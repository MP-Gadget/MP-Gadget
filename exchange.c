#include <mpi.h>
#include <string.h>
/* #include "domain.h" */
#include "mymalloc.h"
#include "allvars.h"
#include "endrun.h"
#include "system.h"
#include "exchange.h"
#include "forcetree.h"
#include "timestep.h"

static MPI_Datatype MPI_TYPE_PARTICLE = 0;
static MPI_Datatype MPI_TYPE_SPHPARTICLE = 0;
static MPI_Datatype MPI_TYPE_BHPARTICLE = 0;

/* 
 * 
 * exchange particles according to layoutfunc.
 * layoutfunc gives the target task of particle p.
*/
static int domain_exchange_once(int (*layoutfunc)(int p), int* toGo, int * toGoSph, int * toGoBh, int *toGet, int *toGetSph, int *toGetBh, enum ExchangeType exchange_type);
static int domain_countToGo(ptrdiff_t nlimit, int (*layoutfunc)(int p), int* toGo, int * toGoSph, int * toGoBh, int *toGet, int *toGetSph, int *toGetBh);

static void domain_count_particles();
static void domain_refresh_totals();

int domain_exchange(int (*layoutfunc)(int p), enum ExchangeType exchange_type) {
    int i;
    int64_t sumtogo;
    int failure = 0;
    /* register the mpi types used in communication if not yet. */
    if (MPI_TYPE_PARTICLE == 0) {
        MPI_Type_contiguous(sizeof(struct particle_data), MPI_BYTE, &MPI_TYPE_PARTICLE);
        MPI_Type_contiguous(sizeof(struct bh_particle_data), MPI_BYTE, &MPI_TYPE_BHPARTICLE);
        MPI_Type_contiguous(sizeof(struct sph_particle_data), MPI_BYTE, &MPI_TYPE_SPHPARTICLE);
        MPI_Type_commit(&MPI_TYPE_PARTICLE);
        MPI_Type_commit(&MPI_TYPE_BHPARTICLE);
        MPI_Type_commit(&MPI_TYPE_SPHPARTICLE);
    }

    /*! toGo[task*NTask + partner] gives the number of particles in task 'task'
     *  that have to go to task 'partner'
     */
    /* flag the particles that need to be exported */
    int * toGo = (int *) mymalloc("toGo", (sizeof(int) * NTask));
    int * toGoSph = (int *) mymalloc("toGoSph", (sizeof(int) * NTask));
    int * toGoBh = (int *) mymalloc("toGoBh", (sizeof(int) * NTask));
    int * toGet = (int *) mymalloc("toGet", (sizeof(int) * NTask));
    int * toGetSph = (int *) mymalloc("toGetSph", (sizeof(int) * NTask));
    int * toGetBh = (int *) mymalloc("toGetBh", (sizeof(int) * NTask));


#pragma omp parallel for
    for(i = 0; i < NumPart; i++)
    {
        int target = layoutfunc(i);
        if(target != ThisTask)
            P[i].OnAnotherDomain = 1;
        P[i].WillExport = 0;
    }

    walltime_measure("/Domain/exchange/init");

    int iter = 0, ret;
    ptrdiff_t exchange_limit;

    do
    {
        exchange_limit = FreeBytes - NTask * (24 * sizeof(int) + 16 * sizeof(MPI_Request));

        if(exchange_limit <= 0)
        {
            endrun(1, "exchange_limit=%d < 0\n", (int) exchange_limit);
        }

        /* determine for each cpu how many particles have to be shifted to other cpus */
        ret = domain_countToGo(exchange_limit, layoutfunc, toGo, toGoSph, toGoBh,toGet, toGetSph, toGetBh);
        walltime_measure("/Domain/exchange/togo");

        for(i = 0, sumtogo = 0; i < NTask; i++)
            sumtogo += toGo[i];

        sumup_longs(1, &sumtogo, &sumtogo);

        message(0, "iter=%d exchange of %013ld particles\n", iter, sumtogo);

        failure = domain_exchange_once(layoutfunc, toGo, toGoSph, toGoBh,toGet, toGetSph, toGetBh, exchange_type);
        if(failure)
            break;
        iter++;
    }
    while(ret > 0);

    myfree(toGetBh);
    myfree(toGetSph);
    myfree(toGet);
    myfree(toGoBh);
    myfree(toGoSph);
    myfree(toGo);
    /* Watch out: domain exchange changes the local number of particles.
     * though the slots has been taken care of in exchange_once, the
     * particle number counts are not updated. */
    domain_count_particles();

    return failure;
}

static int
domain_exchange_once(int (*layoutfunc)(int p), int* toGo, int * toGoSph, int * toGoBh, int *toGet, int *toGetSph, int *toGetBh, enum ExchangeType exchange_type)
{
    if (exchange_type == EXCHANGE_INCREMENTAL) {
        if(!force_tree_allocated()) {
            endrun(0, "Force tree must be allocated to run an incremental domain exchange\n");
        }
    }

    int count_togo = 0, count_togo_sph = 0, count_togo_bh = 0, 
        count_get = 0, count_get_sph = 0, count_get_bh = 0;

    int i, target;
    struct particle_data *partBuf;
    struct sph_particle_data *sphBuf;
    struct bh_particle_data *bhBuf;

    int * count = (int *) alloca(NTask * sizeof(int));
    int * count_sph = (int *) alloca(NTask * sizeof(int));
    int * count_bh = (int *) alloca(NTask * sizeof(int));
    int * offset = (int *) alloca(NTask * sizeof(int));
    int * offset_sph = (int *) alloca(NTask * sizeof(int));
    int * offset_bh = (int *) alloca(NTask * sizeof(int));

    int * count_recv = (int *) alloca(NTask * sizeof(int));
    int * count_recv_sph = (int *) alloca(NTask * sizeof(int));
    int * count_recv_bh = (int *) alloca(NTask * sizeof(int));
    int * offset_recv = (int *) alloca(NTask * sizeof(int));
    int * offset_recv_sph = (int *) alloca(NTask * sizeof(int));
    int * offset_recv_bh = (int *) alloca(NTask * sizeof(int));

    for(i = 1, offset_sph[0] = 0; i < NTask; i++)
        offset_sph[i] = offset_sph[i - 1] + toGoSph[i - 1];

    for(i = 1, offset_bh[0] = 0; i < NTask; i++)
        offset_bh[i] = offset_bh[i - 1] + toGoBh[i - 1];

    offset[0] = 0;

    for(i = 1; i < NTask; i++)
        offset[i] = offset[i - 1] + toGo[i - 1];

    for(i = 0; i < NTask; i++)
    {
        count_togo += toGo[i];
        count_togo_sph += toGoSph[i];
        count_togo_bh += toGoBh[i];

        count_get += toGet[i];
        count_get_sph += toGetSph[i];
        count_get_bh += toGetBh[i];
    }
    int bad_exh=0, bad_exh_s=0;

    /*Check whether the domain exchange will succeed. If not, bail*/
    if(NumPart + count_get - count_togo> All.MaxPart){
        message(1,"Too many particles for exchange: NumPart=%d count_get = %d count_togo=%d All.MaxPart=%d\n", NumPart, count_get, count_togo, All.MaxPart);
        abort();
        bad_exh += 1;
    }
    if(N_sph_slots + count_get_sph - count_togo_sph > All.MaxPart) {
        message(1,"Too many SPH for exchange: N_sph=%d All.MaxPart=%d\n", N_sph_slots + count_get_sph, All.MaxPart);
        bad_exh += 1;
    }
    if(N_bh_slots + count_get_bh - count_togo_bh > All.MaxPartBh) {
        message(1, "Too many BH for exchange: N_bh=%d All.MaxPartBh=%d\n", N_bh_slots + count_get_bh, All.MaxPartBh);
        bad_exh += 1;
    }
    MPI_Allreduce(&bad_exh, &bad_exh_s, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if(bad_exh_s)
        return bad_exh_s;

    partBuf = (struct particle_data *) mymalloc("partBuf", count_togo * sizeof(struct particle_data));
    sphBuf = (struct sph_particle_data *) mymalloc("sphBuf", count_togo_sph * sizeof(struct sph_particle_data));
    bhBuf = (struct bh_particle_data *) mymalloc("bhBuf", count_togo_bh * sizeof(struct bh_particle_data));

    for(i = 0; i < NTask; i++)
        count[i] = count_sph[i] = count_bh[i] = 0;

    /*FIXME: make this omp ! */
    for(i = 0; i < NumPart; i++)
    {
        if(!(P[i].OnAnotherDomain && P[i].WillExport)) continue;
        /* preparing for export */
        P[i].OnAnotherDomain = 0;
        P[i].WillExport = 0;
        target = layoutfunc(i);

        if(P[i].Type == 0)
        {
            sphBuf[offset_sph[target] + count_sph[target]] = SPHP(i);
            /* Set PI to the comm buffer of this rank rather than the slot*/
            P[i].PI = count_sph[target];
            count_sph[target]++;
        } else
        if(P[i].Type == 5)
        {
            bhBuf[offset_bh[target] + count_bh[target]] = BHP(i);
            /* Set PI to the comm buffer of this rank rather than the slot*/
            P[i].PI = count_bh[target];
            count_bh[target]++;
        }

        partBuf[offset[target] + count[target]] = P[i];
        count[target]++;

        if (exchange_type == EXCHANGE_INCREMENTAL) {
            /* mark the particle for removal in GC; TODO : remove it from the tree */
            P[i].Mass = 0;
            force_remove_node(i);
            TimeBinCountType[P[i].Type][P[i].TimeBin] --;
            TimeBinCount[P[i].TimeBin] --;
        } else {
            /* remove this particle from local storage */
            P[i] = P[NumPart - 1];
            NumPart --;
            i--;
        }
    }
    walltime_measure("/Domain/exchange/makebuf");

    for(i = 0; i < NTask; i ++) {
        if(count_sph[i] != toGoSph[i] ) {
            abort();
        }
        if(count_bh[i] != toGoBh[i] ) {
            abort();
        }
    }

    for(i = 0; i < NTask; i++)
    {
        count_recv_sph[i] = toGetSph[i];
        count_recv_bh[i] = toGetBh[i];
        count_recv[i] = toGet[i];
    }

    for(i = 1, offset_recv_sph[0] = N_sph_slots; i < NTask; i++)
        offset_recv_sph[i] = offset_recv_sph[i - 1] + count_recv_sph[i - 1];

    for(i = 1, offset_recv_bh[0] = N_bh_slots; i < NTask; i++)
        offset_recv_bh[i] = offset_recv_bh[i - 1] + count_recv_bh[i - 1];

    offset_recv[0] = NumPart;

    for(i = 1; i < NTask; i++)
        offset_recv[i] = offset_recv[i - 1] + count_recv[i - 1];

    NumPart += count_get;
    N_sph_slots += count_get_sph;
    N_bh_slots += count_get_bh;

    if(NumPart > All.MaxPart) {
        endrun(787878, "Task=%d NumPart=%d All.MaxPart=%d\n", ThisTask, NumPart, All.MaxPart);
    }

    if(N_sph_slots > All.MaxPart)
        endrun(787878, "Task=%d N_sph=%d All.MaxPart=%d\n", ThisTask, N_sph_slots, All.MaxPart);

    if(N_bh_slots > All.MaxPartBh)
        endrun(787878, "Task=%d N_bh=%d All.MaxPartBh=%d\n", ThisTask, N_bh_slots, All.MaxPartBh);

    MPI_Alltoallv_sparse(partBuf, count, offset, MPI_TYPE_PARTICLE,
                 P, count_recv, offset_recv, MPI_TYPE_PARTICLE,
                 MPI_COMM_WORLD);
    walltime_measure("/Domain/exchange/alltoall");

    MPI_Alltoallv_sparse(sphBuf, count_sph, offset_sph, MPI_TYPE_SPHPARTICLE,
                 SphP, count_recv_sph, offset_recv_sph, MPI_TYPE_SPHPARTICLE,
                 MPI_COMM_WORLD);
    walltime_measure("/Domain/exchange/alltoall");

    MPI_Alltoallv_sparse(bhBuf, count_bh, offset_bh, MPI_TYPE_BHPARTICLE,
                BhP, count_recv_bh, offset_recv_bh, MPI_TYPE_BHPARTICLE,
                MPI_COMM_WORLD);
    walltime_measure("/Domain/exchange/alltoall");

    if(count_get_bh > 0) {
        for(target = 0; target < NTask; target++) {
            int i, j;
            for(i = offset_recv[target], j = offset_recv_bh[target];
                i < offset_recv[target] + count_recv[target]; i++) {
                if(P[i].Type != 5) continue;
                P[i].PI = j;
                j++;
            }
            if(j != count_recv_bh[target] + offset_recv_bh[target]) {
                endrun(1, "communication bh inconsistency\n");
            }
        }
    }

    if(count_get_sph > 0) {
        for(target = 0; target < NTask; target++) {
            int i, j;
            for(i = offset_recv[target], j = offset_recv_sph[target];
                i < offset_recv[target] + count_recv[target]; i++) {
                if(P[i].Type != 0) continue;
                P[i].PI = j;
                j++;
            }
            if(j != count_recv_sph[target] + offset_recv_sph[target]) {
                endrun(1, "communication bh inconsistency\n");
            }
        }
    }

    if(exchange_type == EXCHANGE_INCREMENTAL) {
        for(i = NumPart - count_get; i < NumPart; i++) {
            force_insert_particle(i);
        }
        message(0, "Added %d particles to the force tree, Numpart = %d\n", count_get, NumPart);
    }
    myfree(bhBuf);
    myfree(sphBuf);
    myfree(partBuf);

    MPI_Barrier(MPI_COMM_WORLD);

    walltime_measure("/Domain/exchange/finalize");
    return 0;
}


/*This function populates the toGo and toGet arrays*/
static int domain_countToGo(ptrdiff_t nlimit, int (*layoutfunc)(int p), int* toGo, int * toGoSph, int * toGoBh, int *toGet, int *toGetSph, int *toGetBh)
{
    int n, ret, retsum;
    size_t package;

    int * list_NumPart = (int *) alloca(sizeof(int) * NTask);
    int * list_N_sph = (int *) alloca(sizeof(int) * NTask);
    int * list_N_bh = (int *) alloca(sizeof(int) * NTask);

    for(n = 0; n < NTask; n++)
    {
        toGo[n] = 0;
        toGoSph[n] = 0;
        toGoBh[n] = 0;
    }

    package = (sizeof(struct particle_data) + sizeof(struct sph_particle_data) + sizeof(struct bh_particle_data));
    if(package >= nlimit)
        endrun(212, "Package is too large, no free memory.");


    for(n = 0; n < NumPart; n++)
    {
        if(package >= nlimit) break;
        if(!P[n].OnAnotherDomain) continue;

        int target = layoutfunc(n);
        if (target == ThisTask) continue;

        toGo[target] += 1;
        nlimit -= sizeof(struct particle_data);

        if(P[n].Type  == 0)
        {
            toGoSph[target] += 1;
            nlimit -= sizeof(struct sph_particle_data);
        }
        if(P[n].Type  == 5)
        {
            toGoBh[target] += 1;
            nlimit -= sizeof(struct bh_particle_data);
        }
        P[n].WillExport = 1;	/* flag this particle for export */
    }

    MPI_Alltoall(toGo, 1, MPI_INT, toGet, 1, MPI_INT, MPI_COMM_WORLD);
    MPI_Alltoall(toGoSph, 1, MPI_INT, toGetSph, 1, MPI_INT, MPI_COMM_WORLD);
    MPI_Alltoall(toGoBh, 1, MPI_INT, toGetBh, 1, MPI_INT, MPI_COMM_WORLD);

    if(package >= nlimit)
        ret = 1;
    else
        ret = 0;

    MPI_Allreduce(&ret, &retsum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    if(retsum)
    {
        /* in this case, we are not guaranteed that the temporary state after
           the partial exchange will actually observe the particle limits on all
           processors... we need to test this explicitly and rework the exchange
           such that this is guaranteed. This is actually a rather non-trivial
           constraint. */

        MPI_Allgather(&NumPart, 1, MPI_INT, list_NumPart, 1, MPI_INT, MPI_COMM_WORLD);
        MPI_Allgather(&N_bh_slots, 1, MPI_INT, list_N_bh, 1, MPI_INT, MPI_COMM_WORLD);
        MPI_Allgather(&N_sph_slots, 1, MPI_INT, list_N_sph, 1, MPI_INT, MPI_COMM_WORLD);

        int flag, flagsum, ntoomany, ta, i;
        int count_togo, count_toget, count_togo_bh, count_toget_bh, count_togo_sph, count_toget_sph;

        do
        {
            flagsum = 0;

            do
            {
                flag = 0;

                for(ta = 0; ta < NTask; ta++)
                {
                    if(ta == ThisTask)
                    {
                        count_togo = count_toget = 0;
                        count_togo_sph = count_toget_sph = 0;
                        count_togo_bh = count_toget_bh = 0;
                        for(i = 0; i < NTask; i++)
                        {
                            count_togo += toGo[i];
                            count_toget += toGet[i];
                            count_togo_sph += toGoSph[i];
                            count_toget_sph += toGetSph[i];
                            count_togo_bh += toGoBh[i];
                            count_toget_bh += toGetBh[i];
                        }
                    }
                    MPI_Bcast(&count_togo, 1, MPI_INT, ta, MPI_COMM_WORLD);
                    MPI_Bcast(&count_toget, 1, MPI_INT, ta, MPI_COMM_WORLD);
                    MPI_Bcast(&count_togo_sph, 1, MPI_INT, ta, MPI_COMM_WORLD);
                    MPI_Bcast(&count_toget_sph, 1, MPI_INT, ta, MPI_COMM_WORLD);
                    MPI_Bcast(&count_togo_bh, 1, MPI_INT, ta, MPI_COMM_WORLD);
                    MPI_Bcast(&count_toget_bh, 1, MPI_INT, ta, MPI_COMM_WORLD);
                    if((ntoomany = list_N_sph[ta] + count_toget_sph - count_togo_sph - All.MaxPart) > 0)
                    {
                        message (0, "exchange needs to be modified because I can't receive %d SPH-particles on task=%d\n",
                                 ntoomany, ta);
                        if(flagsum > 25) {
                            message(0, "list_N_sph[ta=%d]=%d  count_toget_sph=%d count_togo_sph=%d\n",
                                        ta, list_N_sph[ta], count_toget_sph, count_togo_sph);
                        }
                        flag = 1;
                        i = flagsum % NTask;
                        while(ntoomany)
                        {
                            if(i == ThisTask)
                            {
                                if(toGoSph[ta] > 0)
                                {
                                    toGoSph[ta]--;
                                    count_toget_sph--;
                                    count_toget--;
                                    ntoomany--;
                                }
                            }

                            MPI_Bcast(&ntoomany, 1, MPI_INT, i, MPI_COMM_WORLD);
                            MPI_Bcast(&count_toget, 1, MPI_INT, i, MPI_COMM_WORLD);
                            MPI_Bcast(&count_toget_sph, 1, MPI_INT, i, MPI_COMM_WORLD);
                            i++;
                            if(i >= NTask)
                                i = 0;
                        }
                    }
                    if((ntoomany = list_N_bh[ta] + count_toget_bh - count_togo_bh - All.MaxPartBh) > 0)
                    {
                        message(0, "exchange needs to be modified because I can't receive %d BH-particles on task=%d\n",
                                ntoomany, ta);
                        if(flagsum > 25)
                            message(0, "list_N_bh[ta=%d]=%d  count_toget_bh=%d count_togo_bh=%d\n",
                                    ta, list_N_bh[ta], count_toget_bh, count_togo_bh);

                        flag = 1;
                        i = flagsum % NTask;
                        while(ntoomany)
                        {
                            if(i == ThisTask)
                            {
                                if(toGoBh[ta] > 0)
                                {
                                    toGoBh[ta]--;
                                    count_toget_bh--;
                                    count_toget--;
                                    ntoomany--;
                                }
                            }

                            MPI_Bcast(&ntoomany, 1, MPI_INT, i, MPI_COMM_WORLD);
                            MPI_Bcast(&count_toget, 1, MPI_INT, i, MPI_COMM_WORLD);
                            MPI_Bcast(&count_toget_bh, 1, MPI_INT, i, MPI_COMM_WORLD);
                            i++;
                            if(i >= NTask)
                                i = 0;
                        }
                    }

                    if((ntoomany = list_NumPart[ta] + count_toget - count_togo - All.MaxPart) > 0)
                    {
                        message (0, "exchange needs to be modified because I can't receive %d particles on task=%d\n",
                             ntoomany, ta);
                        if(flagsum > 25)
                            message(0, "list_NumPart[ta=%d]=%d  count_toget=%d count_togo=%d\n",
                                    ta, list_NumPart[ta], count_toget, count_togo);

                        flag = 1;
                        i = flagsum % NTask;
                        while(ntoomany)
                        {
                            if(i == ThisTask)
                            {
                                if(toGo[ta] > 0)
                                {
                                    toGo[ta]--;
                                    count_toget--;
                                    ntoomany--;
                                }
                            }

                            MPI_Bcast(&ntoomany, 1, MPI_INT, i, MPI_COMM_WORLD);
                            MPI_Bcast(&count_toget, 1, MPI_INT, i, MPI_COMM_WORLD);

                            i++;
                            if(i >= NTask)
                                i = 0;
                        }
                    }
                }
                flagsum += flag;

                message(0, "flagsum = %d\n", flagsum);
                if(flagsum > 100)
                    endrun(1013, "flagsum is too big, what does this mean?");
            }
            while(flag);

            if(flagsum)
            {
                int *local_toGo, *local_toGoSph, *local_toGoBh;

                local_toGo = (int *)mymalloc("	      local_toGo", NTask * sizeof(int));
                local_toGoSph = (int *)mymalloc("	      local_toGoSph", NTask * sizeof(int));
                local_toGoBh = (int *)mymalloc("	      local_toGoBh", NTask * sizeof(int));


                for(n = 0; n < NTask; n++)
                {
                    local_toGo[n] = 0;
                    local_toGoSph[n] = 0;
                    local_toGoBh[n] = 0;
                }

                for(n = 0; n < NumPart; n++)
                {
                    if(!P[n].OnAnotherDomain) continue;
                    P[n].WillExport = 0; /* clear 16 */

                    int target = layoutfunc(n);

                    if(P[n].Type == 0)
                    {
                        if(local_toGoSph[target] < toGoSph[target] && local_toGo[target] < toGo[target])
                        {
                            local_toGo[target] += 1;
                            local_toGoSph[target] += 1;
                            P[n].WillExport = 1;
                        }
                    }
                    else
                    if(P[n].Type == 5)
                    {
                        if(local_toGoBh[target] < toGoBh[target] && local_toGo[target] < toGo[target])
                        {
                            local_toGo[target] += 1;
                            local_toGoBh[target] += 1;
                            P[n].WillExport = 1;
                        }
                    }
                    else
                    {
                        if(local_toGo[target] < toGo[target])
                        {
                            local_toGo[target] += 1;
                            P[n].WillExport = 1;
                        }
                    }
                }

                for(n = 0; n < NTask; n++)
                {
                    toGo[n] = local_toGo[n];
                    toGoSph[n] = local_toGoSph[n];
                    toGoBh[n] = local_toGoBh[n];
                }

                MPI_Alltoall(toGo, 1, MPI_INT, toGet, 1, MPI_INT, MPI_COMM_WORLD);
                MPI_Alltoall(toGoSph, 1, MPI_INT, toGetSph, 1, MPI_INT, MPI_COMM_WORLD);
                MPI_Alltoall(toGoBh, 1, MPI_INT, toGetBh, 1, MPI_INT, MPI_COMM_WORLD);
                myfree(local_toGoBh);
                myfree(local_toGoSph);
                myfree(local_toGo);
            }
        }
        while(flagsum);

        return 1;
    }
    else
        return 0;
}

void domain_count_particles()
{
    int i;
    for(i = 0; i < 6; i++)
        NLocal[i] = 0;

#pragma omp parallel private(i)
    {
        int NLocalThread[6] = {0};
#pragma omp for
        for(i = 0; i < NumPart; i++)
        {
            NLocalThread[P[i].Type]++;
        }
#pragma omp critical 
        {
/* avoid omp reduction for now: Craycc doesn't always do it right */
            for(i = 0; i < 6; i ++) {
                NLocal[i] += NLocalThread[i];
            }
        }
    }
    domain_refresh_totals();
}

void
domain_refresh_totals()
{
    int ptype;
    /* because NTotal[] is of type `int64_t', we cannot do a simple
     * MPI_Allreduce() to sum the total particle numbers 
     */
    MPI_Allreduce(NLocal, NTotal, 6, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);

    TotNumPart = 0;
    for(ptype = 0; ptype < 6; ptype ++) {
        TotNumPart += NTotal[ptype];
    }
}

