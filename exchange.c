#include <mpi.h>
#include <string.h>
/* #include "domain.h" */
#include "mymalloc.h"
#include "allvars.h"
#include "endrun.h"
#include "system.h"
#include "exchange.h"
#include "garbage.h"

static MPI_Datatype MPI_TYPE_PARTICLE = 0;
static MPI_Datatype MPI_TYPE_SPHPARTICLE = 0;
static MPI_Datatype MPI_TYPE_BHPARTICLE = 0;

/* 
 * 
 * exchange particles according to layoutfunc.
 * layoutfunc gives the target task of particle p.
*/
static int domain_exchange_once(int (*layoutfunc)(int p), int* toGo, int * toGoSph, int * toGoBh, int *toGet, int *toGetSph, int *toGetBh);
static int domain_countToGo(ptrdiff_t nlimit, int (*layoutfunc)(int p), int* toGo, int * toGoSph, int * toGoBh, int *toGet, int *toGetSph, int *toGetBh);

static void domain_count_particles();

int domain_exchange(int (*layoutfunc)(int p)) {
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

        failure = domain_exchange_once(layoutfunc, toGo, toGoSph, toGoBh,toGet, toGetSph, toGetBh);
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

#define NSP 3

static int domain_exchange_once(int (*layoutfunc)(int p), int* toGo, int * toGoSph, int * toGoBh, int *toGet, int *toGetSph, int *toGetBh)
{
    int i, j, target;
    struct particle_data *partBuf;
    struct sph_particle_data *sphBuf;
    struct bh_particle_data *bhBuf;

    int *ctmem = mymalloc("cts", 4*NSP*NTask*sizeof(int));
    int *count[NSP];
    int *offset[NSP];
    int *count_recv[NSP];
    int *offset_recv[NSP];
    int count_togo[NSP] = {0};
    int count_get[NSP] = {0};

    memset(ctmem, 0, 4*NSP*NTask*sizeof(int));

    for(j=0; j<NSP; j++) {
        count[j] = ctmem+j*NTask;
        offset[j] = ctmem + NSP * NTask +j*NTask;
        count_recv[j] = ctmem + 2 * NSP * NTask +j*NTask;
        offset_recv[j] = ctmem + 3 * NSP * NTask +j*NTask;
    }

    /*Build arrays*/
    int * toGo_arr[NSP];
    toGo_arr[0] = toGo;
    toGo_arr[1] = toGoSph;
    toGo_arr[2] = toGoBh;

    int * toGet_arr[NSP];
    toGet_arr[0] = toGet;
    toGet_arr[1] = toGetSph;
    toGet_arr[2] = toGetBh;

    int bad_exh=0;
    const char *nn[3] = {"particles", "SPH","BH"};

    for(j=0; j<NSP; j++) {
        /*Compute offsets*/
        offset[j][0] = 0;
        for(i = 1; i < NTask; i++)
            offset[j][i] = offset[j][i - 1] + toGo_arr[j][i - 1];
        /*Compute counts*/
        for(i = 0; i < NTask; i++)
        {
            count_togo[j] += toGo_arr[j][i];
            count_get[j] += toGet_arr[j][i];
        }
        /*Check whether the domain exchange will succeed. If not, bail*/
        if(NumPart + count_get[j] - count_togo[j] > All.MaxPart){
            message(1,"Too many %s for exchange: NumPart=%d count_get = %d count_togo=%d All.MaxPart=%d\n", nn[j], NumPart, count_get[j], count_togo[j], All.MaxPart);
            bad_exh = 1;
        }
    }

    MPI_Allreduce(MPI_IN_PLACE, &bad_exh, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
    if(bad_exh) {
        myfree(ctmem);
        return bad_exh;
    }

    partBuf = (struct particle_data *) mymalloc("partBuf", count_togo[0] * sizeof(struct particle_data));
    sphBuf = (struct sph_particle_data *) mymalloc("sphBuf", count_togo[1] * sizeof(struct sph_particle_data));
    bhBuf = (struct bh_particle_data *) mymalloc("bhBuf", count_togo[2] * sizeof(struct bh_particle_data));

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
            sphBuf[offset[1][target] + count[1][target]] = SPHP(i);
            /* Set PI to the comm buffer of this rank rather than the slot*/
            P[i].PI = count[1][target];
            count[1][target]++;
        } else
        if(P[i].Type == 5)
        {
            bhBuf[offset[2][target] + count[2][target]] = BHP(i);
            /* Set PI to the comm buffer of this rank rather than the slot*/
            P[i].PI = count[2][target];
            count[2][target]++;
        }

        partBuf[offset[0][target] + count[0][target]] = P[i];
        count[0][target]++;

        /* mark this particle as a garbage */
        P[i].IsGarbage = 1;
    }
    /* now remove the garbage particles because they have already been copied.
     * eventually we want to fill in the garbage gap or defer the gc, because it breaks the tree.
     * invariance . */
    domain_garbage_collection();

    walltime_measure("/Domain/exchange/makebuf");


    for(j=0; j<NSP; j++) {
        for(i = 0; i < NTask; i ++) {
            if(count[j][i] != toGo_arr[j][i] ) {
                endrun(2, "Count inconsistency %d != %d", count[j][i], toGo_arr[j][i]);
            }
            count_recv[j][i] = toGet_arr[j][i];
        }
    }
    offset_recv[0][0] = NumPart;
    offset_recv[1][0] = N_sph_slots;
    offset_recv[2][0] = N_bh_slots;

    for(j=0; j<NSP; j++)
        for(i = 1; i < NTask; i++)
            offset_recv[j][i] = offset_recv[j][i - 1] + count_recv[j][i - 1];

    NumPart += count_get[0];
    N_sph_slots += count_get[1];
    N_bh_slots += count_get[2];

    if(NumPart > All.MaxPart) {
        endrun(787878, "Task=%d NumPart=%d All.MaxPart=%d\n", ThisTask, NumPart, All.MaxPart);
    }

    if(N_sph_slots > All.MaxPart)
        endrun(787878, "Task=%d N_sph=%d All.MaxPart=%d\n", ThisTask, N_sph_slots, All.MaxPart);

    if(N_bh_slots > All.MaxPartBh)
        endrun(787878, "Task=%d N_bh=%d All.MaxPartBh=%d\n", ThisTask, N_bh_slots, All.MaxPartBh);

    MPI_Alltoallv_sparse(partBuf, count[0], offset[0], MPI_TYPE_PARTICLE,
                 P, count_recv[0], offset_recv[0], MPI_TYPE_PARTICLE,
                 MPI_COMM_WORLD);
    walltime_measure("/Domain/exchange/alltoall");

    MPI_Alltoallv_sparse(sphBuf, count[1], offset[1], MPI_TYPE_SPHPARTICLE,
                 SphP, count_recv[1], offset_recv[1], MPI_TYPE_SPHPARTICLE,
                 MPI_COMM_WORLD);
    walltime_measure("/Domain/exchange/alltoall");

    MPI_Alltoallv_sparse(bhBuf, count[2], offset[2], MPI_TYPE_BHPARTICLE,
                BhP, count_recv[2], offset_recv[2], MPI_TYPE_BHPARTICLE,
                MPI_COMM_WORLD);
    walltime_measure("/Domain/exchange/alltoall");

    if(count_get[2] > 0 || count_get[1] > 0) {
        for(target = 0; target < NTask; target++) {
            int spho = offset_recv[1][target];
            int bho = offset_recv[2][target];
            for(i = offset_recv[0][target]; i < offset_recv[0][target] + count_recv[0][target]; i++) {
                if(P[i].Type == 0) {
                    P[i].PI = spho;
                    spho++;
                }
                if(P[i].Type == 5) {
                    P[i].PI = bho;
                    bho++;
                }
            }
            if(spho != count_recv[1][target] + offset_recv[1][target]) {
                endrun(1, "communication sph inconsistency\n");
            }
            if(bho != count_recv[2][target] + offset_recv[2][target]) {
                endrun(1, "communication bh inconsistency\n");
            }
        }
    }

    myfree(bhBuf);
    myfree(sphBuf);
    myfree(partBuf);
    myfree(ctmem);

    MPI_Barrier(MPI_COMM_WORLD);

    walltime_measure("/Domain/exchange/finalize");

    return 0;
}


/*This function populates the toGo and toGet arrays*/
static int domain_countToGo(ptrdiff_t nlimit, int (*layoutfunc)(int p), int* toGo, int * toGoSph, int * toGoBh, int *toGet, int *toGetSph, int *toGetBh)
{
    int n, ret;
    size_t package;

    int * list_NumPart = ta_malloc("var", int, NTask);
    int * list_N_sph = ta_malloc("var", int, NTask);
    int * list_N_bh = ta_malloc("var", int, NTask);

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

    MPI_Allreduce(MPI_IN_PLACE, &ret, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);

    if(ret)
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

                //message(0, "flagsum = %d\n", flagsum);
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

    }
    ta_reset();
    return ret;
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
    MPI_Allreduce(NLocal, NTotal, 6, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
}
