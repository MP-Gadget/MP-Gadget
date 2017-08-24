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
static MPI_Datatype MPI_TYPE_STARPARTICLE = 0;

/* 
 * 
 * exchange particles according to layoutfunc.
 * layoutfunc gives the target task of particle p.
*/
static int domain_exchange_once(int (*layoutfunc)(int p), int* toGo, int * toGoSph, int * toGoBh, int * toGoStar, int *toGet, int *toGetSph, int *toGetBh, int * toGetStar);
static int domain_countToGo(ptrdiff_t nlimit, int (*layoutfunc)(int p), int* toGo, int * toGoSph, int * toGoBh, int * toGoStar, int *toGet, int *toGetSph, int *toGetBh, int * toGetStar);

static void domain_count_particles();

int domain_exchange(int (*layoutfunc)(int p)) {
    int i;
    int64_t sumtogo;
    int failure = 0;
    /* register the mpi types used in communication if not yet. */
    if (MPI_TYPE_PARTICLE == 0) {
        MPI_Type_contiguous(sizeof(struct particle_data), MPI_BYTE, &MPI_TYPE_PARTICLE);
        MPI_Type_contiguous(sizeof(struct bh_particle_data), MPI_BYTE, &MPI_TYPE_BHPARTICLE);
        MPI_Type_contiguous(sizeof(struct star_particle_data), MPI_BYTE, &MPI_TYPE_STARPARTICLE);
        MPI_Type_contiguous(sizeof(struct sph_particle_data), MPI_BYTE, &MPI_TYPE_SPHPARTICLE);
        MPI_Type_commit(&MPI_TYPE_PARTICLE);
        MPI_Type_commit(&MPI_TYPE_BHPARTICLE);
        MPI_Type_commit(&MPI_TYPE_STARPARTICLE);
        MPI_Type_commit(&MPI_TYPE_SPHPARTICLE);
    }

    /*! toGo[task*NTask + partner] gives the number of particles in task 'task'
     *  that have to go to task 'partner'
     */
    /* flag the particles that need to be exported */
    int * toGo = (int *) mymalloc("toGo", (sizeof(int) * NTask));
    int * toGoSph = (int *) mymalloc("toGoSph", (sizeof(int) * NTask));
    int * toGoBh = (int *) mymalloc("toGoBh", (sizeof(int) * NTask));
    int * toGoStar = (int *) mymalloc("toGoStar", (sizeof(int) * NTask));
    int * toGet = (int *) mymalloc("toGet", (sizeof(int) * NTask));
    int * toGetSph = (int *) mymalloc("toGetSph", (sizeof(int) * NTask));
    int * toGetBh = (int *) mymalloc("toGetBh", (sizeof(int) * NTask));
    int * toGetStar = (int *) mymalloc("toGetStar", (sizeof(int) * NTask));


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
        ret = domain_countToGo(exchange_limit, layoutfunc, toGo, toGoSph, toGoBh, toGoStar, toGet, toGetSph, toGetBh, toGetStar);
        walltime_measure("/Domain/exchange/togo");

        for(i = 0, sumtogo = 0; i < NTask; i++)
            sumtogo += toGo[i];

        sumup_longs(1, &sumtogo, &sumtogo);

        message(0, "iter=%d exchange of %013ld particles\n", iter, sumtogo);

        failure = domain_exchange_once(layoutfunc, toGo, toGoSph, toGoBh, toGoStar, toGet, toGetSph, toGetBh, toGetStar);
        if(failure)
            break;
        iter++;
    }
    while(ret > 0);

    myfree(toGetStar);
    myfree(toGetBh);
    myfree(toGetSph);
    myfree(toGet);
    myfree(toGoStar);
    myfree(toGoBh);
    myfree(toGoSph);
    myfree(toGo);
    /* Watch out: domain exchange changes the local number of particles.
     * though the slots has been taken care of in exchange_once, the
     * particle number counts are not updated. */
    domain_count_particles();

    return failure;
}

#define NSP 4

static int domain_exchange_once(int (*layoutfunc)(int p), int* toGo, int * toGoSph, int * toGoBh, int * toGoStar, int *toGet, int *toGetSph, int *toGetBh, int * toGetStar)
{
    int i, j, target;
    struct particle_data *partBuf;
    struct sph_particle_data *sphBuf;
    struct bh_particle_data *bhBuf;
    struct star_particle_data *starBuf;

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
    toGo_arr[3] = toGoStar;

    int * toGet_arr[NSP];
    toGet_arr[0] = toGet;
    toGet_arr[1] = toGetSph;
    toGet_arr[2] = toGetBh;
    toGet_arr[3] = toGetStar;

    int bad_exh=0;
    const char *nn[NSP] = {"particles", "SPH","BH", "Stars"};

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
    starBuf = (struct star_particle_data *) mymalloc("starBuf", count_togo[3] * sizeof(struct star_particle_data));

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
        } else
        if(P[i].Type == 4)
        {
            starBuf[offset[3][target] + count[3][target]] = STARP(i);
            /* Set PI to the comm buffer of this rank rather than the slot*/
            P[i].PI = count[3][target];
            count[3][target]++;
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
    offset_recv[3][0] = N_star_slots;

    for(j=0; j<NSP; j++)
        for(i = 1; i < NTask; i++)
            offset_recv[j][i] = offset_recv[j][i - 1] + count_recv[j][i - 1];

    NumPart += count_get[0];
    N_sph_slots += count_get[1];
    N_bh_slots += count_get[2];
    N_star_slots += count_get[3];

    if(NumPart > All.MaxPart) {
        endrun(787878, "Task=%d NumPart=%d All.MaxPart=%d\n", ThisTask, NumPart, All.MaxPart);
    }

    if(N_sph_slots > All.MaxPart)
        endrun(787878, "Task=%d N_sph=%d All.MaxPart=%d\n", ThisTask, N_sph_slots, All.MaxPart);

    if(N_bh_slots > All.MaxPartBh)
        endrun(787878, "Task=%d N_bh=%d All.MaxPartBh=%d\n", ThisTask, N_bh_slots, All.MaxPartBh);

    if(N_star_slots > All.MaxPartBh)
        endrun(787878, "Task=%d N_star=%d All.MaxPartBh=%d\n", ThisTask, N_star_slots, All.MaxPartBh);

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

    MPI_Alltoallv_sparse(starBuf, count[3], offset[3], MPI_TYPE_STARPARTICLE,
                StarP, count_recv[3], offset_recv[3], MPI_TYPE_STARPARTICLE,
                MPI_COMM_WORLD);
    walltime_measure("/Domain/exchange/alltoall");

    if(count_get[2] > 0 || count_get[1] > 0 || count_get[3] > 0) {
        for(target = 0; target < NTask; target++) {
            int localo[NSP];
            for(i = 0; i<NSP; i++)
                localo[i] = offset_recv[i][target];
            for(i = offset_recv[0][target]; i < offset_recv[0][target] + count_recv[0][target]; i++) {
                int k;
                if(P[i].Type == 0) {
                    k = 1;
                }
                else if(P[i].Type == 5) {
                    k = 2;
                }
                else if(P[i].Type == 4) {
                    k = 3;
                }
                else
                    continue;
                P[i].PI = localo[k];
                localo[k]++;
            }
            for(i = 1; i<NSP; i++) {
                if(localo[i] != count_recv[i][target] + offset_recv[i][target]) {
                    endrun(1, "communication %s inconsistency\n", nn[i]);
                }
            }
        }
    }

    myfree(starBuf);
    myfree(bhBuf);
    myfree(sphBuf);
    myfree(partBuf);
    myfree(ctmem);

    MPI_Barrier(MPI_COMM_WORLD);

    walltime_measure("/Domain/exchange/finalize");

    return 0;
}


/*This function populates the toGo and toGet arrays*/
static int domain_countToGo(ptrdiff_t nlimit, int (*layoutfunc)(int p), int* toGo, int * toGoSph, int * toGoBh, int * toGoStar, int *toGet, int *toGetSph, int *toGetBh, int * toGetStar)
{
    int n, ret;
    size_t package;

    for(n = 0; n < NTask; n++)
    {
        toGo[n] = 0;
        toGoSph[n] = 0;
        toGoBh[n] = 0;
        toGoStar[n] = 0;
    }

    package = (sizeof(struct particle_data) + sizeof(struct sph_particle_data) + sizeof(struct bh_particle_data)+sizeof(struct star_particle_data));
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
        if(P[n].Type  == 4)
        {
            toGoStar[target] += 1;
            nlimit -= sizeof(struct star_particle_data);
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
    MPI_Alltoall(toGoStar, 1, MPI_INT, toGetStar, 1, MPI_INT, MPI_COMM_WORLD);

    ret = (package >= nlimit);

    MPI_Allreduce(MPI_IN_PLACE, &ret, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);

    if(ret == 0)
        return 0;

    {
        /* in this case, we are not guaranteed that the temporary state after
           the partial exchange will actually observe the particle limits on all
           processors... we need to test this explicitly and rework the exchange
           such that this is guaranteed. This is actually a rather non-trivial
           constraint. */

        int flagsum, i;
        /*Order is: total, sph, bh, star*/
        int *togo_local[4];
        int * list_Npart[4];
        list_Npart[0] = (int *)mymalloc("list_Npart", 4*NTask * sizeof(int));
        for(n=1; n<4; n++)
            list_Npart[n] = list_Npart[n-1]+NTask;
        MPI_Allgather(&NumPart, 1, MPI_INT, list_Npart[0], 1, MPI_INT, MPI_COMM_WORLD);
        MPI_Allgather(&N_sph_slots, 1, MPI_INT, list_Npart[1], 1, MPI_INT, MPI_COMM_WORLD);
        MPI_Allgather(&N_bh_slots, 1, MPI_INT, list_Npart[2], 1, MPI_INT, MPI_COMM_WORLD);
        MPI_Allgather(&N_star_slots, 1, MPI_INT, list_Npart[3], 1, MPI_INT, MPI_COMM_WORLD);
        togo_local[0] = toGo;
        togo_local[1] = toGoSph;
        togo_local[2] = toGoBh;
        togo_local[3] = toGoStar;

        /*FIXME: This algorithm is impossibly slow.*/
        do
        {
            int flag;
            flagsum = 0;

            do
            {
                int ta;
                flag = 0;

                for(ta = 0; ta < NTask; ta++)
                {
                    int count_togo[4]={0}, count_toget[4]={0};
                    if(ta == ThisTask)
                    {
                        for(i = 0; i < NTask; i++)
                        {
                            count_togo[0] += toGo[i];
                            count_toget[0] += toGet[i];
                            count_togo[1] += toGoSph[i];
                            count_toget[1] += toGetSph[i];
                            count_togo[2] += toGoBh[i];
                            count_toget[2] += toGetBh[i];
                            count_togo[3] += toGoStar[i];
                            count_toget[3] += toGetStar[i];
                        }
                    }
                    MPI_Bcast(&count_togo, 4, MPI_INT, ta, MPI_COMM_WORLD);
                    MPI_Bcast(&count_toget, 4, MPI_INT, ta, MPI_COMM_WORLD);
                    for(i=3; i > 0; --i) {
                        int ntoomany = list_Npart[i][ta] + count_toget[i] - count_togo[i] - All.MaxPart;
                        if (ntoomany <= 0)
                            continue;
                        message (0, "Exchange: I can't receive %d particles (array %d) on task=%d\n",ntoomany, i, ta);
                        if(flagsum > 25) {
                            message(0, "list_Npart[%d][ta=%d]=%d  count_toget=%d count_togo=%d\n",
                                        ta, list_Npart[i][ta], count_toget[i], count_togo[i]);
                        }
                        flag = 1;
                        int j = flagsum % NTask;
                        while(ntoomany)
                        {
                            if(j == ThisTask)
                            {
                                if(togo_local[i][ta] > 0)
                                {
                                    togo_local[i][ta]--;
                                    if(i > 0)
                                        count_toget[i]--;
                                    count_toget[0]--;
                                    ntoomany--;
                                }
                            }

                            MPI_Bcast(&ntoomany, 1, MPI_INT, j, MPI_COMM_WORLD);
                            MPI_Bcast(&count_toget[0], 1, MPI_INT, j, MPI_COMM_WORLD);
                            if(i > 0)
                                MPI_Bcast(&count_toget[i], 1, MPI_INT, j, MPI_COMM_WORLD);
                            j = (j+1) % NTask;
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
                int * new_toGo[4];
                new_toGo[0] = (int *)mymalloc("local_toGo", 4*NTask * sizeof(int));
                memset(new_toGo, 0, 4*NTask*sizeof(int));
                for(n=1; n<4; n++)
                    new_toGo[n] = new_toGo[n-1]+NTask;

                for(n = 0; n < NumPart; n++)
                {
                    if(!P[n].OnAnotherDomain) continue;
                    P[n].WillExport = 0; /* clear 16 */

                    int target = layoutfunc(n);

                    int lt = 0;
                    if(P[n].Type == 0)
                        lt = 1;
                    if(P[n].Type == 5)
                        lt = 2;
                    if(P[n].Type == 4)
                        lt = 3;
                    if(new_toGo[lt][target] < togo_local[lt][target] && new_toGo[0][target] < togo_local[0][target])
                    {
                        new_toGo[0][target] += 1;
                        if(lt > 0)
                            new_toGo[lt][target] += 1;
                        P[n].WillExport = 1;
                    }
                }

                for(n = 0; n < NTask; n++)
                {
                    toGo[n] = new_toGo[0][n];
                    toGoSph[n] = new_toGo[1][n];
                    toGoBh[n] = new_toGo[2][n];
                    toGoStar[n] = new_toGo[3][n];
                }

                MPI_Alltoall(toGo, 1, MPI_INT, toGet, 1, MPI_INT, MPI_COMM_WORLD);
                MPI_Alltoall(toGoSph, 1, MPI_INT, toGetSph, 1, MPI_INT, MPI_COMM_WORLD);
                MPI_Alltoall(toGoBh, 1, MPI_INT, toGetBh, 1, MPI_INT, MPI_COMM_WORLD);
                MPI_Alltoall(toGoStar, 1, MPI_INT, toGetStar, 1, MPI_INT, MPI_COMM_WORLD);
                myfree(new_toGo[0]);
            }
        }
        while(flagsum);
        myfree(list_Npart[0]);

    }
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
