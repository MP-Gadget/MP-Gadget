#include <mpi.h>
#include <string.h>
/* #include "domain.h" */
#include "mymalloc.h"
#include "allvars.h"
#include "endrun.h"
#include "system.h"
#include "exchange.h"
#include "garbage.h"

/*Number of structure types for particles*/
#define NSP 4

static MPI_Datatype MPI_TYPE_PARTICLE = 0;
static MPI_Datatype MPI_TYPE_SPHPARTICLE = 0;
static MPI_Datatype MPI_TYPE_BHPARTICLE = 0;
static MPI_Datatype MPI_TYPE_STARPARTICLE = 0;

static void
realloc_secondary_data(int N_bh, int N_star);

/* 
 * 
 * exchange particles according to layoutfunc.
 * layoutfunc gives the target task of particle p.
*/
static int domain_exchange_once(int (*layoutfunc)(int p), int** toGo_arr, int ** toGet_arr);
static int domain_countToGo(ptrdiff_t nlimit, int (*layoutfunc)(int p), int failfast, int** toGo_arr, int** toGet_arr);

int domain_exchange(int (*layoutfunc)(int p), int failfast) {
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

    /*! toGo[0][task*NTask + partner] gives the number of particles in task 'task'
     *  that have to go to task 'partner'
     *  toGo[1] is SPH, toGo[2] is BH and toGo[3] is stars
     */
    /* flag the particles that need to be exported */
    int * toGo_arr[NSP];
    int * toGet_arr[NSP];
    toGo_arr[0] = (int *) mymalloc2("toGo", (NSP * sizeof(int) * NTask));
    toGet_arr[0] = (int *) mymalloc2("toGet", (NSP * sizeof(int) * NTask));
    for(i=1; i<NSP; i++) {
        toGo_arr[i] = toGo_arr[i-1] + NTask;
        toGet_arr[i] = toGet_arr[i-1] + NTask;
    }

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
        ret = domain_countToGo(exchange_limit, layoutfunc, failfast, toGo_arr, toGet_arr);
        walltime_measure("/Domain/exchange/togo");
        if(ret && failfast) {
            failure = 1;
            break;
        }

        for(i = 0, sumtogo = 0; i < NTask; i++)
            sumtogo += toGo_arr[0][i];

        sumup_longs(1, &sumtogo, &sumtogo);

        message(0, "iter=%d exchange of %013ld particles\n", iter, sumtogo);

        failure = domain_exchange_once(layoutfunc, toGo_arr, toGet_arr);
        if(failure)
            break;
        iter++;
    }
    while(ret > 0);

    myfree(toGet_arr[0]);
    myfree(toGo_arr[0]);

    return failure;
}

static int domain_exchange_once(int (*layoutfunc)(int p), int** toGo_arr, int ** toGet_arr)
{
    int i, j, target;
    struct particle_data *partBuf;
    struct sph_particle_data *sphBuf;
    struct bh_particle_data *bhBuf;
    struct star_particle_data *starBuf;

    int *ctmem = mymalloc2("cts", 4*NSP*NTask*sizeof(int));
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

    partBuf = (struct particle_data *) mymalloc2("partBuf", count_togo[0] * sizeof(struct particle_data));
    sphBuf = (struct sph_particle_data *) mymalloc2("sphBuf", count_togo[1] * sizeof(struct sph_particle_data));
    bhBuf = (struct bh_particle_data *) mymalloc2("bhBuf", count_togo[2] * sizeof(struct bh_particle_data));
    starBuf = (struct star_particle_data *) mymalloc2("starBuf", count_togo[3] * sizeof(struct star_particle_data));

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

    if(N_bh_slots > All.MaxPartBh - All.BlackHoleOn * 0.005 * All.MaxPart ||
                N_star_slots > All.MaxPartStar - All.StarformationOn * 0.005* All.MaxPart) {
        int newStar = 1.5*(N_star_slots + All.StarformationOn* 0.005*All.MaxPart);
        int newBh = 1.5*(N_bh_slots + All.BlackHoleOn * 0.005*All.MaxPart);
        message(1, "Need more stars and BHs: (%d, %d) -> (%d, %d)\n", All.MaxPartStar, All.MaxPartBh, newStar, newBh);
        realloc_secondary_data(newBh, newStar);
    }

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
static int domain_countToGo(ptrdiff_t nlimit, int (*layoutfunc)(int p), int failfast, int** toGo_arr, int** toGet_arr)
{
    int n, i, ret;
    size_t package;

    for(i=0; i<NSP; i++)
        for(n = 0; n < NTask; n++)
            toGo_arr[i][n] = 0;

    package = (sizeof(struct particle_data) + sizeof(struct sph_particle_data) + sizeof(struct bh_particle_data)+sizeof(struct star_particle_data));
    if(package >= nlimit)
        endrun(212, "Package is too large, no free memory.");


    for(n = 0; n < NumPart; n++)
    {
        if(package >= nlimit) break;
        if(!P[n].OnAnotherDomain) continue;

        int target = layoutfunc(n);
        if (target == ThisTask) continue;

        toGo_arr[0][target] += 1;
        nlimit -= sizeof(struct particle_data);

        if(P[n].Type  == 0)
        {
            toGo_arr[1][target] += 1;
            nlimit -= sizeof(struct sph_particle_data);
        }
        if(P[n].Type  == 5)
        {
            toGo_arr[2][target] += 1;
            nlimit -= sizeof(struct bh_particle_data);
        }
        if(P[n].Type  == 4)
        {
            toGo_arr[3][target] += 1;
            nlimit -= sizeof(struct star_particle_data);
        }
        P[n].WillExport = 1;	/* flag this particle for export */
    }

    for(i=0; i<NSP; i++)
        MPI_Alltoall(toGo_arr[i], 1, MPI_INT, toGet_arr[i], 1, MPI_INT, MPI_COMM_WORLD);

    ret = (package >= nlimit);

    MPI_Allreduce(MPI_IN_PLACE, &ret, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);

    if(ret == 0)
        return 0;

    if(failfast)
        return 1;

    {
        /* in this case, we are not guaranteed that the temporary state after
           the partial exchange will actually observe the particle limits on all
           processors... we need to test this explicitly and rework the exchange
           such that this is guaranteed. This is actually a rather non-trivial
           constraint. */

        int flagsum;
        /*Order is: total, sph, bh, star*/
        int *togo_local[NSP];
        int * list_Npart[NSP];
        for(n=0; n<NSP; n++)
            togo_local[n] = toGo_arr[n];
        list_Npart[0] = (int *)mymalloc("list_Npart", NSP*NTask * sizeof(int));
        for(n=1; n<NSP; n++)
            list_Npart[n] = list_Npart[n-1]+NTask;
        MPI_Allgather(&NumPart, 1, MPI_INT, list_Npart[0], 1, MPI_INT, MPI_COMM_WORLD);
        MPI_Allgather(&N_sph_slots, 1, MPI_INT, list_Npart[1], 1, MPI_INT, MPI_COMM_WORLD);
        MPI_Allgather(&N_bh_slots, 1, MPI_INT, list_Npart[2], 1, MPI_INT, MPI_COMM_WORLD);
        MPI_Allgather(&N_star_slots, 1, MPI_INT, list_Npart[3], 1, MPI_INT, MPI_COMM_WORLD);

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
                        for(n = 0; n < NSP; n++)
                            for(i = 0; i < NTask; i++)
                            {
                                count_togo[n] += toGo_arr[n][i];
                                count_toget[n] += toGet_arr[n][i];
                            }
                    }
                    MPI_Bcast(&count_togo, NSP, MPI_INT, ta, MPI_COMM_WORLD);
                    MPI_Bcast(&count_toget, NSP, MPI_INT, ta, MPI_COMM_WORLD);
                    for(i=NSP-1; i > 0; --i) {
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
                int * new_toGo[NSP];
                new_toGo[0] = (int *)mymalloc("local_toGo", NSP*NTask * sizeof(int));
                memset(new_toGo, 0, NSP*NTask*sizeof(int));
                for(n=1; n<NSP; n++)
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
                    for(i=0; i< NSP; i++)
                        toGo_arr[i][n] = new_toGo[i][n];

                for(i=0; i< NSP; i++)
                    MPI_Alltoall(toGo_arr[i], 1, MPI_INT, toGet_arr[i], 1, MPI_INT, MPI_COMM_WORLD);
                myfree(new_toGo[0]);
            }
        }
        while(flagsum);
        myfree(list_Npart[0]);

    }
    return ret;
}

static void
realloc_secondary_data(int newMaxPartBh, int newMaxPartStar)
{
    size_t bytes = newMaxPartStar * sizeof(struct star_particle_data) + newMaxPartBh * sizeof(struct bh_particle_data);
    BhP = myrealloc(BhP, bytes);
    StarP = (struct star_particle_data *) (BhP + newMaxPartBh);
    size_t mvstar = (All.MaxPartStar < newMaxPartStar) ? All.MaxPartStar : newMaxPartStar;
    /* We moved the data in realloc, but we still need to shift up the stars for the new number of BHs.
     * Must use addressing relative to new BhP pointer, as StarP was invalidated by the move in realloc*/
    memmove(StarP, BhP + All.MaxPartBh, mvstar * sizeof(struct star_particle_data));
    All.MaxPartBh = newMaxPartBh;
    All.MaxPartStar = newMaxPartStar;
    message(1, "Allocated %g MB for %d stars and %d BHs.\n", bytes / (1024.0 * 1024.0), newMaxPartStar, newMaxPartBh);
}
