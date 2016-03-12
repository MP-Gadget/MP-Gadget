#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <math.h>


#include "allvars.h"
#include "proto.h"
#include "domain.h"
#include "forcetree.h"
#include "mymalloc.h"
#include "mpsort.h"

#define TAG_GRAV_A        18
#define TAG_GRAV_B        19

/*! \file domain.c
 *  \brief code for domain decomposition
 *
 *  This file contains the code for the domain decomposition of the
 *  simulation volume.  The domains are constructed from disjoint subsets
 *  of the leaves of a fiducial top-level tree that covers the full
 *  simulation volume. Domain boundaries hence run along tree-node
 *  divisions of a fiducial global BH tree. As a result of this method, the
 *  tree force are in principle strictly independent of the way the domains
 *  are cut. The domain decomposition can be carried out for an arbitrary
 *  number of CPUs. Individual domains are not cubical, but spatially
 *  coherent since the leaves are traversed in a Peano-Hilbert order and
 *  individual domains form segments along this order.  This also ensures
 *  that each domain has a small surface to volume ratio, which minimizes
 *  communication.
 */


#define REDUC_FAC      0.98

double DomainCorner[3], DomainCenter[3], DomainLen, DomainFac;
int *DomainStartList, *DomainEndList;



double *DomainWork;
int *DomainCount;
int *DomainCountSph;
int *DomainTask;
int *DomainNodeIndex;
int *DomainList, DomainNumChanged;

struct topnode_data *TopNodes;

int NTopnodes, NTopleaves;


/*! toGo[task*NTask + partner] gives the number of particles in task 'task'
 *  that have to go to task 'partner'
 */
static int *toGo, *toGoSph, *toGoBh;
static int *toGet, *toGetSph, *toGetBh;
static int *list_load;
static int *list_loadsph;
static double *list_work;
static double *list_speedfac;
static double *list_cadj_cpu;
static double *list_cadj_cost;

static struct local_topnode_data
{
    peanokey Size;		/*!< number of Peano-Hilbert mesh-cells represented by top-level node */
    peanokey StartKey;		/*!< first Peano-Hilbert key in top-level node */
    int64_t Count;		/*!< counts the number of particles in this top-level node */
    double Cost;
    int Daughter;			/*!< index of first daughter cell (out of 8) of top-level node */
    int Leaf;			/*!< if the node is a leaf, this gives its number when all leaves are traversed in Peano-Hilbert order */
    int Parent;
    int PIndex;			/*!< first particle in node  used only in top-level tree build (this file)*/
}
*topNodes;			/*!< points to the root node of the top-level tree */

static struct peano_hilbert_data
{
    peanokey key;
    int index;
}
*mp;




static void domain_insertnode(struct local_topnode_data *treeA, struct local_topnode_data *treeB, int noA,
        int noB);
static void domain_add_cost(struct local_topnode_data *treeA, int noA, int64_t count, double cost);

static int domain_layoutfunc(int n);
static int domain_countToGo(ptrdiff_t nlimit, int (*layoutfunc)(int p));
static void domain_exchange_once(int (*layoutfunc)(int p) );


static float *domainWork;	/*!< a table that gives the total "work" due to the particles stored by each processor */
static int *domainCount;	/*!< a table that gives the total number of particles held by each processor */
static int *domainCountSph;	/*!< a table that gives the total number of SPH particles held by each processor */

static int domain_allocated_flag = 0;

static int maxLoad, maxLoadsph;

static double totgravcost, totpartcount, gravcost;

static MPI_Datatype MPI_TYPE_PARTICLE = 0;
static MPI_Datatype MPI_TYPE_SPHPARTICLE = 0;
static MPI_Datatype MPI_TYPE_BHPARTICLE = 0;

/*! This is the main routine for the domain decomposition.  It acts as a
 *  driver routine that allocates various temporary buffers, maps the
 *  particles back onto the periodic box if needed, and then does the
 *  domain decomposition, and a final Peano-Hilbert order of all particles
 *  as a tuning measure.
 */
void domain_Decomposition(void)
{
    int i, ret, retsum;
    size_t bytes, all_bytes;
    double t0, t1;

    /* register the mpi types used in communication if not yet. */
    if (MPI_TYPE_PARTICLE == 0) {
        MPI_Type_contiguous(sizeof(struct particle_data), MPI_BYTE, &MPI_TYPE_PARTICLE);
        MPI_Type_contiguous(sizeof(struct bh_particle_data), MPI_BYTE, &MPI_TYPE_BHPARTICLE);
        MPI_Type_contiguous(sizeof(struct sph_particle_data), MPI_BYTE, &MPI_TYPE_SPHPARTICLE);
        MPI_Type_commit(&MPI_TYPE_PARTICLE);
        MPI_Type_commit(&MPI_TYPE_BHPARTICLE);
        MPI_Type_commit(&MPI_TYPE_SPHPARTICLE);
    }

        walltime_measure("/Misc");

        move_particles(All.Ti_Current);

        force_treefree();
        domain_free();

#if defined(SFR) || defined(BLACK_HOLES)
        rearrange_particle_sequence();
#endif

        do_box_wrapping();	/* map the particles back onto the box */

        All.NumForcesSinceLastDomainDecomp = 0;

        if(ThisTask == 0)
        {
            printf("domain decomposition... (presently allocated=%g MB)\n", AllocatedBytes / (1024.0 * 1024.0));
            fflush(stdout);
        }

        t0 = second();

        do
        {
            domain_allocate();

            all_bytes = 0;

            list_cadj_cpu = (double *) mymalloc("list_cadj_cpu", bytes = (sizeof(double) * NTask));
            all_bytes += bytes;
            list_cadj_cost = (double *) mymalloc("list_cadj_cost", bytes = (sizeof(double) * NTask));
            all_bytes += bytes;
            list_load = (int *) mymalloc("list_load", bytes = (sizeof(int) * NTask));
            all_bytes += bytes;
            list_loadsph = (int *) mymalloc("list_loadsph", bytes = (sizeof(int) * NTask));
            all_bytes += bytes;
            list_work = (double *) mymalloc("list_work", bytes = (sizeof(double) * NTask));
            all_bytes += bytes;
            list_speedfac = (double *) mymalloc("list_speedfac", bytes = (sizeof(double) * NTask));
            all_bytes += bytes;
            domainWork = (float *) mymalloc("domainWork", bytes = (MaxTopNodes * sizeof(float)));
            all_bytes += bytes;
            domainCount = (int *) mymalloc("domainCount", bytes = (MaxTopNodes * sizeof(int)));
            all_bytes += bytes;
            domainCountSph = (int *) mymalloc("domainCountSph", bytes = (MaxTopNodes * sizeof(int)));
            all_bytes += bytes;

            topNodes = (struct local_topnode_data *) mymalloc("topNodes", bytes =
                    (MaxTopNodes *
                     sizeof(struct local_topnode_data)));
            all_bytes += bytes;

            if(ThisTask == 0)
            {
                printf
                    ("use of %g MB of temporary storage for domain decomposition... (presently allocated=%g MB)\n",
                     all_bytes / (1024.0 * 1024.0), AllocatedBytes / (1024.0 * 1024.0));
                fflush(stdout);
            }

            maxLoad = (int) (All.MaxPart * REDUC_FAC);
            maxLoadsph = (int) (All.MaxPartSph * REDUC_FAC);

            report_memory_usage("DOMAIN");

            ret = domain_decompose();

            /* copy what we need for the topnodes */
            for(i = 0; i < NTopnodes; i++)
            {
                TopNodes[i].StartKey = topNodes[i].StartKey;
                TopNodes[i].Size = topNodes[i].Size;
                TopNodes[i].Daughter = topNodes[i].Daughter;
                TopNodes[i].Leaf = topNodes[i].Leaf;
            }

            myfree(topNodes);
            myfree(domainCountSph);
            myfree(domainCount);
            myfree(domainWork);
            myfree(list_speedfac);
            myfree(list_work);
            myfree(list_loadsph);
            myfree(list_load);
            myfree(list_cadj_cost);
            myfree(list_cadj_cpu);

            MPI_Allreduce(&ret, &retsum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
            if(retsum)
            {
                domain_free();
                if(ThisTask == 0)
                    printf("Increasing TopNodeAllocFactor=%g  ", All.TopNodeAllocFactor);

                All.TopNodeAllocFactor *= 1.3;

                if(ThisTask == 0)
                {
                    printf("new value=%g\n", All.TopNodeAllocFactor);
                    fflush(stdout);
                }

                if(All.TopNodeAllocFactor > 1000)
                {
                    if(ThisTask == 0)
                        printf("something seems to be going seriously wrong here. Stopping.\n");
                    fflush(stdout);
                    endrun(781);
                }
            }
        }
        while(retsum);

        t1 = second();

        if(ThisTask == 0)
        {
            printf("domain decomposition done. (took %g sec)\n", timediff(t0, t1));
            fflush(stdout);
        }

#ifdef PEANOHILBERT
        peano_hilbert_order();

        walltime_measure("/Domain/Peano");
#endif

        memmove(TopNodes + NTopnodes, DomainTask, NTopnodes * sizeof(int));

        TopNodes = (struct topnode_data *) myrealloc(TopNodes, bytes =
                (NTopnodes * sizeof(struct topnode_data) +
                 NTopnodes * sizeof(int)));
        if(ThisTask == 0)
            printf("Freed %g MByte in top-level domain structure\n",
                    (MaxTopNodes - NTopnodes) * sizeof(struct topnode_data) / (1024.0 * 1024.0));

        DomainTask = (int *) (TopNodes + NTopnodes);

        reconstruct_timebins();
        walltime_measure("/Domain/Misc");
}

/*! This function allocates all the stuff that will be required for the tree-construction/walk later on */
void domain_allocate(void)
{
    size_t bytes, all_bytes = 0;

    MaxTopNodes = (int) (All.TopNodeAllocFactor * All.MaxPart + 1);

    DomainStartList = (int *) mymalloc("DomainStartList", bytes = (NTask * MULTIPLEDOMAINS * sizeof(int)));
    all_bytes += bytes;

    DomainEndList = (int *) mymalloc("DomainEndList", bytes = (NTask * MULTIPLEDOMAINS * sizeof(int)));
    all_bytes += bytes;

    TopNodes = (struct topnode_data *) mymalloc("TopNodes", bytes =
            (MaxTopNodes * sizeof(struct topnode_data) +
             MaxTopNodes * sizeof(int)));
    all_bytes += bytes;

    DomainTask = (int *) (TopNodes + MaxTopNodes);

    if(ThisTask == 0)
        printf("Allocated %g MByte for top-level domain structure\n", all_bytes / (1024.0 * 1024.0));

    domain_allocated_flag = 1;
}

void domain_free(void)
{
    if(domain_allocated_flag)
    {
        myfree(TopNodes);
        myfree(DomainEndList);
        myfree(DomainStartList);
        domain_allocated_flag = 0;
    }
}

static struct topnode_data *save_TopNodes;
static int *save_DomainStartList, *save_DomainEndList;

void domain_free_trick(void)
{
    if(domain_allocated_flag)
    {
        save_TopNodes = TopNodes;
        save_DomainEndList = DomainEndList;
        save_DomainStartList = DomainStartList;
        domain_allocated_flag = 0;
    }
    else
        endrun(131231);
}

void domain_allocate_trick(void)
{
    domain_allocated_flag = 1;

    TopNodes = save_TopNodes;
    DomainEndList = save_DomainEndList;
    DomainStartList = save_DomainStartList;
}




double domain_particle_costfactor(int i)
{
    if(P[i].TimeBin)
        return (1.0 + P[i].GravCost) / (1 << P[i].TimeBin);
    else
        return (1.0 + P[i].GravCost) / TIMEBASE;
}


/*! This function carries out the actual domain decomposition for all
 *  particle types. It will try to balance the work-load for each domain,
 *  as estimated based on the P[i]-GravCost values.  The decomposition will
 *  respect the maximum allowed memory-imbalance given by the value of
 *  PartAllocFactor.
 */
int domain_decompose(void)
{

    int64_t Ntype[6];		/*!< total number of particles of each type */
    int NtypeLocal[6];		/*!< local number of particles of each type */

    int i, status;
    int64_t sumload;
    int maxload;
    double sumwork, sumcpu, sumcost, maxwork, cadj_SpeedFac;


    walltime_measure("/Domain/Decompose/Misc");
#ifdef CPUSPEEDADJUSTMENT
    double min_load, sum_speedfac;
#endif

    for(i = 0; i < 6; i++)
        NtypeLocal[i] = 0;

    gravcost = 0;
#pragma omp parallel private(i)
    {
        int NtypeLocalThread[6] = {0};
        double mygravcost = 0;
#pragma omp for
        for(i = 0; i < NumPart; i++)
        {
            NtypeLocalThread[P[i].Type]++;
            double costfac = domain_particle_costfactor(i);

            mygravcost += costfac;
        }
#pragma omp critical 
        {
/* avoid omp reduction for now: Craycc doesn't always do it right */
            gravcost += mygravcost;
            for(i = 0; i < 6; i ++) {
                NtypeLocal[i] += NtypeLocalThread[i];
            }
        }
    }
    All.Cadj_Cost += gravcost;
    /* because Ntype[] is of type `int64_t', we cannot do a simple
     * MPI_Allreduce() to sum the total particle numbers 
     */
    sumup_large_ints(6, NtypeLocal, Ntype);

    for(i = 0, totpartcount = 0; i < 6; i++)
        totpartcount += Ntype[i];


    MPI_Allreduce(&gravcost, &totgravcost, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    All.Cadj_Cpu *= 0.9;
    All.Cadj_Cost *= 0.9;

    MPI_Allgather(&All.Cadj_Cpu, 1, MPI_DOUBLE, list_cadj_cpu, 1, MPI_DOUBLE, MPI_COMM_WORLD);
    MPI_Allgather(&All.Cadj_Cost, 1, MPI_DOUBLE, list_cadj_cost, 1, MPI_DOUBLE, MPI_COMM_WORLD);

#ifdef CPUSPEEDADJUSTMENT
    MPI_Allreduce(&All.Cadj_Cost, &min_load, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    if(min_load > 0)
    {
        cadj_SpeedFac = All.Cadj_Cpu / All.Cadj_Cost;

        MPI_Allreduce(&cadj_SpeedFac, &sum_speedfac, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        cadj_SpeedFac /= (sum_speedfac / NTask);
    }
    else
        cadj_SpeedFac = 1;

    MPI_Allgather(&cadj_SpeedFac, 1, MPI_DOUBLE, list_speedfac, 1, MPI_DOUBLE, MPI_COMM_WORLD);
#else
    cadj_SpeedFac = 1;
    MPI_Allgather(&cadj_SpeedFac, 1, MPI_DOUBLE, list_speedfac, 1, MPI_DOUBLE, MPI_COMM_WORLD);
#endif

    /* determine global dimensions of domain grid */
    domain_findExtent();

    walltime_measure("/Domain/Decompose/FindExtent");

    if(domain_determineTopTree())
        return 1;
    /* find the split of the domain grid */
    domain_findSplit_work_balanced(MULTIPLEDOMAINS * NTask, NTopleaves);

    walltime_measure("/Domain/Decompose/findworksplit");

    domain_assign_load_or_work_balanced(1);

    walltime_measure("/Domain/Decompose/assignbalance");

    status = domain_check_memory_bound();
    walltime_measure("/Domain/Decompose/memorybound");

    if(status != 0)		/* the optimum balanced solution violates memory constraint, let's try something different */
    {
        if(ThisTask == 0)
            printf
                ("Note: the domain decomposition is suboptimum because the ceiling for memory-imbalance is reached\n");

        domain_findSplit_load_balanced(MULTIPLEDOMAINS * NTask, NTopleaves);

        walltime_measure("/Domain/Decompose/findloadsplit");
        domain_assign_load_or_work_balanced(0);
        walltime_measure("/Domain/Decompose/assignbalance");

        status = domain_check_memory_bound();
        walltime_measure("/Domain/Decompose/memorybound");

        if(status != 0)
        {
            if(ThisTask == 0)
                printf("No domain decomposition that stays within memory bounds is possible.\n");
            endrun(0);
        }
    }


    if(ThisTask == 0)
    {
        sumload = maxload = 0;
        sumwork = sumcpu = sumcost = maxwork = 0;
        for(i = 0; i < NTask; i++)
        {
            sumload += list_load[i];
            sumwork += list_speedfac[i] * list_work[i];
            sumcpu += list_cadj_cpu[i];
            sumcost += list_cadj_cost[i];

            if(list_load[i] > maxload)
                maxload = list_load[i];

            if(list_speedfac[i] * list_work[i] > maxwork)
                maxwork = list_speedfac[i] * list_work[i];
        }

        printf("work-load balance=%g   memory-balance=%g\n",
                maxwork / (sumwork / NTask), maxload / (((double) sumload) / NTask));

#ifdef VERBOSE
        printf("Speedfac:\n");
        for(i = 0; i < NTask; i++)
        {
            printf("Speedfac [%3d]  speedfac=%8.4f  work=%8.4f   load=%8.4f   cpu=%8.4f   cost=%8.4f \n", i,
                    list_speedfac[i], list_speedfac[i] * list_work[i] / (sumwork / NTask),
                    list_load[i] / (((double) sumload) / NTask), list_cadj_cpu[i] / (sumcpu / NTask),
                    list_cadj_cost[i] / (sumcost / NTask));
        }
#endif
    }

    walltime_measure("/Domain/Decompose/Misc");
    domain_exchange(domain_layoutfunc);
    return 0;
}

void checklock() {
    int j;
#ifdef OPENMP_USE_SPINLOCK
    for(j = 0; j < All.MaxPart; j++) {
        if(0 == P[j].SpinLock) {
            printf("lock failed %d, %d\n", j, P[j].SpinLock);
            endrun(12312314); 
        }
    }
#endif
}
/* 
 * 
 * exchange particles according to layoutfunc.
 * layoutfunc gives the target task of particle p.
*/

void domain_exchange(int (*layoutfunc)(int p)) {
    int i;
    int64_t sumtogo;
    /* flag the particles that need to be exported */
    toGo = (int *) mymalloc("toGo", (sizeof(int) * NTask));
    toGoSph = (int *) mymalloc("toGoSph", (sizeof(int) * NTask));
    toGoBh = (int *) mymalloc("toGoBh", (sizeof(int) * NTask));
    toGet = (int *) mymalloc("toGet", (sizeof(int) * NTask));
    toGetSph = (int *) mymalloc("toGetSph", (sizeof(int) * NTask));
    toGetBh = (int *) mymalloc("toGetBh", (sizeof(int) * NTask));


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
            printf("task=%d: exchange_limit=%d\n", ThisTask, (int) exchange_limit);
            endrun(1223);
        }

        /* determine for each cpu how many particles have to be shifted to other cpus */
        ret = domain_countToGo(exchange_limit, layoutfunc);
        walltime_measure("/Domain/exchange/togo");

        for(i = 0, sumtogo = 0; i < NTask; i++)
            sumtogo += toGo[i];

        sumup_longs(1, &sumtogo, &sumtogo);

        if(ThisTask == 0)
        {
            printf("iter=%d exchange of %d%09d particles\n", iter,
                    (int) (sumtogo / 1000000000), (int) (sumtogo % 1000000000));
            fflush(stdout);
        }

        domain_exchange_once(layoutfunc);
        iter++;
    }
    while(ret > 0);

    myfree(toGetBh);
    myfree(toGetSph);
    myfree(toGet);
    myfree(toGoBh);
    myfree(toGoSph);
    myfree(toGo);

}


int domain_check_memory_bound(void)
{
    int ta, m, i;
    int load, sphload, max_load, max_sphload;
    double work;

    max_load = max_sphload = 0;

    for(ta = 0; ta < NTask; ta++)
    {
        load = sphload = 0;
        work = 0;

        for(m = 0; m < MULTIPLEDOMAINS; m++)
            for(i = DomainStartList[ta * MULTIPLEDOMAINS + m]; i <= DomainEndList[ta * MULTIPLEDOMAINS + m]; i++)
            {
                load += domainCount[i];
                sphload += domainCountSph[i];
                work += domainWork[i];
            }

        list_load[ta] = load;
        list_loadsph[ta] = sphload;
        list_work[ta] = work;

        if(load > max_load)
            max_load = load;
        if(sphload > max_sphload)
            max_sphload = sphload;
    }


    if(max_load > maxLoad)
    {
        if(ThisTask == 0)
        {
            printf("desired memory imbalance=%g  (limit=%d, needed=%d)\n",
                    (max_load * All.PartAllocFactor) / maxLoad, maxLoad, max_load);
        }

        return 1;
    }

    if(max_sphload > maxLoadsph)
    {
        if(ThisTask == 0)
        {
            printf("desired memory imbalance=%g  (SPH) (limit=%d, needed=%d)\n",
                    (max_sphload * All.PartAllocFactor) / maxLoadsph, maxLoadsph, max_sphload);
        }

        return 1;
    }

    return 0;
}

static void domain_exchange_once(int (*layoutfunc)(int p))
{
    int count_togo = 0, count_togo_sph = 0, count_togo_bh = 0, 
        count_get = 0, count_get_sph = 0, count_get_bh = 0;
    int *count, *count_sph, *count_bh,
        *offset, *offset_sph, *offset_bh;
    int *count_recv, *count_recv_sph, *count_recv_bh,
        *offset_recv, *offset_recv_sph, *offset_recv_bh;
    int i, n, target;
    struct particle_data *partBuf;
    struct sph_particle_data *sphBuf;
    struct bh_particle_data *bhBuf;

    count = (int *) mymalloc("count", NTask * sizeof(int));
    count_sph = (int *) mymalloc("count_sph", NTask * sizeof(int));
    count_bh = (int *) mymalloc("count_bh", NTask * sizeof(int));
    offset = (int *) mymalloc("offset", NTask * sizeof(int));
    offset_sph = (int *) mymalloc("offset_sph", NTask * sizeof(int));
    offset_bh = (int *) mymalloc("offset_bh", NTask * sizeof(int));

    count_recv = (int *) mymalloc("count_recv", NTask * sizeof(int));
    count_recv_sph = (int *) mymalloc("count_recv_sph", NTask * sizeof(int));
    count_recv_bh = (int *) mymalloc("count_recv_bh", NTask * sizeof(int));
    offset_recv = (int *) mymalloc("offset_recv", NTask * sizeof(int));
    offset_recv_sph = (int *) mymalloc("offset_recv_sph", NTask * sizeof(int));
    offset_recv_bh = (int *) mymalloc("offset_recv_bh", NTask * sizeof(int));

    for(i = 1, offset_sph[0] = 0; i < NTask; i++)
        offset_sph[i] = offset_sph[i - 1] + toGoSph[i - 1];

    for(i = 1, offset_bh[0] = 0; i < NTask; i++)
        offset_bh[i] = offset_bh[i - 1] + toGoBh[i - 1];

    offset[0] = offset_sph[NTask - 1] + toGoSph[NTask - 1];

    for(i = 1; i < NTask; i++)
        offset[i] = offset[i - 1] + (toGo[i - 1] - toGoSph[i - 1]);

    for(i = 0; i < NTask; i++)
    {
        count_togo += toGo[i];
        count_togo_sph += toGoSph[i];
        count_togo_bh += toGoBh[i];

        count_get += toGet[i];
        count_get_sph += toGetSph[i];
        count_get_bh += toGetBh[i];
    }

    partBuf = (struct particle_data *) mymalloc("partBuf", count_togo * sizeof(struct particle_data));
    sphBuf = (struct sph_particle_data *) mymalloc("sphBuf", count_togo_sph * sizeof(struct sph_particle_data));
    bhBuf = (struct bh_particle_data *) mymalloc("bhBuf", count_togo_bh * sizeof(struct bh_particle_data));

    for(i = 0; i < NTask; i++)
        count[i] = count_sph[i] = count_bh[i] = 0;

    /*FIXME: make this omp ! */
    for(n = 0; n < NumPart; n++)
    {
        if(!(P[n].OnAnotherDomain && P[n].WillExport)) continue;
        /* preparing for export */
        P[n].OnAnotherDomain = 0;
        P[n].WillExport = 0;
        target = layoutfunc(n);

        if(P[n].Type == 0)
        {
            partBuf[offset_sph[target] + count_sph[target]] = P[n];
            sphBuf[offset_sph[target] + count_sph[target]] = SPHP(n);
            count_sph[target]++;
        } else
        if(P[n].Type == 5)
        {
            bhBuf[offset_bh[target] + count_bh[target]] = BhP[P[n].PI];
            /* points to the subbuffer */
            P[n].PI = count_bh[target];
            partBuf[offset[target] + count[target]] = P[n];
            count_bh[target]++;
            count[target]++;
        }
        else
        {
            partBuf[offset[target] + count[target]] = P[n];
            count[target]++;
        }


        if(P[n].Type == 0)
        {
            P[n] = P[N_sph - 1];
            P[N_sph - 1] = P[NumPart - 1];
            /* Because SphP doesn't use PI */
            SPHP(n) = SPHP(N_sph - 1);

            NumPart--;
            N_sph--;
            n--;
        }
        else
        {
            P[n] = P[NumPart - 1];
            NumPart--;
            n--;
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

    if(count_get_sph)
    {
        memmove(P + N_sph + count_get_sph, P + N_sph, (NumPart - N_sph) * sizeof(struct particle_data));
    }

    for(i = 0; i < NTask; i++)
    {
        count_recv_sph[i] = toGetSph[i];
        count_recv_bh[i] = toGetBh[i];
        count_recv[i] = toGet[i] - toGetSph[i];
    }

    for(i = 1, offset_recv_sph[0] = N_sph; i < NTask; i++)
        offset_recv_sph[i] = offset_recv_sph[i - 1] + count_recv_sph[i - 1];

    for(i = 1, offset_recv_bh[0] = N_bh; i < NTask; i++)
        offset_recv_bh[i] = offset_recv_bh[i - 1] + count_recv_bh[i - 1];

    offset_recv[0] = NumPart + count_get_sph;

    for(i = 1; i < NTask; i++)
        offset_recv[i] = offset_recv[i - 1] + count_recv[i - 1];

    MPI_Alltoallv_sparse(partBuf, count_sph, offset_sph, MPI_TYPE_PARTICLE,
                 P, count_recv_sph, offset_recv_sph, MPI_TYPE_PARTICLE,
                 MPI_COMM_WORLD);
    walltime_measure("/Domain/exchange/alltoall");

    MPI_Alltoallv_sparse(sphBuf, count_sph, offset_sph, MPI_TYPE_SPHPARTICLE,
                 SphP, count_recv_sph, offset_recv_sph, MPI_TYPE_SPHPARTICLE,
                 MPI_COMM_WORLD);
    walltime_measure("/Domain/exchange/alltoall");

    MPI_Alltoallv_sparse(partBuf, count, offset, MPI_TYPE_PARTICLE,
                 P, count_recv, offset_recv, MPI_TYPE_PARTICLE,
                 MPI_COMM_WORLD);
    walltime_measure("/Domain/exchange/alltoall");

    MPI_Alltoallv_sparse(bhBuf, count_bh, offset_bh, MPI_TYPE_BHPARTICLE,
                BhP, count_recv_bh, offset_recv_bh, MPI_TYPE_BHPARTICLE,
                MPI_COMM_WORLD);
    walltime_measure("/Domain/exchange/alltoall");
                
    for(target = 0; target < NTask; target++) {
        int i, j;
        for(i = offset_recv[target], 
                j = offset_recv_bh[target]; 
                i < offset_recv[target] + count_recv[target]; 
                i++) {
            if(P[i].Type != 5) continue;
            P[i].PI = j;
            j++;
        }
        if(j != count_recv_bh[target] + offset_recv_bh[target]) {
            printf("communitate bh consitency\n");
            endrun(99999);
        }
    }

    NumPart += count_get;
    N_sph += count_get_sph;
    N_bh += count_get_bh;

    if(NumPart > All.MaxPart)
    {
        printf("Task=%d NumPart=%d All.MaxPart=%d\n", ThisTask, NumPart, All.MaxPart);
        endrun(787878);
    }

    if(N_sph > All.MaxPartSph)
        endrun(787879);
    if(N_bh > All.MaxPartBh)
        endrun(787879);

    myfree(bhBuf);
    myfree(sphBuf);
    myfree(partBuf);
    myfree(offset_recv_bh);
    myfree(offset_recv_sph);
    myfree(offset_recv);
    myfree(count_recv_bh);
    myfree(count_recv_sph);
    myfree(count_recv);

    myfree(offset_bh);
    myfree(offset_sph);
    myfree(offset);
    myfree(count_bh);
    myfree(count_sph);
    myfree(count);

    if(ThisTask == 0) {
        fprintf(stderr, "checking ID consistency after exchange\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);

    domain_garbage_collection_bh();
    walltime_measure("/Domain/exchange/finalize");
}
static int bh_cmp_reverse_link(const void * b1in, const void * b2in) {
    const struct bh_particle_data * b1 = (struct bh_particle_data *) b1in;
    const struct bh_particle_data * b2 = (struct bh_particle_data *) b2in;
    if(b1->ReverseLink == -1 && b2->ReverseLink == -1) {
        return 0;
    }
    if(b1->ReverseLink == -1) return 1;
    if(b2->ReverseLink == -1) return -1;
    return (b1->ReverseLink > b2->ReverseLink) - (b1->ReverseLink < b2->ReverseLink);

}

void domain_garbage_collection_bh() {

    /* gc the bh */
    int i, j;
    int total = 0;

    int total0 = 0;

    MPI_Reduce(&N_bh, &total0, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    /* no need to gc if there is no bh to begin with*/
    if (N_bh == 0) goto ex_nobh;

#pragma omp parallel for
    for(i = 0; i < All.MaxPartBh; i++) {
        BhP[i].ReverseLink = -1;
    }
#pragma omp parallel for
    for(i = 0; i < NumPart; i++) {
        if(P[i].Type != 5) continue;
        BhP[P[i].PI].ReverseLink = i;
        if(P[i].PI >= N_bh) {
            printf("bh PI consistency failed1\n");
            endrun(99999); 
        }
        if(BhP[P[i].PI].ID != P[i].ID) {
            printf("bh id consistency failed1\n");
            endrun(99999); 
        }
    }

    /* put unused guys to the end, and sort the used ones
     * by their location in the P array */
    qsort(BhP, N_bh, sizeof(BhP[0]), bh_cmp_reverse_link);

    while(N_bh > 0 && BhP[N_bh - 1].ReverseLink == -1) {
        N_bh --;
    }

    /* Now update the link in BhP */
    for(i = 0; i < N_bh; i ++) {
        P[BhP[i].ReverseLink].PI = i;
    }

    /* Now invalidate ReverseLink */
    for(i = 0; i < N_bh; i ++) {
        BhP[i].ReverseLink = -1;
    }

    j = 0;
#pragma omp parallel for
    for(i = 0; i < NumPart; i++) {
        if(P[i].Type != 5) continue;
        if(P[i].PI >= N_bh) {
            printf("bh PI consistency failed2\n");
            endrun(99999); 
        }
        if(BhP[P[i].PI].ID != P[i].ID) {
            printf("bh id consistency failed2\n");
            endrun(99999); 
        }
#pragma omp atomic
        j ++;
    }
    if(j != N_bh) {
            printf("bh count failed2, j=%d, N_bh=%d\n", j, N_bh);
            endrun(99999); 
    }

ex_nobh:
    MPI_Reduce(&N_bh, &total, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    if(ThisTask == 0 && total != total0) {
        printf("After BH garbage collection, before = %d after= %d\n", total0, total);
    }
}

int domain_fork_particle(int parent) {
    /* this will fork a zero mass particle at the given location of parent.
     *
     * Assumes the particle is protected by locks in threaded env.
     *
     * The Generation of parent is incremented.
     * The child carries the incremented generation number.
     * The ID of the child is modified, with the new generation number set
     * at the highest 8 bits.
     *
     * the new particle's index is returned.
     *
     * Its mass and ptype can be then adjusted. (watchout detached BH /SPH
     * data!)
     * It's PIndex still points to the old Pindex!
     * */

    if(NumPart >= All.MaxPart)
    {
        printf
            ("On Task=%d with NumPart=%d we try to spawn. Sorry, no space left...(All.MaxPart=%d)\n",
             ThisTask, NumPart, All.MaxPart);
        fflush(stdout);
        endrun(8888);
    }
    int child = atomic_fetch_and_add(&NumPart, 1);
    
    NextActiveParticle[child] = FirstActiveParticle;
    FirstActiveParticle = child;

    P[parent].Generation ++;
    uint64_t g = P[parent].Generation;
    /* change the child ID according to the generation. */
    P[child] = P[parent];
    P[child].ID = (P[parent].ID & 0x00ffffffffffffffL) + (g << 56L);

    /* the PIndex still points to the old PIndex */
    P[child].Mass = 0;

    /* FIXME: these are not thread safe !!not !!*/
    TimeBinCount[P[child].TimeBin]++;

    PrevInTimeBin[child] = parent;
    NextInTimeBin[child] = NextInTimeBin[parent];
    if(NextInTimeBin[parent] >= 0)
        PrevInTimeBin[NextInTimeBin[parent]] = child;
    NextInTimeBin[parent] = child;
    if(LastInTimeBin[P[parent].TimeBin] == parent)
        LastInTimeBin[P[parent].TimeBin] = child;

    /* increase NumForceUpdate only if this particle was
     * active */

    /*! When a new additional star particle is created, we can put it into the
     *  tree at the position of the spawning gas particle. This is possible
     *  because the Nextnode[] array essentially describes the full tree walk as a
     *  link list. Multipole moments of tree nodes need not be changed.
     */

    /* we do this only if there is an active force tree 
     * checking Nextnode is not the best way of doing so though.
     * */
    if(Nextnode) {
        int no;

        no = Nextnode[parent];
        Nextnode[parent] = child;
        Nextnode[child] = no;
        Father[child] = Father[parent];
    }
    return child;
}


void domain_findSplit_work_balanced(int ncpu, int ndomain)
{
    int i, start, end;
    double work, workavg, work_before, workavg_before;

    for(i = 0, work = 0; i < ndomain; i++)
        work += domainWork[i];

    workavg = work / ncpu;

    work_before = workavg_before = 0;

    start = 0;

    for(i = 0; i < ncpu; i++)
    {
        work = 0;
        end = start;

        work += domainWork[end] / list_speedfac[i % NTask];

        while((work + work_before < workavg + workavg_before) || (i == ncpu - 1 && end < ndomain - 1))
        {
            if((ndomain - end) > (ncpu - i))
                end++;
            else
                break;

            work += domainWork[end] / list_speedfac[i % NTask];
        }

        DomainStartList[i] = start;
        DomainEndList[i] = end;

        work_before += work;
        workavg_before += workavg;
        start = end + 1;
    }
}


static struct domain_loadorigin_data
{
    double load;
    int origin;
}
*domain;

static struct domain_segments_data
{
    int task, start, end;
}
*domainAssign;



int domain_sort_loadorigin(const void *a, const void *b)
{
    if(((struct domain_loadorigin_data *) a)->load < (((struct domain_loadorigin_data *) b)->load))
        return -1;

    if(((struct domain_loadorigin_data *) a)->load > (((struct domain_loadorigin_data *) b)->load))
        return +1;

    return 0;
}

int domain_sort_segments(const void *a, const void *b)
{
    if(((struct domain_segments_data *) a)->task < (((struct domain_segments_data *) b)->task))
        return -1;

    if(((struct domain_segments_data *) a)->task > (((struct domain_segments_data *) b)->task))
        return +1;

    return 0;
}


void domain_assign_load_or_work_balanced(int mode)
{
    int i, n, ndomains, *target;

    domainAssign =
        (struct domain_segments_data *) mymalloc("domainAssign",
                MULTIPLEDOMAINS * NTask * sizeof(struct domain_segments_data));
    domain =
        (struct domain_loadorigin_data *) mymalloc("domain",
                MULTIPLEDOMAINS * NTask *
                sizeof(struct domain_loadorigin_data));
    target = (int *) mymalloc("target", MULTIPLEDOMAINS * NTask * sizeof(int));

    for(n = 0; n < MULTIPLEDOMAINS * NTask; n++)
        domainAssign[n].task = n;

    ndomains = MULTIPLEDOMAINS * NTask;

    while(ndomains > NTask)
    {
        for(i = 0; i < ndomains; i++)
        {
            domain[i].load = 0;
            domain[i].origin = i;
        }

        for(n = 0; n < MULTIPLEDOMAINS * NTask; n++)
        {
            for(i = DomainStartList[n]; i <= DomainEndList[n]; i++)
                if(mode == 1)
                    domain[domainAssign[n].task].load += domainCount[i];
                else
                    domain[domainAssign[n].task].load += domainWork[i];
        }

        qsort(domain, ndomains, sizeof(struct domain_loadorigin_data), domain_sort_loadorigin);

        for(i = 0; i < ndomains / 2; i++)
        {
            target[domain[i].origin] = i;
            target[domain[ndomains - 1 - i].origin] = i;
        }

        for(n = 0; n < MULTIPLEDOMAINS * NTask; n++)
            domainAssign[n].task = target[domainAssign[n].task];

        ndomains /= 2;
    }

    for(n = 0; n < MULTIPLEDOMAINS * NTask; n++)
    {
        domainAssign[n].start = DomainStartList[n];
        domainAssign[n].end = DomainEndList[n];
    }

    qsort(domainAssign, MULTIPLEDOMAINS * NTask, sizeof(struct domain_segments_data), domain_sort_segments);

    for(n = 0; n < MULTIPLEDOMAINS * NTask; n++)
    {
        DomainStartList[n] = domainAssign[n].start;
        DomainEndList[n] = domainAssign[n].end;

        for(i = DomainStartList[n]; i <= DomainEndList[n]; i++)
            DomainTask[i] = domainAssign[n].task;
    }

    myfree(target);
    myfree(domain);
    myfree(domainAssign);
}



void domain_findSplit_load_balanced(int ncpu, int ndomain)
{
    int i, start, end;
    double load, loadavg, load_before, loadavg_before;

    for(i = 0, load = 0; i < ndomain; i++)
        load += domainCount[i];

    loadavg = load / ncpu;

    load_before = loadavg_before = 0;

    start = 0;

    for(i = 0; i < ncpu; i++)
    {
        load = 0;
        end = start;

        load += domainCount[end];

        while((load + load_before < loadavg + loadavg_before) || (i == ncpu - 1 && end < ndomain - 1))
        {
            if((ndomain - end) > (ncpu - i))
                end++;
            else
                break;

            load += domainCount[end];
        }

        DomainStartList[i] = start;
        DomainEndList[i] = end;

        load_before += load;
        loadavg_before += loadavg;
        start = end + 1;
    }
}







/*! This function determines how many particles that are currently stored
 *  on the local CPU have to be moved off according to the domain
 *  decomposition.
 *
 *  layoutfunc decides the target Task of particle p (used by
 *  subfind_distribute.
 *
 */
static int domain_layoutfunc(int n) {
    int no;
    no = 0;
    peanokey key = KEY(n);
    while(topNodes[no].Daughter >= 0)
        no = topNodes[no].Daughter + (key - topNodes[no].StartKey) / (topNodes[no].Size / 8);

    no = topNodes[no].Leaf;
    return DomainTask[no];
}

static int domain_countToGo(ptrdiff_t nlimit, int (*layoutfunc)(int p))
{
    int n, ret, retsum;
    size_t package;

    int *list_NumPart;
    int *list_N_sph;
    int *list_N_bh;

    list_NumPart = (int *) alloca(sizeof(int) * NTask);
    list_N_sph = (int *) alloca(sizeof(int) * NTask);
    list_N_bh = (int *) alloca(sizeof(int) * NTask);

    for(n = 0; n < NTask; n++)
    {
        toGo[n] = 0;
        toGoSph[n] = 0;
        toGoBh[n] = 0;
    }

    package = (sizeof(struct particle_data) + sizeof(struct sph_particle_data) + sizeof(peanokey));
    if(package >= nlimit)
        endrun(212);


    for(n = 0; n < NumPart; n++)
    {
        if(package >= nlimit) continue;
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
        MPI_Allgather(&N_bh, 1, MPI_INT, list_N_bh, 1, MPI_INT, MPI_COMM_WORLD);
        MPI_Allgather(&N_sph, 1, MPI_INT, list_N_sph, 1, MPI_INT, MPI_COMM_WORLD);

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
                    if((ntoomany = list_N_sph[ta] + count_toget_sph - count_togo_sph - All.MaxPartSph) > 0)
                    {
                        if(ThisTask == 0)
                        {
                            printf
                                ("exchange needs to be modified because I can't receive %d SPH-particles on task=%d\n",
                                 ntoomany, ta);
                            if(flagsum > 25)
                                printf("list_N_sph[ta=%d]=%d  count_toget_sph=%d count_togo_sph=%d\n",
                                        ta, list_N_sph[ta], count_toget_sph, count_togo_sph);
                            fflush(stdout);
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
                        if(ThisTask == 0)
                        {
                            printf
                                ("exchange needs to be modified because I can't receive %d BH-particles on task=%d\n",
                                 ntoomany, ta);
                            if(flagsum > 25)
                                printf("list_N_bh[ta=%d]=%d  count_toget_bh=%d count_togo_bh=%d\n",
                                        ta, list_N_bh[ta], count_toget_bh, count_togo_bh);
                            fflush(stdout);
                        }

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
                        if(ThisTask == 0)
                        {
                            printf
                                ("exchange needs to be modified because I can't receive %d particles on task=%d\n",
                                 ntoomany, ta);
                            if(flagsum > 25)
                                printf("list_NumPart[ta=%d]=%d  count_toget=%d count_togo=%d\n",
                                        ta, list_NumPart[ta], count_toget, count_togo);
                            fflush(stdout);
                        }

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

                if(ThisTask == 0)
                {
                    printf("flagsum = %d\n", flagsum);
                    fflush(stdout);
                    if(flagsum > 100)
                        endrun(1013);
                }
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






/*! This function walks the global top tree in order to establish the
 *  number of leaves it has. These leaves are distributed to different
 *  processors.
 */
void domain_walktoptree(int no)
{
    int i;

    if(topNodes[no].Daughter == -1)
    {
        topNodes[no].Leaf = NTopleaves;
        NTopleaves++;
    }
    else
    {
        for(i = 0; i < 8; i++)
            domain_walktoptree(topNodes[no].Daughter + i);
    }
}


int domain_compare_key(const void *a, const void *b)
{
    if(((struct peano_hilbert_data *) a)->key < (((struct peano_hilbert_data *) b)->key))
        return -1;

    if(((struct peano_hilbert_data *) a)->key > (((struct peano_hilbert_data *) b)->key))
        return +1;

    return 0;
}


int domain_check_for_local_refine(int i, double countlimit, double costlimit)
{
    int j, p, sub, flag = 0;

#ifdef DENSITY_INDEPENDENT_SPH_DEBUG
    costlimit = -1;
    countlimit = -1;
#endif
    if(topNodes[i].Parent >= 0)
    {
        if(topNodes[i].Count > 0.8 * topNodes[topNodes[i].Parent].Count ||
                topNodes[i].Cost > 0.8 * topNodes[topNodes[i].Parent].Cost)
            flag = 1;
    }

    if((topNodes[i].Count > countlimit || topNodes[i].Cost > costlimit || flag == 1) && topNodes[i].Size >= 8)
    {
        if(topNodes[i].Size >= 8)
        {
            if((NTopnodes + 8) <= MaxTopNodes)
            {
                topNodes[i].Daughter = NTopnodes;

                for(j = 0; j < 8; j++)
                {
                    sub = topNodes[i].Daughter + j;
                    topNodes[sub].Daughter = -1;
                    topNodes[sub].Parent = i;
                    topNodes[sub].Size = (topNodes[i].Size >> 3);
                    topNodes[sub].StartKey = topNodes[i].StartKey + j * topNodes[sub].Size;
                    topNodes[sub].PIndex = topNodes[i].PIndex;
                    topNodes[sub].Count = 0;
                    topNodes[sub].Cost = 0;

                }

                NTopnodes += 8;

                sub = topNodes[i].Daughter;

                for(p = topNodes[i].PIndex, j = 0; p < topNodes[i].PIndex + topNodes[i].Count; p++)
                {
                    if(j < 7)
                        while(mp[p].key >= topNodes[sub + 1].StartKey)
                        {
                            j++;
                            sub++;
                            topNodes[sub].PIndex = p;
                            if(j >= 7)
                                break;
                        }

                    topNodes[sub].Cost += domain_particle_costfactor(mp[p].index);
                    topNodes[sub].Count++;
                }

                for(j = 0; j < 8; j++)
                {
                    sub = topNodes[i].Daughter + j;

#ifdef DENSITY_INDEPENDENT_SPH_DEBUG
                    if(topNodes[sub].Count > All.TotNumPart / 
                            (TOPNODEFACTOR * NTask * NTask))
#endif
                        if(domain_check_for_local_refine(sub, countlimit, costlimit))
                            return 1;
                }
            }
            else
                return 1;
        }
    }

    return 0;
}

int domain_nonrecursively_combine_topTree() {
    /* 
     * combine topTree non recursively, this uses MPI_Bcast within a group.
     * it shall be quite a bit faster (~ x2) than the old recursive scheme.
     *
     * it takes less time at higher sep.
     *
     * The communicate should have been done with MPI Inter communicator.
     * but I couldn't figure out how to do it that way.
     * */
    int sep = 1;
    MPI_Datatype MPI_TYPE_TOPNODE;
    MPI_Type_contiguous(sizeof(struct local_topnode_data), MPI_BYTE, &MPI_TYPE_TOPNODE);
    MPI_Type_commit(&MPI_TYPE_TOPNODE);
    int errorflag = 0;
    int errorflagall = 0;

    for(sep = 1; sep < NTask; sep *=2) {
        /* build the subcommunicators for broadcasting */
        int Color = ThisTask / sep;
        int Key = ThisTask % sep;
        int ntopnodes_import = 0;
        struct local_topnode_data * topNodes_import = NULL;

        int recvTask = -1; /* by default do not communicate */

        if(Key != 0) {
            /* non leaders will skip exchanges */
            goto loop_continue;
        }

        /* leaders of even color will combine nodes from next odd color,
         * so that when sep is increased eventually rank 0 will have all
         * nodes */
        if(Color % 2 == 0) {
            /* even guys recv */
            recvTask = ThisTask + sep;
            if(recvTask < NTask) {
                MPI_Recv(
                        &ntopnodes_import, 1, MPI_INT, recvTask, TAG_GRAV_A,
                        MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                topNodes_import = (struct local_topnode_data *) mymalloc("topNodes_import",
                            IMAX(ntopnodes_import, NTopnodes) * sizeof(struct local_topnode_data));

                MPI_Recv(
                        topNodes_import,
                        ntopnodes_import, MPI_TYPE_TOPNODE, 
                        recvTask, TAG_GRAV_B, 
                        MPI_COMM_WORLD, MPI_STATUS_IGNORE);


                if((NTopnodes + ntopnodes_import) > MaxTopNodes) {
                    errorflag = 1;
                } else {
                    if(ntopnodes_import < 0) {
                        fprintf(stderr, "severe domain error using a unintended rank \n");
                        abort();
                    }
                    if(ntopnodes_import > 0 ) {
                        domain_insertnode(topNodes, topNodes_import, 0, 0);
                    } 
                }
                myfree(topNodes_import);
            }
        } else {
            /* odd guys send */
            recvTask = ThisTask - sep;
            if(recvTask >= 0) {
                MPI_Send(&NTopnodes, 1, MPI_INT, recvTask, TAG_GRAV_A,
                        MPI_COMM_WORLD);
                MPI_Send(topNodes,
                        NTopnodes, MPI_TYPE_TOPNODE,
                        recvTask, TAG_GRAV_B,
                        MPI_COMM_WORLD);
            }
            NTopnodes = -1;
        }

loop_continue:
        MPI_Allreduce(&errorflag, &errorflagall, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
        if(errorflagall) {
            break;
        }
    }

    MPI_Bcast(&NTopnodes, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(topNodes, NTopnodes, MPI_TYPE_TOPNODE, 0, MPI_COMM_WORLD);
    MPI_Type_free(&MPI_TYPE_TOPNODE);
    return errorflagall;
}

/*! This function constructs the global top-level tree node that is used
 *  for the domain decomposition. This is done by considering the string of
 *  Peano-Hilbert keys for all particles, which is recursively chopped off
 *  in pieces of eight segments until each segment holds at most a certain
 *  number of particles.
 */
int domain_determineTopTree(void)
{
    int i, count, j, sub, ngrp;
    int recvTask, sendTask, ntopnodes_import, errflag, errsum;
    struct local_topnode_data *topNodes_import, *topNodes_temp;
    double costlimit, countlimit;

    mp = (struct peano_hilbert_data *) mymalloc("mp", sizeof(struct peano_hilbert_data) * NumPart);

    count = 0;
#pragma omp parallel for
    for(i = 0; i < NumPart; i++)
    {
#pragma omp atomic
        count ++;
        mp[i].key = KEY(i);
        mp[i].index = i;
    }

    walltime_measure("/Domain/DetermineTopTree/Misc");
    qsort(mp, NumPart, sizeof(struct peano_hilbert_data), domain_compare_key);
    
    walltime_measure("/Domain/DetermineTopTree/Sort");

    NTopnodes = 1;
    topNodes[0].Daughter = -1;
    topNodes[0].Parent = -1;
    topNodes[0].Size = PEANOCELLS;
    topNodes[0].StartKey = 0;
    topNodes[0].PIndex = 0;
    topNodes[0].Count = count;
    topNodes[0].Cost = gravcost;

    costlimit = totgravcost / (TOPNODEFACTOR * MULTIPLEDOMAINS * NTask);
    countlimit = totpartcount / (TOPNODEFACTOR * MULTIPLEDOMAINS * NTask);

    errflag = domain_check_for_local_refine(0, countlimit, costlimit);
    walltime_measure("/Domain/DetermineTopTree/LocalRefine");

    myfree(mp);

    MPI_Allreduce(&errflag, &errsum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if(errsum)
    {
        if(ThisTask == 0)
            printf
                ("We are out of Topnodes. We'll try to repeat with a higher value than All.TopNodeAllocFactor=%g\n",
                 All.TopNodeAllocFactor);
        fflush(stdout);

        return errsum;
    }


    /* we now need to exchange tree parts and combine them as needed */

    
    errflag = domain_nonrecursively_combine_topTree();

    walltime_measure("/Domain/DetermineTopTree/Combine");
#if 0
    char buf[1000];
    sprintf(buf, "topnodes.bin.%d", ThisTask);
    FILE * fd = fopen(buf, "w");

    /* these PIndex are non-essential in other modules, so we reset them */
    for(i = 0; i < NTopnodes; i ++) {
        topNodes[i].PIndex = -1;
    }
    fwrite(topNodes, sizeof(struct local_topnode_data), NTopnodes, fd);
    fclose(fd);

    //MPI_Barrier(MPI_COMM_WORLD);
    //MPI_Abort(MPI_COMM_WORLD, 0);
#endif
    MPI_Allreduce(&errflag, &errsum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    if(errsum)
    {
        if(ThisTask == 0)
            printf("can't combine trees due to lack of storage. Will try again.\n");
        return errsum;
    }

    /* now let's see whether we should still append more nodes, based on the estimated cumulative cost/count in each cell */


#ifndef DENSITY_INDEPENDENT_SPH_DEBUG
    if(ThisTask == 0)
        printf("Before=%d\n", NTopnodes);

    for(i = 0, errflag = 0; i < NTopnodes; i++)
    {
        if(topNodes[i].Daughter < 0)
            if(topNodes[i].Count > countlimit || topNodes[i].Cost > costlimit)	/* ok, let's add nodes if we can */
                if(topNodes[i].Size > 1)
                {
                    if((NTopnodes + 8) <= MaxTopNodes)
                    {
                        topNodes[i].Daughter = NTopnodes;

                        for(j = 0; j < 8; j++)
                        {
                            sub = topNodes[i].Daughter + j;
                            topNodes[sub].Size = (topNodes[i].Size >> 3);
                            topNodes[sub].Count = topNodes[i].Count / 8;
                            topNodes[sub].Cost = topNodes[i].Cost / 8;
                            topNodes[sub].Daughter = -1;
                            topNodes[sub].Parent = i;
                            topNodes[sub].StartKey = topNodes[i].StartKey + j * topNodes[sub].Size;
                        }

                        NTopnodes += 8;
                    }
                    else
                    {
                        errflag = 1;
                        break;
                    }
                }
    }

    MPI_Allreduce(&errflag, &errsum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if(errsum)
        return errsum;

    if(ThisTask == 0)
        printf("After=%d\n", NTopnodes);
#endif
    walltime_measure("/Domain/DetermineTopTree/Addnodes");
    /* count toplevel leaves */
    domain_sumCost();
    walltime_measure("/Domain/DetermineTopTree/Sumcost");

    if(NTopleaves < MULTIPLEDOMAINS * NTask)
        endrun(112);

    return 0;
}



void domain_sumCost(void)
{
    int i, n;
    float *local_domainWork;
    int *local_domainCount;
    int *local_domainCountSph;

    local_domainWork = (float *) mymalloc("local_domainWork", All.NumThreads * NTopnodes * sizeof(float));
    local_domainCount = (int *) mymalloc("local_domainCount", All.NumThreads * NTopnodes * sizeof(int));
    local_domainCountSph = (int *) mymalloc("local_domainCountSph", All.NumThreads * NTopnodes * sizeof(int));

    NTopleaves = 0;
    domain_walktoptree(0);

    if(ThisTask == 0)
        printf("NTopleaves= %d  NTopnodes=%d (space for %d)\n", NTopleaves, NTopnodes, MaxTopNodes);

#pragma omp parallel private(n, i)
    {
        int tid = omp_get_thread_num();

        float * mylocal_domainWork = local_domainWork + tid * NTopleaves;
        int * mylocal_domainCount = local_domainCount + tid * NTopleaves;
        int * mylocal_domainCountSph = local_domainCountSph + tid * NTopleaves;

        for(i = 0; i < NTopleaves; i++)
        {
            mylocal_domainWork[i] = 0;
            mylocal_domainCount[i] = 0;
            mylocal_domainCountSph[i] = 0;
        }


#pragma omp for
        for(n = 0; n < NumPart; n++)
        {
            int no = 0;
            peanokey key = KEY(n);
            while(topNodes[no].Daughter >= 0)
                no = topNodes[no].Daughter + (key - topNodes[no].StartKey) / (topNodes[no].Size >> 3);

            no = topNodes[no].Leaf;

            mylocal_domainWork[no] += (float) domain_particle_costfactor(n);

            mylocal_domainCount[no] += 1;

            if(P[n].Type == 0) {
                mylocal_domainCountSph[no] += 1;
            }
        }
    }

#pragma omp parallel for
    for(i = 0; i < NTopleaves; i++)
    {
        int tid;
        for(tid = 1; tid < All.NumThreads; tid++) {
            local_domainWork[i] += local_domainWork[i + tid * NTopleaves];
            local_domainCount[i] += local_domainCount[i + tid * NTopleaves];
            local_domainCountSph[i] += local_domainCountSph[i + tid * NTopleaves];
        }
    }

    MPI_Allreduce(local_domainWork, domainWork, NTopleaves, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(local_domainCount, domainCount, NTopleaves, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(local_domainCountSph, domainCountSph, NTopleaves, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    myfree(local_domainCountSph);
    myfree(local_domainCount);
    myfree(local_domainWork);
}


/*! This routine finds the extent of the global domain grid.
*/
void domain_findExtent(void)
{
    int i;
    double len, xmin[3], xmax[3], xmin_glob[3], xmax_glob[3];

    /* determine local extension */
    int j;
    for(j = 0; j < 3; j++)
    {
        xmin[j] = MAX_REAL_NUMBER;
        xmax[j] = -MAX_REAL_NUMBER;
    }

#pragma omp parallel private(i)
    {
        double xminT[3], xmaxT[3];
        int j;
        for(j = 0; j < 3; j++)
        {
            xminT[j] = MAX_REAL_NUMBER;
            xmaxT[j] = -MAX_REAL_NUMBER;
        }

#pragma omp for
        for(i = 0; i < NumPart; i++)
        {
            int j;
            for(j = 0; j < 3; j++)
            {
                if(xminT[j] > P[i].Pos[j])
                    xminT[j] = P[i].Pos[j];

                if(xmaxT[j] < P[i].Pos[j])
                    xmaxT[j] = P[i].Pos[j];
            }
        }
#pragma omp critical 
        {
            for(j = 0; j < 3; j++) {
                if(xmin[j] > xminT[j]) 
                    xmin[j] = xminT[j]; 
                if(xmax[j] < xmaxT[j]) 
                    xmax[j] = xmaxT[j]; 
            
            } 
        }
    }

    MPI_Allreduce(xmin, xmin_glob, 3, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(xmax, xmax_glob, 3, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    len = 0;
    for(j = 0; j < 3; j++)
        if(xmax_glob[j] - xmin_glob[j] > len)
            len = xmax_glob[j] - xmin_glob[j];

    len *= 1.001;

    for(j = 0; j < 3; j++)
    {
        DomainCenter[j] = 0.5 * (xmin_glob[j] + xmax_glob[j]);
        DomainCorner[j] = 0.5 * (xmin_glob[j] + xmax_glob[j]) - 0.5 * len;
    }

    DomainLen = len;
    DomainFac = 1.0 / len * (((peanokey) 1) << (BITS_PER_DIMENSION));
}




void domain_add_cost(struct local_topnode_data *treeA, int noA, int64_t count, double cost)
{
    int i, sub;
    int64_t countA, countB;

    countB = count / 8;
    countA = count - 7 * countB;

    cost = cost / 8;

    for(i = 0; i < 8; i++)
    {
        sub = treeA[noA].Daughter + i;

        if(i == 0)
            count = countA;
        else
            count = countB;

        treeA[sub].Count += count;
        treeA[sub].Cost += cost;

        if(treeA[sub].Daughter >= 0)
            domain_add_cost(treeA, sub, count, cost);
    }
}


void domain_insertnode(struct local_topnode_data *treeA, struct local_topnode_data *treeB, int noA, int noB)
{
    int j, sub;
    int64_t count, countA, countB;
    double cost, costA, costB;

    if(treeB[noB].Size < treeA[noA].Size)
    {
        if(treeA[noA].Daughter < 0)
        {
            if((NTopnodes + 8) <= MaxTopNodes)
            {
                count = treeA[noA].Count - treeB[treeB[noB].Parent].Count;
                countB = count / 8;
                countA = count - 7 * countB;

                cost = treeA[noA].Cost - treeB[treeB[noB].Parent].Cost;
                costB = cost / 8;
                costA = cost - 7 * costB;

                treeA[noA].Daughter = NTopnodes;
                for(j = 0; j < 8; j++)
                {
                    if(j == 0)
                    {
                        count = countA;
                        cost = costA;
                    }
                    else
                    {
                        count = countB;
                        cost = costB;
                    }

                    sub = treeA[noA].Daughter + j;
                    topNodes[sub].Size = (treeA[noA].Size >> 3);
                    topNodes[sub].Count = count;
                    topNodes[sub].Cost = cost;
                    topNodes[sub].Daughter = -1;
                    topNodes[sub].Parent = noA;
                    topNodes[sub].StartKey = treeA[noA].StartKey + j * treeA[sub].Size;
                }
                NTopnodes += 8;
            }
            else
                endrun(88);
        }

        sub = treeA[noA].Daughter + (treeB[noB].StartKey - treeA[noA].StartKey) / (treeA[noA].Size >> 3);
        domain_insertnode(treeA, treeB, sub, noB);
    }
    else if(treeB[noB].Size == treeA[noA].Size)
    {
        treeA[noA].Count += treeB[noB].Count;
        treeA[noA].Cost += treeB[noB].Cost;

        if(treeB[noB].Daughter >= 0)
        {
            for(j = 0; j < 8; j++)
            {
                sub = treeB[noB].Daughter + j;
                domain_insertnode(treeA, treeB, noA, sub);
            }
        }
        else
        {
            if(treeA[noA].Daughter >= 0)
                domain_add_cost(treeA, noA, treeB[noB].Count, treeB[noB].Cost);
        }
    }
    else
        endrun(89);
}

#if defined(SFR) || defined(BLACK_HOLES)
void rearrange_particle_sequence(void)
{
    int i, flag = 0; 

#ifdef SFR
    for(i = 0; i < N_sph; i++) {
        while(P[i].Type != 0 && N_sph - 1 > i) {
            /* remove this particle from SphP, because
             * it is no longer a SPH
             * */
            struct particle_data psave;
            psave = P[i];
            P[i] = P[N_sph - 1];
            SPHP(i) = SPHP(N_sph - 1);
            P[N_sph - 1] = psave;
            flag = 1;
            N_sph --;
        }
    }
#endif

#ifdef BLACK_HOLES
    int count_elim, count_gaselim, tot_elim, tot_gaselim;

    count_elim = 0;
    count_gaselim = 0;

    for(i = 0; i < NumPart; i++)
        if(P[i].Mass == 0)
        {
            TimeBinCount[P[i].TimeBin]--;

            if(P[i].Type == 0)
            {
                TimeBinCountSph[P[i].TimeBin]--;

                P[i] = P[N_sph - 1];
                SPHP(i) = SPHP(N_sph - 1);

                P[N_sph - 1] = P[NumPart - 1];

                N_sph--;

                count_gaselim++;
            } else
            {
                P[i] = P[NumPart - 1];
            }

            NumPart--;
            i--;

            count_elim++;
        }

    MPI_Allreduce(&count_elim, &tot_elim, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&count_gaselim, &tot_gaselim, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    if(count_elim)
        flag = 1;

    if(ThisTask == 0)
    {
        printf("Blackholes: Eliminated %d gas particles and merged away %d black holes.\n",
                tot_gaselim, tot_elim - tot_gaselim);
        fflush(stdout);
    }

#endif
    int flag_sum;

    MPI_Allreduce(&flag, &flag_sum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);


    if(flag_sum)
        reconstruct_timebins();

}
#endif



static void radix_id(const void * data, void * radix, void * arg) {
    ((uint64_t *) radix)[0] = ((MyIDType*) data)[0];
}

void test_id_uniqueness(void)
{
    int i;
    double t0, t1;
    MyIDType *ids, *ids_first;

    if(ThisTask == 0)
    {
        printf("Testing ID uniqueness...\n");
        fflush(stdout);
    }

    if(NumPart == 0)
    {
        printf("need at least one particle per cpu\n");
        endrun(8);
    }

    t0 = second();

    ids = (MyIDType *) mymalloc("ids", NumPart * sizeof(MyIDType));
    ids_first = (MyIDType *) mymalloc("ids_first", NTask * sizeof(MyIDType));

    for(i = 0; i < NumPart; i++)
        ids[i] = P[i].ID;

    mpsort_mpi(ids, NumPart, sizeof(MyIDType), radix_id, 8, NULL, MPI_COMM_WORLD);

    for(i = 1; i < NumPart; i++)
        if(ids[i] == ids[i - 1])
        {
            printf("non-unique ID=%d%09d found on task=%d (i=%d NumPart=%d)\n",
                    (int) (ids[i] / 1000000000), (int) (ids[i] % 1000000000), ThisTask, i, NumPart);

            endrun(12);
        }

    MPI_Allgather(&ids[0], sizeof(MyIDType), MPI_BYTE, ids_first, sizeof(MyIDType), MPI_BYTE, MPI_COMM_WORLD);

    if(ThisTask < NTask - 1)
        if(ids[NumPart - 1] == ids_first[ThisTask + 1])
        {
            printf("non-unique ID=%d found on task=%d\n", (int) ids[NumPart - 1], ThisTask);
            endrun(13);
        }

    myfree(ids_first);
    myfree(ids);

    t1 = second();

    if(ThisTask == 0)
    {
        printf("success.  took=%g sec\n", timediff(t0, t1));
        fflush(stdout);
    }
}
