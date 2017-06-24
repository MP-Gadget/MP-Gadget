#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <math.h>


#include "allvars.h"
#include "proto.h"
#include "forcetree.h"
#include "mymalloc.h"
#include "mpsort.h"
#include "endrun.h"
#include "openmpsort.h"
#include "domain.h"
#include "timestep.h"
#include "system.h"
#include "exchange.h"
#include "garbage.h"

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

/*Only used in forcetree.c*/
int *DomainStartList, *DomainEndList;

int *DomainTask;

struct topnode_data *TopNodes;

int NTopnodes, NTopleaves;

struct local_topnode_data
{
    /*These members are copied into topnode_data*/
    peanokey Size;		/*!< number of Peano-Hilbert mesh-cells represented by top-level node */
    peanokey StartKey;		/*!< first Peano-Hilbert key in top-level node */
    int Daughter;			/*!< index of first daughter cell (out of 8) of top-level node */
    int Leaf;			/*!< if the node is a leaf, this gives its number when all leaves are traversed in Peano-Hilbert order */
    /*Below members are only used in this file*/
    int Parent;
    int PIndex;			/*!< first particle in node  used only in top-level tree build (this file)*/
    int64_t Count;		/*!< counts the number of particles in this top-level node */
    double Cost;
};


static void domain_findSplit_work_balanced(int ncpu, int ndomain, float *domainWork);
static void domain_findSplit_load_balanced(int ncpu, int ndomain, int *domainCount);
static void domain_assign_balanced(float* domainWork, int* domainCount);
static void domain_allocate(void);
int domain_check_memory_bound(const int print_details, float *domainWork, int *domainCount);
static int domain_decompose(void);
int domain_determineTopTree(struct local_topnode_data * topNodes);
static void domain_free(void);
static void domain_sumCost(float *domainWork, int *domainCount);

void domain_insertnode(struct local_topnode_data *treeA, struct local_topnode_data *treeB, int noA, int noB, struct local_topnode_data * topNodes);
static void domain_add_cost(struct local_topnode_data *treeA, int noA, int64_t count, double cost);
int domain_check_for_local_refine(const int i, const struct peano_hilbert_data * mp, struct local_topnode_data * topNodes);

static int domain_layoutfunc(int n);

static int domain_allocated_flag = 0;

/*! This is the main routine for the domain decomposition.  It acts as a
 *  driver routine that allocates various temporary buffers, maps the
 *  particles back onto the periodic box if needed, and then does the
 *  domain decomposition, and a final Peano-Hilbert order of all particles
 *  as a tuning measure.
 */
void domain_Decomposition(void)
{
    int retsum;
    double t0, t1;

    walltime_measure("/Misc");

    /*This drifts all the particles*/
    move_particles(All.Ti_Current);

    if(force_tree_allocated()) force_tree_free();

    domain_garbage_collection();

    domain_free();

    do_box_wrapping();	/* map the particles back onto the box */

    message(0, "domain decomposition... (presently allocated=%g MB)\n", AllocatedBytes / (1024.0 * 1024.0));

    t0 = second();

    do
    {
        int ret;
#ifdef DEBUG
        message(0, "Testing ID Uniqueness before domain decompose\n");
        test_id_uniqueness();
#endif
        domain_allocate();
        ret = domain_decompose();

        MPI_Allreduce(&ret, &retsum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        if(retsum)
        {
            domain_free();
            message(0, "Increasing TopNodeAllocFactor=%g  ", All.TopNodeAllocFactor);

            All.TopNodeAllocFactor *= 1.3;

            message(0, "new value=%g\n", All.TopNodeAllocFactor);

            if(All.TopNodeAllocFactor > 1000)
            {
                if(ThisTask == 0)
                    endrun(781, "something seems to be going seriously wrong here. Stopping.\n");
            }
        }
    }
    while(retsum);

    t1 = second();

    message(0, "domain decomposition done. (took %g sec)\n", timediff(t0, t1));

    peano_hilbert_order();

    walltime_measure("/Domain/Peano");

    memmove(TopNodes + NTopnodes, DomainTask, NTopnodes * sizeof(int));

    TopNodes = (struct topnode_data *) myrealloc(TopNodes,
            (NTopnodes * sizeof(struct topnode_data) + NTopnodes * sizeof(int)));
    message(0, "Freed %g MByte in top-level domain structure\n",
                (MaxTopNodes - NTopnodes) * sizeof(struct topnode_data) / (1024.0 * 1024.0));

    DomainTask = (int *) (TopNodes + NTopnodes);

    reconstruct_timebins();
    walltime_measure("/Domain/Misc");
    force_tree_rebuild();
}

/* This is a cut-down version of the domain decomposition that leaves the
 * domain grid intact, but exchanges the particles*/
void domain_Decomposition_short(void)
{
    int i;

    walltime_measure("/Misc");

    move_particles(All.Ti_Current);

    /* We rebuild the tree every timestep in order to
     * make sure it is consistent.
     * May as well free it here.*/
    if(force_tree_allocated()) force_tree_free();

    /*In case something happened during the timestep*/
    domain_garbage_collection();

    do_box_wrapping();	/* map the particles back onto the box */

    /* Make an array of peano keys so we don't have to
     * recompute them during layout and force tree build.*/
    #pragma omp parallel for
    for(i=0; i<NumPart; i++)
        P[i].Key = KEY(i);

    walltime_measure("/Domain/Short/Misc");
    /*TODO: We should probably check we can satisfy memory constraints here,
     * but that is expensive. Maybe check during exchange (or after exchange)
     * and if not true bail to a full domain_Decomp.*/

    domain_exchange(domain_layoutfunc);

    peano_hilbert_order();
    walltime_measure("/Domain/Short/Peano");

    /* Rebuild active particle list and timebin counts:
     * peano order has changed.*/
    reconstruct_timebins();
    force_tree_rebuild();
}

/*! This function allocates all the stuff that will be required for the tree-construction/walk later on */
void domain_allocate(void)
{
    size_t bytes, all_bytes = 0;

    MaxTopNodes = (int) (All.TopNodeAllocFactor * All.MaxPart + 1);

    DomainStartList = (int *) mymalloc("DomainStartList", bytes = (NTask * All.DomainOverDecompositionFactor * sizeof(int)));
    all_bytes += bytes;

    DomainEndList = (int *) mymalloc("DomainEndList", bytes = (NTask * All.DomainOverDecompositionFactor * sizeof(int)));
    all_bytes += bytes;

    TopNodes = (struct topnode_data *) mymalloc("TopNodes", bytes =
            (MaxTopNodes * sizeof(struct topnode_data) +
             MaxTopNodes * sizeof(int)));
    all_bytes += bytes;

    DomainTask = (int *) (TopNodes + MaxTopNodes);

    message(0, "Allocated %g MByte for top-level domain structure\n", all_bytes / (1024.0 * 1024.0));

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

float domain_particle_costfactor(int i)
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

    int i, status;

    size_t bytes, all_bytes = 0;

    /*!< a table that gives the total "work" due to the particles stored by each processor */
    float *domainWork = (float *) mymalloc("domainWork", bytes = (MaxTopNodes * sizeof(float)));
    all_bytes += bytes;
    /*!< a table that gives the total number of particles held by each processor */
    int * domainCount = (int *) mymalloc("domainCount", bytes = (MaxTopNodes * sizeof(int)));
    all_bytes += bytes;
	/*!< points to the root node of the top-level tree */
    struct local_topnode_data *topNodes = (struct local_topnode_data *) mymalloc("topNodes", bytes =
            (MaxTopNodes * sizeof(struct local_topnode_data)));
    memset(topNodes, 0, sizeof(topNodes[0]) * MaxTopNodes);
    all_bytes += bytes;

    message(0, "use of %g MB of temporary storage for domain decomposition... (presently allocated=%g MB)\n",
             all_bytes / (1024.0 * 1024.0), AllocatedBytes / (1024.0 * 1024.0));

    report_memory_usage("DOMAIN");

    walltime_measure("/Domain/Decompose/Misc");

    /*Make an array of peano keys so we don't have to recompute them inside the domain*/
    #pragma omp parallel for
    for(i=0; i<NumPart; i++)
        P[i].Key = KEY(i);

    if(domain_determineTopTree(topNodes)) {
        myfree(topNodes);
        myfree(domainCount);
        myfree(domainWork);
        return 1;
    }

    /* copy what we need for the topnodes */
    for(i = 0; i < NTopnodes; i++)
    {
        TopNodes[i].StartKey = topNodes[i].StartKey;
        TopNodes[i].Size = topNodes[i].Size;
        TopNodes[i].Daughter = topNodes[i].Daughter;
        TopNodes[i].Leaf = topNodes[i].Leaf;
    }

    myfree(topNodes);
    /* count toplevel leaves */
    domain_sumCost(domainWork, domainCount);
    walltime_measure("/Domain/DetermineTopTree/Sumcost");

    if(NTopleaves < All.DomainOverDecompositionFactor * NTask)
        endrun(112, "Number of Topleaves is less than required over decomposition");


    /* find the split of the domain grid */
    domain_findSplit_work_balanced(All.DomainOverDecompositionFactor * NTask, NTopleaves,domainWork);

    walltime_measure("/Domain/Decompose/findworksplit");

    domain_assign_balanced(domainWork, NULL);

    walltime_measure("/Domain/Decompose/assignbalance");

    status = domain_check_memory_bound(0,domainWork,domainCount);
    walltime_measure("/Domain/Decompose/memorybound");

    if(status != 0)		/* the optimum balanced solution violates memory constraint, let's try something different */
    {
        message(0, "Note: the domain decomposition is suboptimum because the ceiling for memory-imbalance is reached\n");

        domain_findSplit_load_balanced(All.DomainOverDecompositionFactor * NTask, NTopleaves,domainCount);

        walltime_measure("/Domain/Decompose/findloadsplit");
        domain_assign_balanced(NULL, domainCount);
        walltime_measure("/Domain/Decompose/assignbalance");

        status = domain_check_memory_bound(1,domainWork,domainCount);
        walltime_measure("/Domain/Decompose/memorybound");

        if(status != 0)
        {
            endrun(0, "No domain decomposition that stays within memory bounds is possible.\n");
        }
    }

    walltime_measure("/Domain/Decompose/Misc");
    domain_exchange(domain_layoutfunc);

    myfree(domainCount);
    myfree(domainWork);

    return 0;
}

int domain_check_memory_bound(const int print_details, float *domainWork, int *domainCount)
{
    int ta, m, i;
    int load, max_load;
    int64_t sumload;
    double work, max_work, sumwork;
    /*Only used if print_details is true*/
    int list_load[NTask];
    double list_work[NTask];

    max_work = max_load = sumload = sumwork = 0;

    for(ta = 0; ta < NTask; ta++)
    {
        load = 0;
        work = 0;

        for(m = 0; m < All.DomainOverDecompositionFactor; m++)
            for(i = DomainStartList[ta * All.DomainOverDecompositionFactor + m]; i <= DomainEndList[ta * All.DomainOverDecompositionFactor + m]; i++)
            {
                load += domainCount[i];
                work += domainWork[i];
            }

        if(print_details) {
            list_load[ta] = load;
            list_work[ta] = work;
        }

        sumwork += work;
        sumload += load;

        if(load > max_load)
            max_load = load;
        if(work > max_work)
            max_work = work;
    }

    message(0, "Largest deviations from average: work=%g particle load=%g\n",
            max_work / (sumwork / NTask), max_load / (((double) sumload) / NTask));

    if(print_details) {
        message(0, "Balance breakdown:\n");
        for(i = 0; i < NTask; i++)
        {
            message(0, "Task: [%3d]  work=%8.4f  particle load=%8.4f\n", i,
               list_work[i] / (sumwork / NTask), list_load[i] / (((double) sumload) / NTask));
        }
    }

    if(max_load > All.MaxPart)
    {
        message(0, "desired memory imbalance=%g  (limit=%d, needed=%d)\n",
                    (max_load * All.PartAllocFactor) / All.MaxPart, All.MaxPart, max_load);

        return 1;
    }

    return 0;
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


void domain_assign_balanced(float *domainWork, int *domainCount)
{
    int i, n, ndomains, *target;

    domainAssign =
        (struct domain_segments_data *) mymalloc("domainAssign",
                All.DomainOverDecompositionFactor * NTask * sizeof(struct domain_segments_data));
    domain =
        (struct domain_loadorigin_data *) mymalloc("domain",
                All.DomainOverDecompositionFactor * NTask *
                sizeof(struct domain_loadorigin_data));
    target = (int *) mymalloc("target", All.DomainOverDecompositionFactor * NTask * sizeof(int));

    for(n = 0; n < All.DomainOverDecompositionFactor * NTask; n++)
        domainAssign[n].task = n;

    ndomains = All.DomainOverDecompositionFactor * NTask;

    while(ndomains > NTask)
    {
        for(i = 0; i < ndomains; i++)
        {
            domain[i].load = 0;
            domain[i].origin = i;
        }

        for(n = 0; n < All.DomainOverDecompositionFactor * NTask; n++)
        {
            for(i = DomainStartList[n]; i <= DomainEndList[n]; i++)
                if(domainWork)
                    domain[domainAssign[n].task].load += domainWork[i];
                else if(domainCount)
                    domain[domainAssign[n].task].load += domainCount[i];
        }

        qsort(domain, ndomains, sizeof(struct domain_loadorigin_data), domain_sort_loadorigin);

        for(i = 0; i < ndomains / 2; i++)
        {
            target[domain[i].origin] = i;
            target[domain[ndomains - 1 - i].origin] = i;
        }

        for(n = 0; n < All.DomainOverDecompositionFactor * NTask; n++)
            domainAssign[n].task = target[domainAssign[n].task];

        ndomains /= 2;
    }

    for(n = 0; n < All.DomainOverDecompositionFactor * NTask; n++)
    {
        domainAssign[n].start = DomainStartList[n];
        domainAssign[n].end = DomainEndList[n];
    }

    qsort(domainAssign, All.DomainOverDecompositionFactor * NTask, sizeof(struct domain_segments_data), domain_sort_segments);

    for(n = 0; n < All.DomainOverDecompositionFactor * NTask; n++)
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


/* These next two functions are identical except for the type of the thing summed.*/
void domain_findSplit_work_balanced(int ncpu, int ndomain, float *domainWork)
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

        work += domainWork[end];

        while((work + work_before < workavg + workavg_before) || (i == ncpu - 1 && end < ndomain - 1))
        {
            if((ndomain - end) > (ncpu - i))
                end++;
            else
                break;

            work += domainWork[end];
        }

        DomainStartList[i] = start;
        DomainEndList[i] = end;

        work_before += work;
        workavg_before += workavg;
        start = end + 1;
    }
}

void domain_findSplit_load_balanced(int ncpu, int ndomain, int *domainCount)
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


/*This function determines the leaf node for the given particle number.*/
static inline int domain_leafnodefunc(const peanokey key) {
    int no=0;
    while(TopNodes[no].Daughter >= 0)
        no = TopNodes[no].Daughter + (key - TopNodes[no].StartKey) / (TopNodes[no].Size / 8);
    no = TopNodes[no].Leaf;
    return no;
}

/*! This function determines how many particles that are currently stored
 *  on the local CPU have to be moved off according to the domain
 *  decomposition.
 *
 *  layoutfunc decides the target Task of particle p (used by
 *  subfind_distribute).
 *
 */
static int domain_layoutfunc(int n) {
    peanokey key = P[n].Key;
    int no = domain_leafnodefunc(key);
    return DomainTask[no];
}

/*! This function walks the global top tree in order to establish the
 *  number of leaves it has. These leaves are distributed to different
 *  processors.
 */
void domain_walktoptree(int no)
{
    int i;

    if(TopNodes[no].Daughter == -1)
    {
        TopNodes[no].Leaf = NTopleaves;
        NTopleaves++;
    }
    else
    {
        for(i = 0; i < 8; i++)
            domain_walktoptree(TopNodes[no].Daughter + i);
    }
}

/* Refine the local oct-tree, recursively adding costs and particles until
 * either we have chopped off all the peano-hilbert keys and thus have no more
 * refinement to do, or we run out of topNodes.
 * If 1 is returned on any processor we will return to domain_Decomposition,
 * allocate 30% more topNodes, and try again.
 * */
int domain_check_for_local_refine(const int i, const struct peano_hilbert_data * mp, struct local_topnode_data * topNodes)
{
    int j, p;

    /*If there are only 8 particles within this node, we are done refining.*/
    if(topNodes[i].Size < 8)
        return 0;

    /* We need to do refinement if (if we have a parent) we have more than 80%
     * of the parent's particles or costs.*/
    /* If we were below them but we have a parent and somehow got all of its particles, we still
     * need to refine. But if none of these things are true we can return, our work complete. */
    if(topNodes[i].Parent < 0 || (topNodes[i].Count <= 0.8 * topNodes[topNodes[i].Parent].Count &&
            topNodes[i].Cost <= 0.8 * topNodes[topNodes[i].Parent].Cost))
        return 0;

    /* If we want to refine but there is no space for another topNode on this processor,
     * we ran out of top nodes and must get more.*/
    if((NTopnodes + 8) > MaxTopNodes)
        return 1;

    /*Make a new topnode section attached to this node*/
    topNodes[i].Daughter = NTopnodes;
    NTopnodes += 8;

    /* Initialise this topnode with new sub nodes*/
    for(j = 0; j < 8; j++)
    {
        const int sub = topNodes[i].Daughter + j;
        /* The new sub nodes have this node as parent
         * and no daughters.*/
        topNodes[sub].Daughter = -1;
        topNodes[sub].Parent = i;
        /* Shorten the peano key by a factor 8, reflecting the oct-tree level.*/
        topNodes[sub].Size = (topNodes[i].Size >> 3);
        /* This is the region of peanospace covered by this node.*/
        topNodes[sub].StartKey = topNodes[i].StartKey + j * topNodes[sub].Size;
        /* We will compute the cost and initialise the first particle in the node below.
         * This PIndex value is never used*/
        topNodes[sub].PIndex = topNodes[i].PIndex;
        topNodes[sub].Count = 0;
        topNodes[sub].Cost = 0;
    }

    /* Loop over all particles in this node so that the costs of the daughter nodes are correct*/
    for(p = 0, j = 0; p < topNodes[i].Count; p++)
    {
        const int sub = topNodes[i].Daughter;

        /* This identifies which subnode this particle belongs to.
         * Once this particle has passed the StartKey of the next daughter node,
         * we increment the node the particle is added to and set the PIndex.*/
        if(j < 7)
            while(topNodes[sub + j + 1].StartKey <= mp[p + topNodes[i].PIndex].key)
            {
                topNodes[sub + j + 1].PIndex = p;
                j++;
                if(j >= 7)
                    break;
            }

        /*Now we have identified the subnode for this particle, add it to the cost and count*/
        topNodes[sub+j].Cost += domain_particle_costfactor(mp[p + topNodes[i].PIndex].index);
        topNodes[sub+j].Count++;
    }

    /*Check and refine the new daughter nodes*/
    for(j = 0; j < 8; j++)
    {
        const int sub = topNodes[i].Daughter + j;
        /* Refine each sub node. If we could not refine the node as needed,
         * we are out of node space and need more.*/
        if(domain_check_for_local_refine(sub, mp, topNodes))
            return 1;
    }
    return 0;
}

int domain_nonrecursively_combine_topTree(struct local_topnode_data * topNodes)
{
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
                        endrun(1, "severe domain error using a unintended rank \n");
                    }
                    if(ntopnodes_import > 0 ) {
                        domain_insertnode(topNodes, topNodes_import, 0, 0, topNodes);
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
int domain_determineTopTree(struct local_topnode_data * topNodes)
{
    int i, j, sub;
    int errflag, errsum;
    double costlimit, countlimit;

    struct peano_hilbert_data * mp = (struct peano_hilbert_data *) mymalloc("mp", sizeof(struct peano_hilbert_data) * NumPart);

    #pragma omp parallel for
    for(i = 0; i < NumPart; i++)
    {
        mp[i].key = P[i].Key;
        mp[i].index = i;
    }

    walltime_measure("/Domain/DetermineTopTree/Misc");
    qsort_openmp(mp, NumPart, sizeof(struct peano_hilbert_data), peano_compare_key);
    
    walltime_measure("/Domain/DetermineTopTree/Sort");

    double totgravcost, gravcost = 0;
#pragma omp parallel for reduction(+: gravcost)
    for(i = 0; i < NumPart; i++)
    {
        float costfac = domain_particle_costfactor(i);
        gravcost += costfac;
    }

    MPI_Allreduce(&gravcost, &totgravcost, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    NTopnodes = 1;
    topNodes[0].Daughter = -1;
    topNodes[0].Parent = -1;
    topNodes[0].Size = PEANOCELLS;
    topNodes[0].StartKey = 0;
    topNodes[0].PIndex = 0;
    topNodes[0].Count = NumPart;
    topNodes[0].Cost = gravcost;

    costlimit = totgravcost / (TOPNODEFACTOR * All.DomainOverDecompositionFactor * NTask);
    /*We need TotNumPart to be up to date*/
    int64_t NumPart_long = NumPart;
    MPI_Allreduce(&NumPart_long, &TotNumPart, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);

    countlimit = TotNumPart / (TOPNODEFACTOR * All.DomainOverDecompositionFactor * NTask);

    errflag = domain_check_for_local_refine(0, mp, topNodes);
    walltime_measure("/Domain/DetermineTopTree/LocalRefine");

    myfree(mp);

    MPI_Allreduce(&errflag, &errsum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if(errsum)
    {
        message(0, "We are out of Topnodes. We'll try to repeat with a higher value than All.TopNodeAllocFactor=%g\n",
                 All.TopNodeAllocFactor);
        return errsum;
    }


    /* we now need to exchange tree parts and combine them as needed */

    
    errflag = domain_nonrecursively_combine_topTree(topNodes);

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
        message(0, "can't combine trees due to lack of storage. Will try again.\n");
        return errsum;
    }

    /* now let's see whether we should still append more nodes, based on the estimated cumulative cost/count in each cell */

    message(0, "Before=%d\n", NTopnodes);

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

    message(0, "After=%d\n", NTopnodes);
    walltime_measure("/Domain/DetermineTopTree/Addnodes");

    return 0;
}



void domain_sumCost(float *domainWork, int *domainCount)
{
    int i;
    float * local_domainWork = (float *) mymalloc("local_domainWork", All.NumThreads * NTopnodes * sizeof(float));
    int * local_domainCount = (int *) mymalloc("local_domainCount", All.NumThreads * NTopnodes * sizeof(int));

    memset(local_domainWork, 0, All.NumThreads * NTopnodes * sizeof(float));
    memset(local_domainCount, 0, All.NumThreads * NTopnodes * sizeof(float));

    NTopleaves = 0;
    domain_walktoptree(0);

    message(0, "NTopleaves= %d  NTopnodes=%d (space for %d)\n", NTopleaves, NTopnodes, MaxTopNodes);

#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int n;

        float * mylocal_domainWork = local_domainWork + tid * NTopleaves;
        int * mylocal_domainCount = local_domainCount + tid * NTopleaves;

        #pragma omp for
        for(n = 0; n < NumPart; n++)
        {
            int no = domain_leafnodefunc(P[n].Key);

            mylocal_domainWork[no] += domain_particle_costfactor(n);

            mylocal_domainCount[no] += 1;
        }
    }

#pragma omp parallel for
    for(i = 0; i < NTopleaves; i++)
    {
        int tid;
        for(tid = 1; tid < All.NumThreads; tid++) {
            local_domainWork[i] += local_domainWork[i + tid * NTopleaves];
            local_domainCount[i] += local_domainCount[i + tid * NTopleaves];
        }
    }

    MPI_Allreduce(local_domainWork, domainWork, NTopleaves, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(local_domainCount, domainCount, NTopleaves, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    myfree(local_domainCount);
    myfree(local_domainWork);
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


void domain_insertnode(struct local_topnode_data *treeA, struct local_topnode_data *treeB, int noA, int noB, struct local_topnode_data * topNodes)
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
                    /* This is the only use of the global resource in this function,
                     * and adds a node to the toplevel tree.*/
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
                endrun(88, "Too many Topnodes");
        }

        sub = treeA[noA].Daughter + (treeB[noB].StartKey - treeA[noA].StartKey) / (treeA[noA].Size >> 3);
        domain_insertnode(treeA, treeB, sub, noB, topNodes);
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
                domain_insertnode(treeA, treeB, noA, sub,topNodes);
            }
        }
        else
        {
            if(treeA[noA].Daughter >= 0)
                domain_add_cost(treeA, noA, treeB[noB].Count, treeB[noB].Cost);
        }
    }
    else
        endrun(89, "The tree is corrupted, cannot merge them. What is the invariance here?");
}

static void radix_id(const void * data, void * radix, void * arg) {
    ((uint64_t *) radix)[0] = ((MyIDType*) data)[0];
}

void
domain_test_id_uniqueness(void)
{
    int i;
    double t0, t1;
    MyIDType *ids, *ids_first;

    message(0, "Testing ID uniqueness...\n");

    if(NumPart == 0)
    {
        endrun(8, "need at least one particle per cpu\n");
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
            endrun(12, "non-unique ID=%013ld found on task=%d (i=%d NumPart=%d)\n",
                    ids[i], ThisTask, i, NumPart);

        }

    MPI_Allgather(&ids[0], sizeof(MyIDType), MPI_BYTE, ids_first, sizeof(MyIDType), MPI_BYTE, MPI_COMM_WORLD);

    if(ThisTask < NTask - 1)
        if(ids[NumPart - 1] == ids_first[ThisTask + 1])
        {
            endrun(13, "non-unique ID=%d found on task=%d\n", (int) ids[NumPart - 1], ThisTask);
        }

    myfree(ids_first);
    myfree(ids);

    t1 = second();

    message(0, "success.  took=%g sec\n", timediff(t0, t1));
}
