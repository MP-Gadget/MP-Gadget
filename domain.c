#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <math.h>


#include "allvars.h"
#include "forcetree.h"
#include "mymalloc.h"
#include "mpsort.h"
#include "endrun.h"
#include "openmpsort.h"
#include "domain.h"
#include "timestep.h"
#include "timebinmgr.h"
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

struct topnode_data *TopNodes;
struct topleaf_data *TopLeaves;

int MaxTopNodes;		/*!< Maximum number of nodes in the top-level tree used for domain decomposition */

int NTopNodes, NTopLeaves;

static void * TopTreeMemory;

struct local_topnode_data
{
    /*These members are copied into topnode_data*/
    peano_t StartKey;		/*!< first Peano-Hilbert key in top-level node */
    short int Shift;		/*!< log2 of number of Peano-Hilbert mesh-cells represented by top-level node */
    int Daughter;			/*!< index of first daughter cell (out of 8) of top-level node */
    /*Below members are only used in this file*/
    int Parent;
    int PIndex;			/*!< first particle in node  used only in top-level tree build (this file)*/
    int64_t Count;		/*!< counts the number of particles in this top-level node */
    int64_t Cost;
};

struct local_particle_data
{
    peano_t Key;
    int64_t Cost;
};

struct task_data * Tasks;

static int
order_by_key(const void *a, const void *b);
static void
mp_order_by_key(const void * data, void * radix, void * arg);

static void
domain_assign_balanced(int64_t * cost);
static void domain_allocate(void);
static int
domain_check_memory_bound(const int print_details, int64_t *TopLeafWork, int64_t *TopLeafCount);
static int decompose(void);
static void
domain_balance(void);
static int domain_determineTopTree(struct local_topnode_data * topTree);
static void domain_free(void);
static void domain_compute_costs(int64_t *TopLeafWork, int64_t *TopLeafCount);

static void domain_insertnode(struct local_topnode_data *treeA, struct local_topnode_data *treeB, int noA, int noB, struct local_topnode_data * topTree);
static void domain_add_cost(struct local_topnode_data *treeA, int noA, int64_t count, int64_t cost);
static int domain_check_for_local_refine(const int i, struct local_topnode_data * topTree, int64_t countlimit, int64_t costlimit, struct local_particle_data * LP);
static int domain_check_for_local_refine_subsample(
    struct local_topnode_data * topTree,
    struct local_particle_data * LP,
    int sample_step);
static void
domain_create_topleaves(int no, int * next);

static int domain_layoutfunc(int n);

static int domain_allocated_flag = 0;

/*! This is the main routine for the domain decomposition.  It acts as a
 *  driver routine that allocates various temporary buffers, maps the
 *  particles back onto the periodic box if needed, and then does the
 *  domain decomposition, and a final Peano-Hilbert order of all particles
 *  as a tuning measure.
 */
void domain_decompose_full(void)
{
    int retsum;
    double t0, t1;

    walltime_measure("/Misc");

    if(force_tree_allocated()) force_tree_free();

    domain_garbage_collection();

    domain_free();

    message(0, "domain decomposition... (presently allocated=%g MB)\n", AllocatedBytes / (1024.0 * 1024.0));

    t0 = second();

    do
    {
        int ret;
#ifdef DEBUG
        message(0, "Testing ID Uniqueness before domain decompose\n");
        domain_test_id_uniqueness();
#endif
        domain_allocate();

        ret = decompose();

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

    /* Resort the particles such that those of the same type and key are close by.
     * The locality is broken by the exchange. */
    qsort_openmp(P, NumPart, sizeof(struct particle_data), order_by_type_and_key);

    walltime_measure("/Domain/Peano");

    void * OldTopLeaves = TopLeaves;

    TopNodes  = (struct topnode_data *) (TopTreeMemory);
    TopLeaves = (struct topleaf_data *) (TopNodes + NTopNodes);

    memmove(TopLeaves, OldTopLeaves, NTopNodes * sizeof(TopLeaves[0]));

    /* add 1 extra to mark the end of TopLeaves; see assign */
    TopTreeMemory = (struct topnode_data *) myrealloc(TopTreeMemory,
            (NTopNodes * sizeof(TopNodes[0]) + (NTopLeaves + 1) * sizeof(TopLeaves[0])));


    message(0, "Freed %g MByte in top-level domain structure\n",
                (MaxTopNodes - NTopNodes) * (sizeof(TopLeaves[0])  + sizeof(TopNodes[0]))/ (1024.0 * 1024.0));


    walltime_measure("/Domain/Misc");
    force_tree_rebuild();
}

/* This is a cut-down version of the domain decomposition that leaves the
 * domain grid intact, but exchanges the particles and rebuilds the tree */
void domain_maintain(void)
{
    message(0, "Attempting a domain exchange\n");

    walltime_measure("/Misc");

    /* We rebuild the tree every timestep in order to
     * make sure it is consistent.
     * May as well free it here.*/
    if(force_tree_allocated()) force_tree_free();

    domain_garbage_collection();

    walltime_measure("/Domain/Short/Misc");

    /* Try a domain exchange.
     * If we have no memory for the particles,
     * bail and do a full domain*/
    if(domain_exchange(domain_layoutfunc)) {
        domain_decompose_full();
        return;
    }

    force_tree_rebuild();
}

/*! This function allocates all the stuff that will be required for the tree-construction/walk later on */
void domain_allocate(void)
{
    size_t bytes, all_bytes = 0;

    MaxTopNodes = (int) (All.TopNodeAllocFactor * All.MaxPart + 1);

    /* Add a tail item to avoid special treatments */
    Tasks = mymalloc("Tasks", bytes = ((NTask + 1)* sizeof(Tasks[0])));

    all_bytes += bytes;

    TopTreeMemory = mymalloc("TopTree", 
        bytes = (MaxTopNodes * (sizeof(TopNodes[0]) + sizeof(TopLeaves[0]))));

    TopNodes  = (struct topnode_data *) TopTreeMemory;
    TopLeaves = (struct topleaf_data *) (TopNodes + MaxTopNodes);

    all_bytes += bytes;

    message(0, "Allocated %g MByte for top-level domain structure\n", all_bytes / (1024.0 * 1024.0));

    domain_allocated_flag = 1;
}

void domain_free(void)
{
    if(domain_allocated_flag)
    {
        myfree(TopTreeMemory);
        myfree(Tasks);
        domain_allocated_flag = 0;
    }
}

static int64_t
domain_particle_costfactor(int i)
{
    /* We round off GravCost to integer*/
    if(P[i].TimeBin)
        return (1 + P[i].GravCost) * (TIMEBASE / (1 << P[i].TimeBin));
    else
        return (1 + P[i].GravCost); /* assuming on the full step */
}

/*! This function carries out the actual domain decomposition for all
 *  particle types. It will try to balance the work-load for each domain,
 *  as estimated based on the P[i]-GravCost values.  The decomposition will
 *  respect the maximum allowed memory-imbalance given by the value of
 *  PartAllocFactor.
 */
static int
decompose(void)
{

    int i;

    size_t bytes, all_bytes = 0;

    /*!< points to the root node of the top-level tree */
    struct local_topnode_data *topTree = (struct local_topnode_data *) mymalloc("topTree", bytes =
            (MaxTopNodes * sizeof(struct local_topnode_data)));
    memset(topTree, 0, sizeof(topTree[0]) * MaxTopNodes);
    all_bytes += bytes;

    message(0, "use of %g MB of temporary storage for domain decomposition... (presently allocated=%g MB)\n",
             all_bytes / (1024.0 * 1024.0), AllocatedBytes / (1024.0 * 1024.0));

    report_memory_usage("DOMAIN");

    walltime_measure("/Domain/Decompose/Misc");

    if(domain_determineTopTree(topTree)) {
        myfree(topTree);
        return 1;
    }

    /* copy what we need for the topnodes */
    for(i = 0; i < NTopNodes; i++)
    {
        TopNodes[i].StartKey = topTree[i].StartKey;
        TopNodes[i].Shift = topTree[i].Shift;
        TopNodes[i].Daughter = topTree[i].Daughter;
        TopNodes[i].Leaf = -1; /* will be assigned by create_topleaves*/
    }

    myfree(topTree);

    NTopLeaves = 0;
    domain_create_topleaves(0, &NTopLeaves);

    message(0, "NTopLeaves= %d  NTopNodes=%d (space for %d)\n", NTopLeaves, NTopNodes, MaxTopNodes);

    walltime_measure("/Domain/DetermineTopTree/CreateLeaves");

    if(NTopLeaves < All.DomainOverDecompositionFactor * NTask) {
        message(0, "Number of Topleaves is less than required over decomposition");
    }

    /* this is fatal */
    if(NTopLeaves < NTask) {
        endrun(0, "Number of Topleaves is less than NTask");
    }

    domain_balance();

    walltime_measure("/Domain/Decompose/Balance");
    if(domain_exchange(domain_layoutfunc))
        endrun(1929,"Could not exchange particles\n");

    return 0;
}

static void
domain_balance(void)
{
    /*!< a table that gives the total "work" due to the particles stored by each processor */
    int64_t * TopLeafWork = (int64_t *) mymalloc("TopLeafWork",  NTopLeaves * sizeof(TopLeafWork[0]));
    /*!< a table that gives the total number of particles held by each processor */
    int64_t * TopLeafCount = (int64_t *) mymalloc("TopLeafCount",  NTopLeaves * sizeof(TopLeafCount[0]));

    domain_compute_costs(TopLeafWork, TopLeafCount);

    walltime_measure("/Domain/Decompose/Sumcost");

    /* first try work balance */
    domain_assign_balanced(TopLeafWork);

    walltime_measure("/Domain/Decompose/assignbalance");

    int status = domain_check_memory_bound(0, TopLeafWork, TopLeafCount);
    walltime_measure("/Domain/Decompose/memorybound");

    if(status != 0)		/* the optimum balanced solution violates memory constraint, let's try something different */
    {
        message(0, "Note: the domain decomposition is suboptimum because the ceiling for memory-imbalance is reached\n");

        domain_assign_balanced(TopLeafCount);

        walltime_measure("/Domain/Decompose/assignbalance");

        int status = domain_check_memory_bound(1, TopLeafWork, TopLeafCount);
        walltime_measure("/Domain/Decompose/memorybound");

        if(status != 0)
        {
            endrun(0, "No domain decomposition that stays within memory bounds is possible.\n");
        }
    }

    myfree(TopLeafCount);
    myfree(TopLeafWork);
}

static int
domain_check_memory_bound(const int print_details, int64_t *TopLeafWork, int64_t *TopLeafCount)
{
    int ta, i;
    int load, max_load;
    int64_t sumload;
    int64_t work, max_work, sumwork;
    /*Only used if print_details is true*/
    int64_t list_load[NTask];
    int64_t list_work[NTask];

    max_work = max_load = sumload = sumwork = 0;

    for(ta = 0; ta < NTask; ta++)
    {
        load = 0;
        work = 0;

        for(i = Tasks[ta].StartLeaf; i < Tasks[ta].EndLeaf; i ++)
        {
            load += TopLeafCount[i];
            work += TopLeafWork[i];
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

    message(0, "Largest load: work=%g particle=%g\n",
            max_work / ((double)sumwork / NTask), max_load / (((double) sumload) / NTask));

    if(print_details) {
        message(0, "Balance breakdown:\n");
        for(i = 0; i < NTask; i++)
        {
            message(0, "Task: [%3d]  work=%8.4f  particle load=%8.4f\n", i,
               list_work[i] / ((double) sumwork / NTask), list_load[i] / (((double) sumload) / NTask));
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


struct topleaf_extdata {
    peano_t Key;
    int Task;
    int topnode;
    int64_t cost;
};

static int
topleaf_ext_order_by_task_and_key(const void * c1, const void * c2)
{
    const struct topleaf_extdata * p1 = (const struct topleaf_extdata *) c1;
    const struct topleaf_extdata * p2 = (const struct topleaf_extdata *) c2;
    if(p1->Task < p2->Task) return -1;
    if(p1->Task > p2->Task) return 1;
    if(p1->Key < p2->Key) return -1;
    if(p1->Key > p2->Key) return 1;
    return 0;
}

static int
topleaf_ext_order_by_key(const void * c1, const void * c2)
{
    const struct topleaf_extdata * p1 = (const struct topleaf_extdata *) c1;
    const struct topleaf_extdata * p2 = (const struct topleaf_extdata *) c2;
    if(p1->Key < p2->Key) return -1;
    if(p1->Key > p2->Key) return 1;
    return 0;
}


/*
 *
 * This function assigns task to TopLeaves, and sort them by the task,
 * creates the index in Tasks[Task].StartLeaf and Tasks[Task].EndLeaf
 * cost is the cost per TopLeaves
 *
 * */

static void
domain_assign_balanced(int64_t * cost)
{
    /* we work with TopLeafExt then replace TopLeaves*/

    struct topleaf_extdata * TopLeafExt;

    /* A Segment is a subset of the TopLeaf nodes */

    size_t Nsegment = All.DomainOverDecompositionFactor * NTask;

    if(Nsegment > (1 << 31)) {
        endrun(0, "Too many segments requested, overflowing integer\n");
    }

    TopLeafExt = (struct topleaf_extdata *) mymalloc("TopLeafExt", NTopLeaves * sizeof(TopLeafExt[0]));

    /* copy the data over */
    int i;
    for(i = 0; i < NTopLeaves; i ++) {
        TopLeafExt[i].topnode = TopLeaves[i].topnode;
        TopLeafExt[i].Key = TopNodes[TopLeaves[i].topnode].StartKey;
        TopLeafExt[i].Task = -1;
        TopLeafExt[i].cost = cost[i];
    }

    /* make sure TopLeaves are sorted by Key for locality of segments - 
     * likely not necessary be cause when this function
     * is called it is already true */
    qsort_openmp(TopLeafExt, NTopLeaves, sizeof(TopLeafExt[0]), topleaf_ext_order_by_key);

    int64_t totalcost = 0;
    #pragma omp parallel for reduction(+ : totalcost)
    for(i = 0; i < NTopLeaves; i ++) {
        totalcost += TopLeafExt[i].cost;
    }
    int64_t totalcostLeft = totalcost;

    /* start the assignment; We try to create segments that are of the
     * mean_expected cost, then assign them to Tasks in a round-robin fashion.
     */

    double mean_expected = 1.0 * totalcost / Nsegment;
    double mean_task = 1.0 * totalcost /NTask;
    int curleaf = 0;
    int curseg = 0;
    int curtask = 0; /* between 0 and NTask - 1*/
    int nrounds = 0; /*Number of times we looped*/
    int64_t curload = 0; /* cumulative load for the current segment */
    int64_t curtaskload = 0; /* cumulative load for current task */
    int64_t maxleafcost = 0;

    message(0, "Expected segment cost %g\n", mean_expected);
    /* we maintain that after the loop curleaf is the number of leaves scanned,
     * curseg is number of segments created.
     * */
    while(nrounds < NTopLeaves) {
        int append = 0;
        int advance = 0;
        if(TopLeafExt[curleaf].cost > maxleafcost)
            maxleafcost = TopLeafExt[curleaf].cost;
        if(curleaf == NTopLeaves) {
            /* to maintain the invariance */
            advance = 1;
        } else if(NTopLeaves - curleaf == Nsegment - curseg) {
            /* just enough for one segment per leaf; this line ensures 
             * at least Nsegment segments are created. */
            append = 1;
            advance = 1;
        } else {
            /* append a leaf to the segment if there is room left.
             * Calculate room left based on a rolling average of the total so far..*/
            int64_t totalassigned = (totalcost - totalcostLeft) + curload;
            if((mean_expected * (curseg +1) - totalassigned > 0.5 * TopLeafExt[curleaf].cost)
            || curload == 0 /* but at least add one leaf */
                ) {
                append = 1;
            } else {
                /* will be too big of a segment, cut it */
                advance = 1;
            }
        }
        if(append) {
            /* assign the leaf to the task */
            curload += TopLeafExt[curleaf].cost;
            TopLeafExt[curleaf].Task = curtask;
            curleaf ++;
        }

        if(advance) {
            /*Add this segment to the current task*/
            curtaskload += curload;
            /* If we have allocated enough segments to fill this processor,
             * or if we have one segment per task left, proceed to the next task.
             * We do not use round robin so that neighbouring (by key)
             * topTree are on the same processor.*/
            if((mean_task - curtaskload < 0.5 * mean_expected) || (Nsegment - curseg <= NTask - curtask)
               ){
                curtaskload = 0;
                curtask ++;
            }
            /* move on to the next segment.*/
            totalcostLeft -= curload;
            curload = 0;
            curseg ++;

            /* finished a round for all tasks */
            if(curtask == NTask) {
                /*Back to task zero*/
                curtask = 0;
                /* Need a new mean_expected value: we want to
                 * divide the remaining segments evenly between the processors.*/
                mean_expected = 1.0 * totalcostLeft / Nsegment;
                mean_task = 1.0 * totalcostLeft / NTask;
                nrounds++;
            }
            if(curleaf == NTopLeaves) break;
        }
        //message(0, "curleaf = %d advance = %d append = %d, curload = %d cost=%ld left=%ld\n", curleaf, advance, append, curload, TopLeafExt[curleaf].cost, totalcostLeft);
    }

    /*In most 'normal' cases nrounds == 1 here*/
    message(0, "Created %d segments in %d rounds. Max leaf cost: %g\n", curseg, nrounds, (1.0*maxleafcost)/(totalcost) * Nsegment);

    if(curseg < Nsegment) {
        endrun(0, "Not enough segments were created (%d instead of %d). This should not happen.\n", curseg, Nsegment);
    }

    if(totalcostLeft != 0) {
        endrun(0, "Assertion failed. Total cost is not fully assigned to all ranks\n");
    }

    /* lets rearrange the TopLeafExt by task, such that we can build the Tasks table */
    qsort_openmp(TopLeafExt, NTopLeaves, sizeof(TopLeafExt[0]), topleaf_ext_order_by_task_and_key);
    for(i = 0; i < NTopLeaves; i ++) {
        TopNodes[TopLeafExt[i].topnode].Leaf = i;
        TopLeaves[i].Task = TopLeafExt[i].Task;
        TopLeaves[i].topnode = TopLeafExt[i].topnode;
    }

    myfree(TopLeafExt);
    /* here we reduce the number of code branches by adding an item to the end. */
    TopLeaves[NTopLeaves].Task = NTask;
    TopLeaves[NTopLeaves].topnode = -1;

    int ta = 0;
    Tasks[ta].StartLeaf = 0;
    for(i = 0; i <= NTopLeaves; i ++) {

        if(TopLeaves[i].Task == ta) continue;

        Tasks[ta].EndLeaf = i;
        ta ++;
        while(ta < TopLeaves[i].Task) {
            Tasks[ta].EndLeaf = i;
            Tasks[ta].StartLeaf = i;
            ta ++;
        }
        /* the last item will set Tasks[NTask], but we allocated memory for it already */
        Tasks[ta].StartLeaf = i;
    }
    if(ta != NTask) {
        endrun(0, "Assertion failed: not all tasks are assigned. This indicates a bug.\n");
    }
}

/*This function determines the TopLeaves entry for the given key.*/
static inline int
domain_get_topleaf(const peano_t key) {
    int no=0;
    while(TopNodes[no].Daughter >= 0)
        no = TopNodes[no].Daughter + ((key - TopNodes[no].StartKey) >> (TopNodes[no].Shift - 3));
    no = TopNodes[no].Leaf;
    return no;
}

/* this function determines the TopLeaves entry for the given key, and returns the level of the
 * node in terms of `shift`. */
int
domain_get_topleaf_with_shift(const peano_t key, int * shift) {
    int no=0;
    while(TopNodes[no].Daughter >= 0) {
        no = TopNodes[no].Daughter + ((key - TopNodes[no].StartKey) >> (TopNodes[no].Shift - 3));
    }
    *shift = TopNodes[no].Shift;
    no = TopNodes[no].Leaf;
    return no;
}

/*! This function determines chich particles that are currently stored
 *  on the local CPU have to be moved off according to the domain
 *  decomposition.
 *
 *  layoutfunc decides the target Task of particle p (used by
 *  subfind_distribute).
 *
 */
static int
domain_layoutfunc(int n) {
    peano_t key = P[n].Key;
    int no = domain_get_topleaf(key);
    return TopLeaves[no].Task;
}

/*! This function walks the global top tree in order to establish the
 *  number of leaves it has. These leaves are then distributed to different
 *  processors.
 *
 *  the pointer next points to the next free item on TopLeaves array.
 */
static void
domain_create_topleaves(int no, int * next)
{
    int i;
    if(TopNodes[no].Daughter == -1)
    {
        TopNodes[no].Leaf = *next;
        TopLeaves[*next].topnode = no;
        (*next)++;
    }
    else
    {
        for(i = 0; i < 8; i++)
            domain_create_topleaves(TopNodes[no].Daughter + i, next);
    }
}

static int
domain_toptree_get_subnode(struct local_topnode_data * topTree,
        peano_t key)
{
    int no = 0;
    while(topTree[no].Daughter >= 0) {
        no = topTree[no].Daughter + ((key - topTree[no].StartKey) >> (topTree[no].Shift - 3));
    }
    return no;
}

static int
domain_toptree_insert(struct local_topnode_data * topTree,
        peano_t Key, int64_t cost)
{
    /* insert */
    int leaf = domain_toptree_get_subnode(topTree, Key);
    topTree[leaf].Count ++;
    topTree[leaf].Cost += cost;
    return leaf;
}

static int
domain_toptree_split(struct local_topnode_data * topTree,
    int i) 
{
    int j;
    /* we ran out of top nodes and must get more.*/
    if((NTopNodes + 8) > MaxTopNodes)
        return 1;

    /*Make a new topnode section attached to this node*/
    topTree[i].Daughter = NTopNodes;
    NTopNodes += 8;

    /* Initialise this topnode with new sub nodes*/
    for(j = 0; j < 8; j++)
    {
        const int sub = topTree[i].Daughter + j;
        /* The new sub nodes have this node as parent
         * and no daughters.*/
        topTree[sub].Daughter = -1;
        topTree[sub].Parent = i;
        /* Shorten the peano key by a factor 8, reflecting the oct-tree level.*/
        topTree[sub].Shift = topTree[i].Shift - 3;
        /* This is the region of peanospace covered by this node.*/
        topTree[sub].StartKey = topTree[i].StartKey + j * (1L << topTree[sub].Shift);
        /* We will compute the cost and initialise the first particle in the node below.
         * This PIndex value is never used*/
        topTree[sub].PIndex = topTree[i].PIndex;
        topTree[sub].Count = 0;
        topTree[sub].Cost = 0;
    }
    return 0;
}

static void
domain_toptree_update_cost(struct local_topnode_data * topTree, int start)
{
    if(topTree[start].Daughter == -1) return;

    int j = 0;
    for(j = 0; j < 8; j ++) {
        int sub = topTree[start].Daughter + j;
        domain_toptree_update_cost(topTree, sub);
        topTree[start].Count += topTree[sub].Count;
        topTree[start].Cost += topTree[sub].Cost;
    }
}

/* check for local refinements with subsamples*/
static int
domain_check_for_local_refine_subsample(
    struct local_topnode_data * topTree,
    struct local_particle_data * LP,
    int sample_step)
{
    NTopNodes = 1;
    topTree[0].Daughter = -1;
    topTree[0].Parent = -1;
    topTree[0].Shift = BITS_PER_DIMENSION * 3;
    topTree[0].StartKey = 0;
    topTree[0].PIndex = 0;
    topTree[0].Count = 0;
    topTree[0].Cost = 0;

    int i;

    /* this requires the LP structure to be sorted by key;
     * when the LP structure is sorted by key, we either refine
     * the node the last particle lives in, or jump into a new empty node
     * */

    /* sub sample every 16 particles to create a skeleton of the toptree,
     * because otherwise the tree is too fine!  */
    peano_t last_key = -1;
    int last_leaf = -1;

    i = 0;
    while(i < NumPart) {

        int leaf = domain_toptree_get_subnode(topTree, LP[i].Key);

        if (leaf == last_leaf) {
            /* split last leaf to hopefully make place for the new particle
             * we will hold at most one sample particle per leaf */
            if(0 != domain_toptree_split(topTree, leaf)) return 1;
            /* reinsert the last particle */
            topTree[leaf].Count = 0;
            last_leaf = domain_toptree_insert(topTree, last_key, 0);
            continue;
        } else {
            if(topTree[leaf].Count != 0) {
                /* this shall not happen because key is already sorted. (cite paper)*/
                abort();
            }
            last_key = LP[i].Key;
            last_leaf = domain_toptree_insert(topTree, last_key, 0);
            i += sample_step;
            continue;
        }
    }
    /* Now we are ready to insert all particles to the tree */
    for(i = 0; i < NTopNodes; i ++ ) {
        topTree[i].Count = 0;
    }

    for(i = 0; i < NumPart; i ++ ) {
        domain_toptree_insert(topTree, LP[i].Key, LP[i].Cost);
    }

    domain_toptree_update_cost(topTree, 0);
    return 0;
}

/* Refine the local oct-tree, recursively adding costs and particles until
 * either we have chopped off all the peano-hilbert keys and thus have no more
 * refinement to do, or we run out of topTree.
 * If 1 is returned on any processor we will return to domain_Decomposition,
 * allocate 30% more topTree, and try again.
 * */
int domain_check_for_local_refine(const int i, struct local_topnode_data * topTree, int64_t countlimit, int64_t costlimit, struct local_particle_data * LP)
{
    int j, p;
    int need_refine = 1;

    /* if the node is already very cheap, then no need to divide it any further */
    if(topTree[i].Count <= countlimit &&
        topTree[i].Cost <= costlimit) {
        need_refine = 0;
    }

    /* If we were below them but we have a parent and somehow got all of its particles, we still
     * need to refine. This signals we are in a very is because we want to minimize spatially too large of a leaf
     * that overlaps with other processes during the merge.
     * */
    if(topTree[i].Parent > 0) {
        if (topTree[i].Count >= 0.8 * topTree[topTree[i].Parent].Count &&
            topTree[i].Cost >= 0.8 * topTree[topTree[i].Parent].Cost) {
            need_refine = 1;
        }
    }

    /* However, if there are only 8 particles within this node, we are done refining.*/
    if(topTree[i].Shift < 3) need_refine = 0;

    /* done */
    if(!need_refine) return 0;

    /* If we want to refine but there is no space for another topNode on this processor,
     * we ran out of top nodes and must get more.*/
    if((NTopNodes + 8) > MaxTopNodes)
        return 1;

    /*Make a new topnode section attached to this node*/
    topTree[i].Daughter = NTopNodes;
    NTopNodes += 8;

    /* Initialise this topnode with new sub nodes*/
    for(j = 0; j < 8; j++)
    {
        const int sub = topTree[i].Daughter + j;
        /* The new sub nodes have this node as parent
         * and no daughters.*/
        topTree[sub].Daughter = -1;
        topTree[sub].Parent = i;
        /* Shorten the peano key by a factor 8, reflecting the oct-tree level.*/
        topTree[sub].Shift = topTree[i].Shift - 3;
        /* This is the region of peanospace covered by this node.*/
        topTree[sub].StartKey = topTree[i].StartKey + j * (1L << topTree[sub].Shift);
        /* We will compute the cost and initialise the first particle in the node below.
         * This PIndex value is never used*/
        topTree[sub].PIndex = topTree[i].PIndex;
        topTree[sub].Count = 0;
        topTree[sub].Cost = 0;
    }

    /* Loop over all particles in this node so that the costs of the daughter nodes are correct*/
    for(p = 0, j = 0; p < topTree[i].Count; p++)
    {
        const int sub = topTree[i].Daughter;

        /* This identifies which subnode this particle belongs to.
         * Once this particle has passed the StartKey of the next daughter node,
         * we increment the node the particle is added to and set the PIndex.*/
        if(j < 7)
            while(topTree[sub + j + 1].StartKey <= LP[p + topTree[i].PIndex].Key)
            {
                topTree[sub + j + 1].PIndex = p;
                j++;
                if(j >= 7)
                    break;
            }

        /*Now we have identified the subnode for this particle, add it to the cost and count*/
        topTree[sub+j].Cost += LP[p + topTree[i].PIndex].Cost;
        topTree[sub+j].Count++;
    }

    /*Check and refine the new daughter nodes*/
    for(j = 0; j < 8; j++)
    {
        const int sub = topTree[i].Daughter + j;
        /* Refine each sub node. If we could not refine the node as needed,
         * we are out of node space and need more.*/
        if(domain_check_for_local_refine(sub, topTree, countlimit, costlimit, LP))
            return 1;
    }
    return 0;
}

int domain_nonrecursively_combine_topTree(struct local_topnode_data * topTree)
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
        struct local_topnode_data * topTree_import = NULL;

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
                topTree_import = (struct local_topnode_data *) mymalloc("topTree_import",
                            IMAX(ntopnodes_import, NTopNodes) * sizeof(struct local_topnode_data));

                MPI_Recv(
                        topTree_import,
                        ntopnodes_import, MPI_TYPE_TOPNODE, 
                        recvTask, TAG_GRAV_B, 
                        MPI_COMM_WORLD, MPI_STATUS_IGNORE);


                if((NTopNodes + ntopnodes_import) > MaxTopNodes) {
                    errorflag = 1;
                } else {
                    if(ntopnodes_import < 0) {
                        endrun(1, "severe domain error using a unintended rank \n");
                    }
                    if(ntopnodes_import > 0 ) {
                        domain_insertnode(topTree, topTree_import, 0, 0, topTree);
                    } 
                }
                myfree(topTree_import);
            }
        } else {
            /* odd guys send */
            recvTask = ThisTask - sep;
            if(recvTask >= 0) {
                MPI_Send(&NTopNodes, 1, MPI_INT, recvTask, TAG_GRAV_A,
                        MPI_COMM_WORLD);
                MPI_Send(topTree,
                        NTopNodes, MPI_TYPE_TOPNODE,
                        recvTask, TAG_GRAV_B,
                        MPI_COMM_WORLD);
            }
            NTopNodes = -1;
        }

loop_continue:
        MPI_Allreduce(&errorflag, &errorflagall, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
        if(errorflagall) {
            break;
        }
    }

    MPI_Bcast(&NTopNodes, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(topTree, NTopNodes, MPI_TYPE_TOPNODE, 0, MPI_COMM_WORLD);
    MPI_Type_free(&MPI_TYPE_TOPNODE);
    return errorflagall;
}

/*! This function constructs the global top-level tree node that is used
 *  for the domain decomposition. This is done by considering the string of
 *  Peano-Hilbert keys for all particles, which is recursively chopped off
 *  in pieces of eight segments until each segment holds at most a certain
 *  number of particles.
 */
int domain_determineTopTree(struct local_topnode_data * topTree)
{
    int i, j, sub;
    int errflag, errsum;

    int64_t totgravcost, gravcost = 0;

    struct local_particle_data * LP = (struct local_particle_data*) mymalloc("LocalParticleData", NumPart * sizeof(LP[0]));

#pragma omp parallel for reduction(+: gravcost)
    for(i = 0; i < NumPart; i++)
    {
        LP[i].Key = P[i].Key;
        LP[i].Cost = domain_particle_costfactor(i);
        gravcost += LP[i].Cost;
    }

    /*We need TotNumPart to be up to date*/
    sumup_large_ints(1, &NumPart, &TotNumPart);
    MPI_Allreduce(&gravcost, &totgravcost, 1, MPI_INT64, MPI_SUM, MPI_COMM_WORLD);

    int64_t costlimit, countlimit;

    costlimit = totgravcost / (All.TopNodeIncreaseFactor * All.DomainOverDecompositionFactor * NTask);
    countlimit = TotNumPart / (All.TopNodeIncreaseFactor * All.DomainOverDecompositionFactor * NTask);

    NTopNodes = 1;
    topTree[0].Daughter = -1;
    topTree[0].Parent = -1;
    topTree[0].Shift = BITS_PER_DIMENSION * 3;
    topTree[0].StartKey = 0;
    topTree[0].PIndex = 0;
    topTree[0].Count = NumPart;
    topTree[0].Cost = gravcost;

    walltime_measure("/Domain/DetermineTopTree/Misc");

    /* Watchout : must disgard proximity of particle type; this ordering is only required by LocalRefine */
    //mpsort_mpi(LP, NumPart, sizeof(struct local_particle_data), mp_order_by_key, 8, NULL, MPI_COMM_WORLD);
    qsort_openmp(LP, NumPart, sizeof(struct local_particle_data), order_by_key);

    walltime_measure("/Domain/DetermineTopTree/Sort");

//    errflag = domain_check_for_local_refine(0, topTree, countlimit, costlimit, LP);
    errflag = domain_check_for_local_refine_subsample(topTree, LP, 16);
    walltime_measure("/Domain/DetermineTopTree/LocalRefine");

    if(NTopNodes > 4 * All.DomainOverDecompositionFactor * NTask * All.TopNodeIncreaseFactor) {
        message(1, "NTopNodes=%d >> expected = %d; Usually this indicates very bad imbalance, due to a giant density peak.\n",
            NTopNodes, 4 * All.DomainOverDecompositionFactor * NTask * All.TopNodeIncreaseFactor);
    }

    MPI_Allreduce(&errflag, &errsum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    myfree(LP);

    if(errsum)
    {
        message(0, "We are out of Topnodes. We'll try to repeat with a higher value than All.TopNodeAllocFactor=%g\n",
                 All.TopNodeAllocFactor);
        return errsum;
    }


    /* we now need to exchange tree parts and combine them as needed */

    errflag = domain_nonrecursively_combine_topTree(topTree);

    walltime_measure("/Domain/DetermineTopTree/Combine");
#if 0
    char buf[1000];
    sprintf(buf, "topnodes.bin.%d", ThisTask);
    FILE * fd = fopen(buf, "w");

    /* these PIndex are non-essential in other modules, so we reset them */
    for(i = 0; i < NTopNodes; i ++) {
        topTree[i].PIndex = -1;
    }
    fwrite(topTree, sizeof(struct local_topnode_data), NTopNodes, fd);
    fclose(fd);

    //MPI_Barrier(MPI_COMM_WORLD);
    //MPI_Abort(MPI_COMM_WORLD, 0);
#endif

    /* FIXME: the previous function is already collective, so this step is not needed.
     * We shall probably enforce that every 'long' function must be collective.
     *  */
    MPI_Allreduce(&errflag, &errsum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    if(errsum)
    {
        message(0, "can't combine trees due to lack of storage. Will try again.\n");
        return errsum;
    }

    /* now let's see whether we should still append more nodes, based on the estimated cumulative cost/count in each cell */

    message(0, "TopNodes before appending=%d\n", NTopNodes);

    /* At this point we have refined the local particle tree so that each
     * topNode contains a Cost and Count below the cost threshold. We have then
     * done a global merge of the particle tree. Some of our topTree may now contain
     * more particles than the Cost threshold, but doing refinement using the local
     * algorithm is complicated - particles inside any particular topNode may be
     * on another processor. So we do a local volume based refinement here. This
     * just cuts each topNode above the threshold into 8 equal-sized portions by
     * subdividing the peano key.
     * NOTE: this does not correctly preserve costs! Costs are just divided by 8,
     * because recomputing them for the daughter nodes will be expensive.
     * In practice this seems to work fine, probably because the cost distribution
     * is not that unbalanced. */
    /*Note that NTopNodes will change inside the loop*/
    for(i = 0, errflag = 0; i < NTopNodes; i++)
    {
        /*If this node has no children and non-zero size*/
        if(topTree[i].Daughter < 0 && topTree[i].Shift > 0) {
            /*If this node is also more costly than the limit*/
            if(topTree[i].Count > countlimit || topTree[i].Cost > costlimit) {
                /*If we have no space for another 8 topTree, exit */
                if((NTopNodes + 8) > MaxTopNodes) {
                    errflag = 1;
                    break;
                }

                topTree[i].Daughter = NTopNodes;

                for(j = 0; j < 8; j++)
                {
                    sub = topTree[i].Daughter + j;
                    topTree[sub].Shift = topTree[i].Shift - 3;
                    topTree[sub].Count = topTree[i].Count / 8;
                    topTree[sub].Cost = topTree[i].Cost / 8;
                    topTree[sub].Daughter = -1;
                    topTree[sub].Parent = i;
                    topTree[sub].StartKey = topTree[i].StartKey + j * (1L << topTree[sub].Shift);
                }
                NTopNodes += 8;
            }
        }
    }

    MPI_Allreduce(&errflag, &errsum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if(errsum)
        return errsum;

    message(0, "Final NTopNodes = %d per segment = %g.\n", NTopNodes, 1.0 * NTopNodes / (All.DomainOverDecompositionFactor * NTask));
    walltime_measure("/Domain/DetermineTopTree/Addnodes");

    return 0;
}



void domain_compute_costs(int64_t *TopLeafWork, int64_t *TopLeafCount)
{
    int i;
    int64_t * local_TopLeafWork = (int64_t *) mymalloc("local_TopLeafWork", All.NumThreads * NTopLeaves * sizeof(local_TopLeafWork[0]));
    int64_t * local_TopLeafCount = (int64_t *) mymalloc("local_TopLeafCount", All.NumThreads * NTopLeaves * sizeof(local_TopLeafCount[0]));

    memset(local_TopLeafWork, 0, All.NumThreads * NTopLeaves * sizeof(local_TopLeafWork[0]));
    memset(local_TopLeafCount, 0, All.NumThreads * NTopLeaves * sizeof(local_TopLeafCount[0]));

#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int n;

        int64_t * mylocal_TopLeafWork = local_TopLeafWork + tid * NTopLeaves;
        int64_t * mylocal_TopLeafCount = local_TopLeafCount + tid * NTopLeaves;

        #pragma omp for
        for(n = 0; n < NumPart; n++)
        {
            int no = domain_get_topleaf(P[n].Key);

            mylocal_TopLeafWork[no] += domain_particle_costfactor(n);

            mylocal_TopLeafCount[no] += 1;
        }
    }

#pragma omp parallel for
    for(i = 0; i < NTopLeaves; i++)
    {
        int tid;
        for(tid = 1; tid < All.NumThreads; tid++) {
            local_TopLeafWork[i] += local_TopLeafWork[i + tid * NTopLeaves];
            local_TopLeafCount[i] += local_TopLeafCount[i + tid * NTopLeaves];
        }
    }

    MPI_Allreduce(local_TopLeafWork, TopLeafWork, NTopLeaves, MPI_INT64, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(local_TopLeafCount, TopLeafCount, NTopLeaves, MPI_INT64, MPI_SUM, MPI_COMM_WORLD);
    myfree(local_TopLeafCount);
    myfree(local_TopLeafWork);
}

void domain_add_cost(struct local_topnode_data *treeA, int noA, int64_t count, int64_t cost)
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


/* FIXME: this function needs some comments. I used to know what it does --. YF*/
void domain_insertnode(struct local_topnode_data *treeA, struct local_topnode_data *treeB, int noA, int noB, struct local_topnode_data * topTree)
{
    int j, sub;
    int64_t count, countA, countB;
    int64_t cost, costA, costB;

    if(treeB[noB].Shift < treeA[noA].Shift)
    {
        if(treeA[noA].Daughter < 0)
        {
            if((NTopNodes + 8) <= MaxTopNodes)
            {
                count = treeA[noA].Count - treeB[treeB[noB].Parent].Count;
                countB = count / 8;
                countA = count - 7 * countB;

                cost = treeA[noA].Cost - treeB[treeB[noB].Parent].Cost;
                costB = cost / 8;
                costA = cost - 7 * costB;

                treeA[noA].Daughter = NTopNodes;
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
                    topTree[sub].Shift = treeA[noA].Shift - 3;
                    topTree[sub].Count = count;
                    topTree[sub].Cost = cost;
                    topTree[sub].Daughter = -1;
                    topTree[sub].Parent = noA;
                    topTree[sub].StartKey = treeA[noA].StartKey + j * (1L << treeA[sub].Shift);
                }
                NTopNodes += 8;
            }
            else
                endrun(88, "Too many Topnodes");
        }

        sub = treeA[noA].Daughter + ((treeB[noB].StartKey - treeA[noA].StartKey) >> (treeA[noA].Shift - 3));
        domain_insertnode(treeA, treeB, sub, noB, topTree);
    }
    else if(treeB[noB].Shift == treeA[noA].Shift)
    {
        treeA[noA].Count += treeB[noB].Count;
        treeA[noA].Cost += treeB[noB].Cost;

        if(treeB[noB].Daughter >= 0)
        {
            for(j = 0; j < 8; j++)
            {
                sub = treeB[noB].Daughter + j;
                domain_insertnode(treeA, treeB, noA, sub,topTree);
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

/* used only by test uniqueness */
static void
mp_order_by_id(const void * data, void * radix, void * arg) {
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

    mpsort_mpi(ids, NumPart, sizeof(MyIDType), mp_order_by_id, 8, NULL, MPI_COMM_WORLD);

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

static int
order_by_key(const void *a, const void *b)
{
    const struct local_particle_data * pa  = (const struct local_particle_data *) a;
    const struct local_particle_data * pb  = (const struct local_particle_data *) b;
    if(pa->Key < pb->Key)
        return -1;

    if(pa->Key > pb->Key)
        return +1;

    return 0;
}

static void
mp_order_by_key(const void * data, void * radix, void * arg)
{
    const struct local_particle_data * pa  = (const struct local_particle_data *) data;
    ((uint64_t *) radix)[0] = pa->Key;
}

static int
order_by_type_and_key(const void *a, const void *b)
{
    const struct particle_data * pa  = (const struct particle_data *) a;
    const struct particle_data * pb  = (const struct particle_data *) b;

    if(pa->Type < pb->Type)
        return -1;
    if(pa->Type > pb->Type)
        return +1;
    if(pa->Key < pb->Key)
        return -1;
    if(pa->Key > pb->Key)
        return +1;

    return 0;
}

