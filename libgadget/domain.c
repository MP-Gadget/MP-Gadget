#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <omp.h>

#include "utils.h"

#include "mpsort.h"
#include "domain.h"
#include "timestep.h"
#include "timebinmgr.h"
#include "exchange.h"
#include "slotsmanager.h"
#include "partmanager.h"
#include "walltime.h"
#include "utils/paramset.h"
#include "utils/peano.h"

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

/*Parameters of the domain decomposition, set by the input parameter file*/
static struct DomainParams
{
    /* Number of TopLeaves (Peano-Hilbert segments) per processor. TopNodes are refined so that no TopLeaf contains
     * no more than 1/(DODF * NTask) fraction of the work.
     * The load balancer will assign these TopLeaves so that each MPI rank has a similar amount of work.*/
    int DomainOverDecompositionFactor;
    /** Use a global sort for the first few domain policies to try.*/
    int DomainUseGlobalSorting;
    /** Initial number of Top level tree nodes as a fraction of particles */
    double TopNodeAllocFactor;
    /** Fraction of local particle slots to leave free for, eg, star formation*/
    double SetAsideFactor;
} domain_params;

/**
 * Policy for domain decomposition.
 *
 * Several policies will be attempted before we give up and die.
 * */
typedef struct {
    double TopNodeAllocFactor; /** number of Top level tree nodes as a fraction of particles */
    int UseGlobalSort; /** Apply a global sorting on the subsamples before building the top level tree. */
    MPI_Comm GlobalSortComm; /** Communicator to use for the global sort. By default MPI_COMM_WORLD. */
    int SubSampleDistance; /** Frequency of subsampling */
    int PreSort; /** PreSort the local particles before subsampling, creating a fair subsample */
    int NTopLeaves; /** Number of Peano-Hilbert segments to create before balancing. Should be DomainOverDecompositionFactor * NTask*/
} DomainDecompositionPolicy;

int MaxTopNodes;		/*!< Maximum number of nodes in the top-level tree used for domain decomposition */

static void * TopTreeTempMemory;

struct local_topnode_data
{
    /*These members are copied into topnode_data*/
    peano_t StartKey;		/*!< first Peano-Hilbert key in top-level node */
    short int Shift;		/*!< log2 of number of Peano-Hilbert mesh-cells represented by top-level node */
    int Daughter;			/*!< index of first daughter cell (out of 8) of top-level node */
    /*Below members are only used in this file*/
    int Parent;
    int PIndex;         /* FIXME: this appears to be useless now. first particle in node used only in top-level tree build (this file)*/
    int64_t Count;      /* the number of 'subsample' particles in this top-level node */
    int64_t Cost;       /* the cost of 'subsample' particle in this top-level node */
};

struct local_particle_data
{
    peano_t Key;
    int64_t Cost;
};

/*Set the parameters of the domain module*/
void set_domain_params(ParameterSet * ps)
{
    int ThisTask;
    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
    if(ThisTask == 0) {
        domain_params.DomainOverDecompositionFactor = param_get_int(ps, "DomainOverDecompositionFactor");
        domain_params.TopNodeAllocFactor = param_get_double(ps, "TopNodeAllocFactor");
        domain_params.DomainUseGlobalSorting = param_get_int(ps, "DomainUseGlobalSorting");
        domain_params.SetAsideFactor = 1.;
        if((param_get_int(ps, "StarformationOn") && param_get_double(ps, "QuickLymanAlphaProbability") == 0.)
            || param_get_int(ps, "BlackHoleOn"))
            domain_params.SetAsideFactor = 0.95;
    }
    MPI_Bcast(&domain_params, sizeof(struct DomainParams), MPI_BYTE, 0, MPI_COMM_WORLD);
}

static int
order_by_key(const void *a, const void *b);
static void
mp_order_by_key(const void * data, void * radix, void * arg);

static void
domain_assign_balanced(DomainDecomp * ddecomp, int64_t * cost, const int NsegmentPerTask);

static void domain_allocate(DomainDecomp * ddecomp, DomainDecompositionPolicy * policy);

static int
domain_check_memory_bound(const DomainDecomp * ddecomp, const int print_details, int64_t *TopLeafWork, int64_t *TopLeafCount);

static int domain_attempt_decompose(DomainDecomp * ddecomp, DomainDecompositionPolicy * policy);

static void
domain_balance(DomainDecomp * ddecomp);

static int domain_determine_global_toptree(DomainDecompositionPolicy * policy, struct local_topnode_data * topTree, int * topTreeSize, MPI_Comm DomainComm);
static void domain_free(DomainDecomp * ddecomp);

static void
domain_compute_costs(const DomainDecomp * ddecomp, int64_t *TopLeafWork, int64_t *TopLeafCount);

static void
domain_toptree_merge(struct local_topnode_data *treeA, struct local_topnode_data *treeB, int noA, int noB, int * treeASize);

static int domain_check_for_local_refine_subsample(
    DomainDecompositionPolicy * policy,
    struct local_topnode_data * topTree, int * topTreeSize
    );

static int
domain_global_refine(struct local_topnode_data * topTree, int * topTreeSize, int64_t countlimit, int64_t costlimit);

static void
domain_create_topleaves(DomainDecomp * ddecomp, int no, int * next);

static int domain_layoutfunc(int n, const void * userdata);

static int
domain_policies_init(DomainDecompositionPolicy policies[],
        const int NincreaseAlloc,
        const int SwitchToGlobal);

/*! This is the main routine for the domain decomposition.  It acts as a
 *  driver routine that allocates various temporary buffers, maps the
 *  particles back onto the periodic box if needed, and then does the
 *  domain decomposition, and a final Peano-Hilbert order of all particles
 *  as a tuning measure.
 */
void domain_decompose_full(DomainDecomp * ddecomp)
{
    static DomainDecompositionPolicy policies[16];
    static int Npolicies = 0;

    /* start from last successful policy to avoid retries */
    static int LastSuccessfulPolicy = 0;

    if (Npolicies == 0) {
        const int NincreaseAlloc = 16;
        Npolicies = domain_policies_init(policies, NincreaseAlloc, 8);
    }

    walltime_measure("/Misc");

    message(0, "domain decomposition... (presently allocated=%g MB)\n", mymalloc_usedbytes() / (1024.0 * 1024.0));

    int decompose_failed = 1;
    int i;
    for(i = LastSuccessfulPolicy; i < Npolicies; i ++)
    {
        domain_free(ddecomp);

#ifdef DEBUG
        domain_test_id_uniqueness();
#endif
        domain_allocate(ddecomp, &policies[i]);

        message(0, "Attempting new domain decomposition policy: TopNodeAllocFactor=%g, UseglobalSort=%d, SubSampleDistance=%d UsePreSort=%d\n",
                policies[i].TopNodeAllocFactor, policies[i].UseGlobalSort, policies[i].SubSampleDistance, policies[i].PreSort);

        decompose_failed = MPIU_Any(0 != domain_attempt_decompose(ddecomp, &policies[i]), ddecomp->DomainComm);

        if(!decompose_failed) {
            LastSuccessfulPolicy = i;
            break;
        }
    }

    if(decompose_failed) {
        endrun(0, "No suitable domain decomposition policy worked for this particle distribution\n");
    }

    domain_balance(ddecomp);

    walltime_measure("/Domain/Decompose/Balance");

    /* copy the used nodes from temp to the true. */
    void * OldTopLeaves = ddecomp->TopLeaves;
    void * OldTopNodes = ddecomp->TopNodes;

    ddecomp->TopNodes  = (struct topnode_data *) mymalloc2("TopNodes", sizeof(ddecomp->TopNodes[0]) * ddecomp->NTopNodes);
    /* add 1 extra to mark the end of TopLeaves; see assign */
    ddecomp->TopLeaves = (struct topleaf_data *) mymalloc2("TopLeaves", sizeof(ddecomp->TopLeaves[0]) * (ddecomp->NTopLeaves + 1));

    memcpy(ddecomp->TopLeaves, OldTopLeaves, ddecomp->NTopLeaves* sizeof(ddecomp->TopLeaves[0]));
    memcpy(ddecomp->TopNodes, OldTopNodes, ddecomp->NTopNodes * sizeof(ddecomp->TopNodes[0]));

    /* no longer useful */
    myfree(TopTreeTempMemory);
    TopTreeTempMemory = NULL;

    if(domain_exchange(domain_layoutfunc, ddecomp, 0, ddecomp->DomainComm))
        endrun(1929,"Could not exchange particles\n");

    /*Do a garbage collection so that the slots are ordered
     *the same as the particles, garbage is at the end and all particles are in peano order.*/
    slots_gc_sorted();

    /*Ensure collective*/
    MPIU_Barrier(ddecomp->DomainComm);
    message(0, "Domain decomposition done.\n");

    report_memory_usage("DOMAIN");

    walltime_measure("/Domain/PeanoSort");
}

/* This is a cut-down version of the domain decomposition that leaves the
 * domain grid intact, but exchanges the particles and rebuilds the tree */
void domain_maintain(DomainDecomp * ddecomp)
{
    message(0, "Attempting a domain exchange\n");

    walltime_measure("/Misc");

    /* Try a domain exchange.
     * If we have no memory for the particles,
     * bail and do a full domain*/
    if(0 != domain_exchange(domain_layoutfunc, ddecomp, 0, ddecomp->DomainComm)) {
        domain_decompose_full(ddecomp);
        return;
    }
}

/* this function generates several domain decomposition policies for attempting
 * creating the domain. */
static int
domain_policies_init(DomainDecompositionPolicy policies[],
        const int NincreaseAlloc,
        const int SwitchToGlobal)
{
    int NTask;
    MPI_Comm_size(MPI_COMM_WORLD, &NTask);

    int i;
    for(i = 0; i < NincreaseAlloc; i ++) {
        policies[i].TopNodeAllocFactor = domain_params.TopNodeAllocFactor * pow(1.3, i);
        policies[i].UseGlobalSort = domain_params.DomainUseGlobalSorting;
        /* The extent of the global sorting may be different from the extent of the Domain communicator*/
        policies[i].GlobalSortComm = MPI_COMM_WORLD;
        policies[i].PreSort = 0;
        policies[i].SubSampleDistance = 16;
        /* Desired number of TopLeaves should scale like the total number of processors*/
        policies[i].NTopLeaves = domain_params.DomainOverDecompositionFactor * NTask;
    }

    for(i = SwitchToGlobal; i < NincreaseAlloc; i ++) {
        /* global sorting is slower than a local sorting, but tends to produce a more
         * balanced domain tree that is easier to merge.
         * */
        policies[i].UseGlobalSort = 1;
        /* global sorting of particles is slow, so we add a slower presort to even the local
         * particle distribution before subsampling, improves the balance, too */
        policies[i].PreSort = 1;
    }

    return NincreaseAlloc;
}

/*! This function allocates all the stuff that will be required for the tree-construction/walk later on */
static void
domain_allocate(DomainDecomp * ddecomp, DomainDecompositionPolicy * policy)
{
    size_t bytes, all_bytes = 0;

    MaxTopNodes = (int) (policy->TopNodeAllocFactor * PartManager->MaxPart + 1);

    /* Build the domain over the global all-processors communicator.
     * We use a symbol in case we want to do fancy things in the future.*/
    ddecomp->DomainComm = MPI_COMM_WORLD;

    int NTask;
    MPI_Comm_size(ddecomp->DomainComm, &NTask);

    /* Add a tail item to avoid special treatments */
    ddecomp->Tasks = mymalloc2("Tasks", bytes = ((NTask + 1)* sizeof(ddecomp->Tasks[0])));

    all_bytes += bytes;

    TopTreeTempMemory = mymalloc("TopTreeWorkspace",
        bytes = (MaxTopNodes * (sizeof(ddecomp->TopNodes[0]) + sizeof(ddecomp->TopLeaves[0]))));

    ddecomp->TopNodes  = (struct topnode_data *) TopTreeTempMemory;
    ddecomp->TopLeaves = (struct topleaf_data *) (ddecomp->TopNodes + MaxTopNodes);

    all_bytes += bytes;

    message(0, "Allocated %g MByte for top-level domain structure\n", all_bytes / (1024.0 * 1024.0));

    ddecomp->domain_allocated_flag = 1;
}

void domain_free(DomainDecomp * ddecomp)
{
    if(ddecomp->domain_allocated_flag)
    {
        myfree(ddecomp->TopLeaves);
        myfree(ddecomp->TopNodes);
        myfree(ddecomp->Tasks);
        ddecomp->domain_allocated_flag = 0;
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
 *  particle types. It will try to balance the work-load for each ddecomp,
 *  as estimated based on the P[i]-GravCost values.  The decomposition will
 *  respect the maximum allowed memory-imbalance given by the value of
 *  PartAllocFactor.
 */
static int
domain_attempt_decompose(DomainDecomp * ddecomp, DomainDecompositionPolicy * policy)
{

    int i;

    size_t bytes, all_bytes = 0;

    /* points to the root node of the top-level tree */
    struct local_topnode_data *topTree = (struct local_topnode_data *) mymalloc("LocaltopTree", bytes =
            (MaxTopNodes * sizeof(struct local_topnode_data)));
    memset(topTree, 0, sizeof(topTree[0]) * MaxTopNodes);
    all_bytes += bytes;

    message(0, "use of %g MB of temporary storage for domain decomposition... (presently allocated=%g MB)\n",
             all_bytes / (1024.0 * 1024.0), mymalloc_usedbytes() / (1024.0 * 1024.0));

    report_memory_usage("DOMAIN");

    walltime_measure("/Domain/Decompose/Misc");

    if(domain_determine_global_toptree(policy, topTree, &ddecomp->NTopNodes, ddecomp->DomainComm)) {
        myfree(topTree);
        return 1;
    }

    /* copy what we need for the topnodes */
    for(i = 0; i < ddecomp->NTopNodes; i++)
    {
        ddecomp->TopNodes[i].StartKey = topTree[i].StartKey;
        ddecomp->TopNodes[i].Shift = topTree[i].Shift;
        ddecomp->TopNodes[i].Daughter = topTree[i].Daughter;
        ddecomp->TopNodes[i].Leaf = -1; /* will be assigned by create_topleaves*/
    }

    myfree(topTree);

    ddecomp->NTopLeaves = 0;
    domain_create_topleaves(ddecomp, 0, &ddecomp->NTopLeaves);

    message(0, "NTopLeaves= %d  NTopNodes=%d (space for %d)\n", ddecomp->NTopLeaves, ddecomp->NTopNodes, MaxTopNodes);

    walltime_measure("/Domain/DetermineTopTree/CreateLeaves");

    if(ddecomp->NTopLeaves < policy->NTopLeaves) {
        message(0, "Number of Topleaves is less than required over decomposition");
    }

    int NTask;
    MPI_Comm_size(ddecomp->DomainComm, &NTask);

    /* this is fatal */
    if(ddecomp->NTopLeaves < NTask) {
        endrun(0, "Number of Topleaves is less than NTask");
    }

    return 0;
}

/**
 * attempt to assign segments to tasks such that the load or work is balanced.
 *
 * */
static void
domain_balance(DomainDecomp * ddecomp)
{
    /*!< a table that gives the total "work" due to the particles stored by each processor */
    int64_t * TopLeafWork = (int64_t *) mymalloc("TopLeafWork",  ddecomp->NTopLeaves * sizeof(TopLeafWork[0]));
    /*!< a table that gives the total number of particles held by each processor */
    int64_t * TopLeafCount = (int64_t *) mymalloc("TopLeafCount",  ddecomp->NTopLeaves * sizeof(TopLeafCount[0]));

    domain_compute_costs(ddecomp, TopLeafWork, TopLeafCount);

    walltime_measure("/Domain/Decompose/Sumcost");

    /* first try work balance */
    domain_assign_balanced(ddecomp, TopLeafWork, 1);

    walltime_measure("/Domain/Decompose/assignbalance");

    int status = domain_check_memory_bound(ddecomp, 0, TopLeafWork, TopLeafCount);
    walltime_measure("/Domain/Decompose/memorybound");

    if(status != 0)		/* the optimum balanced solution violates memory constraint, let's try something different */
    {
        message(0, "Note: the domain decomposition is suboptimum because the ceiling for memory-imbalance is reached\n");

        domain_assign_balanced(ddecomp, TopLeafCount, 1);

        walltime_measure("/Domain/Decompose/assignbalance");

        int status = domain_check_memory_bound(ddecomp, 1, TopLeafWork, TopLeafCount);
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
domain_check_memory_bound(const DomainDecomp * ddecomp, const int print_details, int64_t *TopLeafWork, int64_t *TopLeafCount)
{
    int ta, i;
    int load, max_load;
    int64_t sumload;
    int64_t work, max_work, sumwork;
    int NTask;
    MPI_Comm_size(ddecomp->DomainComm, &NTask);

    /*Only used if print_details is true*/
    int64_t *list_load = NULL;
    int64_t *list_work = NULL;
    if(print_details) {
        list_load = ta_malloc("list_load",int64_t, 2*NTask);
        list_work = list_load + NTask;
    }

    max_work = max_load = sumload = sumwork = 0;

    for(ta = 0; ta < NTask; ta++)
    {
        load = 0;
        work = 0;

        for(i = ddecomp->Tasks[ta].StartLeaf; i < ddecomp->Tasks[ta].EndLeaf; i ++)
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
        ta_free(list_load);
    }

    /*Leave a small number of particles for star formation */
    if(max_load > PartManager->MaxPart * domain_params.SetAsideFactor)
    {
        message(0, "desired memory imbalance=%g  (limit=%g, needed=%d)\n",
                    (max_load * ((double) sumload ) / NTask ) / PartManager->MaxPart, domain_params.SetAsideFactor * PartManager->MaxPart, max_load);

        return 1;
    }

    return 0;
}


/* Data of TopTree leaf nodes that are used for assignment to tasks */
struct topleaf_extdata {
    peano_t Key;
    int Task;        /** The task that receives the node */
    int topnode;     /** The node */
    int64_t cost;    /** cost value, can be either number of calculations or number of particles. */
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


/**
 * This function assigns TopLeaves to Segments, trying to ensure uniform cost
 * by assigning TopLeaves (contiguously) until a Segment has a desired size.
 * Then each Segment is assigned to a Task in a round-robin fashion starting from the largest Segment
 * until all Segments are assigned.
 * At the moment the Segment/Task distinction is not used: we call this with NSegmentPerTask = 1, because
 * we are able to create Segments of roughly equal cost from the TopLeaves.
 *
 * This creates the index in Tasks[Task].StartLeaf and Tasks[Task].EndLeaf
 * cost is the cost per TopLeaves
 *
 * */
static void
domain_assign_balanced(DomainDecomp * ddecomp, int64_t * cost, const int NsegmentPerTask)
{
    int NTask;
    MPI_Comm_size(ddecomp->DomainComm, &NTask);

    int Nsegment = NTask * NsegmentPerTask;
    /* we work with TopLeafExt then replace TopLeaves*/

    struct topleaf_extdata * TopLeafExt;

    /* A Segment is a subset of the TopLeaf nodes */

    TopLeafExt = (struct topleaf_extdata *) mymalloc("TopLeafExt", ddecomp->NTopLeaves * sizeof(TopLeafExt[0]));

    /* copy the data over */
    int i;
    for(i = 0; i < ddecomp->NTopLeaves; i ++) {
        TopLeafExt[i].topnode = ddecomp->TopLeaves[i].topnode;
        TopLeafExt[i].Key = ddecomp->TopNodes[ddecomp->TopLeaves[i].topnode].StartKey;
        TopLeafExt[i].Task = -1;
        TopLeafExt[i].cost = cost[i];
    }

    /* make sure TopLeaves are sorted by Key for locality of segments -
     * likely not necessary be cause when this function
     * is called it is already true */
    qsort_openmp(TopLeafExt, ddecomp->NTopLeaves, sizeof(TopLeafExt[0]), topleaf_ext_order_by_key);

    int64_t totalcost = 0;
    #pragma omp parallel for reduction(+ : totalcost)
    for(i = 0; i < ddecomp->NTopLeaves; i ++) {
        totalcost += TopLeafExt[i].cost;
    }
    int64_t totalcostLeft = totalcost;

    /* start the assignment; We try to create segments that are of the
     * mean_expected cost, then assign them to Tasks in a round-robin fashion.
     * The default is that there is one segment per task, which makes the assignment trivial.
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
    while(nrounds < ddecomp->NTopLeaves) {
        int append = 0;
        int advance = 0;
        if(curleaf == ddecomp->NTopLeaves) {
            /* to maintain the invariance */
            advance = 1;
        } else if(ddecomp->NTopLeaves - curleaf == Nsegment - curseg) {
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
            if(curleaf == ddecomp->NTopLeaves)
                break;
            if(TopLeafExt[curleaf].cost > maxleafcost)
                maxleafcost = TopLeafExt[curleaf].cost;
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
    qsort_openmp(TopLeafExt, ddecomp->NTopLeaves, sizeof(TopLeafExt[0]), topleaf_ext_order_by_task_and_key);
    for(i = 0; i < ddecomp->NTopLeaves; i ++) {
        ddecomp->TopNodes[TopLeafExt[i].topnode].Leaf = i;
        ddecomp->TopLeaves[i].Task = TopLeafExt[i].Task;
        ddecomp->TopLeaves[i].topnode = TopLeafExt[i].topnode;
    }

    myfree(TopLeafExt);
    /* here we reduce the number of code branches by adding an item to the end. */
    ddecomp->TopLeaves[ddecomp->NTopLeaves].Task = NTask;
    ddecomp->TopLeaves[ddecomp->NTopLeaves].topnode = -1;

    int ta = 0;
    ddecomp->Tasks[ta].StartLeaf = 0;
    for(i = 0; i <= ddecomp->NTopLeaves; i ++) {

        if(ddecomp->TopLeaves[i].Task == ta) continue;

        ddecomp->Tasks[ta].EndLeaf = i;
        ta ++;
        while(ta < ddecomp->TopLeaves[i].Task) {
            ddecomp->Tasks[ta].EndLeaf = i;
            ddecomp->Tasks[ta].StartLeaf = i;
            ta ++;
        }
        /* the last item will set Tasks[NTask], but we allocated memory for it already */
        ddecomp->Tasks[ta].StartLeaf = i;
    }
    if(ta != NTask) {
        endrun(0, "Assertion failed: not all tasks are assigned. This indicates a bug.\n");
    }
}

/*! This function determines which particles that are currently stored
 *  on the local CPU have to be moved off according to the domain
 *  decomposition.
 *
 *  layoutfunc decides the target Task of particle p (used by
 *  subfind_distribute).
 *
 */
static int
domain_layoutfunc(int n, const void * userdata) {
    const DomainDecomp * ddecomp = (const DomainDecomp *) userdata;
    peano_t key = P[n].Key;
    int no = domain_get_topleaf(key, ddecomp);
    return ddecomp->TopLeaves[no].Task;
}

/*! This function walks the global top tree in order to establish the
 *  number of leaves it has. These leaves are then distributed to different
 *  processors.
 *
 *  the pointer next points to the next free item on TopLeaves array.
 */
static void
domain_create_topleaves(DomainDecomp * ddecomp, int no, int * next)
{
    int i;
    if(ddecomp->TopNodes[no].Daughter == -1)
    {
        ddecomp->TopNodes[no].Leaf = *next;
        ddecomp->TopLeaves[*next].topnode = no;
        (*next)++;
    }
    else
    {
        for(i = 0; i < 8; i++)
            domain_create_topleaves(ddecomp, ddecomp->TopNodes[no].Daughter + i, next);
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
domain_toptree_split(struct local_topnode_data * topTree, int * topTreeSize,
    int i)
{
    int j;
    /* we ran out of top nodes and must get more.*/
    if((*topTreeSize + 8) > MaxTopNodes)
        return 1;

    if(topTree[i].Shift < 3) {
        endrun(1, "Failed to build a TopTree -- particles overly clustered.\n");
    }

    /*Make a new topnode section attached to this node*/
    topTree[i].Daughter = *topTreeSize;
    (*topTreeSize) += 8;

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

/* This function recurively identify and terminate tree branches that are cheap.*/
static void
domain_toptree_truncate_r(struct local_topnode_data * topTree, int start, int64_t countlimit, int64_t costlimit)
{
    if(topTree[start].Daughter == -1) return;

    if(topTree[start].Count < countlimit &&
       topTree[start].Cost < costlimit) {
        /* truncate here */
        topTree[start].Daughter = -1;
        return;
    }

    int j;
    for(j = 0; j < 8; j ++) {
        int sub = topTree[start].Daughter + j;
        domain_toptree_truncate_r(topTree, sub, countlimit, costlimit);
    }
}

/* remove the nodes that are no longer useful after the truncation.
 *
 * We walk the topTree top-down to collect useful nodes, and move them to
 * the head of the topTree list.
 *
 * This works because any child node is stored after the parent in the list
 * -- we are destroying the old tree just slow enough
 * */

static void
domain_toptree_garbage_collection(struct local_topnode_data * topTree, int start, int * last_free)
{

    if(topTree[start].Daughter == -1) return;
    int j;

    int oldd = topTree[start].Daughter;
    int newd = *last_free;

    topTree[start].Daughter = newd;

    (*last_free) += 8;

    /* copy first in case oldd + j is overwritten by the recursed gc
     * if last_free is less than oldd */
    for(j = 0; j < 8; j ++) {
        topTree[newd + j] = topTree[oldd + j];
        topTree[newd + j].Parent = start;
    }

    for(j = 0; j < 8; j ++) {
        domain_toptree_garbage_collection(topTree, newd + j, last_free);
    }
}

static void
domain_toptree_truncate(
    struct local_topnode_data * topTree, int * topTreeSize,
    int64_t countlimit, int64_t costlimit)
{

    /* first terminate the tree.*/
    domain_toptree_truncate_r(topTree, 0, countlimit, costlimit);

    /* then remove the unused nodes from the topTree storage. This is important
     * for efficient global merge */
    *topTreeSize = 1; /* put in the root node -- it's never a garbage . */
    domain_toptree_garbage_collection(topTree, 0, topTreeSize);
}

/**
 * This function performs local refinement of the topTree.
 *
 * It creates the local refinement by quickly going through
 * a skeleton tree of a fraction (subsample) of all local particles.
 *
 * Next, we add flesh to the skeleton: all particles are deposited
 * to the tree.
 *
 * Finally, in _truncation(), we chop off the cheap branches from the skeleton,
 * terminating the branches when the cost and count are both sufficiently small.
 *
 * We do not use the full particle data. Sufficient to use LP, which contains
 * the peano key and cost of each particle, and sorted by the peano key.
 * */
static int
domain_check_for_local_refine_subsample(
    DomainDecompositionPolicy * policy,
    struct local_topnode_data * topTree, int * topTreeSize
    )
{

    int i;

    struct local_particle_data * LP = (struct local_particle_data*) mymalloc("LocalParticleData", PartManager->NumPart * sizeof(LP[0]));

    /* Watchout : Peano/Morton ordering is required by the tree
     * building algorithm in local_refine.
     *
     * We can either use a global or a local sorting here; the code will run
     * without crashing.
     *
     * A global sorting is chosen to ensure the local topTrees are really local
     * and the leaves almost disjoint. This makes the merged topTree a more accurate
     * representation of the true cost / load distribution, for merging
     * and secondary refinement are approximated.
     *
     * A local sorting may be faster but makes the tree less accurate due to
     * more likely running into overlapped local topTrees.
     * */

    int Nsample = PartManager->NumPart / policy->SubSampleDistance;

    if(Nsample == 0 && PartManager->NumPart != 0) Nsample = 1;

#pragma omp parallel for
    for(i = 0; i < PartManager->NumPart; i ++)
    {
        LP[i].Key = P[i].Key;
        LP[i].Cost = domain_particle_costfactor(i);
    }

    /* First sort to ensure spatially 'even' subsamples; FIXME: This can probably
     * be omitted in most cases. Usually the particles in memory won't be very far off
     * from a peano order. */
    if(policy->PreSort)
        qsort_openmp(LP, PartManager->NumPart, sizeof(struct local_particle_data), order_by_key);

    /* now subsample */
    for(i = 0; i < Nsample; i ++)
    {
        LP[i].Key = LP[i * policy->SubSampleDistance].Key;
        LP[i].Cost = LP[i * policy->SubSampleDistance].Cost;
    }

    if(policy->UseGlobalSort) {
        mpsort_mpi(LP, Nsample, sizeof(struct local_particle_data), mp_order_by_key, 8, NULL, policy->GlobalSortComm);
    } else {
        qsort_openmp(LP, Nsample, sizeof(struct local_particle_data), order_by_key);
    }

    walltime_measure("/Domain/DetermineTopTree/Sort");

    *topTreeSize = 1;
    topTree[0].Daughter = -1;
    topTree[0].Parent = -1;
    topTree[0].Shift = BITS_PER_DIMENSION * 3;
    topTree[0].StartKey = 0;
    topTree[0].PIndex = 0;
    topTree[0].Count = 0;
    topTree[0].Cost = 0;

    /* The tree building algorithm here requires the LP structure to
     * be sorted by key, in which case we either refine
     * the node the last particle lives in, or jump into a new empty node,
     * as the particles are scanned through.
     *
     * I unfortunately cannot find the direct reference of this with
     * the proof. I found it around 2012~2013 when writing psphray2
     * and didn't cite it properly then!
     *
     * Here is a blog link that is related:
     *
     * https://devblogs.nvidia.com/parallelforall/thinking-parallel-part-iii-tree-construction-gpu/
     *
     * A few interesting papers that may make this faster and cheaper.
     *
     * http://sc07.supercomputing.org/schedule/pdf/pap117.pdf
     *
     * */

    /* 1. create a skeleton of the toptree.
     * We do not use all particles to avoid excessive memory consumption.
     *
     * */


    /* During the loop, topTree[?].Count is a flag to indicate if
     * the node has been finished. We either refine the last leaf node
     * or create a new leaf because of the peano/morton sorting.
     * */
    peano_t last_key = -1;
    int last_leaf = -1;
    i = 0;
    while(i < Nsample) {

        int leaf = domain_toptree_get_subnode(topTree, LP[i].Key);

        if (leaf == last_leaf && topTree[leaf].Shift >= 3) {
            /* two particles in a node? need refinement if possible. */
            if(0 != domain_toptree_split(topTree, topTreeSize, leaf)) {
                /* out of memory, retry */
                myfree(LP);
                return 1;
            }
            /* pop the last particle and reinsert it */
            topTree[leaf].Count = 0;

            last_leaf = domain_toptree_insert(topTree, last_key, 0);
            /* retry the current particle. */
            continue;
        } else {
            if(topTree[leaf].Count != 0) {
                /* meeting a node that's already been finished ?
                 * This shall not happen when key is already sorted;
                 * due to the sorting.
                 */
                endrun(10, "Failed to build the toptree\n");
            }
            /* this will create a new node. */
            last_key = LP[i].Key;
            last_leaf = domain_toptree_insert(topTree, last_key, 0);
            i += policy->SubSampleDistance;
            continue;
        }
    }
    /* Alternativly we could have kept last_cost in the above loop and avoid
     * the second and third step: */

    /* 2. Remove the skelton particles from the tree to make it a skeleton;
     * note that we never bothered to add Cost when the skeleton was built.
     * otherwise we shall clean it here too.*/
    for(i = 0; i < *topTreeSize; i ++ ) {
        topTree[i].Count = 0;
    }

    /* 3. insert all particles to the skeleton tree; Count will be correct because we cleaned them.*/
    for(i = 0; i < Nsample; i ++ ) {
        domain_toptree_insert(topTree, LP[i].Key, LP[i].Cost);
    }

    myfree(LP);

    /* then compute the costs of the internal nodes. */
    domain_toptree_update_cost(topTree, 0);

    /* we leave truncation in another function, for costlimit and countlimit must be
     * used in secondary refinement*/
    return 0;
}

/* Combine the toptree. Returns a (collective) error code which is non-zero if an error occured*/
int
domain_nonrecursively_combine_topTree(struct local_topnode_data * topTree, int * topTreeSize, MPI_Comm DomainComm)
{
    /*
     * combine topTree non recursively, this uses MPI_Bcast within a group.
     * it shall be quite a bit faster (~ x2) than the old recursive scheme.
     *
     * it takes less time at higher sep.
     *
     * The communication should have been done with MPI Inter communicator.
     * but I couldn't figure out how to do it that way.
     * */
    int sep = 1;
    MPI_Datatype MPI_TYPE_TOPNODE;
    MPI_Type_contiguous(sizeof(struct local_topnode_data), MPI_BYTE, &MPI_TYPE_TOPNODE);
    MPI_Type_commit(&MPI_TYPE_TOPNODE);
    int errorflag = 0;
    int errorflagall = 0;
    /*Number of tasks to decompose to*/
    int NTask;
    MPI_Comm_size(DomainComm, &NTask);
    /*Which task is this?*/
    int ThisTask;
    MPI_Comm_rank(DomainComm, &ThisTask);

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
                        DomainComm, MPI_STATUS_IGNORE);
                topTree_import = (struct local_topnode_data *) mymalloc("topTree_import",
                            (ntopnodes_import > *topTreeSize ? ntopnodes_import : *topTreeSize) * sizeof(struct local_topnode_data));

                MPI_Recv(
                        topTree_import,
                        ntopnodes_import, MPI_TYPE_TOPNODE,
                        recvTask, TAG_GRAV_B,
                        DomainComm, MPI_STATUS_IGNORE);


                if((*topTreeSize + ntopnodes_import) > MaxTopNodes) {
                    errorflag = 1;
                } else {
                    if(ntopnodes_import < 0) {
                        endrun(1, "severe domain error using a unintended rank \n");
                    }
                    if(ntopnodes_import > 0 ) {
                        domain_toptree_merge(topTree, topTree_import, 0, 0, topTreeSize);
                    }
                }
                myfree(topTree_import);
            }
        } else {
            /* odd guys send */
            recvTask = ThisTask - sep;
            if(recvTask >= 0) {
                MPI_Send(topTreeSize, 1, MPI_INT, recvTask, TAG_GRAV_A,
                        DomainComm);
                MPI_Send(topTree,
                        *topTreeSize, MPI_TYPE_TOPNODE,
                        recvTask, TAG_GRAV_B,
                        DomainComm);
            }
            *topTreeSize = -1;
        }

loop_continue:
        MPI_Allreduce(&errorflag, &errorflagall, 1, MPI_INT, MPI_LOR, DomainComm);
        if(errorflagall) {
            break;
        }
    }

    MPI_Bcast(topTreeSize, 1, MPI_INT, 0, DomainComm);
    MPI_Bcast(topTree, *topTreeSize, MPI_TYPE_TOPNODE, 0, DomainComm);
    MPI_Type_free(&MPI_TYPE_TOPNODE);
    return errorflagall;
}

/*! This function constructs the global top-level tree node that is used
 *  for the domain decomposition. This is done by considering the string of
 *  Peano-Hilbert keys for all particles, which is recursively chopped off
 *  in pieces of eight segments until each segment holds at most a certain
 *  number of particles.
 */
int domain_determine_global_toptree(DomainDecompositionPolicy * policy,
        struct local_topnode_data * topTree, int * topTreeSize, MPI_Comm DomainComm)
{
    walltime_measure("/Domain/DetermineTopTree/Misc");

    /*
     * Build local refinement with a subsample of particles
     * 1/16 is used because each local topTree node takes about 32 bytes.
     **/

    int local_refine_failed = MPIU_Any(
                0 != domain_check_for_local_refine_subsample(policy, topTree, topTreeSize),
                        DomainComm);

    walltime_measure("/Domain/DetermineTopTree/LocalRefine/Init");

    if(local_refine_failed) {
        message(0, "We are out of Topnodes. \n");
        return 1;
    }

    int64_t TotCost = 0, TotCount = 0;
    int64_t costlimit, countlimit;

    MPI_Allreduce(&topTree[0].Cost, &TotCost, 1, MPI_INT64, MPI_SUM, DomainComm);
    MPI_Allreduce(&topTree[0].Count, &TotCount, 1, MPI_INT64, MPI_SUM, DomainComm);

    costlimit = TotCost / (policy->NTopLeaves);
    countlimit = TotCount / (policy->NTopLeaves);

    domain_toptree_truncate(topTree, topTreeSize, countlimit, costlimit);

    walltime_measure("/Domain/DetermineTopTree/LocalRefine/truncate");

    if(*topTreeSize > 4 * policy->NTopLeaves) {
        message(1, "local TopTree Size =%d >> expected = %d; Usually this indicates very bad imbalance, due to a giant density peak.\n",
            *topTreeSize, 4 * policy->NTopLeaves);
    }

#if 0
    char buf[1000];
    sprintf(buf, "topnodes.bin.%d", ThisTask);
    FILE * fd = fopen(buf, "w");

    /* these PIndex are non-essential in other modules, so we reset them */
    for(i = 0; i < *topTreeSize; i ++) {
        topTree[i].PIndex = -1;
    }
    fwrite(topTree, sizeof(struct local_topnode_data), *topTreeSize, fd);
    fclose(fd);

    //MPI_Barrier(DomainComm);
    //MPI_Abort(DomainComm, 0);
#endif

    /* we now need to exchange tree parts and combine them as needed */
    int combine_failed = domain_nonrecursively_combine_topTree(topTree, topTreeSize, DomainComm);

    walltime_measure("/Domain/DetermineTopTree/Combine");

    if(combine_failed)
    {
        message(0, "can't combine trees due to lack of storage. Will try again.\n");
        return 1;
    }

    /* now let's see whether we should still more refinements, based on the estimated cumulative cost/count in each cell */

    int global_refine_failed = MPIU_Any(0 != domain_global_refine(topTree, topTreeSize, countlimit, costlimit), DomainComm);

    walltime_measure("/Domain/DetermineTopTree/Addnodes");

    if(global_refine_failed)
        return 1;

    message(0, "Final local topTree size = %d per segment = %g.\n", *topTreeSize, 1.0 * (*topTreeSize) / (policy->NTopLeaves));

    return 0;
}

static int
domain_global_refine(
    struct local_topnode_data * topTree, int * topTreeSize,
    int64_t countlimit, int64_t costlimit)
{
    int i;

    /* At this point we have refined the local particle tree so that each
     * topNode contains a Cost and Count below the cost threshold. We have then
     * done a global merge of the particle tree. Some of our topTree may now contain
     * more particles than the Cost threshold, but repeating refinement using the local
     * algorithm is complicated - particles inside any particular topNode may be
     * on another processor. So we do a local volume based refinement here. This
     * just cuts each topNode above the threshold into 8 equal-sized portions by
     * subdividing the peano key.
     *
     * NOTE: Just like the merge, this does not correctly preserve costs!
     * Costs are just divided by 8, because recomputing them for the daughter nodes
     * will be expensive.
     * In practice this seems to work fine, probably because the cost distribution
     * is not that unbalanced. */

    message(0, "local topTree size before appending=%d\n", *topTreeSize);

    /*Note that *topTreeSize will change inside the loop*/
    for(i = 0; i < *topTreeSize; i++)
    {
        /*If this node has no children and non-zero size*/
        if(topTree[i].Daughter >= 0 || topTree[i].Shift <= 0) continue;

        /*If this node is also more costly than the limit*/
        if(topTree[i].Count < countlimit && topTree[i].Cost < costlimit) continue;

        /*If we have no space for another 8 topTree, exit */
        if((*topTreeSize + 8) > MaxTopNodes) {
            return 1;
        }

        topTree[i].Daughter = *topTreeSize;
        int j;
        for(j = 0; j < 8; j++)
        {
            int sub = topTree[i].Daughter + j;
            topTree[sub].Shift = topTree[i].Shift - 3;
            topTree[sub].Count = topTree[i].Count / 8;
            topTree[sub].Cost = topTree[i].Cost / 8;
            topTree[sub].Daughter = -1;
            topTree[sub].Parent = i;
            topTree[sub].StartKey = topTree[i].StartKey + j * (1L << topTree[sub].Shift);
        }
        (*topTreeSize) += 8;
    }
    return 0;
}


static void
domain_compute_costs(const DomainDecomp * ddecomp, int64_t *TopLeafWork, int64_t *TopLeafCount)
{
    int i;
    int NumThreads = omp_get_max_threads();
    int64_t * local_TopLeafWork = (int64_t *) mymalloc("local_TopLeafWork", NumThreads * ddecomp->NTopLeaves * sizeof(local_TopLeafWork[0]));
    int64_t * local_TopLeafCount = (int64_t *) mymalloc("local_TopLeafCount", NumThreads * ddecomp->NTopLeaves * sizeof(local_TopLeafCount[0]));

    memset(local_TopLeafWork, 0, NumThreads * ddecomp->NTopLeaves * sizeof(local_TopLeafWork[0]));
    memset(local_TopLeafCount, 0, NumThreads * ddecomp->NTopLeaves * sizeof(local_TopLeafCount[0]));

#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int n;

        int64_t * mylocal_TopLeafWork = local_TopLeafWork + tid * ddecomp->NTopLeaves;
        int64_t * mylocal_TopLeafCount = local_TopLeafCount + tid * ddecomp->NTopLeaves;

        #pragma omp for
        for(n = 0; n < PartManager->NumPart; n++)
        {
            /* Skip garbage particles: they have zero work
             * and can be removed by exchange if under memory pressure.*/
            if(P[n].IsGarbage)
                continue;
            int no = domain_get_topleaf(P[n].Key, ddecomp);

            mylocal_TopLeafWork[no] += domain_particle_costfactor(n);

            mylocal_TopLeafCount[no] += 1;
        }
    }

#pragma omp parallel for
    for(i = 0; i < ddecomp->NTopLeaves; i++)
    {
        int tid;
        for(tid = 1; tid < NumThreads; tid++) {
            local_TopLeafWork[i] += local_TopLeafWork[i + tid * ddecomp->NTopLeaves];
            local_TopLeafCount[i] += local_TopLeafCount[i + tid * ddecomp->NTopLeaves];
        }
    }

    MPI_Allreduce(local_TopLeafWork, TopLeafWork, ddecomp->NTopLeaves, MPI_INT64, MPI_SUM, ddecomp->DomainComm);
    MPI_Allreduce(local_TopLeafCount, TopLeafCount, ddecomp->NTopLeaves, MPI_INT64, MPI_SUM, ddecomp->DomainComm);
    myfree(local_TopLeafCount);
    myfree(local_TopLeafWork);
}

/**
 * Merge treeB into treeA.
 *
 * The function recursively merge the cost and refinement of treeB into treeA.
 *
 * At the initial call, noA and noB must both be the root node.
 *
 * When the structure is mismatched, e.g. a leaf in A meets a branch in B or
 * a leaf in B meets a branch in A, the cost of the leaf is splitted evenly.
 * This is only an approximation then the leaf is not empty.
 *
 * We therefore have an incentive to minimize overlaps between treeB and treeA.
 * local_refinement does a global sorting of keys to help that.
 *
 * */

static void
domain_toptree_merge(struct local_topnode_data *treeA,
                     struct local_topnode_data *treeB,
                     int noA, int noB, int * treeASize)
{
    int j, sub;
    int64_t count;
    int64_t cost;

    if(treeB[noB].Shift < treeA[noA].Shift)
    {
        /* Add B to A */
        /* Create a daughter to a, since we will merge B to A's daughter*/
        if(treeA[noA].Daughter < 0)
        {
            if((*treeASize + 8) >= MaxTopNodes) {
                endrun(88, "Too many Topnodes; this shall not happen because we ensure there is enough and bailed earlier than this\n");
            }
            /* noB must have a parent if we are here, since noB is lower than noA;
             * noA must have already merged in the cost of noB's parent.
             * This is the first time we create these children in A, thus,
             * We shall evenly divide the non-B part of the cost in these children;
             * */

            count = treeA[noA].Count - treeB[treeB[noB].Parent].Count;
            cost = treeA[noA].Cost - treeB[treeB[noB].Parent].Cost;

            treeA[noA].Daughter = *treeASize;
            for(j = 0; j < 8; j++)
            {

                sub = treeA[noA].Daughter + j;
                treeA[sub].Shift = treeA[noA].Shift - 3;
                treeA[sub].Count = (j + 1) * count / 8 - j * count / 8;
                treeA[sub].Cost  = (j + 1) * cost / 8 - j * cost / 8;
                treeA[sub].Daughter = -1;
                treeA[sub].Parent = noA;
                treeA[sub].StartKey = treeA[noA].StartKey + j * (1L << treeA[sub].Shift);
            }
            (*treeASize) += 8;
        }

        /* find the sub node in A for me and merge, this would bring noB and sub on the same shift, drop to next case */
        sub = treeA[noA].Daughter + ((treeB[noB].StartKey - treeA[noA].StartKey) >> (treeA[noA].Shift - 3));
        domain_toptree_merge(treeA, treeB, sub, noB, treeASize);
    }
    else if(treeB[noB].Shift == treeA[noA].Shift)
    {
        treeA[noA].Count += treeB[noB].Count;
        treeA[noA].Cost += treeB[noB].Cost;

        /* Prefer to go down B; this would trigger the previous case  */
        if(treeB[noB].Daughter >= 0)
        {
            for(j = 0; j < 8; j++)
            {
                sub = treeB[noB].Daughter + j;
                if(treeB[sub].Shift >= treeB[noB].Shift) {
                    endrun(1, "Child node %d has shift %d, parent %d has shift %d. treeB is corrupt. \n",
                        sub, treeB[sub].Shift, noB, treeB[noB].Shift);
                }
                domain_toptree_merge(treeA, treeB, noA, sub, treeASize);
            }
        }
        else
        {
            /* We can't divide by B so we do it for A, this may trigger the next branch, since
             * we are lowering A */
            if(treeA[noA].Daughter >= 0) {
                for(j = 0; j < 8; j++) {
                    sub = treeA[noA].Daughter + j;
                    domain_toptree_merge(treeA, treeB, sub, noB, treeASize);
                }
            }
        }
    }
    else if(treeB[noB].Shift > treeA[noA].Shift)
    {
        /* Since we only know how to split A, here we simply add a spatial average to A */
        uint64_t n = 1L << (treeB[noB].Shift - treeA[noA].Shift);

        if(treeB[noB].Shift - treeA[noA].Shift > 60) {
            message(1, "Warning: Refusing to merge two tree nodes of wildly different depth: %d %d;\n ", treeB[noB].Shift, treeA[noA].Shift);
            n = 0;
        }

        count = treeB[noB].Count;
        cost = treeB[noB].Cost;

        if (n > 0) {
            /* this is no longer conserving total cost but it should be fine .. */
            treeA[noA].Count += count / n;
            treeA[noA].Cost += cost / n;

            // message(1, "adding cost to %d %td, %td\n", noA, count / n, cost / n);
            if(treeA[noA].Daughter >= 0) {
                for(j = 0; j < 8; j++) {
                    sub = treeA[noA].Daughter + j;
                    domain_toptree_merge(treeA, treeB, sub, noB, treeASize);
                }
            }
        }
    }
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
