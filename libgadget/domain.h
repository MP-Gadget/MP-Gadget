#ifndef DOMAIN_H
#define DOMAIN_H

#include <mpi.h>
#include "utils/peano.h"
#include "utils/paramset.h"
#include "drift.h"

/*These variables are used externally in forcetree.c.
 * DomainTask is also used in treewalk and NTopLeaves is used in gravpm.c*/

struct topnode_data
{
    peano_t StartKey;		/*!< first Peano-Hilbert key in top-level node */
    int Daughter;			/*!< index of first daughter cell (out of 8) of top-level node */
    int Shift;    /*!< level = log 2 of number of Peano-Hilbert mesh-cells represented by top-level node */
    int Leaf ;			/*!< if the node is a leaf, this gives its index in topleaf_data */
};

struct topleaf_data {
    int Task;
    int topnode; /* used during domain_decompose for balancing the decomposition */
    int treenode; /* used during life span of the tree for looking up in the tree Nodes */
};

struct task_data {
    int StartLeaf;
    int EndLeaf;
};

typedef struct DomainDecomp {
    int domain_allocated_flag;
    /* TopNode and TopLeaf data arrays*/
    struct topnode_data * TopNodes;
    struct topleaf_data * TopLeaves;
    /*Size of the above arrays*/
    int NTopNodes;
    int NTopLeaves;
    struct task_data * Tasks;
    /* MPI Communicator over which to build the Domain.
     * Currently this is always MPI_COMM_WORLD.*/
    MPI_Comm DomainComm;
} DomainDecomp;

/*Parameters of the domain decomposition, set by the input parameter file*/
typedef struct DomainParams
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
} DomainParams;

/*Set the parameters of the domain module*/
void set_domain_params(ParameterSet * ps);
/* Test helper*/
void set_domain_par(DomainParams dp);

/* Do a full domain decomposition, which splits the particles into even clumps*/
void domain_decompose_full(DomainDecomp * ddecomp, MPI_Comm DomainComm);
/* Exchange particles which have moved into the new domains, not re-doing the split unless we have to*/
int domain_maintain(DomainDecomp * ddecomp, struct DriftData * drift);

/** This function determines the TopLeaves entry for the given key.*/
static inline int
domain_get_topleaf(const peano_t key, const DomainDecomp * ddecomp) {
    int no=0;
    while(ddecomp->TopNodes[no].Daughter >= 0)
        no = ddecomp->TopNodes[no].Daughter + ((key - ddecomp->TopNodes[no].StartKey) >> (ddecomp->TopNodes[no].Shift - 3));
    no = ddecomp->TopNodes[no].Leaf;
    return no;
};

void domain_free(DomainDecomp * ddecomp);

#endif
