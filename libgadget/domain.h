#ifndef DOMAIN_H
#define DOMAIN_H

#include "utils/peano.h"

/*These variables are used externally in forcetree.c.
 * DomainTask is also used in treewalk and NTopLeaves is used in gravpm.c*/

struct topnode_data
{
    peano_t StartKey;		/*!< first Peano-Hilbert key in top-level node */
    int Daughter;			/*!< index of first daughter cell (out of 8) of top-level node */
    short int Shift;    /*!< level = log 2 of number of Peano-Hilbert mesh-cells represented by top-level node */
    int Leaf ;			/*!< if the node is a leaf, this gives its index in topleaf_data */
};

struct topleaf_data {
    int Task;
    union {
        int topnode; /* used during domain_decompose for balancing the decomposition */
        int treenode; /* used during life span of the tree for looking up in the tree Nodes */
    };
};

struct task_data {
    int StartLeaf;
    int EndLeaf;
};

typedef struct Domain {
    int domain_allocated_flag;
    /* TopNode and TopLeaf data arrays*/
    struct topnode_data * TopNodes;
    struct topleaf_data * TopLeaves;
    /*Size of the above arrays*/
    int NTopNodes;
    int NTopLeaves;
    struct task_data * Tasks;
} Domain;

void domain_decompose_full(Domain * domain);
void domain_maintain(Domain * domain);

/** This function determines the TopLeaves entry for the given key.*/
static inline int
domain_get_topleaf(const peano_t key, const Domain * domain) {
    int no=0;
    while(domain->TopNodes[no].Daughter >= 0)
        no = domain->TopNodes[no].Daughter + ((key - domain->TopNodes[no].StartKey) >> (domain->TopNodes[no].Shift - 3));
    no = domain->TopNodes[no].Leaf;
    return no;
};

#endif
