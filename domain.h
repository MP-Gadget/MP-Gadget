#ifndef DOMAIN_H
#define DOMAIN_H

#include "peano.h"

/*These variables are used externally in forcetree.c.
 * DomainTask is also used in treewalk and NTopleaves is used in gravpm.c*/
extern int *DomainStartList, *DomainEndList;
extern int *DomainTask;

extern struct topnode_data
{
    peanokey Size;		/*!< number of Peano-Hilbert mesh-cells represented by top-level node */
    peanokey StartKey;		/*!< first Peano-Hilbert key in top-level node */
    int Daughter;			/*!< index of first daughter cell (out of 8) of top-level node */
    int Leaf;			/*!< if the node is a leaf, this gives its number when all leaves are traversed in Peano-Hilbert order */
} *TopNodes;
extern int MaxTopNodes;	        /*!< Maximum number of nodes in the top-level tree used for domain decomposition */


extern int NTopnodes, NTopleaves;

void domain_test_id_uniqueness();
void domain_decompose_full(void);
void domain_maintain(void);
#endif
