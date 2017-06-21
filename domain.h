#ifndef DOMAIN_H
#define DOMAIN_H

#include "peano.h"

/*Several of these variables are used externally only in restart.c*/
extern double DomainCorner[3], DomainCenter[3], DomainLen, DomainFac;
extern int *DomainStartList, *DomainEndList;
extern int *DomainTask;

extern struct topnode_data
{
    peanokey Size;		/*!< number of Peano-Hilbert mesh-cells represented by top-level node */
    peanokey StartKey;		/*!< first Peano-Hilbert key in top-level node */
    int64_t Count;		/*!< counts the number of particles in this top-level node */
    MyFloat GravCost;
    int Daughter;			/*!< index of first daughter cell (out of 8) of top-level node */
    int Leaf;			/*!< if the node is a leaf, this gives its number when all leaves are traversed in Peano-Hilbert order */
} *TopNodes;

extern int NTopnodes, NTopleaves;

void domain_exchange(int (*layoutfunc)(int p));
void domain_refresh_totals();
int domain_garbage_collection(void);
void domain_test_id_uniqueness();
void domain_Decomposition(void);
int domain_fork_particle();
#endif
