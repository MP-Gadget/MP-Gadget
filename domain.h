#ifndef DOMAIN_H
#define DOMAIN_H

#include "peano.h"

/*Several of these variables are used externally only in restart.c*/
extern double DomainCorner[3], DomainCenter[3], DomainLen, DomainFac;
extern int *DomainStartList, *DomainEndList;
extern int *DomainTask;

extern struct topnode_data
{
    peanokey Size;
    peanokey StartKey;
    int64_t Count;
    MyFloat GravCost;
    int Daughter;
    int Pstart;
    int Blocks;
    int Leaf;
} *TopNodes;

extern int NTopnodes, NTopleaves;

void domain_exchange(int (*layoutfunc)(int p));
void domain_refresh_totals();
int domain_garbage_collection(void);
void domain_test_id_uniqueness();
void domain_Decomposition(void);
int domain_fork_particle();
#endif
