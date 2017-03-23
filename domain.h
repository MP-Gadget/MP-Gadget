#ifndef DOMAIN_H
#define DOMAIN_H

#include "peano.h"

/*Several of these variables are used externally only in restart.c*/
extern double DomainCorner[3], DomainCenter[3], DomainLen, DomainFac;
extern int *DomainStartList, *DomainEndList;
extern int *DomainTask;
extern int *DomainNodeIndex;
extern int *DomainList, DomainNumChanged;

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


int domain_fork_particle();
void domain_findSplit_work_balanced(int ncpu, int ndomain);
void domain_findSplit_load_balanced(int ncpu, int ndomain);
int domain_sort_loadorigin(const void *a, const void *b);
int domain_sort_segments(const void *a, const void *b);
void domain_assign_load_or_work_balanced(int mode);
void domain_allocate(void);
void domain_Decomposition(void);
int domain_check_memory_bound(const int print_details);
int domain_compare_key(const void *a, const void *b);
int domain_compare_key(const void *a, const void *b);
int domain_compare_toplist(const void *a, const void *b);
double domain_particle_costfactor(int i);
int domain_decompose(void);
int domain_determineTopTree(void);
void domain_exchange(int (*layoutfunc)(int p));
void domain_findExchangeNumbers(int task, int partner, int sphflag, int *send, int *recv);
void domain_findExtent(void);
void domain_findSplit(int cpustart, int ncpu, int first, int last);
void domain_findSplit_balanced(int cpustart, int ncpu, int first, int last);
void domain_free(void);
void domain_shiftSplit(void);
void domain_sumCost(void);
int domain_topsplit(int node, peanokey startkey);
int domain_topsplit_local(int node, peanokey startkey, int mode);
int domain_topsplit_special(void);
int domain_compare_key(const void *a, const void *b);
int domain_check_for_local_refine(const int i, const double countlimit, const double costlimit, const struct peano_hilbert_data * mp);
void domain_free_trick(void);
void domain_allocate_trick(void);
int domain_recursively_combine_topTree(int start, int ncpu);
void domain_walktoptree(int no);
void mysort_domain(void *b, size_t n, size_t s);

int domain_garbage_collection(void);

void domain_test_id_uniqueness();
void domain_refresh_totals();


#endif
