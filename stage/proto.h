#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

void subfind_distlinklist_set_all(long long index, long long head, long long tail, int len, long long next);
void subfind_distlinklist_get_tailandlen(long long index, long long *tail, int *len);
void subfind_distlinklist_set_tailandlen(long long index, long long tail, int len);
int subfind_distlinklist_get_tail_set_tail_increaselen(long long index, long long *tail, long long newtail);
void subfind_distlinklist_set_headandnext(long long index, long long head, long long next);
void subfind_distlinklist_get_two_heads(long long ngb_index1, long long ngb_index2,
					long long *head, long long *head_attach);
long long subfind_distlinklist_get_head(long long index);
long long subfind_distlinklist_get_rank(long long index);
long long subfind_distlinklist_get_next(long long index);
void subfind_distlinklist_set_next(long long index, long long next);
long long subfind_distlinklist_set_head_get_next(long long index, long long head);
long long subfind_distlinklist_setrank_and_get_next(long long index, long long *rank);


void subfind_col_find_candidates(int totgrouplen);

void write_candidates(int num);
void load_sd(int num);


void *mymalloc(size_t n);

void myfree(void *p);


void read_parameter_file(char *fname);

size_t my_fread(void *ptr, size_t size, size_t nmemb, FILE * stream);
size_t my_fwrite(void *ptr, size_t size, size_t nmemb, FILE * stream);

void endrun(int nr);
