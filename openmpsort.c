#ifdef _OPENMP
#include <omp.h>
#endif
#include <stdlib.h>
#include <stdio.h>
#include <stddef.h>
#include <string.h>
#include "mymalloc.h"

static void merge(void * base1, size_t nmemb1, void * base2, size_t nmemb2, void * output, size_t size,
         int(*compar)(const void *, const void *)) {
    char * p1 = base1;
    char * p2 = base2;
    char * po = output;
    char * s1 = p1 + nmemb1 * size, *s2 = p2 + nmemb2 * size;
    while(p1 < s1 && p2 < s2) {
        int cmp = compar(p1, p2 );
        if(cmp <= 0) {
            memcpy(po, p1, size);
            p1 += size;
        } else {
            memcpy(po, p2, size);
            p2 += size;
        }
        po += size;
    }
    if(p1 < s1) {
        memcpy(po, p1, s1 - p1);
    }
    if(p2 < s2) {
        memcpy(po, p2, s2 - p2);
    }
}

void qsort_openmp(void *base, size_t nmemb, size_t size,
                         int(*compar)(const void *, const void *)) {
    int Nt = omp_get_max_threads();
    ptrdiff_t Anmemb[Nt];
    ptrdiff_t Anmemb_old[Nt];
    void * Abase_store[Nt];
    void * Atmp_store[Nt];
    void ** Abase = Abase_store;
    void ** Atmp = Atmp_store;

    void * tmp;
    tmp = mymalloc("qsort", size * nmemb);

#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        /* actual number of threads*/
        int Nt = omp_get_num_threads();

        /* domain decomposition */
        ptrdiff_t start = tid * nmemb / Nt;
        ptrdiff_t end = (tid + 1) * nmemb / Nt;
        /* save decoposition to global array for merging */
        Anmemb[tid] = end - start;
        Anmemb_old[tid] = end - start;
        Abase[tid] = ((char*) base) + start * size;
        Atmp[tid] = ((char*) tmp) + start * size;

        qsort( Abase[tid], Anmemb[tid], size, compar);

        /* now each sub array is sorted, kick start the merging */

        int sep;
        for (sep = 1; sep < Nt; sep *=2 ) {
#pragma omp barrier
            int color = tid / sep;
            int key = tid % sep;
#if 0
            if(tid == 0) {
                printf("sep = %d Abase[0] %p base %p, Atmp[0] %p tmp %p\n", 
                        sep, Abase[0], base, Atmp[0], tmp);
                printf("base 73134 = %d 73135 = %d\n", ((int*) base)[73134], ((int*) base)[73135]);
                printf("tmp  73134 = %d 73135 = %d\n", ((int*) tmp )[73134], ((int*) tmp )[73135]);
            }
#endif
#pragma omp barrier
            /* only group leaders work */
            if(key == 0 && color % 2 == 0) {
                int nextT = tid + sep;
                /*merge with next guy */
                /* only even leaders arrives to this point*/
                if(nextT >= Nt) {
                    /* no next guy, copy directly.*/
                    merge(Abase[tid], Anmemb[tid], NULL, 0, Atmp[tid], size, compar);
                }  else {
#if 0                    
                    printf("%d + %d merging with %td/%td:%td %td/%td:%td\n", tid, nextT,
                            ((char*)Abase[tid] - (char*) base) / size,
                            ((char*)Abase[tid] - (char*) tmp) / size,
                            Anmemb[tid], 
                            ((char*)Abase[nextT] - (char*) base) / size,
                            ((char*)Abase[nextT] - (char*) tmp) / size,
                            Anmemb[nextT]);
#endif
                    merge(Abase[tid], Anmemb[tid], Abase[nextT], Anmemb[nextT], Atmp[tid], size, compar);
                    /* merge two lists */
                    Anmemb[tid] = Anmemb[tid] + Anmemb[nextT];
                    Anmemb[nextT] = 0;
                }
            }

            /* now swap Abase and Atmp for next merge */
#pragma omp barrier
            if(tid == 0) {
                void ** a = Abase;
                Abase = Atmp;
                Atmp = a;
            }
            /* at this point Abase contains the sorted array */
        }

#pragma omp barrier
        /* output was written to the tmp rather than desired location, copy it */
        if(Abase[0] != base) {
            memcpy(Atmp[tid], Abase[tid], Anmemb_old[tid] * size);
        }
    }
    myfree(tmp);
}
