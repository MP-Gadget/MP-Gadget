#include <omp.h>
#include <stdlib.h>
#include <stdio.h>
#include <stddef.h>
#include <string.h>
#include <stdint.h>
#include "mymalloc.h"
/* Below is a merge-sort routine copied directly from glibc version 2.26
 * (although the code is the same since Dec. 2010, glibc 2.13).
 * The copy is so that we can control our memory allocation
 * to use mymalloc instead of the system malloc.
 * It also means that on machines where libc is not glibc we will
 * use glibc's routine, which is almost certainly better.*/

/*============================================*/
/* An alternative to qsort, with an identical interface.
   This file is part of the GNU C Library.
   Copyright (C) 1992-2017 Free Software Foundation, Inc.
   Written by Mike Haertel, September 1988.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, see
   <http://www.gnu.org/licenses/>.  */

/*Define for convenience*/
typedef int(*__compar_fn_t)(const void *, const void *);

struct msort_param
{
  size_t s;
  size_t var;
  __compar_fn_t cmp;
  void *arg;
  char *t;
};
static void msort_with_tmp (const struct msort_param *p, void *b, size_t n);

static void
msort_with_tmp (const struct msort_param *p, void *b, size_t n)
{
  char *b1, *b2;
  size_t n1, n2;

  if (n <= 1)
    return;

  n1 = n / 2;
  n2 = n - n1;
  b1 = (char *) b;
  b2 = (char *) b + (n1 * p->s);

  msort_with_tmp (p, b1, n1);
  msort_with_tmp (p, b2, n2);

  char *tmp = p->t;
  const size_t s = p->s;
  __compar_fn_t cmp = p->cmp;
  switch (p->var)
    {
    case 0:
      while (n1 > 0 && n2 > 0)
        {
          if ((*cmp) (b1, b2) <= 0)
            {
              *(uint32_t *) tmp = *(uint32_t *) b1;
              b1 += sizeof (uint32_t);
              --n1;
            }
          else
            {
              *(uint32_t *) tmp = *(uint32_t *) b2;
              b2 += sizeof (uint32_t);
              --n2;
            }
          tmp += sizeof (uint32_t);
        }
      break;
    case 1:
      while (n1 > 0 && n2 > 0)
        {
          if ((*cmp) (b1, b2) <= 0)
            {
              *(uint64_t *) tmp = *(uint64_t *) b1;
              b1 += sizeof (uint64_t);
              --n1;
            }
          else
            {
              *(uint64_t *) tmp = *(uint64_t *) b2;
              b2 += sizeof (uint64_t);
              --n2;
            }
          tmp += sizeof (uint64_t);
        }
      break;
    case 2:
      while (n1 > 0 && n2 > 0)
        {
          unsigned long *tmpl = (unsigned long *) tmp;
          unsigned long *bl;

          tmp += s;
          if ((*cmp) (b1, b2) <= 0)
            {
              bl = (unsigned long *) b1;
              b1 += s;
              --n1;
            }
          else
            {
              bl = (unsigned long *) b2;
              b2 += s;
              --n2;
            }
          while (tmpl < (unsigned long *) tmp)
            *tmpl++ = *bl++;
        }
      break;
    case 3:
      while (n1 > 0 && n2 > 0)
        {
          if ((*cmp) (*(const void **) b1, *(const void **) b2) <= 0)
            {
              *(void **) tmp = *(void **) b1;
              b1 += sizeof (void *);
              --n1;
            }
          else
            {
              *(void **) tmp = *(void **) b2;
              b2 += sizeof (void *);
              --n2;
            }
          tmp += sizeof (void *);
        }
      break;
    default:
      while (n1 > 0 && n2 > 0)
        {
          if ((*cmp) (b1, b2) <= 0)
            {
              tmp = (char *) memcpy (tmp, b1, s) + s;
              b1 += s;
              --n1;
            }
          else
            {
              tmp = (char *) memcpy (tmp, b2, s) + s;
              b2 += s;
              --n2;
            }
        }
      break;
    }

  if (n1 > 0)
    memcpy (tmp, b1, n1 * s);
  memcpy (b, p->t, (n - n2) * s);
}

/*End code copied from glibc*/
/*=====================================================*/

static void merge(void * base1, size_t nmemb1, void * base2, size_t nmemb2, void * output, size_t size,
         int(*compar)(const void *, const void *), int indirect) {
    char * p1 = (char *) base1;
    char * p2 = (char *) base2;
    char * po = (char *) output;
    char * s1 = p1 + nmemb1 * size, *s2 = p2 + nmemb2 * size;
    while(p1 < s1 && p2 < s2) {
        int cmp;
        if(indirect) {
            cmp = compar(*(void **)p1, *(void **)p2);
        }
        else {
            cmp = compar(p1, p2);
        }
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
    ptrdiff_t * Anmemb = ta_malloc("Anmemb", ptrdiff_t, Nt);
    ptrdiff_t * Anmemb_old = ta_malloc("Anmemb_old", ptrdiff_t, Nt);

    void ** Abase_store = ta_malloc("Abase", void *, Nt);
    void ** Atmp_store = ta_malloc("Atmp", void *, Nt);
    void ** Abase = Abase_store;
    void ** Atmp = Atmp_store;

    /*Should I use indirect sorting?*/
    int indirect = 0;
    if(size > 32)
        indirect = 1;
    void * tmp;
    /*NOTE: if this allocation becomes a problem,
     * switch to glibc's quicksort (serial!) */
    if(indirect)
        tmp = mymalloc("qsort",2*nmemb*sizeof(void *) + size);
    else
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

        struct msort_param p;
        p.t = ((char **) Atmp)[tid];
        p.s = size;
        p.var = 4;
        p.cmp = compar;
        /*For large arrays we use an indirect sort following glibc*/
        if(indirect)
        {
            /* Indirect sorting: copy everything in this thread to the new pointer space,
             * which is after the tmp space. */
            char *ip = (char *) Abase[tid];
            void **tp = (void **) ((char *)tmp + (nmemb + start)* sizeof (void *));
            void **t = tp;
            void *end = (void *) (tp + Anmemb[tid]);

            while ((void *) t < end)
            {
              *t++ = ip;
              ip += size;
            }
            p.s = sizeof (void *);
            p.var = 3;
            Abase[tid] = tp;
            Atmp[tid] = ((char*) tmp) + start * p.s;
            p.t = ((char **)Atmp)[tid];
        }
        else {
            /*Copied from glibc*/
            if ((size & (sizeof (uint32_t) - 1)) == 0
                && ((char *) base - (char *) 0) % __alignof__ (uint32_t) == 0)
              {
                if (size == sizeof (uint32_t))
                  p.var = 0;
                else if (size == sizeof (uint64_t)
                         && ((char *) base - (char *) 0) % __alignof__ (uint64_t) == 0)
                  p.var = 1;
                else if ((size & (sizeof (unsigned long) - 1)) == 0
                         && ((char *) base - (char *) 0)
                            % __alignof__ (unsigned long) == 0)
                  p.var = 2;
              }
            /*End copied from glibc*/
        }
        msort_with_tmp (&p, Abase[tid], Anmemb[tid]);
        /* now each sub array is sorted, kick start the merging */
        int sep;
        for (sep = 1; sep < Nt; sep *=2 ) {
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
#pragma omp critical
		    {
                /* only even leaders arrives to this point*/
                if(nextT >= Nt) {
                    /* no next guy, copy directly.*/
                    merge(Abase[tid], Anmemb[tid], NULL, 0, Atmp[tid], p.s, compar, indirect);
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
                    merge(Abase[tid], Anmemb[tid], Abase[nextT], Anmemb[nextT], Atmp[tid], p.s, compar, indirect);
                    /* merge two lists */
                    Anmemb[tid] += Anmemb[nextT];
                    Anmemb[nextT] = 0;
		    }
                }
            }

            /* now swap Abase and Atmp for next merge */
#pragma omp barrier
#pragma omp master
	    {
                void ** a = Abase;
                Abase = Atmp;
                Atmp = a;
	    }
#pragma omp barrier
            /* at this point Abase contains the sorted array */
        }
        /* output was written to the tmp rather than desired location, copy it */
        if((!indirect && Abase[0] != base)
                || (indirect && Abase[0] != (char *) tmp + nmemb * sizeof(void *))) {
            memmove(Atmp[tid], Abase[tid], Anmemb_old[tid] * p.s);
        }
    }

    ta_free(Atmp_store);
    ta_free(Abase_store);
    ta_free(Anmemb_old);
    ta_free(Anmemb);

    /*Copied from glibc */
    /* tp[0] .. tp[n - 1] is now sorted, copy around entries of
       the original array (done serially).  Knuth vol. 3 (2nd ed.) exercise 5.2-10.  */
    if(indirect) {
        char *kp, *ip;
        size_t i;
        void **tp = (void **) ((char *) tmp + nmemb * sizeof (void *));
        void *tmp_storage = (void *) (tp + nmemb);
        for (i = 0, ip = (char *) base; i < nmemb; i++, ip += size)
          if ((kp = ((char **)tp)[i]) != ip)
            {
              size_t j = i;
              char *jp = ip;
              memcpy (tmp_storage, ip, size);

              do {
                size_t k = (kp - (char *) base) / size;
                tp[j] = jp;
                memcpy (jp, kp, size);
                j = k;
                jp = kp;
                kp = ((char **)tp)[k];
              } while (kp != ip);

              tp[j] = jp;
              memcpy (jp, tmp_storage, size);
            }
    }
    myfree(tmp);
}
