#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <unistd.h>
#include <signal.h>
#include <gsl/gsl_rng.h>


#include "allvars.h"
#include "proto.h"

#define TAG_DATAIN  100
#define TAG_DATAOUT 101


static void serial_sort(char *base, size_t nmemb, size_t size, int (*compar) (const void *, const void *));
static void msort_serial_with_tmp(char *base, size_t n, size_t s, int (*compar) (const void *, const void *),
				  char *t);
static void parallel_merge(int master, int ncpu, size_t * nlist, size_t nmax, char *base, size_t nmemb,
			   size_t size, int (*compar) (const void *, const void *));



void parallel_sort(void *base, size_t nmemb, size_t size, int (*compar) (const void *, const void *))
{
  int i, ncpu_in_group, master, groupnr;
  size_t nmax, *nlist;

  serial_sort((char *) base, nmemb, size, compar);

  nlist = (size_t *) mymalloc("nlist", NTask * sizeof(size_t));

  MPI_Allgather(&nmemb, sizeof(size_t), MPI_BYTE, nlist, sizeof(size_t), MPI_BYTE, MPI_COMM_WORLD);

  for(i = 0, nmax = 0; i < NTask; i++)
    if(nlist[i] > nmax)
      nmax = nlist[i];

  for(ncpu_in_group = 2; ncpu_in_group <= (1 << PTask); ncpu_in_group *= 2)
    {
      groupnr = ThisTask / ncpu_in_group;

      master = ncpu_in_group * groupnr;

      parallel_merge(master, ncpu_in_group, nlist, nmax, (char *) base, nmemb, size, compar);
    }

  myfree(nlist);
}

void parallel_merge(int master, int ncpu, size_t * nlist, size_t nmax,
		    char *base, size_t nmemb, size_t size, int (*compar) (const void *, const void *))
{
  size_t na, nb, nr;
  int cpua, cpub, cpur;
  char *list_a, *list_b, *list_r;

  if(master + ncpu / 2 >= NTask)	/* nothing to do */
    return;

  if(ThisTask != master)
    {
      if(nmemb)
	{
	  list_r = (char *) mymalloc("	  list_r", nmemb * size);

	  MPI_Request requests[2];

	  MPI_Isend(base, nmemb * size, MPI_BYTE, master, TAG_DATAIN, MPI_COMM_WORLD, &requests[0]);
	  MPI_Irecv(list_r, nmemb * size, MPI_BYTE, master, TAG_DATAOUT, MPI_COMM_WORLD, &requests[1]);
	  MPI_Waitall(2, requests, MPI_STATUSES_IGNORE);

	  memcpy(base, list_r, nmemb * size);
	  myfree(list_r);
	}
    }
  else
    {
      list_a = (char *) mymalloc("list_a", nmax * size);
      list_b = (char *) mymalloc("list_b", nmax * size);
      list_r = (char *) mymalloc("list_r", nmax * size);

      cpua = master;
      cpub = master + ncpu / 2;
      cpur = master;

      na = 0;
      nb = 0;
      nr = 0;

      memcpy(list_a, base, nmemb * size);
      if(nlist[cpub])
	MPI_Recv(list_b, nlist[cpub] * size, MPI_BYTE, cpub, TAG_DATAIN, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      while(cpur < master + ncpu && cpur < NTask)
	{
	  while(na >= nlist[cpua] && cpua < master + ncpu / 2 - 1)
	    {
	      cpua++;
	      if(nlist[cpua])
		MPI_Recv(list_a, nlist[cpua] * size, MPI_BYTE, cpua, TAG_DATAIN, MPI_COMM_WORLD,
			 MPI_STATUS_IGNORE);
	      na = 0;
	    }
	  while(nb >= nlist[cpub] && cpub < master + ncpu - 1 && cpub < NTask - 1)
	    {
	      cpub++;
	      if(nlist[cpub])
		MPI_Recv(list_b, nlist[cpub] * size, MPI_BYTE, cpub, TAG_DATAIN, MPI_COMM_WORLD,
			 MPI_STATUS_IGNORE);
	      nb = 0;
	    }

	  while(nr >= nlist[cpur])
	    {
	      if(cpur == master)
		memcpy(base, list_r, nr * size);
	      else
		{
		  if(nlist[cpur])
		    MPI_Send(list_r, nlist[cpur] * size, MPI_BYTE, cpur, TAG_DATAOUT, MPI_COMM_WORLD);
		}
	      nr = 0;
	      cpur++;
	      if(cpur >= master + ncpu)
		break;
	    }

	  if(na < nlist[cpua] && nb < nlist[cpub])
	    {
	      if(compar(list_a + na * size, list_b + nb * size) < 0)
		{
		  memcpy(list_r + nr * size, list_a + na * size, size);
		  na++;
		  nr++;
		}
	      else
		{
		  memcpy(list_r + nr * size, list_b + nb * size, size);
		  nb++;
		  nr++;
		}
	    }
	  else if(na < nlist[cpua])
	    {
	      memcpy(list_r + nr * size, list_a + na * size, size);
	      na++;
	      nr++;
	    }
	  else if(nb < nlist[cpub])
	    {
	      memcpy(list_r + nr * size, list_b + nb * size, size);
	      nb++;
	      nr++;
	    }

	}

      myfree(list_r);
      myfree(list_b);
      myfree(list_a);
    }
}


static void serial_sort(char *base, size_t nmemb, size_t size, int (*compar) (const void *, const void *))
{
  const size_t storage = nmemb * size;

  char *tmp = (char *) mymalloc("char *tmp", storage);

  msort_serial_with_tmp(base, nmemb, size, compar, tmp);

  myfree(tmp);
}


static void msort_serial_with_tmp(char *base, size_t n, size_t s, int (*compar) (const void *, const void *),
				  char *t)
{
  char *tmp;
  char *b1, *b2;
  size_t n1, n2;

  if(n <= 1)
    return;

  n1 = n / 2;
  n2 = n - n1;
  b1 = base;
  b2 = base + n1 * s;

  msort_serial_with_tmp(b1, n1, s, compar, t);
  msort_serial_with_tmp(b2, n2, s, compar, t);

  tmp = t;

  while(n1 > 0 && n2 > 0)
    {
      if(compar(b1, b2) < 0)
	{
	  --n1;
	  memcpy(tmp, b1, s);
	  tmp += s;
	  b1 += s;
	}
      else
	{
	  --n2;
	  memcpy(tmp, b2, s);
	  tmp += s;
	  b2 += s;
	}
    }

  if(n1 > 0)
    memcpy(tmp, b1, n1 * s);

  memcpy(base, t, (n - n2) * s);
}
