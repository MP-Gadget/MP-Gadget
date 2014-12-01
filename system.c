#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <unistd.h>
#include <signal.h>
#include <gsl/gsl_rng.h>


#include "allvars.h"
#include "proto.h"



#ifdef DEBUG
#include <fenv.h>
void enable_core_dumps_and_fpu_exceptions(void)
{
  struct rlimit rlim;
  extern int feenableexcept(int __excepts);

  /* enable floating point exceptions */

  /*
     feenableexcept(FE_DIVBYZERO | FE_INVALID);
   */

  /* Note: FPU exceptions appear not to work properly 
   * when the Intel C-Compiler for Linux is used
   */

  /* set core-dump size to infinity */
  getrlimit(RLIMIT_CORE, &rlim);
  rlim.rlim_cur = RLIM_INFINITY;
  setrlimit(RLIMIT_CORE, &rlim);

  /* MPICH catches the signales SIGSEGV, SIGBUS, and SIGFPE....
   * The following statements reset things to the default handlers,
   * which will generate a core file.  
   */
  /*
     signal(SIGSEGV, catch_fatal);
     signal(SIGBUS, catch_fatal);
     signal(SIGFPE, catch_fatal);
     signal(SIGINT, catch_fatal);
   */

  signal(SIGSEGV, SIG_DFL);
  signal(SIGBUS, SIG_DFL);
  signal(SIGFPE, SIG_DFL);
  signal(SIGINT, SIG_DFL);

  /* Establish a handler for SIGABRT signals. */
  signal(SIGABRT, catch_abort);
}


void catch_abort(int sig)
{
  MPI_Finalize();
  exit(0);
}

void catch_fatal(int sig)
{
  MPI_Finalize();

  signal(sig, SIG_DFL);
  raise(sig);
}

void write_pid_file(void)
{
  pid_t my_pid;
  char mode[8], buf[500];
  FILE *fd;
  int i;

  my_pid = getpid();

  sprintf(buf, "%s%s", All.OutputDir, "PIDs.txt");

  if(RestartFlag == 0)
    strcpy(mode, "w");
  else
    strcpy(mode, "a");

  for(i = 0; i < NTask; i++)
    {
      if(ThisTask == i)
	{
	  if(ThisTask == 0)
	    sprintf(mode, "w");
	  else
	    sprintf(mode, "a");

	  if((fd = fopen(buf, mode)))
	    {
	      fprintf(fd, "%s %d\n", getenv("HOST"), (int) my_pid);
	      fclose(fd);
	    }
	}

      MPI_Barrier(MPI_COMM_WORLD);
    }
}
#endif


/*
double get_random_number(unsigned int id)
{
  return RndTable[(id % RNDTABLE)];
}
*/

double get_random_number(MyIDType id)
{
  return RndTable[(int)(id % RNDTABLE)];
}

void set_random_numbers(void)
{
  int i;

  for(i = 0; i < RNDTABLE; i++)
    RndTable[i] = gsl_rng_uniform(random_generator);
}


/* returns the number of cpu-ticks in seconds that
 * have elapsed. (or the wall-clock time)
 */
double second(void)
{
#ifdef WALLCLOCK
  return MPI_Wtime();
#else
  return ((double) clock()) / CLOCKS_PER_SEC;
#endif

  /* note: on AIX and presumably many other 32bit systems, 
   * clock() has only a resolution of 10ms=0.01sec 
   */
}

/* returns the time difference between two measurements 
 * obtained with second(). The routine takes care of the 
 * possible overflow of the tick counter on 32bit systems.
 */
double timediff(double t0, double t1)
{
  double dt;

  dt = t1 - t0;

  if(dt < 0)			/* overflow has occured (for systems with 32bit tick counter) */
    {
#ifdef WALLCLOCK
      dt = 0;
#else
      dt = t1 + pow(2, 32) / CLOCKS_PER_SEC - t0;
#endif
    }

  return dt;
}




#ifdef X86FIX

#define _FPU_SETCW(x) asm volatile ("fldcw %0": :"m" (x));
#define _FPU_GETCW(x) asm volatile ("fnstcw %0":"=m" (x));
#define _FPU_EXTENDED 0x0300
#define _FPU_DOUBLE   0x0200

void x86_fix(void)
{
  unsigned short dummy, new_cw;
  unsigned short *old_cw;

  old_cw = &dummy;

  _FPU_GETCW(*old_cw);
  new_cw = (*old_cw & ~_FPU_EXTENDED) | _FPU_DOUBLE;
  _FPU_SETCW(new_cw);
}

#endif


void sumup_large_ints(int n, int *src, int64_t *res)
{
  int i, j, *numlist;

  numlist = (int *) mymalloc("numlist", NTask * n * sizeof(int));
  MPI_Allgather(src, n, MPI_INT, numlist, n, MPI_INT, MPI_COMM_WORLD);

  for(j = 0; j < n; j++)
    res[j] = 0;

  for(i = 0; i < NTask; i++)
    for(j = 0; j < n; j++)
      res[j] += numlist[i * n + j];

  myfree(numlist);
}

void sumup_longs(int n, int64_t *src, int64_t *res)
{
  int i, j;
  int64_t *numlist;

  numlist = (int64_t *) mymalloc("numlist", NTask * n * sizeof(int64_t));
  MPI_Allgather(src, n * sizeof(int64_t), MPI_BYTE, numlist, n * sizeof(int64_t), MPI_BYTE,
		MPI_COMM_WORLD);

  for(j = 0; j < n; j++)
    res[j] = 0;

  for(i = 0; i < NTask; i++)
    for(j = 0; j < n; j++)
      res[j] += numlist[i * n + j];

  myfree(numlist);
}

int64_t count_to_offset(int64_t countLocal) {
    int64_t offsetLocal;
    int64_t count[NTask];
    int64_t offset[NTask];
    MPI_Gather(&countLocal, 1, MPI_LONG, &count[0], 1, MPI_LONG, 0, MPI_COMM_WORLD);
    if(ThisTask == 0) {
        offset[0] = 0;
        int i;
        for(i = 1; i < NTask; i ++) {
            offset[i] = offset[i-1] + count[i-1];
        }
    }
    MPI_Scatter(&offset[0], 1, MPI_LONG, &offsetLocal, 1, MPI_LONG, 0, MPI_COMM_WORLD);
    return offsetLocal;
}

int64_t count_sum(int64_t countLocal) {
    int64_t sum = 0;
    MPI_Allreduce(&countLocal, &sum, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
    return sum;
}

size_t sizemax(size_t a, size_t b)
{
  if(a < b)
    return b;
  else
    return a;
}


void report_VmRSS(void)
{
  pid_t my_pid;
  FILE *fd;
  char buf[1024];

  my_pid = getpid();

  sprintf(buf, "/proc/%d/status", my_pid);

  if((fd = fopen(buf, "r")))
    {
      while(1)
	{
	  if(fgets(buf, 500, fd) != buf)
	    break;

	  if(strncmp(buf, "VmRSS", 5) == 0)
	    {
	      printf("ThisTask=%d: %s", ThisTask, buf);
	    }
	  if(strncmp(buf, "VmSize", 6) == 0)
	    {
	      printf("ThisTask=%d: %s", ThisTask, buf);
	    }
	}
      fclose(fd);
    }
}

int MPI_Alltoallv_sparse(void *sendbuf, int *sendcnts, int *sdispls,
        MPI_Datatype sendtype, void *recvbuf, int *recvcnts,
        int *rdispls, MPI_Datatype recvtype, MPI_Comm comm) {

    int ThisTask;
    int NTask;
    MPI_Comm_rank(comm, &ThisTask);
    MPI_Comm_size(comm, &NTask);
    int PTask;
    int ngrp;

    for(PTask = 0; NTask > (1 << PTask); PTask++);

    ptrdiff_t lb;
    ptrdiff_t send_elsize;
    ptrdiff_t recv_elsize;

    MPI_Type_get_extent(sendtype, &lb, &send_elsize);
    MPI_Type_get_extent(recvtype, &lb, &recv_elsize);

#ifndef NO_ISEND_IRECV_IN_DOMAIN
    int n_requests;
    MPI_Request requests[NTask * 2];
    n_requests = 0;


    for(ngrp = 0; ngrp < (1 << PTask); ngrp++)
    {
        int target = ThisTask ^ ngrp;

        if(target >= NTask) continue;
        if(recvcnts[target] == 0) continue;
        MPI_Irecv(
                ((char*) recvbuf) + recv_elsize * rdispls[target], 
                recvcnts[target],
                recvtype, target, 101934, comm, &requests[n_requests++]);
    }

    MPI_Barrier(comm);	/* not really necessary, but this will guarantee that all receives are
                                       posted before the sends, which helps the stability of MPI on 
                                       bluegene, and perhaps some mpich1-clusters */

    for(ngrp = 0; ngrp < (1 << PTask); ngrp++)
    {
        int target = ThisTask ^ ngrp;
        if(target >= NTask) continue;
        if(sendcnts[target] == 0) continue;
        MPI_Isend(((char*) sendbuf) + send_elsize * sdispls[target], 
                sendcnts[target],
                sendtype, target, 101934, comm, &requests[n_requests++]);
    }

    MPI_Waitall(n_requests, requests, MPI_STATUSES_IGNORE);

#else
    for(ngrp = 0; ngrp < (1 << PTask); ngrp++)
    {
        int target = ThisTask ^ ngrp;

        if(target >= NTask) continue;
        if(sendcnts[target] == 0 && recvcnts[target] == 0) continue;
        MPI_Sendrecv(((char*)sendbuf) + send_elsize * sdispls[target], 
                sendcnts[target], sendtype, 
                target, 101934,
                ((char*)recvbuf) + recv_elsize * rdispls[target],
                recvcnts[target], recvtype, 
                target, 101934, 
                comm, MPI_STATUS_IGNORE);

    }
#endif
    /* ensure the collective-ness */
    MPI_Barrier(comm);

    return 0;
} 
