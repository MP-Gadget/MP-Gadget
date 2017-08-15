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
#include "mymalloc.h"



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

  strcpy(mode, "a+");

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
  return MPI_Wtime();
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
      dt = t1 + pow(2, 32) / CLOCKS_PER_SEC - t0;
    }

  return dt;
}

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

int MPI_Alltoallv_smart(void *sendbuf, int *sendcnts, int *sdispls,
        MPI_Datatype sendtype, void *recvbuf, int *recvcnts,
        int *rdispls, MPI_Datatype recvtype, MPI_Comm comm) 
/* 
 * sdispls, recvcnts rdispls can be NULL,
 *
 * if recvbuf is NULL, returns total number of item required to hold the
 * data.
 * */
{
    int ThisTask;
    int NTask;
    MPI_Comm_rank(comm, &ThisTask);
    MPI_Comm_size(comm, &NTask);
    int i;
    int nn = 0;
    for(i = 0; i < NTask; i ++) {
        if(sendcnts[i] > 0) {
            nn ++;
        }
    }
    if(recvcnts == NULL) {
        recvcnts = alloca(sizeof(int) * NTask);
        MPI_Alltoall(sendcnts, 1, MPI_INT,
                     recvcnts, 1, MPI_INT, comm);
    }
    if(recvbuf == NULL) {
        int totalrecv = 0;
        for(i = 0; i < NTask; i ++) {
            totalrecv += recvcnts[i];
        }
        return totalrecv;
    }
    if(sdispls == NULL) {
        sdispls = alloca(sizeof(int) * NTask);
        sdispls[0] = 0;
        for (i = 1; i < NTask; i++) {
            sdispls[i] = sdispls[i - 1] + sendcnts[i - 1];
        }
    }
    if(rdispls == NULL) {
        rdispls = alloca(sizeof(int) * NTask);
        rdispls[0] = 0;
        for (i = 1; i < NTask; i++) {
            rdispls[i] = rdispls[i - 1] + recvcnts[i - 1];
        }
    }

    int dense = nn < NTask * 0.2;
    int tot_dense = 0;
    MPI_Allreduce(&dense, &tot_dense, 1, MPI_INT, MPI_SUM, comm);

    if(tot_dense != 0) {
        return MPI_Alltoallv(sendbuf, sendcnts, sdispls, 
                    sendtype, recvbuf, 
                    recvcnts, rdispls, recvtype, comm);
    } else {
        return MPI_Alltoallv_sparse(sendbuf, sendcnts, sdispls, 
                    sendtype, recvbuf, 
                    recvcnts, rdispls, recvtype, comm);

    }
}

int MPI_Alltoallv_sparse(void *sendbuf, int *sendcnts, int *sdispls,
        MPI_Datatype sendtype, void *recvbuf, int *recvcnts,
        int *rdispls, MPI_Datatype recvtype, MPI_Comm comm) {

        return MPI_Alltoallv(sendbuf, sendcnts, sdispls,
                    sendtype, recvbuf,
                    recvcnts, rdispls, recvtype, comm);
        //return MPI_Alltoallv_fix_buserror(sendbuf, sendcnts, sdispls,
        //            sendtype, recvbuf,
        //            recvcnts, rdispls, recvtype, comm);
    }

int MPI_Alltoallv_fix_buserror(void *sendbuf, int *sendcnts, int *sdispls,
        MPI_Datatype sendtype, void *recvbuf, int *recvcnts,
        int *rdispls, MPI_Datatype recvtype, MPI_Comm comm)
{
   int i, j, ncut;
   int ThisTask, NTask;
   int *sendcnts_ncut, *sdispls_ncut, *recvcnts_ncut, *rdispls_ncut;
   MPI_Comm_rank(comm, &ThisTask);
   MPI_Comm_size(comm, &NTask);
   
   ncut = 4;
   sendcnts_ncut = (int *) malloc(NTask * sizeof(int));
   recvcnts_ncut = (int *) malloc(NTask * sizeof(int));
   sdispls_ncut = (int *) malloc(NTask * sizeof(int));
   rdispls_ncut = (int *) malloc(NTask * sizeof(int));

   for (j =0; j<ncut;j++){
       for (i=0; i<NTask; i++){
           sdispls_ncut[i] = sendcnts[i] * j/ ncut; 
           sendcnts_ncut[i] = sendcnts[i] * (j+1)/ ncut - sdispls_ncut[i];
           sdispls_ncut[i] += sdispls[i];
           rdispls_ncut[i] = recvcnts[i] * j/ ncut;       
           recvcnts_ncut[i] = recvcnts[i] * (j+1)/ ncut - rdispls_ncut[i];
           rdispls_ncut[i] += rdispls[i];
       }
       MPI_Alltoallv(sendbuf, sendcnts_ncut, sdispls_ncut, sendtype, recvbuf, recvcnts_ncut, rdispls_ncut, recvtype, comm);
   }
   return 1;       
}
