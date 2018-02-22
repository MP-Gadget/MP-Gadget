#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <unistd.h>
#include <signal.h>
#include <gsl/gsl_rng.h>


#include "system.h"
#include "mymalloc.h"


#define  RNDTABLE 8192


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

#endif


static double RndTable[RNDTABLE];

static gsl_rng *random_generator;	/*!< the random number generator used */

double get_random_number(uint64_t id)
{
    return RndTable[(int)(id % RNDTABLE)];
}

void set_random_numbers(int seed)
{
    random_generator = gsl_rng_alloc(gsl_rng_ranlxd1);

    gsl_rng_set(random_generator, seed);	/* start-up seed */

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

void
sumup_large_ints(int n, int *src, int64_t *res)
{
    MPI_Comm comm = MPI_COMM_WORLD;
    int NTask;
    int ThisTask;
    MPI_Comm_size(comm, &NTask);
    MPI_Comm_rank(comm, &ThisTask);

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
    MPI_Comm comm = MPI_COMM_WORLD;
    int NTask;
    int ThisTask;
    MPI_Comm_size(comm, &NTask);
    MPI_Comm_rank(comm, &ThisTask);
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

int64_t
MPIU_cumsum(int64_t countLocal, MPI_Comm comm)
{
    int NTask;
    int ThisTask;
    MPI_Comm_size(comm, &NTask);
    MPI_Comm_rank(comm, &ThisTask);

    int64_t offsetLocal;
    int64_t count[NTask];
    int64_t offset[NTask];
    MPI_Gather(&countLocal, 1, MPI_INT64, &count[0], 1, MPI_INT64, 0, MPI_COMM_WORLD);
    if(ThisTask == 0) {
        offset[0] = 0;
        int i;
        for(i = 1; i < NTask; i ++) {
            offset[i] = offset[i-1] + count[i-1];
        }
    }
    MPI_Scatter(&offset[0], 1, MPI_INT64, &offsetLocal, 1, MPI_INT64, 0, MPI_COMM_WORLD);
    return offsetLocal;
}

int64_t count_sum(int64_t countLocal) {
    int64_t sum = 0;
    MPI_Allreduce(&countLocal, &sum, 1, MPI_INT64, MPI_SUM, MPI_COMM_WORLD);
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
    int *a_sdispls=NULL, *a_recvcnts=NULL, *a_rdispls=NULL;
    for(i = 0; i < NTask; i ++) {
        if(sendcnts[i] > 0) {
            nn ++;
        }
    }
    if(recvcnts == NULL) {
        a_recvcnts = ta_malloc("recvcnts", int, NTask);
        recvcnts = a_recvcnts;
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
        a_sdispls = ta_malloc("sdispls", int, NTask);
        sdispls = a_sdispls;
        sdispls[0] = 0;
        for (i = 1; i < NTask; i++) {
            sdispls[i] = sdispls[i - 1] + sendcnts[i - 1];
        }
    }
    if(rdispls == NULL) {
        a_rdispls = ta_malloc("rdispls", int, NTask);
        rdispls = a_rdispls;
        rdispls[0] = 0;
        for (i = 1; i < NTask; i++) {
            rdispls[i] = rdispls[i - 1] + recvcnts[i - 1];
        }
    }

    int dense = nn < NTask * 0.2;
    int tot_dense = 0, ret;
    MPI_Allreduce(&dense, &tot_dense, 1, MPI_INT, MPI_SUM, comm);

    if(tot_dense != 0) {
        ret = MPI_Alltoallv(sendbuf, sendcnts, sdispls,
                    sendtype, recvbuf, 
                    recvcnts, rdispls, recvtype, comm);
    } else {
        ret = MPI_Alltoallv_sparse(sendbuf, sendcnts, sdispls,
                    sendtype, recvbuf, 
                    recvcnts, rdispls, recvtype, comm);

    }
    if(a_rdispls)
        ta_free(a_rdispls);
    if(a_sdispls)
        ta_free(a_sdispls);
    if(a_recvcnts)
        ta_free(a_recvcnts);
    return ret;
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
    MPI_Request *requests = mymalloc("requests", NTask * 2 * sizeof(MPI_Request));
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

    MPI_Barrier(comm);
    /* not really necessary, but this will guarantee that all receives are
       posted before the sends, which helps the stability of MPI on
       bluegene, and perhaps some mpich1-clusters */
    /* Note 08/2016: Even on modern hardware this barrier leads to a slight speedup.
     * Probably because it allows the code to hit a fast path transfer.*/

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
    myfree(requests);
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

int
cluster_get_hostid()
{
    /* Find a unique hostid for the computing rank. */
    char hostname[1024];
    int i;
    gethostname(hostname, 1024);

    MPI_Barrier(MPI_COMM_WORLD);

    int l = strlen(hostname) + 4;
    int ml = 0;
    int NTask;
    int ThisTask;
    char * buffer;
    int * nid;
    MPI_Comm_size(MPI_COMM_WORLD, &NTask);
    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
    MPI_Allreduce(&l, &ml, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    
    buffer = malloc(ml * NTask);
    nid = malloc(sizeof(int) * NTask);
    MPI_Allgather(hostname, ml, MPI_BYTE, buffer, ml, MPI_BYTE, MPI_COMM_WORLD);

    typedef int(*compar_fn)(const void *, const void *);
    qsort(buffer, NTask, ml, (compar_fn) strcmp);
    
    nid[0] = 0;
    for(i = 1; i < NTask; i ++) {
        if(strcmp(buffer + i * ml, buffer + (i - 1) *ml)) {
            nid[i] = nid[i - 1] + 1;
        } else {
            nid[i] = nid[i - 1];
        }
    }
    for(i = 0; i < NTask; i ++) {
        if(!strcmp(hostname, buffer + i * ml)) {
            break;
        }
    }
    int rt = nid[i];
    free(buffer);
    free(nid);
    MPI_Barrier(MPI_COMM_WORLD);
    return rt;
}

int
cluster_get_num_hosts()
{
    /* return the number of hosts */
    int id = cluster_get_hostid();
    int maxid;
    MPI_Allreduce(&id, &maxid, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    return maxid + 1;
}

double
get_physmem_bytes()
{
#if defined _SC_PHYS_PAGES && defined _SC_PAGESIZE
    { /* This works on linux-gnu, solaris2 and cygwin.  */
        double pages = sysconf (_SC_PHYS_PAGES);
        double pagesize = sysconf (_SC_PAGESIZE);
        if (0 <= pages && 0 <= pagesize)
            return pages * pagesize;
    }
#endif

#if defined HW_PHYSMEM
    { /* This works on *bsd and darwin.  */
        unsigned int physmem;
        size_t len = sizeof physmem;
        static int mib[2] = { CTL_HW, HW_PHYSMEM };

        if (sysctl (mib, ARRAY_SIZE (mib), &physmem, &len, NULL, 0) == 0
                && len == sizeof (physmem))
            return (double) physmem;
    }
#endif
    return 64 * 1024 * 1024;
}

int
MPIU_Any(int condition, MPI_Comm comm)
{
    MPI_Allreduce(MPI_IN_PLACE, &condition, 1, MPI_INT, MPI_LOR, comm);
    return condition;
}

void
MPIU_write_pids(char * filename)
{
    MPI_Comm comm = MPI_COMM_WORLD;
    int NTask;
    int ThisTask;
    MPI_Comm_size(comm, &NTask);
    MPI_Comm_rank(comm, &ThisTask);
    pid_t my_pid;
    char mode[8], buf[500];
    FILE *fd;
    int i;

    my_pid = getpid();

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
