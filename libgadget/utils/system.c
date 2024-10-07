#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <stdint.h>
#include <stdarg.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <unistd.h>
#include <signal.h>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#include <omp.h>

#define __UTILS_SYSTEM_C
#include "system.h"
#include "mymalloc.h"
#include "endrun.h"

/* NOTE:
 *
 * The MPIU_xxx functions must be called after the memory module is initalized.
 * Shall split them to a new module.
 *
 * */

#if 0
#include <fenv.h>
void enable_core_dumps_and_fpu_exceptions(void)
{
  /* enable floating point exceptions */

  /* extern int feenableexcept(int __excepts);

     feenableexcept(FE_DIVBYZERO | FE_INVALID);
   */

  /* Note: FPU exceptions appear not to work properly
   * when the Intel C-Compiler for Linux is used
   */

  /* set core-dump size to infinity.
   * Don't do this because it may be there for a reason:
   * the cluster does not like us to dump 2PB of core! */
  /* getrlimit(RLIMIT_CORE, &rlim);
  struct rlimit rlim;
  rlim.rlim_cur = RLIM_INFINITY;
  setrlimit(RLIMIT_CORE, &rlim);*/
}

#endif

double get_random_number(const uint64_t id, const RandTable * const rnd)
{
    if(!rnd->Table)
        endrun(1, "Random number called with non-allocated random table\n")
    /* Draw from the table: note that only the lowest bits of the ID are used*/;
    return rnd->Table[id % rnd->tablesize];
}

RandTable set_random_numbers(uint64_t seed, const size_t rndtablesize)
{
    RandTable rnd;
    rnd.Table = (double *) mymalloc2("Random", rndtablesize * sizeof(double));
    rnd.tablesize = rndtablesize;
    /* start-up seed */

    boost::random::mt19937 random_generator(seed);
    boost::random::uniform_real_distribution<double> dist(0, 1);
    /* Populate a table with uniform random numbers between 0 and 1*/
    size_t i;
    for(i = 0; i < rndtablesize; i++)
        rnd.Table[i] = dist(random_generator);

    return rnd;
}

void free_random_numbers(RandTable * rnd)
{
    myfree(rnd->Table);
    rnd->Table = NULL;
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

static int
putline(const char * prefix, const char * line)
{
    const char * p, * q;
    p = q = line;
    int newline = 1;
    while(*p != 0) {
        if(newline)
            write(STDOUT_FILENO, prefix, strlen(prefix));
        if (*p == '\n') {
            write(STDOUT_FILENO, q, p - q + 1);
            q = p + 1;
            newline = 1;
            p ++;
            continue;
        }
        newline = 0;
        p++;
    }
    /* if the last line did not end with a new line, fix it here. */
    if (q != p) {
        const char * warning = "LASTMESSAGE did not end with new line: ";
        write(STDOUT_FILENO, warning, strlen(warning));
        write(STDOUT_FILENO, q, p - q);
        write(STDOUT_FILENO, "\n", 1);
    }
    return 0;
}


/* Watch out:
 *
 * On some versions of OpenMPI with CPU frequency scaling we see negative time
 * due to a bug in OpenMPI https://github.com/open-mpi/ompi/issues/3003
 *
 * But they have fixed it.
 */

static double _timestart = -1;
/*
 * va_list version of MPIU_Trace.
 * */
void
MPIU_Tracev(MPI_Comm comm, int where, int error, const char * fmt, va_list va)
{
    if(_timestart == -1) {
        _timestart = MPI_Wtime();
    }
    int ThisTask;
    MPI_Comm_rank(comm, &ThisTask);
    char prefix[128];

    char buf[4096];
    vsnprintf(buf, 4096, fmt, va);
    buf[4095] = '\0';
    char err[] = "ERROR: ";
    /* Print nothing if not an error*/
    if(!error)
        err[0] = '\0';

    if(where > 0) {
        sprintf(prefix, "[ %09.2f ] %sTask %d: ", MPI_Wtime() - _timestart, err, ThisTask);
    } else {
        sprintf(prefix, "[ %09.2f ] %s", MPI_Wtime() - _timestart, err);
    }

    if(ThisTask == 0 || where > 0) {
        putline(prefix, buf);
    }
}

/*
 * Write a trace message to the communicator.
 * if where > 0, write from all ranks.
 * if where == 0, only write from root rank.
 * */
void MPIU_Trace(MPI_Comm comm, int where, const char * fmt, ...)
{
    va_list va;
    va_start(va, fmt);
    MPIU_Tracev(comm, where, 0, fmt, va);
    va_end(va);
}

int64_t
MPIU_cumsum(int64_t countLocal, MPI_Comm comm)
{
    int NTask;
    int ThisTask;
    MPI_Comm_size(comm, &NTask);
    MPI_Comm_rank(comm, &ThisTask);

    int64_t offsetLocal;
    int64_t * count = ta_malloc("counts", int64_t, NTask);
    int64_t * offset = ta_malloc("offsets", int64_t, NTask);
    MPI_Gather(&countLocal, 1, MPI_INT64, &count[0], 1, MPI_INT64, 0, MPI_COMM_WORLD);
    if(ThisTask == 0) {
        offset[0] = 0;
        int i;
        for(i = 1; i < NTask; i ++) {
            offset[i] = offset[i-1] + count[i-1];
        }
    }
    MPI_Scatter(&offset[0], 1, MPI_INT64, &offsetLocal, 1, MPI_INT64, 0, MPI_COMM_WORLD);
    ta_free(offset);
    ta_free(count);
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

    int n_requests;
    MPI_Request *requests = (MPI_Request *) mymalloc("requests", NTask * 2 * sizeof(MPI_Request));
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
    /* ensure the collective-ness */
    MPI_Barrier(comm);

    return 0;
}

/* return the number of hosts */
int
cluster_get_num_hosts(void)
{
    /* Find a unique hostid for the computing rank. */
    int NTask;
    int ThisTask;
    MPI_Comm_size(MPI_COMM_WORLD, &NTask);
    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);

    /* Size is set by the size of the temp heap:
     * this fills it and should be changed if needed.*/
    const int bufsz = 256;
    char * buffer = ta_malloc("buffer", char, bufsz * NTask);
    memset(buffer, 0, bufsz * NTask);
    int i, j;
    gethostname(&buffer[bufsz*ThisTask], bufsz);
    buffer[bufsz * ThisTask + bufsz - 1] = '\0';
    MPI_Allgather(MPI_IN_PLACE, bufsz, MPI_CHAR, buffer, bufsz, MPI_CHAR, MPI_COMM_WORLD);

    int nunique = 0;
    /* Count unique entries*/
    for(j = 0; j < NTask; j++) {
        for(i = j+1; i < NTask; i++) {
            if(strncmp(buffer + i * bufsz, buffer + j * bufsz, bufsz) == 0)
                break;
        }
        if(i == NTask)
            nunique++;
    }
    ta_free(buffer);
    return nunique;
}

double
get_physmem_bytes(void)
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

/**
 * A fancy MPI barrier (use MPIU_Barrier macro)
 *
 *  - aborts if barrier mismatch occurs
 *  - warn if some ranks are very imbalanced.
 *
 */
int
_MPIU_Barrier(const char * fn, const int line, MPI_Comm comm)
{
    int ThisTask, NTask;
    MPI_Comm_size(comm, &NTask);
    MPI_Comm_rank(comm, &ThisTask);
    int * recvbuf = ta_malloc("tags", int, NTask);
    int tag = 0;
    int i;
    for(i = 0; fn[i]; i ++) {
        tag += (int)fn[i] * 8;
    }
    tag += line;

    MPI_Request request;
    MPI_Igather(&tag, 1, MPI_INT, recvbuf, 1, MPI_INT, 0, comm, &request);
    i = 0;
    int flag = 1;
    int tsleep = 0;
    while(flag) {
        MPI_Test(&request, &flag, MPI_STATUS_IGNORE);
        if(flag) break;
        usleep(i * 1000);
        tsleep += i * 1000;
        i = i + 1;
        if(i == 50) {
            if(ThisTask == 0) {
                MPIU_Trace(comm, 0, "Waited more than %g seconds during barrier %s : %d \n", tsleep / 1000000., fn, line);
            }
            break;
        }
    }
    MPI_Wait(&request, MPI_STATUS_IGNORE);
    /* now check if all ranks indeed hit the same barrier. Some MPIs do allow them to mix up! */
    if (ThisTask == 0) {
        for(i = 0; i < NTask; i ++) {
            if(recvbuf[i] != tag) {
                MPIU_Trace(comm, 0, "Task %d Did not hit barrier at %s : %d; expecting %d, got %d\n", i, fn, line, tag, recvbuf[i]);
            }
        }
    }
    ta_free(recvbuf);
    return 0;
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

    int my_pid = getpid();
    int * pids = ta_malloc("pids", int, NTask);
    /* Smaller buffer than in cluster_get_num_hosts because
     * here an overflow is harmless but running out of memory isn't*/
    int bufsz = 64;
    char * hosts = ta_malloc("hosts", char, (NTask+1) * bufsz);
    char * thishost = hosts + NTask * bufsz;
    gethostname(thishost, bufsz);
    thishost[bufsz - 1] = '\0';
    /* MPI_IN_PLACE is not used here because the MPI on travis doesn't like it*/
    MPI_Gather(thishost, bufsz, MPI_CHAR, hosts, bufsz, MPI_CHAR, 0, comm);
    MPI_Gather(&my_pid, 1, MPI_INT, pids, 1, MPI_INT, 0, comm);

    if(ThisTask == 0)
    {
        int i;
        FILE *fd = fopen(filename, "w");
        if(!fd)
            endrun(5, "Could not open pidfile %s\n", filename);
        for(i = 0; i < NTask; i++)
            fprintf(fd, "host: %s pid: %d\n", hosts+i*bufsz, pids[i]);
        fclose(fd);
    }
    myfree(hosts);
    myfree(pids);
}

size_t gadget_compact_thread_arrays(int ** dest, gadget_thread_arrays * arrays)
{
    int i;
    size_t asize = 0;
    size_t * offsets = ta_malloc("tmp", size_t, arrays->narrays);
    for(i = 0; i < arrays->narrays; i++)
    {
        offsets[i] = asize;
        asize += arrays->sizes[i];
    }

    #pragma omp parallel for
    for(i = 1; i < arrays->narrays; i++)
        memmove(arrays->dest + offsets[i], arrays->srcs[i], sizeof(int) * arrays->sizes[i]);
    ta_free(offsets);
    for(i = arrays->narrays-1; i >= 1; i--)
        myfree(arrays->srcs[i]);
    ta_free(arrays->srcs);
    ta_free(arrays->sizes);
    *dest = arrays->dest;
    return asize;
}

gadget_thread_arrays gadget_setup_thread_arrays(const char * destname, int alloc_high, size_t total_size)
{
    gadget_thread_arrays threadarray = {0};
    const int narrays = omp_get_max_threads();
    if (alloc_high)
        threadarray.dest = (int *) mymalloc2(destname, sizeof(int) * total_size);
    else
        threadarray.dest = (int *) mymalloc(destname, sizeof(int) * total_size);
    threadarray.sizes = ta_malloc("nexthr", size_t, narrays);
    threadarray.srcs = ta_malloc("threx", int *, narrays);
    threadarray.total_size = total_size;
    threadarray.schedsz = total_size/narrays + 1;
    threadarray.narrays = narrays;
    int i;
    threadarray.srcs[0] = threadarray.dest;
    threadarray.sizes[0] = 0;
    for(i=1; i < narrays; i++) {
        if(alloc_high)
            threadarray.srcs[i] = (int*) mymalloc2("threx", sizeof(int) * total_size);
        else
            threadarray.srcs[i] = (int*) mymalloc("threx", sizeof(int) * total_size);
        threadarray.sizes[i] = 0;
    }
    return threadarray;
}
