#ifndef __UTILS_SYSTEM_H__
#define __UTILS_SYSTEM_H__

#include <stdint.h>

/* Note on a 32-bit architecture MPI_LONG may be 32-bit,
 * so these should be MPI_LONG_LONG. But in
 * the future MPI_LONG_LONG may become 128-bit.*/
#define MPI_UINT64 MPI_UNSIGNED_LONG
#define MPI_INT64 MPI_LONG

/* check the version of OPENMP */
#if _OPENMP < 201107
#error MP-Gadget requires OpenMP >= 3.1 if openmp is enabled. \
       Try to compile without openmp or use a newer compiler (gcc >= 4.7) .
#endif

#ifdef DEBUG
void catch_abort(int sig);
void catch_fatal(int sig);
void enable_core_dumps_and_fpu_exceptions(void);
#endif

int cluster_get_num_hosts(void);
double get_physmem_bytes(void);

/* Gets a random number in the range [0, 1). Only the low bits of the id are used,
 * and random deviates are drawn from a pre-seeded table so that they are independent of processor.*/
double get_random_number(uint64_t id);
void set_random_numbers(int seed);
void sumup_large_ints(int n, int *src, int64_t *res);
void sumup_longs(int n, int64_t *src, int64_t *res);
int64_t count_sum(int64_t countLocal);
//int64_t count_to_offset(int64_t countLocal);

/* Returns true if condition is true on ANY processor*/
int MPIU_Any(int condition, MPI_Comm comm);

void MPIU_write_pids(char * filename);

/* Compact an array which has segments (usually corresponding to different threads).
 * After this is run, it will be a single contiguous array. The memory can then be realloced.
 * Function returns size of the final array.*/
size_t gadget_compact_thread_arrays(int * dest, int * srcs[], size_t sizes[], int narrays);

/* Set up pointers to different parts of a single segmented array (usually corresponding to different threads).*/
void gadget_setup_thread_arrays(int * dest, int * srcs[], size_t sizes[], size_t total_size, int narrays);

int MPI_Alltoallv_smart(void *sendbuf, int *sendcnts, int *sdispls,
        MPI_Datatype sendtype, void *recvbuf, int *recvcnts,
        int *rdispls, MPI_Datatype recvtype, MPI_Comm comm);

int MPI_Alltoallv_sparse(void *sendbuf, int *sendcnts, int *sdispls,
        MPI_Datatype sendtype, void *recvbuf, int *recvcnts,
        int *rdispls, MPI_Datatype recvtype, MPI_Comm comm);

double timediff(double t0, double t1);
double second(void);
size_t sizemax(size_t a, size_t b);

static inline int64_t atomic_fetch_and_add_64(int64_t * ptr, int64_t value) {
    int64_t k;
#pragma omp atomic capture
    {
      k = (*ptr);
      (*ptr)+=value;
    }
    return k;
}

static inline int atomic_fetch_and_add(int * ptr, int value) {
    int k;
#pragma omp atomic capture
    {
      k = (*ptr);
      (*ptr)+=value;
    }
    return k;
}
static inline int atomic_add_and_fetch(int * ptr, int value) {
    int k;
#pragma omp atomic capture
    {
      (*ptr)+=value;
      k = (*ptr);
    }
    return k;
}

void MPIU_Trace(MPI_Comm comm, int where, const char * fmt, ...);
void MPIU_Tracev(MPI_Comm comm, int where, int error, const char * fmt, va_list va);

int _MPIU_Barrier(const char * fn, const int ln, MPI_Comm comm);

/* Fancy barrier which warns if there is a lot of imbalance. */
#define MPIU_Barrier(comm) _MPIU_Barrier(__FILE__, __LINE__, comm)

#ifdef MOREDEBUG
/* Checked versions of some MPI routines which make sure that the number we get out is sensible*/
int MPI_Allreduce_Checked(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm, const int line, const char * file);
int MPI_Reduce_Checked(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm, const int line, const char * file);
/* Defines, guarded so the implementation can still use the original*/
#ifndef __UTILS_SYSTEM_C
#define MPI_Allreduce(sendbuf, recvbuf, count, datatype, op, comm) \
MPI_Allreduce_Checked(sendbuf, recvbuf, count, datatype, op, comm, __LINE__, __FILE__)
#define MPI_Reduce(sendbuf, recvbuf, count, datatype, op, root, comm) \
MPI_Reduce_Checked(sendbuf, recvbuf, count, datatype, op, root, comm, __LINE__, __FILE__)
#endif

#endif

#endif //_UTILS_SYSTEM_H
