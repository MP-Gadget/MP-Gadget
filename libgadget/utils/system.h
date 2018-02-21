#ifndef __UTILS_SYSTEM_H__
#define __UTILS_SYSTEM_H__

/* Note on a 32-bit architecture MPI_LONG may be 32-bit,
 * so these should be MPI_LONG_LONG. But in
 * the future MPI_LONG_LONG may become 128-bit.*/
#define MPI_UINT64 MPI_UNSIGNED_LONG
#define MPI_INT64 MPI_LONG

/* check the version of OPENMP */
#if defined(_OPENMP)
#if _OPENMP < 201107
#error MP-Gadget requires OpenMP >= 3.1 if openmp is enabled. \
       Try to compile without openmp or use a newer compiler (gcc >= 4.7) .
#endif
#endif

#ifdef DEBUG
void catch_abort(int sig);
void catch_fatal(int sig);
void enable_core_dumps_and_fpu_exceptions(void);
#endif

int cluster_get_num_hosts();
int cluster_get_hostid();
double get_physmem_bytes();

double get_random_number(uint64_t id);
void set_random_numbers(int seed);
void sumup_large_ints(int n, int *src, int64_t *res);
void sumup_longs(int n, int64_t *src, int64_t *res);
int64_t count_sum(int64_t countLocal);
//int64_t count_to_offset(int64_t countLocal);

int MPIU_Any(int condition, MPI_Comm comm);
void MPIU_write_pids(char * filename);

int MPI_Alltoallv_smart(void *sendbuf, int *sendcnts, int *sdispls,
        MPI_Datatype sendtype, void *recvbuf, int *recvcnts,
        int *rdispls, MPI_Datatype recvtype, MPI_Comm comm);

int MPI_Alltoallv_sparse(void *sendbuf, int *sendcnts, int *sdispls,
        MPI_Datatype sendtype, void *recvbuf, int *recvcnts,
        int *rdispls, MPI_Datatype recvtype, MPI_Comm comm);

double timediff(double t0, double t1);
double second(void);
size_t sizemax(size_t a, size_t b);

static inline int atomic_fetch_and_add(int * ptr, int value) {
    int k;
#if _OPENMP >= 201107
#pragma omp atomic capture
    {
      k = (*ptr);
      (*ptr)+=value;
    }
#else
#if defined(OPENMP_USE_SPINLOCK) && !defined(__INTEL_COMPILER)
    k = __sync_fetch_and_add(ptr, value);
#else /* non spinlock*/
#pragma omp critical
    {
      k = (*ptr);
      (*ptr)+=value;
    }
#endif
#endif
    return k;
}
static inline int atomic_add_and_fetch(int * ptr, int value) {
    int k;
#if _OPENMP >= 201107
#pragma omp atomic capture
    {
      (*ptr)+=value;
      k = (*ptr);
    }
#else
#ifdef OPENMP_USE_SPINLOCK
    k = __sync_add_and_fetch(ptr, value);
#else /* non spinlock */
#pragma omp critical
    {
      (*ptr)+=value;
      k = (*ptr);
    }
#endif
#endif
    return k;
}

#endif
