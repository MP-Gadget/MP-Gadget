#ifndef __UTILS_SYSTEM_H__
#define __UTILS_SYSTEM_H__

#ifdef DEBUG
void catch_abort(int sig);
void catch_fatal(int sig);
void enable_core_dumps_and_fpu_exceptions(void);
void write_pid_file(void);
#endif

int cluster_get_num_hosts();
int cluster_get_hostid();
double get_physmem_bytes();

double get_random_number(MyIDType id);
void set_random_numbers(void);
void sumup_large_ints(int n, int *src, int64_t *res);
void sumup_longs(int n, int64_t *src, int64_t *res);
int64_t count_sum(int64_t countLocal);
//int64_t count_to_offset(int64_t countLocal);

int MPI_Alltoallv_smart(void *sendbuf, int *sendcnts, int *sdispls,
        MPI_Datatype sendtype, void *recvbuf, int *recvcnts,
        int *rdispls, MPI_Datatype recvtype, MPI_Comm comm);

int MPI_Alltoallv_sparse(void *sendbuf, int *sendcnts, int *sdispls,
        MPI_Datatype sendtype, void *recvbuf, int *recvcnts,
        int *rdispls, MPI_Datatype recvtype, MPI_Comm comm);

double timediff(double t0, double t1);
double second(void);
size_t sizemax(size_t a, size_t b);


#endif
