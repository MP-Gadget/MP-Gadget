#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#include <mpi.h>
#include "mp-mpiu.h"
static void * default_mpiu_malloc_func(const char * name, size_t size, const char * file, const int line, void * userdata) { return malloc(size); }
static void default_mpiu_free_func(void * ptr, const char * file, const int line, void * userdata) { free(ptr); }

static void * verbose_mpiu_malloc_func(const char * name, size_t size, const char * file, const int line, void * userdata) {
    MPI_Comm comm = (MPI_Comm) (intptr_t) userdata;
    int rank;
    MPI_Comm_rank(comm, &rank);
    void * ptr = malloc(size);
    fprintf(stderr, "MPIU_Malloc: T%04d %16p : %s size = %ld, %s:%d\n", rank, ptr, name, size, file, line);
    return ptr;
}

static void verbose_mpiu_free_func(void * ptr, const char * file, const int line, void * userdata) {
    MPI_Comm comm = (MPI_Comm) (intptr_t) userdata;
    int rank;
    MPI_Comm_rank(comm, &rank);
    fprintf(stderr, "MPIU_Free: T%04d %16p : %s:%d\n", rank, ptr, file, line);
    free(ptr);
}

static struct {
    mpiu_malloc_func malloc_func;
    mpiu_free_func free_func;
    void * userdata;
} _MPIUMem = {
    default_mpiu_malloc_func,
    default_mpiu_free_func,
    NULL
};

void
mpiu_set_malloc(mpiu_malloc_func malloc, mpiu_free_func free, void * userdata)
{
    _MPIUMem.malloc_func = malloc;
    _MPIUMem.free_func = free;
    _MPIUMem.userdata = userdata;
}

void
MPIU_Set_verbose_malloc(MPI_Comm comm)
{
    mpiu_set_malloc(verbose_mpiu_malloc_func, verbose_mpiu_free_func, (void*) (intptr_t) comm);
}

void * mpiu_malloc(const char * name, size_t size, const char * file, const int line) {
    return _MPIUMem.malloc_func(name, size, file, line, _MPIUMem.userdata);
}

void mpiu_free(void * ptr, const char * file, const int line) {
    _MPIUMem.free_func(ptr, file, line, _MPIUMem.userdata);
}

/* The following two functions are taken from MP-Gadget. The hope
 * is that when the exchange is sparse posting requests is
 * faster than Alltoall on some implementations. */

static int MPI_Alltoallv_sparse(void *sendbuf, int *sendcnts, int *sdispls,
        MPI_Datatype sendtype, void *recvbuf, int *recvcnts,
        int *rdispls, MPI_Datatype recvtype, MPI_Comm comm);

int MPIU_Alltoallv(void *sendbuf, int *sendcnts, int *sdispls,
        MPI_Datatype sendtype, void *recvbuf, int *recvcnts,
        int *rdispls, MPI_Datatype recvtype, MPI_Comm comm,
        enum MPIU_AlltoallvSparsePolicy policy
)
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
        a_recvcnts = malloc(sizeof(int) * NTask);
        recvcnts = a_recvcnts;
        MPI_Alltoall(sendcnts, 1, MPI_INT,
                     recvcnts, 1, MPI_INT, comm);
    }
    if(recvbuf == NULL) {
        int totalrecv = 0;
        for(i = 0; i < NTask; i ++) {
            totalrecv += recvcnts[i];
        }
        if(a_recvcnts)
            free(a_recvcnts);
        return totalrecv;
    }
    if(sdispls == NULL) {
        a_sdispls = malloc(sizeof(int) * NTask);
        sdispls = a_sdispls;
        sdispls[0] = 0;
        for (i = 1; i < NTask; i++) {
            sdispls[i] = sdispls[i - 1] + sendcnts[i - 1];
        }
    }
    if(rdispls == NULL) {
        a_rdispls = malloc(sizeof(int) * NTask);
        rdispls = a_rdispls;
        rdispls[0] = 0;
        for (i = 1; i < NTask; i++) {
            rdispls[i] = rdispls[i - 1] + recvcnts[i - 1];
        }
    }

    int dense;

    if(policy == AUTO) {
        dense = nn > 128;
        MPI_Allreduce(MPI_IN_PLACE, &dense, 1, MPI_INT, MPI_SUM, comm);
    }
    if(policy == DISABLED) {
        dense = 1;
    }
    if(policy == REQUIRED) {
        dense = 0;
    }

    int ret;
    if(dense != 0) {
        ret = MPI_Alltoallv(sendbuf, sendcnts, sdispls,
                    sendtype, recvbuf,
                    recvcnts, rdispls, recvtype, comm);
    } else {
        ret = MPI_Alltoallv_sparse(sendbuf, sendcnts, sdispls,
                    sendtype, recvbuf,
                    recvcnts, rdispls, recvtype, comm);
    }

    if(a_rdispls)
        free(a_rdispls);
    if(a_sdispls)
        free(a_sdispls);
    if(a_recvcnts)
        free(a_recvcnts);
    return ret;
}

static int MPI_Alltoallv_sparse(void *sendbuf, int *sendcnts, int *sdispls,
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
    MPI_Request *requests = malloc(NTask * 2 * sizeof(MPI_Request));
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
    free(requests);
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

/* Find the rank that has the value of MPI_MIN, or MPI_MAX.
 * If there is degeneracy, return the lower rank.
 * Avoids MPI_MINLOC and MPI_MAXLOC.
 * */
int
MPIU_GetLoc(const void * base, MPI_Datatype type, MPI_Op op, MPI_Comm comm)
{
    ptrdiff_t lb;
    ptrdiff_t elsize;
    MPI_Type_get_extent(type, &lb, &elsize);

    void * tmp = malloc(elsize);
    /* find the result of the reduction. */
    MPI_Allreduce(base, tmp, 1, type, op, comm);

    int ThisTask;
    int NTask;
    MPI_Comm_size(comm, &NTask);
    MPI_Comm_rank(comm, &ThisTask);
    int rank = NTask;
    int ret = -1;
    if (memcmp(base, tmp, elsize) == 0) {
        rank = ThisTask;
    }
    /* find the rank that is the same as the reduction result */
    /* avoid MPI_IN_PLACE, since if we are using this code, we have assumed we are using
     * a crazy MPI impl...
     * */
    MPI_Allreduce(&rank, &ret, 1, MPI_INT, MPI_MIN, comm);
    free(tmp);
    return ret;
}

void *
MPIU_Gather (MPI_Comm comm, int root, const void * sendbuffer, void * recvbuffer, int nsend, size_t elsize, int * totalnrecv)
{
    int NTask;
    int ThisTask;

    MPI_Comm_size(comm, &NTask);
    MPI_Comm_rank(comm, &ThisTask);

    MPI_Datatype dtype;
    MPI_Type_contiguous(elsize, MPI_BYTE, &dtype);
    MPI_Type_commit(&dtype);

    int recvcount[NTask];
    int rdispls[NTask + 1];
    int i;
    MPI_Gather(&nsend, 1, MPI_INT, recvcount, 1, MPI_INT, root, comm);

    rdispls[0] = 0;
    for(i = 1; i <= NTask; i ++) {
        rdispls[i] = rdispls[i - 1] + recvcount[i - 1];
    }

    if(ThisTask == root) {
        if(recvbuffer == NULL)
            recvbuffer = MPIU_Malloc("recvbuffer", elsize, rdispls[NTask]);
        if(totalnrecv)
            *totalnrecv = rdispls[NTask];
    } else {
        if(totalnrecv)
            *totalnrecv = 0;
    }

    MPI_Gatherv(sendbuffer, nsend, dtype, recvbuffer, recvcount, rdispls, dtype, root, comm);

    MPI_Type_free(&dtype);

    return recvbuffer;
}

void *
MPIU_Scatter (MPI_Comm comm, int root, const void * sendbuffer, void * recvbuffer, int nrecv, size_t elsize, int * totalnsend)
{
    int NTask;
    int ThisTask;
    MPI_Comm_size(comm, &NTask);
    MPI_Comm_rank(comm, &ThisTask);

    MPI_Datatype dtype;
    MPI_Type_contiguous(elsize, MPI_BYTE, &dtype);
    MPI_Type_commit(&dtype);

    int sendcount[NTask];
    int sdispls[NTask + 1];
    int i;

    MPI_Gather(&nrecv, 1, MPI_INT, sendcount, 1, MPI_INT, root, comm);

    sdispls[0] = 0;
    for(i = 1; i <= NTask; i ++) {
        sdispls[i] = sdispls[i - 1] + sendcount[i - 1];
    }

    if(recvbuffer == NULL)
        recvbuffer = MPIU_Malloc("recvbuffer", elsize, nrecv);

    if(ThisTask == root) {
        if(totalnsend)
            *totalnsend = sdispls[NTask];
    } else {
        if(totalnsend)
            *totalnsend = 0;
    }
    MPI_Scatterv(sendbuffer, sendcount, sdispls, dtype, recvbuffer, nrecv, dtype, root, comm);

    MPI_Type_free(&dtype);

    return recvbuffer;
}

int
_MPIU_Segmenter_assign_colors(size_t glocalsize, size_t * sizes, size_t * sizes2, int * ncolor, MPI_Comm comm)
{
    int NTask;
    int ThisTask;
    MPI_Comm_rank(comm, &ThisTask);
    MPI_Comm_size(comm, &NTask);

    if (sizes2 == NULL) {
        sizes2 = sizes;
    }

    int i;
    int mycolor = -1;
    size_t current_size = 0;
    size_t current_sizes2 = 0;
    int current_color = 0;
    int lastcolor = 0;
    for(i = 0; i < NTask; i ++) {
        current_size += sizes[i];
        current_sizes2 += sizes2[i];

        lastcolor = current_color;

        if(i == ThisTask) {
            mycolor = lastcolor;
        }

        if(current_size > glocalsize || current_sizes2 > glocalsize) {
            current_size = 0;
            current_sizes2 = 0;
            current_color ++;
        }
    }
    /* no data for color of -1; exclude them later with special cases */
    if(sizes[ThisTask] == 0 && sizes2[ThisTask] == 0) {
        mycolor = -1;
    }

    *ncolor = lastcolor + 1;
    return mycolor;
}

size_t
MPIU_Segmenter_collect_sizes(size_t localsize, size_t * sizes, size_t * myoffset, MPI_Comm comm)
{

    int ThisTask, NTask;

    MPI_Comm_size(comm, &NTask);
    MPI_Comm_rank(comm, &ThisTask);

    size_t totalsize;

    sizes[ThisTask] = localsize;

    MPI_Datatype MPI_PTRDIFFT;

    if(sizeof(ptrdiff_t) == sizeof(long)) {
        MPI_PTRDIFFT = MPI_LONG;
    } else if(sizeof(ptrdiff_t) == sizeof(int)) {
        MPI_PTRDIFFT = MPI_INT;
    } else { abort(); }

    MPI_Allreduce(&sizes[ThisTask], &totalsize, 1, MPI_PTRDIFFT, MPI_SUM, comm);
    MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, sizes, 1, MPI_PTRDIFFT, comm);

    int i;
    *myoffset = 0;
    for(i = 0; i < ThisTask; i ++) {
        (*myoffset) += sizes[i];
    }

    return totalsize;
}

void
MPIU_Segmenter_init(MPIU_Segmenter * segmenter,
               size_t * sizes,
               size_t * sizes2,
               size_t avgsegsize,
               int Ngroup,
               MPI_Comm comm)
{
    int ThisTask, NTask;

    MPI_Comm_size(comm, &NTask);
    MPI_Comm_rank(comm, &ThisTask);

    segmenter->ThisSegment = _MPIU_Segmenter_assign_colors(avgsegsize, sizes, sizes2, &segmenter->Nsegments, comm);

    if(segmenter->ThisSegment >= 0) {
        /* assign segments to groups.
         * if Nsegments < Ngroup, some groups will have no segments, and thus no ranks belong to them. */
        segmenter->GroupID = ((size_t) segmenter->ThisSegment) * Ngroup / segmenter->Nsegments;
    } else {
        segmenter->GroupID = Ngroup + 1;
        segmenter->ThisSegment = NTask + 1;
    }

    segmenter->Ngroup = Ngroup;

    MPI_Comm_split(comm, segmenter->GroupID, ThisTask, &segmenter->Group);

    MPI_Allreduce(&segmenter->ThisSegment, &segmenter->segment_start, 1, MPI_INT, MPI_MIN, segmenter->Group);
    MPI_Allreduce(&segmenter->ThisSegment, &segmenter->segment_end, 1, MPI_INT, MPI_MAX, segmenter->Group);

    segmenter->segment_end ++;

    int rank;

    MPI_Comm_rank(segmenter->Group, &rank);

    /* rank with most data in a group is the leader of the group. */
    segmenter->group_leader_rank = MPIU_GetLoc(&sizes[ThisTask], MPI_LONG, MPI_MAX, segmenter->Group);

    segmenter->is_group_leader = rank == segmenter->group_leader_rank;

    MPI_Comm_split(comm, (rank == segmenter->group_leader_rank)? 0 : 1, ThisTask, &segmenter->Leaders);

    MPI_Comm_split(segmenter->Group, segmenter->ThisSegment, ThisTask, &segmenter->Segment);

    /* rank with least data in a segment is the leader of the segment. */
    segmenter->segment_leader_rank = MPIU_GetLoc(&sizes[ThisTask], MPI_LONG, MPI_MIN, segmenter->Segment);
}

void
MPIU_Segmenter_destroy(MPIU_Segmenter * segmenter)
{
    MPI_Comm_free(&segmenter->Segment);
    MPI_Comm_free(&segmenter->Group);
    MPI_Comm_free(&segmenter->Leaders);
}

