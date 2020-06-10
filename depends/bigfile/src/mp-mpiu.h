#ifndef _MPIU_H_
#define _MPIU_H_

typedef void * (*mpiu_malloc_func)(const char * name, size_t size, const char * file, const int line, void * userdata);
typedef void (*mpiu_free_func)(void * ptr, const char * file, const int line, void * userdata);

void * mpiu_malloc(const char * name, size_t size, const char * file, const int line);
void mpiu_free(void * ptr, const char * file, const int line);

void mpiu_set_malloc(mpiu_malloc_func malloc, mpiu_free_func free, void * userdata);
void MPIU_Set_verbose_malloc(MPI_Comm comm);

/*
 * Set the MPIU memory allocator. MPIU uses the allocator to provided a hook for tracking allocations of significance.
 * */
#define MPIU_SetMalloc mpiu_set_malloc
#define MPIU_MallocT(name, type, nmemb) MPIU_Malloc(name, sizeof(type), nmemb, __FILE__, __LINE__)
#define MPIU_Malloc(name, elsize, nmemb) mpiu_malloc(name, ((size_t)(elsize)) * nmemb, __FILE__, __LINE__)
#define MPIU_Free(ptr) mpiu_free(ptr, __FILE__, __LINE__)

/*
 * MPIU_Alltoallv:
 * a Alltoallv can automatically switch to a sparse implementation
 */
enum MPIU_AlltoallvSparsePolicy {
    AUTO = 0,
    DISABLED = 1,
    REQUIRED = 2
};

int MPIU_Alltoallv(void *sendbuf, int *sendcnts, int *sdispls,
        MPI_Datatype sendtype, void *recvbuf, int *recvcnts,
        int *rdispls, MPI_Datatype recvtype, MPI_Comm comm,
        enum MPIU_AlltoallvSparsePolicy policy);

/*
 * Returns the rank that contains the first result matching the MPI_Op.
 * op can be MPI_MIN or MPI_MAX. This function works around potentially buggy
 * MPI_MINLOC and MPI_MAXLOC implemenations.
 * */
int
MPIU_GetLoc(const void * base, MPI_Datatype type, MPI_Op op, MPI_Comm comm);

/*
 * Gathers from all ranks to root. if recvbuffer is NULL, allocate new memory with MPIU_Malloc.
 */
void *
MPIU_Gather (MPI_Comm comm, int root, const void * sendbuffer, void * recvbuffer, int nsend, size_t elsize, int * totalnrecv);

/*
 * Scatter from root to all ranks. if recvbuffer is NULL, allocate new memory with MPIU_Malloc.
 */
void *
MPIU_Scatter (MPI_Comm comm, int root, const void * sendbuffer, void * recvbuffer, int nrecv, size_t elsize, int * totalnsend);

/* Segment a MPI Comm into 'groups', such that distributed data in each group is roughly even.
 * NOTE: this API needs some revision to incorporate some of the downstream behaviors. Currently
 * the internal data structure is directly accessed by downstream.
 * */
typedef struct MPIU_Segmenter {
    /* data model: rank <- segment <- group */
    int Ngroup;
    int Nsegments;
    int GroupID; /* ID of the group of this rank */
    int ThisSegment; /* SegmentID of the local data chunk on this rank*/

    size_t totalsize;
    int segment_start; /* segments responsible in this group */
    int segment_end;

    int is_group_leader;
    int group_leader_rank;
    int segment_leader_rank;
    MPI_Comm Group;  /* communicator for all ranks in the group */
    MPI_Comm Leaders; /* communicator for all ranks by leaders vs nonleaders */
    MPI_Comm Segment; /* communicator for all ranks in this segment */
} MPIU_Segmenter;

size_t
MPIU_Segmenter_collect_sizes(size_t localsize, size_t * sizes, size_t * myoffset, MPI_Comm comm);

/* MPIU_segmenter_init: Create a Segmenter.
 * the total number of items according to both sizes and sizes2 will not
 * exceed the epxected_segsize by too much.
 * */
void
MPIU_Segmenter_init(MPIU_Segmenter * segmenter,
               size_t * sizes,   /* IN: size per rank, used to bound the number of ranks in a group. */
               size_t * sizes2,  /* IN: if given, secondary size used to bound the number of ranks in a group. */
               size_t expected_segsize, /* desired size per segment */
               int Ngroup,  /* number of groups to form. */
               MPI_Comm comm);
void
MPIU_Segmenter_destroy(MPIU_Segmenter * segmenter);

#endif
