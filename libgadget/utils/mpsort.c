
#include <stdio.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

#include <mpi.h>

#include "mpsort.h"
#include "system.h"
#include "mymalloc.h"
#include "openmpsort.h"
#include "endrun.h"

typedef int (*_compar_fn_t)(const void * r1, const void * r2, size_t rsize);
typedef void (*_bisect_fn_t)(void * r, const void * r1, const void * r2, size_t rsize);

struct crstruct {
    void * base;
    size_t nmemb;
    size_t size;
    size_t rsize;
    void * arg;
    void (*radix)(const void * ptr, void * radix, void * arg);
    _compar_fn_t compar;
    _bisect_fn_t bisect;
};

#define DEFTYPE(type) \
static int _compar_radix_ ## type ( \
        const type * u1,  \
        const type * u2,  \
        size_t junk) { \
    return (signed) (*u1 > *u2) - (signed) (*u1 < *u2); \
} \
static void _bisect_radix_ ## type ( \
        type * u, \
        const type * u1,  \
        const type * u2,  \
        size_t junk) { \
    *u = *u1 + ((*u2 - *u1) >> 1); \
}
DEFTYPE(uint16_t)
DEFTYPE(uint32_t)
DEFTYPE(uint64_t)
static int _compar_radix(const void * r1, const void * r2, size_t rsize, int dir) {
    size_t i;
    /* from most significant */
    const unsigned char * u1 = (const unsigned char *) r1;
    const unsigned char * u2 = (const unsigned char *) r2;
    if(dir < 0) {
        u1 += rsize - 1;
        u2 += rsize - 1;;
    }
    for(i = 0; i < rsize; i ++) {
        if(*u1 < *u2) return -1;
        if(*u1 > *u2) return 1;
        u1 += dir;
        u2 += dir;
    }
    return 0;
}
static int _compar_radix_u8(const void * r1, const void * r2, size_t rsize, int dir) {
    size_t i;
    /* from most significant */
    const uint64_t * u1 = (const uint64_t *) r1;
    const uint64_t * u2 = (const uint64_t *) r2;
    if(dir < 0) {
        u1 = (const uint64_t *) ((const char*) u1 + rsize - 8);
        u2 = (const uint64_t *) ((const char*) u2 + rsize - 8);
    }
    for(i = 0; i < rsize; i += 8) {
        if(*u1 < *u2) return -1;
        if(*u1 > *u2) return 1;
        u1 += dir;
        u2 += dir;
    }
    return 0;
}
static int _compar_radix_le(const void * r1, const void * r2, size_t rsize) {
    return _compar_radix(r1, r2, rsize, -1);
}
static int _compar_radix_be(const void * r1, const void * r2, size_t rsize) {
    return _compar_radix(r1, r2, rsize, +1);
}
static int _compar_radix_le_u8(const void * r1, const void * r2, size_t rsize) {
    return _compar_radix_u8(r1, r2, rsize, -1);
}
static int _compar_radix_be_u8(const void * r1, const void * r2, size_t rsize) {
    return _compar_radix_u8(r1, r2, rsize, +1);
}
static void _bisect_radix(void * r, const void * r1, const void * r2, size_t rsize, int dir) {
    size_t i;
    const unsigned char * u1 = (const unsigned char *) r1;
    const unsigned char * u2 = (const unsigned char *) r2;
    unsigned char * u = (unsigned char *) r;
    unsigned int carry = 0;
    if(dir > 0) {
        u1 += rsize - 1;
        u2 += rsize - 1;
    }
    /* from most significant */
    for(i = 0; i < rsize; i ++) {
        unsigned int tmp = (unsigned int) *u2 + *u1 + carry;
        if(tmp >= 256) carry = 1;
        else carry = 0;
        *u = tmp % (UINT8_MAX+1);
        u -= dir;
        u1 -= dir;
        u2 -= dir;
    }
    u += dir;
    for(i = 0; i < rsize; i ++) {
        unsigned int tmp = *u + carry * 256;
        carry = tmp & 1;
        *u = (tmp >> 1) ;
        u += dir;
    }
}
static void _bisect_radix_le(void * r, const void * r1, const void * r2, size_t rsize) {
    _bisect_radix(r, r1, r2, rsize, -1);
}
static void _bisect_radix_be(void * r, const void * r1, const void * r2, size_t rsize) {
    _bisect_radix(r, r1, r2, rsize, +1);
}

void _setup_radix_sort(
        struct crstruct *d,
        void * base,
        size_t nmemb,
        size_t size,
        void (*radix)(const void * ptr, void * radix, void * arg),
        size_t rsize,
        void * arg) {

    /* Cheers stack overflow*/
    union {
        uint32_t i;
        char c[4];
    } be_detect = {0x01020304};
    d->base = base;
    d->nmemb = nmemb;
    d->rsize = rsize;
    d->arg = arg;
    d->radix = radix;
    d->size = size;
    switch(rsize) {
        case 2:
            d->compar = (_compar_fn_t) _compar_radix_uint16_t;
            d->bisect = (_bisect_fn_t) _bisect_radix_uint16_t;
            break;
        case 4:
            d->compar = (_compar_fn_t) _compar_radix_uint32_t;
            d->bisect = (_bisect_fn_t) _bisect_radix_uint32_t;
            break;
        case 8:
            d->compar = (_compar_fn_t) _compar_radix_uint64_t;
            d->bisect = (_bisect_fn_t) _bisect_radix_uint64_t;
            break;
        default:
            if(be_detect.c[0] != 1) {
                if(rsize % 8 == 0) {
                    d->compar = _compar_radix_le_u8;
                } else{
                    d->compar = _compar_radix_le;
                }
                d->bisect = _bisect_radix_le;
            } else {
                if(rsize % 8 == 0) {
                    d->compar = _compar_radix_be_u8;
                } else{
                    d->compar = _compar_radix_be;
                }
                d->bisect = _bisect_radix_be;
            }
    }
}

/****
 * sort by radix;
 * internally this uses qsort_r of glibc.
 *
 **** */
static struct crstruct _cacr_d;

/* implementation ; internal */
static int _compute_and_compar_radix(const void * p1, const void * p2) {
    char r1[_cacr_d.rsize], r2[_cacr_d.rsize];
    _cacr_d.radix(p1, r1, _cacr_d.arg);
    _cacr_d.radix(p2, r2, _cacr_d.arg);
    int c1 = _cacr_d.compar(r1, r2, _cacr_d.rsize);
    return c1;
}

static void radix_sort(void * base, size_t nmemb, size_t size,
        void (*radix)(const void * ptr, void * radix, void * arg),
        size_t rsize,
        void * arg) {

    memset(&_cacr_d, 0, sizeof(struct crstruct));
    _setup_radix_sort(&_cacr_d, base, nmemb, size, radix, rsize, arg);

    qsort_openmp(_cacr_d.base, _cacr_d.nmemb, _cacr_d.size, _compute_and_compar_radix);
}


/*
 * returns index of the last item satisfying
 * [item] < P,
 *
 * returns -1 if [all] < P
 * */

static ptrdiff_t _bsearch_last_lt(void * P,
    void * base, size_t nmemb,
    struct crstruct * d) {

    if (nmemb == 0) return -1;

    char tmpradix[d->rsize];
    ptrdiff_t left = 0;
    ptrdiff_t right = nmemb - 1;

    d->radix((char*) base, tmpradix, d->arg);
    if(d->compar(tmpradix, P, d->rsize) >= 0) {
        return - 1;
    }
    d->radix((char*) base + right * d->size, tmpradix, d->arg);
    if(d->compar(tmpradix, P, d->rsize) < 0) {
        return nmemb - 1;
    }

    /* left <= i <= right*/
    /* [left] < P <= [right] */
    while(right > left + 1) {
        ptrdiff_t mid = ((right - left + 1) >> 1) + left;
        d->radix((char*) base + mid * d->size, tmpradix, d->arg);
        /* if [mid] < P , move left to mid */
        /* if [mid] >= P , move right to mid */
        int c1 = d->compar(tmpradix, P, d->rsize);
        if(c1 < 0) {
            left = mid;
        } else {
            right = mid;
        }
    }
    return left;
}

/*
 * returns index of the last item satisfying
 * [item] <= P,
 *
 * */
static ptrdiff_t _bsearch_last_le(void * P,
    void * base, size_t nmemb,
    struct crstruct * d) {

    if (nmemb == 0) return -1;

    char tmpradix[d->rsize];
    ptrdiff_t left = 0;
    ptrdiff_t right = nmemb - 1;

    d->radix((char*) base, tmpradix, d->arg);
    if(d->compar(tmpradix, P, d->rsize) > 0) {
        return -1;
    }
    d->radix((char*) base + right * d->size, tmpradix, d->arg);
    if(d->compar(tmpradix, P, d->rsize) <= 0) {
        return nmemb - 1;
    }

    /* left <= i <= right*/
    /* [left] <= P < [right] */
    while(right > left + 1) {
        ptrdiff_t mid = ((right - left + 1) >> 1) + left;
        d->radix((char*) base + mid * d->size, tmpradix, d->arg);
        /* if [mid] <= P , move left to mid */
        /* if [mid] > P , move right to mid*/
        int c1 = d->compar(tmpradix, P, d->rsize);
        if(c1 <= 0) {
            left = mid;
        } else {
            right = mid;
        }
    }
    return left;
}

/*
 * do a histogram of mybase, based on bins defined in P.
 * P is an array of radix of length Plength,
 * myCLT, myCLE are of length Plength + 2
 *
 * myCLT[i + 1] is the count of items less than P[i]
 * myCLE[i + 1] is the count of items less than or equal to P[i]
 *
 * myCLT[0] is always 0
 * myCLT[Plength + 1] is always mynmemb
 *
 * */
static void _histogram(char * P, int Plength, void * mybase, size_t mynmemb,
        ptrdiff_t * myCLT, ptrdiff_t * myCLE,
        struct crstruct * d) {
    int it;

    if(myCLT) {
        myCLT[0] = 0;
        for(it = 0; it < Plength; it ++) {
            /* No need to start from the beginging of mybase, since myubase and P are both sorted */
            ptrdiff_t offset = myCLT[it];
            myCLT[it + 1] = _bsearch_last_lt(P + it * d->rsize,
                            ((char*) mybase) + offset * d->size,
                            mynmemb - offset, d)
                            + 1 + offset;
        }
        myCLT[it + 1] = mynmemb;
    }
    if(myCLE) {
        myCLE[0] = 0;
        for(it = 0; it < Plength; it ++) {
            /* No need to start from the beginging of mybase, since myubase and P are both sorted */
            ptrdiff_t offset = myCLE[it];
            myCLE[it + 1] = _bsearch_last_le(P + it * d->rsize,
                            ((char*) mybase) + offset * d->size,
                            mynmemb - offset, d)
                            + 1 + offset;
        }
        myCLE[it + 1] = mynmemb;
    }
}

struct piter {
    int * stable;
    int * narrow;
    int Plength;
    char * Pleft;
    char * Pright;
    struct crstruct * d;
};
static void piter_init(struct piter * pi,
        char * Pmin, char * Pmax, int Plength,
        struct crstruct * d) {
    pi->stable = ta_malloc("stable", int, Plength);
    memset(pi->stable, 0, Plength * sizeof(int));
    pi->narrow = ta_malloc("narrow", int, Plength);
    memset(pi->narrow, 0, Plength * sizeof(int));
    pi->d = d;
    pi->Pleft = ta_malloc("left", char, Plength * d->rsize);
    memset(pi->Pleft, 0, Plength * d->rsize * sizeof(char));
    pi->Pright = ta_malloc("right", char, Plength * d->rsize);
    memset(pi->Pright, 0, Plength * d->rsize * sizeof(char));
    pi->Plength = Plength;

    int i;
    for(i = 0; i < pi->Plength; i ++) {
        memcpy(&pi->Pleft[i * d->rsize], Pmin, d->rsize);
        memcpy(&pi->Pright[i * d->rsize], Pmax, d->rsize);
    }
}
static void piter_destroy(struct piter * pi) {
    myfree(pi->Pright);
    myfree(pi->Pleft);
    myfree(pi->narrow);
    myfree(pi->stable);
}

/*
 * this will bisect the left / right in piter.
 * note that piter goes [left, right], thus we need
 * to maintain an internal status to make sure we go over
 * the additional 'right]'. (usual bisect range is
 * '[left, right)' )
 * */
static void piter_bisect(struct piter * pi, char * P) {
    struct crstruct * d = pi->d;
    int i;
    for(i = 0; i < pi->Plength; i ++) {
        if(pi->stable[i]) continue;
        if(pi->narrow[i]) {
            /* The last iteration, test Pright directly */
            memcpy(&P[i * d->rsize],
                &pi->Pright[i * d->rsize],
                d->rsize);
            pi->stable[i] = 1;
        } else {
            /* ordinary iteration */
            d->bisect(&P[i * d->rsize],
                    &pi->Pleft[i * d->rsize],
                    &pi->Pright[i * d->rsize], d->rsize);
            /* in case the bisect can't move P beyond left,
             * the range is too small, so we set flag narrow,
             * and next iteration we will directly test Pright */
            if(d->compar(&P[i * d->rsize],
                &pi->Pleft[i * d->rsize], d->rsize) <= 0) {
                pi->narrow[i] = 1;
            }
        }
#if 0
        printf("bisect %d %u %u %u\n", i, *(int*) &P[i * d->rsize],
                *(int*) &pi->Pleft[i * d->rsize],
                *(int*) &pi->Pright[i * d->rsize]);
#endif
    }
}
static int piter_all_done(struct piter * pi) {
    int i;
    int done = 1;
#if 0
#pragma omp single
    for(i = 0; i < pi->Plength; i ++) {
        printf("P %d stable %d narrow %d\n",
            i, pi->stable[i], pi->narrow[i]);
    }
#endif
    for(i = 0; i < pi->Plength; i ++) {
        if(!pi->stable[i]) {
            done = 0;
            break;
        }
    }
    return done;
}

/*
 * bisection acceptance test.
 *
 * test if the counts satisfies CLT < C <= CLE.
 * move Pleft / Pright accordingly.
 * */
static void piter_accept(struct piter * pi, char * P,
        ptrdiff_t * C, ptrdiff_t * CLT, ptrdiff_t * CLE) {
    struct crstruct * d = pi->d;
    int i;
#if 0
    for(i = 0; i < pi->Plength + 1; i ++) {
        printf("counts %d LT %ld C %ld LE %ld\n",
                i, CLT[i], C[i], CLE[i]);
    }
#endif
    for(i = 0; i < pi->Plength; i ++) {
        if( CLT[i + 1] < C[i + 1] && C[i + 1] <= CLE[i + 1]) {
            pi->stable[i] = 1;
            continue;
        } else {
            if(CLT[i + 1] >= C[i + 1]) {
                /* P[i] is too big */
                memcpy(&pi->Pright[i * d->rsize], &P[i * d->rsize], d->rsize);
            } else {
                /* P[i] is too small */
                memcpy(&pi->Pleft[i * d->rsize], &P[i * d->rsize], d->rsize);
            }
        }
    }
}

static int _mpsort_mpi_options = 0;

/* mpi version of radix sort;
 *
 * each caller provides the distributed array and number of items.
 * the sorted array is returned to the original array pointed to by
 * mybase. (AKA no rebalancing is done.)
 *
 * NOTE: may need an api to return a balanced array!
 *
 * uses the same amount of temporary storage space for communication
 * and local sort. (this will be allocated via malloc)
 *
 *
 * */

static MPI_Datatype MPI_TYPE_PTRDIFF = 0;

struct crmpistruct {
    MPI_Datatype MPI_TYPE_RADIX;
    MPI_Datatype MPI_TYPE_DATA;
    MPI_Comm comm;
    void * mybase;
    void * myoutbase;
    size_t mynmemb;
    size_t nmemb;
    size_t myoutnmemb;
    size_t outnmemb;
    int NTask;
    int ThisTask;
};

static void
_setup_mpsort_mpi(struct crmpistruct * o,
                  struct crstruct * d,
                  void * myoutbase, size_t myoutnmemb,
                  MPI_Comm comm)
{

    o->comm = comm;

    MPI_Comm_size(comm, &o->NTask);
    MPI_Comm_rank(comm, &o->ThisTask);

    o->mybase = d->base;
    o->mynmemb = d->nmemb;
    o->myoutbase = myoutbase;
    o->myoutnmemb = myoutnmemb;

    MPI_Allreduce(&o->mynmemb, &o->nmemb, 1, MPI_TYPE_PTRDIFF, MPI_SUM, comm);
    MPI_Allreduce(&o->myoutnmemb, &o->outnmemb, 1, MPI_TYPE_PTRDIFF, MPI_SUM, comm);

    if(o->outnmemb != o->nmemb) {
        endrun(4, "total number of items in the item does not match the input %ld != %ld\n", o->outnmemb, o->nmemb);
    }

    MPI_Type_contiguous(d->rsize, MPI_BYTE, &o->MPI_TYPE_RADIX);
    MPI_Type_commit(&o->MPI_TYPE_RADIX);

    MPI_Type_contiguous(d->size, MPI_BYTE, &o->MPI_TYPE_DATA);
    MPI_Type_commit(&o->MPI_TYPE_DATA);

}
static void _destroy_mpsort_mpi(struct crmpistruct * o) {
    MPI_Type_free(&o->MPI_TYPE_RADIX);
    MPI_Type_free(&o->MPI_TYPE_DATA);
}

static void _find_Pmax_Pmin_C(void * mybase, size_t mynmemb,
        size_t myoutnmemb,
        char * Pmax, char * Pmin,
        ptrdiff_t * C,
        struct crstruct * d,
        struct crmpistruct * o);

static int _solve_for_layout_mpi (
        int NTask,
        ptrdiff_t * C,
        ptrdiff_t * myT_CLT,
        ptrdiff_t * myT_CLE,
        ptrdiff_t * myT_C,
        MPI_Comm comm);

struct TIMER {
    double time;
    char name[20];
};

static struct TIMERS {
    struct TIMER * tmr;
    int curtmr;
    int ntimer;
} _TIMERS;

static int
_assign_colors(size_t glocalsize, size_t * sizes, int * ncolor, MPI_Comm comm)
{
    int NTask;
    int ThisTask;
    MPI_Comm_rank(comm, &ThisTask);
    MPI_Comm_size(comm, &NTask);

    int i;
    int mycolor = -1;
    size_t current_size = 0;
    size_t current_outsize = 0;
    int current_color = 0;
    int lastcolor = 0;
    for(i = 0; i < NTask; i ++) {
        current_size += sizes[i];

        lastcolor = current_color;

        if(i == ThisTask) {
            mycolor = lastcolor;
        }

        if(current_size > glocalsize || current_outsize > glocalsize) {
            current_size = 0;
            current_outsize = 0;
            current_color ++;
        }
    }
    /* no data for color of -1; exclude them later with special cases */
    if(sizes[ThisTask] == 0) {
        mycolor = -1;
    }

    *ncolor = lastcolor + 1;
    return mycolor;
}

static size_t
_collect_sizes(size_t localsize, size_t * sizes, size_t * myoffset, MPI_Comm comm)
{

    int ThisTask, NTask;

    MPI_Comm_size(comm, &NTask);
    MPI_Comm_rank(comm, &ThisTask);

    size_t totalsize;

    sizes[ThisTask] = localsize;

    MPI_Allreduce(&sizes[ThisTask], &totalsize, 1, MPI_TYPE_PTRDIFF, MPI_SUM, comm);
    MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, sizes, 1, MPI_TYPE_PTRDIFF, comm);

    int i;
    *myoffset = 0;
    for(i = 0; i < ThisTask; i ++) {
        (*myoffset) += sizes[i];
    }

    return totalsize;
}

struct SegmentGroupDescr {
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
};

/* Find the rank that has the value of MPI_MIN, or MPI_MAX.
 * If there is degeneracy, return the lower rank.
 * Avoids MPI_MINLOC and MPI_MAXLOC.
 * */
static int
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

static void
_create_segment_group(struct SegmentGroupDescr * descr, size_t * sizes, size_t avgsegsize, int Ngroup, MPI_Comm comm)
{
    int ThisTask, NTask;

    MPI_Comm_size(comm, &NTask);
    MPI_Comm_rank(comm, &ThisTask);

    descr->ThisSegment = _assign_colors(avgsegsize, sizes, &descr->Nsegments, comm);

    if(descr->ThisSegment >= 0) {
        /* assign segments to groups.
         * if Nsegments < Ngroup, some groups will have no segments, and thus no ranks belong to them. */
        descr->GroupID = ((size_t) descr->ThisSegment) * Ngroup / descr->Nsegments;
    } else {
        descr->GroupID = Ngroup + 1;
        descr->ThisSegment = NTask + 1;
    }

    descr->Ngroup = Ngroup;

    MPI_Comm_split(comm, descr->GroupID, ThisTask, &descr->Group);

    MPI_Allreduce(&descr->ThisSegment, &descr->segment_start, 1, MPI_INT, MPI_MIN, descr->Group);
    MPI_Allreduce(&descr->ThisSegment, &descr->segment_end, 1, MPI_INT, MPI_MAX, descr->Group);

    descr->segment_end ++;

    int rank;

    MPI_Comm_rank(descr->Group, &rank);

    /* rank with most data in a group is the leader of the group. */
    descr->group_leader_rank = MPIU_GetLoc(&sizes[ThisTask], MPI_LONG, MPI_MAX, descr->Group);

    descr->is_group_leader = rank == descr->group_leader_rank;

    MPI_Comm_split(comm, (rank == descr->group_leader_rank)? 0 : 1, ThisTask, &descr->Leaders);

    MPI_Comm_split(descr->Group, descr->ThisSegment, ThisTask, &descr->Segment);

    /* rank with least data in a segment is the leader of the segment. */
    descr->segment_leader_rank = MPIU_GetLoc(&sizes[ThisTask], MPI_LONG, MPI_MIN, descr->Segment);
}

static void
_destroy_segment_group(struct SegmentGroupDescr * descr)
{
    MPI_Comm_free(&descr->Segment);
    MPI_Comm_free(&descr->Group);
    MPI_Comm_free(&descr->Leaders);
}

static void
mpsort_increment_timer(const char * name, int erase)
{
    if(!(_TIMERS.tmr))
        return;
    struct TIMER * tmr = _TIMERS.tmr+_TIMERS.curtmr;
    if(erase && _TIMERS.curtmr > 0)
        _TIMERS.curtmr --;
    tmr->time = MPI_Wtime();
    strncpy(tmr->name, name, 19);
    tmr->name[19] = '\0';
    _TIMERS.curtmr++;
}

void mpsort_setup_timers(int ntimers)
{
    if(!(_TIMERS.tmr)) {
        _TIMERS.tmr = (struct TIMER *) mymalloc2("timers", ntimers * sizeof(struct TIMER));
        _TIMERS.ntimer = ntimers;
        _TIMERS.curtmr = 0;
    }
}

void mpsort_free_timers(void)
{
    if(!(_TIMERS.tmr)) {
        myfree(_TIMERS.tmr);
        _TIMERS.tmr = NULL;
        _TIMERS.ntimer = 0;
        _TIMERS.curtmr = 0;
    }
}


void mpsort_mpi_report_last_run() {
    if(!(_TIMERS.tmr))
        return;
    double last = _TIMERS.tmr[0].time;
    int i;
    for(i = 1; i < _TIMERS.curtmr; i++) {
        struct TIMER * tmr = &_TIMERS.tmr[i];
        if(0 == strncmp(tmr->name, "END", 20))
            break;
        message(0, "%s: %g\n", tmr->name, tmr->time - last);
        last = tmr->time;
    }
}

int mpsort_mpi_find_ntimers(struct TIMERS * timers) {
    return _TIMERS.curtmr;
}

void
mpsort_mpi_impl (void * mybase, size_t mynmemb, size_t size,
        void (*radix)(const void * ptr, void * radix, void * arg),
        size_t rsize,
        void * arg,
        MPI_Comm comm,
        const int line, const char * file)
{
    mpsort_mpi_newarray_impl(mybase, mynmemb,
        mybase, mynmemb,
        size, radix, rsize, arg, comm, line, file);
}

static int
mpsort_mpi_histogram_sort(struct crstruct d, struct crmpistruct o);

static void
MPIU_Scatter (MPI_Comm comm, int root, const void * sendbuffer, void * recvbuffer, int nrecv, size_t elsize, int * totalnsend);
static void
MPIU_Gather (MPI_Comm comm, int root, const void * sendbuffer, void * recvbuffer, int nsend, size_t elsize, int * totalnrecv);

static uint64_t
checksum(void * base, size_t nbytes, MPI_Comm comm)
{
    uint64_t sum = 0;
    char * ptr = (char *) base;
    size_t i;
    for(i = 0; i < nbytes; i ++) {
        sum += ptr[i];
    }
    MPI_Allreduce(MPI_IN_PLACE, &sum, 1, MPI_LONG, MPI_SUM, comm);
    return sum;
}

void
mpsort_mpi_newarray_impl (void * mybase, size_t mynmemb,
        void * myoutbase, size_t myoutnmemb,
        size_t elsize,
        void (*radix)(const void * ptr, void * radix, void * arg),
        size_t rsize,
        void * arg,
        MPI_Comm comm,
        const int line,
        const char * file)
{
    if(MPI_TYPE_PTRDIFF == 0) {
        if(MPI_SUCCESS != MPI_Type_match_size(MPI_TYPECLASS_INTEGER, sizeof(ptrdiff_t), &MPI_TYPE_PTRDIFF))
            endrun(3, "Ptrdiff size %ld not recognised\n", sizeof(ptrdiff_t));
    }

    struct SegmentGroupDescr seggrp[1];

    uint64_t sum1 = checksum(mybase, elsize * mynmemb, comm);

    int NTask;
    int ThisTask;
    MPI_Comm_size(comm, &NTask);
    MPI_Comm_rank(comm, &ThisTask);

    if(elsize > 8 && elsize % 8 != 0) {
        if(ThisTask == 0) {
            endrun(12, "MPSort: element size is large (%ld) but not aligned to 8 bytes. "
                            "This is known to frequently trigger MPI bugs. "
                            "Caller site: %s:%d\n",
                            elsize, file, line);
        }
    }
    if(rsize > 8 && rsize % 8 != 0) {
        if(ThisTask == 0) {
            endrun(12, "MPSort: radix size is large (%ld) but not aligned to 8 bytes. "
                            "This is known to frequently trigger MPI bugs. "
                            "Caller site: %s:%d\n",
                            rsize, file, line);
        }
    }

    size_t * sizes = ta_malloc("sizes", size_t, NTask);
    size_t myoffset;
    size_t totalsize = _collect_sizes(mynmemb, sizes, &myoffset, comm);

    size_t avgsegsize = NTask; /* combine very small ranks to segments */
    if (avgsegsize * elsize > 4 * 1024 * 1024) {
        /* do not use more than 4MB in a segment */
        avgsegsize = 4 * 1024 * 1024 / elsize;
    }
    if(mpsort_mpi_has_options(MPSORT_REQUIRE_GATHER_SORT)) {
        message(0, "MPSort: gathering all data to a single rank for sorting due to MPSORT_REQUIRE_GATHER_SORT. "
                   "Total number of items is %ld. Caller site: %s:%d\n",
                   totalsize, file, line);
        avgsegsize = totalsize;
    }

    if(mpsort_mpi_has_options(MPSORT_DISABLE_GATHER_SORT)) {
        avgsegsize = 0;
        message(0, "MPSort: disable gathering data into larger chunks due to MPSORT_DISABLE_GATHER_SORT. "
                   "Caller site: %s:%d\n",
                   file, line);
    }

    /* use as many groups as possible (some will be empty) but at most 1 segment per group */
    _create_segment_group(seggrp, sizes, avgsegsize, NTask, comm);

    myfree(sizes);
    /* group comm == seg comm */

    void * mysegmentbase = NULL;
    void * myoutsegmentbase = NULL;
    size_t mysegmentnmemb;
    size_t myoutsegmentnmemb;

    int groupsize;
    int grouprank;
    MPI_Comm_size(seggrp->Group, &groupsize);
    MPI_Comm_rank(seggrp->Group, &grouprank);

    MPI_Allreduce(&mynmemb, &mysegmentnmemb, 1, MPI_TYPE_PTRDIFF, MPI_SUM, seggrp->Group);
    MPI_Allreduce(&myoutnmemb, &myoutsegmentnmemb, 1, MPI_TYPE_PTRDIFF, MPI_SUM, seggrp->Group);

    if (groupsize > 1) {
        if(grouprank == seggrp->group_leader_rank) {
            mysegmentbase = mymalloc("segmentbase", mysegmentnmemb * elsize);
            myoutsegmentbase = mymalloc("outsegment", myoutsegmentnmemb * elsize);
        }
        MPIU_Gather(seggrp->Group, seggrp->group_leader_rank, mybase, mysegmentbase, mynmemb, elsize, NULL);
    } else {
        mysegmentbase = mybase;
        myoutsegmentbase = myoutbase;
    }

    /* only do sorting on the group leaders for each segment */
    if(seggrp->is_group_leader) {

        struct crstruct d;
        struct crmpistruct o;

        _setup_radix_sort(&d, mysegmentbase, mysegmentnmemb, elsize, radix, rsize, arg);

        _setup_mpsort_mpi(&o, &d, myoutsegmentbase, myoutsegmentnmemb, seggrp->Leaders);

        mpsort_mpi_histogram_sort(d, o);

        _destroy_mpsort_mpi(&o);
    }

    if(groupsize > 1) {
        MPIU_Scatter(seggrp->Group, seggrp->group_leader_rank, myoutsegmentbase, myoutbase, myoutnmemb, elsize, NULL);
    }

/*    {
        int ntmr;
        if(seggrp->is_group_leader)
            ntmr = (mpsort_mpi_find_ntimers(tmr) + 1);

        MPI_Bcast(&ntmr, 1, MPI_INT, seggrp->group_leader_rank, seggrp->Group);
        MPI_Bcast(tmr, sizeof(tmr[0]) * ntmr, MPI_BYTE, seggrp->group_leader_rank, seggrp->Group);
    }*/

    if(grouprank == seggrp->group_leader_rank) {
        if(myoutsegmentbase != myoutbase)
            myfree(myoutsegmentbase);
        if(mysegmentbase != mybase)
            myfree(mysegmentbase);
    }

    _destroy_segment_group(seggrp);

    uint64_t sum2 = checksum(myoutbase, elsize * myoutnmemb, comm);
    if (sum1 != sum2) {
        endrun(5, "Data changed after sorting; checksum mismatch.\n");
    }
}

static void
MPIU_Gather (MPI_Comm comm, int root, const void * sendbuffer, void * recvbuffer, int nsend, size_t elsize, int * totalnrecv)
{
    int NTask;
    int ThisTask;

    MPI_Comm_size(comm, &NTask);
    MPI_Comm_rank(comm, &ThisTask);

    MPI_Datatype dtype;
    MPI_Type_contiguous(elsize, MPI_BYTE, &dtype);
    MPI_Type_commit(&dtype);

    int * recvcount = ta_malloc("recvcount", int, NTask);
    int * rdispls = ta_malloc("rdispls", int, NTask+1);
    int i;
    MPI_Gather(&nsend, 1, MPI_INT, recvcount, 1, MPI_INT, root, comm);

    rdispls[0] = 0;
    for(i = 1; i <= NTask; i ++) {
        rdispls[i] = rdispls[i - 1] + recvcount[i - 1];
    }

    if(ThisTask == root) {
        if(totalnrecv)
            *totalnrecv = rdispls[NTask];
    } else {
        if(totalnrecv)
            *totalnrecv = 0;
    }

    MPI_Gatherv(sendbuffer, nsend, dtype, recvbuffer, recvcount, rdispls, dtype, root, comm);

    ta_free(rdispls);
    ta_free(recvcount);
    MPI_Type_free(&dtype);
}

static void
MPIU_Scatter (MPI_Comm comm, int root, const void * sendbuffer, void * recvbuffer, int nrecv, size_t elsize, int * totalnsend)
{
    int NTask;
    int ThisTask;
    MPI_Comm_size(comm, &NTask);
    MPI_Comm_rank(comm, &ThisTask);

    MPI_Datatype dtype;
    MPI_Type_contiguous(elsize, MPI_BYTE, &dtype);
    MPI_Type_commit(&dtype);

    int * sendcount = ta_malloc("sendcount", int, NTask);
    int * sdispls = ta_malloc("sdispls", int, NTask+1);
    int i;

    MPI_Gather(&nrecv, 1, MPI_INT, sendcount, 1, MPI_INT, root, comm);

    sdispls[0] = 0;
    for(i = 1; i <= NTask; i ++) {
        sdispls[i] = sdispls[i - 1] + sendcount[i - 1];
    }

    if(ThisTask == root) {
        if(totalnsend)
            *totalnsend = sdispls[NTask];
    } else {
        if(totalnsend)
            *totalnsend = 0;
    }
    MPI_Scatterv(sendbuffer, sendcount, sdispls, dtype, recvbuffer, nrecv, dtype, root, comm);

    ta_free(sdispls);
    ta_free(sendcount);
    MPI_Type_free(&dtype);
}

static int
mpsort_mpi_histogram_sort(struct crstruct d, struct crmpistruct o)
{
    ptrdiff_t * myC = (ptrdiff_t *) mymalloc("myhistC", (o.NTask + 1) * sizeof(ptrdiff_t));

    /* Desired counts*/
    ptrdiff_t * C = (ptrdiff_t *) mymalloc("histC", (o.NTask + 1) * sizeof(ptrdiff_t));
    /* counts of less than P */
    ptrdiff_t * myCLT = (ptrdiff_t *) mymalloc("myhistC", (o.NTask + 1) * sizeof(ptrdiff_t));
    ptrdiff_t * CLT = (ptrdiff_t *) mymalloc("histCLT", (o.NTask + 1) * sizeof(ptrdiff_t));
    /* counts of less than or equal to P */
    ptrdiff_t * myCLE = (ptrdiff_t *) mymalloc("myhistCLE", (o.NTask + 1) * sizeof(ptrdiff_t));
    ptrdiff_t * CLE = (ptrdiff_t *) mymalloc("CLE", (o.NTask + 1) * sizeof(ptrdiff_t));

    int iter = 0;
    int done = 0;
    char * buffer;
    int i;

    mpsort_increment_timer("START", 0);

    /* and sort the local array */
    radix_sort(d.base, d.nmemb, d.size, d.radix, d.rsize, d.arg);

    MPI_Barrier(o.comm);

    mpsort_increment_timer("FirstSort", 0);

    char * P = ta_malloc("PP", char, d.rsize * (o.NTask - 1));
    memset(P, 0, d.rsize * (o.NTask -1));

    char Pmax[d.rsize];
    char Pmin[d.rsize];

    _find_Pmax_Pmin_C(o.mybase, o.mynmemb, o.myoutnmemb, Pmax, Pmin, C, &d, &o);

    mpsort_increment_timer("PmaxPmin", 0);

    struct piter pi;

    piter_init(&pi, Pmin, Pmax, o.NTask - 1, &d);

    while(!done) {
        iter ++;
        piter_bisect(&pi, P);

        _histogram(P, o.NTask - 1, o.mybase, o.mynmemb, myCLT, myCLE, &d);

        MPI_Allreduce(myCLT, CLT, o.NTask + 1,
                MPI_TYPE_PTRDIFF, MPI_SUM, o.comm);
        MPI_Allreduce(myCLE, CLE, o.NTask + 1,
                MPI_TYPE_PTRDIFF, MPI_SUM, o.comm);

        char bisectnum[20];
        snprintf(bisectnum, 20, "bisect%04d", iter);
        mpsort_increment_timer(bisectnum, iter > 10);

        piter_accept(&pi, P, C, CLT, CLE);
#if 0
        {
            int k;
            for(k = 0; k < o.NTask; k ++) {
                MPI_Barrier(o.comm);
                int i;
                if(o.ThisTask != k) continue;

                printf("P (%d): PMin %d PMax %d P ",
                        o.ThisTask,
                        *(int*) Pmin,
                        *(int*) Pmax
                        );
                for(i = 0; i < o.NTask - 1; i ++) {
                    printf(" %d ", ((int*) P) [i]);
                }
                printf("\n");

                printf("C (%d): ", o.ThisTask);
                for(i = 0; i < o.NTask + 1; i ++) {
                    printf("%ld ", C[i]);
                }
                printf("\n");
                printf("CLT (%d): ", o.ThisTask);
                for(i = 0; i < o.NTask + 1; i ++) {
                    printf("%ld ", CLT[i]);
                }
                printf("\n");
                printf("CLE (%d): ", o.ThisTask);
                for(i = 0; i < o.NTask + 1; i ++) {
                    printf("%ld ", CLE[i]);
                }
                printf("\n");

            }
        }
#endif
        done = piter_all_done(&pi);
    }

    piter_destroy(&pi);

    _histogram(P, o.NTask - 1, o.mybase, o.mynmemb, myCLT, myCLE, &d);

    ta_free(P);

    mpsort_increment_timer("findP", 0);

    ptrdiff_t * myT_C = (ptrdiff_t *) mymalloc("myhistT_C", (o.NTask) * sizeof(ptrdiff_t));
    ptrdiff_t * myT_CLT = (ptrdiff_t *) mymalloc("myhistCLT", (o.NTask) * sizeof(ptrdiff_t));
    ptrdiff_t * myT_CLE = (ptrdiff_t *) mymalloc("myhistCLE", (o.NTask) * sizeof(ptrdiff_t));

    /* transpose the matrix, could have been done with a new datatype */
    /*
    MPI_Alltoall(myCLT, 1, MPI_TYPE_PTRDIFF,
            myT_CLT, 1, MPI_TYPE_PTRDIFF, o.comm);
    */
    MPI_Alltoall(myCLT + 1, 1, MPI_TYPE_PTRDIFF,
            myT_CLT, 1, MPI_TYPE_PTRDIFF, o.comm);

    /*MPI_Alltoall(myCLE, 1, MPI_TYPE_PTRDIFF,
            myT_CLE, 1, MPI_TYPE_PTRDIFF, o.comm); */
    MPI_Alltoall(myCLE + 1, 1, MPI_TYPE_PTRDIFF,
            myT_CLE, 1, MPI_TYPE_PTRDIFF, o.comm);

    mpsort_increment_timer("LayDistr", 0);

    _solve_for_layout_mpi(o.NTask, C, myT_CLT, myT_CLE, myT_C, o.comm);

    myfree(myT_CLE);
    myfree(myT_CLT);

    myC[0] = 0;
    MPI_Alltoall(myT_C, 1, MPI_TYPE_PTRDIFF,
            myC + 1, 1, MPI_TYPE_PTRDIFF, o.comm);

    myfree(myT_C);

#if 0
    for(i = 0;i < o.NTask; i ++) {
        int j;
        MPI_Barrier(o.comm);
        if(o.ThisTask != i) continue;
        for(j = 0; j < o.NTask + 1; j ++) {
            printf("%d %d %d, ",
                    myCLT[j],
                    myC[j],
                    myCLE[j]);
        }
        printf("\n");

    }
#endif


    /* Desired counts*/
    myfree(CLE);
    myfree(myCLE);
    myfree(CLT);
    myfree(myCLT);
    myfree(C);

    int * SendCount = ta_malloc("SendCount", int, o.NTask);
    int * SendDispl = ta_malloc("SendDispl", int, o.NTask);
    int * RecvCount = ta_malloc("RecvCount", int, o.NTask);
    int * RecvDispl = ta_malloc("RecvDispl", int, o.NTask);

    mpsort_increment_timer("LaySolve", 0);

    for(i = 0; i < o.NTask; i ++) {
        SendCount[i] = myC[i + 1] - myC[i];
    }

    MPI_Alltoall(SendCount, 1, MPI_INT,
            RecvCount, 1, MPI_INT, o.comm);

    SendDispl[0] = 0;
    RecvDispl[0] = 0;
    size_t totrecv = RecvCount[0];
    for(i = 1; i < o.NTask; i ++) {
        SendDispl[i] = SendDispl[i - 1] + SendCount[i - 1];
        RecvDispl[i] = RecvDispl[i - 1] + RecvCount[i - 1];
        if(SendDispl[i] != myC[i]) {
            endrun(7, "SendDispl error\n");
        }
        totrecv += RecvCount[i];
    }
    if(totrecv != o.myoutnmemb) {
        endrun(8, "totrecv = %td, mismatch with %td\n", totrecv, o.myoutnmemb);
    }
#if 0
    {
        int k;
        for(k = 0; k < o.NTask; k ++) {
            MPI_Barrier(o.comm);

            if(o.ThisTask != k) continue;

            printf("P (%d): ", o.ThisTask);
            for(i = 0; i < o.NTask - 1; i ++) {
                printf("%d ", ((int*) P) [i]);
            }
            printf("\n");

            printf("C (%d): ", o.ThisTask);
            for(i = 0; i < o.NTask + 1; i ++) {
                printf("%d ", C[i]);
            }
            printf("\n");
            printf("CLT (%d): ", o.ThisTask);
            for(i = 0; i < o.NTask + 1; i ++) {
                printf("%d ", CLT[i]);
            }
            printf("\n");
            printf("CLE (%d): ", o.ThisTask);
            for(i = 0; i < o.NTask + 1; i ++) {
                printf("%d ", CLE[i]);
            }
            printf("\n");

            printf("MyC (%d): ", o.ThisTask);
            for(i = 0; i < o.NTask + 1; i ++) {
                printf("%d ", myC[i]);
            }
            printf("\n");
            printf("MyCLT (%d): ", o.ThisTask);
            for(i = 0; i < o.NTask + 1; i ++) {
                printf("%d ", myCLT[i]);
            }
            printf("\n");

            printf("MyCLE (%d): ", o.ThisTask);
            for(i = 0; i < o.NTask + 1; i ++) {
                printf("%d ", myCLE[i]);
            }
            printf("\n");

            printf("Send Count(%d): ", o.ThisTask);
            for(i = 0; i < o.NTask; i ++) {
                printf("%d ", SendCount[i]);
            }
            printf("\n");
            printf("My data(%d): ", o.ThisTask);
            for(i = 0; i < mynmemb; i ++) {
                printf("%d ", ((int*) mybase)[i]);
            }
            printf("\n");
        }
    }
#endif
    if(o.myoutbase == o.mybase)
        buffer = (char *) mymalloc("mpsortbuffer", d.size * o.myoutnmemb);
    else
        buffer = (char *) o.myoutbase;

    MPI_Alltoallv_smart(
            o.mybase, SendCount, SendDispl, o.MPI_TYPE_DATA,
            buffer, RecvCount, RecvDispl, o.MPI_TYPE_DATA,
            o.comm);

    if(o.myoutbase == o.mybase) {
        memcpy(o.myoutbase, buffer, o.myoutnmemb * d.size);
        myfree(buffer);
    }

    myfree(RecvDispl);
    myfree(RecvCount);
    myfree(SendDispl);
    myfree(SendCount);

    myfree(myC);
    MPI_Barrier(o.comm);
    mpsort_increment_timer("Exchange", 0);

    radix_sort(o.myoutbase, o.myoutnmemb, d.size, d.radix, d.rsize, d.arg);

    MPI_Barrier(o.comm);

    mpsort_increment_timer("SecondSort", 0);

    mpsort_increment_timer("End", 0);

    return 0;
}

static void _find_Pmax_Pmin_C(void * mybase, size_t mynmemb,
        size_t myoutnmemb,
        char * Pmax, char * Pmin,
        ptrdiff_t * C,
        struct crstruct * d,
        struct crmpistruct * o) {
    memset(Pmax, 0, d->rsize);
    memset(Pmin, -1, d->rsize);

    char myPmax[d->rsize];
    char myPmin[d->rsize];

    size_t * eachnmemb = ta_malloc("eachnmemb", size_t, o->NTask);
    size_t * eachoutnmemb = ta_malloc("eachoutnmemb", size_t, o->NTask);
    char * eachPmax = (char *) mymalloc("eachPmax", d->rsize * o->NTask * sizeof(char));
    char * eachPmin = (char *) mymalloc("eachPmin", d->rsize * o->NTask * sizeof(char));
    int i;

    if(mynmemb > 0) {
        d->radix((char*) mybase + (mynmemb - 1) * d->size, myPmax, d->arg);
        d->radix(mybase, myPmin, d->arg);
    } else {
        memset(myPmin, 0, d->rsize);
        memset(myPmax, 0, d->rsize);
    }

    MPI_Allgather(&mynmemb, 1, MPI_TYPE_PTRDIFF,
            eachnmemb, 1, MPI_TYPE_PTRDIFF, o->comm);
    MPI_Allgather(&myoutnmemb, 1, MPI_TYPE_PTRDIFF,
            eachoutnmemb, 1, MPI_TYPE_PTRDIFF, o->comm);
    MPI_Allgather(myPmax, 1, o->MPI_TYPE_RADIX,
            eachPmax, 1, o->MPI_TYPE_RADIX, o->comm);
    MPI_Allgather(myPmin, 1, o->MPI_TYPE_RADIX,
            eachPmin, 1, o->MPI_TYPE_RADIX, o->comm);


    C[0] = 0;
    for(i = 0; i < o->NTask; i ++) {
        C[i + 1] = C[i] + eachoutnmemb[i];
        if(eachnmemb[i] == 0) continue;

        if(d->compar(eachPmax + i * d->rsize, Pmax, d->rsize) > 0) {
            memcpy(Pmax, eachPmax + i * d->rsize, d->rsize);
        }
        if(d->compar(eachPmin + i * d->rsize, Pmin, d->rsize) < 0) {
            memcpy(Pmin, eachPmin + i * d->rsize, d->rsize);
        }
    }

    myfree(eachPmin);
    myfree(eachPmax);
    myfree(eachoutnmemb);
    myfree(eachnmemb);
}

static int
_solve_for_layout_mpi (
        int NTask,
        ptrdiff_t * C,
        ptrdiff_t * myT_CLT,
        ptrdiff_t * myT_CLE,
        ptrdiff_t * myT_C,
        MPI_Comm comm) {
    int i, j;
    int ThisTask;
    MPI_Comm_rank(comm, &ThisTask);

    /* first assume we just send according to myT_CLT */
    for(i = 0; i < NTask; i ++) {
        myT_C[i] = myT_CLT[i];
    }

    /* Solve for each receiving task i
     *
     * this solves for GL_C[..., i + 1], which depends on GL_C[..., i]
     *
     * and we have GL_C[..., 0] == 0 by definition.
     *
     * this cannot be done in parallel wrt i because of the dependency.
     *
     *  a solution is guaranteed because GL_CLE and GL_CLT
     *  brackes the total counts C (we've found it with the
     *  iterative counting.
     *
     * */

    ptrdiff_t sure = 0;

    /* how many will I surely receive? */
    for(j = 0; j < NTask; j ++) {
        ptrdiff_t recvcount = myT_C[j];
        sure += recvcount;
    }
    /* let's see if we have enough */
    ptrdiff_t deficit = C[ThisTask + 1] - sure;

    for(j = 0; j < NTask; j ++) {
        /* deficit solved */
        if(deficit == 0) break;
        if(deficit < 0) {
            endrun(10, "More items than there should be at j=%d: deficit=%ld\n (C: %ld sure %ld)", j, deficit, C[ThisTask+1], sure);
        }
        /* how much task j can supply ? */
        ptrdiff_t supply = myT_CLE[j] - myT_C[j];
        if(supply < 0) {
            endrun(10, "Less items than there should be at j=%d: supply=%ld (myTCLE %ld myTC %ld)\n", j, supply, myT_CLE[j], myT_C[j]);
        }
        if(supply <= deficit) {
            myT_C[j] += supply;
            deficit -= supply;
        } else {
            myT_C[j] += deficit;
            deficit = 0;
        }
    }

    return 0;
}


static void _mpsort_mpi_parse_env()
{
    static int _mpsort_env_parsed = 0;
    if(_mpsort_env_parsed) return;

    _mpsort_env_parsed = 1;
    if(getenv("MPSORT_DISABLE_GATHER_SORT"))
        mpsort_mpi_set_options(MPSORT_DISABLE_GATHER_SORT);
    if(getenv("MPSORT_REQUIRE_GATHER_SORT "))
        mpsort_mpi_set_options(MPSORT_REQUIRE_GATHER_SORT );
}

void
mpsort_mpi_set_options(int options)
{
    _mpsort_mpi_parse_env();
    _mpsort_mpi_options |= options;
}

int
mpsort_mpi_has_options(int options)
{
    _mpsort_mpi_parse_env();
    return _mpsort_mpi_options & options;
}

void
mpsort_mpi_unset_options(int options)
{
    _mpsort_mpi_parse_env();
    _mpsort_mpi_options &= ~options;

}
