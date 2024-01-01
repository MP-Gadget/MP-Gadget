#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>
#include <stdio.h>

#include "stub.h"

#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <string.h>
#include <unistd.h>

#include <mpi.h>
#include "../utils/endrun.h"
#include "../utils/mymalloc.h"
#include "../utils/mpsort.h"

struct BaseGroup {
    int OriginalTask;
    int Length;
    uint64_t MinID;
    int MinIDTask;
};

static void radix_int(const void * ptr, void * radix, void * arg) {
    *(uint64_t*)radix = *(const int64_t*) ptr + INT64_MIN;
}

static int64_t
checksum(int64_t * data, size_t localsize, MPI_Comm comm)
{
    int64_t sum = 0;
    size_t i;
    for(i = 0; i < localsize; i ++) {
        sum += data[i];
    }

    MPI_Allreduce(MPI_IN_PLACE, &sum, 1, MPI_LONG, MPI_SUM, comm);
    return sum;
}
static void
generate(int64_t * data, size_t localsize, int bits, int seed)
{
    /* only keep bits of precision. */
    srandom(seed);

    size_t i;
    unsigned shift = 64u - bits;
    for(i = 0; i < localsize; i ++) {
        uint64_t value = (int64_t) random() * (int64_t) random() * random() * random();
        data[i] = (signed) ((value << shift));
    }
}

static void
check_sorted(void * data, int elsize, size_t localsize, int compar(void * d1, void * d2), MPI_Comm comm)
{
    size_t i;
    int ThisTask, NTask;
    MPI_Comm_rank(comm, &ThisTask);
    MPI_Comm_size(comm, &NTask);

    const int TAG = 0xbeef;

    for(i = 1; i < localsize; i ++) {
        if(compar(data + i*elsize, data + (i - 1)*elsize) < 0) {
//             struct BaseGroup * i1 = ((struct BaseGroup *)data) + i;
//             struct BaseGroup * i2= ((struct BaseGroup *)data) +(i-1);
//             message(1, "cur=(%ld %ld %ld) prev= (%ld %ld %ld)\n", labs(i1->OriginalTask - i1->MinIDTask), i1->MinID, i1->Length, labs(i2->OriginalTask - i2->MinIDTask), i2->MinID, i2->Length);
            endrun(12, "Ordering of local array is broken i=%ld, d=%ld d-1=%ld. \n", i, ((int64_t *)data)[i], ((int64_t *)data)[i-1]);
        }
        assert_true(compar(data + i*elsize, data + (i - 1)*elsize) >=0);
    }

    if(NTask == 1) return;

    char * prev = ta_malloc("prev", char, elsize);
    memset(prev, -1000000, elsize);

    while(1) {
        if(ThisTask == 0) {
            void * ptr = prev;
            if(localsize > 0) {
                ptr = data + elsize * (localsize - 1);
            }
            MPI_Send(ptr, elsize, MPI_BYTE, ThisTask + 1, TAG, comm);
            break;
        }
        if(ThisTask == NTask - 1) {
            MPI_Recv(prev, elsize, MPI_BYTE,
                    ThisTask - 1, TAG, comm, MPI_STATUS_IGNORE);
            break;
        }
        /* else */
        if(localsize == 0) {
            /* simply pass through whatever we get */
            MPI_Recv(prev, elsize, MPI_BYTE, ThisTask - 1, TAG, comm, MPI_STATUS_IGNORE);
            MPI_Send(prev, elsize, MPI_BYTE, ThisTask + 1, TAG, comm);
            break;
        }
        else
        {
            MPI_Sendrecv(
                    data+(localsize - 1)*elsize, elsize, MPI_BYTE,
                    ThisTask + 1, TAG,
                    prev, elsize, MPI_BYTE,
                    ThisTask - 1, TAG, comm, MPI_STATUS_IGNORE);
            break;
        }
    }

    if(ThisTask > 1) {
        if(localsize > 0) {
//                printf("ThisTask = %d prev = %d\n", ThisTask, prev);
            if(compar(prev, data) > 0) {
                endrun(12, "Ordering of global array is broken prev=%d d=%ld (comp: %d). \n", *prev, *(int64_t *)data, compar(prev, data));
            }
            assert_true(compar(prev, data) <= 0);
        }
    }
}

int compar_int(void * d1, void * d2)
{
    int64_t * i1 = d1;
    int64_t * i2 = d2;
    if(*i1 < *i2)
        return -1;
    else if(*i1 == *i2)
        return 0;
    else
        return 1;
}

static void
do_mpsort_test(int64_t srcsize, int bits, int staggered, int gather)
{
    int ThisTask;
    int NTask;

    MPI_Comm_size(MPI_COMM_WORLD, &NTask);
    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);

    if(gather == 1)
        mpsort_mpi_set_options(MPSORT_REQUIRE_GATHER_SORT);
    if(gather == 0)
        mpsort_mpi_set_options(MPSORT_DISABLE_GATHER_SORT);

//     message(0, "NTask = %d\n", NTask);
//     message(0, "src size = %ld\n", srcsize);

    if(staggered && (ThisTask % 2 == 0)) srcsize = 0;

    int64_t csize;

    MPI_Allreduce(&srcsize, &csize, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);

    int64_t destsize = csize * (ThisTask + 1) /  NTask - csize * (ThisTask) / NTask;

    message(0, "dest size = %ld\n", destsize);
//     message(0, "csize = %ld\n", csize);

    int64_t * src = mymalloc("src", srcsize * sizeof(int64_t));
    int64_t * dest = mymalloc("dest", destsize * sizeof(int64_t));

    int seed = 9999 * ThisTask;
    generate(src, srcsize, bits, seed);

    int64_t srcsum = checksum(src, srcsize, MPI_COMM_WORLD);
//         if(ThisTask == 0)
//        mpsort_setup_timers(512);
    {
        double start = MPI_Wtime();

        mpsort_setup_timers(512);
        mpsort_mpi_newarray(src, srcsize,
                            dest, destsize,
                            sizeof(int64_t),
                            radix_int, sizeof(int64_t), NULL,
                            MPI_COMM_WORLD);

        MPI_Barrier(MPI_COMM_WORLD);
        double end = MPI_Wtime();

        int64_t destsum = checksum(dest, destsize, MPI_COMM_WORLD);

        if(destsum != srcsum) {
            endrun(5, "MPSort checksum is inconsistent.\n");
            MPI_Abort(MPI_COMM_WORLD, -1);
        }

        check_sorted(dest, sizeof(int64_t), destsize, compar_int, MPI_COMM_WORLD);

        message(0, "MPSort total time: %g\n", end - start);
//         if(ThisTask == 0) {
//             mpsort_mpi_report_last_run();
//                mpsort_free_timers();
//         }
    }

    mpsort_mpi_unset_options(MPSORT_REQUIRE_GATHER_SORT + MPSORT_DISABLE_GATHER_SORT);
    myfree(dest);
    myfree(src);
}

static void fof_radix_Group_TotalCountTaskDiffMinID(const void * a, void * radix, void * arg) {
    uint64_t * u = (uint64_t *) radix;
    struct BaseGroup * f = (struct BaseGroup *) a;
    u[0] = labs(f->OriginalTask - f->MinIDTask);
    u[1] = f->MinID;
    u[2] = UINT64_MAX - (f->Length);
}

int compar_bg(void * d1, void * d2)
{
    struct BaseGroup * i1 = d1;
    struct BaseGroup * i2 = d2;
    int dist1 = labs(i1->OriginalTask - i1->MinIDTask);
    int dist2 = labs(i2->OriginalTask - i2->MinIDTask);
    /* Note reversed sign! We want the largest groups first.*/
    if(i1->Length < i2->Length)
        return 1;
    else if(i1->Length > i2->Length)
        return -1;
    if(i1->MinID < i2->MinID)
        return -1;
    else if(i1->MinID > i2->MinID)
        return 1;
    else if(dist1 < dist2)
        return -1;
    else if(dist1 > dist2)
        return 1;
    return 0;
}

static uint64_t
checksum_minid(struct BaseGroup * data, size_t localsize, MPI_Comm comm)
{
    uint64_t sum = 0;
    size_t i;
    for(i = 0; i < localsize; i ++) {
        sum += data[i].MinID;
    }

    MPI_Allreduce(MPI_IN_PLACE, &sum, 1, MPI_LONG, MPI_SUM, comm);
    return sum;
}

/* Tests sorting with a very long radix, as is done in the FOF code*/
static void
do_long_radix_test(int srcsize)
{
    int ThisTask;
    int NTask;

    MPI_Comm_size(MPI_COMM_WORLD, &NTask);
    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);

    struct BaseGroup * base = mymalloc("base", srcsize * sizeof(struct BaseGroup));

    int i;
    for(i = 0; i < srcsize; i++)
    {
        base[i].OriginalTask = ThisTask;	/* original task */
        base[i].MinIDTask = random() % NTask;
        base[i].MinID = random();
        base[i].Length = 10;
    }

    int64_t srcsum = checksum_minid(base, srcsize, MPI_COMM_WORLD);
    {
        double start = MPI_Wtime();

        mpsort_mpi_newarray(base, srcsize,
                            base, srcsize,
                            sizeof(struct BaseGroup),
                            fof_radix_Group_TotalCountTaskDiffMinID, 24, NULL,
                            MPI_COMM_WORLD);

        MPI_Barrier(MPI_COMM_WORLD);
        double end = MPI_Wtime();

        int64_t destsum = checksum_minid(base, srcsize, MPI_COMM_WORLD);

        if(destsum != srcsum) {
            endrun(5, "MPSort checksum is inconsistent.\n");
            MPI_Abort(MPI_COMM_WORLD, -1);
        }

        check_sorted(base, sizeof(struct BaseGroup), srcsize, compar_bg, MPI_COMM_WORLD);

        message(0, "MPSort total time: %g\n", end - start);
    }
    myfree(base);
}

static void
test_basegroup(void ** state)
{
    do_long_radix_test(50);
    /* This is chosen because it fails in travis*/
    do_long_radix_test(0);
}

static void
test_mpsort_bits(void ** state)
{
    message(0, "16 bits!\n");
    /* With whatever gather we like*/
    do_mpsort_test(2000, 16, 0, -1);
    message(0, "32 bits!\n");
    do_mpsort_test(2000, 32, 0, -1);
    message(0, "64 bits!\n");
    do_mpsort_test(2000, 64, 0, -1);
}

static void
test_mpsort_stagger(void ** state)
{
    /* With stagger*/
    do_mpsort_test(2000, 32, 1, -1);
    /* Use a number that doesn't divide evenly so we get a different destsize*/
    do_mpsort_test(1999, 32, 0, -1);
}

static void
test_mpsort_gather(void ** state)
{
    /* With forced gather*/
    do_mpsort_test(2000, 32, 0, 1);
    /* Without forced gather*/
    do_mpsort_test(2000, 32, 0, 0);
}

int main(void)
{
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_mpsort_bits),
        cmocka_unit_test(test_mpsort_stagger),
        cmocka_unit_test(test_basegroup),
        cmocka_unit_test(test_mpsort_gather),
    };
    return cmocka_run_group_tests_mpi(tests, NULL, NULL);
}
