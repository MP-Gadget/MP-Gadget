#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>
#include <math.h>
#include <stdio.h>

#include "stub.h"

static void
test_allocator(void ** state)
{
    Allocator A0[1];
    allocator_init(A0, "Default", 4096 * 1024, 1, NULL);

    int * p1 = allocator_alloc_bot(A0, "M+1", 1024*sizeof(int));
    int * p2 = allocator_alloc_bot(A0, "M+2", 2048*sizeof(int));

    int * q1 = allocator_alloc_top(A0, "M-1", 1024*sizeof(int));
    int * q2 = allocator_alloc_top(A0, "M-2", 2048*sizeof(int));

    p1[0] = 1;
    p2[1] = 1;
    allocator_print(A0);

    q2[2000] = 1;
    q2 = allocator_realloc(A0, q2, 3072*sizeof(int));
    assert_int_equal(q2[2000] ,1);

    /* Assert that reallocing does not move the base pointer.
     * Note this is true only for bottom allocations*/
    int * p2new = allocator_realloc(A0, p2, 3072*sizeof(int));
    assert_int_equal(p2new, p2);

    assert_int_equal(allocator_dealloc(A0, p1), ALLOC_EMISMATCH);
    assert_int_equal(allocator_dealloc(A0, q1), ALLOC_EMISMATCH);

    assert_int_equal(allocator_dealloc(A0, p2), 0);
    assert_int_equal(allocator_dealloc(A0, q2), 0);

    allocator_free(p1);
    allocator_free(q1);

    allocator_print(A0);

    allocator_destroy(A0);
}

static void
test_sub_allocator(void ** state)
{
    Allocator A0[1];
    Allocator A1[1];
    allocator_init(A0, "Default", 4096 * 1024 * 2, 1, NULL);
    allocator_init(A1, "A1", 4096 * 1024, 1, A0);

    allocator_print(A0);
    void * p1 = allocator_alloc_bot(A1, "M+1", 1024);
    void * p2 = allocator_alloc_bot(A0, "M+2", 2048);

    allocator_print(A0);
    allocator_print(A1);

    allocator_free(p2);
    allocator_free(p1);

    allocator_destroy(A1);
    allocator_destroy(A0);
}
static void
test_allocator_malloc(void ** state)
{
    Allocator A0[1];
    allocator_malloc_init(A0, "libc based", 4096 * 1024, 1, NULL);

    int * p1 = allocator_alloc_bot(A0, "M+1", 2048);
    int * p2 = allocator_alloc_bot(A0, "M+2", 2048);

    int * q1 = allocator_alloc_top(A0, "M-1", 2048);
    int * q2 = allocator_alloc_top(A0, "M-2", 128*sizeof(int));

    allocator_print(A0);
    p1[0] = 1;
    p2[1] = 1;
    q2[100] = 1;

    p2 = allocator_realloc(A0, p2, 3072);
    assert_int_equal(p2[1],1);
    q2 = allocator_realloc(A0, q2, 1000*sizeof(int));
    q2[500] = 1;
    assert_int_equal(q2[100] ,1);

    assert_int_equal(allocator_dealloc(A0, p1), ALLOC_EMISMATCH);
    assert_int_equal(allocator_dealloc(A0, q1), ALLOC_EMISMATCH);

    assert_int_equal(allocator_dealloc(A0, p2), 0);
    assert_int_equal(allocator_dealloc(A0, q2), 0);

    allocator_free(p1);
    allocator_free(q1);

    allocator_print(A0);

    allocator_destroy(A0);
}

int main(void)
{
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_allocator),
        cmocka_unit_test(test_allocator_malloc),
        cmocka_unit_test(test_sub_allocator),
    };
    return cmocka_run_group_tests_mpi(tests, NULL, NULL);
}
