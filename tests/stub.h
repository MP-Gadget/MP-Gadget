#include <libgadget/utils.h>

int
_cmocka_run_group_tests_mpi(const char * name, const struct CMUnitTest tests[], size_t ntests, void * p1, void * p2);

#define cmocka_run_group_tests_mpi(tests, p1, p2) _cmocka_run_group_tests_mpi(#tests, tests, sizeof(tests) / sizeof(tests[0]), p1, p2)
#define assert_all_true(x) assert_true(!MPIU_Any(!x, MPI_COMM_WORLD));
