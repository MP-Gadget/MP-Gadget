/*Tests for the timebinmgr module, to ensure we got the bitwise arithmetic right.*/

#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>
#include <math.h>
#include <string.h>
#include <stdio.h>
#include "stub.h"
//So All.OutputList is defined
#include "../allvars.h"
#include "../timebinmgr.h"

struct global_data_all_processes All;

/*timebinmgr has no state*/
/*First test conversions between float and integer timelines*/
static void test_conversions(void ** state) {

    double outs[4] = {log(0.1), log(0.2), log(0.8), log(1)};
    memcpy(All.OutputListTimes, outs, 4*sizeof(double));
    All.OutputListLength = 4;

    All.TimeInit = 0.1;
    All.TimeMax = 1.0;

    setup_sync_points();

    /*Convert an integer to and from loga*/
    /* double loga_from_ti(unsigned int ti); */
    assert_true(fabs(loga_from_ti(0) - outs[0]) < 1e-6);
    assert_true(fabs(loga_from_ti(TIMEBASE) - outs[1]) < 1e-6);
    assert_true(fabs(loga_from_ti(TIMEBASE-1) - (outs[0] + (outs[1]-outs[0])*(TIMEBASE-1)/TIMEBASE)) < 1e-6);
    assert_true(fabs(loga_from_ti(TIMEBASE+1) - (outs[1] + (outs[2]-outs[1])/TIMEBASE)) < 1e-6);
    assert_true(fabs(loga_from_ti(2*TIMEBASE) - outs[2]) < 1e-6);
    /* unsigned int ti_from_loga(double loga); */
    assert_true(ti_from_loga(outs[0]) == 0);
    assert_true(ti_from_loga(outs[1]) == TIMEBASE);
    assert_true(ti_from_loga(outs[2]) == 2*TIMEBASE);
    double midpt = (outs[2] + outs[1])/2;
    assert_true(ti_from_loga(midpt) == TIMEBASE+TIMEBASE/2);
    assert_true(fabs(loga_from_ti(TIMEBASE+TIMEBASE/2)-midpt)< 1e-6);

    /*Check behaviour past end*/
    assert_true(ti_from_loga(0) == 3*TIMEBASE);
    assert_true(fabs(loga_from_ti(ti_from_loga(0.1)) - 0.1) < 1e-6);

    /*! this function returns the next output time after ti_curr.*/
    assert_int_equal(find_next_sync_point(0)->ti , TIMEBASE);
    assert_int_equal(find_next_sync_point(TIMEBASE)->ti , 2 * TIMEBASE);
    assert_int_equal(find_next_sync_point(TIMEBASE-1)->ti , TIMEBASE);
    assert_int_equal(find_next_sync_point(TIMEBASE+1)->ti , 2*TIMEBASE);
    assert_int_equal(find_next_sync_point(4 * TIMEBASE) , NULL);

    assert_int_equal(find_current_sync_point(0)->ti , 0);
    assert_int_equal(find_current_sync_point(TIMEBASE)->ti , TIMEBASE);
    assert_int_equal(find_current_sync_point(-1) , NULL);
    assert_int_equal(find_current_sync_point(TIMEBASE-1) , NULL);

    assert_int_equal(find_current_sync_point(0)->write_snapshot, 1);
    assert_int_equal(find_current_sync_point(TIMEBASE)->write_snapshot, 1);
    assert_int_equal(find_current_sync_point(2 * TIMEBASE)->write_snapshot, 1);
    assert_int_equal(find_current_sync_point(3 * TIMEBASE)->write_snapshot, 1);
}

static void test_dloga(void ** state) {

    double outs[4] = {log(0.1), log(0.2), log(0.8), log(1)};
    memcpy(All.OutputListTimes, outs, 4*sizeof(double));
    All.OutputListLength = 4;
    All.Ti_Current = ti_from_loga(log(0.55));
    /* unsigned int dti_from_dloga(double loga); */
    /* double dloga_from_dti(unsigned int ti); */

    /*Get dloga from a timebin*/
    /* double get_dloga_for_bin(int timebin); */
    assert_true(fabs(get_dloga_for_bin(0))<1e-6);
    assert_true(fabs(get_dloga_for_bin(TIMEBINS)-(outs[2]-outs[1]))<1e-6);
    assert_true(fabs(get_dloga_for_bin(TIMEBINS-2)-(outs[2]-outs[1])/4)<1e-6);

    /*Enforce that an integer time is a power of two*/
    /* unsigned int round_down_power_of_two(unsigned int ti); */
    assert_true(round_down_power_of_two(TIMEBASE)==TIMEBASE);
    assert_true(round_down_power_of_two(TIMEBASE+1)==TIMEBASE);
    assert_true(round_down_power_of_two(TIMEBASE-1)==TIMEBASE/2);
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_conversions),
        cmocka_unit_test(test_dloga),
    };
    return cmocka_run_group_tests_mpi(tests, NULL, NULL);
}
