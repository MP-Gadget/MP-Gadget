#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>
#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include "stub.h"
#include "../utils/interp.h"

#define DMAX(x,y) ((x) > (y) ? (x) : (y))
#define DMIN(x,y) ((x) < (y) ? (x) : (y))

/*Modified modulus function which deals with negative numbers better*/
double modulus(double x, double mod) {
    if (x >= 0)
        return fmod(x,mod);
    else
        return fmod(x+mod,mod);
}

#define DSIZE 3

static void test_interp(void ** state) {
    Interp ip;
    int64_t dims[] = {DSIZE, DSIZE};
    double ydata[DSIZE][DSIZE], ydata_sum[DSIZE][DSIZE];
    interp_init(&ip, 2, dims);
    interp_init_dim(&ip, 0, 0, DSIZE-1);
    interp_init_dim(&ip, 1, 0, DSIZE-1);
    /*Initialise the data*/
    {
        int i, j;
        for(i = 0; i < DSIZE; i++) {
            for(j = 0; j < DSIZE; j++) {
                ydata[i][j] = fabs((1. - j) * (1. - i));
                ydata_sum[i][j] = i+j;
            }
        }
    }

    double i, j;
    int status[2];
    for(i = -0.4; i <= DSIZE; i += 0.4) {
        for(j = -0.4; j <= DSIZE; j += 0.4) {
            double x[] = {i, j};
            double y = interp_eval(&ip, x, (double*) ydata, status);
            assert_true(status[0] == -1*(i < 0)  + (i > DSIZE-1));
            assert_true(status[1] == -1*(j < 0)  + (j > DSIZE-1));
            double yp = interp_eval_periodic (&ip, x, (double*) ydata);
            /*Note boundaries: without periodic use the maximum, with no remainers.*/
            double y_truth = fabs((1.-DMAX(DMIN(i,DSIZE-1),0))*(1.-DMAX(DMIN(j,DSIZE-1),0)));
            /* With a periodic boundary we normally use the modulus. However for this specific case
             * the boundaries happen to be identical so variation in the bin between them 
             * is ignored by the interpolator and y == yp. This is not true if you increase the range on i.*/
/*             printf("(%g %g ) %3.2f/%3.2f/%3.2f \n", i,j, y, yp, yp_truth); */
            assert_true(fabs(y - y_truth) <= 1e-5*y);
            assert_true(fabs(yp - y_truth) <= 1e-5*yp);
        }
    }
    i=0;
    for(i = -0.4; i <= 3.0; i += 0.4) {
        for(j = -0.4; j <= 3.0; j += 0.4) {
            double x[] = {i, j};
            double y = interp_eval(&ip, x, (double*) ydata_sum, status);
            double yp = interp_eval_periodic (&ip, x, (double*) ydata_sum);
            double y_truth = DMAX(DMIN(i,DSIZE-1),0)+DMAX(DMIN(j,DSIZE-1),0);
            double yp_truth = modulus(i,DSIZE)+ modulus(j, DSIZE);
/*             printf("(%g %g ) %3.2f/%3.2f/%3.2f \n", i,j, y, yp, yp_truth); */
            /*Linear interpolation is very inaccurate outside the boundaries in the periodic case!*/
            if(i >= 0 && j >= 0 && i <= DSIZE-1 && j <= DSIZE-1)
                assert_true(fabs(yp - yp_truth) <= 1e-5*yp);
            assert_true(fabs(y - y_truth) <= 1e-5*y);
        }
    }
    interp_destroy(&ip);
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_interp),
    };
    return cmocka_run_group_tests_mpi(tests, NULL, NULL);
}
