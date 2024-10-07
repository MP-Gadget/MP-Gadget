#ifndef __TIMEFAC_H
#define __TIMEFAC_H

#include "types.h"
#include "cosmology.h"
#include "timebinmgr.h"
#include <functional>  // For std::function

/* Get the exact drift and kick factors at given time by integrating. */
double get_exact_drift_factor(Cosmology * CP, inttime_t ti0, inttime_t ti1);
double get_exact_hydrokick_factor(Cosmology * CP, inttime_t ti0, inttime_t ti1);
double get_exact_gravkick_factor(Cosmology * CP, inttime_t ti0, inttime_t ti1);
double compute_comoving_distance(Cosmology *CP, double a0, double a1, const double UnitVelocity_in_cm_per_s);
double tanh_sinh_integrate_adaptive(
    std::function<double(double)> func, 
    double a, 
    double b, 
    double* estimated_error, 
    double rel_tol = 1e-8, 
    double abs_tol = 0,
    int max_refinements_limit = 30, 
    int init_refine = 5, 
    int step = 5
);

#endif
