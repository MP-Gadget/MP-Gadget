#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "physconst.h"
#include "timefac.h"
#include "timebinmgr.h"
#include "utils.h"

#include <stdio.h>
#include <math.h>
#include <boost/math/quadrature/gauss.hpp>
#include <boost/math/special_functions/fpclassify.hpp>  // For isnan and isinf
#include <functional>
#include <boost/math/quadrature/tanh_sinh.hpp>

// Function to perform tanh-sinh integration with adaptive max_refinements
double tanh_sinh_integrate_adaptive(
    std::function<double(double)> func, double a, double b, 
    double* estimated_error, double rel_tol, double abs_tol, 
    int max_refinements_limit, int init_refine, int step) 
{
    double result_prev = 0.0;
    double result_current = 0.0;
    int max_refine = init_refine;

    // Loop until reaching the max refinements limit or satisfying the tolerance
    for (; max_refine <= max_refinements_limit; max_refine += step) {
        // Create a Tanh-Sinh integrator with the current max_refinements
        boost::math::quadrature::tanh_sinh<double> integrator(max_refine);

        // Perform the integration
        result_current = integrator.integrate(func, a, b);

        // If this is not the first iteration, compute the absolute and relative errors
        if (max_refine > init_refine) {
            double abs_error = fabs(result_current - result_prev);  // Absolute error
            double rel_error = abs_error / fabs(result_current);    // Relative error

            *estimated_error = abs_error;  // Store the absolute error

            // Check if either the relative or absolute error is within the target tolerance
            if (rel_error < rel_tol || abs_error < abs_tol) {
                break;  // Stop refining if either error is within the tolerance
            }
        }

        // Update the previous result for the next iteration
        result_prev = result_current;
    }

    // If we exited the loop without achieving the desired tolerance, print a warning
    if (*estimated_error > abs_tol && (*estimated_error / fabs(result_current)) > rel_tol) {
        message(1, 
            "Warning: Tanh-Sinh integration reached neither the desired relative tolerance of %g nor absolute tolerance of %g. "
            "Final absolute error: %g, relative error: %g\n", 
            rel_tol, abs_tol, *estimated_error, (*estimated_error / fabs(result_current)));
    }

    // Return the final result
    return result_current;
}

/* Integrand for the drift table*/
static double drift_integ(double a, void *param)
{
  Cosmology * CP = (Cosmology *) param;
  double h = hubble_function(CP, a);
  return 1 / (h * a * a * a);
}

/* Integrand for the gravkick table*/
static double gravkick_integ(double a, void *param)
{
  Cosmology * CP = (Cosmology *) param;
  double h = hubble_function(CP, a);

  return 1 / (h * a * a);
}

/* Integrand for the hydrokick table.
 * Note this is the same function as drift.*/
static double hydrokick_integ(double a, void *param)
{
  double h;

  Cosmology * CP = (Cosmology *) param;
  h = hubble_function(CP, a);

  return 1 / (h * pow(a, 3 * GAMMA_MINUS1) * a);
}

// Function to compute a factor using Tanh-Sinh adaptive integration
static double get_exact_factor(Cosmology *CP, inttime_t t0, inttime_t t1, double (*factor)(double, void *))
{
    if (t0 == t1) {
        return 0;
    }

    // Calculate the scale factors
    double a0 = exp(loga_from_ti(t0));
    double a1 = exp(loga_from_ti(t1));
    double abserr;

    // Define the integrand as a lambda function, wrapping the existing factor function
    auto integrand = [CP, factor](double a) {
        return factor(a, (void*)CP);
    };

    // Call the adaptive Tanh-Sinh integrator
    double result = tanh_sinh_integrate_adaptive(integrand, a0, a1, &abserr);

    return result;
}

/*Get the exact drift factor*/
double get_exact_drift_factor(Cosmology * CP, inttime_t ti0, inttime_t ti1)
{
    return get_exact_factor(CP, ti0, ti1, &drift_integ);
}

/*Get the exact drift factor*/
double get_exact_gravkick_factor(Cosmology * CP, inttime_t ti0, inttime_t ti1)
{
    return get_exact_factor(CP, ti0, ti1, &gravkick_integ);
}

double get_exact_hydrokick_factor(Cosmology * CP, inttime_t ti0, inttime_t ti1)
{
    return get_exact_factor(CP, ti0, ti1, &hydrokick_integ);
}

/* Integrand for comoving distance */
static double comoving_distance_integ(double a, void *param)
{
    // Cosmology *CP = (Cosmology *) param;
    // double h = hubble_function(CP, a);
    // return 1. / (h * a * a); 
    return gravkick_integ(a, param);
}

/* Function to compute comoving distance using the adaptive integrator */
double compute_comoving_distance(Cosmology *CP, double a0, double a1, const double UnitVelocity_in_cm_per_s)
{   
    // relative error tolerance
    // double epsrel = 1e-8;
    double result, abserr;
    // Define the integrand as a lambda function, wrapping comoving_distance_integ
    auto integrand = [CP](double a) {
        return comoving_distance_integ(a, (void*)CP);
    };

    // Call the generic adaptive integration function
    // result = adaptive_integrate(integrand, a0, a1, &abserr);
    result = tanh_sinh_integrate_adaptive(integrand, a0, a1, &abserr);

    // Convert the result using the provided units
    return (LIGHTCGS / UnitVelocity_in_cm_per_s) * result;
}
