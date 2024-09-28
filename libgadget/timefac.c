#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <gsl/gsl_integration.h>

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

#define WORKSIZE 10000

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

/*Do the integral required to get a factor.*/
static double get_exact_factor(Cosmology * CP, inttime_t t0, inttime_t t1, double (*factor) (double, void *))
{
    double result, abserr;
    if(t0 == t1)
        return 0;
    double a0 = exp(loga_from_ti(t0));
    double a1 = exp(loga_from_ti(t1));
    gsl_function F;
    gsl_integration_workspace *workspace;
    workspace = gsl_integration_workspace_alloc(WORKSIZE);
    F.function = factor;
    F.params = CP;
    gsl_integration_qag(&F, a0, a1, 0, 1.0e-8, WORKSIZE, GSL_INTEG_GAUSS61, workspace, &result, &abserr);
    gsl_integration_workspace_free(workspace);
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
    Cosmology *CP = (Cosmology *) param;
    double h = hubble_function(CP, a);
    return 1. / (h * a * a); 
}

/* Function to compute the comoving distance between two scale factors */
// double compute_comoving_distance(Cosmology *CP, double a0, double a1, const double UnitVelocity_in_cm_per_s)
// {
//     double result, abserr;
//     gsl_function F;
//     gsl_integration_workspace *workspace = gsl_integration_workspace_alloc(WORKSIZE);
    
//     F.function = comoving_distance_integ;
//     F.params = CP;

//     // Using GSL to perform the integration
//     gsl_integration_qag(&F, a0, a1, 0, 1.0e-8, WORKSIZE, GSL_INTEG_GAUSS61, workspace, &result, &abserr);
//     gsl_integration_workspace_free(workspace);

//     return (LIGHTCGS/UnitVelocity_in_cm_per_s) * result;
// }

/* Adaptive integration function with error control */
double adaptive_integrate(std::function<double(double)> integrand, double a0, double a1, double *abserr, double epsrel = 1e-8, size_t max_points = 1024)
{
    double result_prev = 0.0;
    double result_current = 0.0;
    size_t points = 15;  // Start with 15-point Gauss-Legendre quadrature

    while (true) {
        result_prev = result_current;

        // Use switch-case to handle different compile-time fixed point values
        switch (points) {
            case 15:
                result_current = boost::math::quadrature::gauss<double, 15>::integrate(integrand, a0, a1);
                break;
            case 31:
                result_current = boost::math::quadrature::gauss<double, 31>::integrate(integrand, a0, a1);
                break;
            case 63:
                result_current = boost::math::quadrature::gauss<double, 63>::integrate(integrand, a0, a1);
                break;
            case 127:
                result_current = boost::math::quadrature::gauss<double, 127>::integrate(integrand, a0, a1);
                break;
            case 255:
                result_current = boost::math::quadrature::gauss<double, 255>::integrate(integrand, a0, a1);
                break;
            case 511:
                result_current = boost::math::quadrature::gauss<double, 511>::integrate(integrand, a0, a1);
                break;
            case 1024:
                result_current = boost::math::quadrature::gauss<double, 1024>::integrate(integrand, a0, a1);
                break;
            default:
                printf("Unsupported number of points: %zu\n", points);
                return result_current;
        }

        // Estimate the absolute error as the difference between successive results
        *abserr = fabs(result_current - result_prev);

        // Check if the relative error is within the tolerance
        if (fabs(result_current) > 0 && (*abserr / fabs(result_current)) < epsrel) {
            break;
        }

        // If we've reached the max allowed points without satisfying error tolerance, stop
        if (points == max_points) {
            printf("Warning: Maximum points reached. Desired relative error not achieved.\n");
            break;
        }

        // Double the number of quadrature points for the next iteration
        size_t next_points = points * 2;
        if (next_points > max_points) {
            points = max_points;
        } else {
            points = next_points;
        }
    }

    return result_current;
}


// Function to perform tanh-sinh integration with adaptive max_refinements
double tanh_sinh_integrate_adaptive(std::function<double(double)> func, double a, double b, double* estimated_error, double rel_tol = 1e-8, int max_refinements_limit = 30, int init_refine = 5, int step = 5) {
    double result_prev = 0.0;
    double result_current = 0.0;
    *estimated_error = 1.0;  // Start with a large relative error
    int max_refine = init_refine;

    // Loop until reaching the max refinements limit or satisfying the tolerance
    for (; max_refine <= max_refinements_limit; max_refine += step) {
        // Create a Tanh-Sinh integrator with the current max_refinements
        boost::math::quadrature::tanh_sinh<double> integrator(max_refine);

        // Perform the integration
        result_current = integrator.integrate(func, a, b);

        // If this is not the first iteration, compute the relative error
        if (max_refine > init_refine) {
            *estimated_error = fabs(result_current - result_prev) / fabs(result_current);

            // Check if the relative error is within the target tolerance
            if (*estimated_error < rel_tol) {
                break;  // Stop refining if the result is within the tolerance
            }
        }

        // Update the previous result for the next iteration
        result_prev = result_current;
    }

    // Return the final result
    return result_current;
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
