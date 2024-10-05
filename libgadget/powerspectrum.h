#ifndef POWERSPEC_H
#define POWERSPEC_H

#include <stddef.h>
#include <stdint.h>

// Undefine P before including Boost
#ifdef P
#undef P
#endif
#include <boost/math/interpolators/barycentric_rational.hpp>

#define P PartManager->Base

typedef struct _powerspectrum {
    double * kk;
    double * Power;
    int64_t * Nmodes;
    int size;
    int nalloc;
    int nonzero;
    double Norm;
    /* Used to set the output units of the power to Mpc*/
    double BoxSize_in_MPC;
    /*These are for the LRA neutrino code*/
    /*log k bins and ratio of Pnu to Pcdm: stored so interpolation is accurate*/
    double * logknu;
    double * delta_nu_ratio;
    double nu_prefac;
    boost::math::interpolators::barycentric_rational<double>* nu_spline;

} Power;

/*Allocate memory for the power spectrum*/
void powerspectrum_alloc(Power * ps, const int nbins, const int nthreads, const int MassiveNuLinResp, const double BoxSize_in_cm);

/*Zero memory for the power spectrum*/
void powerspectrum_zero(Power * ps);

/*Free power spectrum memory*/
void powerspectrum_free(Power * ps);

/* Sum the different modes on each thread and processor together to get a power spectrum,
 * and fix the units.*/
void powerspectrum_sum(Power * ps);

/*Save the power spectrum to a file*/
void powerspectrum_save(Power * ps, const char * OutputDir, const char * filename, const double Time, const double D1);
#endif
