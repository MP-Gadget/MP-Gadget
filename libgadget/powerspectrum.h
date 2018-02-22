#ifndef POWERSPEC_H
#define POWERSPEC_H

#include <stdint.h>

struct _powerspectrum {
    double * kk;
    double * Power;
    double * logknu;
    double * Pnuratio;
    int64_t * Nmodes;
    size_t size;
    size_t nalloc;
    double Norm;
};

/*Allocate memory for the power spectrum*/
void powerspectrum_alloc(struct _powerspectrum * PowerSpectrum, const int nbins, const int nthreads);

/*Zero memory for the power spectrum*/
void powerspectrum_zero(struct _powerspectrum * PowerSpectrum);

/* Sum the different modes on each thread and processor together to get a power spectrum,
 * and fix the units.*/
void powerspectrum_sum(struct _powerspectrum * PowerSpectrum, const double BoxSize_in_cm);

/*Save the power spectrum to a file*/
void powerspectrum_save(struct _powerspectrum * PowerSpectrum, const char * OutputDir, const double Time, const double D1);

#endif
