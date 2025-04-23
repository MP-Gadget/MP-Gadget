#ifndef LIGHTCONE_H
#define LIGHTCONE_H
#include "cosmology.h"
#include "types.h"

/* Initialise the lightcone code module. */
void lightcone_init(Cosmology * CP, double timeBegin, const double UnitLength_in_cm, const char * OutputDir);
void lightcone_compute(double a, double BoxSize, Cosmology * CP, inttime_t ti_curr, inttime_t ti_next, const RandTable * const rnd);
#endif
