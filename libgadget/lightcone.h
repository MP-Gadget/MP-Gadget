#ifndef LIGHTCONE_H
#define LIGHTCONE_H

/* Initialise the lightcone code module. */
void lightcone_init(Cosmology * CP, double timeBegin, const double UnitLength_in_cm, const char * OutputDir);
void lightcone_compute(double a, double BoxSize, Cosmology * CP, inttime_t ti_curr, inttime_t ti_next);
#endif
