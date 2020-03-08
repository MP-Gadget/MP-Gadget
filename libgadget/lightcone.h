#ifndef LIGHTCONE_H
#define LIGHTCONE_H

/* Initialise the lightcone code module. */
void lightcone_init(Cosmology * CP, double timeBegin);
void lightcone_compute(double a, Cosmology * CP, inttime_t ti_curr, inttime_t ti_next);
#endif
