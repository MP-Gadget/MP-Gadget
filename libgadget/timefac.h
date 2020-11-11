#ifndef __TIMEFAC_H
#define __TIMEFAC_H

#include "types.h"
#include "cosmology.h"
#include "timebinmgr.h"

void init_drift_table(Cosmology * CP, double timeBegin, double timeMax);
double get_hydrokick_factor(inttime_t ti0, inttime_t ti1);
double get_gravkick_factor(inttime_t ti0, inttime_t ti1);

/* Get the exact drift and kick factors at given time by integrating. */
double get_exact_drift_factor(Cosmology * CP, inttime_t ti0, inttime_t ti1);

#endif
