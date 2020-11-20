#ifndef __TIMEFAC_H
#define __TIMEFAC_H

#include "types.h"
#include "cosmology.h"
#include "timebinmgr.h"

/* Get the exact drift and kick factors at given time by integrating. */
double get_exact_drift_factor(Cosmology * CP, inttime_t ti0, inttime_t ti1);
double get_exact_hydrokick_factor(Cosmology * CP, inttime_t ti0, inttime_t ti1);
double get_exact_gravkick_factor(Cosmology * CP, inttime_t ti0, inttime_t ti1);

#endif
