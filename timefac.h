#ifndef __TIMEFAC_H
#define __TIMEFAC_H

void init_drift_table(double timeBegin, double timeMax);
double get_hydrokick_factor(inttime_t ti0, inttime_t ti1);
double get_gravkick_factor(inttime_t ti0, inttime_t ti1);
double get_drift_factor(inttime_t ti0, inttime_t ti1);

#endif
