#ifndef __TIMEFAC_H
#define __TIMEFAC_H

void init_drift_table(double timeBegin, double timeMax, int timebase);
double get_hydrokick_factor(int time0, int time1);
double get_gravkick_factor(int time0, int time1);
double get_drift_factor(int time0, int time1);

#endif
