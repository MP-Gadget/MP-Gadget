#ifndef __TIMEFAC_H
#define __TIMEFAC_H

void init_drift_table(double timeBegin, double timeMax);
double get_hydrokick_factor(unsigned int ti0, unsigned int ti1);
double get_gravkick_factor(unsigned int ti0, unsigned int ti1);
double get_drift_factor(unsigned int ti0, unsigned int ti1);

#endif
