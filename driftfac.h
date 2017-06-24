#ifndef __DRIFTFAC_H
#define __DRIFTFAC_H

double get_hydrokick_factor(int time0, int time1);
double get_gravkick_factor(int time0, int time1);
double drift_integ(double a, void *param);
double gravkick_integ(double a, void *param);
double hydrokick_integ(double a, void *param);
void init_drift_table(double timeBegin, double timeMax);
double get_drift_factor(int time0, int time1);

#endif
