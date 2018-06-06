#ifndef TIMESTEP_H
#define TIMESTEP_H

#include "timebinmgr.h"
/*Flat array containing all active particles:
set in rebuild_activelist.*/
extern int NumActiveParticle;
extern int *ActiveParticle;

int rebuild_activelist(inttime_t ti_current, int NumCurrentTiStep);
void free_activelist(void);
void set_global_time(double newtime);
int find_timesteps(int * MinTimeBin);
void apply_half_kick(void);
void apply_PM_half_kick(void);

int is_timebin_active(int i, inttime_t current);
void set_timebin_active(binmask_t mask);

void sph_VelPred(int i, double * VelPred);

inttime_t find_next_kick(inttime_t Ti_Current, int minTimeBin);

void init_timebins(double TimeInit);

int is_PM_timestep(inttime_t ti);

#endif
