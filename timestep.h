#ifndef TIMESTEP_H
#define TIMESTEP_H

#include "timebinmgr.h"
/*Flat array containing all active particles:
set in run.c: find_next_sync_point_and_drift*/
extern int NumActiveParticle;
extern int *ActiveParticle;

void timestep_allocate_memory(int MaxPart);
int rebuild_activelist(inttime_t ti_current);
void set_global_time(double newtime);
int find_timesteps(int * MinTimeBin);
void apply_half_kick(void);
void apply_PM_half_kick(void);

void print_timebin_statistics(int NumCurrentTiStep);
int is_timebin_active(int i, inttime_t current);
void set_timebin_active(binmask_t mask);

void sph_VelPred(int i, double * VelPred);

inttime_t find_next_kick(inttime_t Ti_Current, int minTimeBin);

void init_timebins(double TimeInit);

int is_PM_timestep(inttime_t ti);

#endif
