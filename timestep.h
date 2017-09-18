#ifndef TIMESTEP_H
#define TIMESTEP_H

#include "timebinmgr.h"
/*Flat array containing all active particles:
set in run.c: find_next_sync_point_and_drift*/
extern int NumActiveParticle;
extern int *ActiveParticle;

extern int TimeBinCount[TIMEBINS];
extern int TimeBinCountType[6][TIMEBINS];

void timestep_allocate_memory(int MaxPart);
int update_active_timebins(inttime_t next_kick);
void rebuild_activelist(void);
void set_global_time(double newtime);
void find_timesteps(void);
void apply_half_kick(void);
void apply_PM_half_kick(void);

int is_timebin_active(int i);
void set_timebin_active(binmask_t mask);

void sph_VelPred(int i, double * VelPred);
double EntropyPred(int i);
double PressurePred(int i);

inttime_t find_next_kick(inttime_t Ti_Current);

void init_timebins(double TimeInit);

int is_PM_timestep(inttime_t ti);

#endif
