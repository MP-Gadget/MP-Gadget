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
int update_active_timebins(unsigned int next_kick);
void rebuild_activelist(void);
void set_global_time(double newtime);
void advance_and_find_timesteps(int do_half_kick);
void apply_half_kick(void);
double find_dloga_displacement_constraint(void);
int is_timebin_active(int i);
void set_timebin_active(binmask_t mask);

void sph_VelPred(int i, double * VelPred);
double EntropyPred(int i);
double PressurePred(int i);

unsigned int find_next_kick(unsigned int Ti_Current);
unsigned int find_next_outputtime(unsigned int ti_curr);
void init_timebins(void);

int is_PM_timestep(unsigned int ti);

#endif
