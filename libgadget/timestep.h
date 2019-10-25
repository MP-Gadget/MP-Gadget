#ifndef TIMESTEP_H
#define TIMESTEP_H

#include "timebinmgr.h"
/*Flat array containing all active particles:
set in rebuild_activelist.*/
typedef struct ActiveParticles
{
    int MaxActiveParticle;
    int NumActiveParticle;
    int *ActiveParticle;
} ActiveParticles;

/* variables for organizing PM steps of discrete timeline */
typedef struct {
    inttime_t length; /*!< Duration of the current PM integer timestep*/
    inttime_t start;           /* current start point of the PM step*/
    inttime_t Ti_kick;  /* current inttime of PM Kick (velocity) */
} TimeSpan;

extern TimeSpan PM;

int rebuild_activelist(ActiveParticles * act, inttime_t ti_current, int NumCurrentTiStep);
void free_activelist(ActiveParticles * act);
void set_global_time(double newtime);

/* This function assigns new short-range timesteps to particles.
 * It will also advance the PM timestep and set the new timestep length.
 * Returns the minimum timestep found.*/
int find_timesteps(const ActiveParticles * act, inttime_t Ti_Current);

/* Apply half a kick to the particles: short-range and long-range.
 * These functions sync drift and kick times.*/
void apply_half_kick(const ActiveParticles * act);
void apply_PM_half_kick(void);

int is_timebin_active(int i, inttime_t current);

inttime_t find_next_kick(inttime_t Ti_Current, int minTimeBin);

void init_timebins(double TimeInit);

int is_PM_timestep(inttime_t ti);

#endif
