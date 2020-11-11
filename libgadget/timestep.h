#ifndef TIMESTEP_H
#define TIMESTEP_H

#include "utils/paramset.h"
#include "timebinmgr.h"
#include "timefac.h"
/*Flat array containing all active particles:
set in rebuild_activelist.*/
typedef struct ActiveParticles
{
    int MaxActiveParticle;
    int NumActiveParticle;
    int *ActiveParticle;
} ActiveParticles;

/* Structure to hold the kickfactors around the current position on the integer timeline*/
typedef struct
{
    /*Minimum and maximum active and occupied timebins. Initially (but never again) zero*/
    int mintimebin;
    int maxtimebin;
    /* Kick times per bin*/
    inttime_t Ti_kick[TIMEBINS+1];
    /* Current drift time, which is universal.*/
    inttime_t Ti_Current;
} DriftKickTimes;

int rebuild_activelist(ActiveParticles * act, inttime_t ti_current, int NumCurrentTiStep);
void free_activelist(ActiveParticles * act);
void set_global_time(const inttime_t Ti_Current);

/* This function assigns new short-range timesteps to particles.
 * It will also advance the PM timestep and set the new timestep length.
 * Returns the minimum timestep found.*/
void find_timesteps(const ActiveParticles * act, DriftKickTimes * times);

/* Apply half a kick to the particles: short-range and long-range.
 * These functions sync drift and kick times.*/
void apply_half_kick(const ActiveParticles * act, DriftKickTimes * times);
void apply_PM_half_kick(void);

int is_timebin_active(int i, inttime_t current);

inttime_t find_next_kick(inttime_t Ti_Current, int minTimeBin);

inttime_t init_timebins(double TimeInit);

int is_PM_timestep(inttime_t ti);

/* Gets the kick time of the PM step*/
inttime_t get_pm_kick(void);

void set_timestep_params(ParameterSet * ps);

#endif
