#ifndef TIMESTEP_H
#define TIMESTEP_H

#include "utils/paramset.h"
#include "timebinmgr.h"
#include "timefac.h"
#include "petapm.h"
#include "domain.h"

/*Flat array containing all active particles:
set in rebuild_activelist.*/
typedef struct ActiveParticles
{
    int64_t MaxActiveParticle;
    int64_t NumActiveParticle;
    int64_t NumActiveGravity;
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
    /* Drift time when this timebin was last active*/
    inttime_t Ti_lastactivedrift[TIMEBINS+1];
    /* Current drift time, which is universal.*/
    inttime_t Ti_Current;
    /* PM Timesteps*/
    inttime_t PM_length; /*!< Duration of the current PM integer timestep*/
    inttime_t PM_start;           /* current start point of the PM step*/
    inttime_t PM_kick;  /* current inttime of PM Kick (velocity) */
} DriftKickTimes;

int rebuild_activelist(ActiveParticles * act, const DriftKickTimes * const times, int NumCurrentTiStep, const double Time);
void free_activelist(ActiveParticles * act);
double get_atime(const inttime_t Ti_Current);

/* This function assigns new short-range timesteps to particles.
 * It will also advance the PM timestep and set the new timestep length.
 * Arguments:
 * ActiveParticles: particles to assign new timesteps for.
 * times: structure of kick times.
 * atime: current scale factor.
 * FastParticleType: particle type (generally neutrinos, 2) which has only long-range timestepping.
 * Cosmology: to compute hubble scaling factors.
 * asmth: size of PM smoothing cell in internal units. asmth = All.Asmth * PartManager->BoxSize / Nmesh
 * isFirstTimeStep: Flags to do special things for BHs on first time step.
 * Returns 0 if success, 1 if timestep is bad.*/
int find_timesteps(const ActiveParticles * act, DriftKickTimes * times, const double atime, int FastParticleType, const Cosmology * CP, const double asmth, const int isFirstTimeStep);
/* Apply half a kick to the particles: short-range and long-range.
 * These functions sync drift and kick times.*/
void apply_half_kick(const ActiveParticles * act, Cosmology * CP, DriftKickTimes * times, const double atime, const double MinEgySpec);
void apply_PM_half_kick(Cosmology * CP, DriftKickTimes * times);

int is_timebin_active(int i, inttime_t current);

inttime_t find_next_kick(inttime_t Ti_Current, int minTimeBin);

inttime_t init_timebins(double TimeInit);

/* Update the table of active bin drift times */
void update_lastactive_drift(DriftKickTimes * times);

DriftKickTimes init_driftkicktime(inttime_t Ti_Current);

int is_PM_timestep(const DriftKickTimes * const times);

void set_timestep_params(ParameterSet * ps);

/* Assigns new short-range timesteps, computes short-range gravitational forces
 * and does the gravitational half-step kicks.
 * Note this does not compute the initial accelerations: _second_half should be run FIRST. */
int do_hierarchical_gravity_first_half(const ActiveParticles * act, PetaPM * pm, DomainDecomp * ddecomp, DriftKickTimes * times, const double atime, int HybridNuGrav, int FastParticleType, Cosmology * CP, const char * EmergencyOutputDir);

/* Computes short-range gravitational forces at the second half of the step and
 * does the gravitational half-step kicks.*/
int do_hierarchical_gravity_second_half(int minTimeBin, const ActiveParticles * act, PetaPM * pm, DomainDecomp * ddecomp, DriftKickTimes * times, const double atime, int HybridNuGrav, int FastParticleType, Cosmology * CP, const char * EmergencyOutputDir);

#endif
