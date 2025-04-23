#ifndef TIMESTEP_H
#define TIMESTEP_H

#include "utils/paramset.h"
#include "timebinmgr.h"
#include "petapm.h"
#include "domain.h"

/* Structure to hold the kickfactors around the current position on the integer timeline*/
typedef struct
{
    /*Minimum and maximum active and occupied timebins. Initially (but never again) zero*/
    int mintimebin;
    int maxtimebin;
    int mingravtimebin;
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

/* Structure to hold data about the currently active particles. Similar to the WorkQueue structure in treebuild.
 * ActiveParticle may be NULL: if so accesses should be forwarded to the particle manager.*/
typedef struct ActiveParticles
{
    int64_t MaxActiveParticle;
    int64_t NumActiveParticle;
    int64_t NumActiveGravity;
    int64_t NumActiveHydro;
    int *ActiveParticle;
    struct particle_data * Particles;
} ActiveParticles;

/* Initialise an empty active particle list,
 * which will forward requests to the particle manager.
 * No heap memory is allocated.*/
ActiveParticles init_empty_active_particles(struct part_manager_type * PartManager);
/* Build a list of active particles from the particle manager, allocating memory for the active particle list.*/
void build_active_particles(ActiveParticles * act, const DriftKickTimes * const times, const int NumCurrentTiStep, const double Time, const struct part_manager_type * const PartManager);

/* Free the active particle list if necessary*/
void free_active_particles(ActiveParticles * act);
/* Get the current scale factor*/
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
int find_hydro_timesteps(const ActiveParticles * act, DriftKickTimes * times, const double atime, const Cosmology * CP, const int isFirstTimeStep);

/* Apply half a kick to the particles: short-range and long-range.
 * These functions sync drift and kick times.*/
void apply_half_kick(const ActiveParticles * act, Cosmology * CP, DriftKickTimes * times, const double atime);
/* Do hydro kick only*/
void apply_hydro_half_kick(const ActiveParticles * act, Cosmology * CP, DriftKickTimes * times, const double atime);
void apply_PM_half_kick(Cosmology * CP, DriftKickTimes * times);

int is_timebin_active(int i, inttime_t current);

inttime_t find_next_kick(inttime_t Ti_Current, int minTimeBin);

inttime_t init_timebins(double TimeInit);

/* Update the table of active bin drift times */
void update_lastactive_drift(DriftKickTimes * times);

DriftKickTimes init_driftkicktime(inttime_t Ti_Current);

int is_PM_timestep(const DriftKickTimes * const times);

void set_timestep_params(ParameterSet * ps);

/* Stored accelerations.*/
struct grav_accel_store
{
    MyFloat (* GravAccel ) [3];
    int64_t nstore;
};

/* Assigns new short-range timesteps, computes short-range gravitational forces
 * and does the gravitational half-step kicks.
 * Note this does not compute the initial accelerations: hierarchical_gravity_accelerations should be run FIRST.
 * Re-uses the gravity memory from StoredGravAccel.*/
int hierarchical_gravity_and_timesteps(const ActiveParticles * act, PetaPM * pm, DomainDecomp * ddecomp, struct grav_accel_store StoredGravAccel, DriftKickTimes * times, const double atime, int HybridNuGrav, int FastParticleType, Cosmology * CP, const char * EmergencyOutputDir);

/* Computes short-range gravitational forces and
 * do the gravitational half-step kicks. Places the gravitational force into StoredGravAccel.*/
int hierarchical_gravity_accelerations(const ActiveParticles * act, PetaPM * pm, DomainDecomp * ddecomp, struct grav_accel_store StoredGravAccel, DriftKickTimes * times, int HybridNuGrav, Cosmology * CP, const char * EmergencyOutputDir);

/* Updates the Ti_kick times a half-step for this bin*/
void update_kick_times(DriftKickTimes * times);


#endif
