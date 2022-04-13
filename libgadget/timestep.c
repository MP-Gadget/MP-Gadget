#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

#include "utils.h"
#include "physconst.h"
#include "timebinmgr.h"
#include "domain.h"
#include "timefac.h"
#include "cosmology.h"
#include "checkpoint.h"
#include "slotsmanager.h"
#include "partmanager.h"
#include "hydra.h"
#include "walltime.h"
#include "timestep.h"
#include "gravity.h"

/*! \file timestep.c
 *  \brief routines for 'kicking' particles in
 *  momentum space and assigning new timesteps
 */
static struct timestep_params
{
    /* adjusts accuracy of time-integration */

    double ErrTolIntAccuracy;   /*!< accuracy tolerance parameter \f$ \eta \f$ for timestep criterion. The
                                  timesteps is \f$ \Delta t = \sqrt{\frac{2 \eta eps}{a}} \f$ */

    int ForceEqualTimesteps; /*If true, all timesteps have the same timestep, the smallest allowed.*/
    double MinSizeTimestep,     /*!< minimum allowed timestep. Normally, the simulation terminates if the
                              timestep determined by the timestep criteria falls below this limit. */
           MaxSizeTimestep;     /*!< maximum allowed timestep */

    double MaxRMSDisplacementFac;   /*!< this determines a global timestep criterion for cosmological simulations
                                      in comoving coordinates.  To this end, the code computes the rms velocity
                                      of all particles, and limits the timestep such that the rms displacement
                                      is a fraction of the mean particle separation (determined from the
                                      particle mass and the cosmological parameters). This parameter specifies
                                      this fraction. */

    double MaxGasVel; /* Limit on Gas velocity */
    double CourantFac;		/*!< SPH-Courant factor */
} TimestepParams;

/*Set the parameters of the hydro module*/
void
set_timestep_params(ParameterSet * ps)
{
    int ThisTask;
    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
    if(ThisTask == 0) {
        TimestepParams.ErrTolIntAccuracy = param_get_double(ps, "ErrTolIntAccuracy");
        TimestepParams.MaxGasVel = param_get_double(ps, "MaxGasVel");
        TimestepParams.MaxSizeTimestep = param_get_double(ps, "MaxSizeTimestep");

        TimestepParams.MinSizeTimestep = param_get_double(ps, "MinSizeTimestep");
        TimestepParams.ForceEqualTimesteps = param_get_int(ps, "ForceEqualTimesteps");
        TimestepParams.MaxRMSDisplacementFac = param_get_double(ps, "MaxRMSDisplacementFac");
        TimestepParams.CourantFac = param_get_double(ps, "CourantFac");
    }
    MPI_Bcast(&TimestepParams, sizeof(struct timestep_params), MPI_BYTE, 0, MPI_COMM_WORLD);
}

static inline int get_active_particle(const ActiveParticles * act, int pa)
{
    if(act->ActiveParticle)
        return act->ActiveParticle[pa];
    else
        return pa;
}

static int
timestep_eh_slots_fork(EIBase * event, void * userdata)
{
    /*Update the active particle list:
     * if the parent is active the child should also be active.
     * Stars must always be active on formation, but
     * BHs need not be: a halo can be seeded when the particle in question is inactive.*/

    EISlotsFork * ev = (EISlotsFork *) event;

    int parent = ev->parent;
    int child = ev->child;
    ActiveParticles * act = (ActiveParticles *) userdata;

    if(is_timebin_active(P[parent].TimeBinHydro, P[parent].Ti_drift)) {
        int64_t childactive = atomic_fetch_and_add_64(&act->NumActiveParticle, 1);
        if(act->ActiveParticle) {
            /* This should never happen because we allocate as much space for active particles as we have space
             * for particles, but just in case*/
            if(childactive >= act->MaxActiveParticle)
                endrun(5, "Tried to add %ld active particles, more than %ld allowed\n", childactive, act->MaxActiveParticle);
            act->ActiveParticle[childactive] = child;
        }
    }
    return 0;
}

/* Enum for keeping track of which
 * timestep criterion is limiting each particles'
 * timestep evolution*/
enum TimeStepType
{
    TI_ACCEL = 0,
    TI_COURANT = 1,
    TI_ACCRETE = 2,
    TI_NEIGH = 3,
    TI_HSML = 4,
};

static double get_timestep_gravity_dloga(const int p, const double atime, const double hubble);
static double get_timestep_hydro_dloga(const int p, const inttime_t Ti_Current, const double atime, const double hubble, enum TimeStepType * titype);
static inttime_t convert_timestep_to_ti(double dloga, const int p, const inttime_t dti_max, const inttime_t Ti_Current, enum TimeStepType titype);
static int get_timestep_bin(inttime_t dti);
static void do_grav_short_range_kick(struct particle_data * part, const double Fgravkick);
static void do_hydro_kick(int i, double dt_entr, double Fgravkick, double Fhydrokick, const double atime, const double MinEgySpec);

/* Hierarchical gravity functions*/
/* Build a sublist of particles gravitationally active and smaller than a timebin*/
int build_active_sublist(ActiveParticles * sub_act, const ActiveParticles * act, const int maxtimebin);

/* Get the current PM (global) timestep.*/
static inttime_t get_PM_timestep_ti(const DriftKickTimes * const times, const double atime, const Cosmology * CP, const int FastParticleType, const double asmth);

/*Initialise the integer timeline*/
inttime_t
init_timebins(double TimeInit)
{
    inttime_t Ti_Current = ti_from_loga(log(TimeInit));
    /*Enforce Ti_Current is initially even*/
    if(Ti_Current % 2 == 1)
        Ti_Current++;
    message(0, "Initial TimeStep at TimeInit %g Ti_Current = %d \n", TimeInit, Ti_Current);
    return Ti_Current;
}

DriftKickTimes init_driftkicktime(inttime_t Ti_Current)
{
    DriftKickTimes times = {0};
    times.Ti_Current = Ti_Current;
    int i;
    for(i = 0; i <= TIMEBINS; i++) {
        times.Ti_kick[i] = times.Ti_Current;
        times.Ti_lastactivedrift[i] = times.Ti_Current;
    }
    /* this makes sure the first step is a PM step. */
    times.PM_length = 0;
    times.PM_kick = times.Ti_Current;
    times.PM_start = times.Ti_Current;
    return times;
}

int is_timebin_active(int i, inttime_t current) {
    /*Bin 0 is always active and at time 0 all bins are active*/
    if(i <= 0 || current <= 0)
        return 1;
    if(current % dti_from_timebin(i) == 0)
        return 1;
    return 0;
}

/*Report whether the current timestep is the end of the PM timestep*/
int
is_PM_timestep(const DriftKickTimes * const times)
{
    if(times->Ti_Current > times->PM_start + times->PM_length)
        endrun(12, "Passed end of PM step! ti=%d, PM = %d + %d\n",times->Ti_Current, times->PM_start, times->PM_length);
    return times->Ti_Current == times->PM_start + times->PM_length;
}

double
get_atime(const inttime_t Ti_Current) {
    return exp(loga_from_ti(Ti_Current));
}

static int
get_timebin_from_dti(inttime_t dti, int binold, int * badstepsizecount, DriftKickTimes * times)
{
        /* make it a power 2 subdivision */
        dti = round_down_power_of_two(dti);

        int bin = get_timestep_bin(dti);
        if(bin < 1) {
//             message(1, "Time-step of integer size %d not allowed, id = %lu, debugging info follows.\n", dti, P[i].ID);
            (*badstepsizecount)++;
        }

        /* timestep wants to increase */
        if(bin > binold)
        {
            /* make sure the new step is currently active,
             * so that particles do not miss a step */
            while(!is_timebin_active(bin, times->Ti_Current) && bin > binold && bin > 1)
                bin--;
        }
        return bin;
}

/* Find the single global timestep for when the timesteps are supposed to be synchronised.*/
static inttime_t
find_global_timestep(DriftKickTimes * times, const inttime_t dti_max, const double atime, const double hubble)
{
        inttime_t dti_min;
        int i;
        #pragma omp parallel for reduction(min:dti_min)
        for(i = 0; i < PartManager->NumPart; i++)
        {
            enum TimeStepType titype;
            /* Because we don't GC on short timesteps, there can be garbage here.
             * Avoid making it active. */
            if(P[i].IsGarbage || P[i].Swallowed)
                continue;
            /* Use hydro timestep as it is always shorter*/
            double dloga = get_timestep_hydro_dloga(i, times->Ti_Current, atime, hubble, &titype);
            inttime_t dti = convert_timestep_to_ti(dloga, i, dti_max, times->Ti_Current, titype);
            if(dti < dti_min)
                dti_min = dti;
        }
        MPI_Allreduce(MPI_IN_PLACE, &dti_min, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        return dti_min;
}


/* Assigns new short-range timesteps, computes short-range gravitational forces
 * and does the gravitational half-step kicks.*/
int
hierarchical_gravity_and_timesteps(const ActiveParticles * act, PetaPM * pm, DomainDecomp * ddecomp, DriftKickTimes * times, const double atime, int HybridNuGrav, int FastParticleType, Cosmology * CP, const char * EmergencyOutputDir)
{
    walltime_measure("/Misc");

    /*Update the PM timestep size */
    const int isPM = is_PM_timestep(times);
    inttime_t dti_max = times->PM_length;

    if(isPM) {
        dti_max = get_PM_timestep_ti(times, atime, CP, FastParticleType, pm->Asmth);
        times->PM_length = dti_max;
        times->PM_start = times->PM_kick;
    }

    const double hubble = hubble_function(CP, atime);
    /* Find the longest active timebin. Usually the PM step*/
    int ti, largest_active = TIMEBASE;
    for(ti = dti_max; ti > 0; ti--) {
        if(is_timebin_active(ti, times->Ti_Current) && dti_from_timebin(ti) <= times->PM_length) {
            largest_active = ti;
            break;
        }
    }
    /* Move all particles to largest active timebin, then drop them down below.*/
    int i;
    for(i = 0; i < act->NumActiveGravity; i++) {
        int pa = act->ActiveParticle ? act->ActiveParticle[i] : i;
        P[pa].TimeBinGravity = largest_active;
    }
    int mTimeBin = largest_active, maxTimeBin = times->maxtimebin;
    int64_t badstepsizecount = 0;
    int64_t last_active, total_part, last_active_loc = act->NumActiveGravity;
    MPI_Allreduce(&PartManager->NumPart, &total_part, 1, MPI_INT64, MPI_SUM, MPI_COMM_WORLD);
    /* Set timebins to largest value */
    for(ti = largest_active; ti > 0; ti--) {
        /* Do nothing if no particles are active*/
        ActiveParticles subact[1] = {0};
        build_active_sublist(subact, act, ti);
        int64_t tot_active;
        MPI_Allreduce(&last_active_loc, &last_active, 1, MPI_INT64, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&subact->NumActiveGravity, &tot_active, 1, MPI_INT64, MPI_SUM, MPI_COMM_WORLD);
        if(tot_active == 0) {
            mTimeBin = ti;
            break;
        }
        /* Do not need to do the accelerations for timesteps where all particles are still active.
         * In this case just shrink the timesteps more.*/
        if(tot_active < last_active) {
            /* Tree with only particle timesteps below this value*/
            ForceTree Tree = {0};
            force_tree_rebuild(&Tree, ddecomp, subact, HybridNuGrav, 1, EmergencyOutputDir);
            const double rho0 = CP->Omega0 * 3 * CP->Hubble * CP->Hubble / (8 * M_PI * CP->GravInternal);
            grav_short_tree(act, pm, &Tree, rho0, HybridNuGrav, FastParticleType, times->Ti_Current);
            force_tree_free(&Tree);
        }
        /* This finds the smallest timestep which contains all particles and later shrinks the PM timestep to this value.*/
        if(isPM && (tot_active == total_part))
            maxTimeBin = ti;
        /* We need to compute the new timestep here based on the acceleration at the current level,
         * because we will over-write the acceleration*/
        int i, nactive_next = 0;
        #pragma omp parallel for reduction(+: nactive_next) reduction(max:maxTimeBin)
        for(i = 0; i < subact->NumActiveGravity; i++) {
            int pa = subact->ActiveParticle ? subact->ActiveParticle[i] : i;
            double dloga_gravity = get_timestep_gravity_dloga(i, atime, hubble);
            inttime_t dti_gravity = convert_timestep_to_ti(dloga_gravity, i, dti_max, times->Ti_Current, TI_ACCEL);
            /* Reduce the timebin by 1 if needed by this current acceleration.*/
            if(dti_gravity < dti_from_timebin(ti)) {
                P[pa].TimeBinGravity = ti -1;
                nactive_next++;
            }
        }
        if(ti == 1)
            badstepsizecount = nactive_next;
        /* This tests COLLECTIVELY for the timestep needing to shrink.
           If we are the topmost timestep and it needs to shrink for more than 33% of the particles,
           shrink it for all of them. Then we don't need to recompute the accelerations (because
           they are still the same, and are from all particles).*/
        int64_t nactive_next_tot;
        sumup_large_ints(1, &nactive_next, &nactive_next_tot);
        if(tot_active == total_part && nactive_next_tot < tot_active && nactive_next_tot > tot_active / 3 ) {
            #pragma omp parallel for reduction(+: nactive_next) reduction(max:maxTimeBin)
            for(i = 0; i < subact->NumActiveGravity; i++) {
                int pa = subact->ActiveParticle ? subact->ActiveParticle[i] : i;
                P[pa].TimeBinGravity = ti -1;
            }
        }

        /* Now we do the gravity kicks using each half-step acceleration.*/
        inttime_t dti = dti_from_timebin(ti);
        /* Compute kick factors for occupied bins*/
        /* Go forwards a halfstep for the current bin*/
        double gravkick = get_exact_gravkick_factor(CP, times->Ti_kick[ti], times->Ti_kick[ti] + dti/2);
        /* Go backwards a halfstep for the timestep above this one*/;
        if(ti < largest_active) {
            inttime_t lowerdti = dti_from_timebin(ti+1);
            const double lowerkick = get_exact_gravkick_factor(CP, times->Ti_kick[ti+1], times->Ti_kick[ti+1] + lowerdti/2);
            gravkick -= lowerkick;
        }
        /* Do the kick, changing velocity.*/
        for(i = 0; i < subact->NumActiveGravity; i++) {
            int pa = subact->ActiveParticle ? subact->ActiveParticle[i] : i;
            do_grav_short_range_kick(&P[pa], gravkick);
        }
        last_active = subact->NumActiveGravity;
        myfree(subact->ActiveParticle);
    }
    MPI_Allreduce(MPI_IN_PLACE, &badstepsizecount, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &mTimeBin, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &maxTimeBin, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

    /* Ensure that the PM timestep is not longer than the longest tree timestep;
     * this prevents particles in the longest timestep being active and moving into a higher bin
     * between PM timesteps, thus skipping the PM step entirely.*/
    if(isPM && times->PM_length > dti_from_timebin(maxTimeBin))
        times->PM_length = dti_from_timebin(maxTimeBin);
    message(0, "PM timebin: %x (dloga: %g Max: %g).\n", times->PM_length, dloga_from_dti(times->PM_length, times->Ti_Current), TimestepParams.MaxSizeTimestep);
    times->mintimebin = mTimeBin;
    times->maxtimebin = maxTimeBin;
    return badstepsizecount;;
}

/* Computes short-range gravitational forces at the second half of the step and
 * does the gravitational half-step kicks.*/
int hierarchical_gravity_accelerations(int minTimeBin, const ActiveParticles * act, PetaPM * pm, DomainDecomp * ddecomp, DriftKickTimes * times, int HybridNuGrav, int FastParticleType, Cosmology * CP, const char * EmergencyOutputDir)
{
    walltime_measure("/Misc");

    /* Find the longest active timebin. Usually the PM step*/
    int ti, largest_active = TIMEBASE;
    for(ti = times->PM_length; ti > 0; ti--) {
        if(is_timebin_active(ti, times->Ti_Current) && dti_from_timebin(ti) <= times->PM_length) {
            largest_active = ti;
            break;
        }
    }
    /* Set timebins to largest value */
    for(ti = minTimeBin; ti <= largest_active; ti++) {
        ActiveParticles subact[1] = {0};
        build_active_sublist(subact, act, ti);
        /* Tree with moments but only particle timesteps below this value*/
        ForceTree Tree = {0};
        force_tree_rebuild(&Tree, ddecomp, subact, HybridNuGrav, 1, EmergencyOutputDir);
        const double rho0 = CP->Omega0 * 3 * CP->Hubble * CP->Hubble / (8 * M_PI * CP->GravInternal);
        grav_short_tree(act, pm, &Tree, rho0, HybridNuGrav, FastParticleType, times->Ti_Current);
        force_tree_free(&Tree);
        /* We need to compute the new timestep here based on the acceleration at the current level,
         * because we will over-write the acceleration*/
        /* Now we do the gravity kicks using each half-step acceleration.*/
        inttime_t dti = dti_from_timebin(ti);
        /* Compute kick factors for occupied bins*/
        /* Go forwards a halfstep for the current bin*/
        double gravkick = get_exact_gravkick_factor(CP, times->Ti_kick[ti], times->Ti_kick[ti] + dti/2);
        if(ti < largest_active) {
            inttime_t lowerdti = dti_from_timebin(ti+1)
            /* Go backwards a halfstep for the timestep above this one*/;
            const double lowerkick = get_exact_gravkick_factor(CP, times->Ti_kick[ti+1], times->Ti_kick[ti+1] + lowerdti/2);
            gravkick -= lowerkick;
        }
        int i;
        /* Do the kick, changing velocity.*/
        for(i = 0; i < subact->NumActiveGravity; i++) {
            int pa = subact->ActiveParticle ? subact->ActiveParticle[i] : i;
            do_grav_short_range_kick(&P[pa], gravkick);
        }
        myfree(subact->ActiveParticle);
        /* FIXME: Set SPH Predicted velocities for inactive particles.*/
    }
    return 0;
}

/* This function assigns new short-range timesteps to particles.
 * It will also shrink the PM timestep to the longest short-range timestep.
 * Stores the maximum and minimum timesteps in the DriftKickTimes structure.*/
int
find_timesteps(const ActiveParticles * act, DriftKickTimes * times, const double atime, int FastParticleType, const Cosmology * CP, const double asmth, const int isFirstTimeStep)
{
    int pa;
    inttime_t dti_min = TIMEBASE;

    walltime_measure("/Misc");

    /*Update the PM timestep size */
    const int isPM = is_PM_timestep(times);
    inttime_t dti_max = times->PM_length;

    if(isPM) {
        dti_max = get_PM_timestep_ti(times, atime, CP, FastParticleType, asmth);
        times->PM_length = dti_max;
        times->PM_start = times->PM_kick;
    }

    const double hubble = hubble_function(CP, atime);
    /* Now assign new timesteps and kick */
    if(TimestepParams.ForceEqualTimesteps) {
        dti_min = find_global_timestep(times, dti_max, atime, hubble);
    }

    int64_t ntiaccel=0, nticourant=0, ntiaccrete=0, ntineighbour=0, ntihsml=0;
    int badstepsizecount = 0;
    int mTimeBin = TIMEBINS, maxTimeBin = 0;
    #pragma omp parallel for reduction(min: mTimeBin) reduction(+: badstepsizecount, ntiaccel, nticourant, ntiaccrete, ntineighbour, ntihsml) reduction(max:maxTimeBin)
    for(pa = 0; pa < act->NumActiveParticle; pa++)
    {
        const int i = get_active_particle(act, pa);

        if(P[i].IsGarbage || P[i].Swallowed)
            continue;

        enum TimeStepType titype = TI_ACCEL;
        inttime_t dti_gravity, dti_hydro;
        if(TimestepParams.ForceEqualTimesteps) {
            dti_gravity = dti_min;
            dti_hydro = dti_min;
        } else {
            /* Compute gravity timestep*/
            double dloga_gravity = get_timestep_gravity_dloga(i, atime, hubble);
            dti_gravity = convert_timestep_to_ti(dloga_gravity, i, dti_max, times->Ti_Current, titype);
            /* Do hydro timestep for gas or BHs. Always shorter*/
            if(P[i].Type == 0 || P[i].Type == 5) {
                double dloga_hydro = get_timestep_hydro_dloga(i, times->Ti_Current, atime, hubble, &titype);
                dti_hydro = convert_timestep_to_ti(dloga_hydro, i, dti_max, times->Ti_Current, titype);
            }
            /* Just in case*/
            else
                dti_hydro = dti_gravity;
            /* Type of shortest timestep criterion. Note that gravity is always TI_ACCEL.*/
            if(titype == TI_ACCEL)
                ntiaccel++;
            else if (titype == TI_COURANT)
                nticourant++;
            else if (titype == TI_ACCRETE)
                ntiaccrete++;
            else if (titype == TI_NEIGH)
                ntineighbour++;
            else if (titype == TI_HSML)
                ntihsml++;
        }
        /* Find a new particle bin.
         * active particles always remain active
         * until rebuild_activelist is called
         * (after domain, on new timestep).*/
        int bin_gravity = get_timebin_from_dti(dti_gravity, P[i].TimeBinGravity, &badstepsizecount, times);
        int bin_hydro = get_timebin_from_dti(dti_hydro, P[i].TimeBinHydro, &badstepsizecount, times);
        /* Enforce that the hydro timestep is always shorter than or equal to the gravity timestep*/
        if(bin_hydro > bin_gravity)
            bin_hydro = bin_gravity;
        /* Enforce that the gravity bin is the hydro bin. This is to ensure the code continues to
         * work while we implement the Hamiltonian timesplitting */
        bin_gravity = bin_hydro;
        /* Only update if both the old and new timebins are currently active.
         * We know that the shorter hydro timestep is active, but we need to check
         * the gravity timestep.*/
        if(is_timebin_active(P[i].TimeBinHydro, times->Ti_Current) && is_timebin_active(bin_hydro, times->Ti_Current))
            P[i].TimeBinHydro = bin_hydro;
        if(is_timebin_active(P[i].TimeBinGravity, times->Ti_Current) && is_timebin_active(bin_gravity, times->Ti_Current))
            P[i].TimeBinGravity= bin_gravity;
        /*Find max and min*/
        if(bin_hydro < mTimeBin)
            mTimeBin = bin_hydro;
        if(bin_gravity > maxTimeBin)
            maxTimeBin = bin_gravity;
    }

    MPI_Allreduce(MPI_IN_PLACE, &badstepsizecount, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &mTimeBin, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &maxTimeBin, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

    MPI_Allreduce(MPI_IN_PLACE, &ntiaccel, 1, MPI_INT64, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &nticourant, 1, MPI_INT64, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &ntihsml, 1, MPI_INT64, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &ntiaccrete, 1, MPI_INT64, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &ntineighbour, 1, MPI_INT64, MPI_SUM, MPI_COMM_WORLD);

    /* Ensure that the PM timestep is not longer than the longest tree timestep;
     * this prevents particles in the longest timestep being active and moving into a higher bin
     * between PM timesteps, thus skipping the PM step entirely.*/
    if(isPM && times->PM_length > dti_from_timebin(maxTimeBin))
        times->PM_length = dti_from_timebin(maxTimeBin);
    message(0, "PM timebin: %x (dloga: %g Max: %g). Criteria: Accel: %ld Soundspeed: %ld DivVel: %ld Accrete: %ld Neighbour: %ld\n",
            times->PM_length, dloga_from_dti(times->PM_length, times->Ti_Current), TimestepParams.MaxSizeTimestep,
            ntiaccel, nticourant, ntihsml, ntiaccrete, ntineighbour);

    /* BH particles have their timesteps set by a timestep limiter.
     * On the first timestep this is not effective because all the particles have zero timestep.
     * So on the first timestep only set all BH particles to the smallest allowable timestep.
     * Note we can leave the gravitational timestep as set by the acceleration: repositioning may take care of it.*/
    if(isFirstTimeStep) {
        #pragma omp parallel for
        for(pa = 0; pa < PartManager->NumPart; pa++)
        {
            if(P[pa].Type == 5)
                P[pa].TimeBinHydro = mTimeBin;
        }
    }
    walltime_measure("/Timeline");
    times->mintimebin = mTimeBin;
    times->maxtimebin = maxTimeBin;
    return badstepsizecount;
}

/* Update the last active drift times for all bins*/
void
update_lastactive_drift(DriftKickTimes * times)
{
    int bin;
    #pragma omp parallel for
    for(bin = 0; bin <= TIMEBINS; bin++)
    {
        /* Update active timestep even if no particles in it*/
        if(is_timebin_active(bin, times->Ti_Current))
            times->Ti_lastactivedrift[bin] = times->Ti_Current;
    }
}

/* Apply half a kick, for the second half of the timestep.*/
void
apply_half_kick(const ActiveParticles * act, Cosmology * CP, DriftKickTimes * times, const double atime, const double MinEgySpec)
{
    int pa, bin;
    walltime_measure("/Misc");
    double gravkick[TIMEBINS+1] = {0}, hydrokick[TIMEBINS+1] = {0};
    /* Do nothing for the first timestep when the kicks are always zero*/
    if(times->mintimebin == 0 && times->maxtimebin == 0)
        return;
    #pragma omp parallel for
    for(bin = times->mintimebin; bin <= TIMEBINS; bin++)
    {
        /* Kick the active timebins*/
        if(is_timebin_active(bin, times->Ti_Current)) {
            /* do the kick for half a step*/
            inttime_t dti = dti_from_timebin(bin);
            inttime_t newkick = times->Ti_kick[bin] + dti/2;
            /* Compute kick factors for occupied bins*/
            gravkick[bin] = get_exact_gravkick_factor(CP, times->Ti_kick[bin], newkick);
            hydrokick[bin] = get_exact_hydrokick_factor(CP, times->Ti_kick[bin], newkick);
      //      message(0, "drift %d bin %d kick: %d->%d\n", times->Ti_Current, bin, times->Ti_kick[bin], newkick);
            times->Ti_kick[bin] = newkick;
        }
    }
    /* Advance the shorter bins without particles by the minimum occupied timestep.*/
    for(bin=1; bin < times->mintimebin; bin++)
        times->Ti_kick[bin] += dti_from_timebin(times->mintimebin)/2;
    //    message(0, "drift %d bin %d kick: %d\n", times->Ti_Current, bin, times->Ti_kick[bin]);
    /* Now assign new timesteps and kick */
    #pragma omp parallel for
    for(pa = 0; pa < act->NumActiveParticle; pa++)
    {
        const int i = get_active_particle(act, pa);
        if(P[i].Swallowed || P[i].IsGarbage)
            continue;
        int bin_gravity = P[i].TimeBinGravity;
        if(bin_gravity > TIMEBINS)
            endrun(4, "Particle %d (type %d, id %ld) had unexpected timebin %d\n", i, P[i].Type, P[i].ID, P[i].TimeBinGravity);
        /* Kick active gravity particles*/
        if(is_timebin_active(bin_gravity, times->Ti_Current)) {
            do_grav_short_range_kick(&P[i], gravkick[bin_gravity]);
#ifdef DEBUG
            if(P[i].Ti_kick_grav != times->Ti_kick[bin_gravity] - dti_from_timebin(bin_gravity)/2)
                endrun(4, "Particle %d (type %d, id %ld bin %d dt %x gen %d) had grav kick time %x not %x\n",
                       i, P[i].Type, P[i].ID, P[i].TimeBinGravity, dti_from_timebin(bin_gravity)/2, P[i].Generation, P[i].Ti_kick_grav, times->Ti_kick[bin_gravity] - dti_from_timebin(bin_gravity)/2);
            P[i].Ti_kick_grav = times->Ti_kick[bin_gravity];
#endif
        }
        /* Hydro kick for hydro particles*/
        if(P[i].Type == 0 || P[i].Type == 5) {
            int bin_hydro = P[i].TimeBinHydro;
            inttime_t dti = dti_from_timebin(bin_hydro);
            const double dt_entr = dloga_from_dti(dti/2, times->Ti_Current);
            /*This only changes particle i, so is thread-safe.*/
            do_hydro_kick(i, dt_entr, gravkick[bin_hydro], hydrokick[bin_hydro], atime, MinEgySpec);
#ifdef DEBUG
            if(P[i].Ti_kick_hydro != times->Ti_kick[bin_hydro] - dti_from_timebin(bin_hydro)/2)
                endrun(4, "Particle %d (type %d, id %ld bin %d) had hydro kick time %ld not %ld\n",
                       i, P[i].Type, P[i].ID, P[i].TimeBinHydro, P[i].Ti_kick_hydro, times->Ti_kick[bin_hydro] - dti_from_timebin(bin_hydro)/2);
            P[i].Ti_kick_hydro = times->Ti_kick[bin_hydro];
#endif
        }
    }
    walltime_measure("/Timeline/HalfKick/Short");
}

void
apply_PM_half_kick(Cosmology * CP, DriftKickTimes * times)
{
    /*Always do a PM half-kick, because this should be called just after a PM step*/
    const inttime_t tistart = times->PM_kick;
    const inttime_t tiend =  tistart + times->PM_length / 2;
    /* Do long-range kick */
    int i;
    const double Fgravkick = get_exact_gravkick_factor(CP, tistart, tiend);

    #pragma omp parallel for
    for(i = 0; i < PartManager->NumPart; i++)
    {
        int j;
        if(P[i].Swallowed || P[i].IsGarbage)
            continue;
        for(j = 0; j < 3; j++)	/* do the kick */
            P[i].Vel[j] += P[i].GravPM[j] * Fgravkick;
    }
    times->PM_kick = tiend;
    walltime_measure("/Timeline/HalfKick/Long");
}

/* Add gravitational kick to current particle*/
void
do_grav_short_range_kick(struct particle_data * part, const double Fgravkick)
{
    int j;
    for(j = 0; j < 3; j++)
        part->Vel[j] += part->GravAccel[j] * Fgravkick;
}

void
do_hydro_kick(int i, double dt_entr, double Fgravkick, double Fhydrokick, const double atime, const double MinEgySpec)
{
    int j;
    /* Add kick from dynamic friction and hydro drag for BHs. */
    if(P[i].Type == 5) {
        for(j = 0; j < 3; j++){
            P[i].Vel[j] += BHP(i).DFAccel[j] * Fgravkick;
            P[i].Vel[j] += BHP(i).DragAccel[j] * Fgravkick;
        }
    }

    if(P[i].Type == 0) {
        /* Add kick from hydro and SPH stuff */
        for(j = 0; j < 3; j++) {
            P[i].Vel[j] += SPHP(i).HydroAccel[j] * Fhydrokick;
        }
        /* Code here imposes a hard limit (default to speed of light)
         * on the gas velocity. This should rarely be hit.*/
        double vv=0;
        for(j=0; j < 3; j++)
            vv += P[i].Vel[j] * P[i].Vel[j];
        vv = sqrt(vv);

        if(vv > 0 && vv/atime > TimestepParams.MaxGasVel) {
            message(1,"Gas Particle ID %ld exceeded the gas velocity limit: %g > %g\n",P[i].ID, vv / atime, TimestepParams.MaxGasVel);
            for(j=0;j < 3; j++)
                P[i].Vel[j] *= TimestepParams.MaxGasVel * atime / vv;
        }

        /* Update entropy for adiabatic change*/
        SPHP(i).Entropy += SPHP(i).DtEntropy * dt_entr;

        /* Limit entropy in simulations with cooling disabled*/
        double a3inv = 1/(atime * atime * atime);
        const double enttou = pow(SPHP(i).Density * a3inv, GAMMA_MINUS1) / GAMMA_MINUS1;
        if(SPHP(i).Entropy < MinEgySpec/enttou)
            SPHP(i).Entropy = MinEgySpec / enttou;
    }
#ifdef DEBUG
    /* Check we have reasonable velocities. If we do not, try to explain why*/
    if(isnan(P[i].Vel[0]) || isnan(P[i].Vel[1]) || isnan(P[i].Vel[2])) {
        message(1, "Vel = %g %g %g Type = %d gk = %g a_g = %g %g %g\n",
                P[i].Vel[0], P[i].Vel[1], P[i].Vel[2], P[i].Type,
                Fgravkick, P[i].GravAccel[0], P[i].GravAccel[1], P[i].GravAccel[2]);
    }
#endif
}

static double
get_timestep_gravity_dloga(const int p, const double atime, const double hubble)
{
    /*Compute physical acceleration*/
    const double a2inv = 1/(atime * atime);
    double ax = a2inv * P[p].GravAccel[0];
    double ay = a2inv * P[p].GravAccel[1];
    double az = a2inv * P[p].GravAccel[2];

    ay += a2inv * P[p].GravPM[1];
    ax += a2inv * P[p].GravPM[0];
    az += a2inv * P[p].GravPM[2];

    double ac = sqrt(ax * ax + ay * ay + az * az);	/* this is now the physical acceleration */

    if(ac == 0)
        ac = 1.0e-30;

    /* mind the factor 2.8 difference between gravity and softening used here. */
    double dt = sqrt(2 * TimestepParams.ErrTolIntAccuracy * atime * (FORCE_SOFTENING(p, P[p].Type) / 2.8) / ac);

    /* d a / a = dt * H */
    double dloga = dt * hubble;
    return dloga;
}

static double
get_timestep_hydro_dloga(const int p, const inttime_t Ti_Current, const double atime, const double hubble, enum TimeStepType * titype)
{
    double ac = 0;
    double dt = 0, dt_courant = 0, dt_hsml = 0;

    /*Compute physical acceleration*/
    {
        const double a2inv = 1/(atime * atime);

        double ax = a2inv * P[p].GravAccel[0];
        double ay = a2inv * P[p].GravAccel[1];
        double az = a2inv * P[p].GravAccel[2];

        ay += a2inv * P[p].GravPM[1];
        ax += a2inv * P[p].GravPM[0];
        az += a2inv * P[p].GravPM[2];

        if(P[p].Type == 0)
        {
            const double fac2 = 1 / pow(atime, 3 * GAMMA - 2);
            ax += fac2 * SPHP(p).HydroAccel[0];
            ay += fac2 * SPHP(p).HydroAccel[1];
            az += fac2 * SPHP(p).HydroAccel[2];
        }

        ac = sqrt(ax * ax + ay * ay + az * az);	/* this is now the physical acceleration */
    }

    if(ac == 0)
        ac = 1.0e-30;

    /* mind the factor 2.8 difference between gravity and softening used here. */
    dt = sqrt(2 * TimestepParams.ErrTolIntAccuracy * atime * (FORCE_SOFTENING(p, P[p].Type) / 2.8) / ac);
    *titype = TI_ACCEL;

    if(P[p].Type == 0)
    {
        const double fac3 = pow(atime, 3 * (1 - GAMMA) / 2.0);
        dt_courant = 2 * TimestepParams.CourantFac * atime * P[p].Hsml / (fac3 * SPHP(p).MaxSignalVel);
        if(dt_courant < dt) {
            dt = dt_courant;
            *titype = TI_COURANT;
        }
        /* This timestep criterion is from Gadget-4, eq. 0 of 2010.03567 and stops
         * particles having too large a density change.*/
        dt_hsml = TimestepParams.CourantFac * atime * atime * fabs(P[p].Hsml / (P[p].DtHsml + 1e-20));
        if(dt_hsml < dt) {
            dt = dt_hsml;
            *titype = TI_HSML;
        }
    }

    if(P[p].Type == 5)
    {
        if(BHP(p).Mdot > 0 && BHP(p).Mass > 0)
        {
            double dt_accr = 0.25 * BHP(p).Mass / BHP(p).Mdot;
            if(dt_accr < dt) {
                dt = dt_accr;
                *titype = TI_ACCRETE;
            }
        }
        if(BHP(p).minTimeBin > 0 && BHP(p).minTimeBin+1 < TIMEBINS) {
            double dt_limiter = get_dloga_for_bin(BHP(p).minTimeBin+1, Ti_Current) / hubble;
            /* Set the black hole timestep to the minimum timesteps of neighbouring gas particles.
             * It should be at least this for accretion accuracy, and it does not make sense to
             * make it less than this. We go one timestep up because often the smallest
             * timebin particle is cooling, and so increases its timestep. Then the smallest timebin
             * contains only the BH which doesn't make much numerical sense. Accretion accuracy is not much changed
             * by one timestep difference.*/
            dt = dt_limiter;
            *titype = TI_NEIGH;
        }
    }

    /* d a / a = dt * H */
    double dloga = dt * hubble;

    return dloga;
}

/*! This function returns the maximum allowed timestep of a particle, expressed in
 *  terms of the integer mapping that is used to represent the total simulated timespan.
 *  Arguments:
 *  dloga -> timestep in dloga units
 *  p -> particle index
 *  dti_max -> maximal timestep.
 *  Ti_Current -> current integer timeline
 *  titype -> type of timestep limit that was used for debugging. */
static inttime_t
convert_timestep_to_ti(double dloga, const int p, const inttime_t dti_max, const inttime_t Ti_Current, enum TimeStepType titype)
{
    inttime_t dti;
    /*Give a useful message if we are broken*/
    if(dti_max == 0)
        return 0;

    if(dloga < TimestepParams.MinSizeTimestep)
        dloga = TimestepParams.MinSizeTimestep;

    dti = dti_from_dloga(dloga, Ti_Current);

    /* Check for overflow*/
    if(dti > dti_max || dti < 0)
        dti = dti_max;

    if(dti <= 1 || dti > (inttime_t) TIMEBASE)
    {
        if(P[p].Type == 0)
            message(1, "Bad timestep (%x)! titype %d. ID=%lu Type=%d dloga=%g dtmax=%x xyz=(%g|%g|%g) tree=(%g|%g|%g) PM=(%g|%g|%g) hydro-frc=(%g|%g|%g) dens=%g hsml=%g dh = %g Entropy=%g, dtEntropy=%g maxsignal=%g\n",
                dti, titype, P[p].ID, P[p].Type, dloga, dti_max,
                P[p].Pos[0], P[p].Pos[1], P[p].Pos[2],
                P[p].GravAccel[0], P[p].GravAccel[1], P[p].GravAccel[2],
                P[p].GravPM[0], P[p].GravPM[1], P[p].GravPM[2],
                SPHP(p).HydroAccel[0], SPHP(p).HydroAccel[1], SPHP(p).HydroAccel[2],
                SPHP(p).Density, P[p].Hsml, P[p].DtHsml, SPHP(p).Entropy, SPHP(p).DtEntropy, SPHP(p).MaxSignalVel);
        else
            message(1, "Bad timestep (%x)! titype %d. ID=%lu Type=%d dloga=%g dtmax=%x xyz=(%g|%g|%g) tree=(%g|%g|%g) PM=(%g|%g|%g)\n",
                dti, titype, P[p].ID, P[p].Type, dloga, dti_max,
                P[p].Pos[0], P[p].Pos[1], P[p].Pos[2],
                P[p].GravAccel[0], P[p].GravAccel[1], P[p].GravAccel[2],
                P[p].GravPM[0], P[p].GravPM[1], P[p].GravPM[2]
              );
    }

    return dti;
}


/*! This function computes the PM timestep of the system based on
 *  the rms velocities of particles. For cosmological simulations, the criterion used is that the rms
 *  displacement should be at most a fraction MaxRMSDisplacementFac of the mean particle separation.
 *  Note that the latter is estimated using the assigned particle masses, separately for each particle type.
 */
double
get_long_range_timestep_dloga(const double atime, const Cosmology * CP, const int FastParticleType, const double asmth)
{
    int i, type;
    int count[6];
    int64_t count_sum[6];
    double v[6], v_sum[6], mim[6], min_mass[6];
    double dloga = TimestepParams.MaxSizeTimestep;

    for(type = 0; type < 6; type++)
    {
        count[type] = 0;
        v[type] = 0;
        mim[type] = 1.0e30;
    }

    for(i = 0; i < PartManager->NumPart; i++)
    {
        v[P[i].Type] += P[i].Vel[0] * P[i].Vel[0] + P[i].Vel[1] * P[i].Vel[1] + P[i].Vel[2] * P[i].Vel[2];
        if(P[i].Mass > 0)
        {
            if(mim[P[i].Type] > P[i].Mass)
                mim[P[i].Type] = P[i].Mass;
        }
        count[P[i].Type]++;
    }

    MPI_Allreduce(v, v_sum, 6, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(mim, min_mass, 6, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);

    sumup_large_ints(6, count, count_sum);

    /* add star, gas and black hole particles together to treat them on equal footing,
     * using the original gas particle spacing. */
    v_sum[0] += v_sum[4];
    count_sum[0] += count_sum[4];
    v_sum[4] = v_sum[0];
    count_sum[4] = count_sum[0];
    v_sum[0] += v_sum[5];
    count_sum[0] += count_sum[5];
    v_sum[5] = v_sum[0];
    count_sum[5] = count_sum[0];

    min_mass[5] = min_mass[0];

    const double hubble = hubble_function(CP, atime);
    for(type = 0; type < 6; type++)
    {
        if(count_sum[type] > 0)
        {
            double omega, dmean, dloga1;
            /* Type 4 is stars, type 5 is BHs, both baryons*/
            if(type == 0 || type == 4 || type == 5) {
                omega = CP->OmegaBaryon;
            }
            /* In practice usually FastParticleType == 2
             * so this doesn't matter. */
            else if (type == 2) {
                omega = get_omega_nu(&CP->ONu, 1);
            } else {
                omega = CP->OmegaCDM;
            }
            /* "Avg. radius" of smallest particle: (min_mass/total_mass)^1/3 */
            dmean = pow(min_mass[type] / (omega * CP->RhoCrit), 1.0 / 3);

            dloga1 = TimestepParams.MaxRMSDisplacementFac * hubble * atime * atime * DMIN(asmth, dmean) / sqrt(v_sum[type] / count_sum[type]);
            message(0, "type=%d  dmean=%g asmth=%g minmass=%g a=%g  sqrt(<p^2>)=%g  dloga=%g\n",
                    type, dmean, asmth, min_mass[type], atime, sqrt(v_sum[type] / count_sum[type]), dloga1);

            /* don't constrain the step to the neutrinos */
            if(type != FastParticleType && dloga1 < dloga)
                dloga = dloga1;
        }
    }

    if(dloga < TimestepParams.MinSizeTimestep) {
        dloga = TimestepParams.MinSizeTimestep;
    }

    return dloga;
}

/* backward compatibility with the old loop. */
inttime_t
get_PM_timestep_ti(const DriftKickTimes * const times, const double atime, const Cosmology * CP, const int FastParticleType, const double asmth)
{
    double dloga = get_long_range_timestep_dloga(atime, CP, FastParticleType, asmth);

    inttime_t dti = dti_from_dloga(dloga, times->Ti_Current);
    dti = round_down_power_of_two(dti);

    SyncPoint * next = find_next_sync_point(times->Ti_Current);
    if(next == NULL)
        endrun(0, "Trying to go beyond the last sync point. This happens only at TimeMax \n");

    /* go no more than the next sync point */
    inttime_t dti_max = next->ti - times->PM_kick;

    if(dti > dti_max)
        dti = dti_max;
    return dti;
}

int get_timestep_bin(inttime_t dti)
{
   int bin = -1;

   if(dti <= 0)
       return 0;

   if(dti == 1)
       return -1;

   while(dti)
   {
       bin++;
       dti >>= 1;
   }

   return bin;
}

/*! This function finds the next synchronization point of the system
 * (i.e. the earliest point of time any of the particles needs a force
 * computation), and drifts the system to this point of time.  If the
 * system drifts over the desired time of a snapshot file, the
 * function will drift to this moment, generate an output, and then
 * resume the drift.
 */
inttime_t find_next_kick(inttime_t Ti_Current, int minTimeBin)
{
    /* Current value plus the increment for the smallest active bin. */
    return Ti_Current + dti_from_timebin(minTimeBin);
}

static void print_timebin_statistics(const DriftKickTimes * const times, const int NumCurrentTiStep, int * TimeBinCountType, const double Time);

/* mark the bins that will be active before the next kick*/
int rebuild_activelist(ActiveParticles * act, const DriftKickTimes * const times, int NumCurrentTiStep, const double Time)
{
    int i;

    int NumThreads = omp_get_max_threads();
    /*Since we use a static schedule, only need NumPart/NumThreads elements per thread.*/
    size_t narr = PartManager->NumPart / NumThreads + NumThreads;

    /*We know all particles are active on a PM timestep*/
    if(is_PM_timestep(times)) {
        act->ActiveParticle = NULL;
        act->NumActiveParticle = PartManager->NumPart;
        act->NumActiveGravity = PartManager->NumPart;
    }
    else {
        /*Need space for more particles than we have, because of star formation*/
        act->ActiveParticle = (int *) mymalloc("ActiveParticle", narr * NumThreads * sizeof(int));
        act->NumActiveParticle = 0;
        act->NumActiveGravity = 0;
    }

    int * TimeBinCountType = (int *) mymalloc("TimeBinCountType", 6*(TIMEBINS+1)*NumThreads * sizeof(int));
    memset(TimeBinCountType, 0, 6 * (TIMEBINS+1) * NumThreads * sizeof(int));

    /*We want a lockless algorithm which preserves the ordering of the particle list.*/
    size_t *NActiveThread = ta_malloc("NActiveThread", size_t, NumThreads);
    int **ActivePartSets = ta_malloc("ActivePartSets", int *, NumThreads);
    gadget_setup_thread_arrays(act->ActiveParticle, ActivePartSets, NActiveThread, narr, NumThreads);

    /* We enforce schedule static to imply monotonic, ensure that each thread executes on contiguous particles
     * and ensure no thread gets more than narr particles.*/
    size_t schedsz = PartManager->NumPart / NumThreads + 1;
    int64_t nactivegrav=act->NumActiveGravity;
    #pragma omp parallel for schedule(static, schedsz) reduction(+: nactivegrav)
    for(i = 0; i < PartManager->NumPart; i++)
    {
        const int bin_hydro = P[i].TimeBinHydro;
        const int bin_gravity = P[i].TimeBinGravity;
        const int tid = omp_get_thread_num();
        if(P[i].IsGarbage || P[i].Swallowed)
            continue;
        /* when we are in PM, all particles must have been synced. */
        if (P[i].Ti_drift != times->Ti_Current) {
            endrun(5, "Particle %d type %d has drift time %x not ti_current %x!",i, P[i].Type, P[i].Ti_drift, times->Ti_Current);
        }
        /* For now build active particles with either hydro or gravity active*/
        int hydro_active = is_timebin_active(bin_hydro, times->Ti_Current);
        int gravity_active = is_timebin_active(bin_gravity, times->Ti_Current);
        if(act->ActiveParticle && gravity_active)
            nactivegrav++;
        if(act->ActiveParticle && (hydro_active || gravity_active))
        {
            /* Store this particle in the ActiveSet for this thread*/
            ActivePartSets[tid][NActiveThread[tid]] = i;
            NActiveThread[tid]++;

        }
        /* Account gas and BHs to their hydro bin and other particles to their gravity bin*/
        int bin = bin_gravity;
        if(P[i].Type == 0 || P[i].Type == 5)
            bin = bin_hydro;
        TimeBinCountType[(TIMEBINS + 1) * (6* tid + P[i].Type) + bin] ++;
    }
    if(act->ActiveParticle) {
        /*Now we want a merge step for the ActiveParticle list.*/
        act->NumActiveParticle = gadget_compact_thread_arrays(act->ActiveParticle, ActivePartSets, NActiveThread, NumThreads);
    }
    ta_free(ActivePartSets);
    ta_free(NActiveThread);
    act->NumActiveGravity = nactivegrav;
    /*Print statistics for this time bin*/
    print_timebin_statistics(times, NumCurrentTiStep, TimeBinCountType, Time);
    myfree(TimeBinCountType);

    /* Shrink the ActiveParticle array. We still need extra space for star formation,
     * but we do not need space for the known-inactive particles*/
    if(act->ActiveParticle) {
        act->ActiveParticle = (int *) myrealloc(act->ActiveParticle, sizeof(int)*(act->NumActiveParticle + PartManager->MaxPart - PartManager->NumPart));
        act->MaxActiveParticle = act->NumActiveParticle + PartManager->MaxPart - PartManager->NumPart;
        /* listen to the slots events such that we can set timebin of new particles */
    }
    event_listen(&EventSlotsFork, timestep_eh_slots_fork, act);
    walltime_measure("/Timeline/Active");

    return 0;
}

/* Build a sublist of particles, selected from the currently active particles, which have a gravity timebin no larger than ti.*/
int
build_active_sublist(ActiveParticles * sub_act, const ActiveParticles * act, const int maxtimebin)
{
    int i;

    int NumThreads = omp_get_max_threads();
    /*Since we use a static schedule, only need NumPart/NumThreads elements per thread.*/
    size_t narr = PartManager->NumPart / NumThreads + NumThreads;

    /*Need space for more particles than we have, because of star formation*/
    sub_act->ActiveParticle = (int *) mymalloc("ActiveParticle", narr * NumThreads * sizeof(int));
    sub_act->NumActiveParticle = 0;
    sub_act->NumActiveGravity = 0;

    /*We want a lockless algorithm which preserves the ordering of the particle list.*/
    size_t *NActiveThread = ta_malloc("NActiveThread", size_t, NumThreads);
    int **ActivePartSets = ta_malloc("ActivePartSets", int *, NumThreads);
    gadget_setup_thread_arrays(sub_act->ActiveParticle, ActivePartSets, NActiveThread, narr, NumThreads);

    /* We enforce schedule static to imply monotonic, ensure that each thread executes on contiguous particles
     * and ensure no thread gets more than narr particles.*/
    size_t schedsz = PartManager->NumPart / NumThreads + 1;
    #pragma omp parallel for schedule(static, schedsz)
    for(i = 0; i < act->NumActiveGravity; i++)
    {
        int pi = act->ActiveParticle ? act->ActiveParticle[i] : i;
        const int bin_gravity = P[pi].TimeBinGravity;
        const int tid = omp_get_thread_num();
        if(P[pi].IsGarbage || P[pi].Swallowed)
            continue;
        if(bin_gravity > maxtimebin)
            continue;
        /* Store this particle in the ActiveSet for this thread*/
        ActivePartSets[tid][NActiveThread[tid]] = i;
        NActiveThread[tid]++;
    }
    /*Now we want a merge step for the ActiveParticle list.*/
    sub_act->NumActiveParticle = gadget_compact_thread_arrays(sub_act->ActiveParticle, ActivePartSets, NActiveThread, NumThreads);
    sub_act->NumActiveGravity = sub_act->NumActiveParticle;
    sub_act->MaxActiveParticle = sub_act->NumActiveParticle;

    ta_free(ActivePartSets);
    ta_free(NActiveThread);

    sub_act->ActiveParticle = (int *) myrealloc(sub_act->ActiveParticle, sizeof(int)*(sub_act->NumActiveParticle));
    return 0;
}

void free_activelist(ActiveParticles * act)
{
    if(act->ActiveParticle) {
        myfree(act->ActiveParticle);
    }
    event_unlisten(&EventSlotsFork, timestep_eh_slots_fork, act);
}

/*! This routine writes one line for every timestep.
 * FdCPU the cumulative cpu-time consumption in various parts of the
 * code is stored.
 */
static void print_timebin_statistics(const DriftKickTimes * const times, const int NumCurrentTiStep, int * TimeBinCountType, const double Time)
{
    int i;
    int64_t tot = 0, tot_type[6] = {0};
    int64_t tot_count[TIMEBINS+1] = {0};
    int64_t tot_count_type[6][TIMEBINS+1] = {{0}};
    int64_t tot_num_force = 0;
    int64_t TotNumPart = 0, TotNumType[6] = {0};

    int NumThreads = omp_get_max_threads();
    /*Sum the thread-local memory*/
    for(i = 1; i < NumThreads; i ++) {
        int j;
        for(j=0; j < 6 * (TIMEBINS+1); j++)
            TimeBinCountType[j] += TimeBinCountType[6 * (TIMEBINS+1) * i + j];
    }

    for(i = 0; i < 6; i ++) {
        sumup_large_ints(TIMEBINS+1, &TimeBinCountType[(TIMEBINS+1) * i], tot_count_type[i]);
    }

    for(i = 0; i<TIMEBINS+1; i++) {
        int j;
        for(j=0; j<6; j++) {
            tot_count[i] += tot_count_type[j][i];
            /*Note j*/
            TotNumType[j] += tot_count_type[j][i];
            TotNumPart += tot_count_type[j][i];
        }
        if(is_timebin_active(i, times->Ti_Current))
            tot_num_force += tot_count[i];
    }

    char extra[20] = {0};

    if(is_PM_timestep(times))
        strcat(extra, "PM-Step");

    const double dloga = get_dloga_for_bin(times->mintimebin, times->Ti_Current);
    const double z = 1.0 / (Time) - 1;
    message(0, "Begin Step %d, Time: %g (%x), Redshift: %g, Nf = %014ld, Systemstep: %g, Dloga: %g, status: %s\n",
                NumCurrentTiStep, Time, times->Ti_Current, z, tot_num_force,
                dloga * Time, dloga,
                extra);

    message(0, "TotNumPart: %013ld SPH %013ld BH %010ld STAR %013ld \n",
                TotNumPart, TotNumType[0], TotNumType[5], TotNumType[4]);
    message(0,     "Occupied: % 12ld % 12ld % 12ld % 12ld % 12ld % 12ld dt\n", 0L, 1L, 2L, 3L, 4L, 5L);

    for(i = TIMEBINS;  i >= 0; i--) {
        if(tot_count[i] == 0) continue;
        message(0, " %c bin=%2d % 12ld % 12ld % 12ld % 12ld % 12ld % 12ld %6g\n",
                is_timebin_active(i, times->Ti_Current) ? 'X' : ' ',
                i,
                tot_count_type[0][i],
                tot_count_type[1][i],
                tot_count_type[2][i],
                tot_count_type[3][i],
                tot_count_type[4][i],
                tot_count_type[5][i],
                get_dloga_for_bin(i, times->Ti_Current));

        if(is_timebin_active(i, times->Ti_Current))
        {
            tot += tot_count[i];
            int ptype;
            for(ptype = 0; ptype < 6; ptype ++) {
                tot_type[ptype] += tot_count_type[ptype][i];
            }
        }
    }
    message(0,     "               -----------------------------------\n");
    message(0,     "Total:    % 12ld % 12ld % 12ld % 12ld % 12ld % 12ld  Sum:% 14ld\n",
        tot_type[0], tot_type[1], tot_type[2], tot_type[3], tot_type[4], tot_type[5], tot);

}
