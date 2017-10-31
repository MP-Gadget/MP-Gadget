#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "allvars.h"
#include "domain.h"
#include "openmpsort.h"
#include "proto.h"
#include "timefac.h"
#include "cosmology.h"
#include "cooling.h"
#include "mymalloc.h"
#include "endrun.h"
#include "slotsmanager.h"
#include "system.h"
#include "timestep.h"

/*! \file timestep.c
 *  \brief routines for 'kicking' particles in
 *  momentum space and assigning new timesteps
 */


/* variables for organizing PM steps of discrete timeline */
typedef struct {
    inttime_t length; /*!< Duration of the current PM integer timestep*/
    inttime_t start;           /* current start point of the PM step*/
    inttime_t Ti_kick;  /* current inttime of PM Kick (velocity) */
} TimeSpan;

static TimeSpan PM;

/*Get the dti from the timebin*/
static inline inttime_t dti_from_timebin(int bin) {
    return bin ? (1 << bin) : 0;
}
/*Flat array containing all active particles*/
int NumActiveParticle;
int *ActiveParticle;

static int TimeBinCountType[6][TIMEBINS+1];

void timestep_allocate_memory(int MaxPart)
{
    ActiveParticle = (int *) mymalloc("ActiveParticle", MaxPart * sizeof(int));
}

static void reverse_and_apply_gravity();
static inttime_t get_timestep_ti(const int p, const inttime_t dti_max);
static int get_timestep_bin(inttime_t dti);
static void do_the_short_range_kick(int i, inttime_t tistart, inttime_t tiend);
static void do_the_long_range_kick(inttime_t tistart, inttime_t tiend);
static inttime_t get_long_range_timestep_ti(const inttime_t dti_max);

/*Initialise the integer timeline*/
void
init_timebins(double TimeInit)
{
    All.Ti_Current = ti_from_loga(TimeInit);
    /*Enforce Ti_Current is initially even*/
    if(All.Ti_Current % 2 == 1)
        All.Ti_Current++;
    /* this makes sure the first step is a PM step. */
    PM.length = 0;
    PM.Ti_kick = All.Ti_Current;
    PM.start = All.Ti_Current;
}

int is_timebin_active(int i, inttime_t current) {
    /*Bin 0 is always active and at time 0 all bins are active*/
    if(i == 0 || current == 0)
        return 1;
    if(current % dti_from_timebin(i) == 0)
        return 1;
    return 0;
}

/*Report whether the current timestep is the end of the PM timestep*/
int
is_PM_timestep(inttime_t ti)
{
    if(ti > PM.start + PM.length)
        endrun(12, "Passed end of PM step! ti=%d, PM = %d + %d\n",ti, PM.start, PM.length);
    return ti == PM.start + PM.length;

}

void
set_global_time(double newtime) {
    All.TimeStep = newtime - All.Time;
    All.Time = newtime;
    All.cf.a = All.Time;
    All.cf.a2inv = 1 / (All.Time * All.Time);
    All.cf.a3inv = 1 / (All.Time * All.Time * All.Time);
    All.cf.fac_egy = pow(All.Time, 3 * GAMMA_MINUS1);
    All.cf.hubble = hubble_function(All.Time);
    All.cf.hubble_a2 = All.Time * All.Time * hubble_function(All.Time);

#ifdef LIGHTCONE
    lightcone_set_time(All.cf.a);
#endif
    IonizeParams();
}

/* This function assigns new timesteps to particles and PM */
int
find_timesteps(int * MinTimeBin)
{
    int pa;
    inttime_t dti_min = TIMEBASE;

    walltime_measure("/Misc");

    if(All.MakeGlassFile)
        reverse_and_apply_gravity();

    /*Update the PM timestep size */
    if(is_PM_timestep(All.Ti_Current)) {
        SyncPoint * next = find_next_sync_point(All.Ti_Current);
        inttime_t dti_max;
        if(next == NULL) {

            endrun(0, "Trying to go beyond the last sync point. This happens only at TimeMax \n");

            /* use a unlimited pm step size*/
            dti_max = TIMEBASE;
        } else {
            /* go no more than the next sync point */
            dti_max = next->ti - PM.Ti_kick;
        }
        PM.length = get_long_range_timestep_ti(dti_max);
        PM.start = PM.Ti_kick;
    }

    /* Now assign new timesteps and kick */
    if(All.ForceEqualTimesteps) {
        #pragma omp parallel for
        for(pa = 0; pa < NumActiveParticle; pa++)
        {
            const int i = ActiveParticle[pa];
            inttime_t dti = get_timestep_ti(i, PM.length);

            if(dti < dti_min)
                dti_min = dti;
        }

        /* FIXME : this assumes inttime_t is int*/
        MPI_Allreduce(MPI_IN_PLACE, &dti_min, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    }

    int badstepsizecount = 0;
    int mTimeBin = TIMEBINS;
    #pragma omp parallel for reduction(min: mTimeBin) reduction(+: badstepsizecount)
    for(pa = 0; pa < NumActiveParticle; pa++)
    {
        const int i = ActiveParticle[pa];

        if(P[i].Ti_kick != P[i].Ti_drift) {
            endrun(1, "Inttimes out of sync: Particle %d (ID=%ld) Kick=%o != Drift=%o\n", i, P[i].ID, P[i].Ti_kick, P[i].Ti_drift);
        }

        int dti;
        if(All.ForceEqualTimesteps) {
            dti = dti_min;
        } else {
            dti = get_timestep_ti(i, PM.length);
        }

        /* make it a power 2 subdivision */
        dti = round_down_power_of_two(dti);

        int bin = get_timestep_bin(dti);
        if(bin < 1) {
            message(1, "Time-step of integer size %d not allowed, id = %lu, debugging info follows. %d\n", dti, P[i].ID);
            badstepsizecount++;
        }
        int binold = P[i].TimeBin;

        if(bin > binold)		/* timestep wants to increase */
        {
            /* make sure the new step is currently active,
             * so that particles do not miss a step */
            while(!is_timebin_active(bin, All.Ti_Current) && bin > binold && bin > 1)
                bin--;
        }
        /* This moves particles between time bins:
         * active particles always remain active
         * until reconstruct_timebins is called
         * (during domain, on new timestep).*/
        P[i].TimeBin = bin;
        if(bin < mTimeBin)
            mTimeBin = bin;
    }

    MPI_Allreduce(MPI_IN_PLACE, &badstepsizecount, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &mTimeBin, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

    if(badstepsizecount) {
        message(0, "bad timestep spotted: terminating and saving snapshot.\n");
        savepositions(999999);
        endrun(0, "Ending due to bad timestep");
    }
    walltime_measure("/Timeline");
    *MinTimeBin = mTimeBin;
    return 0;
}


/* Apply half a kick, for the second half of the timestep.*/
void
apply_half_kick(void)
{
    int pa;
    walltime_measure("/Misc");
    /* Now assign new timesteps and kick */
    #pragma omp parallel for
    for(pa = 0; pa < NumActiveParticle; pa++)
    {
        const int i = ActiveParticle[pa];
        int bin = P[i].TimeBin;
        inttime_t dti = dti_from_timebin(bin);
        /* current Kick time */
        inttime_t tistart = P[i].Ti_kick;
        /* half of a step */
        inttime_t tiend = P[i].Ti_kick + dti / 2;
        /*This only changes particle i, so is thread-safe.*/
        do_the_short_range_kick(i, tistart, tiend);
    }
    walltime_measure("/Timeline/HalfKick/Short");
}

void
apply_PM_half_kick(void)
{
    /*Always do a PM half-kick, because this should be called just after a PM step*/
    const inttime_t tistart = PM.Ti_kick;
    const inttime_t tiend =  PM.Ti_kick + PM.length / 2;
    /* Do long-range kick */
    do_the_long_range_kick(tistart, tiend);
    walltime_measure("/Timeline/HalfKick/Long");
}

/*Advance a long-range timestep and do the desired kick.*/
void
do_the_long_range_kick(inttime_t tistart, inttime_t tiend)
{
    int i;
    const double Fgravkick = get_gravkick_factor(tistart, tiend);

    #pragma omp parallel for
    for(i = 0; i < NumPart; i++)
    {
        int j;
        for(j = 0; j < 3; j++)	/* do the kick */
            P[i].Vel[j] += P[i].GravPM[j] * Fgravkick;
    }
    PM.Ti_kick = tiend;
}

void
do_the_short_range_kick(int i, inttime_t tistart, inttime_t tiend)
{
    const double Fgravkick = get_gravkick_factor(tistart, tiend);

    int j;
#ifdef DEBUG
    if(P[i].Ti_kick != tistart) {
        endrun(1, "Ti kick mismatch\n");
    }

#endif
    /* update the time stamp */
    P[i].Ti_kick = tiend;

    /* do the kick */

    for(j = 0; j < 3; j++)
    {
        P[i].Vel[j] += P[i].GravAccel[j] * Fgravkick;
    }

    if(P[i].Type == 0) {
        const double Fhydrokick = get_hydrokick_factor(tistart, tiend);
        double dt_entr = dloga_from_dti(tiend-tistart); /* XXX: the kick factor of entropy is dlog a? */
        /* Add kick from hydro and SPH stuff */
        for(j = 0; j < 3; j++) {
            P[i].Vel[j] += SPHP(i).HydroAccel[j] * Fhydrokick;
        }

        /* Code here imposes a hard limit (default to speed of light)
         * on the gas velocity. Then a limit on the change in entropy
         * FIXME: This should probably not be needed!*/
        const double velfac = sqrt(All.cf.a3inv);
        double vv=0;
        for(j=0; j < 3; j++)
            vv += P[i].Vel[j] * P[i].Vel[j];
        vv = sqrt(vv);

        if(vv > All.MaxGasVel * velfac) {
            for(j=0;j < 3; j++)
            {
                P[i].Vel[j] *= All.MaxGasVel * velfac / vv;
            }
        }

        /* In case of cooling, we prevent that the entropy (and
           hence temperature) decreases by more than a factor 0.5.
           FIXME: Why is this and the last thing here? Should not be needed. */

        if(SPHP(i).DtEntropy * dt_entr < -0.5 * SPHP(i).Entropy)
            SPHP(i).Entropy *= 0.5;
        else
            SPHP(i).Entropy += SPHP(i).DtEntropy * dt_entr;

        /* Implement an entropy floor*/
        if(All.MinEgySpec)
        {
            const double minentropy = All.MinEgySpec * GAMMA_MINUS1 / pow(SPHP(i).EOMDensity * All.cf.a3inv, GAMMA_MINUS1);
            if(SPHP(i).Entropy < minentropy)
            {
                SPHP(i).Entropy = minentropy;
                SPHP(i).DtEntropy = 0;
            }
        }

        /* In case the timestep increases in the new step, we
           make sure that we do not 'overcool' by bounding the entropy rate of next step */
        double dt_entr_next = get_dloga_for_bin(P[i].TimeBin) / 2;

        if(SPHP(i).DtEntropy * dt_entr_next < - 0.5 * SPHP(i).Entropy)
            SPHP(i).DtEntropy = -0.5 * SPHP(i).Entropy / dt_entr_next;
    }

}

/*Get the predicted velocity for a particle
 * at the Force computation time, which always coincides with the Drift inttime.
 * for gravity and hydro forces.
 * This is mostly used for artificial viscosity.*/
void
sph_VelPred(int i, double * VelPred)
{
    const int ti = P[i].Ti_drift;
    const double Fgravkick2 = get_gravkick_factor(P[i].Ti_kick, ti);
    const double Fhydrokick2 = get_hydrokick_factor(P[i].Ti_kick, ti);
    const double FgravkickB = get_gravkick_factor(PM.Ti_kick, ti);
    int j;
    for(j = 0; j < 3; j++) {
        VelPred[j] = P[i].Vel[j] + Fgravkick2 * P[i].GravAccel[j]
            + P[i].GravPM[j] * FgravkickB + Fhydrokick2 * SPHP(i).HydroAccel[j];
    }
}

/*Helper function for predicting the entropy*/
static inline double _EntPred(int i)
{
    const double Fentr = dloga_from_dti(P[i].Ti_drift - P[i].Ti_kick);
    double epred = SPHP(i).Entropy + SPHP(i).DtEntropy * Fentr;
    /*This mirrors the entropy limiter in do_the_short_range_kick*/
    if(epred < 0.5 * SPHP(i).Entropy)
        epred = 0.5 * SPHP(i).Entropy;
    return epred;
}

/* This gives the predicted entropy at the particle Kick timestep
 * for the density independent SPH code.
 * Watchout: with kddk, when the second k is applied, Ti_kick < Ti_drift. */
double
EntropyPred(int i)
{
    double epred = _EntPred(i);
    return pow(epred, 1/GAMMA);
}

double
PressurePred(int i)
{
    double epred = _EntPred(i);
    return epred * pow(SPHP(i).EOMDensity, GAMMA);
}

double
get_timestep_dloga(const int p)
{
    double ac = 0;
    double dt = 0, dt_courant = 0;

    /*Compute physical acceleration*/
    {
        double ax = All.cf.a2inv * P[p].GravAccel[0];
        double ay = All.cf.a2inv * P[p].GravAccel[1];
        double az = All.cf.a2inv * P[p].GravAccel[2];

        ax += All.cf.a2inv * P[p].GravPM[0];
        ay += All.cf.a2inv * P[p].GravPM[1];
        az += All.cf.a2inv * P[p].GravPM[2];

        if(P[p].Type == 0)
        {
            const double fac2 = 1 / pow(All.Time, 3 * GAMMA - 2);
            ax += fac2 * SPHP(p).HydroAccel[0];
            ay += fac2 * SPHP(p).HydroAccel[1];
            az += fac2 * SPHP(p).HydroAccel[2];
        }

        ac = sqrt(ax * ax + ay * ay + az * az);	/* this is now the physical acceleration */
    }

    if(ac == 0)
        ac = 1.0e-30;

    /* mind the factor 2.8 difference between gravity and softening used here. */
    dt = sqrt(2 * All.ErrTolIntAccuracy * All.cf.a * (FORCE_SOFTENING(p) / 2.8) / ac);

    if(P[p].Type == 0)
    {
        const double fac3 = pow(All.Time, 3 * (1 - GAMMA) / 2.0);
        dt_courant = 2 * All.CourantFac * All.Time * P[p].Hsml / (fac3 * SPHP(p).MaxSignalVel);
        if(dt_courant < dt)
            dt = dt_courant;
    }

#ifdef BLACK_HOLES
    if(P[p].Type == 5)
    {
        if(BHP(p).Mdot > 0 && BHP(p).Mass > 0)
        {
            double dt_accr = 0.25 * BHP(p).Mass / BHP(p).Mdot;
            if(dt_accr < dt)
                dt = dt_accr;
        }
        if(BHP(p).TimeBinLimit > 0) {
            double dt_limiter = get_dloga_for_bin(BHP(p).TimeBinLimit) / All.cf.hubble;
            if (dt_limiter < dt) dt = dt_limiter;
        }
    }
#endif

    /* d a / a = dt * H */
    double dloga = dt * All.cf.hubble;

    return dloga;
}

/*! This function returns the maximum allowed timestep of a particle, expressed in
 *  terms of the integer mapping that is used to represent the total simulated timespan.
 *  Arguments:
 *  p -> particle index
 *  dti_max -> maximal timestep.  */
static inttime_t
get_timestep_ti(const int p, const inttime_t dti_max)
{
    int dti;
    /*Give a useful message if we are broken*/
    if(dti_max == 0)
        return 0;

    /*Set to max timestep allowed if the tree is off*/
    if(!All.TreeGravOn)
        return dti_max;

    double dloga = get_timestep_dloga(p);

    if(dloga < All.MinSizeTimestep)
        dloga = All.MinSizeTimestep;

    dti = dti_from_dloga(dloga);

    if(dti > dti_max)
        dti = dti_max;

    /*
    sqrt(2 * All.ErrTolIntAccuracy * All.cf.a * All.SofteningTable[P[p].Type] / ac) * All.cf.hubble,
    */
    if(dti <= 1 || dti > TIMEBASE)
    {
        message(1, "Bad timestep (%x) assigned! ID=%lu Type=%d dloga=%g dtmax=%x xyz=(%g|%g|%g) tree=(%g|%g|%g) PM=(%g|%g|%g)\n",
                dti, P[p].ID, P[p].Type, dloga, dti_max,
                P[p].Pos[0], P[p].Pos[1], P[p].Pos[2],
                P[p].GravAccel[0], P[p].GravAccel[1], P[p].GravAccel[2],
                P[p].GravPM[0], P[p].GravPM[1], P[p].GravPM[2]
              );
        if(P[p].Type == 0)
            message(1, "hydro-frc=(%g|%g|%g) dens=%g hsml=%g numngb=%g\n", SPHP(p).HydroAccel[0], SPHP(p).HydroAccel[1],
                    SPHP(p).HydroAccel[2], SPHP(p).Density, P[p].Hsml, P[p].NumNgb);
#ifdef DENSITY_INDEPENDENT_SPH
        if(P[p].Type == 0)
            message(1, "egyrho=%g entvarpred=%g dhsmlegydensityfactor=%g Entropy=%g, dtEntropy=%g, Pressure=%g\n", SPHP(p).EgyWtDensity, EntropyPred(p),
                    SPHP(p).DhsmlEgyDensityFactor, SPHP(p).Entropy, SPHP(p).DtEntropy, PressurePred(p));
#endif
#ifdef SFR
        if(P[p].Type == 0) {
            message(1, "sfr = %g\n" , SPHP(p).Sfr);
        }
#endif
#ifdef BLACK_HOLES
        if(P[p].Type == 0) {
            message(1, "injected_energy = %g\n" , SPHP(p).Injected_BH_Energy);
        }
#endif
    }

    return dti;
}


/*! This function computes the PM timestep of the system based on
 *  the rms velocities of particles. For cosmological simulations, the criterion used is that the rms
 *  displacement should be at most a fraction MaxRMSDisplacementFac of the mean particle separation. 
 *  Note that the latter is estimated using the assigned particle masses, separately for each particle type.
 */
double
get_long_range_timestep_dloga()
{
    int i, type;
    int count[6];
    int64_t count_sum[6];
    double v[6], v_sum[6], mim[6], min_mass[6];
    double dloga = All.MaxSizeTimestep;

    for(type = 0; type < 6; type++)
    {
        count[type] = 0;
        v[type] = 0;
        mim[type] = 1.0e30;
    }

    for(i = 0; i < NumPart; i++)
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

#ifdef SFR
    /* add star and gas particles together to treat them on equal footing, using the original gas particle
       spacing. */
    v_sum[0] += v_sum[4];
    count_sum[0] += count_sum[4];
    v_sum[4] = v_sum[0];
    count_sum[4] = count_sum[0];
#ifdef BLACK_HOLES
    v_sum[0] += v_sum[5];
    count_sum[0] += count_sum[5];
    v_sum[5] = v_sum[0];
    count_sum[5] = count_sum[0];
    min_mass[5] = min_mass[0];
#endif
#endif

    for(type = 0; type < 6; type++)
    {
        if(count_sum[type] > 0)
        {
            double omega, dmean, dloga1;
            const double asmth = All.Asmth * All.BoxSize / All.Nmesh;
            if(type == 0 || (type == 4 && All.StarformationOn)
#ifdef BLACK_HOLES
                || (type == 5)
#endif
                ) {
                omega = All.CP.OmegaBaryon;
            }
            /* Neutrinos are counted here as CDM. They should be counted separately!
             * In practice usually FastParticleType == 2
             * so this doesn't matter. Also the neutrinos
             * are either Way Too Fast, or basically CDM anyway. */
            else {
                omega = All.CP.OmegaCDM;
            }
            /* "Avg. radius" of smallest particle: (min_mass/total_mass)^1/3 */
            dmean = pow(min_mass[type] / (omega * 3 * All.CP.Hubble * All.CP.Hubble / (8 * M_PI * All.G)), 1.0 / 3);

            dloga1 = All.MaxRMSDisplacementFac * All.cf.hubble * All.cf.a * All.cf.a * DMIN(asmth, dmean) / sqrt(v_sum[type] / count_sum[type]);
            message(0, "type=%d  dmean=%g asmth=%g minmass=%g a=%g  sqrt(<p^2>)=%g  dlogmax=%g\n",
                    type, dmean, asmth, min_mass[type], All.Time, sqrt(v_sum[type] / count_sum[type]), dloga);

            /* don't constrain the step to the neutrinos */
            if(type != All.FastParticleType && dloga1 < dloga)
                dloga = dloga1;
        }
    }
    return dloga;
}

/* backward compatibility with the old loop. */
inttime_t
get_long_range_timestep_ti(const inttime_t dti_max)
{
    double dloga = get_long_range_timestep_dloga();
    int dti = dti_from_dloga(dloga);
    dti = round_down_power_of_two(dti);
    if(dti > dti_max)
        dti = dti_max;
    message(0, "Maximal PM timestep: dloga = %g  (%g)\n", dloga_from_dti(dti), All.MaxSizeTimestep);
    return dti;
}

int get_timestep_bin(inttime_t dti)
{
   int bin = -1;

   if(dti == 0)
       return 0;

   if(dti == 1)
   {
       return -1;
   }

   while(dti)
   {
       bin++;
       dti >>= 1;
   }

   return bin;
}

/* This function reverse the direction of the gravitational force.
 * This is only useful for making Lagrangian glass files*/
void reverse_and_apply_gravity()
{
    double dispmax=0, globmax;
    int i;
    for(i = 0; i < NumPart; i++)
    {
        int j;
        /*Reverse the direction of acceleration*/
        for(j = 0; j < 3; j++)
        {
            P[i].GravAccel[j] *= -1;
            P[i].GravAccel[j] -= P[i].GravPM[j];
            P[i].GravPM[j] = 0;
        }

        double disp = sqrt(P[i].GravAccel[0] * P[i].GravAccel[0] +
                P[i].GravAccel[1] * P[i].GravAccel[1] + P[i].GravAccel[2] * P[i].GravAccel[2]);

        disp *= 2.0 / (3 * All.CP.Hubble * All.CP.Hubble);

        if(disp > dispmax)
            dispmax = disp;
    }

    MPI_Allreduce(&dispmax, &globmax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    double dmean = pow(P[0].Mass / (All.CP.Omega0 * 3 * All.CP.Hubble * All.CP.Hubble / (8 * M_PI * All.G)), 1.0 / 3);

    const double fac = DMIN(1.0, dmean / globmax);

    message(0, "Glass-making: dmean= %g  global disp-maximum= %g\n", dmean, globmax);

    /* Move the actual particles according to the (reversed) gravitational force.
     * Not sure why this is here rather than in the main code.*/
    for(i = 0; i < NumPart; i++)
    {
        int j;
        for(j = 0; j < 3; j++)
        {
            P[i].Vel[j] = 0;
            P[i].Pos[j] += fac * P[i].GravAccel[j] * 2.0 / (3 * All.CP.Hubble * All.CP.Hubble);
            P[i].GravAccel[j] = 0;
        }
    }

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

/* mark the bins that will be active before the next kick*/
int rebuild_activelist(inttime_t Ti_Current)
{
    int i;

    memset(TimeBinCountType, 0, 6*(TIMEBINS+1)*sizeof(int));
    NumActiveParticle = 0;

    for(i = 0; i < NumPart; i++)
    {
        int bin = P[i].TimeBin;

        if(is_timebin_active(bin, Ti_Current))
        {
            ActiveParticle[NumActiveParticle] = i;
            NumActiveParticle++;
        }
        TimeBinCountType[P[i].Type][bin]++;
    }
    return 0;
}

/*! This routine writes one line for every timestep.
 * FdCPU the cumulative cpu-time consumption in various parts of the
 * code is stored.
 */
void print_timebin_statistics(int NumCurrentTiStep)
{
    double z;
    int i;
    int64_t tot = 0, tot_type[6] = {0};
    int64_t tot_count[TIMEBINS+1] = {0};
    int64_t tot_count_type[6][TIMEBINS+1] = {0};
    int64_t tot_num_force = 0;
    int64_t TotNumPart = 0, TotNumType[6] = {0};

    for(i = 0; i < 6; i ++) {
        sumup_large_ints(TIMEBINS+1, TimeBinCountType[i], tot_count_type[i]);
    }

    for(i = 0; i<TIMEBINS+1; i++) {
        int j;
        for(j=0; j<6; j++) {
            tot_count[i] += tot_count_type[j][i];
            /*Note j*/
            TotNumType[j] += tot_count_type[j][i];
            TotNumPart += tot_count_type[j][i];
        }
        if(is_timebin_active(i, All.Ti_Current))
            tot_num_force += tot_count[i];
    }

    char extra[1024] = {0};

    if(is_PM_timestep(All.Ti_Current))
        strcat(extra, "PM-Step");

    z = 1.0 / (All.Time) - 1;
    message(0, "Begin Step %d, Time: %g (%x), Redshift: %g, Nf = %014ld, Systemstep: %g, Dloga: %g, status: %s\n",
                NumCurrentTiStep, All.Time, All.Ti_Current, z, tot_num_force,
                All.TimeStep, log(All.Time) - log(All.Time - All.TimeStep),
                extra);

    message(0, "TotNumPart: %013ld SPH %013ld BH %010ld STAR %013ld \n",
                TotNumPart, TotNumType[0], TotNumType[5], TotNumType[4]);
    message(0,     "Occupied: % 12ld % 12ld % 12ld % 12ld % 12ld % 12ld dt\n", 0L, 1L, 2L, 3L, 4L, 5L);

    for(i = TIMEBINS;  i >= 0; i--) {
        if(tot_count[i] == 0) continue;
        message(0, " %c bin=%2d % 12ld % 12ld % 12ld % 12ld % 12ld % 12ld %6g\n",
                is_timebin_active(i, All.Ti_Current) ? 'X' : ' ',
                i,
                tot_count_type[0][i],
                tot_count_type[1][i],
                tot_count_type[2][i],
                tot_count_type[3][i],
                tot_count_type[4][i],
                tot_count_type[5][i],
                get_dloga_for_bin(i));

        if(is_timebin_active(i, All.Ti_Current))
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

