#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "allvars.h"
#include "proto.h"
#include "timefac.h"
#include "cosmology.h"
#include "cooling.h"
#include "mymalloc.h"
#include "endrun.h"
#include "system.h"
#include "timestep.h"

/*! \file timestep.c
 *  \brief routines for 'kicking' particles in
 *  momentum space and assigning new timesteps
 */


/* variables for organizing discrete timeline */
static struct time_vars {
    int step; /*!< Maximum (PM) integer timestep from global displacements*/
    int start;           /* current start point of the PM step*/
} PM_Ti;

/*Get the kick time for a timestep, given a start point and a step size.*/
inline int get_kick_ti(int start, int step)
{
    return start + step/2;
}

/*Flat array containing all active particles*/
int NumActiveParticle;
int *ActiveParticle;

int TimeBinCount[TIMEBINS];
int TimeBinCountType[6][TIMEBINS];
int TimeBinActive[TIMEBINS];

void timestep_allocate_memory(int MaxPart)
{
    ActiveParticle = (int *) mymalloc("ActiveParticle", MaxPart * sizeof(int));
}

static void reverse_and_apply_gravity();
static int get_timestep_ti(int p, int dti_max);
static int get_timestep_bin(int dti);
static void do_the_short_range_kick(int i, int tistart, int tiend);
static void do_the_long_range_kick(int tistart, int tiend);
static int get_long_range_timestep_ti(void);

/*Initialise the integer timeline*/
void init_timebins(double TimeInit, double TimeMax)
{
    init_integer_timeline(TimeInit, TimeMax);

    PM_Ti.step = 0;
    PM_Ti.start = 0;
    update_active_timebins(0);

    All.Ti_Current = 0;
}

int is_timebin_active(int i) {
    return TimeBinActive[i];
}

/*Report whether the current timestep is the end of the PM timestep*/
int is_PM_timestep(int ti)
{
    return ti == PM_Ti.start + PM_Ti.step;
}

void set_timebin_active(binmask_t binmask) {
    int bin;
    for(bin = 0; bin < TIMEBINS; bin ++) {
        if(BINMASK(bin) & binmask) {
            TimeBinActive[bin] = 1;
        } else {
            TimeBinActive[bin] = 0;
        }
    }
}

/*! This function sets the (comoving) softening length of all particle
 *  types in the table All.SofteningTable[...].  We check that the physical
 *  softening length is bounded by the Softening-MaxPhys values.
 */
void set_softenings(const double time)
{
    int i;

    if(All.SofteningGas * time > All.SofteningGasMaxPhys)
        All.SofteningTable[0] = All.SofteningGasMaxPhys / time;
    else
        All.SofteningTable[0] = All.SofteningGas;

    if(All.SofteningHalo * time > All.SofteningHaloMaxPhys)
        All.SofteningTable[1] = All.SofteningHaloMaxPhys / time;
    else
        All.SofteningTable[1] = All.SofteningHalo;

    if(All.SofteningDisk * time > All.SofteningDiskMaxPhys)
        All.SofteningTable[2] = All.SofteningDiskMaxPhys / time;
    else
        All.SofteningTable[2] = All.SofteningDisk;

    if(All.SofteningBulge * time > All.SofteningBulgeMaxPhys)
        All.SofteningTable[3] = All.SofteningBulgeMaxPhys / time;
    else
        All.SofteningTable[3] = All.SofteningBulge;

    if(All.SofteningStars * time > All.SofteningStarsMaxPhys)
        All.SofteningTable[4] = All.SofteningStarsMaxPhys / time;
    else
        All.SofteningTable[4] = All.SofteningStars;

    if(All.SofteningBndry * time > All.SofteningBndryMaxPhys)
        All.SofteningTable[5] = All.SofteningBndryMaxPhys / time;
    else
        All.SofteningTable[5] = All.SofteningBndry;

    for(i = 0; i < 6; i++)
        All.ForceSoftening[i] = 2.8 * All.SofteningTable[i];

    All.MinGasHsml = All.MinGasHsmlFractional * All.ForceSoftening[0];
}

void set_global_time(double newtime) {
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
    set_softenings(newtime);
}

/*! This function advances the system in momentum space, i. it does apply the 'kick' operation after the
 *  forces have been computed. Additionally, it assigns new timesteps to particles. At start-up, a
 *  half-timestep is carried out, as well as at the end of the simulation. In between, the half-step kick that
 *  ends the previous timestep and the half-step kick for the new timestep are combined into one operation.
 */
void advance_and_find_timesteps(void)
{
    int pa;

    walltime_measure("/Misc");

    /* FgravkickB is (now - PM0) - (PMhalf - PM0) = now - PMhalf*/
    if(All.MakeGlassFile)
        reverse_and_apply_gravity();

    int new_PM_Ti_step = PM_Ti.step;

    /*Update the displacement timestep*/
    if(is_PM_timestep(All.Ti_Current))
        new_PM_Ti_step = get_long_range_timestep_ti();

    /* Now assign new timesteps and kick */
#ifdef FORCE_EQUAL_TIMESTEPS
    int ti_min=TIMEBASE;
    #pragma omp parallel for
    for(pa = 0; pa < NumActiveParticle; pa++)
    {
        const int i = ActiveParticle[pa];
        int dti = get_timestep_ti(i, new_PM_Ti_step);

        if(dti < ti_min)
            ti_min = dti;
    }

    int ti_min_glob;

    MPI_Allreduce(&ti_min, &ti_min_glob, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
#endif

    int badstepsizecount = 0;
    #pragma omp parallel for
    for(pa = 0; pa < NumActiveParticle; pa++)
    {
        const int i = ActiveParticle[pa];
#ifdef FORCE_EQUAL_TIMESTEPS
        int dti = ti_min_glob;
#else
        int dti = get_timestep_ti(i, new_PM_Ti_step);
#endif
        /* make it a power 2 subdivision */
        int ti_min = TIMEBASE;
        while(ti_min > dti)
            ti_min >>= 1;
        dti = ti_min;

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
            while(TimeBinActive[bin] == 0 && bin > binold)
                bin--;

            dti = bin ? (1 << bin) : 0;
        }

        if(All.Ti_Current >= TIMEBASE)	/* we here finish the last timestep. */
        {
            dti = 0;
            bin = 0;
        }

        if((TIMEBASE - All.Ti_Current) < dti)	/* check that we don't run beyond the end */
        {
            endrun(888, "Integer timeline ran past the end of the bins: %d - %d  < %d\n",TIMEBASE, All.Ti_Current, dti);
        }

        /* This moves particles between time bins:
         * active particles always remain active
         * until reconstruct_timebins is called
         * (during domain, on new timestep).*/
        if(bin != binold)
        {
            /*Update time bin counts*/
            atomic_fetch_and_add(&TimeBinCount[binold],-1);
            atomic_fetch_and_add(&TimeBinCount[bin],1);

            atomic_fetch_and_add(&TimeBinCountType[P[i].Type][binold],-1);
            atomic_fetch_and_add(&TimeBinCountType[P[i].Type][bin],1);

            P[i].TimeBin = bin;
        }

        int dti_old = binold ? (1 << binold) : 0;

        int tistart = get_kick_ti(P[i].Ti_begstep, dti_old);	/* midpoint of old step */
        int tiend = get_kick_ti(P[i].Ti_begstep + dti_old, dti);	/* midpoint of new step */

        P[i].Ti_begstep += dti_old;

        /*This only changes particle i, so is thread-safe.*/
        do_the_short_range_kick(i, tistart, tiend);
    }

    /*Check whether any particles had a bad timestep*/
    int badstepsizecount_global=0;
    MPI_Allreduce(&badstepsizecount, &badstepsizecount_global, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    if(badstepsizecount_global) {
        message(0, "bad timestep spotted: terminating and saving snapshot.\n");
        savepositions(999999, 0);
        endrun(0, "Ending due to bad timestep");
    }


    if(is_PM_timestep(All.Ti_Current))
    {
        /* Note this means we do a half kick on the first timestep.
         * Which means we should also do a half-kick just before output.*/
        const int tistart = get_kick_ti(PM_Ti.start, PM_Ti.step);
        const int tiend =  get_kick_ti(PM_Ti.start + PM_Ti.step, new_PM_Ti_step);
        /* Do long-range kick */
        do_the_long_range_kick(tistart, tiend);
        PM_Ti.start += PM_Ti.step;
        PM_Ti.step = new_PM_Ti_step;
    }

    walltime_measure("/Timeline");
}

/*Advance a long-range timestep and do the desired kick.*/
void do_the_long_range_kick(int tistart, int tiend)
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
}

void do_the_short_range_kick(int i, int tistart, int tiend)
{
    const double Fgravkick = get_gravkick_factor(tistart, tiend);

    int j;

    if(P[i].Ti_kick != tistart) {
        endrun(1, "Ti kick mismatch\n");
    }

    /* do the kick */

    for(j = 0; j < 3; j++)
    {
        P[i].Vel[j] += P[i].GravAccel[j] * Fgravkick;
    }

    P[i].Ti_kick = tiend;

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
 * at the momentum (kick) timestep, accounting
 * for gravity and hydro forces.
 * This is mostly used for artificial viscosity.*/
void sph_VelPred(int i, double * VelPred)
{
    const int ti = P[i].Ti_drift;
    const double Fgravkick2 = get_gravkick_factor(ti, P[i].Ti_kick);
    const double Fhydrokick2 = get_hydrokick_factor(ti, P[i].Ti_kick);
    const double FgravkickB = get_gravkick_factor(ti, get_kick_ti(PM_Ti.start, PM_Ti.step));
    int j;
    for(j = 0; j < 3; j++) {
        VelPred[j] = P[i].Vel[j] - Fgravkick2 * P[i].GravAccel[j]
            - P[i].GravPM[j] * FgravkickB - Fhydrokick2 * SPHP(i).HydroAccel[j];
    }
}

/* This gives the predicted entropy at the particle Kick timestep
 * for the density independent SPH code.*/
double EntropyPred(int i)
{
    const double Fentr = dloga_from_dti(P[i].Ti_drift - P[i].Ti_kick);
    return pow(SPHP(i).Entropy + SPHP(i).DtEntropy * Fentr, 1/GAMMA);
}

double PressurePred(int i)
{
    const double Fentr = dloga_from_dti(P[i].Ti_drift - P[i].Ti_kick);
    return (SPHP(i).Entropy + SPHP(i).DtEntropy * Fentr) * pow(SPHP(i).EOMDensity, GAMMA);
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

    dt = sqrt(2 * All.ErrTolIntAccuracy * All.cf.a * All.SofteningTable[P[p].Type] / ac);
#ifdef ADAPTIVE_GRAVSOFT_FORGAS
    if(P[p].Type == 0)
        dt = sqrt(2 * All.ErrTolIntAccuracy * All.cf.a * P[p].Hsml / 2.8 / ac);
#endif

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
static int
get_timestep_ti(const int p, const int dti_max)
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
    if(!(dti > 1 && dti < TIMEBASE))
    {
        message(1, "Error: A timestep of size zero was assigned on the integer timeline!\n"
                "We better stop.\n"
                "Task=%d type %d Part-ID=%lu dloga=%g, dtmax=%g dti=%d xyz=(%g|%g|%g) tree=(%g|%g|%g), ErrTolIntAccuracy=%g\n\n",
                ThisTask, P[p].Type, (MyIDType)P[p].ID, dloga, dti_max, dti,
                P[p].Pos[0], P[p].Pos[1], P[p].Pos[2], P[p].GravAccel[0], P[p].GravAccel[1],
                P[p].GravAccel[2],
                All.ErrTolIntAccuracy
              );

        message(1, "pm_force=(%g|%g|%g)\n", P[p].GravPM[0], P[p].GravPM[1], P[p].GravPM[2]);

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
double get_long_range_timestep_dloga()
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
            dmean = pow(min_mass[type] / (omega * 3 * All.Hubble * All.Hubble / (8 * M_PI * All.G)), 1.0 / 3);

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
int get_long_range_timestep_ti()
{
    double dloga = get_long_range_timestep_dloga();
    int dti = dti_from_dloga(dloga);
    /* make it a power 2 subdivision */
    int ti_min = TIMEBASE;
    while(ti_min > dti)
        ti_min >>= 1;
    dti = ti_min;
    message(0, "Maximal PM timestep: dloga = %g  (%g)\n", dloga_from_dti(dti), All.MaxSizeTimestep);
    return dti;
}

int get_timestep_bin(int dti)
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

        disp *= 2.0 / (3 * All.Hubble * All.Hubble);

        if(disp > dispmax)
            dispmax = disp;
    }

    MPI_Allreduce(&dispmax, &globmax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    double dmean = pow(P[0].Mass / (All.CP.Omega0 * 3 * All.Hubble * All.Hubble / (8 * M_PI * All.G)), 1.0 / 3);

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
            P[i].Pos[j] += fac * P[i].GravAccel[j] * 2.0 / (3 * All.Hubble * All.Hubble);
            P[i].GravAccel[j] = 0;
        }
    }

}

void rebuild_activelist(void)
{
    int i;

    for(i = 0; i < TIMEBINS; i++)
    {
        TimeBinCount[i] = 0;
        int ptype;
        for(ptype = 0; ptype < 6; ptype ++)
            TimeBinCountType[ptype][i] = 0;
    }

    NumActiveParticle = 0;

    for(i = 0; i < NumPart; i++)
    {
        int bin = P[i].TimeBin;

        if(TimeBinActive[bin])
        {
            ActiveParticle[NumActiveParticle] = i;
            NumActiveParticle++;
        }
        TimeBinCount[bin]++;
        TimeBinCountType[P[i].Type][bin]++;
    }
}


/*! this function returns the next output time that is equal or larger to
 *  ti_curr
 */
int find_next_outputtime(int ti_curr)
{
    int i, ti_next=-1;

    for(i = 0; i < All.OutputListLength; i++)
    {
        const double time = All.OutputListTimes[i];

        if(time >= All.TimeInit && time <= All.TimeMax)
        {
            const int ti = (int) ti_from_loga(log(time));

            if(ti >= ti_curr)
            {
                ti_next = ti;
                break;
            }
        }
    }
    if(ti_next == -1)
    {
        /* Next output is at TimeMax*/
        ti_next = TIMEBASE;
    }
    const double next = exp(loga_from_ti(ti_next));
    message(0, "Setting next time for snapshot file to Time_next= %g \n", next);
    return ti_next;
}

/*! This function finds the next synchronization point of the system
 * (i.e. the earliest point of time any of the particles needs a force
 * computation), and drifts the system to this point of time.  If the
 * system drifts over the desired time of a snapshot file, the
 * function will drift to this moment, generate an output, and then
 * resume the drift.
 */
int find_next_kick(int ti_nextoutput)
{
    int n, ti_next_kick_global;
    int ti_next_kick = TIMEBASE;
    const double timeold = All.Time;
    /*This repopulates all timebins on the first timestep*/
    if(TimeBinCount[0])
        ti_next_kick = All.Ti_Current;

    /* find the next kick time */
    for(n = 1; n < TIMEBINS; n++)
    {
        if(!TimeBinCount[n])
            continue;
	    /* next kick time for this timebin */
        const int dt_bin = (1 << n);
        const int ti_next_for_bin = (All.Ti_Current / dt_bin) * dt_bin + dt_bin;
        if(ti_next_for_bin < ti_next_kick)
            ti_next_kick = ti_next_for_bin;
    }

    /*All processors sync timesteps*/
    MPI_Allreduce(&ti_next_kick, &ti_next_kick_global, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

    /*Convert back to floating point time*/
    double nexttime = exp(loga_from_ti(ti_next_kick_global));

    set_global_time(nexttime);

    All.TimeStep = All.Time - timeold;

    walltime_measure("/Misc");

    return ti_next_kick_global;
}

/* mark the bins that will be active before the next kick*/
int update_active_timebins(int next_kick)
{
    int n;
    int NumForceUpdate = TimeBinCount[0];

    for(n = 1, TimeBinActive[0] = 1; n < TIMEBINS; n++)
    {
        int dti_bin = (1 << n);

        if((next_kick % dti_bin) == 0)
        {
            TimeBinActive[n] = 1;
            NumForceUpdate += TimeBinCount[n];
        }
        else
            TimeBinActive[n] = 0;
    }
    return NumForceUpdate;
}
