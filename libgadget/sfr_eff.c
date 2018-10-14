/***
 * Multi-Phase star formaiton
 *
 * The algorithm here is based on Springel Hernequist 2003, and Okamoto 2010.
 *
 * The source code originally came from sfr_eff.c in Gadget-3. This version has
 * been heavily rewritten to add support for new wind models, new star formation
 * criterions, and more importantly, use the new tree walker routines.
 *
 * I (Yu Feng) feel it is appropriate to release most of this file with a free license,
 * because the implementation here has diverged from the original code by too far.
 *
 * The largest remaining concern are a few functions there were obtained from Gadget-P. 
 * Functions for self-gravity starformation condition and H2 are derived from Gadget-P
 * and used with permission of Phil Hopkins. Please cite the requisite papers if you use them.
 *
 * */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "allvars.h"
#include "sfr_eff.h"
#include "cooling.h"
#include "slotsmanager.h"
#include "timestep.h"
#include "treewalk.h"
#include "winds.h"

/*Cooling only: no star formation*/
static void cooling_direct(int i);

/*
 * This routine does cooling and star formation for
 * the effective multi-phase model.
 */
static int
sfr_cooling_haswork(int target, TreeWalk * tw)
{
    return P[target].Type == 0 && P[target].Mass > 0;
}

/* these guys really shall be local to cooling_and_starformation, but
 * I am too lazy to pass them around to subroutines.
 */
static int * stars_converted;
static int * stars_spawned;
static double * sum_mass_stars;
static double * sum_sm;
static double * localsfr;

static void cooling_relaxed(int i, double egyeff, double dtime, double trelax);

static int get_sfr_condition(int i);
static int make_particle_star(int i);
static void starformation(int i);
static void quicklyastarformation(int i);
static double get_sfr_factor_due_to_selfgravity(int i);
static double get_sfr_factor_due_to_h2(int i);
static double get_starformation_rate_full(int i, double dtime, MyFloat * ne_new, double * trelax, double * egyeff);
static double find_star_mass(int i);

static void
sfr_cool_postprocess(int i, TreeWalk * tw)
{
        int flag;
#ifdef SFR
        /*Remove a wind particle from the delay mode if the (physical) density has dropped sufficiently.*/
        if(SPHP(i).DelayTime > 0 && SPHP(i).Density * All.cf.a3inv < All.WindFreeTravelDensFac * All.PhysDensThresh) {
                SPHP(i).DelayTime = 0;
        }
        /*Reduce the time until the particle can form stars again by the current timestep*/
        if(SPHP(i).DelayTime > 0) {
            const double dloga = get_dloga_for_bin(P[i].TimeBin);
            /*  the proper time duration of the step */
            const double dtime = dloga / All.cf.hubble;
            SPHP(i).DelayTime = DMAX(SPHP(i).DelayTime - dtime, 0);
        }
#endif
        /* check whether conditions for star formation are fulfilled.
         *
         * f=1  normal cooling
         * f=0  star formation
         */
        flag = get_sfr_condition(i);

        /* normal implicit isochoric cooling */
        if(flag == 1 || (All.QuickLymanAlphaProbability > 0 && All.QuickLymanAlphaProbability < 1)) {
            cooling_direct(i);
        }
        if(flag == 0) {
            /* active star formation */
            if(All.QuickLymanAlphaProbability > 0)
                quicklyastarformation(i);
            else
                starformation(i);
        }
}

static void
cool_postprocess(int i, TreeWalk * tw)
{
    cooling_direct(i);
}

/* cooling and star formation routine.*/
void cooling_and_starformation(void)
{
    walltime_measure("/Misc");

    if(!All.CoolingOn)
        return;

    /*When we switch to OpenMP 4.5, which supports array reduction,
     * we can perhaps remove this*/
    stars_spawned = ta_malloc("stars_spawned", int, All.NumThreads);
    stars_converted = ta_malloc("stars_converted", int, All.NumThreads);
    sum_sm = ta_malloc("sum_sm", double, All.NumThreads);
    sum_mass_stars = ta_malloc("sum_mass_stars", double, All.NumThreads);
    localsfr = ta_malloc("localsfr", double, All.NumThreads);
    memset(stars_spawned, 0, All.NumThreads * sizeof(int));
    memset(stars_converted, 0, All.NumThreads * sizeof(int));
    memset(sum_sm, 0, All.NumThreads * sizeof(double));
    memset(sum_mass_stars, 0, All.NumThreads * sizeof(double));
    memset(localsfr, 0, All.NumThreads * sizeof(double));

    TreeWalk tw[1] = {{0}};

    tw->visit = NULL; /* no tree walk */
    tw->ev_label = "SFR_COOL";
    tw->haswork = sfr_cooling_haswork;
    if(All.StarformationOn)
        tw->postprocess = (TreeWalkProcessFunction) sfr_cool_postprocess;
    else
        tw->postprocess = (TreeWalkProcessFunction) cool_postprocess;

    treewalk_run(tw, ActiveParticle, NumActiveParticle);

    if(!All.StarformationOn)
        return;

    int i;
    for(i = 1; i < All.NumThreads; i++)
    {
        sum_mass_stars[0] += sum_mass_stars[i];
        localsfr[0] += localsfr[i];
        sum_sm[0] += sum_sm[i];
        stars_spawned[0] += stars_spawned[i];
        stars_converted[0] += stars_converted[i];
    }

    int tot_spawned, tot_converted;
    MPI_Allreduce(&stars_spawned[0], &tot_spawned, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&stars_converted[0], &tot_converted, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    if(tot_spawned > 0 || tot_converted > 0)
    {
        message(0, "SFR: spawned %d stars, converted %d gas particles into stars\n",
                    tot_spawned, tot_converted);

        /* Note: N_sph is only reduced once domain_garbage_collection is called */

        /* Note: New tree construction can be avoided because of  `force_add_star_to_tree()' */
    }

    double total_sum_mass_stars, total_sm, totsfrrate;

    MPI_Reduce(&localsfr[0], &totsfrrate, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&sum_sm[0], &total_sm, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&sum_mass_stars[0], &total_sum_mass_stars, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if(ThisTask == 0)
    {
        double rate = 0;
        if(All.TimeStep > 0)
            rate = total_sm / (All.TimeStep / (All.Time * All.cf.hubble));

        /* convert to solar masses per yr */

        double rate_in_msunperyear = rate * (All.UnitMass_in_g / SOLAR_MASS) / (All.UnitTime_in_s / SEC_PER_YEAR);

        fprintf(FdSfr, "%g %g %g %g %g\n", All.Time, total_sm, totsfrrate, rate_in_msunperyear,
                total_sum_mass_stars);
        fflush(FdSfr);
    }

    ta_free(localsfr);
    ta_free(sum_mass_stars);
    ta_free(sum_sm);
    ta_free(stars_converted);
    ta_free(stars_spawned);

    walltime_measure("/Cooling/StarFormation");

#ifdef SFR
    /* Now apply the wind model: has to use the new NumActiveParticle.*/
    winds_and_feedback(ActiveParticle, NumActiveParticle);
#endif
}

static void
cooling_direct(int i) {

    /*  the actual time-step */
    double dloga = get_dloga_for_bin(P[i].TimeBin);
    double dtime = dloga / All.cf.hubble;

    double ne = SPHP(i).Ne;	/* electron abundance (gives ionization state and mean molecular weight) */

    double unew = DMAX(All.MinEgySpec,
            (SPHP(i).Entropy + SPHP(i).DtEntropy * dloga) /
            GAMMA_MINUS1 * pow(SPHP(i).EOMDensity * All.cf.a3inv, GAMMA_MINUS1));

#ifdef BLACK_HOLES
    if(SPHP(i).Injected_BH_Energy)
    {
        if(P[i].Mass == 0) {
            endrun(-1, "Encoutered zero mass particle during sfr;"
                      " We haven't implemented tracer particles and this shall not happen\n");
            /* This shall not happend */
            SPHP(i).Injected_BH_Energy = 0;
        }

        unew += SPHP(i).Injected_BH_Energy / P[i].Mass;
        const double u_to_temp_fac = (4 / (8 - 5 * (1 - HYDROGEN_MASSFRAC))) * PROTONMASS / BOLTZMANN * GAMMA_MINUS1
        * All.UnitEnergy_in_cgs / All.UnitMass_in_g;

        double temp = u_to_temp_fac * unew;


        if(temp > 5.0e9)
            unew = 5.0e9 / u_to_temp_fac;

        SPHP(i).Injected_BH_Energy = 0;
    }
#endif

    struct UVBG uvbg;
    GetParticleUVBG(i, &uvbg);
    unew = DoCooling(unew, SPHP(i).Density * All.cf.a3inv, dtime, &uvbg, &ne, SPHP(i).Metallicity);

    SPHP(i).Ne = ne;

    /* upon start-up, we need to protect against dt==0 */
    if(dloga > 0)
    {
        /* note: the adiabatic rate has been already added in ! */
        SPHP(i).DtEntropy = (unew * GAMMA_MINUS1 /
                pow(SPHP(i).EOMDensity * All.cf.a3inv,
                    GAMMA_MINUS1) - SPHP(i).Entropy) / dloga;

        if(SPHP(i).DtEntropy < -0.5 * SPHP(i).Entropy / dloga)
            SPHP(i).DtEntropy = -0.5 * SPHP(i).Entropy / dloga;
    }
}

/* returns 0 if the particle is actively forming stars */
static int get_sfr_condition(int i) {
    int flag = 1;
/* no sfr !*/
    if(!All.StarformationOn) {
        return flag;
    }
    if(SPHP(i).Density * All.cf.a3inv >= All.PhysDensThresh)
        flag = 0;

    if(SPHP(i).Density < All.OverDensThresh)
        flag = 1;

    /* massless particles never form stars! */
    if(P[i].Mass == 0) {
        endrun(-1, "Encoutered zero mass particle during sfr ;"
                  " We haven't implemented tracer particles and this shall not happen\n");
    }
#ifdef SFR
    if(SPHP(i).DelayTime > 0)
        flag = 1;		/* only normal cooling for particles in the wind */
#endif
    if(All.QuickLymanAlphaProbability > 0) {
        double dloga = get_dloga_for_bin(P[i].TimeBin);
        double unew = DMAX(All.MinEgySpec,
                (SPHP(i).Entropy + SPHP(i).DtEntropy * dloga) /
                GAMMA_MINUS1 * pow(SPHP(i).EOMDensity * All.cf.a3inv, GAMMA_MINUS1));

        const double u_to_temp_fac = (4 / (8 - 5 * (1 - HYDROGEN_MASSFRAC))) * PROTONMASS / BOLTZMANN * GAMMA_MINUS1
        * All.UnitEnergy_in_cgs / All.UnitMass_in_g;

        double temp = u_to_temp_fac * unew;

        if(SPHP(i).Density > All.OverDensThresh && temp < 1.0e5)
            flag = 0;
        else
            flag = 1;
    }

    return flag;
}

static int make_particle_star(int i) {
    double mass_of_star = find_star_mass(i);
    int child;
    if(P[i].Type != 0)
        endrun(7772, "Only gas forms stars, what's wrong?");

    int tid = omp_get_thread_num();
    /*Store the SPH particle slot properties, overwritten in slots_convert*/
    struct sph_particle_data oldslot = SPHP(i);
    /* ok, make a star */
    if(P[i].Mass < 1.1 * mass_of_star || All.QuickLymanAlphaProbability > 0)
    {
        /* here the gas particle is eliminated because remaining mass is all converted. */
        stars_converted[tid]++;

        /*If all the mass, just convert the slot*/
        child = slots_convert(i, 4);
    }
    else
    {
        stars_spawned[tid]++;
        /* if we get a fraction of the mass*/
        child = slots_fork(i, 4);

        P[child].Mass = mass_of_star;
        P[i].Mass -= P[child].Mass;
    }

    /*Set properties*/
    sum_mass_stars[tid] += P[child].Mass;
    STARP(child).FormationTime = All.Time;
    STARP(child).BirthDensity = oldslot.Density;
    /*Copy metallicity*/
    STARP(child).Metallicity = oldslot.Metallicity;
    P[child].IsNewParticle = 1;
    return 0;
}

static void cooling_relaxed(int i, double egyeff, double dtime, double trelax) {
    const double densityfac = pow(SPHP(i).EOMDensity * All.cf.a3inv, GAMMA_MINUS1) / GAMMA_MINUS1;
    double egycurrent = SPHP(i).Entropy *  densityfac;

#ifdef BLACK_HOLES
    if(SPHP(i).Injected_BH_Energy > 0)
    {
        struct UVBG uvbg;
        GetParticleUVBG(i, &uvbg);
        egycurrent += SPHP(i).Injected_BH_Energy / P[i].Mass;

        const double u_to_temp_fac = (4 / (8 - 5 * (1 - HYDROGEN_MASSFRAC))) * PROTONMASS / BOLTZMANN * GAMMA_MINUS1
        * All.UnitEnergy_in_cgs / All.UnitMass_in_g;

        double temp = u_to_temp_fac * egycurrent;

        if(temp > 5.0e9)
            egycurrent = 5.0e9 / u_to_temp_fac;

        if(egycurrent > egyeff)
        {
            double ne = SPHP(i).Ne;
            double tcool = GetCoolingTime(egycurrent, SPHP(i).Density * All.cf.a3inv, &uvbg, &ne, SPHP(i).Metallicity);

            if(tcool < trelax && tcool > 0)
                trelax = tcool;
        }

        SPHP(i).Injected_BH_Energy = 0;
    }
#endif

    SPHP(i).Entropy =  (egyeff + (egycurrent - egyeff) * exp(-dtime / trelax)) /densityfac;

    SPHP(i).DtEntropy = 0;

}

/*Forms stars according to the quick lyman alpha star formation criterion,
 * which forms stars with a constant probability (usually 1) if they are star forming*/
static void
quicklyastarformation(int i)
{
    if(get_random_number(P[i].ID + 1) < All.QuickLymanAlphaProbability) {
        make_particle_star(i);
    }
}

/*Forms stars, computes various global counters, and forms winds.*/
static void
starformation(int i)
{
    /*  the proper time-step */
    double dloga = get_dloga_for_bin(P[i].TimeBin);
    double dtime = dloga / All.cf.hubble;

    double egyeff, trelax;
    double rateOfSF = get_starformation_rate_full(i, dtime, &SPHP(i).Ne, &trelax, &egyeff);

    /* amount of stars expect to form */

    double sm = rateOfSF * dtime;

    double p = sm / P[i].Mass;

    int tid = omp_get_thread_num();

    sum_sm[tid] += P[i].Mass * (1 - exp(-p));
    /* convert to Solar per Year.*/
    localsfr[tid] += rateOfSF * (All.UnitMass_in_g / SOLAR_MASS) / (All.UnitTime_in_s / SEC_PER_YEAR);

    double w = get_random_number(P[i].ID);
    SPHP(i).Metallicity += w * METAL_YIELD * (1 - exp(-p));

    if(dloga > 0 && P[i].TimeBin)
    {
      	/* upon start-up, we need to protect against dloga ==0 */
        cooling_relaxed(i, egyeff, dtime, trelax);
    }

    double mass_of_star = find_star_mass(i);
    double prob = P[i].Mass / mass_of_star * (1 - exp(-p));

    if(get_random_number(P[i].ID + 1) < prob) {
        make_particle_star(i);
    }

    if(P[i].Type == 0)	{
        /* to protect using a particle that has been turned into a star */
        SPHP(i).Metallicity += (1 - w) * METAL_YIELD * (1 - exp(-p));
#ifdef SFR
        if(All.WindOn && HAS(All.WindModel, WIND_SUBGRID)) {
            /* Here comes the Springel Hernquist 03 wind model */
            double pw = All.WindEfficiency * sm / P[i].Mass;
            double prob = 1 - exp(-pw);
            double zero[3] = {0, 0, 0};
            if(get_random_number(P[i].ID + 2) < prob)
                make_particle_wind(P[i].ID, i, All.WindSpeed * All.cf.a, zero);
        }
#endif
    }
}

double get_starformation_rate(int i) {
    /* returns SFR in internal units */
    return get_starformation_rate_full(i, 0, NULL, NULL, NULL);
}

static double get_starformation_rate_full(int i, double dtime, MyFloat * ne_new, double * trelax, double * egyeff) {
    double rateOfSF;
    int flag;
    double tsfr;
    double factorEVP, egyhot, ne, tcool, y, x, cloudmass;
    struct UVBG uvbg;

    if(!All.StarformationOn)
        return 0;
    flag = get_sfr_condition(i);

    if(flag == 1) {
        /* this shall not happen but let's put in some safe
         * numbers in case the code run wary!
         *
         * the only case trelax and egyeff are
         * required is in starformation(i)
         * */
        if (trelax) {
            *trelax = All.MaxSfrTimescale;
        }
        if (egyeff) {
            *egyeff = All.EgySpecCold;
        }
        return 0;
    }

    tsfr = sqrt(All.PhysDensThresh / (SPHP(i).Density * All.cf.a3inv)) * All.MaxSfrTimescale;
    /*
     * gadget-p doesn't have this cap.
     * without the cap sm can be bigger than cloudmass.
    */
    if(tsfr < dtime)
        tsfr = dtime;

    GetParticleUVBG(i, &uvbg);

    factorEVP = pow(SPHP(i).Density * All.cf.a3inv / All.PhysDensThresh, -0.8) * All.FactorEVP;

    egyhot = All.EgySpecSN / (1 + factorEVP) + All.EgySpecCold;

    ne = SPHP(i).Ne;

    tcool = GetCoolingTime(egyhot, SPHP(i).Density * All.cf.a3inv, &uvbg, &ne, SPHP(i).Metallicity);
    y = tsfr / tcool * egyhot / (All.FactorSN * All.EgySpecSN - (1 - All.FactorSN) * All.EgySpecCold);

    x = 1 + 1 / (2 * y) - sqrt(1 / y + 1 / (4 * y * y));

    cloudmass = x * P[i].Mass;

    rateOfSF = (1 - All.FactorSN) * cloudmass / tsfr;

    if (ne_new ) {
        *ne_new = ne;
    }

    if (trelax) {
        *trelax = tsfr * (1 - x) / x / (All.FactorSN * (1 + factorEVP));
    }
    if (egyeff) {
        *egyeff = egyhot * (1 - x) + All.EgySpecCold * x;
    }

    if (HAS(All.StarformationCriterion, SFR_CRITERION_MOLECULAR_H2)) {
        rateOfSF *= get_sfr_factor_due_to_h2(i);
    }
    if (HAS(All.StarformationCriterion, SFR_CRITERION_SELFGRAVITY)) {
        rateOfSF *= get_sfr_factor_due_to_selfgravity(i);
    }
    return rateOfSF;
}

void init_cooling_and_star_formation(void)
{
    InitCool();

    /* mean molecular weight assuming ZERO ionization NEUTRAL GAS*/
    double meanweight = 4.0 / (1 + 3 * HYDROGEN_MASSFRAC);

    /*Used for cooling and for timestepping*/
    All.MinEgySpec = 1 / meanweight * (1.0 / GAMMA_MINUS1) * (BOLTZMANN / PROTONMASS) * All.MinGasTemp;
    All.MinEgySpec *= All.UnitMass_in_g / All.UnitEnergy_in_cgs;

    if(!All.StarformationOn)
        return;

    All.OverDensThresh =
        All.CritOverDensity * All.CP.OmegaBaryon * 3 * All.CP.Hubble * All.CP.Hubble / (8 * M_PI * All.G);

    All.PhysDensThresh = All.CritPhysDensity * PROTONMASS / HYDROGEN_MASSFRAC / All.UnitDensity_in_cgs;

    All.EgySpecCold = 1 / meanweight * (1.0 / GAMMA_MINUS1) * (BOLTZMANN / PROTONMASS) * All.TempClouds;
    All.EgySpecCold *= All.UnitMass_in_g / All.UnitEnergy_in_cgs;

    /* mean molecular weight assuming FULL ionization */
    meanweight = 4 / (8 - 5 * (1 - HYDROGEN_MASSFRAC));

    All.EgySpecSN = 1 / meanweight * (1.0 / GAMMA_MINUS1) * (BOLTZMANN / PROTONMASS) * All.TempSupernova;
    All.EgySpecSN *= All.UnitMass_in_g / All.UnitEnergy_in_cgs;

#ifdef SFR
    if(All.WindOn) {
        if(HAS(All.WindModel, WIND_FIXED_EFFICIENCY)) {
            All.WindSpeed = sqrt(2 * All.WindEnergyFraction * All.FactorSN * All.EgySpecSN / (1 - All.FactorSN) / All.WindEfficiency);
            message(0, "Windspeed: %g\n", All.WindSpeed);
        } else {
            All.WindSpeed = sqrt(2 * All.WindEnergyFraction * All.FactorSN * All.EgySpecSN / (1 - All.FactorSN) / 1.0);
            message(0, "Reference Windspeed: %g\n", All.WindSigma0 * All.WindSpeedFactor);
        }
    }
#endif

    if(All.PhysDensThresh == 0)
    {
        double A0, dens, tcool, ne, coolrate, egyhot, x, u4;
        double tsfr, y, peff, fac, neff, egyeff, factorEVP, sigma, thresholdStarburst;

        A0 = All.FactorEVP;

        egyhot = All.EgySpecSN / A0;

        meanweight = 4 / (8 - 5 * (1 - HYDROGEN_MASSFRAC));	/* note: assuming FULL ionization */

        u4 = 1 / meanweight * (1.0 / GAMMA_MINUS1) * (BOLTZMANN / PROTONMASS) * 1.0e4;
        u4 *= All.UnitMass_in_g / All.UnitEnergy_in_cgs;


        dens = 1.0e6 * 3 * All.CP.Hubble * All.CP.Hubble / (8 * M_PI * All.G);

        /* to be guaranteed to get z=0 rate */
        set_global_time(1.0);

        ne = 1.0;

        SetZeroIonization();
        struct UVBG uvbg;
        GetGlobalUVBG(&uvbg);
        /*XXX: We set the threshold without metal cooling;
         * It probably make sense to set the parameters with
         * a metalicity dependence.
         * */
        tcool = GetCoolingTime(egyhot, dens, &uvbg, &ne, 0.0);

        coolrate = egyhot / tcool / dens;

        x = (egyhot - u4) / (egyhot - All.EgySpecCold);

        All.PhysDensThresh =
            x / pow(1 - x,
                    2) * (All.FactorSN * All.EgySpecSN - (1 -
                            All.FactorSN) * All.EgySpecCold) /
                        (All.MaxSfrTimescale * coolrate);

        message(0, "A0= %g  \n", A0);
        message(0, "Computed: PhysDensThresh= %g  (int units)         %g h^2 cm^-3\n", All.PhysDensThresh,
                All.PhysDensThresh / (PROTONMASS / HYDROGEN_MASSFRAC / All.UnitDensity_in_cgs));
        message(0, "EXPECTED FRACTION OF COLD GAS AT THRESHOLD = %g\n", x);
        message(0, "tcool=%g dens=%g egyhot=%g\n", tcool, dens, egyhot);

        dens = All.PhysDensThresh * 10;

        do
        {
            tsfr = sqrt(All.PhysDensThresh / (dens)) * All.MaxSfrTimescale;
            factorEVP = pow(dens / All.PhysDensThresh, -0.8) * All.FactorEVP;
            egyhot = All.EgySpecSN / (1 + factorEVP) + All.EgySpecCold;

            ne = 0.5;
            tcool = GetCoolingTime(egyhot, dens, &uvbg, &ne, 0.0);

            y = tsfr / tcool * egyhot / (All.FactorSN * All.EgySpecSN - (1 - All.FactorSN) * All.EgySpecCold);
            x = 1 + 1 / (2 * y) - sqrt(1 / y + 1 / (4 * y * y));
            egyeff = egyhot * (1 - x) + All.EgySpecCold * x;

            peff = GAMMA_MINUS1 * dens * egyeff;

            fac = 1 / (log(dens * 1.025) - log(dens));
            dens *= 1.025;

            neff = -log(peff) * fac;

            tsfr = sqrt(All.PhysDensThresh / (dens)) * All.MaxSfrTimescale;
            factorEVP = pow(dens / All.PhysDensThresh, -0.8) * All.FactorEVP;
            egyhot = All.EgySpecSN / (1 + factorEVP) + All.EgySpecCold;

            ne = 0.5;
            tcool = GetCoolingTime(egyhot, dens, &uvbg, &ne, 0.0);

            y = tsfr / tcool * egyhot / (All.FactorSN * All.EgySpecSN - (1 - All.FactorSN) * All.EgySpecCold);
            x = 1 + 1 / (2 * y) - sqrt(1 / y + 1 / (4 * y * y));
            egyeff = egyhot * (1 - x) + All.EgySpecCold * x;

            peff = GAMMA_MINUS1 * dens * egyeff;

            neff += log(peff) * fac;
        }
        while(neff > 4.0 / 3);

        thresholdStarburst = dens;

        message(0, "Run-away sets in for dens=%g\n", thresholdStarburst);
        message(0, "Dynamic range for quiescent star formation= %g\n", thresholdStarburst / All.PhysDensThresh);

        sigma = 10.0 / All.CP.Hubble * 1.0e-10 / pow(1.0e-3, 2);

        message(0, "Isotherm sheet central density: %g   z0=%g\n",
                M_PI * All.G * sigma * sigma / (2 * GAMMA_MINUS1) / u4,
                GAMMA_MINUS1 * u4 / (2 * M_PI * All.G * sigma));

    }
}

static double
find_star_mass(int i)
{
    double mass_of_star =  All.MassTable[0] / GENERATIONS;
    if(mass_of_star > P[i].Mass) {
        /* if some mass has been stolen by BH, e.g */
        mass_of_star = P[i].Mass;
    }
    /* if we are the last particle */
    if(fabs(mass_of_star - P[i].Mass) / mass_of_star < 0.5) {
        mass_of_star = P[i].Mass;
    }
    return mass_of_star;
}

/********************
 *
 * The follow functions are from Desika and Gadget-P.
 * We really are mostly concerned about H2 here.
 *
 * You may need a license to run with these modess.
 
 * */
#if defined SPH_GRAD_RHO
static double ev_NH_from_GradRho(MyFloat gradrho[3], double hsml, double rho, double include_h)
{
    /* column density from GradRho, copied from gadget-p; what is it
     * calculating? */
    double gradrho_mag;
    if(rho<=0) {
        gradrho_mag = 0;
    } else {
        gradrho_mag = sqrt(gradrho[0]*gradrho[0]+gradrho[1]*gradrho[1]+gradrho[2]*gradrho[2]);
        if(gradrho_mag > 0) {gradrho_mag = rho*rho/gradrho_mag;} else {gradrho_mag=0;}
        if(include_h > 0) gradrho_mag += include_h*rho*hsml;
    }
    return gradrho_mag; // *(Z/Zsolar) add metallicity dependence
}
#endif

static double get_sfr_factor_due_to_h2(int i) {
    /*  Krumholz & Gnedin fitting function for f_H2 as a function of local
     *  properties, from gadget-p; we return the enhancement on SFR in this
     *  function */

#if ! defined SPH_GRAD_RHO
    /* if SPH_GRAD_RHO is not enabled, disable H2 molecular gas
     * this really shall not happen because begrun will check against the
     * condition. Ditto if not metal tracking.
     * */
    return 1.0;
#else
    double tau_fmol;
    double zoverzsun = SPHP(i).Metallicity/METAL_YIELD;
    tau_fmol = ev_NH_from_GradRho(SPHP(i).GradRho,P[i].Hsml,SPHP(i).Density,1) * All.cf.a2inv;
    tau_fmol *= (0.1 + zoverzsun);
    if(tau_fmol>0) {
        tau_fmol *= 434.78*All.UnitDensity_in_cgs*All.CP.HubbleParam*All.UnitLength_in_cm;
        double y = 0.756*(1+3.1*pow(zoverzsun,0.365));
        y = log(1+0.6*y+0.01*y*y)/(0.6*tau_fmol);
        y = 1-0.75*y/(1+0.25*y);
        if(y<0) y=0;
        if(y>1) y=1;
        return y;

    } // if(tau_fmol>0)
    return 1.0;
#endif
}

static double get_sfr_factor_due_to_selfgravity(int i) {
    double divv = SPHP(i).DivVel * All.cf.a2inv;

    divv += 3.0*All.cf.hubble_a2; // hubble-flow correction

    if(HAS(All.StarformationCriterion, SFR_CRITERION_CONVERGENT_FLOW)) {
        if( divv>=0 ) return 0; // restrict to convergent flows (optional) //
    }

    double dv2abs = (divv*divv
            + (SPHP(i).CurlVel*All.cf.a2inv)
            * (SPHP(i).CurlVel*All.cf.a2inv)
           ); // all in physical units
    double alpha_vir = 0.2387 * dv2abs/(All.G * SPHP(i).Density*All.cf.a3inv);

    double y = 1.0;

    if((alpha_vir < 1.0)
    || (SPHP(i).Density * All.cf.a3inv > 100. * All.PhysDensThresh)
    )  {
        y = 66.7;
    } else {
        y = 0.1;
    }
    // PFH: note the latter flag is an arbitrary choice currently set
    // -by hand- to prevent runaway densities from this prescription! //

    if (HAS(All.StarformationCriterion, SFR_CRITERION_CONTINUOUS_CUTOFF)) {
        // continuous cutoff w alpha_vir instead of sharp (optional) //
        y *= 1.0/(1.0 + alpha_vir);
    }
    return y;
}
