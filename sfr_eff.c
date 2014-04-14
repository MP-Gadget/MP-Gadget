#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "allvars.h"
#include "proto.h"
#include "forcetree.h"

#ifdef COOLING
static double u_to_temp_fac; /* assuming very hot !*/
static unsigned int bits;

/* these guys really shall be local to cooling_and_starformation, but
 * I am too lazy to pass them around to subroutines.
 */
static int stars_converted;
static int stars_spawned;
static double sum_sm;
static double sum_mass_stars;

static int get_sfr_condition(int i);
static void cooling_relaxed(int i, double egyeff, double dtime, double trelax);
static void cooling_direct(int i);
static void starformation(int i);
static int make_particle_wind(int i, double efficiency);
static int make_particle_star(int i, double p);
static double get_sfr_factor_due_to_selfgravity(int i);
static double get_sfr_factor_due_to_h2(int i);

/*
 * This routine does cooling and star formation for
 * the effective multi-phase model.
 */


void cooling_and_starformation(void)
    /* cooling routine when star formation is enabled */
{
    int i, bin, flag;
    u_to_temp_fac = (4 / (8 - 5 * (1 - HYDROGEN_MASSFRAC))) * PROTONMASS / BOLTZMANN * GAMMA_MINUS1
        * All.UnitEnergy_in_cgs / All.UnitMass_in_g;

#ifdef FLTROUNDOFFREDUCTION
#if defined(BH_THERMALFEEDBACK) || defined(BH_KINETICFEEDBACK)
    for(i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i])
        if(P[i].Type == 0)
            SPHP(i).i.Injected_BH_Energy = FLT(SPHP(i).i.dInjected_BH_Energy);
#endif
#endif

    for(bin = 0; bin < TIMEBINS; bin++) {
        if(!TimeBinActive[bin]) continue;
        TimeBinSfr[bin] = 0;
    }

    stars_spawned = stars_converted = 0;
    sum_sm = sum_mass_stars = 0;

    for(i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i])
    {
        if(P[i].Type != 0) continue;
#ifdef MAGNETIC
        SPHP(i).XColdCloud = x;
#endif
#ifdef WINDS
        if(SPHP(i).DelayTime > 0) {
            double dt = (P[i].TimeBin ? (1 << P[i].TimeBin) : 0) * All.Timebase_interval;
                /*  the actual time-step */

            double dtime;

            dtime = dt / All.cf.hubble;

            SPHP(i).DelayTime -= dtime;
        }

        if(SPHP(i).DelayTime > 0) {
            if(SPHP(i).d.Density * All.cf.a3inv < All.WindFreeTravelDensFac * All.PhysDensThresh)
                SPHP(i).DelayTime = 0;
        } else {
            SPHP(i).DelayTime = 0;
        }
#endif
#ifdef MAGNETIC
        x=0.;
#endif
        /* check whether conditions for star formation are fulfilled.
         *  
         * f=1  normal cooling
         * f=0  star formation
         */
        flag = get_sfr_condition(i);

#if !defined(NOISMPRESSURE) && !defined(QUICK_LYALPHA)
        /* normal implicit isochoric cooling */
        if(flag == 1) {
            cooling_direct(i);
        }
#else
        /* always do direct cooling in these cases */
        cooling_direct(i);
#endif
        if(flag == 0) {
            /* active star formation */
            starformation(i);
        }
    }				/* end of main loop over active particles */


    int tot_spawned, tot_converted;
    MPI_Allreduce(&stars_spawned, &tot_spawned, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&stars_converted, &tot_converted, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    if(tot_spawned > 0 || tot_converted > 0)
    {
        if(ThisTask == 0)
        {
            printf("SFR: spawned %d stars, converted %d gas particles into stars\n",
                    tot_spawned, tot_converted);
            fflush(stdout);
        }


        All.TotNumPart += tot_spawned;
        All.TotN_sph -= tot_converted;
        NumPart += stars_spawned;

        /* Note: N_sph is only reduced once rearrange_particle_sequence is called */

        /* Note: New tree construction can be avoided because of  `force_add_star_to_tree()' */
    }

    double sfrrate = 0, totsfrrate;
    for(bin = 0; bin < TIMEBINS; bin++)
        if(TimeBinCount[bin])
            sfrrate += TimeBinSfr[bin];

    MPI_Allreduce(&sfrrate, &totsfrrate, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    double total_sum_mass_stars, total_sm;

    MPI_Reduce(&sum_sm, &total_sm, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&sum_mass_stars, &total_sum_mass_stars, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if(ThisTask == 0)
    {
        double rate;
        double rate_in_msunperyear;
        if(All.TimeStep > 0)
            rate = total_sm / (All.TimeStep / (All.Time * All.cf.hubble));
        else
            rate = 0;

        /* convert to solar masses per yr */

        rate_in_msunperyear = rate * (All.UnitMass_in_g / SOLAR_MASS) / (All.UnitTime_in_s / SEC_PER_YEAR);

        fprintf(FdSfr, "%g %g %g %g %g\n", All.Time, total_sm, totsfrrate, rate_in_msunperyear,
                total_sum_mass_stars);
        fflush(FdSfr);
    }
}

static void cooling_direct(int i) {

#ifdef COSMIC_RAYS
    int CRpop;

#ifdef CR_SN_INJECTION
    double tinj = 0.0, instant_reheat = 0.0;
    int InjPopulation;
#endif
#endif



    double dt = (P[i].TimeBin ? (1 << P[i].TimeBin) : 0) * All.Timebase_interval;
        /*  the actual time-step */

    double dtime;

    dtime = dt / All.cf.hubble;

    SPHP(i).Sfr = 0;
#if defined(COSMIC_RAYS) && defined(CR_OUTPUT_INJECTION)
    SPHP(i).CR_Specific_SupernovaHeatingRate = 0;
#endif
    double ne = SPHP(i).Ne;	/* electron abundance (gives ionization state and mean molecular weight) */

    double unew = DMAX(All.MinEgySpec,
            (SPHP(i).Entropy + SPHP(i).e.DtEntropy * dt) /
            GAMMA_MINUS1 * pow(SPHP(i).EOMDensity * All.cf.a3inv, GAMMA_MINUS1));

#if defined(BH_THERMALFEEDBACK) || defined(BH_KINETICFEEDBACK)
    if(SPHP(i).i.Injected_BH_Energy)
    {
        if(P[i].Mass == 0)
            SPHP(i).i.Injected_BH_Energy = 0;
        else
            unew += SPHP(i).i.Injected_BH_Energy / P[i].Mass;

        double temp = u_to_temp_fac * unew;


        if(temp > 5.0e9)
            unew = 5.0e9 / u_to_temp_fac;

#ifdef FLTROUNDOFFREDUCTION
        SPHP(i).i.dInjected_BH_Energy = 0;
#else
        SPHP(i).i.Injected_BH_Energy = 0;
#endif
    }
#endif
#ifdef RT_COOLING_PHOTOHEATING
    unew = radtransfer_cooling_photoheating(i, dtime);

    if(P[i].TimeBin)	/* upon start-up, we need to protect against dt==0 */
    {
        /* note: the adiabatic rate has been already added in ! */

        if(dt > 0)
        {
            SPHP(i).e.DtEntropy += unew * GAMMA_MINUS1 /
                pow(SPHP(i).EOMDensity * All.cf.a3inv, GAMMA_MINUS1) / dt;

            if(SPHP(i).e.DtEntropy < -0.5 * SPHP(i).Entropy / dt)
                SPHP(i).e.DtEntropy = -0.5 * SPHP(i).Entropy / dt;
        }
    }
#else
    unew = DoCooling(unew, SPHP(i).d.Density * All.cf.a3inv, dtime, &ne, P[i].Metallicity);

    SPHP(i).Ne = ne;

    if(P[i].TimeBin)	/* upon start-up, we need to protect against dt==0 */
    {
        /* note: the adiabatic rate has been already added in ! */

        if(dt > 0)
        {
#ifdef COSMIC_RAYS
            for(CRpop = 0; CRpop < NUMCRPOP; CRpop++)
                unew += CR_Particle_ThermalizeAndDissipate(SphP + i, dtime, CRpop);
#endif

            SPHP(i).e.DtEntropy = (unew * GAMMA_MINUS1 /
                    pow(SPHP(i).EOMDensity * All.cf.a3inv,
                        GAMMA_MINUS1) - SPHP(i).Entropy) / dt;

            if(SPHP(i).e.DtEntropy < -0.5 * SPHP(i).Entropy / dt)
                SPHP(i).e.DtEntropy = -0.5 * SPHP(i).Entropy / dt;
        }
    }
#endif
}

#endif /* closing of COOLING-conditional */


#if defined(SFR)


/* returns 0 if the particle is actively forming stars */
static int get_sfr_condition(int i) {
    int flag = 1;
/* no sfr !*/
    if(!All.StarformationOn) {
        return flag;
    }
    if(SPHP(i).d.Density * All.cf.a3inv >= All.PhysDensThresh)
        flag = 0;

    if(All.ComovingIntegrationOn)
        if(SPHP(i).d.Density < All.OverDensThresh)
            flag = 1;

#ifdef BLACK_HOLES
    if(P[i].Mass == 0)
        flag = 1;
#endif

#ifdef WINDS
    if(SPHP(i).DelayTime > 0)
        flag = 1;		/* only normal cooling for particles in the wind */
#endif

#ifdef QUICK_LYALPHA
    temp = u_to_temp_fac * (SPHP(i).Entropy + SPHP(i).e.DtEntropy * dt) /
        GAMMA_MINUS1 * pow(SPHP(i).EOMDensity * All.cf.a3inv, GAMMA_MINUS1);

    if(SPHP(i).d.Density > All.OverDensThresh && temp < 1.0e5)
        flag = 0;
    else
        flag = 1;
#endif
    return flag;
}


static int make_particle_wind(int i, double efficiency) {
    /* returns 0 if particle i is converteed to wind. */
    int j;
    double prob = 1 - exp(-efficiency);

    if(get_random_number(P[i].ID + 2) >= prob)	return 1;
    /* ok, make the particle go into the wind */
    double v =
        sqrt(2 * All.WindEnergyFraction * All.FactorSN *
                All.EgySpecSN / (1 - All.FactorSN) / All.WindEfficiency);
    double dir[3];
#ifdef ISOTROPICWINDS
    double theta = acos(2 * get_random_number(P[i].ID + 3) - 1);
    double phi = 2 * M_PI * get_random_number(P[i].ID + 4);

    dir[0] = sin(theta) * cos(phi);
    dir[1] = sin(theta) * sin(phi);
    dir[2] = cos(theta);
#else
    dir[0] = P[i].g.GravAccel[1] * P[i].Vel[2] - P[i].g.GravAccel[2] * P[i].Vel[1];
    dir[1] = P[i].g.GravAccel[2] * P[i].Vel[0] - P[i].g.GravAccel[0] * P[i].Vel[2];
    dir[2] = P[i].g.GravAccel[0] * P[i].Vel[1] - P[i].g.GravAccel[1] * P[i].Vel[0];
#endif

    double norm = 0;
    for(j = 0; j < 3; j++)
        norm += dir[j] * dir[j];

    norm = sqrt(norm);
    if(get_random_number(P[i].ID + 5) < 0.5)
        norm = -norm;

    if(norm != 0)
    {
        for(j = 0; j < 3; j++)
            dir[j] /= norm;

        for(j = 0; j < 3; j++)
        {
            P[i].Vel[j] += v * All.cf.a * dir[j];
            SPHP(i).VelPred[j] += v * All.cf.a * dir[j];
        }

        SPHP(i).DelayTime = All.WindFreeTravelLength / v;
    }
    return 0;
}
static int make_particle_star(int i, double p) {

    int number_of_stars_generated;
    if(bits == 0)
        number_of_stars_generated = 0;
    else
        number_of_stars_generated = (P[i].ID >> (sizeof(MyIDType)*8 - bits));

    double mass_of_star = P[i].Mass / (GENERATIONS - number_of_stars_generated);


#ifdef QUICK_LYALPHA
    double prob = 2; /* always make a star */
    /* this preserves the random sequence; if there is one. */
#else
    double prob = P[i].Mass / mass_of_star * (1 - exp(-p));
#endif
    if(get_random_number(P[i].ID + 1) >= prob)	return -1;

    /* ok, make a star */
    if(number_of_stars_generated == (GENERATIONS - 1))
    {
        /* here we turn the gas particle itself into a star */
        Stars_converted++;
        stars_converted++;

        sum_mass_stars += P[i].Mass;

        P[i].Type = 4;
        TimeBinCountSph[P[i].TimeBin]--;
        TimeBinSfr[P[i].TimeBin] -= SPHP(i).Sfr;

#ifdef STELLARAGE
        P[i].StellarAge = All.Time;
#endif
    }
    else
    {
        /* here we spawn a new star particle */

        if(NumPart + stars_spawned >= All.MaxPart)
        {
            printf
                ("On Task=%d with NumPart=%d we try to spawn %d particles. Sorry, no space left...(All.MaxPart=%d)\n",
                 ThisTask, NumPart, stars_spawned, All.MaxPart);
            fflush(stdout);
            endrun(8888);
        }

        P[NumPart + stars_spawned] = P[i];
        P[NumPart + stars_spawned].Type = 4;
#ifdef SNIA_HEATING
        P[NumPart + stars_spawned].Hsml = All.SofteningTable[0];
#endif

        NextActiveParticle[NumPart + stars_spawned] = FirstActiveParticle;
        FirstActiveParticle = NumPart + stars_spawned;
        NumForceUpdate++;

        TimeBinCount[P[NumPart + stars_spawned].TimeBin]++;

        PrevInTimeBin[NumPart + stars_spawned] = i;
        NextInTimeBin[NumPart + stars_spawned] = NextInTimeBin[i];
        if(NextInTimeBin[i] >= 0)
            PrevInTimeBin[NextInTimeBin[i]] = NumPart + stars_spawned;
        NextInTimeBin[i] = NumPart + stars_spawned;
        if(LastInTimeBin[P[i].TimeBin] == i)
            LastInTimeBin[P[i].TimeBin] = NumPart + stars_spawned;

        P[i].ID += ((MyIDType) 1 << (sizeof(MyIDType)*8 - bits));

        P[NumPart + stars_spawned].Mass = mass_of_star;
        P[i].Mass -= P[NumPart + stars_spawned].Mass;
        sum_mass_stars += P[NumPart + stars_spawned].Mass;
#ifdef STELLARAGE
        P[NumPart + stars_spawned].StellarAge = All.Time;
#endif
        force_add_star_to_tree(i, NumPart + stars_spawned);

        stars_spawned++;
    }
    return 0;
}
static void cooling_relaxed(int i, double egyeff, double dtime, double trelax) {
    double egycurrent =
        SPHP(i).Entropy * pow(SPHP(i).EOMDensity * All.cf.a3inv, GAMMA_MINUS1) / GAMMA_MINUS1;

#ifdef COSMIC_RAYS
#ifdef CR_SN_INJECTION
    if(All.CR_SNEff > 0)
    {
        if(NUMCRPOP > 1)
            InjPopulation = CR_Find_Alpha_to_InjectTo(All.CR_SNAlpha);
        else
            InjPopulation = 0;

        tinj =
            SPHP(i).CR_E0[InjPopulation] / (p * All.FeedbackEnergy * All.CR_SNEff / dtime);

        instant_reheat =
            CR_Particle_SupernovaFeedback(&SPHP(i), p * All.FeedbackEnergy * All.CR_SNEff,
                    tinj);
    }
    else
        instant_reheat = 0;

#if defined(COSMIC_RAYS) && defined(CR_OUTPUT_INJECTION)
    SPHP(i).CR_Specific_SupernovaHeatingRate =
        (p * All.FeedbackEnergy * All.CR_SNEff - instant_reheat) / dtime;
#endif
    egycurrent += instant_reheat;
#endif
    for(CRpop = 0; CRpop < NUMCRPOP; CRpop++)
        egycurrent += CR_Particle_ThermalizeAndDissipate(SphP + i, dtime, CRpop);
#endif /* COSMIC_RAYS */


#if defined(BH_THERMALFEEDBACK) || defined(BH_KINETICFEEDBACK)
    if(SPHP(i).i.Injected_BH_Energy > 0)
    {
        egycurrent += SPHP(i).i.Injected_BH_Energy / P[i].Mass;

        double temp = u_to_temp_fac * egycurrent;

        if(temp > 5.0e9)
            egycurrent = 5.0e9 / u_to_temp_fac;

        if(egycurrent > egyeff)
        {
            double ne = SPHP(i).Ne;
            double tcool = GetCoolingTime(egycurrent, SPHP(i).d.Density * All.cf.a3inv, &ne, P[i].Metallicity);

            if(tcool < trelax && tcool > 0)
                trelax = tcool;
        }

        SPHP(i).i.Injected_BH_Energy = 0;
    }
#endif
#ifdef MAGNETICSEED
    SPHP(i).MagSeed =  egyhot * factorEVP *  1E-2;   // This is the definition of how much energy we will put in MF (we neglect cooling here, we have to check if thios has sense)
    egyeff-= SPHP(i).MagSeed;                 //Here we also substract that to the feedback

    SPHP(i).MagSeed *= All.UnitMass_in_g / All.UnitEnergy_in_cgs * SPHP(i).d.Density * (1.-x) * All.cf.a3inv *All.Time ;// * All.cf.a3inv
#endif



#if !defined(NOISMPRESSURE)
    SPHP(i).Entropy =
        (egyeff +
         (egycurrent -
          egyeff) * exp(-dtime / trelax)) * GAMMA_MINUS1 /
        pow(SPHP(i).EOMDensity * All.cf.a3inv, GAMMA_MINUS1);

    SPHP(i).e.DtEntropy = 0;
#endif

}

static void starformation(int i) {
    /* the upper bits of the gas particle ID store how man stars this gas
       particle gas already generated */


#if !defined(QUICK_LYALPHA)
    double dt = (P[i].TimeBin ? (1 << P[i].TimeBin) : 0) * All.Timebase_interval;
        /*  the actual time-step */

    double dtime = dt / All.cf.hubble;

    /* 
     * gadget-p doesn't have this cap.
     * without the cap sm can be bigger than cloudmass.
        if(tsfr < dtime)
            tsfr = dtime;
    */

    double egyeff, trelax;
    double rateOfSF = get_starformation_rate(i, &trelax, &egyeff);

    /* amount of stars expect to form */

    double sm = rateOfSF * dtime;	

    double p = sm / P[i].Mass;

    sum_sm += P[i].Mass * (1 - exp(-p));

    /* convert to Solar per Year but is this damn variable otherwise used 
     * at all? */
    SphP[i].Sfr = rateOfSF *
        (All.UnitMass_in_g / SOLAR_MASS) / (All.UnitTime_in_s / SEC_PER_YEAR);

    TimeBinSfr[P[i].TimeBin] += SPHP(i).Sfr;

#ifdef METALS
    double w = get_random_number(P[i].ID);
    P[i].Metallicity += w * METAL_YIELD * (1 - exp(-p));
#endif

    if(dt > 0 && P[i].TimeBin)
    {
      	/* upon start-up, we need to protect against dt==0 */
        cooling_relaxed(i, egyeff, dtime, trelax);
    }


#else /* belongs to ifndef(QUICK_LYALPHA) */

    SPHP(i).Sfr = 0;

#endif /* ends to QUICK_LYALPHA */

    make_particle_star(i, p);

    if(P[i].Type == 0)	{
    /* to protect using a particle that has been turned into a star */
#ifdef METALS
        P[i].Metallicity += (1 - w) * METAL_YIELD * (1 - exp(-p));
#endif

#ifdef WINDS
    /* Here comes the wind model */
        double pw = All.WindEfficiency * sm / P[i].Mass;
        make_particle_wind(i, pw);
#endif
    }


}

double get_starformation_rate(int i, double * trelax, double * egyeff)
{
    double rateOfSF;
    int flag;
    double tsfr;
    double factorEVP, egyhot, ne, tcool, y, x, cloudmass;

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

    tsfr = sqrt(All.PhysDensThresh / (SPHP(i).d.Density * All.cf.a3inv)) * All.MaxSfrTimescale;

    factorEVP = pow(SPHP(i).d.Density * All.cf.a3inv / All.PhysDensThresh, -0.8) * All.FactorEVP;

    egyhot = All.EgySpecSN / (1 + factorEVP) + All.EgySpecCold;

    ne = SPHP(i).Ne;

    tcool = GetCoolingTime(egyhot, SPHP(i).d.Density * All.cf.a3inv, &ne, P[i].Metallicity);
    y = tsfr / tcool * egyhot / (All.FactorSN * All.EgySpecSN - (1 - All.FactorSN) * All.EgySpecCold, P[i].Metallicity);

    x = 1 + 1 / (2 * y) - sqrt(1 / y + 1 / (4 * y * y));

    cloudmass = x * P[i].Mass;

    rateOfSF = (1 - All.FactorSN) * cloudmass / tsfr;

    if (trelax) {
        *trelax = tsfr * (1 - x) / x / (All.FactorSN * (1 + factorEVP));
    }
    if (egyeff) {
        *egyeff = egyhot * (1 - x) + All.EgySpecCold * x;
    }

    if (All.StarformationCriterion & SFR_CRITERION_MOLECULAR_H2) {
        rateOfSF *= get_sfr_factor_due_to_h2(i);
    }
    if (All.StarformationCriterion & SFR_CRITERION_SELFGRAVITY) {
        rateOfSF *= get_sfr_factor_due_to_selfgravity(i);
    }
    return rateOfSF;
}

void init_clouds(void)
{
    double A0, dens, tcool, ne, coolrate, egyhot, x, u4, meanweight;
    double tsfr, y, peff, fac, neff, egyeff, factorEVP, sigma, thresholdStarburst;

    if(All.PhysDensThresh == 0)
    {
        A0 = All.FactorEVP;

        egyhot = All.EgySpecSN / A0;

        meanweight = 4 / (8 - 5 * (1 - HYDROGEN_MASSFRAC));	/* note: assuming FULL ionization */

        u4 = 1 / meanweight * (1.0 / GAMMA_MINUS1) * (BOLTZMANN / PROTONMASS) * 1.0e4;
        u4 *= All.UnitMass_in_g / All.UnitEnergy_in_cgs;


        if(All.ComovingIntegrationOn)
            dens = 1.0e6 * 3 * All.Hubble * All.Hubble / (8 * M_PI * All.G);
        else
            dens = 1.0e6 * 3 * All.Hubble * All.Hubble / (8 * M_PI * All.G);

        if(All.ComovingIntegrationOn)
        {
            /* to be guaranteed to get z=0 rate */
            set_global_time(1.0);
            IonizeParams();
        }

        ne = 1.0;

        SetZeroIonization();

        /*XXX: We set the threshold without metal cooling;
         * It probably make sense to set the parameters with
         * a metalicity dependence.
         * */
        tcool = GetCoolingTime(egyhot, dens, &ne, 0.0);

        coolrate = egyhot / tcool / dens;

        x = (egyhot - u4) / (egyhot - All.EgySpecCold);

        All.PhysDensThresh =
            x / pow(1 - x,
                    2) * (All.FactorSN * All.EgySpecSN - (1 -
                            All.FactorSN) * All.EgySpecCold) /
                        (All.MaxSfrTimescale * coolrate);

        if(ThisTask == 0)
        {
            printf("\nA0= %g  \n", A0);
            printf("Computed: PhysDensThresh= %g  (int units)         %g h^2 cm^-3\n", All.PhysDensThresh,
                    All.PhysDensThresh / (PROTONMASS / HYDROGEN_MASSFRAC / All.UnitDensity_in_cgs));
            printf("EXPECTED FRACTION OF COLD GAS AT THRESHOLD = %g\n\n", x);
            printf("tcool=%g dens=%g egyhot=%g\n", tcool, dens, egyhot);
        }

        dens = All.PhysDensThresh * 10;

        do
        {
            tsfr = sqrt(All.PhysDensThresh / (dens)) * All.MaxSfrTimescale;
            factorEVP = pow(dens / All.PhysDensThresh, -0.8) * All.FactorEVP;
            egyhot = All.EgySpecSN / (1 + factorEVP) + All.EgySpecCold;

            ne = 0.5;
            tcool = GetCoolingTime(egyhot, dens, &ne, 0.0);

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
            tcool = GetCoolingTime(egyhot, dens, &ne, 0.0);

            y = tsfr / tcool * egyhot / (All.FactorSN * All.EgySpecSN - (1 - All.FactorSN) * All.EgySpecCold);
            x = 1 + 1 / (2 * y) - sqrt(1 / y + 1 / (4 * y * y));
            egyeff = egyhot * (1 - x) + All.EgySpecCold * x;

            peff = GAMMA_MINUS1 * dens * egyeff;

            neff += log(peff) * fac;
        }
        while(neff > 4.0 / 3);

        thresholdStarburst = dens;

#ifdef MODIFIEDBONDI
        All.BlackHoleRefDensity = thresholdStarburst;
        All.BlackHoleRefSoundspeed = sqrt(GAMMA * GAMMA_MINUS1 * egyeff);
#endif


        if(ThisTask == 0)
        {
            printf("Run-away sets in for dens=%g\n", thresholdStarburst);
            printf("Dynamic range for quiescent star formation= %g\n", thresholdStarburst / All.PhysDensThresh);
            fflush(stdout);
        }

        integrate_sfr();

        if(ThisTask == 0)
        {
            sigma = 10.0 / All.Hubble * 1.0e-10 / pow(1.0e-3, 2);

            printf("Isotherm sheet central density: %g   z0=%g\n",
                    M_PI * All.G * sigma * sigma / (2 * GAMMA_MINUS1) / u4,
                    GAMMA_MINUS1 * u4 / (2 * M_PI * All.G * sigma));
            fflush(stdout);

        }

        if(All.ComovingIntegrationOn)
        {
            set_global_time(All.TimeBegin);
            IonizeParams();
        }

#ifdef WINDS
        if(All.WindEfficiency > 0)
            if(ThisTask == 0)
                printf("Windspeed: %g\n",
                        sqrt(2 * All.WindEnergyFraction * All.FactorSN * All.EgySpecSN / (1 - All.FactorSN) /
                            All.WindEfficiency));
#endif
    }
}

void integrate_sfr(void)
{
    double rho0, rho, rho2, q, dz, gam, sigma = 0, sigma_u4, sigmasfr = 0, ne, P1;
    double x = 0, y, P, P2, x2, y2, tsfr2, factorEVP2, egyhot2, tcool2, drho, dq;
    double meanweight, u4, z, tsfr, tcool, egyhot, factorEVP, egyeff, egyeff2;
    FILE *fd;


    meanweight = 4 / (8 - 5 * (1 - HYDROGEN_MASSFRAC));	/* note: assuming FULL ionization */
    u4 = 1 / meanweight * (1.0 / GAMMA_MINUS1) * (BOLTZMANN / PROTONMASS) * 1.0e4;
    u4 *= All.UnitMass_in_g / All.UnitEnergy_in_cgs;

    if(All.ComovingIntegrationOn)
    {
        /* to be guaranteed to get z=0 rate */
        set_global_time(1.0);
        IonizeParams();
    }

    if(ThisTask == 0)
        fd = fopen("eos.txt", "w");
    else
        fd = 0;

    for(rho = All.PhysDensThresh; rho <= 1e7 * All.PhysDensThresh; rho *= 1.2)
    {
        tsfr = sqrt(All.PhysDensThresh / rho) * All.MaxSfrTimescale;

        factorEVP = pow(rho / All.PhysDensThresh, -0.8) * All.FactorEVP;

        egyhot = All.EgySpecSN / (1 + factorEVP) + All.EgySpecCold;

        ne = 1.0;
        tcool = GetCoolingTime(egyhot, rho, &ne, 0.0);

        y = tsfr / tcool * egyhot / (All.FactorSN * All.EgySpecSN - (1 - All.FactorSN) * All.EgySpecCold);
        x = 1 + 1 / (2 * y) - sqrt(1 / y + 1 / (4 * y * y));

        egyeff = egyhot * (1 - x) + All.EgySpecCold * x;

        P = GAMMA_MINUS1 * rho * egyeff;

        if(ThisTask == 0)
        {
            fprintf(fd, "%g %g %g %g %g\n", rho, P, x, tcool, egyhot);
        }
    }

    if(ThisTask == 0)
        fclose(fd);


    if(ThisTask == 0)
        fd = fopen("sfrrate.txt", "w");
    else
        fd = 0;

    for(rho0 = All.PhysDensThresh; rho0 <= 10000 * All.PhysDensThresh; rho0 *= 1.02)
    {
        z = 0;
        rho = rho0;
        q = 0;
        dz = 0.001;

        sigma = sigmasfr = sigma_u4 = 0;

        while(rho > 0.0001 * rho0)
        {
            if(rho > All.PhysDensThresh)
            {
                tsfr = sqrt(All.PhysDensThresh / rho) * All.MaxSfrTimescale;

                factorEVP = pow(rho / All.PhysDensThresh, -0.8) * All.FactorEVP;

                egyhot = All.EgySpecSN / (1 + factorEVP) + All.EgySpecCold;

                ne = 1.0;
                tcool = GetCoolingTime(egyhot, rho, &ne, 0.0);

                y =
                    tsfr / tcool * egyhot / (All.FactorSN * All.EgySpecSN - (1 - All.FactorSN) * All.EgySpecCold);
                x = 1 + 1 / (2 * y) - sqrt(1 / y + 1 / (4 * y * y));

                egyeff = egyhot * (1 - x) + All.EgySpecCold * x;

                P = P1 = GAMMA_MINUS1 * rho * egyeff;

                rho2 = 1.1 * rho;
                tsfr2 = sqrt(All.PhysDensThresh / rho2) * All.MaxSfrTimescale;
                factorEVP2 = pow(rho2 / All.PhysDensThresh, -0.8) * All.FactorEVP;
                egyhot2 = All.EgySpecSN / (1 + factorEVP) + All.EgySpecCold;
                tcool2 = GetCoolingTime(egyhot2, rho2, &ne, 0.0);
                y2 =
                    tsfr2 / tcool2 * egyhot2 / (All.FactorSN * All.EgySpecSN -
                            (1 - All.FactorSN) * All.EgySpecCold);
                x2 = 1 + 1 / (2 * y2) - sqrt(1 / y2 + 1 / (4 * y2 * y2));
                egyeff2 = egyhot2 * (1 - x2) + All.EgySpecCold * x2;
                P2 = GAMMA_MINUS1 * rho2 * egyeff2;

                gam = log(P2 / P1) / log(rho2 / rho);
            }
            else
            {
                tsfr = 0;

                P = GAMMA_MINUS1 * rho * u4;
                gam = 1.0;


                sigma_u4 += rho * dz;
            }



            drho = q;
            dq = -(gam - 2) / rho * q * q - 4 * M_PI * All.G / (gam * P) * rho * rho * rho;

            sigma += rho * dz;
            if(tsfr > 0)
            {
                sigmasfr += (1 - All.FactorSN) * rho * x / tsfr * dz;
            }

            rho += drho * dz;
            q += dq * dz;
        }


        sigma *= 2;		/* to include the other side */
        sigmasfr *= 2;
        sigma_u4 *= 2;


        if(ThisTask == 0)
        {
            fprintf(fd, "%g %g %g %g\n", rho0, sigma, sigmasfr, sigma_u4);
        }
    }


    if(All.ComovingIntegrationOn)
    {
        set_global_time(All.TimeBegin);
        IonizeParams();
    }

    if(ThisTask == 0)
        fclose(fd);
}

void set_units_sfr(void)
{
    for(bits = 0; GENERATIONS > (1 << bits); bits++);

    double meanweight;

#ifdef COSMIC_RAYS
    double feedbackenergyinergs;
#endif

    All.OverDensThresh =
        All.CritOverDensity * All.OmegaBaryon * 3 * All.Hubble * All.Hubble / (8 * M_PI * All.G);

    All.PhysDensThresh = All.CritPhysDensity * PROTONMASS / HYDROGEN_MASSFRAC / All.UnitDensity_in_cgs;

    meanweight = 4 / (1 + 3 * HYDROGEN_MASSFRAC);	/* note: assuming NEUTRAL GAS */

    All.EgySpecCold = 1 / meanweight * (1.0 / GAMMA_MINUS1) * (BOLTZMANN / PROTONMASS) * All.TempClouds;
    All.EgySpecCold *= All.UnitMass_in_g / All.UnitEnergy_in_cgs;

    meanweight = 4 / (8 - 5 * (1 - HYDROGEN_MASSFRAC));	/* note: assuming FULL ionization */

    All.EgySpecSN = 1 / meanweight * (1.0 / GAMMA_MINUS1) * (BOLTZMANN / PROTONMASS) * All.TempSupernova;
    All.EgySpecSN *= All.UnitMass_in_g / All.UnitEnergy_in_cgs;


#ifdef COSMIC_RAYS
    if(All.CR_SNEff < 0.0)
        /* if CR_SNeff < 0.0, then substract CR Feedback energy from thermal
         * feedback energy
         */
    {
        if(ThisTask == 0)
        {
            printf("%g percent of thermal feedback go into Cosmic Rays.\nRemaining ", -100.0 * All.CR_SNEff);
        }

        All.EgySpecSN *= (1.0 + All.CR_SNEff);
        All.CR_SNEff = -All.CR_SNEff / (1.0 + All.CR_SNEff);

    }

    All.FeedbackEnergy = All.FactorSN / (1 - All.FactorSN) * All.EgySpecSN;

    feedbackenergyinergs = All.FeedbackEnergy / All.UnitMass_in_g * (All.UnitEnergy_in_cgs * SOLAR_MASS);

    if(ThisTask == 0)
    {
        printf("Feedback energy per formed solar mass in stars= %g  ergs\n", feedbackenergyinergs);
        printf("OverDensThresh= %g\nPhysDensThresh= %g (internal units)\n", All.OverDensThresh,
                All.PhysDensThresh);
    }
#endif
}

static double evaluate_NH_from_GradRho(MyFloat gradrho[3], double hsml, double rho, double include_h)
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

static double get_sfr_factor_due_to_h2(int i) {
    /*  Krumholz & Gnedin fitting function for f_H2 as a function of local
     *  properties, from gadget-p; we return the enhancement on SFR in this
     *  function */

#ifndef SPH_GRAD_RHO
    /* if SPH_GRAD_RHO is not enabled, disable H2 molecular gas
     * this really shall not happen because begrun will check against the
     * condition.
     * */
    return 1.0;
#else
    double tau_fmol;
    double zoverzsun = P[i].Metallicity/METAL_YIELD;
    tau_fmol = evaluate_NH_from_GradRho(SPHP(i).GradRho,P[i].Hsml,SPHP(i).d.Density,1) * All.cf.a2inv;
    tau_fmol *= (0.1 + zoverzsun);
    if(tau_fmol>0) {
        tau_fmol *= 434.78*All.UnitDensity_in_cgs*All.HubbleParam*All.UnitLength_in_cm;
        double y = 0.756*(1+3.1*pow(zoverzsun,0.365));
        y = log(1+0.6*y+0.01*y*y)/(0.6*tau_fmol);
        y = 1-0.75*y/(1+0.25*y);
        if(y<0) y=0; if(y>1) y=1;
        return y;

    } // if(tau_fmol>0)
    return 1.0;
#endif
}
static double get_sfr_factor_due_to_selfgravity(int i) {
#ifdef SPH_GRAD_RHO
    double divv = SPHP(i).v.DivVel * All.cf.a2inv; 
    if(All.ComovingIntegrationOn) {
        divv += 3.0*All.cf.hubble_a2; // hubble-flow correction
    }

    if(All.StarformationCriterion & SFR_CRITERION_CONVERGENT_FLOW) {
        if( divv>=0 ) return 0; // restrict to convergent flows (optional) //
    }

    double dv2abs = (divv*divv 
            + (SPHP(i).r.CurlVel*All.cf.a2inv)
            * (SPHP(i).r.CurlVel*All.cf.a2inv)
           ); // all in physical units
    double alpha_vir = 0.2387 * dv2abs/(All.G * SPHP(i).d.Density*All.cf.a3inv);

    double y = 1.0;
    if(All.ComovingIntegrationOn)
    {
        if((alpha_vir < 1.0) 
        || (SPHP(i).d.Density * All.cf.a3inv > 100. * All.PhysDensThresh)
        )  {
            y = 66.7;
        } else {
            y = 0.1;
        }
        // PFH: note the latter flag is an arbitrary choice currently set 
        // -by hand- to prevent runaway densities from this prescription! //
    } else {
        if(alpha_vir < 1.0) {
            y = 66.7;
        } else {
            y = 0.1;
        }
    }
    if (All.StarformationCriterion & SFR_CRITERION_CONTINUOUS_CUTOFF) {
        // continuous cutoff w alpha_vir instead of sharp (optional) //
        y *= 1.0/(1.0 + alpha_vir); 
    }
    return y;
#else
    return 1.0;
#endif
}

#endif /* closes COOLING */

