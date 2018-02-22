/**************
 * This file is based on the Gadget-3 cooling.c.
 *
 * The routines for the atomic rates are identical. 
 *
 * Otherwise, this version by Yu Feng:
 *  - implements additional support for fluctuating UV.
 *  - is heavily refactored to make it reentrance friendly.
 *
 * There is no public version of original cooling.c freely available.
 *
 * Only reference is 2004 version by Brian O'Shea in ENZO repo,
 * which contains a deferred freedom claimer depending on the action
 * of Volker Springel.
 *
 * The LICENSE of this file is therefore still in limbo.
 *
 ************** */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <bigfile.h>


#include "utils.h"

#include "allvars.h"
#include "partmanager.h"
#include "cooling.h"

#define NCOOLTAB  2000

#define SMALLNUM 1.0e-60
#define COOLLIM  0.1
#define HEATLIM	 20.0

struct abundance {
    double ne;
    double nH0;
    double nHp;
    double nHe0;
    double nHep;
    double nHepp;
};

struct rates {
    double aHp;
    double aHep;
    double aHepp;
    double ad;
    double geH0;
    double geHe0;
    double geHep;
    double bH0;
    double bHep;
    double bff;
};

struct {
    int NRedshift_bins;
    double * Redshift_bins;

    int NHydrogenNumberDensity_bins;
    double * HydrogenNumberDensity_bins;

    int NTemperature_bins;
    double * Temperature_bins;

    double * Lmet_table; /* metal cooling @ one solar metalicity*/

    Interp interp;
} MC;

static void find_abundances_and_rates(double logT, double nHcgs, struct UVBG * uvbg, struct abundance * y, struct rates * r);
static double solve_equilibrium_temp(double u, double nHcgs, struct UVBG * uvbg, struct abundance * y);
static double * h5readdouble(char * filename, char * dataset, int * Nread);

double PrimordialCoolingRate(double logT, double nHcgs, struct UVBG * uvbg, double *nelec);
double CoolingRateFromU(double u, double nHcgs, struct UVBG * uvbg, double *ne_guess, double Z);
static void IonizeParamsTable(void);
static void InitMetalCooling();
static double TableMetalCoolingRate(double redshift, double logT, double lognH);

static double XH = HYDROGEN_MASSFRAC;	/* hydrogen abundance by mass */
static double yhelium;

#define eV_to_K   11606.0
#define eV_to_erg 1.60184e-12


static int CoolingNoMetal;
static int CoolingNoPrimordial;


static double mhboltz;		/* hydrogen mass over Boltzmann constant */
static double ethmin;		/* minimum internal energy for neutral gas */

static double Tmin = 0.0;	/* in log10 */
static double Tmax = 9.0;
static double deltaT;

/* These tables are readonly after initialized */
static double *BetaH0, *BetaHep, *Betaff;
static double *AlphaHp, *AlphaHep, *Alphad, *AlphaHepp;
static double *GammaeH0, *GammaeHe0, *GammaeHep;

struct UVBG GlobalUVBG = {0};

/* This function modifies the photoheating rates by
 * a density dependent factor.
 * This is a hack to attempt to account for helium reionisation,
 * especially for the Lyman alpha forest.
 * It is not a good model for helium reionisation, and needs to be replaced!
 * Takes hydrogen number density in cgs units.
 */
double he_reion_factor(double nHcgs)
{
  const double rhoc = 3.0 * pow(All.CP.HubbleParam*HUBBLE,2.0) /(8.0*M_PI*GRAVITY);
  const double rho = PROTONMASS * nHcgs / XH;
  const double overden = rho/(All.CP.OmegaBaryon * rhoc * pow(All.Time,-3.0));
  if (overden >= All.HeliumHeatThresh)
      return All.HeliumHeatAmp*pow(All.HeliumHeatThresh, All.HeliumHeatExp);
  else
      return All.HeliumHeatAmp*pow(overden, All.HeliumHeatExp);
}

/* returns new internal energy per unit mass. 
 * Arguments are passed in code units, density is proper density.
 */
double DoCooling(double u_old, double rho, double dt, struct UVBG * uvbg, double *ne_guess, double Z)
{
    if(CoolingNoPrimordial) return 0;

    double u, du;
    double u_lower, u_upper;
    double ratefact;
    double LambdaNet;
    int iter = 0;

    rho *= All.UnitDensity_in_cgs * All.CP.HubbleParam * All.CP.HubbleParam;	/* convert to physical cgs units */
    u_old *= All.UnitPressure_in_cgs / All.UnitDensity_in_cgs;
    dt *= All.UnitTime_in_s / All.CP.HubbleParam;

    double nHcgs = XH * rho / PROTONMASS;	/* hydrogen number dens in cgs units */
    ratefact = nHcgs * nHcgs / rho;

    u = u_old;
    u_lower = u;
    u_upper = u;

    LambdaNet = CoolingRateFromU(u, nHcgs, uvbg, ne_guess, Z);

    /* bracketing */

    if(u - u_old - ratefact * LambdaNet * dt < 0)	/* heating */
    {
        u_upper *= sqrt(1.1);
        u_lower /= sqrt(1.1);
            while(u_upper - u_old - ratefact * CoolingRateFromU(u_upper, nHcgs, uvbg, ne_guess, Z) * dt < 0)
            {
                u_upper *= 1.1;
                u_lower *= 1.1;
            }

    }

    if(u - u_old - ratefact * LambdaNet * dt > 0)
    {
        u_lower /= sqrt(1.1);
        u_upper *= sqrt(1.1);
            while(u_lower - u_old - ratefact * CoolingRateFromU(u_lower, nHcgs, uvbg, ne_guess, Z) * dt > 0)
            {
                u_upper /= 1.1;
                u_lower /= 1.1;
            }
    }

    do
    {
        u = 0.5 * (u_lower + u_upper);

        LambdaNet = CoolingRateFromU(u, nHcgs, uvbg, ne_guess, Z);

        if(u - u_old - ratefact * LambdaNet * dt > 0)
        {
            u_upper = u;
        }
        else
        {
            u_lower = u;
        }

        du = u_upper - u_lower;

        iter++;

        if(iter >= (MAXITER - 10))
            message(1, "u= %g\n", u);
    }
    while(fabs(du / u) > 1.0e-6 && iter < MAXITER);

    if(iter >= MAXITER)
    {
        endrun(10, "failed to converge in DoCooling()\n");
    }

    u *= All.UnitDensity_in_cgs / All.UnitPressure_in_cgs;	/* to internal units */

    return u;
}



/* returns cooling time. 
 * NOTE: If we actually have heating, a cooling time of 0 is returned.
 */
double GetCoolingTime(double u_old, double rho, struct UVBG * uvbg, double *ne_guess, double Z)
{
    if(CoolingNoPrimordial) return 0;

    double u;
    double ratefact;
    double LambdaNet, coolingtime;

    rho *= All.UnitDensity_in_cgs * All.CP.HubbleParam * All.CP.HubbleParam;	/* convert to physical cgs units */
    u_old *= All.UnitPressure_in_cgs / All.UnitDensity_in_cgs;


    double nHcgs = XH * rho / PROTONMASS;	/* hydrogen number dens in cgs units */
    ratefact = nHcgs * nHcgs / rho;

    u = u_old;

    LambdaNet = CoolingRateFromU(u, nHcgs, uvbg, ne_guess, Z);

    /* bracketing */

    if(LambdaNet >= 0)		/* ups, we have actually heating due to UV background */
        return 0;

    coolingtime = u_old / (-ratefact * LambdaNet);

    coolingtime *= All.CP.HubbleParam / All.UnitTime_in_s;

    return coolingtime;
}




void cool_test(void)
{
    double uin, rhoin, muin, nein;

    //tempin = 34.0025;
    uin = 6.01329e+09;
    rhoin = 7.85767e-29;
    muin = 0.691955;

    nein = (1 + 4 * yhelium) / muin - (1 + yhelium);
    struct abundance y;
    y.ne = nein;
    double nHcgs = rhoin * XH / PROTONMASS;
    message(0, "%g\n", solve_equilibrium_temp(uin, nHcgs, &GlobalUVBG, &y));
}

/* this function determines the electron fraction, and hence the mean 
 * molecular weight. With it arrives at a self-consistent temperature.
 * Element abundances and the rates for the emission are also computed
 */
static double solve_equilibrium_temp(double u, double nHcgs, struct UVBG * uvbg, struct abundance * y)
{
    double temp, temp_old, temp_new, max = 0, ne_old;
    double mu;
    struct rates r;
    int iter = 0;
/*     double u_input, rho_input, ne_input; */

    mu = (1 + 4 * yhelium) / (1 + yhelium + y->ne);
    temp = GAMMA_MINUS1 / BOLTZMANN * u * PROTONMASS * mu;

    do
    {
        ne_old = y->ne;

        find_abundances_and_rates(log10(temp), nHcgs, uvbg, y, &r);

        temp_old = temp;

        mu = (1 + 4 * yhelium) / (1 + yhelium + y->ne);

        temp_new = GAMMA_MINUS1 / BOLTZMANN * u * PROTONMASS * mu;

        max =
            DMAX(max,
                    temp_new / (1 + yhelium + y->ne) * fabs((y->ne - ne_old) / (temp_new - temp_old + 1.0)));

        temp = temp_old + (temp_new - temp_old) / (1 + max);
        iter++;

        if(iter > (MAXITER - 10))
            message(1, "-> temp= %g ne=%g\n", temp, y->ne);
    }
    while(fabs(temp - temp_old) > 1.0e-3 * temp && iter < MAXITER);

    if(iter >= MAXITER)
        {
            endrun(12, "failed to converge in solve_equilibrium_temp()\n");
        }

    return temp;
}


/* this function computes the actual abundance ratios 
*/
static void find_abundances_and_rates(double logT, double nHcgs, struct UVBG * uvbg, struct abundance * y, struct rates * r)
{
    double neold, nenew;
    int j, niter;
    double flow, fhi, t;

/*     double logT_input, rho_input, ne_input; */

    if(logT <= Tmin)		/* everything neutral */
    {
        memset(r, 0, sizeof(struct rates));
        y->nH0 = 1.0;
        y->nHe0 = yhelium;
        y->nHp = 0;
        y->nHep = 0;
        y->nHepp = 0;
        y->ne = 0;
        return;
    }

    if(logT >= Tmax)		/* everything is ionized */
    {
        memset(r, 0, sizeof(struct rates));
        y->nH0 = 0;
        y->nHe0 = 0;
        y->nHp = 1.0;
        y->nHep = 0;
        y->nHepp = yhelium;
        y->ne = y->nHp + 2.0 * y->nHepp; /* note: in units of the hydrogen number density */
        return;
    }

    t = (logT - Tmin) / deltaT;
    j = (int) t;
    fhi = t - j;
    flow = 1 - fhi;

    if(y->ne== 0)
        y->ne = 1.0;

    niter = 0;
    double necgs = y->ne * nHcgs;

    /* evaluate number densities iteratively (cf KWH eqns 33-38) in units of nH */
    do
    {
        double gJH0ne, gJHe0ne, gJHepne;
        niter++;

        r->aHp = flow * AlphaHp[j] + fhi * AlphaHp[j + 1];
        r->aHep = flow * AlphaHep[j] + fhi * AlphaHep[j + 1];
        r->aHepp = flow * AlphaHepp[j] + fhi * AlphaHepp[j + 1];
        r->ad = flow * Alphad[j] + fhi * Alphad[j + 1];
        r->geH0 = flow * GammaeH0[j] + fhi * GammaeH0[j + 1];
        r->geHe0 = flow * GammaeHe0[j] + fhi * GammaeHe0[j + 1];
        r->geHep = flow * GammaeHep[j] + fhi * GammaeHep[j + 1];

        if(necgs <= 1.e-25 || uvbg->J_UV == 0)
        {
            gJH0ne = gJHe0ne = gJHepne = 0;
        }
        else
        {
            gJH0ne = uvbg->gJH0 / necgs;
            gJHe0ne = uvbg->gJHe0 / necgs;
            gJHepne = uvbg->gJHep / necgs;
        }

        y->nH0 = r->aHp / (r->aHp + r->geH0 + gJH0ne);	/* eqn (33) */
        y->nHp = 1.0 - y->nH0;		/* eqn (34) */

        if((gJHe0ne + r->geHe0) <= SMALLNUM)	/* no ionization at all */
        {
            y->nHep = 0.0;
            y->nHepp = 0.0;
            y->nHe0 = yhelium;
        }
        else
        {
            y->nHep = yhelium / (1.0 + (r->aHep + r->ad) / (r->geHe0 + gJHe0ne) + (r->geHep + gJHepne) / r->aHepp);	/* eqn (35) */
            y->nHe0 = y->nHep * (r->aHep + r->ad) / (r->geHe0 + gJHe0ne);	/* eqn (36) */
            y->nHepp = y->nHep * (r->geHep + gJHepne) / r->aHepp;	/* eqn (37) */
        }

        neold = y->ne;

        y->ne = y->nHp + y->nHep + 2 * y->nHepp;	/* eqn (38) */

        if(uvbg->J_UV == 0)
            break;

        nenew = 0.5 * (y->ne + neold);
        y->ne = nenew;
        necgs = y->ne * nHcgs;

        if(fabs(y->ne - neold) < 1.0e-4)
            break;

        if(niter > (MAXITER - 10))
            message(1, "ne= %g  niter=%d\n", y->ne, niter);
    }
    while(niter < MAXITER);

    if(niter >= MAXITER)
    {
        endrun(13, "no convergence reached in find_abundances_and_rates()\n");
    }

    r->bH0 = flow * BetaH0[j] + fhi * BetaH0[j + 1];
    r->bHep = flow * BetaHep[j] + fhi * BetaHep[j + 1];
    r->bff = flow * Betaff[j] + fhi * Betaff[j + 1];
}



/*  this function first computes the self-consistent temperature
 *  and abundance ratios, and then it calculates 
 *  (heating rate-cooling rate)/n_h^2 in cgs units 
 */
double CoolingRateFromU(double u, double nHcgs, struct UVBG * uvbg, double *ne_guess, double Z)
{
    if(CoolingNoPrimordial) return 0;

    double temp;
    struct abundance y;

    y.ne = *ne_guess;
    temp = solve_equilibrium_temp(u, nHcgs, uvbg, &y);
    *ne_guess = y.ne;
    double logT = log10(temp);
    double redshift = 1 / All.cf.a -  1.;
    double LambdaNet = PrimordialCoolingRate(logT, nHcgs, uvbg, ne_guess);
    if(! CoolingNoMetal) {
        double lognH = log10(nHcgs);
        LambdaNet -= Z * TableMetalCoolingRate(redshift, logT, lognH);
    }
    return LambdaNet;
}


/*  this function computes the self-consistent temperature
 *  and abundance ratios 
 */
double AbundanceRatios(double u, double rho, struct UVBG * uvbg, double *ne_guess, double *nH0_pointer, double *nHeII_pointer)
{
    if(CoolingNoPrimordial) return 0;

    double temp;
    struct abundance y;

    rho *= All.UnitDensity_in_cgs * All.CP.HubbleParam * All.CP.HubbleParam;	/* convert to physical cgs units */
    u *= All.UnitPressure_in_cgs / All.UnitDensity_in_cgs;

    double nHcgs = rho / PROTONMASS * XH;
    y.ne = *ne_guess;
    temp = solve_equilibrium_temp(u, nHcgs, uvbg, &y);
    *ne_guess = y.ne;

    *nH0_pointer = y.nH0;
    *nHeII_pointer = y.nHep;

    return temp;
}

double ConvertInternalEnergy2Temperature(double u, double ne)
{
    if(CoolingNoPrimordial) return 0;

    double mu;
    double temp;
    mu = (1 + 4 * yhelium) / (1 + yhelium + ne);

    u *= All.UnitPressure_in_cgs / All.UnitDensity_in_cgs;
    temp = GAMMA_MINUS1 / BOLTZMANN * u * PROTONMASS * mu;
    return temp;
}

extern FILE *fd;

/*  Calculates (heating rate-cooling rate)/n_h^2 in cgs units 
*/
double PrimordialCoolingRate(double logT, double nHcgs, struct UVBG * uvbg, double *nelec)
{
    double Lambda, Heat;
    double redshift;
    double T;
    struct abundance y;
    struct rates r;

    if(logT <= Tmin)
        logT = Tmin + 0.5 * deltaT;	/* floor at Tmin */


    y.ne = *nelec;

    if(logT < Tmax)
    {
        double LambdaExc, LambdaIon, LambdaRec, LambdaFF;
        double LambdaExcH0, LambdaExcHep, LambdaIonH0, LambdaIonHe0, LambdaIonHep;
        double LambdaRecHp, LambdaRecHep, LambdaRecHepp, LambdaRecHepd;
        find_abundances_and_rates(logT, nHcgs, uvbg, &y, &r);
        *nelec = y.ne;
        /* Compute cooling and heating rate (cf KWH Table 1) in units of nH**2 */
        T = pow(10.0, logT);

        LambdaExcH0 = r.bH0 * y.ne * y.nH0;
        LambdaExcHep = r.bHep * y.ne * y.nHep;
        LambdaExc = LambdaExcH0 + LambdaExcHep;	/* excitation */

        LambdaIonH0 = 2.18e-11 * r.geH0 * y.ne * y.nH0;
        LambdaIonHe0 = 3.94e-11 * r.geHe0 * y.ne * y.nHe0;
        LambdaIonHep = 8.72e-11 * r.geHep * y.ne * y.nHep;
        LambdaIon = LambdaIonH0 + LambdaIonHe0 + LambdaIonHep;	/* ionization */

        LambdaRecHp = 1.036e-16 * T * y.ne * (r.aHp * y.nHp);
        LambdaRecHep = 1.036e-16 * T * y.ne * (r.aHep * y.nHep);
        LambdaRecHepp = 1.036e-16 * T * y.ne * (r.aHepp * y.nHepp);
        LambdaRecHepd = 6.526e-11 * r.ad * y.ne * y.nHep;
        LambdaRec = LambdaRecHp + LambdaRecHep + LambdaRecHepp + LambdaRecHepd;

        LambdaFF = r.bff * (y.nHp + y.nHep + 4 * y.nHepp) * y.ne;

        Lambda = LambdaExc + LambdaIon + LambdaRec + LambdaFF;

        redshift = 1 / All.cf.a - 1;
        double LambdaCmptn = 5.65e-36 * y.ne * (T - 2.73 * (1. + redshift)) * pow(1. + redshift, 4.) / nHcgs;

        Lambda += LambdaCmptn;

        Heat = 0;
        double hefac=1.;
        if(All.HeliumHeatOn)
            hefac = he_reion_factor(nHcgs);
        if(uvbg->J_UV != 0)
            Heat += hefac * (y.nH0 * uvbg->epsH0 + y.nHe0 * uvbg->epsHe0 + y.nHep * uvbg->epsHep) / nHcgs;

    }
    else				/* here we're outside of tabulated rates, T>Tmax K */
    {
        /* at high T (fully ionized); only free-free and Compton cooling are present.  
           Assumes no heating. */
        double LambdaFF, LambdaCmptn;
        Heat = 0;

        /* very hot: H and He both fully ionized */
        y.nHp = 1.0;
        y.nHep = 0;
        y.nHepp = yhelium;
        y.ne = y.nHp + 2.0 * y.nHepp;
        *nelec = y.ne;		/* note: in units of the hydrogen number density */

        T = pow(10.0, logT);
        LambdaFF =
            1.42e-27 * sqrt(T) * (1.1 + 0.34 * exp(-(5.5 - logT) * (5.5 - logT) / 3)) * (y.nHp + 4 * y.nHepp) * y.ne;

        redshift = 1 / All.cf.a - 1;
        /* add inverse Compton cooling off the microwave background */
        LambdaCmptn = 5.65e-36 * y.ne * (T - 2.73 * (1. + redshift)) * pow(1. + redshift, 4.) / nHcgs;

        Lambda = LambdaFF + LambdaCmptn;
    }

    return (Heat - Lambda);
}

static void InitUVF(void);
void InitCoolMemory(void)
{
    BetaH0 = (double *) mymalloc("BetaH0", (NCOOLTAB + 1) * sizeof(double));
    BetaHep = (double *) mymalloc("BetaHep", (NCOOLTAB + 1) * sizeof(double));
    AlphaHp = (double *) mymalloc("AlphaHp", (NCOOLTAB + 1) * sizeof(double));
    AlphaHep = (double *) mymalloc("AlphaHep", (NCOOLTAB + 1) * sizeof(double));
    Alphad = (double *) mymalloc("Alphad", (NCOOLTAB + 1) * sizeof(double));
    AlphaHepp = (double *) mymalloc("AlphaHepp", (NCOOLTAB + 1) * sizeof(double));
    GammaeH0 = (double *) mymalloc("GammaeH0", (NCOOLTAB + 1) * sizeof(double));
    GammaeHe0 = (double *) mymalloc("GammaeHe0", (NCOOLTAB + 1) * sizeof(double));
    GammaeHep = (double *) mymalloc("GammaeHep", (NCOOLTAB + 1) * sizeof(double));
    Betaff = (double *) mymalloc("Betaff", (NCOOLTAB + 1) * sizeof(double));
}


void MakeCoolingTable(void)
    /* Set up interpolation tables in T for cooling rates given in KWH, ApJS, 105, 19 
       Hydrogen, Helium III recombination rates and collisional ionization cross-sections are updated */
{
    int i;
    double T;
    double Tfact;

#ifdef NEW_RATES
    double dE, P, A, X, K, U, T_eV;
    double b0, b1, b2, b3, b4, b5, c0, c1, c2, c3, c4, c5, y;	/* used in Scholz-Walter fit */
    double E1s_2, Gamma1s_2s, Gamma1s_2p;
#endif

    XH = 0.76;
    yhelium = (1 - XH) / (4 * XH);

    mhboltz = PROTONMASS / BOLTZMANN;

    if(All.MinGasTemp > 0.0)
        Tmin = log10(0.1 * All.MinGasTemp);
    else
        Tmin = 1.0;

    deltaT = (Tmax - Tmin) / NCOOLTAB;

    ethmin = pow(10.0, Tmin) * (1. + yhelium) / ((1. + 4. * yhelium) * mhboltz * GAMMA_MINUS1);
    /* minimum internal energy for neutral gas */

    for(i = 0; i <= NCOOLTAB; i++)
    {
        BetaH0[i] =
            BetaHep[i] =
            Betaff[i] =
            AlphaHp[i] = AlphaHep[i] = AlphaHepp[i] = Alphad[i] = GammaeH0[i] = GammaeHe0[i] = GammaeHep[i] = 0;


        T = pow(10.0, Tmin + deltaT * i);

        Tfact = 1.0 / (1 + sqrt(T / 1.0e5));

        if(118348 / T < 70)
            BetaH0[i] = 7.5e-19 * exp(-118348 / T) * Tfact;

#ifdef NEW_RATES
        /* Scholtz-Walters 91 fit */
        if(T >= 2.0e3 && T < 1e8)
        {

            if(T >= 2.0e3 && T < 6.0e4)
            {
                b0 = -3.299613e1;
                b1 = 1.858848e1;
                b2 = -6.052265;
                b3 = 8.603783e-1;
                b4 = -5.717760e-2;
                b5 = 1.451330e-3;

                c0 = -1.630155e2;
                c1 = 8.795711e1;
                c2 = -2.057117e1;
                c3 = 2.359573;
                c4 = -1.339059e-1;
                c5 = 3.021507e-3;
            }
            else
            {
                if(T >= 6.0e4 && T < 6.0e6)
                {
                    b0 = 2.869759e2;
                    b1 = -1.077956e2;
                    b2 = 1.524107e1;
                    b3 = -1.080538;
                    b4 = 3.836975e-2;
                    b5 = -5.467273e-4;

                    c0 = 5.279996e2;
                    c1 = -1.939399e2;
                    c2 = 2.718982e1;
                    c3 = -1.883399;
                    c4 = 6.462462e-2;
                    c5 = -8.811076e-4;
                }
                else
                {
                    b0 = -2.7604708e3;
                    b1 = 7.9339351e2;
                    b2 = -9.1198462e1;
                    b3 = 5.1993362;
                    b4 = -1.4685343e-1;
                    b5 = 1.6404093e-3;

                    c0 = -2.8133632e3;
                    c1 = 8.1509685e2;
                    c2 = -9.4418414e1;
                    c3 = 5.4280565;
                    c4 = -1.5467120e-1;
                    c5 = 1.7439112e-3;
                }

                y = log(T);
                E1s_2 = 10.2;	/* eV */

                Gamma1s_2s =
                    exp(b0 + b1 * y + b2 * y * y + b3 * y * y * y + b4 * y * y * y * y + b5 * y * y * y * y * y);
                Gamma1s_2p =
                    exp(c0 + c1 * y + c2 * y * y + c3 * y * y * y + c4 * y * y * y * y + c5 * y * y * y * y * y);

                T_eV = T / eV_to_K;

                BetaH0[i] = E1s_2 * eV_to_erg * (Gamma1s_2s + Gamma1s_2p) * exp(-E1s_2 / T_eV);
            }
        }
#endif


        if(473638 / T < 70)
            BetaHep[i] = 5.54e-17 * pow(T, -0.397) * exp(-473638 / T) * Tfact;

        Betaff[i] = 1.43e-27 * sqrt(T) * (1.1 + 0.34 * exp(-(5.5 - log10(T)) * (5.5 - log10(T)) / 3));


#ifdef NEW_RATES
        AlphaHp[i] = 6.28e-11 * pow(T / 1000, -0.2) / (1. + pow(T / 1.0e6, 0.7)) / sqrt(T);
#else
        AlphaHp[i] = 8.4e-11 * pow(T / 1000, -0.2) / (1. + pow(T / 1.0e6, 0.7)) / sqrt(T);	/* old Cen92 fit */
#endif


        AlphaHep[i] = 1.5e-10 * pow(T, -0.6353);


#ifdef NEW_RATES
        AlphaHepp[i] = 3.36e-10 * pow(T / 1000, -0.2) / (1. + pow(T / 4.0e6, 0.7)) / sqrt(T);
#else
        AlphaHepp[i] = 4. * AlphaHp[i];	/* old Cen92 fit */
#endif

        if(470000 / T < 70)
            Alphad[i] = 1.9e-3 * pow(T, -1.5) * exp(-470000 / T) * (1. + 0.3 * exp(-94000 / T));


#ifdef NEW_RATES
        T_eV = T / eV_to_K;

        /* Voronov 97 fit */
        /* hydrogen */
        dE = 13.6;
        P = 0.0;
        A = 0.291e-7;
        X = 0.232;
        K = 0.39;

        U = dE / T_eV;
        GammaeH0[i] = A * (1.0 + P * sqrt(U)) * pow(U, K) * exp(-U) / (X + U);

        /* Helium */
        dE = 24.6;
        P = 0.0;
        A = 0.175e-7;
        X = 0.18;
        K = 0.35;

        U = dE / T_eV;
        GammaeHe0[i] = A * (1.0 + P * sqrt(U)) * pow(U, K) * exp(-U) / (X + U);

        /* Hellium II */
        dE = 54.4;
        P = 1.0;
        A = 0.205e-8;
        X = 0.265;
        K = 0.25;

        U = dE / T_eV;
        GammaeHep[i] = A * (1.0 + P * sqrt(U)) * pow(U, K) * exp(-U) / (X + U);

#else
        if(157809.1 / T < 70)
            GammaeH0[i] = 5.85e-11 * sqrt(T) * exp(-157809.1 / T) * Tfact;

        if(285335.4 / T < 70)
            GammaeHe0[i] = 2.38e-11 * sqrt(T) * exp(-285335.4 / T) * Tfact;

        if(631515.0 / T < 70)
            GammaeHep[i] = 5.68e-12 * sqrt(T) * exp(-631515.0 / T) * Tfact;
#endif

    }


}

static double * h5readdouble(char * filename, char * dataset, int * Nread) {
    void * buffer;
    int N;
    if(ThisTask == 0) {
        BigFile bf[1];
        big_file_open(bf, filename);
        BigBlock bb[1];
        if(0 != big_file_open_block(bf, bb, dataset)) {
            endrun(-1, "Cannot open %s %s\n", filename, dataset);
        }

        N = bb->size;
        MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

        BigArray array[1];

        big_block_read_simple(bb, 0, N, array, "f8");
        /* steal the buffer */
        buffer = array->data;
        big_block_close(bb);
        big_file_close(bf);
    } else {
        MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
        buffer = malloc(N * sizeof(double));
    }

    MPI_Bcast(buffer, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    *Nread = N;
    return buffer;
}



/* table input (from file TREECOOL) for ionizing parameters */

#define JAMPL	1.0		/* amplitude factor relative to input table */
#define TABLESIZE 500		/* Max # of lines in TREECOOL */

static float inlogz[TABLESIZE];
static float gH0[TABLESIZE], gHe[TABLESIZE], gHep[TABLESIZE];
static float eH0[TABLESIZE], eHe[TABLESIZE], eHep[TABLESIZE];
static int nheattab;		/* length of table */

void ReadIonizeParams(char *fname)
{
    int i;
    FILE *fdcool;

    if(!(fdcool = fopen(fname, "r")))
    {
        endrun(456, " Cannot read ionization table in file `%s'\n", fname);
    }

    for(i = 0; i < TABLESIZE; i++)
        gH0[i] = 0;

    for(i = 0; i < TABLESIZE; i++)
        if(fscanf(fdcool, "%g %g %g %g %g %g %g",
                    &inlogz[i], &gH0[i], &gHe[i], &gHep[i], &eH0[i], &eHe[i], &eHep[i]) == EOF)
            break;

    fclose(fdcool);

    /*  nheattab is the number of entries in the table */

    for(i = 0, nheattab = 0; i < TABLESIZE; i++)
        if(gH0[i] != 0.0)
            nheattab++;
        else
            break;

    message(0, "Read ionization table for z=%g - %g, with %d rows in file `%s'.\n", pow(10,inlogz[0])-1,pow(10,inlogz[nheattab-1])-1,nheattab, fname);
}


void IonizeParams(void)
{
    if(CoolingNoPrimordial) return;
    IonizeParamsTable();
}



static void IonizeParamsTable(void)
{
    int i, ilow;
    double logz, dzlow, dzhi;
    double redshift;

    redshift = 1 / All.cf.a - 1;

    logz = log10(redshift + 1.0);
    ilow = 0;
    for(i = 0; i < nheattab; i++)
    {
        if(inlogz[i] < logz)
            ilow = i;
        else
            break;
    }

    dzlow = logz - inlogz[ilow];
    dzhi = inlogz[ilow + 1] - logz;

    if(logz > inlogz[nheattab - 1] || gH0[ilow] == 0 || gH0[ilow + 1] == 0 || nheattab == 0)
    {
        memset(&GlobalUVBG, 0, sizeof(GlobalUVBG));
        return;
    }
    else
        GlobalUVBG.J_UV = 1.e-21;		/* irrelevant as long as it's not 0 */

    GlobalUVBG.gJH0 = JAMPL * pow(10., (dzhi * log10(gH0[ilow]) + dzlow * log10(gH0[ilow + 1])) / (dzlow + dzhi));
    GlobalUVBG.gJHe0 = JAMPL * pow(10., (dzhi * log10(gHe[ilow]) + dzlow * log10(gHe[ilow + 1])) / (dzlow + dzhi));
    GlobalUVBG.gJHep = JAMPL * pow(10., (dzhi * log10(gHep[ilow]) + dzlow * log10(gHep[ilow + 1])) / (dzlow + dzhi));
    GlobalUVBG.epsH0 = JAMPL * pow(10., (dzhi * log10(eH0[ilow]) + dzlow * log10(eH0[ilow + 1])) / (dzlow + dzhi));
    GlobalUVBG.epsHe0 = JAMPL * pow(10., (dzhi * log10(eHe[ilow]) + dzlow * log10(eHe[ilow + 1])) / (dzlow + dzhi));
    GlobalUVBG.epsHep = JAMPL * pow(10., (dzhi * log10(eHep[ilow]) + dzlow * log10(eHep[ilow + 1])) / (dzlow + dzhi));

    return;
}


void SetZeroIonization(void)
{
    memset(&GlobalUVBG, 0, sizeof(GlobalUVBG));
}

void InitCool(void)
{
    /* The table will be initialized to z=0 */
    if(!All.CoolingOn) {
        CoolingNoPrimordial = 1;
        CoolingNoMetal = 1;
        return;
    }

    InitCoolMemory();
    MakeCoolingTable();
    if(strlen(All.TreeCoolFile) == 0) {
        CoolingNoPrimordial = 1;
        message(0, "No TreeCool file is provided. Cooling is broken. OK for DM only runs. \n");
    } else {
        CoolingNoPrimordial = 0;
        ReadIonizeParams(All.TreeCoolFile);
    }
    /* now initialize the metal cooling table from cloudy; we got this file
     * from vogelsberger's Arepo simulations; it is supposed to be 
     * cloudy + UVB - H and He; look so.
     * the table contains only 1 Z_sun values. Need to be scaled to the 
     * metallicity.
     *
     * */
            /* let's see if the Metal Cool File is magic NoMetal */ 
    if(strlen(All.MetalCoolFile) == 0) {
        CoolingNoMetal = 1;
    } else {
        CoolingNoMetal = 0;
        InitMetalCooling();
    }

    InitUVF();
}

static void InitMetalCooling() {
    int size;
    //This is never used if All.MetalCoolFile == ""
    double * tabbedmet = h5readdouble(All.MetalCoolFile, "MetallicityInSolar_bins", &size);

    if(size != 1 || tabbedmet[0] != 0.0) {
        endrun(123, "MetalCool file %s is wrongly tabulated\n", All.MetalCoolFile);
    }
    free(tabbedmet);
    
    MC.Redshift_bins = h5readdouble(All.MetalCoolFile, "Redshift_bins", &MC.NRedshift_bins);
    MC.HydrogenNumberDensity_bins = h5readdouble(All.MetalCoolFile, "HydrogenNumberDensity_bins", &MC.NHydrogenNumberDensity_bins);
    MC.Temperature_bins = h5readdouble(All.MetalCoolFile, "Temperature_bins", &MC.NTemperature_bins);
    MC.Lmet_table = h5readdouble(All.MetalCoolFile, "NetCoolingRate", &size);

    int dims[] = {MC.NRedshift_bins, MC.NHydrogenNumberDensity_bins, MC.NTemperature_bins};

    interp_init(&MC.interp, 3, dims);
    interp_init_dim(&MC.interp, 0, MC.Redshift_bins[0], MC.Redshift_bins[MC.NRedshift_bins - 1]);
    interp_init_dim(&MC.interp, 1, MC.HydrogenNumberDensity_bins[0], 
                    MC.HydrogenNumberDensity_bins[MC.NHydrogenNumberDensity_bins - 1]);
    interp_init_dim(&MC.interp, 2, MC.Temperature_bins[0], 
                    MC.Temperature_bins[MC.NTemperature_bins - 1]);
}

static double TableMetalCoolingRate(double redshift, double logT, double lognH) {
    double x[] = {redshift, lognH, logT};
    int status[3];
    double rate = interp_eval(&MC.interp, x, MC.Lmet_table, status);
    /* XXX: in case of very hot / very dense we just use whatever the table says at
     * the limit. should be OK. */
    return rate;
}

static struct {
    int disabled;
    Interp interp;
    Interp Finterp;
    double * Table;
    ptrdiff_t Nside;
    double * Fraction;
    double * Zbins;
    int N_Zbins;
} UVF;

static void InitUVF(void) {
    /* The UV fluctation file is a bigfile with these tables:
     * ReionizedFraction: values of the reionized fraction as function of
     * redshift.
     * Redshift_Bins: uniform redshifts of the reionized fraction values 
     *
     * XYZ_Bins: the uniform XYZ points where Z_reion is tabulated. (length of Nside)
     *
     * Zreion_Table: a Nside (X) x Nside (Y)x Nside (z) C ordering double array,
     * the reionization redshift as function of space, on a grid give by
     * XYZ_Bins.
     *
     * Notice that this table is broadcast to all MPI ranks, thus it can't be
     * too big. (400x400x400 is around 400 MBytes)
     *
     * */
    if(strlen(All.UVFluctuationFile) == 0) {
        message(0, "Using UNIFORM UV BG from %s\n", All.TreeCoolFile);
        UVF.disabled = 1;
        return;
    } else {
        message(0, "Using NON-UNIFORM UV BG from %s and %s\n", All.TreeCoolFile, All.UVFluctuationFile);
        UVF.disabled = 0;
    }
    int size;
    {
        /* read the reionized fraction */
        UVF.Zbins = h5readdouble(All.UVFluctuationFile, "Redshift_Bins", &UVF.N_Zbins);
        UVF.Fraction = h5readdouble(All.UVFluctuationFile, "ReionizedFraction", &UVF.N_Zbins);
        int dims[] = {UVF.N_Zbins};
        interp_init(&UVF.Finterp, 1, dims);
        interp_init_dim(&UVF.Finterp, 0, UVF.Zbins[0], UVF.Zbins[UVF.N_Zbins - 1]);
    }

    UVF.Nside = size;

    int Nside = UVF.Nside;
    /* This is kinda big, so we move it to mymalloc (leaving more free space for
     * system /MPI */
    UVF.Table = mymalloc("Zreion", (sizeof(double) * Nside) * (Nside * Nside));
    int i;
    double * data = h5readdouble(All.UVFluctuationFile, "Zreion_Table", &size);
    /* convert to float internally, saving memory */
    for(i = 0; i < size; i ++) {
        UVF.Table[i] = data[i];
    }
    free(data);

    if(UVF.Table[0] < 0.01 || UVF.Table[0] > 100.0) {
        endrun(123, "UV Flucutaiton doesn't seem right\n");
    }

    double * XYZ_Bins = h5readdouble(All.UVFluctuationFile, "XYZ_Bins", &size);
    int dims[] = {Nside, Nside, Nside};
    interp_init(&UVF.interp, 3, dims);
    interp_init_dim(&UVF.interp, 0, XYZ_Bins[0], XYZ_Bins[Nside - 1]);
    interp_init_dim(&UVF.interp, 1, XYZ_Bins[0], XYZ_Bins[Nside - 1]);
    interp_init_dim(&UVF.interp, 2, XYZ_Bins[0], XYZ_Bins[Nside - 1]);
    free(XYZ_Bins);
}

#if 0
/* Fraction of total universe that is ionized.
 * currently unused. Unclear if the UVBG in Treecool shall be adjusted
 * by the factor or not. seems to be NOT after reading Giguere's paper.
 * */
static double GetReionizedFraction(double time) {
    if(UVF.disabled) {
        return 1.0;
    }
    int status[1];
    double redshift = 1 / time - 1;
    double x[] = {redshift};
    double fraction = interp_eval(&UVF.Finterp, x, UVF.Fraction, status);
    if(status[0] < 0) return 0.0;
    if(status[0] > 0) return 1.0;
    return fraction;
}

#endif

/* 
 * returns the spatial dependent UVBG if UV fluctuation is enabled. 
 *
 * */
void GetParticleUVBG(int i, struct UVBG * uvbg) {
    double z = 1 / All.cf.a - 1;
    if(All.UVRedshiftThreshold >= 0.0 && z > All.UVRedshiftThreshold) {
        /* if a threshold is set, disable UV bg above that redshift */
        memset(uvbg, 0, sizeof(struct UVBG));
        return;
    }
    if(UVF.disabled) {
        /* directly use the TREECOOL table if UVF is disabled */
        memcpy(uvbg, &GlobalUVBG, sizeof(struct UVBG));
        return;
    }
    double pos[3];
    int k;
    for(k = 0; k < 3; k ++) {
        pos[k] = P[i].Pos[k];
    }
    double zreion = interp_eval_periodic(&UVF.interp, pos, UVF.Table);
    if(zreion < z) {
        memset(uvbg, 0, sizeof(struct UVBG));
    } else {
        memcpy(uvbg, &GlobalUVBG, sizeof(struct UVBG));
    }
}
void GetGlobalUVBG(struct UVBG * uvbg) {
    memcpy(uvbg, &GlobalUVBG, sizeof(struct UVBG));
}
