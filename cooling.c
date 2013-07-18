#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "allvars.h"
#include "proto.h"

#include "cooling.h"

#ifdef CS_MODEL
#include "cs_metals.h"
#endif

#ifdef COOLING

#define NCOOLTAB  2000

#define SMALLNUM 1.0e-60
#define COOLLIM  0.1
#define HEATLIM	 20.0


#ifndef CS_MODEL
static double XH = HYDROGEN_MASSFRAC;	/* hydrogen abundance by mass */
static double yhelium;
#endif

#define eV_to_K   11606.0
#define eV_to_erg 1.60184e-12


static double mhboltz;		/* hydrogen mass over Boltzmann constant */
static double ethmin;		/* minimum internal energy for neutral gas */

static double Tmin = 0.0;	/* in log10 */
static double Tmax = 9.0;
static double deltaT;

static double *BetaH0, *BetaHep, *Betaff;
static double *AlphaHp, *AlphaHep, *Alphad, *AlphaHepp;
static double *GammaeH0, *GammaeHe0, *GammaeHep;

static double J_UV = 0, gJH0 = 0, gJHep = 0, gJHe0 = 0, epsH0 = 0, epsHep = 0, epsHe0 = 0;

static double ne, necgs, nHcgs;
static double bH0, bHep, bff, aHp, aHep, aHepp, ad, geH0, geHe0, geHep;
static double gJH0ne, gJHe0ne, gJHepne;
static double nH0, nHp, nHep, nHe0, nHepp;



static double DoCool_u_old_input, DoCool_rho_input, DoCool_dt_input, DoCool_ne_guess_input;

/* returns new internal energy per unit mass. 
 * Arguments are passed in code units, density is proper density.
 */
double DoCooling(double u_old, double rho, double dt, double *ne_guess)
{
    double u, du;
    double u_lower, u_upper;
    double ratefact;
    double LambdaNet;
    int iter = 0;

    DoCool_u_old_input = u_old;
    DoCool_rho_input = rho;
    DoCool_dt_input = dt;
    DoCool_ne_guess_input = *ne_guess;


    rho *= All.UnitDensity_in_cgs * All.HubbleParam * All.HubbleParam;	/* convert to physical cgs units */
    u_old *= All.UnitPressure_in_cgs / All.UnitDensity_in_cgs;
    dt *= All.UnitTime_in_s / All.HubbleParam;

    nHcgs = XH * rho / PROTONMASS;	/* hydrogen number dens in cgs units */
    ratefact = nHcgs * nHcgs / rho;

    u = u_old;
    u_lower = u;
    u_upper = u;

    LambdaNet = CoolingRateFromU(u, rho, ne_guess);

    /* bracketing */

    if(u - u_old - ratefact * LambdaNet * dt < 0)	/* heating */
    {
        u_upper *= sqrt(1.1);
        u_lower /= sqrt(1.1);
            while(u_upper - u_old - ratefact * CoolingRateFromU(u_upper, rho, ne_guess) * dt < 0)
            {
                u_upper *= 1.1;
                u_lower *= 1.1;
            }

    }

    if(u - u_old - ratefact * LambdaNet * dt > 0)
    {
        u_lower /= sqrt(1.1);
        u_upper *= sqrt(1.1);
            while(u_lower - u_old - ratefact * CoolingRateFromU(u_lower, rho, ne_guess) * dt > 0)
            {
                u_upper /= 1.1;
                u_lower /= 1.1;
            }
    }

    do
    {
        u = 0.5 * (u_lower + u_upper);

        LambdaNet = CoolingRateFromU(u, rho, ne_guess);

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
            printf("u= %g\n", u);
    }
    while(fabs(du / u) > 1.0e-6 && iter < MAXITER);

    if(iter >= MAXITER)
    {
        printf("failed to converge in DoCooling()\n");
        printf("DoCool_u_old_input=%g\nDoCool_rho_input= %g\nDoCool_dt_input= %g\nDoCool_ne_guess_input= %g\n",
                DoCool_u_old_input, DoCool_rho_input, DoCool_dt_input, DoCool_ne_guess_input);
        endrun(10);
    }

    u *= All.UnitDensity_in_cgs / All.UnitPressure_in_cgs;	/* to internal units */

    return u;
}



/* returns cooling time. 
 * NOTE: If we actually have heating, a cooling time of 0 is returned.
 */
double GetCoolingTime(double u_old, double rho, double *ne_guess)
{
    double u;
    double ratefact;
    double LambdaNet, coolingtime;

    DoCool_u_old_input = u_old;
    DoCool_rho_input = rho;
    DoCool_ne_guess_input = *ne_guess;

    rho *= All.UnitDensity_in_cgs * All.HubbleParam * All.HubbleParam;	/* convert to physical cgs units */
    u_old *= All.UnitPressure_in_cgs / All.UnitDensity_in_cgs;


    nHcgs = XH * rho / PROTONMASS;	/* hydrogen number dens in cgs units */
    ratefact = nHcgs * nHcgs / rho;

    u = u_old;

    LambdaNet = CoolingRateFromU(u, rho, ne_guess);

    /* bracketing */

    if(LambdaNet >= 0)		/* ups, we have actually heating due to UV background */
        return 0;

    coolingtime = u_old / (-ratefact * LambdaNet);

    coolingtime *= All.HubbleParam / All.UnitTime_in_s;

    return coolingtime;
}


/* returns new internal energy per unit mass. 
 * Arguments are passed in code units, density is proper density.
 */
double DoInstabilityCooling(double m_old, double u, double rho, double dt, double fac, double *ne_guess)
{
    double m, dm;
    double m_lower, m_upper;
    double ratefact;
    double LambdaNet;
    int iter = 0;

    DoCool_u_old_input = u;
    DoCool_rho_input = rho;
    DoCool_dt_input = dt;
    DoCool_ne_guess_input = *ne_guess;

    if(fac <= 0)			/* the hot phase is actually colder than the cold reservoir! */
    {
        return 0.01 * m_old;
    }

    rho *= All.UnitDensity_in_cgs * All.HubbleParam * All.HubbleParam;	/* convert to physical cgs units */
    u *= All.UnitPressure_in_cgs / All.UnitDensity_in_cgs;
    dt *= All.UnitTime_in_s / All.HubbleParam;
    fac *= All.UnitMass_in_g / All.UnitEnergy_in_cgs;

    nHcgs = XH * rho / PROTONMASS;	/* hydrogen number dens in cgs units */
    ratefact = nHcgs * nHcgs / rho * fac;

    m = m_old;
    m_lower = m;
    m_upper = m;

    LambdaNet = CoolingRateFromU(u, rho, ne_guess);

    /* bracketing */

    if(m - m_old - m * m / m_old * ratefact * LambdaNet * dt < 0)	/* heating */
    {
        m_upper *= sqrt(1.1);
        m_lower /= sqrt(1.1);
        while(m_upper - m_old -
            m_upper * m_upper / m_old * ratefact * CoolingRateFromU(u, rho * m_upper / m_old,
                    ne_guess) * dt < 0)
                    {
                        m_upper *= 1.1;
                        m_lower *= 1.1;
                    }
    }

    if(m - m_old - m_old * ratefact * LambdaNet * dt > 0)
    {
        m_lower /= sqrt(1.1);
        m_upper *= sqrt(1.1);
        while(m_lower - m_old -
            m_lower * m_lower / m_old * ratefact * CoolingRateFromU(u, rho * m_lower / m_old,
                    ne_guess) * dt > 0)
                    {
                        m_upper /= 1.1;
                        m_lower /= 1.1;
                    }
    }

    do
    {
        m = 0.5 * (m_lower + m_upper);

        LambdaNet = CoolingRateFromU(u, rho * m / m_old, ne_guess);

        if(m - m_old - m * m / m_old * ratefact * LambdaNet * dt > 0)
        {
            m_upper = m;
        }
        else
        {
            m_lower = m;
        }

        dm = m_upper - m_lower;

        iter++;

        if(iter >= (MAXITER - 10))
            printf("m= %g\n", m);
    }
    while(fabs(dm / m) > 1.0e-6 && iter < MAXITER);

    if(iter >= MAXITER)
    {
        printf("failed to converge in DoCooling()\n");
        printf("DoCool_u_old_input=%g\nDoCool_rho_input= %g\nDoCool_dt_input= %g\nDoCool_ne_guess_input= %g\n",
                DoCool_u_old_input, DoCool_rho_input, DoCool_dt_input, DoCool_ne_guess_input);
        printf("m_old= %g\n", m_old);
        endrun(11);
    }

    return m;
}








void cool_test(void)
{
    double uin, rhoin, tempin, muin, nein;

    uin = 6.01329e+09;
    rhoin = 7.85767e-29;
    tempin = 34.0025;
    muin = 0.691955;

    nein = (1 + 4 * yhelium) / muin - (1 + yhelium);

    printf("%g\n", convert_u_to_temp(uin, rhoin, &nein));
}


/* this function determines the electron fraction, and hence the mean 
 * molecular weight. With it arrives at a self-consistent temperature.
 * Element abundances and the rates for the emission are also computed
 */
double convert_u_to_temp(double u, double rho, double *ne_guess)
{
    double temp, temp_old, temp_new, max = 0, ne_old;
    double mu, dmax1, dmax2;
    int iter = 0;

    double u_input, rho_input, ne_input;

    u_input = u;
    rho_input = rho;
    ne_input = *ne_guess;

    mu = (1 + 4 * yhelium) / (1 + yhelium + *ne_guess);
    temp = GAMMA_MINUS1 / BOLTZMANN * u * PROTONMASS * mu;

    do
    {
        ne_old = *ne_guess;

        find_abundances_and_rates(log10(temp), rho, ne_guess);
        temp_old = temp;

        mu = (1 + 4 * yhelium) / (1 + yhelium + *ne_guess);

        temp_new = GAMMA_MINUS1 / BOLTZMANN * u * PROTONMASS * mu;

        max =
            DMAX(max,
                    temp_new / (1 + yhelium + *ne_guess) * fabs((*ne_guess - ne_old) / (temp_new - temp_old + 1.0)));

        temp = temp_old + (temp_new - temp_old) / (1 + max);
        iter++;

        if(iter > (MAXITER - 10))
                printf("-> temp= %g ne=%g\n", temp, *ne_guess);
    }
    while(fabs(temp - temp_old) > 1.0e-3 * temp && iter < MAXITER);

    if(iter >= MAXITER)
        {
            printf("failed to converge in convert_u_to_temp()\n");
            printf("u_input= %g\nrho_input=%g\n ne_input=%g\n", u_input, rho_input, ne_input);
            printf
                ("DoCool_u_old_input=%g\nDoCool_rho_input= %g\nDoCool_dt_input= %g\nDoCool_ne_guess_input= %g\n",
                 DoCool_u_old_input, DoCool_rho_input, DoCool_dt_input, DoCool_ne_guess_input);

            endrun(12);
        }

    return temp;
}


/* this function computes the actual abundance ratios 
*/
void find_abundances_and_rates(double logT, double rho, double *ne_guess)
{
    double neold, nenew;
    int j, niter;
    double Tlow, Thi, flow, fhi, t;

    double logT_input, rho_input, ne_input;

    logT_input = logT;
    rho_input = rho;
    ne_input = *ne_guess;

    if(logT <= Tmin)		/* everything neutral */
    {
        nH0 = 1.0;
        nHe0 = yhelium;
        nHp = 0;
        nHep = 0;
        nHepp = 0;
        ne = 0;
        *ne_guess = 0;
        return;
    }

    if(logT >= Tmax)		/* everything is ionized */
    {
        nH0 = 0;
        nHe0 = 0;
        nHp = 1.0;
        nHep = 0;
        nHepp = yhelium;
        ne = nHp + 2.0 * nHepp;
        *ne_guess = ne;		/* note: in units of the hydrogen number density */
        return;
    }

    t = (logT - Tmin) / deltaT;
    j = (int) t;
    Tlow = Tmin + deltaT * j;
    Thi = Tlow + deltaT;
    fhi = t - j;
    flow = 1 - fhi;

    if(*ne_guess == 0)
        *ne_guess = 1.0;

    nHcgs = XH * rho / PROTONMASS;	/* hydrogen number dens in cgs units */

    ne = *ne_guess;
    neold = ne;
    niter = 0;
    necgs = ne * nHcgs;

    /* evaluate number densities iteratively (cf KWH eqns 33-38) in units of nH */
    do
    {
        niter++;

        aHp = flow * AlphaHp[j] + fhi * AlphaHp[j + 1];
        aHep = flow * AlphaHep[j] + fhi * AlphaHep[j + 1];
        aHepp = flow * AlphaHepp[j] + fhi * AlphaHepp[j + 1];
        ad = flow * Alphad[j] + fhi * Alphad[j + 1];
        geH0 = flow * GammaeH0[j] + fhi * GammaeH0[j + 1];
        geHe0 = flow * GammaeHe0[j] + fhi * GammaeHe0[j + 1];
        geHep = flow * GammaeHep[j] + fhi * GammaeHep[j + 1];

        if(necgs <= 1.e-25 || J_UV == 0)
        {
            gJH0ne = gJHe0ne = gJHepne = 0;
        }
        else
        {
            gJH0ne = gJH0 / necgs;
            gJHe0ne = gJHe0 / necgs;
            gJHepne = gJHep / necgs;
        }

        nH0 = aHp / (aHp + geH0 + gJH0ne);	/* eqn (33) */
        nHp = 1.0 - nH0;		/* eqn (34) */

        if((gJHe0ne + geHe0) <= SMALLNUM)	/* no ionization at all */
        {
            nHep = 0.0;
            nHepp = 0.0;
            nHe0 = yhelium;
        }
        else
        {
            nHep = yhelium / (1.0 + (aHep + ad) / (geHe0 + gJHe0ne) + (geHep + gJHepne) / aHepp);	/* eqn (35) */
            nHe0 = nHep * (aHep + ad) / (geHe0 + gJHe0ne);	/* eqn (36) */
            nHepp = nHep * (geHep + gJHepne) / aHepp;	/* eqn (37) */
        }

        neold = ne;

        ne = nHp + nHep + 2 * nHepp;	/* eqn (38) */
        necgs = ne * nHcgs;

        if(J_UV == 0)
            break;

        nenew = 0.5 * (ne + neold);
        ne = nenew;
        necgs = ne * nHcgs;

        if(fabs(ne - neold) < 1.0e-4)
            break;

        if(niter > (MAXITER - 10))
            printf("ne= %g  niter=%d\n", ne, niter);
    }
    while(niter < MAXITER);

    if(niter >= MAXITER)
    {
        printf("no convergence reached in find_abundances_and_rates()\n");
        printf("logT_input= %g  rho_input= %g  ne_input= %g\n", logT_input, rho_input, ne_input);
        printf("DoCool_u_old_input=%g\nDoCool_rho_input= %g\nDoCool_dt_input= %g\nDoCool_ne_guess_input= %g\n",
                DoCool_u_old_input, DoCool_rho_input, DoCool_dt_input, DoCool_ne_guess_input);
        endrun(13);
    }

    bH0 = flow * BetaH0[j] + fhi * BetaH0[j + 1];
    bHep = flow * BetaHep[j] + fhi * BetaHep[j + 1];
    bff = flow * Betaff[j] + fhi * Betaff[j + 1];

    *ne_guess = ne;
}



/*  this function first computes the self-consistent temperature
 *  and abundance ratios, and then it calculates 
 *  (heating rate-cooling rate)/n_h^2 in cgs units 
 */
double CoolingRateFromU(double u, double rho, double *ne_guess)
{
    double temp;

    temp = convert_u_to_temp(u, rho, ne_guess);

#ifndef CS_MODEL
    return CoolingRate(log10(temp), rho, ne_guess);
#else
    return cs_CoolingRate_SD(log10(temp), rho, ne_guess);
#endif
}


/*  this function computes the self-consistent temperature
 *  and abundance ratios 
 */
double AbundanceRatios(double u, double rho, double *ne_guess, double *nH0_pointer, double *nHeII_pointer)
{
    double temp;

    DoCool_u_old_input = u;
    DoCool_rho_input = rho;
    DoCool_ne_guess_input = *ne_guess;

    rho *= All.UnitDensity_in_cgs * All.HubbleParam * All.HubbleParam;	/* convert to physical cgs units */
    u *= All.UnitPressure_in_cgs / All.UnitDensity_in_cgs;

    temp = convert_u_to_temp(u, rho, ne_guess);

    *nH0_pointer = nH0;
    *nHeII_pointer = nHep;

    return temp;
}




extern FILE *fd;

/*  Calculates (heating rate-cooling rate)/n_h^2 in cgs units 
*/
double CoolingRate(double logT, double rho, double *nelec)
{
    double Lambda, Heat;
    double LambdaExc, LambdaIon, LambdaRec, LambdaFF, LambdaCmptn = 0.0;
    double LambdaExcH0, LambdaExcHep, LambdaIonH0, LambdaIonHe0, LambdaIonHep;
    double LambdaRecHp, LambdaRecHep, LambdaRecHepp, LambdaRecHepd;
    double redshift;
    double T;

#if defined (UM_METAL_COOLING) && defined (UM_MET_IN_LT_COOLING)
    double LambdaMetalLines;
#endif

    if(logT <= Tmin)
        logT = Tmin + 0.5 * deltaT;	/* floor at Tmin */


    nHcgs = XH * rho / PROTONMASS;	/* hydrogen number dens in cgs units */


    if(logT < Tmax)
    {
        find_abundances_and_rates(logT, rho, nelec);

        /* Compute cooling and heating rate (cf KWH Table 1) in units of nH**2 */
        T = pow(10.0, logT);

#if defined (UM_CHEMISTRY) && defined (UM_METAL_COOLING) && defined (UM_MET_IN_LT_COOLING)
#ifdef UM_e_MET_IMPACTS
        LambdaMetalLines = um_metal_line_cooling(T, rho, LT_NMet, ne * nHcgs) / (nHcgs * nHcgs);
#else
        LambdaMetalLines = um_metal_line_cooling(T, rho, LT_NMet) / (nHcgs * nHcgs);
#endif
#endif

        LambdaExcH0 = bH0 * ne * nH0;
        LambdaExcHep = bHep * ne * nHep;
        LambdaExc = LambdaExcH0 + LambdaExcHep;	/* excitation */

        LambdaIonH0 = 2.18e-11 * geH0 * ne * nH0;
        LambdaIonHe0 = 3.94e-11 * geHe0 * ne * nHe0;
        LambdaIonHep = 8.72e-11 * geHep * ne * nHep;
        LambdaIon = LambdaIonH0 + LambdaIonHe0 + LambdaIonHep;	/* ionization */

        LambdaRecHp = 1.036e-16 * T * ne * (aHp * nHp);
        LambdaRecHep = 1.036e-16 * T * ne * (aHep * nHep);
        LambdaRecHepp = 1.036e-16 * T * ne * (aHepp * nHepp);
        LambdaRecHepd = 6.526e-11 * ad * ne * nHep;
        LambdaRec = LambdaRecHp + LambdaRecHep + LambdaRecHepp + LambdaRecHepd;

        LambdaFF = bff * (nHp + nHep + 4 * nHepp) * ne;

        Lambda = LambdaExc + LambdaIon + LambdaRec + LambdaFF;

#if defined (UM_CHEMISTRY) && defined (UM_METAL_COOLING) && defined (UM_MET_IN_LT_COOLING)
        Lambda += LambdaMetalLines;
#endif

        if(All.ComovingIntegrationOn)
        {
            redshift = 1 / All.Time - 1;
            LambdaCmptn = 5.65e-36 * ne * (T - 2.73 * (1. + redshift)) * pow(1. + redshift, 4.) / nHcgs;

            Lambda += LambdaCmptn;
        }
        else
            LambdaCmptn = 0;

        Heat = 0;
        if(J_UV != 0)
            Heat += (nH0 * epsH0 + nHe0 * epsHe0 + nHep * epsHep) / nHcgs;

    }
    else				/* here we're outside of tabulated rates, T>Tmax K */
    {
        /* at high T (fully ionized); only free-free and Compton cooling are present.  
           Assumes no heating. */

        Heat = 0;

        LambdaExcH0 = LambdaExcHep = LambdaIonH0 = LambdaIonHe0 = LambdaIonHep =
            LambdaRecHp = LambdaRecHep = LambdaRecHepp = LambdaRecHepd = 0;

        /* very hot: H and He both fully ionized */
        nHp = 1.0;
        nHep = 0;
        nHepp = yhelium;
        ne = nHp + 2.0 * nHepp;
        *nelec = ne;		/* note: in units of the hydrogen number density */

        T = pow(10.0, logT);
        LambdaFF =
            1.42e-27 * sqrt(T) * (1.1 + 0.34 * exp(-(5.5 - logT) * (5.5 - logT) / 3)) * (nHp + 4 * nHepp) * ne;

        if(All.ComovingIntegrationOn)
        {
            redshift = 1 / All.Time - 1;
            /* add inverse Compton cooling off the microwave background */
            LambdaCmptn = 5.65e-36 * ne * (T - 2.73 * (1. + redshift)) * pow(1. + redshift, 4.) / nHcgs;
        }
        else
            LambdaCmptn = 0;

        Lambda = LambdaFF + LambdaCmptn;
    }

    /*      
            printf("Lambda= %g\n", Lambda);

            fprintf(fd,"%g %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g\n", pow(10, logT),Lambda,
            LambdaExcH0, LambdaExcHep, 
            LambdaIonH0, LambdaIonHe0, LambdaIonHep,
            LambdaRecHp, LambdaRecHep, LambdaRecHepp, LambdaRecHepd,
            LambdaFF, LambdaCmptn, Heat,
            ne, nHp, nHep, nHepp);
            */

    return (Heat - Lambda);
}





double LogTemp(double u, double ne)	/* ne= electron density in terms of hydrogen density */
{
    double T;

    if(u < ethmin)
        u = ethmin;

    T = log10(GAMMA_MINUS1 * u * mhboltz * (1 + 4 * yhelium) / (1 + ne + yhelium));

    return T;
}



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





/* table input (from file TREECOOL) for ionizing parameters */

#define JAMPL	1.0		/* amplitude factor relative to input table */
#define TABLESIZE 200		/* Max # of lines in TREECOOL */

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
        printf(" Cannot read ionization table in file `%s'\n", fname);
        endrun(456);
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

    if(ThisTask == 0)
        printf("\n\nread ionization table with %d entries in file `%s'.\n\n", nheattab, fname);
}


void IonizeParams(void)
{
    IonizeParamsTable();

    /*
       IonizeParamsFunction();
       */
}



void IonizeParamsTable(void)
{
    int i, ilow;
    double logz, dzlow, dzhi;
    double redshift;

    if(All.ComovingIntegrationOn)
        redshift = 1 / All.Time - 1;
    else
    {
        gJHe0 = gJHep = gJH0 = 0;
        epsHe0 = epsHep = epsH0 = 0;
        J_UV = 0;
        return;
    }

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
        gJHe0 = gJHep = gJH0 = 0;
        epsHe0 = epsHep = epsH0 = 0;
        J_UV = 0;
        return;
    }
    else
        J_UV = 1.e-21;		/* irrelevant as long as it's not 0 */

    gJH0 = JAMPL * pow(10., (dzhi * log10(gH0[ilow]) + dzlow * log10(gH0[ilow + 1])) / (dzlow + dzhi));
    gJHe0 = JAMPL * pow(10., (dzhi * log10(gHe[ilow]) + dzlow * log10(gHe[ilow + 1])) / (dzlow + dzhi));
    gJHep = JAMPL * pow(10., (dzhi * log10(gHep[ilow]) + dzlow * log10(gHep[ilow + 1])) / (dzlow + dzhi));
    epsH0 = JAMPL * pow(10., (dzhi * log10(eH0[ilow]) + dzlow * log10(eH0[ilow + 1])) / (dzlow + dzhi));
    epsHe0 = JAMPL * pow(10., (dzhi * log10(eHe[ilow]) + dzlow * log10(eHe[ilow + 1])) / (dzlow + dzhi));
    epsHep = JAMPL * pow(10., (dzhi * log10(eHep[ilow]) + dzlow * log10(eHep[ilow + 1])) / (dzlow + dzhi));

    return;
}


void SetZeroIonization(void)
{
    gJHe0 = gJHep = gJH0 = 0;
    epsHe0 = epsHep = epsH0 = 0;
    J_UV = 0;
}


void IonizeParamsFunction(void)
{
    int i, nint;
    double a0, planck, ev, e0_H, e0_He, e0_Hep;
    double gint, eint, t, tinv, fac, eps;
    double at, beta, s;
    double pi;

#define UVALPHA         1.0
    double Jold = -1.0;
    double redshift;

    J_UV = 0.;
    gJHe0 = gJHep = gJH0 = 0.;
    epsHe0 = epsHep = epsH0 = 0.;


    if(All.ComovingIntegrationOn)	/* analytically compute params from power law J_nu */
    {
        redshift = 1 / All.Time - 1;

        if(redshift >= 6)
            J_UV = 0.;
        else
        {
            if(redshift >= 3)
                J_UV = 4e-22 / (1 + redshift);
            else
            {
                if(redshift >= 2)
                    J_UV = 1e-22;
                else
                    J_UV = 1.e-22 * pow(3.0 / (1 + redshift), -3.0);
            }
        }

        if(J_UV == Jold)
            return;


        Jold = J_UV;

        if(J_UV == 0)
            return;


        a0 = 6.30e-18;
        planck = 6.6262e-27;
        ev = 1.6022e-12;
        e0_H = 13.6058 * ev;
        e0_He = 24.59 * ev;
        e0_Hep = 54.4232 * ev;

        gint = 0.0;
        eint = 0.0;
        nint = 5000;
        at = 1. / ((double) nint);

        for(i = 1; i <= nint; i++)
        {
            t = (double) i;
            t = (t - 0.5) * at;
            tinv = 1. / t;
            eps = sqrt(tinv - 1.);
            fac = exp(4. - 4. * atan(eps) / eps) / (1. - exp(-2. * M_PI / eps)) * pow(t, UVALPHA + 3.);
            gint += fac * at;
            eint += fac * (tinv - 1.) * at;
        }

        gJH0 = a0 * gint / planck;
        epsH0 = a0 * eint * (e0_H / planck);
        gJHep = gJH0 * pow(e0_H / e0_Hep, UVALPHA) / 4.0;
        epsHep = epsH0 * pow((e0_H / e0_Hep), UVALPHA - 1.) / 4.0;

        at = 7.83e-18;
        beta = 1.66;
        s = 2.05;

        gJHe0 = (at / planck) * pow((e0_H / e0_He), UVALPHA) *
            (beta / (UVALPHA + s) + (1. - beta) / (UVALPHA + s + 1));
        epsHe0 = (e0_He / planck) * at * pow(e0_H / e0_He, UVALPHA) *
            (beta / (UVALPHA + s - 1) + (1 - 2 * beta) / (UVALPHA + s) - (1 - beta) / (UVALPHA + s + 1));

        pi = M_PI;
        gJH0 *= 4. * pi * J_UV;
        gJHep *= 4. * pi * J_UV;
        gJHe0 *= 4. * pi * J_UV;
        epsH0 *= 4. * pi * J_UV;
        epsHep *= 4. * pi * J_UV;
        epsHe0 *= 4. * pi * J_UV;
    }
}


void InitCool(void)
{
    InitCoolMemory();
    MakeCoolingTable();

#ifdef CS_MODEL

    cs_read_coolrate_table();

#ifdef CS_SNII
    cs_read_yield_table();

    cs_imf();
#endif

#endif



    ReadIonizeParams("TREECOOL");

    All.Time = All.TimeBegin;
    IonizeParams();

}

#ifdef CS_MODEL

double cs_CoolingRate_SD(double logT, double rho, double *nelec)
{
    double Lambda, Heat;
    double LambdaCmptn = 0.0;
    double redshift;
    double T;

    if(logT <= Tmin)
        logT = Tmin + 0.5 * deltaT;	/* floor at Tmin */

    nHcgs = XH * rho / PROTONMASS;	/* hydrogen number dens in cgs units */

    if(logT < Tmax)
    {

        find_abundances_and_rates(logT, rho, nelec);

        T = pow(10.0, logT);

        /* METALS: the second input parameter is the abundance Fe/H
        */

        Lambda = cs_get_Lambda_SD(logT, FeHgas);


        if(All.ComovingIntegrationOn)
        {
            redshift = 1 / All.Time - 1;
            LambdaCmptn = 5.65e-36 * ne * (T - 2.73 * (1. + redshift)) * pow(1. + redshift, 4.) / nHcgs;

            Lambda += LambdaCmptn;
        }
        else
            LambdaCmptn = 0;

        Heat = 0;
        if(J_UV != 0)
            Heat += (nH0 * epsH0 + nHe0 * epsHe0 + nHep * epsHep) / nHcgs;
    }
    else				/* here we're outside of tabulated rates, T>Tmax K */
    {
        /* at high T (fully ionized); only free-free and Compton cooling are present.
           Assumes no heating. */

        Heat = 0;

        /* very hot: H and He both fully ionized */
        nHp = 1.0;
        nHep = 0;
        nHepp = yhelium;
        ne = nHp + 2.0 * nHepp;
        *nelec = ne;		/* note: in units of the hydrogen number density */

        T = pow(10.0, logT);

        Lambda = cs_get_Lambda_SD(logT, FeHgas);

        if(All.ComovingIntegrationOn)
        {
            redshift = 1 / All.Time - 1;
            /* add inverse Compton cooling off the microwave background */
            LambdaCmptn = 5.65e-36 * ne * (T - 2.73 * (1. + redshift)) * pow(1. + redshift, 4.) / nHcgs;
        }
        else
            LambdaCmptn = 0;

        Lambda += LambdaCmptn;
    }

    return (Heat - Lambda);
}

#endif


#endif
