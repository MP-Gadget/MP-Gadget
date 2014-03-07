#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <hdf5.h>

#define  BOLTZMANN   1.38066e-16
#ifndef  GAMMA
#define  GAMMA         (5.0/3.0)	/*!< adiabatic index of simulated gas */
#endif
#define  PROTONMASS  1.6726e-24
#define MAXITER 400
#define COMPTONFACTOR 5.65e-36 /* old gadget has 5.65e-36*/
#define  GAMMA_MINUS1  (GAMMA-1)
#define  HYDROGEN_MASSFRAC 0.76 //0.76	/*!< mass fraction of hydrogen, relevant only for radiative cooling */

//#define NEW_RATES

#define NCOOLTAB  2000

#define SMALLNUM 1.0e-60
#define COOLLIM  0.1
#define HEATLIM	 20.0

double CoolingRate(double logT, double lognH);
void IonizeParams(void);
void write_bins(hid_t fid, double * bins, int Nbins, char * name);
void write_component(hid_t fid, double * table, char * name);
static double Time;

static double XH = HYDROGEN_MASSFRAC;	/* hydrogen abundance by mass */
static double yhelium;

#define eV_to_K   11606.0
#define eV_to_erg 1.60184e-12


static double mhboltz;		/* hydrogen mass over Boltzmann constant */
static double ethmin;		/* minimum internal energy for neutral gas */

static double Tmin = 1.0;	/* in log10 */
static double Tmax = 9.0;
static double deltaT;

static double *BetaH0, *BetaHep, *Betaff;
static double *AlphaHp, *AlphaHep, *Alphad, *AlphaHepp;
static double *GammaeH0, *GammaeHe0, *GammaeHep;

static double J_UV = 0, gJH0 = 0, gJHep = 0, gJHe0 = 0, epsH0 = 0, epsHep = 0, epsHe0 = 0;

static double ne;
static double bH0, bHep, bff, aHp, aHep, aHepp, ad, geH0, geHe0, geHep;
static double gJH0ne, gJHe0ne, gJHepne;
static double nH0, nHp, nHep, nHe0, nHepp;
static double Heat, Lambda, LambdaCmptn;


static double DoCool_u_old_input, DoCool_rho_input, DoCool_dt_input, DoCool_ne_guess_input;


/* this function computes the actual abundance ratios 
*/
void find_abundances_and_rates(double logT, double rho, double ne_guess)
{
    double neold, nenew;
    int j, niter;
    double Tlow, Thi, flow, fhi, t;

    double logT_input, rho_input, ne_input;

    logT_input = logT;
    rho_input = rho;
    ne_input = ne_guess;

    if(logT <= Tmin)		/* everything neutral */
    {
        nH0 = 1.0;
        nHe0 = yhelium;
        nHp = 0;
        nHep = 0;
        nHepp = 0;
        ne = 0;
        return;
    }

    if(logT >= Tmax)		/* everything is ionized */
    {
        nH0 = 0;
        nHe0 = 0;
        nHp = 1.0;
        nHep = 0;
        nHepp = yhelium;
        ne = nHp + 2.0 * nHepp; /* note: in units of the hydrogen number density */
        return;
    }

    t = (logT - Tmin) / deltaT;
    j = (int) t;
    Tlow = Tmin + deltaT * j;
    Thi = Tlow + deltaT;
    fhi = t - j;
    flow = 1 - fhi;

    if(ne_guess == 0) ne_guess = 1.0;

    double nHcgs = XH * rho / PROTONMASS;	/* hydrogen number dens in cgs units */

    ne = ne_guess;
    neold = ne;
    niter = 0;
    double necgs = ne * nHcgs;

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
        if( fabs(1 / Time - 1 - 2.0) < 0.1) {
            if( fabs(logT - 3.96064) < 0.2) {
                 if(nHcgs >2.e-6 && nHcgs < 2.9e-6) {
                     printf("%g %g %d %g\n", logT, nHcgs, niter, nH0);
                 }
            }
        }
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
        abort();
    }

    bH0 = flow * BetaH0[j] + fhi * BetaH0[j + 1];
    bHep = flow * BetaHep[j] + fhi * BetaHep[j + 1];
    bff = flow * Betaff[j] + fhi * Betaff[j + 1];
}



/*  Calculates (heating rate-cooling rate)/n_h^2 in cgs units 
*/
double CoolingRate(double logT, double lognHcgs)
{
    double LambdaExc, LambdaIon, LambdaRec, LambdaFF;
    double LambdaExcH0, LambdaExcHep, LambdaIonH0, LambdaIonHe0, LambdaIonHep;
    double LambdaRecHp, LambdaRecHep, LambdaRecHepp, LambdaRecHepd;
    double redshift;
    double T;

    if(logT <= Tmin)
        logT = Tmin + 0.5 * deltaT;	/* floor at Tmin */


    double nHcgs = pow(10., lognHcgs); //XH * rho / PROTONMASS;	/* hydrogen number dens in cgs units */

    double rho = nHcgs / XH * PROTONMASS;

    if(logT < Tmax)
    {
        find_abundances_and_rates(logT, rho, 0.0);

        /* Compute cooling and heating rate (cf KWH Table 1) in units of nH**2 */
        T = pow(10.0, logT);

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

        redshift = 1 / Time - 1;
        LambdaCmptn = COMPTONFACTOR * ne * (T - 2.73 * (1. + redshift)) * pow(1. + redshift, 4.) / nHcgs;

        Lambda += LambdaCmptn;

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
        ne = nHp + 2.0 * nHepp; /* note: in units of the hydrogen number density */

        T = pow(10.0, logT);
        LambdaFF =
            1.42e-27 * sqrt(T) * (1.1 + 0.34 * exp(-(5.5 - logT) * (5.5 - logT) / 3)) * (nHp + 4 * nHepp) * ne;

        redshift = 1 / Time - 1;
            /* add inverse Compton cooling off the microwave background */
        LambdaCmptn = 5.65e-36 * ne * (T - 2.73 * (1. + redshift)) * pow(1. + redshift, 4.) / nHcgs;

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





void InitCoolMemory(void)
{
    BetaH0 = (double *) malloc((NCOOLTAB + 1) * sizeof(double));
    BetaHep = (double *) malloc((NCOOLTAB + 1) * sizeof(double));
    AlphaHp = (double *) malloc((NCOOLTAB + 1) * sizeof(double));
    AlphaHep = (double *) malloc((NCOOLTAB + 1) * sizeof(double));
    Alphad = (double *) malloc((NCOOLTAB + 1) * sizeof(double));
    AlphaHepp = (double *) malloc((NCOOLTAB + 1) * sizeof(double));
    GammaeH0 = (double *) malloc((NCOOLTAB + 1) * sizeof(double));
    GammaeHe0 = (double *) malloc((NCOOLTAB + 1) * sizeof(double));
    GammaeHep = (double *) malloc((NCOOLTAB + 1) * sizeof(double));
    Betaff = (double *) malloc((NCOOLTAB + 1) * sizeof(double));
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

    XH = HYDROGEN_MASSFRAC;
    yhelium = (1 - XH) / (4 * XH);

    mhboltz = PROTONMASS / BOLTZMANN;

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
#define TABLESIZE 1000 /* Max # of lines in TREECOOL */

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
        abort();
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

    printf("\n\nread ionization table with %d entries in file `%s'.\n\n", nheattab, fname);
}


void IonizeParamsTable(void)
{
    int i, ilow;
    double logz, dzlow, dzhi;
    double redshift;

    redshift = 1 / Time - 1;

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


void IonizeParams(void)
{
    IonizeParamsTable();

}



void SetZeroIonization(void)
{
    gJHe0 = gJHep = gJH0 = 0;
    epsHe0 = epsHep = epsH0 = 0;
    J_UV = 0;
}


int main(int argc, char * argv[]) {
    InitCoolMemory();
    ReadIonizeParams(argv[1]);

    MakeCoolingTable();

    double table[9][51][51][200];
    double Redshift_bins[51];
    double HydrogenNumberDensity_bins[51];
    double Temperature_bins[200];
    double z = 0.0;
    double logn = 0.0;
    double logT = 0.0;
    int i, j, k, l;
    hid_t fid;

    for(i = 0; i < 51; i ++) {
        z = i * 0.2;
        Redshift_bins[i] = z;
    }
    for(j = 0; j < 51; j ++) {
        logn = -8 + 0.2 * j;
        HydrogenNumberDensity_bins[j] = logn;
    }
    for(k = 0; k < 200; k ++) {
        logT = 1 + 8. / 199.95651587694084 * k;
        Temperature_bins[k] = logT;
    }
    for(i = 0; i < 51; i ++) {
        z = Redshift_bins[i];
        Time = 1 / ( z + 1.);
        IonizeParams();

        for(j = 0; j < 51; j ++) {
            logn = HydrogenNumberDensity_bins[j];
            double nHcgs = pow(10., logn);

            for(k = 0; k < 200; k ++) {

                /* to match the AREPO table from Vogelsburger */
                logT = Temperature_bins[k];
                double T = pow(10., logT);
                CoolingRate(logT, logn);
                double a[] = {
                    ne, nH0, nHp, nHep, nHe0, 
                    nHepp, Heat, Lambda - LambdaCmptn, LambdaCmptn
                };
                for(l = 0; l < 9; l ++) {
                    table[l][i][j][k] = a[l];
                }
            }
        }
    }

    fid = H5Fcreate(argv[2], H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    write_component(fid, (double*)table[0], "ElectronAbundance");
    write_component(fid, (double*)table[1], "nH0");
    write_component(fid, (double*)table[2], "nHp");
    write_component(fid, (double*)table[3], "nHep");
    write_component(fid, (double*)table[4], "nHe0");
    write_component(fid, (double*)table[5], "nHepp");
    write_component(fid, (double*)table[6], "Heat");
    write_component(fid, (double*)table[7], "PrimordialCoolingRate");
    write_component(fid, (double*)table[8], "ComptonCoolingRate");
    write_bins(fid, Redshift_bins, 51, "Redshift_bins");
    write_bins(fid, HydrogenNumberDensity_bins, 51, "HydrogenNumberDensity_bins");
    write_bins(fid, Temperature_bins, 200, "Temperature_bins");
    H5Fclose(fid);
}

void write_bins(hid_t fid, double * bins, int Nbins, char * name) {
    hsize_t dims[1] = {Nbins};
    hid_t dsid = H5Screate_simple(1, dims, NULL);
    hid_t did = H5Dcreate2(fid, name, H5T_NATIVE_DOUBLE, dsid, 
            H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(did, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, bins);
    H5Dclose(did);
    H5Sclose(dsid);

}
void write_component(hid_t fid, double * table, char * name) {
    hsize_t dims[3] = {51, 51, 200};
    hid_t dsid = H5Screate_simple(3, dims, NULL);
    hid_t did = H5Dcreate2(fid, name, H5T_NATIVE_DOUBLE, dsid, 
            H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(did, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, table);
    H5Dclose(did);
    H5Sclose(dsid);
}
