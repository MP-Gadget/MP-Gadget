#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <hdf5.h>
#include "allvars.h"
#include "proto.h"
#include "interp.h"
#include "cooling.h"
#ifdef COOLING
/* used at init for caculating the star formation threshold*/
static int ZeroIonizationFlag = 0;
static int CoolingNoMetal = 1;
/* hydrogen abundance by mass */

#define XH HYDROGEN_MASSFRAC

#define yhelium ((1 - XH) / (4 * XH))

struct abundance {
    double ne;
    double nH0;
    double nHp;
    double nHe0;
    double nHep;
    double nHepp;
};

static double TableTemperature(double redshift, double logU, double lognH);
static double TableCoolingRate(double redshift, double logU, double lognH);
static void TableAbundance(double redshift, double logT, double lognH, struct abundance * y);
static double TableMetalCoolingRate(double redshift, double logT, double lognH);

static double * h5readdouble(char * filename, char * dataset, int * Nread) {
    void * buffer;
    int N;
    if(ThisTask == 0) {
        hid_t hfile = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
        hid_t dset = H5Dopen2(hfile, dataset, H5P_DEFAULT);
        hid_t dspace = H5Dget_space(dset);
        int ndims = H5Sget_simple_extent_ndims(dspace);
        hsize_t dims[10];
        H5Sget_simple_extent_dims(dspace, dims, NULL);
        N = 1;
        int i;
        for(i = 0; i < ndims; i ++) {
            N *= dims[i];
        }
        H5Sclose(dspace);

        MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

        buffer = mymalloc(dataset, N * sizeof(double));

        H5Dread(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, buffer);
        H5Dclose(dset);
        H5Fclose(hfile);
    } else {
        MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
        buffer = mymalloc(dataset, N * sizeof(double));
    }

    MPI_Bcast(buffer, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    *Nread = N;
    return buffer;
}

struct {
    int NRedshift_bins;
    double * Redshift_bins;

    int NHydrogenNumberDensity_bins;
    double * HydrogenNumberDensity_bins;

    int NTemperature_bins;
    double * Temperature_bins;

    int NSpecInternalEnergy_bins;
    double * SpecInternalEnergy_bins;

    double * nHp_table;
    double * nHep_table;
    double * nHepp_table;
    double * Lpnet_table; /* primordial cooling - UVB heating + CMB Compton*/
    double * T_table;

    Interp interp;
    Interp interpT;
} PC;

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

void InitCool(void) {
    /* XXX: we assume uniform bins. otherwise interp will fail! */

    int size;  /* this variable is not checked. if the file was inconsistent we ruin the sim! */

    PC.Redshift_bins = h5readdouble(All.TreeCoolFile, "Redshift_bins", &PC.NRedshift_bins);
    PC.HydrogenNumberDensity_bins = h5readdouble(All.TreeCoolFile, "HydrogenNumberDensity_bins", &PC.NHydrogenNumberDensity_bins);
    PC.Temperature_bins = h5readdouble(All.TreeCoolFile, "Temperature_bins", &PC.NTemperature_bins);
    PC.SpecInternalEnergy_bins = h5readdouble(All.TreeCoolFile, "SpecInternalEnergy_bins", &PC.NSpecInternalEnergy_bins);

    PC.nHp_table = h5readdouble(All.TreeCoolFile, "nHp", &size);
    PC.nHep_table = h5readdouble(All.TreeCoolFile, "nHep", &size);
    PC.nHepp_table = h5readdouble(All.TreeCoolFile, "nHepp", &size);
    PC.Lpnet_table = h5readdouble(All.TreeCoolFile, "NetCoolingRate", &size);
    PC.T_table = h5readdouble(All.TreeCoolFile, "EquilibriumTemperature", &size);


    /* 
     *
     * initialize the temperature / spec internal energy look up table,
     * This table is based on Primordial cooling and UVB. Metal is ignored, as
     * argued in Vogelsberger et al  2013 Arepo paper.
     *
     * */
    {

    int dimsT[] = {PC.NRedshift_bins, PC.NHydrogenNumberDensity_bins, PC.NSpecInternalEnergy_bins};
    interp_init(&PC.interpT, 3, dimsT);
    interp_init_dim(&PC.interpT, 0, PC.Redshift_bins[0], PC.Redshift_bins[PC.NRedshift_bins - 1]);
    interp_init_dim(&PC.interpT, 1, PC.HydrogenNumberDensity_bins[0], 
                    PC.HydrogenNumberDensity_bins[PC.NHydrogenNumberDensity_bins - 1]);
    interp_init_dim(&PC.interpT, 2, PC.SpecInternalEnergy_bins[0], 
                    PC.SpecInternalEnergy_bins[PC.NSpecInternalEnergy_bins - 1]);

    }

    printf("z = %g log nH = %g log U = %g logT = %g\n",
            157., -0.1, 10.0,
            TableTemperature(157, 10.0, -0.1));


    /* initialize the primordial cooling table (H and He + UVB) */
    {
        int dims[] = {PC.NRedshift_bins, PC.NHydrogenNumberDensity_bins, PC.NTemperature_bins};

        interp_init(&PC.interp, 3, dims);
        interp_init_dim(&PC.interp, 0, PC.Redshift_bins[0], PC.Redshift_bins[PC.NRedshift_bins - 1]);
        interp_init_dim(&PC.interp, 1, PC.HydrogenNumberDensity_bins[0], 
                        PC.HydrogenNumberDensity_bins[PC.NHydrogenNumberDensity_bins - 1]);
        interp_init_dim(&PC.interp, 2, PC.Temperature_bins[0], 
                        PC.Temperature_bins[PC.NTemperature_bins - 1]);
    }

    /* now initialize the metal cooling table from cloudy; we got this file
     * from vogelsberger's Arepo simulations; it is supposed to be 
     * cloudy + UVB - H and He; look so.
     * the table contains only 1 Z_sun values. Need to be scaled to the 
     * metallicity.
     *
     * */
            /* let's see if the Metal Cool File is magic NoMetal */ 
    if(!strcmp(All.MetalCoolFile, "NoMetal")) {
        CoolingNoMetal = 0;
    } else {
        CoolingNoMetal = 1;
        
        double * tabbedmet = h5readdouble(All.MetalCoolFile, "MetallicityInSolar_bins", &size);

        if(ThisTask == 0 && size != 1 || tabbedmet[0] != 0.0) {
            fprintf(stderr, "MetalCool file %s is wrongly tabulated\n", All.MetalCoolFile);
            endrun(124214);
        }
        
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

    if(ThisTask == 0) {
        double z, T = 5, n = -6;
        for(z = 0.0; z < 12.0; z += 1.0) {
            printf("z = %g log nH = %g log T = %g Lmet = %g Lpnet = %g\n",
                    z, n, T,
                    TableMetalCoolingRate(z, T, n),
                    TableCoolingRate(z, T, n)
                    );
        }
    }
}

static void TableAbundance(double redshift, double logT, double lognH, struct abundance * y) {
    if(ZeroIonizationFlag) redshift = -1;

    double x[] = {redshift, lognH, logT};
    int status[3];
    y->nHp = interp_eval(&PC.interp, x, PC.nHp_table, status);
    if(status[2] > 0) {
        /* very hot: H and He both fully ionized */
        y->nHp = 1.0;
        y->nHep = 0.0;
        y->nHepp = yhelium;
    } else if(status[2] < 0) {
        /* very cold: H and He both fully neutral */
        y->nHp = 0.0;
        y->nHep = 0.0;
        y->nHepp = 0.0;
    } else {
        y->nHep = interp_eval(&PC.interp, x, PC.nHep_table, status);
        y->nHepp = interp_eval(&PC.interp, x, PC.nHepp_table, status);
    }

    y->nH0 = 1.0 - y->nHp;
    y->nHe0 = 1.0 - (y->nHepp + y->nHep);
    y->ne = y->nHp + y->nHep + 2.0 * y->nHepp;
}

/* the table stores net cooling rate of primoridal H + He + UVB heating. */
static double TableCoolingRate(double redshift, double logT, double lognH) {
    if(ZeroIonizationFlag) redshift = -1;

    double x[] = {redshift, lognH, logT};
    int status[3];
    double rate = interp_eval(&PC.interp, x, PC.Lpnet_table, status);
    if(status[2] > 0) {
        /* very hot: H and He both fully ionized */
        double T = pow(10.0, logT);
        double nH = pow(10.0, lognH);
        double ne = 1 + 2 * yhelium;
        double LambdaFF = 1.42e-27 * sqrt(T) * (1.1 + 0.34 * exp(-(5.5 - logT) * (5.5 - logT) / 3)) * (1.0 + 4 * yhelium) * ne;

            /* add inverse Compton cooling off the microwave background */
        double LambdaCmptn = 5.65e-36 * ne * (T - 2.73 * (1. + redshift)) * pow(1. + redshift, 4.) / nH;
        rate = LambdaFF + LambdaCmptn;
    } else {
        /* very cold: H and He both fully ionized */
        /* simply use the lowest temperature rates */
    }
    return rate;
}


/* the table stores net cooling rate of Metals emerged in the same UVB at 1 * Z_sun. */
static double TableMetalCoolingRate(double redshift, double logT, double lognH) {
    double x[] = {redshift, lognH, logT};
    int status[3];
    double rate = interp_eval(&MC.interp, x, MC.Lmet_table, status);
    /* XXX: in case of very hot / very dense we just use whatever the table says at
     * the limit. should be OK. */
    return rate;
}

/* this function determines the electron fraction, and hence the mean 
 * molecular weight. With it arrives at a self-consistent temperature.
 *
 * This is precalculated from MakePrimodialCoolingTable.c
 *
 * returns log T/ K.
 */

static double TableTemperature(double redshift, double logU, double lognH) {
    if(ZeroIonizationFlag) redshift = -1;

    double x[] = {redshift, lognH, logU};
    int status[3];
    double logT = interp_eval(&PC.interpT, x, PC.T_table, status);
    if(status[2] > 0) {
        /* very hot: H and He both fully ionized */
        double U = pow(10.0, logU);
        double mu = (1 + 4 * yhelium) / (1 + yhelium + 1 + 2 * yhelium);
        logT = log10(GAMMA_MINUS1 / BOLTZMANN * U * PROTONMASS * mu);
    } else if(status[2] < 0) {
        /* very cold neutral */
        double U = pow(10.0, logU);
        double mu = (4.0) / (1 + 3 * XH);
        logT = log10(GAMMA_MINUS1 / BOLTZMANN * U * PROTONMASS * mu);
    }
    return logT;
}


/* calculate cooling rate from u
 * rho shall be in cgs.
 *
 * u is in cgs
 * u will be converted to T first
 *
 * This is different from old GADGET. 
 *
 * */
static double HeatingRateU(double redshift, double u, double lognH, double Z) {

    double logT = TableTemperature(redshift, log10(u), lognH);
    double rate = - TableCoolingRate(redshift, logT, lognH);

    if (Z != 0 && !CoolingNoMetal ) {
        rate -= Z * TableCoolingRate(redshift, logT, lognH);
    }
    return rate;
}

/* returns abundance ratios
 * Arguments are passed in code units, density is proper density.
 *
 * the electron abundance is calculated too.
 *
 */
double AbundanceRatios(double u, double rho, double *ne, double *nH0, double *nHeII) {
    struct abundance y;
    double redshift = 0;
    if(All.ComovingIntegrationOn) {
        redshift = 1.0 / All.Time - 1;
    }

    rho *= All.UnitDensity_in_cgs * All.HubbleParam * All.HubbleParam;	/* convert to physical cgs units */
    u *= All.UnitPressure_in_cgs / All.UnitDensity_in_cgs;

    double nH = XH * rho / PROTONMASS;	/* hydrogen number dens in cgs units */

    double lognH = log10(nH);

    double logT = TableTemperature(redshift, log10(u), lognH);
    TableAbundance(redshift, logT, lognH, &y);

    *ne= y.ne;
    *nH0 = y.nH0;
    *nHeII = y.nHepp;

    return pow(10., logT); 
}


/* returns new internal energy per unit mass. 
 * Arguments are passed in code units, density is proper density.
 *
 * the electron abundance is calculated too.
 *
 */
double DoCooling(double u_old, double rho, double dt, double *ne_guess, double Z)
{
    double u, du;
    double u_lower, u_upper;
    double LambdaNet;

    double redshift = 0;

    double DoCool_u_old_input, DoCool_rho_input, DoCool_dt_input, DoCool_ne_guess_input;
    if(All.ComovingIntegrationOn) {
        redshift = 1.0 / All.Time - 1;
    }

    int iter = 0;

    DoCool_u_old_input = u_old;
    DoCool_rho_input = rho;
    DoCool_dt_input = dt;
    DoCool_ne_guess_input = *ne_guess;


    rho *= All.UnitDensity_in_cgs * All.HubbleParam * All.HubbleParam;	/* convert to physical cgs units */
    u_old *= All.UnitPressure_in_cgs / All.UnitDensity_in_cgs;
    dt *= All.UnitTime_in_s / All.HubbleParam;

    double nH = XH * rho / PROTONMASS;	/* hydrogen number dens in cgs units */
    double lognH = log10(nH);

    double ratefact = nH * nH / rho;


    u = u_old;
    u_lower = u;
    u_upper = u;

    /**/
    LambdaNet = HeatingRateU(redshift, u, lognH, Z);

    /* bracketing */

    if(u - u_old - ratefact * LambdaNet * dt < 0)	/* heating */
    {
        u_upper *= sqrt(1.1);
        u_lower /= sqrt(1.1);
            while(u_upper - u_old - ratefact * HeatingRateU(redshift, u_upper, lognH, Z) * dt < 0)
            {
                u_upper *= 1.1;
                u_lower *= 1.1;
            }

    }

    if(u - u_old - ratefact * LambdaNet * dt > 0)
    {
        u_lower /= sqrt(1.1);
        u_upper *= sqrt(1.1);
            while(u_lower - u_old - ratefact * HeatingRateU(redshift, u_lower, lognH, Z) * dt > 0)
            {
                u_upper /= 1.1;
                u_lower /= 1.1;
            }
    }

    do
    {
        u = 0.5 * (u_lower + u_upper);

        LambdaNet = HeatingRateU(redshift, u, lognH, Z);

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

    double junk;
    AbundanceRatios(u, rho, ne_guess, &junk, &junk);

    return u;
}

/* returns cooling time. 
 * NOTE: If we actually have heating, a cooling time of 0 is returned.
 */
double GetCoolingTime(double u_old, double rho, double *ne_guess, double Z)
{
    double u;
    double LambdaNet, coolingtime;

    double redshift = 0;
    if(All.ComovingIntegrationOn) {
        redshift = 1.0 / All.Time - 1;
    }


    double junk;
    /* u_old is still in internal units AbundanceRatios expects internal units */
    AbundanceRatios(u_old, rho, ne_guess, &junk, &junk);

    rho *= All.UnitDensity_in_cgs * All.HubbleParam * All.HubbleParam;	/* convert to physical cgs units */
    u_old *= All.UnitPressure_in_cgs / All.UnitDensity_in_cgs;


    double nH = XH * rho / PROTONMASS;	/* hydrogen number dens in cgs units */
    double ratefact = nH * nH / rho;
    double lognH = log10(nH);

    u = u_old;

    LambdaNet = HeatingRateU(redshift, u, lognH, Z);

    /* bracketing */

    if(LambdaNet >= 0)		/* ups, we have actually heating due to UV background */
        return 0;

    coolingtime = u_old / (-ratefact * LambdaNet);

    coolingtime *= All.HubbleParam / All.UnitTime_in_s;

    return coolingtime;
}

void IonizeParams() {
    ZeroIonizationFlag = 0;
}
void SetZeroIonization() {
    ZeroIonizationFlag = 1;
}
#endif
