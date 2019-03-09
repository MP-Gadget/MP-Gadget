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

#include "utils/endrun.h"
#include "utils/mymalloc.h"
#include "utils/interp.h"
#include "physconst.h"
#include "cooling.h"
#include "cooling_rates.h"

/*Metal cooling functions*/
static void InitMetalCooling(const char * MetalCoolFile);
static double TableMetalCoolingRate(double redshift, double logT, double lognH);

static struct cooling_units coolunits;

/*Do initialisation for the cooling module*/
void InitCool(const char * TreeCoolFile, const char * MetalCoolFile, const char * UVFluctuationFile, struct cooling_units cu, struct cooling_params coolpar)
{
    coolunits = cu;
    /* The table will be initialized to z=0 */
    if(!coolunits.CoolingOn) {
        coolunits.CoolingNoMetal = 1;
        return;
    }

    if(strlen(TreeCoolFile) == 0) {
        coolunits.CoolingOn = 0;
        coolpar.PhotoIonizationOn = 0;
        message(0, "No TreeCool file is provided. Cooling is broken. OK for DM only runs. \n");
    }

    init_cooling_rates(TreeCoolFile, coolpar);

    /* now initialize the metal cooling table from cloudy; we got this file
     * from vogelsberger's Arepo simulations; it is supposed to be
     * cloudy + UVB - H and He; look so.
     * the table contains only 1 Z_sun values. Need to be scaled to the
     * metallicity.
     *
     * */
            /* let's see if the Metal Cool File is magic NoMetal */
    if(strlen(MetalCoolFile) == 0) {
        coolunits.CoolingNoMetal = 1;
    } else {
        coolunits.CoolingNoMetal = 0;
        InitMetalCooling(MetalCoolFile);
    }

    init_uvf_table(UVFluctuationFile, coolunits.UVRedshiftThreshold);
}


/*  this function sums the cooling and heating rate from primordial and metal cooling, returning
 *  (heating rate-cooling rate)/n_h^2 in cgs units and setting ne_guess to the new electron temperature.
 */
static double
CoolingRateFromU(double redshift, double u, double nHcgs, struct UVBG * uvbg, double *ne_guess, double Z)
{
    if(!coolunits.CoolingOn) return 0;

    double LambdaNet = get_heatingcooling_rate(nHcgs, u, 1 - HYDROGEN_MASSFRAC, redshift, uvbg, ne_guess);

    if(! coolunits.CoolingNoMetal) {
        double lognH = log10(nHcgs);
        double temp = get_temp(nHcgs, u, 1- HYDROGEN_MASSFRAC, redshift, uvbg);
        double logT = log10(temp);
        LambdaNet -= Z * TableMetalCoolingRate(redshift, logT, lognH);
    }
    return LambdaNet;
}

#define MAXITER 1000

/* returns new internal energy per unit mass.
 * Arguments are passed in code units, density is proper density.
 */
double DoCooling(double redshift, double u_old, double rho, double dt, struct UVBG * uvbg, double *ne_guess, double Z)
{
    if(!coolunits.CoolingOn) return 0;

    double u, du;
    double u_lower, u_upper;
    double LambdaNet;
    int iter = 0;

    rho *= coolunits.density_in_phys_cgs;	/* convert to physical cgs units */
    u_old *= coolunits.uu_in_cgs;
    dt *= coolunits.tt_in_s;

    double nHcgs = HYDROGEN_MASSFRAC * rho / PROTONMASS;	/* hydrogen number dens in cgs units */
    double ratefact = nHcgs * nHcgs / rho;

    u = u_old;
    u_lower = u;
    u_upper = u;

    LambdaNet = CoolingRateFromU(redshift, u, nHcgs, uvbg, ne_guess, Z);

    /* bracketing */

    if(u - u_old - ratefact * LambdaNet * dt < 0)	/* heating */
    {
        u_upper *= sqrt(1.1);
        u_lower /= sqrt(1.1);
            while(u_upper - u_old - ratefact * CoolingRateFromU(redshift, u_upper, nHcgs, uvbg, ne_guess, Z) * dt < 0)
            {
                u_upper *= 1.1;
                u_lower *= 1.1;
            }

    }

    if(u - u_old - ratefact * LambdaNet * dt > 0)
    {
        u_lower /= sqrt(1.1);
        u_upper *= sqrt(1.1);
            while(u_lower - u_old - ratefact * CoolingRateFromU(redshift, u_lower, nHcgs, uvbg, ne_guess, Z) * dt > 0)
            {
                u_upper /= 1.1;
                u_lower /= 1.1;
            }
    }

    do
    {
        u = 0.5 * (u_lower + u_upper);

        LambdaNet = CoolingRateFromU(redshift, u, nHcgs, uvbg, ne_guess, Z);

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

    u /= coolunits.uu_in_cgs;   /*convert back to internal units */

    return u;
}

/* returns cooling time. 
 * NOTE: If we actually have heating, a cooling time of 0 is returned.
 */
double GetCoolingTime(double redshift, double u_old, double rho, struct UVBG * uvbg, double *ne_guess, double Z)
{
    if(!coolunits.CoolingOn) return 0;

    /* convert to physical cgs units */
    rho *= coolunits.density_in_phys_cgs;
    u_old *= coolunits.uu_in_cgs;

    /* hydrogen number dens in cgs units */
    const double nHcgs = HYDROGEN_MASSFRAC * rho / PROTONMASS;

    double LambdaNet = CoolingRateFromU(redshift, u_old, nHcgs, uvbg, ne_guess, Z);

    /* bracketing */

    if(LambdaNet >= 0)		/* ups, we have actually heating due to UV background */
        return 0;

    double ratefact = nHcgs * nHcgs / rho;
    double coolingtime = u_old / (-ratefact * LambdaNet);

    /*Convert back to internal units*/
    coolingtime /= coolunits.tt_in_s;

    return coolingtime;
}

/*Here comes the Metal Cooling code*/
struct {
    int NRedshift_bins;
    double * Redshift_bins;

    int NHydrogenNumberDensity_bins;
    double * HydrogenNumberDensity_bins;

    int NTemperature_bins;
    double * Temperature_bins;

    double * Lmet_table; /* metal cooling @ one solar metalicity*/

    Interp interp;
} MetalCool;


static void InitMetalCooling(const char * MetalCoolFile) {
    int size;
    //This is never used if MetalCoolFile == ""
    double * tabbedmet = read_big_array(MetalCoolFile, "MetallicityInSolar_bins", &size);

    if(size != 1 || tabbedmet[0] != 0.0) {
        endrun(123, "MetalCool file %s is wrongly tabulated\n", MetalCoolFile);
    }
    myfree(tabbedmet);
    
    MetalCool.Redshift_bins = read_big_array(MetalCoolFile, "Redshift_bins", &MetalCool.NRedshift_bins);
    MetalCool.HydrogenNumberDensity_bins = read_big_array(MetalCoolFile, "HydrogenNumberDensity_bins", &MetalCool.NHydrogenNumberDensity_bins);
    MetalCool.Temperature_bins = read_big_array(MetalCoolFile, "Temperature_bins", &MetalCool.NTemperature_bins);
    MetalCool.Lmet_table = read_big_array(MetalCoolFile, "NetCoolingRate", &size);

    int dims[] = {MetalCool.NRedshift_bins, MetalCool.NHydrogenNumberDensity_bins, MetalCool.NTemperature_bins};

    interp_init(&MetalCool.interp, 3, dims);
    interp_init_dim(&MetalCool.interp, 0, MetalCool.Redshift_bins[0], MetalCool.Redshift_bins[MetalCool.NRedshift_bins - 1]);
    interp_init_dim(&MetalCool.interp, 1, MetalCool.HydrogenNumberDensity_bins[0],
                    MetalCool.HydrogenNumberDensity_bins[MetalCool.NHydrogenNumberDensity_bins - 1]);
    interp_init_dim(&MetalCool.interp, 2, MetalCool.Temperature_bins[0],
                    MetalCool.Temperature_bins[MetalCool.NTemperature_bins - 1]);
}

static double TableMetalCoolingRate(double redshift, double logT, double lognH) {
    double x[] = {redshift, lognH, logT};
    int status[3];
    double rate = interp_eval(&MetalCool.interp, x, MetalCool.Lmet_table, status);
    /* XXX: in case of very hot / very dense we just use whatever the table says at
     * the limit. should be OK. */
    return rate;
}
