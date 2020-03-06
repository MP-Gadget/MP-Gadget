/**************
 * This file is based on the Gadget-3 cooling.c.
 *
 * The main atomic rates have been rewritten from scratch and now exist in cooling_rates.c.
 * There is also support for metal cooling and a fluctuating UV background in cooling_uvfluc.c
 *
 * There is no public version of original cooling.c freely available.
 *
 * Only reference is 2004 version by Brian O'Shea in ENZO repo,
 * which contains a deferred freedom claimer depending on the action
 * of Volker Springel.
 *
 * The LICENSE of the DoCooling function is therefore still in limbo,
 * although everything else is now unencumbered.
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
#include "cooling_qso_lightup.h"

static struct cooling_units coolunits;

/*Do initialisation for the cooling module*/
void init_cooling(char * TreeCoolFile, char * MetalCoolFile, char * reion_hist_file, struct cooling_units cu, Cosmology * CP)
{
    coolunits = cu;

    /*Initialize the cooling rates*/
    init_cooling_rates(TreeCoolFile, MetalCoolFile, CP);
    /* Initialize the helium reionization model*/
    init_qso_lightup(reion_hist_file);
}

#define MAXITER 1000

/* Wrapper function which returns the rate of change of internal energy in units of
 * erg/s/g. Arguments:
 * rho: density in protons/cm^3 (physical)
 * u: internal energy in units of erg/g
 * Z: metallicity
 * redshift: redshift
 * isHeIIIionized: flags whether the particle has been HeII reionized.
 */
static double
get_lambdanet(double rho, double u, double redshift, double Z, struct UVBG * uvbg, double * ne_guess, int isHeIIIionized)
{
    double LambdaNet = get_heatingcooling_rate(rho, u, 1 - HYDROGEN_MASSFRAC, redshift, Z, uvbg, ne_guess);
    if(!isHeIIIionized) {
        /* get_long_mean_free_path_heating returns the heating in units of erg/s/cm^3,
         * the factor of rho converts to erg/s/proton and then PROTONMASS to erg/s/g */
        LambdaNet += get_long_mean_free_path_heating(redshift)  / (rho  * PROTONMASS);
    }
    return LambdaNet;
}

/* returns new internal energy per unit mass.
 * Arguments are passed in code units, density is proper density.
 */
double DoCooling(double redshift, double u_old, double rho, double dt, struct UVBG * uvbg, double *ne_guess, double Z, double MinEgySpec, int isHeIIIionized)
{
    if(!coolunits.CoolingOn) return 0;

    double u, du;
    double u_lower, u_upper;
    double LambdaNet;
    int iter = 0;

    rho *= coolunits.density_in_phys_cgs / PROTONMASS;	/* convert to (physical) protons/cm^3 */
    u_old *= coolunits.uu_in_cgs;
    MinEgySpec *= coolunits.uu_in_cgs;
    if(u_old < MinEgySpec)
        u_old = MinEgySpec;
    dt *= coolunits.tt_in_s;

    u = u_old;
    u_lower = u;
    u_upper = u;

    LambdaNet = get_lambdanet(rho, u, redshift, Z, uvbg, ne_guess, isHeIIIionized);

    /* bracketing */
    if(u - u_old - LambdaNet * dt < 0)	/* heating */
    {
        do
        {
            u_lower = u_upper;
            u_upper *= 1.1;
        } while(u_upper - u_old - get_lambdanet(rho, u_upper, redshift, Z, uvbg, ne_guess, isHeIIIionized) * dt < 0);
    }
    else
    {
        do {
            u_upper = u_lower;
            u_lower /= 1.1;
            /* This means that we don't need an initial bracket*/
            if(u_upper <= MinEgySpec) {
                break;
            }
        } while(u_lower - u_old - get_lambdanet(rho, u_lower, redshift, Z, uvbg, ne_guess, isHeIIIionized) * dt > 0);
    }

    do
    {
        u = 0.5 * (u_lower + u_upper);
        /* If we know that the new energy
         * is below the minimum gas internal energy, we are done here.*/
        if(u_upper <= MinEgySpec) {
            u = MinEgySpec;
            break;
        }

        LambdaNet = get_lambdanet(rho, u, redshift, Z, uvbg, ne_guess, isHeIIIionized);

        if(u - u_old - LambdaNet * dt > 0)
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
    rho *= coolunits.density_in_phys_cgs / PROTONMASS;
    u_old *= coolunits.uu_in_cgs;

    /* Note: this does not include the long mean free path heating from helium reionization*/
    double LambdaNet = get_heatingcooling_rate(rho, u_old, 1 - HYDROGEN_MASSFRAC, redshift, Z, uvbg, ne_guess);

    if(LambdaNet >= 0)		/* ups, we have actually heating due to UV background */
        return 0;

    double coolingtime = u_old / (- LambdaNet);

    /*Convert back to internal units*/
    coolingtime /= coolunits.tt_in_s;

    return coolingtime;
}

/*Gets the neutral fraction from density and internal energy in internal units*/
double
GetNeutralFraction(double u_old, double rho, const struct UVBG * uvbg, double ne_init)
{
    /* convert to physical cgs units */
    rho *= coolunits.density_in_phys_cgs / PROTONMASS;
    u_old *= coolunits.uu_in_cgs;
    double nh0 = get_neutral_fraction_phys_cgs(rho, u_old, 1 - HYDROGEN_MASSFRAC, uvbg, &ne_init);
    return nh0;
}

/*Gets the helium ion fraction from density and internal energy in internal units*/
double
GetHeliumIonFraction(int ion, double u_old, double rho, const struct UVBG * uvbg, double ne_init)
{
    /* convert to physical cgs units */
    rho *= coolunits.density_in_phys_cgs / PROTONMASS;
    u_old *= coolunits.uu_in_cgs;
    double helium_ion = get_helium_ion_phys_cgs(ion, rho, u_old, 1 - HYDROGEN_MASSFRAC, uvbg, ne_init);
    return helium_ion;
}
