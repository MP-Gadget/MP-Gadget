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

static struct cooling_units coolunits;

/*Do initialisation for the cooling module*/
void init_cooling(char * TreeCoolFile, char * MetalCoolFile, char * UVFluctuationFile, struct cooling_units cu, Cosmology * CP)
{
    coolunits = cu;

    /*Initialize the cooling rates*/
    init_cooling_rates(TreeCoolFile, MetalCoolFile, CP);
    /*Initialize the uv fluctuation table*/
    init_uvf_table(UVFluctuationFile);
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

    rho *= coolunits.density_in_phys_cgs / PROTONMASS;	/* convert to (physical) protons/cm^3 */
    u_old *= coolunits.uu_in_cgs;
    dt *= coolunits.tt_in_s;

    u = u_old;
    u_lower = u;
    u_upper = u;

    LambdaNet = get_heatingcooling_rate(rho, u, 1 - HYDROGEN_MASSFRAC, redshift, Z, uvbg, ne_guess);

    /* bracketing */

    if(u - u_old - LambdaNet * dt < 0)	/* heating */
    {
        u_upper *= sqrt(1.1);
        u_lower /= sqrt(1.1);
            while(u_upper - u_old - get_heatingcooling_rate(rho, u_upper, 1 - HYDROGEN_MASSFRAC, redshift, Z, uvbg, ne_guess) * dt < 0)
            {
                u_upper *= 1.1;
                u_lower *= 1.1;
            }

    }

    if(u - u_old - LambdaNet * dt > 0)
    {
        u_lower /= sqrt(1.1);
        u_upper *= sqrt(1.1);
            while(u_lower - u_old - get_heatingcooling_rate(rho, u_lower, 1 - HYDROGEN_MASSFRAC, redshift, Z, uvbg, ne_guess) * dt > 0)
            {
                u_upper /= 1.1;
                u_lower /= 1.1;
            }
    }

    do
    {
        u = 0.5 * (u_lower + u_upper);

        LambdaNet = get_heatingcooling_rate(rho, u, 1 - HYDROGEN_MASSFRAC, redshift, Z, uvbg, ne_guess);

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
