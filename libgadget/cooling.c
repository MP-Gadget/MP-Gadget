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

static struct cooling_units coolunits;

/*Do initialisation for the cooling module*/
void init_cool_units(struct cooling_units cu)
{
    coolunits = cu;
}


/*  this function sums the cooling and heating rate from primordial and metal cooling, returning
 *  (heating rate-cooling rate)/n_h^2 in cgs units and setting ne_guess to the new electron temperature.
 */
static double
CoolingRateFromU(double redshift, double u, double nHcgs, struct UVBG * uvbg, double *ne_guess, double Z)
{
    if(!coolunits.CoolingOn) return 0;

    double LambdaNet = get_heatingcooling_rate(nHcgs, u, 1 - HYDROGEN_MASSFRAC, redshift, Z, uvbg, ne_guess);

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
