#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "allvars.h"
/*! \file longrange.c
 *  \brief driver routines for computation of long-range gravitational PM force
 */


/*Defined in gravpm.c and only used here*/
void  gravpm_init_periodic();

static void fill_ntab();

/*! Driver routine to call initializiation of periodic or/and non-periodic FFT
 *  routines.
 */
void grav_init(void)
{
    fill_ntab();
    gravpm_init_periodic();
}

/*! length of lock-up table for short-range force kernel in TreePM algorithm */
#define NTAB 1000
/*! variables for short-range lookup table */
static float shortrange_table[NTAB], shortrange_table_potential[NTAB], shortrange_table_tidal[NTAB];

static void
fill_ntab()
{
    int i;
    for(i = 0; i < NTAB; i++)
    {
        double u = 3.0 / NTAB * (i + 0.5);
        shortrange_table[i] = erfc(u) + 2.0 * u / sqrt(M_PI) * exp(-u * u);
        shortrange_table_potential[i] = erfc(u);
        shortrange_table_tidal[i] = 4.0 * u * u * u / sqrt(M_PI) * exp(-u * u);
    }
}

/* multiply force factor (*fac) and potential (*pot) by the shortrange force window function*/
int
grav_apply_short_range_window(double r, double * fac, double * pot)
{
    const double asmth = All.Asmth * All.BoxSize / All.Nmesh;
    const double asmthfac = 0.5 / asmth * (NTAB / 3.0);
    int tabindex = (int) (asmthfac * r);
    if(tabindex < NTAB)
    {
        *fac *= shortrange_table[tabindex];
        *pot *= shortrange_table_potential[tabindex];
        return 0;
    } else {
        return 1;
    }
}
