#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "allvars.h"
#include "proto.h"
/*! \file longrange.c
 *  \brief driver routines for computation of long-range gravitational PM force
 */


/*Defined in gravpm.c and only used here*/
void  gravpm_init_periodic();
void  gravpm_force();

static void fill_ntab();

/*! Driver routine to call initializiation of periodic or/and non-periodic FFT
 *  routines.
 */
void long_range_init(void)
{
    fill_ntab();
    gravpm_init_periodic();
}


/*! This function computes the long-range PM force for all particles.
 */
void long_range_force(void)
{
  int i;

  for(i = 0; i < NumPart; i++)
    {
      P[i].GravPM[0] = P[i].GravPM[1] = P[i].GravPM[2] = 0;
      P[i].PM_Potential = 0;

    }

  gravpm_force();
}

/*! This function sets the (comoving) softening length of all particle
 *  types in the table All.SofteningTable[...].  We check that the physical
 *  softening length is bounded by the Softening-MaxPhys values.
 */
void set_softenings(void)
{
    int i;

    if(All.SofteningGas * All.Time > All.SofteningGasMaxPhys)
        All.SofteningTable[0] = All.SofteningGasMaxPhys / All.Time;
    else
        All.SofteningTable[0] = All.SofteningGas;

    if(All.SofteningHalo * All.Time > All.SofteningHaloMaxPhys)
        All.SofteningTable[1] = All.SofteningHaloMaxPhys / All.Time;
    else
        All.SofteningTable[1] = All.SofteningHalo;

    if(All.SofteningDisk * All.Time > All.SofteningDiskMaxPhys)
        All.SofteningTable[2] = All.SofteningDiskMaxPhys / All.Time;
    else
        All.SofteningTable[2] = All.SofteningDisk;

    if(All.SofteningBulge * All.Time > All.SofteningBulgeMaxPhys)
        All.SofteningTable[3] = All.SofteningBulgeMaxPhys / All.Time;
    else
        All.SofteningTable[3] = All.SofteningBulge;

    if(All.SofteningStars * All.Time > All.SofteningStarsMaxPhys)
        All.SofteningTable[4] = All.SofteningStarsMaxPhys / All.Time;
    else
        All.SofteningTable[4] = All.SofteningStars;

    if(All.SofteningBndry * All.Time > All.SofteningBndryMaxPhys)
        All.SofteningTable[5] = All.SofteningBndryMaxPhys / All.Time;
    else
        All.SofteningTable[5] = All.SofteningBndry;

    for(i = 0; i < 6; i++)
        All.ForceSoftening[i] = 2.8 * All.SofteningTable[i];

    All.MinGasHsml = All.MinGasHsmlFractional * All.ForceSoftening[0];
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
    const double asmth = ASMTH * All.BoxSize / All.Nmesh;
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
