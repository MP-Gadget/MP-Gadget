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

/*! Driver routine to call initializiation of periodic or/and non-periodic FFT
 *  routines.
 */
void long_range_init(void)
{
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

#ifdef NOGRAVITY
  return;
#endif
  gravpm_force();
}


