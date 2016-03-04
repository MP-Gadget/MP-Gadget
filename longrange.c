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

#ifdef PETAPM

/*Defined in gravpm.c and only used here*/
void  gravpm_init_periodic();
void  gravpm_force();

/*! Driver routine to call initializiation of periodic or/and non-periodic FFT
 *  routines.
 */
void long_range_init(void)
{
#ifdef PETAPM
  gravpm_init_periodic();
#endif /*PETAPM*/
}


void long_range_init_regionsize(void)
{
#ifdef PERIODIC
#ifdef PLACEHIGHRESREGION
  if(RestartFlag != 1)
    pm_init_regionsize();
  pm_setup_nonperiodic_kernel();
#endif
#else
  if(RestartFlag != 1)
    pm_init_regionsize();
  pm_setup_nonperiodic_kernel();
#endif
}


/*! This function computes the long-range PM force for all particles.
 */
void long_range_force(void)
{
  int i;

#ifndef PERIODIC
  int j;
  double fac;
#endif

  for(i = 0; i < NumPart; i++)
    {
      P[i].GravPM[0] = P[i].GravPM[1] = P[i].GravPM[2] = 0;
      P[i].PM_Potential = 0;

    }

#ifdef NOGRAVITY

  return;
#endif


#ifdef PERIODIC
#ifdef PETAPM
  gravpm_force();
#else
  do_box_wrapping();	/* map the particles back onto the box */
  pmforce_periodic(0, NULL);
#endif
#ifdef PLACEHIGHRESREGION
  i = pmforce_nonperiodic(1);
  if(i == 1)			/* this is returned if a particle lied outside allowed range */
    {
      pm_init_regionsize();
      pm_setup_nonperiodic_kernel();
      i = pmforce_nonperiodic(1);	/* try again */

    }
  if(i == 1)
    endrun(68686);
#endif
#else
  i = pmforce_nonperiodic(0);

  if(i == 1)			/* this is returned if a particle lied outside allowed range */
    {
      pm_init_regionsize();
      pm_setup_nonperiodic_kernel();
      i = pmforce_nonperiodic(0);	/* try again */
    }
  if(i == 1)
    endrun(68687);
#ifdef PLACEHIGHRESREGION
  i = pmforce_nonperiodic(1);
  if(i == 1)			/* this is returned if a particle lied outside allowed range */
    {
      pm_init_regionsize();
      pm_setup_nonperiodic_kernel();

      /* try again */

      for(i = 0; i < NumPart; i++)
	P[i].GravPM[0] = P[i].GravPM[1] = P[i].GravPM[2] = 0;

      i = pmforce_nonperiodic(0) + pmforce_nonperiodic(1);

    }
  if(i != 0)
    endrun(68688);
#endif
#endif


#ifndef PERIODIC
  if(All.ComovingIntegrationOn)
    {
      fac = 0.5 * All.Hubble * All.Hubble * All.Omega0;

      for(i = 0; i < NumPart; i++)
	for(j = 0; j < 3; j++)
	  P[i].GravPM[j] += fac * P[i].Pos[j];
    }


  /* Finally, the following factor allows a computation of cosmological simulation 
     with vacuum energy in physical coordinates */

  if(All.ComovingIntegrationOn == 0)
    {
      fac = All.OmegaLambda * All.Hubble * All.Hubble;

      for(i = 0; i < NumPart; i++)
	for(j = 0; j < 3; j++)
	  P[i].GravPM[j] += fac * P[i].Pos[j];
    }
#endif
#if 0
#ifdef PETAPM
  char * fnt = "longrange-peta-3.%d";
#else
  char * fnt = "longrange-pm.%d";
#endif
  char fn[1024];
  sprintf(fn, fnt, ThisTask);

  FILE * fp = fopen(fn, "w");
  double * buf = malloc(NumPart * sizeof(double) * 7);
  for(i = 0; i < NumPart; i ++) {
      buf[i * 7] = P[i].PM_Potential;
      int k;
      for(k = 0; k < 3; k ++) {
          buf[i * 7 + 1 + k] = P[i].GravPM[k];
      }
      for(k = 0; k < 3; k ++) {
          buf[i * 7 + 4 + k] = P[i].Pos[k];
      }
  }
  fwrite(buf, sizeof(double) * 7, NumPart, fp);
  fclose(fp);
  MPI_Barrier(MPI_COMM_WORLD);
  abort();
#endif
}


#endif
