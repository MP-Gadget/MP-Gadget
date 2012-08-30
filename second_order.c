#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <gsl/gsl_math.h>

#include "allvars.h"
#include "proto.h"

/*! \file second_order.c
 *  \brief produce actual ICs from special 2nd order lpt ICs
 */



double F1_Omega(double a)
{
  double omega_a;

  omega_a = All.Omega0 / (All.Omega0 + a * (1 - All.Omega0 - All.OmegaLambda) + a * a * a * All.OmegaLambda);

  return pow(omega_a, 5.0 / 9);
}

double F2_Omega(double a)
{
  double omega_a;

  omega_a = All.Omega0 / (All.Omega0 + a * (1 - All.Omega0 - All.OmegaLambda) + a * a * a * All.OmegaLambda);

  return 2.0 * pow(omega_a, 6.0 / 11);
}


void second_order_ics(void)
{
  int i, j;
  double a, hubble, fac1, fac2, fac3;
  double ax, ay, az;

  if(ThisTask == 0)
    {
      printf("\nNow producing ICs based on 2nd-order LPT\n\n");
      fflush(stdout);
    }

  a = All.TimeBegin;

  hubble = hubble_function(a);

  fac1 = 1.0 / (a * a * hubble * F1_Omega(a));	/* this factor converts the Zeldovich peculiar velocity 
						   (expressed as  a^2*dX/dt  to comoving displacement */
  fac2 = header.lpt_scalingfactor;

  fac3 = fac2 * a * a * hubble * F2_Omega(a);

  if(ThisTask == 0)
    printf("fac1=%g  fac2=%g  fac3=%g\n", fac1, fac2, fac3);

  for(i = 0; i < NumPart; i++)
    {
      for(j = 0; j < 3; j++)
	P[i].Pos[j] += fac1 * P[i].Vel[j];	/* Zeldovich displacement */

#ifdef PMGRID
      for(j = 0; j < 3; j++)
	P[i].Pos[j] += fac2 * (P[i].GravPM[j] + P[i].g.GravAccel[j]);	/* second order lpt displacement */

      for(j = 0; j < 3; j++)
	P[i].Vel[j] += fac3 * (P[i].GravPM[j] + P[i].g.GravAccel[j]);	/* second order lpt velocity correction */
#else
      for(j = 0; j < 3; j++)
	P[i].Pos[j] += fac2 * (P[i].g.GravAccel[j]);	/* second order lpt displacement */

      for(j = 0; j < 3; j++)
	P[i].Vel[j] += fac3 * (P[i].g.GravAccel[j]);	/* second order lpt velocity correction */
#endif
    }


  /* now set the masses correctly */

  for(i = 0; i < NumPart; i++)
    {
      P[i].Mass = P[i].OldAcc;

      if(All.MassTable[P[i].Type] != 0)
	P[i].Mass = All.MassTable[P[i].Type];
    }

  if(All.ComovingIntegrationOn)
    if(All.PeriodicBoundariesOn == 1)
      check_omega();		/* check again whether we have plausible mass density */

  /* recompute long range force */

  All.DoDynamicUpdate = 1;
  domain_Decomposition();	/* redo domain decomposition because particles may have shifted */


  CPU_Step[CPU_MISC] += measure_time();

#ifdef PMGRID
  long_range_force();
#endif

  CPU_Step[CPU_MESH] += measure_time();

  gravity_tree();

  for(i = 0; i < NumPart; i++)
    {
#ifdef PMGRID
      ax = (P[i].g.GravAccel[0] + P[i].GravPM[0]) / All.G;
      ay = (P[i].g.GravAccel[1] + P[i].GravPM[1]) / All.G;
      az = (P[i].g.GravAccel[2] + P[i].GravPM[2]) / All.G;
#else
      ax = P[i].g.GravAccel[0] / All.G;
      ay = P[i].g.GravAccel[1] / All.G;
      az = P[i].g.GravAccel[2] / All.G;
#endif
      P[i].OldAcc = sqrt(ax * ax + ay * ay + az * az);
    }

  if(All.TypeOfOpeningCriterion == 1)
    {
      All.ErrTolTheta = 0;	/* This will switch to the relative opening criterion for the following force computations */
      gravity_tree();
    }

}
