#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_integration.h>

#include "allvars.h"
#include "proto.h"


#ifdef BUBBLES

#ifdef EBUB_PROPTO_BHAR
/* Di Matteo et al. ApJ, 593, 2003; eq. [11] */

double rho_dot(double z, void *params)
{
  double a = 5. / 4.;
  double b = 3. / 2.;
  double zm = 4.8;
  double ebh = 3.0e-4;		/* [Msun/yr/Mpc^3] */
  double c1, c2, H0;
  double rho_dot;

  H0 = 100. * All.HubbleParam / CM_PER_MPC * 1.0e5 * SEC_PER_YEAR;	/* [yrs] */

  /* dt = -3/2 * c1*c2 * (1+z)^(-5/2) * 1/sqrt(1 + c2^2/(1+z)^3) dz */

  /* c1 = 2/3/H0/sqrt(1-AllOmega0) 
     c2 = sqrt((1-AllOmega0)/AllOmega0) */

  c1 = 2.0 / 3.0 / H0 / pow((1.0 - All.Omega0), 1. / 2.);
  c2 = pow(((1.0 - All.Omega0) / All.Omega0), 1. / 2.);

  rho_dot = ebh * (b * exp(a * (z - zm))) / (b - a + a * exp(b * (z - zm)));

  /* now times dt = (dt/dz)*dz */

  rho_dot = rho_dot * (3.0 / 2.0 * c1 * c2 / pow((1.0 + z), 5.0 / 2.0) /
		       pow((1.0 + c2 * c2 / pow((1.0 + z), 3.0)), 1. / 2.));

  return rho_dot;
}

double bhgrowth(double z1, double z2)
{
#define WORKSIZE 1000
  double result, abserr;
  gsl_function F;
  gsl_integration_workspace *workspace;

  workspace = gsl_integration_workspace_alloc(WORKSIZE);

  F.function = &rho_dot;

  gsl_integration_qag(&F, z2, z1, 0, 1.0e-4, WORKSIZE, GSL_INTEG_GAUSS41, workspace, &result, &abserr);

  gsl_integration_workspace_free(workspace);

  if(ThisTask == 0)
    printf("Integration result is: %g \n", result);

  return result;
}

#endif

void bubble(void)
{
  double phi, theta;
  double dx, dy, dz, rr, r2, dE;
  double E_bubble, totE_bubble, hubble_a;
  double BubbleDistance = 0.0, BubbleRadius = 0.0, BubbleEnergy = 0.0;
  double Mass_bubble, totMass_bubble;
  MyFloat pos[3];
  int numngb, tot_numngb, startnode, numngb_inbox;
  int n, i, j, dummy;

#ifdef EBUB_PROPTO_BHAR
  double c1, c2, c3, H0;
  double epsilon = 0.1, eff = 0.05, tsp = 4.5e7;
  double t1, t2;
  double z1, z2;
  double Enorm = 2455.156;
#endif

  if(ThisTask == 0)
    printf("Bubble radius: %g, Bubble distance: %g, Bubble energy: %g \n",
	   All.BubbleRadius, All.BubbleDistance, All.BubbleEnergy);

  if(All.ComovingIntegrationOn)
    {
      hubble_a = hubble_function(All.Time) / All.Hubble;
#ifdef FOF

#ifndef EBUB_PROPTO_BHAR

      /* distances and energy are riscaled according to the mass of the MP 
         in rispect to its final mass and according to the cosmology */

      BubbleDistance = All.BubbleDistance * 1. / All.Time *
	pow(All.BiggestGroupMass / 1.0e5, 1. / 3.) / pow(hubble_a, 2. / 3.);

      BubbleRadius = All.BubbleRadius * 1. / All.Time *
	pow(All.BiggestGroupMass / 1.0e5, 1. / 3.) / pow(hubble_a, 2. / 3.);

      BubbleEnergy = All.BubbleEnergy * pow(All.BiggestGroupMass / 1.0e5, 5. / 3.) * pow(hubble_a, 2. / 3.);
#else

      /* energy is scaled according to the BH growth with time. The total 
         energy injected is the same as in the "default" case (so the 
         normalization is fixed) and also the final BH mass is the same 
         (so the intergation constant of BHAR is fixed) */

      H0 = 100. * All.HubbleParam / CM_PER_MPC * 1.0e5 * SEC_PER_YEAR;	/* [yrs] */
      c1 = 2.0 / 3.0 / H0 / pow((1.0 - All.Omega0), 1. / 2.);
      c2 = pow(((1.0 - All.Omega0) / All.Omega0), 1. / 2.);
      c3 = c2 / pow((1.0 / All.Time), 3. / 2.);

      BubbleDistance = All.BubbleDistance * 1. / All.Time *
	pow(All.BiggestGroupMass / 1.0e5, 1. / 3.) / pow(hubble_a, 2. / 3.);

      BubbleRadius = All.BubbleRadius * 1. / All.Time *
	pow(All.BiggestGroupMass / 1.0e5, 1. / 3.) / pow(hubble_a, 2. / 3.);

      t1 = 2.0 / 3.0 / H0 / pow((1.0 - All.Omega0), 1. / 2.) * log(c3 + pow((c3 * c3 + 1.0), 1. / 2.));

      t2 = t1 + tsp;

      z1 = 1.0 / All.Time - 1.0;
      z2 = pow((c2 / sinh(3.0 * H0 * t2 * pow((1.0 - All.Omega0), 1. / 2.) / 2.)), 2.0 / 3.0) - 1.0;

      BubbleEnergy = eff * epsilon * pow(C, 2) * bhgrowth(z1, z2) * SOLAR_MASS * Enorm;

#endif
#endif
    }
  else
    {
      BubbleDistance = All.BubbleDistance;
      BubbleRadius = All.BubbleRadius;
      BubbleEnergy = All.BubbleEnergy;
    }

  if(ThisTask == 0)
    {
      printf("Bubble radius: %g, Bubble distance: %g, Bubble energy: %g \n",
	     BubbleRadius, BubbleDistance, BubbleEnergy);
#ifdef EBUB_PROPTO_BHAR
      printf("tini: %g, tfin: %g, zini: %g, zfin: %g \n", t1, t2, z1, z2);
#endif
    }

  phi = 2 * M_PI * get_random_number(0);
  theta = acos(2 * get_random_number(0) - 1);
  rr = pow(get_random_number(0), 1. / 3.) * BubbleDistance;

  pos[0] = sin(theta) * cos(phi);
  pos[1] = sin(theta) * sin(phi);
  pos[2] = cos(theta);

  for(i = 0; i < 3; i++)
    pos[i] *= rr;

  if(ThisTask == 0)
    printf("Make Bubble! (%g|%g|%g)\n", pos[0], pos[1], pos[2]);


#ifdef FOF
  for(i = 0; i < 3; i++)
    pos[i] += All.BiggestGroupCM[i];

  if(ThisTask == 0)
    {
      printf("CM of biggest group! (%g|%g|%g)\n",
	     All.BiggestGroupCM[0], All.BiggestGroupCM[1], All.BiggestGroupCM[2]);
      printf("Make Bubble! (%g|%g|%g)\n", pos[0], pos[1], pos[2]);
    }
#endif


  Ngblist = (int *) mymalloc("Ngblist", NumPart * sizeof(int));

  /* First, let's see how many particles are in the bubble */

  numngb = 0;
  E_bubble = 0.;
  Mass_bubble = 0.;

  startnode = All.MaxPart;
  do
    {
      numngb_inbox = ngb_treefind_variable(pos, BubbleRadius, -1, &startnode, 0, &dummy, &dummy);

      for(n = 0; n < numngb_inbox; n++)
	{
	  j = Ngblist[n];
	  dx = pos[0] - P[j].Pos[0];
	  dy = pos[1] - P[j].Pos[1];
	  dz = pos[2] - P[j].Pos[2];

#ifdef PERIODIC			/*  now find the closest image in the given box size  */
	  if(dx > boxHalf_X)
	    dx -= boxSize_X;
	  if(dx < -boxHalf_X)
	    dx += boxSize_X;
	  if(dy > boxHalf_Y)
	    dy -= boxSize_Y;
	  if(dy < -boxHalf_Y)
	    dy += boxSize_Y;
	  if(dz > boxHalf_Z)
	    dz -= boxSize_Z;
	  if(dz < -boxHalf_Z)
	    dz += boxSize_Z;
#endif
	  r2 = dx * dx + dy * dy + dz * dz;

	  if(r2 < BubbleRadius * BubbleRadius && P[j].Type == 0)
	    {
	      numngb++;

	      if(All.ComovingIntegrationOn)
		E_bubble +=
		  SphP[j].Entropy * P[j].Mass * pow(SphP[j].d.Density / pow(All.Time, 3),
						    GAMMA_MINUS1) / GAMMA_MINUS1;
	      else
		E_bubble += SphP[j].Entropy * P[j].Mass * pow(SphP[j].d.Density, GAMMA_MINUS1) / GAMMA_MINUS1;

	      Mass_bubble += P[j].Mass;

	    }
	}
    }
  while(startnode >= 0);


  MPI_Allreduce(&numngb, &tot_numngb, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&E_bubble, &totE_bubble, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&Mass_bubble, &totMass_bubble, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  if(ThisTask == 0)
    {
      if(tot_numngb == 0)
	printf("no particles in bubble found!\n");
      
      if(totE_bubble == 0)
	printf("Bubble has no energy!\n");
      
      fflush(stdout);
    }
  
  totE_bubble *= All.UnitEnergy_in_cgs;
  

  if(ThisTask == 0)
    {
      printf("found %d particles in bubble with energy %g and total mass %g \n", tot_numngb, totE_bubble,
	     totMass_bubble);
      printf("energy shall be increased by: (Eini+Einj)/Eini = %g \n",
	     (BubbleEnergy + totE_bubble) / totE_bubble);
      fflush(stdout);
    }

  /* now find particles in Bubble again, and inject energy */

  startnode = All.MaxPart;

  do
    {
      numngb_inbox = ngb_treefind_variable(pos, BubbleRadius, -1, &startnode, 0, &dummy, &dummy);

      for(n = 0; n < numngb_inbox; n++)
	{
	  j = Ngblist[n];
	  dx = pos[0] - P[j].Pos[0];
	  dy = pos[1] - P[j].Pos[1];
	  dz = pos[2] - P[j].Pos[2];

#ifdef PERIODIC			/*  now find the closest image in the given box size  */
	  if(dx > boxHalf_X)
	    dx -= boxSize_X;
	  if(dx < -boxHalf_X)
	    dx += boxSize_X;
	  if(dy > boxHalf_Y)
	    dy -= boxSize_Y;
	  if(dy < -boxHalf_Y)
	    dy += boxSize_Y;
	  if(dz > boxHalf_Z)
	    dz -= boxSize_Z;
	  if(dz < -boxHalf_Z)
	    dz += boxSize_Z;
#endif
	  r2 = dx * dx + dy * dy + dz * dz;

	  if(r2 < BubbleRadius * BubbleRadius && P[j].Type == 0 && totMass_bubble > 0)
	    {
	      /* with sf on gas particles have mass that is not fixed */

	      if(All.StarformationOn)
		dE = ((BubbleEnergy / All.UnitEnergy_in_cgs) / totMass_bubble) * P[j].Mass;
	      else
		dE = (BubbleEnergy / All.UnitEnergy_in_cgs) / tot_numngb;

	      /* energy we want to inject in this particle */
	      
	      if(All.ComovingIntegrationOn)
		SphP[j].Entropy +=
		  GAMMA_MINUS1 * dE / P[j].Mass / pow(SphP[j].d.Density / pow(All.Time, 3), GAMMA_MINUS1);
	      else
		SphP[j].Entropy += GAMMA_MINUS1 * dE / P[j].Mass / pow(SphP[j].d.Density, GAMMA_MINUS1);
	      
	    }
	}
    }
  while(startnode >= 0);
  
  myfree(Ngblist);
}

#endif
