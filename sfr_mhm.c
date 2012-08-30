#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "allvars.h"
#include "proto.h"
#include "forcetree.h"

#ifdef MHM

void cooling_and_starformation(void)	/* cooling routine when star formation is enabled */
{
  int i, stars_spawned, tot_spawned, stars_converted, tot_converted;
  unsigned int gen, bits;
  double dt, dtime, a3inv, tdyn, rateOfSF;
  double hubble_a, time_hubble_a;
  double sum_sm, total_sm, sm, rate, sum_mass_stars, total_sum_mass_stars;
  double p, prob;
  double rate_in_msunperyear;
  double sfrrate, totsfrrate;


  if(All.ComovingIntegrationOn)
    {
      a3inv = 1 / (All.Time * All.Time * All.Time);
      hubble_a = hubble_function(All.Time);
      time_hubble_a = All.Time * hubble_a;
    }
  else
    a3inv = hubble_a = time_hubble_a = 1;



  stars_spawned = stars_converted = 0;
  sum_sm = sum_mass_stars = 0;

  for(bits = 0; GENERATIONS > (1 << bits); bits++);

  for(i = 0; i < N_gas; i++)
    {
      if(P[i].Type == 0)
	if(P[i].Ti_endstep == All.Ti_Current)
	  {
	    dt = (P[i].Ti_endstep - P[i].Ti_begstep) * All.Timebase_interval;
	    /*  the actual time-step */

	    if(All.ComovingIntegrationOn)
	      dtime = All.Time * dt / time_hubble_a;
	    else
	      dtime = dt;


	    tdyn = 1 / sqrt(4 * M_PI * All.G * SphP[i].Density * a3inv);

	    rateOfSF = All.WindEfficiency * P[i].Mass / tdyn;

	    sm = rateOfSF * dtime;	/* amount of stars expect to form */

	    SphP[i].FeedbackEnergy = All.WindEnergyFraction * sm * All.FeedbackEnergy;

	    p = sm / P[i].Mass;

	    sum_sm += P[i].Mass * (1 - exp(-p));


	    if(bits == 0)
	      number_of_stars_generated = 0;
	    else
	      number_of_stars_generated = (P[i].ID >> (sizeof(MyIDType)*8 - bits));

	    mass_of_star = P[i].Mass / (GENERATIONS - number_of_stars_generated);


	    SphP[i].Sfr = rateOfSF * (All.UnitMass_in_g / SOLAR_MASS) / (All.UnitTime_in_s / SEC_PER_YEAR);

	    prob = P[i].Mass / mass_of_star * (1 - exp(-p));

	    if(get_random_number(P[i].ID + 1) < prob)	/* ok, make a star */
	      {
		if(number_of_stars_generated == (GENERATIONS - 1))
		  {
		    /* here we turn the gas particle itself into a star */
		    Stars_converted++;
		    stars_converted++;

		    sum_mass_stars += P[i].Mass;

		    P[i].Type = 4;
#ifdef STELLARAGE
		    P[i].StellarAge = All.Time;
#endif
		  }
		else
		  {
		    /* here we spawn a new star particle */
		    gen = (int) (1.1 * P[i].Mass / (All.OrigGasMass / GENERATIONS)) - 1;   //do we need this ?
		    
		    for(bits = 0; GENERATIONS > (1 << bits); bits++);

		    gen <<= (32 - bits);

		    if(NumPart + stars_spawned >= All.MaxPart)
		      {
			printf
			  ("On Task=%d with NumPart=%d we try to spawn %d particles. Sorry, no space left...(All.MaxPart=%d)\n",
			   ThisTask, NumPart, stars_spawned, All.MaxPart);
			fflush(stdout);
			endrun(8888);
		      }

		    P[NumPart + stars_spawned] = P[i];
		    P[NumPart + stars_spawned].Type = 4;

		    P[i].ID += ((MyIDType) 1 << (sizeof(MyIDType)*8 - bits));

		    P[NumPart + stars_spawned].Mass = mass_of_star;
		    P[i].Mass -= P[NumPart + stars_spawned].Mass;
		    sum_mass_stars += P[NumPart + stars_spawned].Mass;
#ifdef STELLARAGE
		    P[NumPart + stars_spawned].StellarAge = All.Time;
#endif
		    force_add_star_to_tree(i, NumPart + stars_spawned);

		    stars_spawned++;
		  }
	      }
	  }

    }				/* end of main loop over active particles */


  MPI_Allreduce(&stars_spawned, &tot_spawned, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&stars_converted, &tot_converted, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  if(tot_spawned > 0 || tot_converted > 0)
    {
      if(ThisTask == 0)
	{
	  printf("\n----> spawned %d stars, converted %d gas particles into stars\n\n",
		 tot_spawned, tot_converted);
	  fflush(stdout);
	}


      All.TotNumPart += tot_spawned;
      All.TotN_gas -= tot_converted;
      NumPart += stars_spawned;
      NumForceUpdate += stars_spawned;

      /* Note: N_gas is only reduced once rearrange_particle_sequence is called */
      /* Note: New tree construction can be avoided because of  `force_add_star_to_tree()' */
    }

  for(i = 0, sfrrate = 0; i < N_gas; i++)
    if(P[i].Type == 0)
      sfrrate += SphP[i].Sfr;

  MPI_Allreduce(&sfrrate, &totsfrrate, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  MPI_Reduce(&sum_sm, &total_sm, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&sum_mass_stars, &total_sum_mass_stars, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  if(ThisTask == 0)
    {
      if(All.TimeStep > 0)
	rate = total_sm / (All.TimeStep / time_hubble_a);
      else
	rate = 0;

      /* convert to solar masses per yr */

      rate_in_msunperyear = rate * (All.UnitMass_in_g / SOLAR_MASS) / (All.UnitTime_in_s / SEC_PER_YEAR);

      fprintf(FdSfr, "%g %g %g %g %g\n", All.Time, total_sm, totsfrrate, rate_in_msunperyear,
	      total_sum_mass_stars);
      fflush(FdSfr);
    }
}


double get_starformation_rate(int i)
{
  double rateOfSF;
  double a3inv;
  double tdyn;


  if(All.ComovingIntegrationOn)
    a3inv = 1 / (All.Time * All.Time * All.Time);
  else
    a3inv = 1;

  tdyn = 1 / sqrt(4 * M_PI * All.G * SphP[i].Density * a3inv);

  rateOfSF = All.MaxSfrTimescale * P[i].Mass / tdyn;

  /* convert to solar masses per yr */

  rateOfSF *= (All.UnitMass_in_g / SOLAR_MASS) / (All.UnitTime_in_s / SEC_PER_YEAR);

  return rateOfSF;
}

#endif
