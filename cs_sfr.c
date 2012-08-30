#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "allvars.h"
#include "proto.h"
#include "forcetree.h"

/*! \file cs_sfr.c 
 *  
 *  This file contains the star formation scheme for CS_MODEL
 */

#ifdef COOLING
#ifdef CS_MODEL

#include "cs_metals.h"

void cs_cooling_and_starformation(void)	/* cooling routine when star formation is enabled */
{
  int i, bin, flag, stars_spawned, tot_spawned, stars_converted, tot_converted, number_of_stars_generated,
    bits;
  double dt, dtime, ascale = 1, hubble_a = 0, a3inv, ne = 1;
  double time_hubble_a, unew, mass_of_star;
  double sum_sm, total_sm, sm, rate, rate2 = 0, sum_mass_stars, total_sum_mass_stars;
  double p, prob, dmax1, dmax2;
  double tdyn, soundspeed, sfrrate, totsfrrate;
  double rate_in_msunperyear;
  int ik;
  char buf[500];
  char mode[2];
  double zs, zg;

#ifdef CS_TESTS
  double uold;
#endif

  if(ThisTask == 0)
    {
      printf("... start cooling and star formation ...\n");
      fflush(stdout);
    }

#ifdef CS_TESTS
  Energy_cooling = 0;
#endif

  strcpy(mode, "a");

  for(bin = 0; bin < TIMEBINS; bin++)
    if(TimeBinActive[bin])
      TimeBinSfr[bin] = 0;

  if(All.ComovingIntegrationOn)
    {
      /* Factors for comoving integration of hydro */
      a3inv = 1 / (All.Time * All.Time * All.Time);
      hubble_a = All.Hubble * sqrt(All.Omega0 / (All.Time * All.Time * All.Time)
				   + (1 - All.Omega0 - All.OmegaLambda) / (All.Time * All.Time) +
				   All.OmegaLambda);

      time_hubble_a = All.Time * hubble_a;
      ascale = All.Time;
    }
  else
    a3inv = ascale = time_hubble_a = 1;

  stars_spawned = stars_converted = 0;
  sum_sm = sum_mass_stars = 0;

  for(bits = 0; GENERATIONS > (1 << bits); bits++);


  for(i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i])
    {
#ifdef SFR
      if(P[i].Type == 0)
#endif
	{
	  XH = P[i].Zm[6] / P[i].Mass;
	  yhelium = (1 - XH) / (4. * XH);

	  if(P[i].Zm[4] > 0)
	    FeHgas = log10(P[i].Zm[4] / 56. / P[i].Zm[6]) + 4.5;
	  else
	    FeHgas = -4;

	  dt = (P[i].TimeBin ? (1 << P[i].TimeBin) : 0) * All.Timebase_interval;	/*  the actual time-step */

	  if(All.ComovingIntegrationOn)
	    dtime = All.Time * dt / time_hubble_a;
	  else
	    dtime = dt;

	  /* check whether conditions for star formation are fulfilled.
	   *  
	   * f=1  star formation allowed
	   * f=0  no star formation 
	   */
	  flag = 1;		/* default is: star formation allowed */

	  if(SphP[i].d.Density * a3inv < All.PhysDensThresh)
	    flag = 0;

	  if(All.ComovingIntegrationOn)
	    if(SphP[i].d.Density < All.OverDensThresh)
	      flag = 0;

	  if(SphP[i].v.DivVel > 0.)	/* if flow is divergent -> no sfr */
	    flag = 0;

	  soundspeed = sqrt(GAMMA * SphP[i].Pressure / SphP[i].d.Density);

	  if(All.ComovingIntegrationOn)
	    {
	      soundspeed /= pow(All.Time, 1.5 * GAMMA_MINUS1);

	      tdyn = 1 / sqrt(4 * M_PI * All.G * SphP[i].d.Density * a3inv);

	    }
	  else
	    {
	      tdyn = 1 / sqrt(4 * M_PI * All.G * SphP[i].d.Density);
	    }


	  /* now do the cooling */

	  ne = SphP[i].Ne;	/* electron abundance (gives ionization state and mean molecular weight) */

#ifdef CS_TESTS
	  uold = DMAX(All.MinEgySpec,
		      (SphP[i].Entropy + SphP[i].e.DtEntropy * dt) /
		      GAMMA_MINUS1 * pow(SphP[i].d.Density * a3inv, GAMMA_MINUS1));
#endif

	  unew = DoCooling(DMAX(All.MinEgySpec,
				(SphP[i].Entropy + SphP[i].e.DtEntropy * dt) /
				GAMMA_MINUS1 * pow(SphP[i].d.Density * a3inv, GAMMA_MINUS1)),
			   SphP[i].d.Density * a3inv, dtime, &ne);

#ifdef CS_TESTS
	  Energy_cooling += (unew - uold) * P[i].Mass;
#endif

	  if(ne > 2)
	    {
	      printf("Part=%d Density=%g ne=%g Entropy=%g XH=%g yhelium=%g FeHgas=%g\n",
		     P[i].ID, SphP[i].d.Density, ne, SphP[i].Entropy, XH, yhelium, FeHgas);
	      fflush(stdout);
	    }

	  SphP[i].Ne = ne;

	  if(P[i].TimeBin)	/* upon start-up, we need to protect against dt==0 */
	    {
	      /* note: the adiabatic rate has been already added in ! */

	      if(dt > 0)
		{
		  SphP[i].e.DtEntropy = (unew * GAMMA_MINUS1 /
					 pow(SphP[i].d.Density * a3inv, GAMMA_MINUS1) - SphP[i].Entropy) / dt;

		  if(SphP[i].e.DtEntropy < -0.5 * SphP[i].Entropy / dt)
		    SphP[i].e.DtEntropy = -0.5 * SphP[i].Entropy / dt;
		}
	    }

	  SphP[i].Sfr = 0;


	  if(flag == 1)		/* now do star formation if allowed */
	    {
	      /* active star formation */

	      p = All.FactorSFR / tdyn * dtime;

	      sm = P[i].Mass * (1 - exp(-p));

	      sum_sm += sm;

	      SphP[i].Sfr = P[i].Mass * All.FactorSFR / tdyn;

	      /* the upper bits of the gas particle ID store how many stars this gas 
	         particle gas already generated */

	      TimeBinSfr[P[i].TimeBin] += SphP[i].Sfr;

	      if(bits == 0)
		number_of_stars_generated = 0;
	      else
		number_of_stars_generated = (P[i].ID >> (sizeof(MyIDType)*8 - bits));

	      mass_of_star = P[i].Mass / (GENERATIONS - number_of_stars_generated);

	      /* now check whether we stochastically produce a star particle */

	      prob = sm / mass_of_star;

	      if(get_random_number(P[i].ID) < prob)	/* ok, make a star */
		{
		  if(number_of_stars_generated == (GENERATIONS - 1))
		    {
		      /* here we turn the gas particle itself into a star */
#ifdef CS_FEEDBACK
		      P[i].EnergySNCold = P[i].EnergySN;
		      P[i].EnergySN = 0;
#endif
		      Stars_converted++;
		      stars_converted++;

		      sum_mass_stars += P[i].Mass;

		      P[i].Type = 4;
		      TimeBinCountSph[P[i].TimeBin]--;
		      TimeBinSfr[P[i].TimeBin] -= SphP[i].Sfr;
#ifdef STELLARAGE
		      P[i].StellarAge = All.Time;
#endif
		    }
		  else
		    {
		      /* here we spawn a new star particle */

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

		      NextActiveParticle[NumPart + stars_spawned] = FirstActiveParticle;
		      FirstActiveParticle = NumPart + stars_spawned;
		      NumForceUpdate++;



		      TimeBinCount[P[NumPart + stars_spawned].TimeBin]++;

		      PrevInTimeBin[NumPart + stars_spawned] = i;
		      NextInTimeBin[NumPart + stars_spawned] = NextInTimeBin[i];
		      if(NextInTimeBin[i] >= 0)
			PrevInTimeBin[NextInTimeBin[i]] = NumPart + stars_spawned;
		      NextInTimeBin[i] = NumPart + stars_spawned;
		      if(LastInTimeBin[P[i].TimeBin] == i)
			LastInTimeBin[P[i].TimeBin] = NumPart + stars_spawned;

		      P[i].ID += ((MyIDType) 1 << (sizeof(MyIDType)*8 - bits));

#ifdef CS_FEEDBACK
		      P[NumPart + stars_spawned].EnergySN = 0;


		      /* This is for the gas particle to keep the same EnergyReservoir/Mass */
		      /* The rest of the Reservoir goes to the EnergySNCold of the star */
		      P[NumPart + stars_spawned].EnergySNCold = P[i].EnergySN * mass_of_star / P[i].Mass;
		      P[i].EnergySN *= (P[i].Mass - mass_of_star) / P[i].Mass;
#endif
		      zs = 0;
		      zg = 0;
		      for(ik = 0; ik < 12; ik++)
			{
			  P[NumPart + stars_spawned].Zm[ik] *= mass_of_star / P[i].Mass;
			  P[i].Zm[ik] -= P[NumPart + stars_spawned].Zm[ik];
			  zs += P[NumPart + stars_spawned].Zm[ik];
			  zg += P[i].Zm[ik];
			}

		      P[NumPart + stars_spawned].Mass = mass_of_star;
		      P[i].Mass -= P[NumPart + stars_spawned].Mass;
		      sum_mass_stars += P[NumPart + stars_spawned].Mass;

		      if((zg / P[i].Mass > 1.01 || zg / P[i].Mass < 0.99) ||
			 (zs / P[NumPart + stars_spawned].Mass > 1.01
			  || zs / P[NumPart + stars_spawned].Mass < 0.99))
			{
			  printf("WARNING ID_Gas=%d Zg=%g  Zs=%g\n", P[i].ID, zg / P[i].Mass,
				 zs / P[NumPart + stars_spawned].Mass);
			  endrun(84760);
			}

#ifdef STELLARAGE
		      P[NumPart + stars_spawned].StellarAge = All.Time;
#endif
		      force_add_star_to_tree(i, NumPart + stars_spawned);

		      stars_spawned++;
		    }
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

      /* Note: N_gas is only reduced once rearrange_particle_sequence is called */

      /* Note: New tree construction can be avoided because of  `force_add_star_to_tree()' */
    }

  for(bin = 0, sfrrate = 0; bin < TIMEBINS; bin++)
    if(TimeBinCount[bin])
      sfrrate += TimeBinSfr[bin];

  MPI_Allreduce(&sfrrate, &totsfrrate, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  MPI_Reduce(&sum_sm, &total_sm, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&sum_mass_stars, &total_sum_mass_stars, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  if(ThisTask == 0)
    {
      if(All.TimeStep > 0)
	{
	  rate = total_sm / (All.TimeStep / time_hubble_a);
	  rate2 = total_sum_mass_stars / (All.TimeStep / time_hubble_a);
	}
      else
	rate = rate2 = 0;

      /* convert to solar masses per yr */

      rate_in_msunperyear = rate * (All.UnitMass_in_g / SOLAR_MASS) / (All.UnitTime_in_s / SEC_PER_YEAR);

      fprintf(FdSfr, "%g %g %g %g %g %g\n", All.Time, total_sm,
	      totsfrrate, rate_in_msunperyear, total_sum_mass_stars, rate2);

      fclose(FdSfr);
      sprintf(buf, "%s%s", All.OutputDir, "sfr.txt");
      FdSfr = fopen(buf, mode);
    }

  if(ThisTask == 0)
    {
      printf("... cooling and star formation done ...\n");
      fflush(stdout);
    }

}


double get_starformation_rate(int i)
{
  double rateOfSF;
  double a3inv;
  int flag;
  double tdyn, soundspeed, hubble_a;

  if(All.ComovingIntegrationOn)
    {
      a3inv = 1 / (All.Time * All.Time * All.Time);
      hubble_a = All.Hubble * sqrt(All.Omega0 / (All.Time * All.Time * All.Time)
				   + (1 - All.Omega0 - All.OmegaLambda) / (All.Time * All.Time) +
				   All.OmegaLambda);
    }
  else
    {
      a3inv = 1;
      hubble_a = 1;
    }

  /*
   * f=1 star formation allowed 
   * f=0 no star formation
   */
  flag = 1;			/* default is: star formation allowed */

  if(SphP[i].d.Density * a3inv < All.PhysDensThresh)
    flag = 0;

  if(All.ComovingIntegrationOn)
    if(SphP[i].d.Density < All.OverDensThresh)
      flag = 0;

  if(SphP[i].v.DivVel > 0.)	/* if flow is divergent -> no sfr */
    flag = 0;

  soundspeed = sqrt(GAMMA * SphP[i].Pressure / SphP[i].d.Density);

  if(All.ComovingIntegrationOn)
    {
      soundspeed /= pow(All.Time, 1.5 * GAMMA_MINUS1);

      tdyn = 1 / sqrt(4 * M_PI * All.G * SphP[i].d.Density * a3inv);

    }
  else
    {
      tdyn = 1 / sqrt(4 * M_PI * All.G * SphP[i].d.Density);

    }

  if(flag == 1)			/* now do star formation if allowed */
    {
      /* active star formation */
      if(All.ComovingIntegrationOn)
	rateOfSF = All.FactorSFR / tdyn;
      else
	rateOfSF = All.FactorSFR / tdyn;

      rateOfSF *= P[i].Mass;

      /* convert to solar masses per yr */

      rateOfSF *= (All.UnitMass_in_g / SOLAR_MASS) / (All.UnitTime_in_s / SEC_PER_YEAR);
    }
  else
    rateOfSF = 0;

  return rateOfSF;
}




#endif /* end of CS_MODEL */
#endif /* end of COOLING */
