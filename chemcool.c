#ifdef CHEMCOOL

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>

#include "allvars.h"
#include "proto.h"

double do_chemcool(int part_index, double dt)
{
  int i, j, flag, nsp, ipar[NIPAR];
  double a, a3inv, hubble_param, hubble_param2, hubble_a, dmin1, dmin2, dmax1, dmax2;
  double timestep, yn, energy, energy_old, entropy_new, dl, divv, column_est, t_start, Utherm_new, du;
  double abundances[TRAC_NUM], y[NSPEC], ydot[NSPEC], rpar[NRPAR];

  CPU_Step[CPU_MISC] += measure_time();

  column_est = divv = 0;

  if(All.ComovingIntegrationOn)
    {
      a = All.Time;
      a3inv = 1.0 / (a * a * a);
      hubble_param = All.HubbleParam;
      hubble_param2 = hubble_param * hubble_param;
      hubble_a = hubble_function(a);
      COOLR.redshift = (1.0 / a) - 1.0;
    }
  else
    a = a3inv = hubble_param = hubble_param2 = hubble_a = COOLR.redshift = 1;

  if(part_index < 0)
    {
      for(i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i])
	{
	  if(P[i].Type == 0)
	    {
	      dt = (P[i].TimeBin ? (1 << P[i].TimeBin) : 0) * All.Timebase_interval;

	      if(dt > 0)
		{
		  timestep = All.UnitTime_in_s * dt / hubble_a / hubble_param;

		  COOLI.index_current = i;
		  COOLI.id_current = P[i].ID;

		  yn = All.UnitDensity_in_cgs * HYDROGEN_MASSFRAC * SphP[i].d.Density * a3inv * hubble_param2 / PROTONMASS;

		  energy = energy_old = All.UnitEnergy_in_cgs / pow(All.UnitLength_in_cm, 3.0) * SphP[i].d.Density * a3inv * hubble_param2 * DMAX(All.MinEgySpec, (SphP[i].Entropy + SphP[i].e.DtEntropy * dt) * pow(SphP[i].d.Density * a3inv, GAMMA_MINUS1) / GAMMA_MINUS1);

		  dl = All.UnitLength_in_cm * SphP[i].Hsml * a / hubble_param;

		  for(j = 0; j < TRAC_NUM; j++)
		    abundances[j] = SphP[i].TracAbund[j];

		  EVOLVE_ABUNDANCES(&timestep, &dl, &yn, &divv, &energy, abundances, &column_est);

		  if(fabs(energy - energy_old) > 1.1 * DTCOOL_SCALE_FACTOR * energy_old)
		    {
		      printf("ID = %d\n", P[i].ID);
		      printf("Thermal energy has changed too much!\n");
		      endrun(6634);
		    }

		  entropy_new = energy / All.UnitEnergy_in_cgs * pow(All.UnitLength_in_cm, 3.0) / hubble_param2 / pow(SphP[i].d.Density * a3inv, GAMMA) * GAMMA_MINUS1;

		  SphP[i].e.DtEntropy = (entropy_new - SphP[i].Entropy) / dt;

		  for(j = 0; j < TRAC_NUM; j++)
		    SphP[i].TracAbund[j] = abundances[j];
		}
	    }
	}
    }
  else
    {
      i = part_index;

      yn = All.UnitDensity_in_cgs * HYDROGEN_MASSFRAC * SphP[i].d.Density * a3inv * hubble_param2 / PROTONMASS;

      energy = energy_old = SphP[i].Entropy * All.UnitEnergy_in_cgs / pow(All.UnitLength_in_cm, 3.0) * hubble_param2 * pow(SphP[i].d.Density * a3inv, GAMMA) / GAMMA_MINUS1;

      dl = All.UnitLength_in_cm * SphP[i].Hsml * a / hubble_param;

      nsp = NSPEC;

      t_start = ipar[0] = 0;

      rpar[0] = yn;
      rpar[1] = dl;
      rpar[2] = divv;
      rpar[3] = column_est;

      for(j = 0; j < TRAC_NUM; j++)
	y[j] = abundances[j] = SphP[i].TracAbund[j];

      y[ITMP] = energy;

      RATE_EQ(&nsp, &t_start, y, ydot, rpar, ipar);

      if(ydot[ITMP] > 0)
	dt = DMIN(DTCOOL_SCALE_FACTOR * y[ITMP] / sqrt(ydot[ITMP] * ydot[ITMP]), dt);

      flag = 0;

      while(flag == 0)
	{
	  energy = energy_old;

	  for(j = 0; j < TRAC_NUM; j++)
	    abundances[j] = SphP[i].TracAbund[j];

	  EVOLVE_ABUNDANCES(&dt, &dl, &yn, &divv, &energy, abundances, &column_est);

	  if(fabs(energy - energy_old) > DTCOOL_SCALE_FACTOR * energy_old)
	    dt /= 2;
	  else
	    flag = 1;
	}
      
      dt *= hubble_param / All.UnitTime_in_s;

      return dt;
    }

  CPU_Step[CPU_COOLINGSFR] += measure_time();
}
#endif
