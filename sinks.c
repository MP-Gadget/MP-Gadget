#ifdef SINKS

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>

#include "allvars.h"
#include "proto.h"
#include "domain.h"

void do_sinks(void)
{
  int i, j, k, index, sink_index, iter, count, count_global, prev, num_acc, num_acc_glob, num_update,
    num_update_glob;
  int numforceupdate, globnumforceupdate, count_list[NTask], count_list_bytes[NTask], prev_list[NTask],
    prev_list_bytes[NTask];
  double a, a2, a3inv, hubble_param, hubble_param2, nh, semimajor_axis, alpha, min_alpha, xtmp,
    mass_cm;
  double dx, dy, dz, r, r2, r2_min, h, u, wp, e_grav, e_kin, e_therm, divv, sink_pos[3], pos_cm[3], vel_cm[3];
#ifdef MAGNETIC
  double e_mag = 0, mu0 = 0;
#endif


  struct particle_data psave;

  struct
  {
    double nh_max;
    int task;
  } local, global;

  struct sink_list
  {
    int task;
    int index;
    int active;
    double pos[3];
    double vel[3];
#ifdef MAGNETIC
	  double bfld[3];
	  double rho;
#endif
    double mass;
    double soft;
    double utherm;
  } *sink, *sink_send;


#ifdef MAGNETIC
#ifndef MU0_UNITY
    mu0 *= (4 * M_PI);
	mu0 /= All.UnitTime_in_s * All.UnitTime_in_s *
		   All.UnitLength_in_cm / All.UnitMass_in_g;
	if(All.ComovingIntegrationOn)
      mu0 /= (All.HubbleParam * All.HubbleParam);
#endif
#endif

  /* Sink Preparation */

  CPU_Step[CPU_MISC] += measure_time();

  if(ThisTask == 0)
    {
      printf("\nDoing sinks...\n");
      fflush(stdout);
    }

  if(All.ComovingIntegrationOn)
    {
      a = All.Time;
      a2 = a * a;
      a3inv = 1.0 / (a * a2);
      hubble_param = All.HubbleParam;
      hubble_param2 = hubble_param * hubble_param;
    }
  else
    a = a2 = a3inv = hubble_param = hubble_param2 = 1.0;

  for(i = NumSinks = 0; i < NumPart; i++)
    if(P[i].Type == 5)
      NumSinks++;

  MPI_Allreduce(&NumSinks, &All.TotNumSinks, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  num_update = num_update_glob = 0;

  if(All.TotNumSinks > 0)
    {
      if(!(sink_send = (struct sink_list *) malloc(All.TotNumSinks * sizeof(struct sink_list))))
	terminate("Failed to allocate memory!\n");

      if(!(sink = (struct sink_list *) malloc(All.TotNumSinks * sizeof(struct sink_list))))
	terminate("Failed to allocate memory!\n");

      MPI_Allgather(&NumSinks, 1, MPI_INT, count_list, 1, MPI_INT, MPI_COMM_WORLD);

      for(i = 1, prev = prev_list[0] = prev_list_bytes[0] = 0, count_list_bytes[0] =
	  count_list[0] * sizeof(struct sink_list); i < NTask; i++)
	{
	  prev += count_list[i - 1];
	  prev_list[i] = prev;

	  count_list_bytes[i] = count_list[i] * sizeof(struct sink_list);
	  prev_list_bytes[i] = prev_list[i] * sizeof(struct sink_list);
	}

      for(i = 0, index = prev_list[ThisTask]; i < NumSinks; i++)
	{
	  sink_send[index].task = ThisTask;
	  sink_send[index].index = i;

	  if(TimeBinActive[P[NumPart - NumSinks + i].TimeBin])
	    sink_send[index].active = 1;
	  else
	    sink_send[index].active = 0;

	  for(j = 0; j < 3; j++)
	    {
	      sink_send[index].pos[j] = P[NumPart - NumSinks + i].Pos[j];
	      sink_send[index].vel[j] = P[NumPart - NumSinks + i].Vel[j];
#ifdef MAGNETIC
          sink_send[index].bfld[j] = SphP[NumPart - NumSinks + i].BPred[j];
#endif	   
	    }

	  sink_send[index].mass = P[NumPart - NumSinks + i].Mass;

#ifdef MAGNETIC
      sink_send[index].rho = SphP[NumPart - NumSinks + i].d.Density;
#endif	  
	  index++;
	}

      MPI_Allgatherv(&sink_send[prev_list[ThisTask]], NumSinks * sizeof(struct sink_list), MPI_BYTE, sink,
		     count_list_bytes, prev_list_bytes, MPI_BYTE, MPI_COMM_WORLD);

      free(sink_send);
    }



  /* Sink Mergers */

  for(i = 0; i < All.TotNumSinks; i++)
    if(sink[i].active)
      {
	iter = min_alpha = semimajor_axis = 0;
	index = -1;

	for(j = i + 1; j < All.TotNumSinks; j++)
	  if(sink[j].active)
	    {
	      dx = NGB_PERIODIC_LONG_X(sink[i].pos[0] - sink[j].pos[0]);
	      dy = NGB_PERIODIC_LONG_Y(sink[i].pos[1] - sink[j].pos[1]);
	      dz = NGB_PERIODIC_LONG_Z(sink[i].pos[2] - sink[j].pos[2]);

	      r = sqrt(dx * dx + dy * dy + dz * dz);

	      if(r < All.SinkHsml / a * hubble_param)
		{
		  h = All.ForceSoftening[5];

		  if(r > h)
		    wp = 1 / r;
		  else
		    {
		      u = r / h;

		      if(u < 0.5)
			wp = -1.0 / h * (-2.8 + u * u * (5.333333333333 + u * u * (6.4 * u - 9.6)));
		      else
			wp =
			  -1.0 / h * (-3.2 + 0.066666666667 / u +
				      u * u * (10.666666666667 +
					       u * (-16.0 + u * (9.6 - 2.133333333333 * u))));
		    }

		  e_grav = All.G * sink[i].mass * sink[j].mass * wp / a;

		  for(k = e_kin = 0; k < 3; k++)
		    {
		      vel_cm[k] =
			(sink[i].mass * sink[i].vel[k] + sink[j].mass * sink[j].vel[k]) / (sink[i].mass +
											   sink[j].mass);

		      e_kin +=
			0.5 * sink[i].mass * (sink[i].vel[k] - vel_cm[k]) * (sink[i].vel[k] - vel_cm[k]) / a2;
		      e_kin +=
			0.5 * sink[j].mass * (sink[j].vel[k] - vel_cm[k]) * (sink[j].vel[k] - vel_cm[k]) / a2;
		    }

		  alpha = e_kin / e_grav;

		  if(iter == 0 || alpha < min_alpha)
		    {
		      index = j;

		      min_alpha = alpha;

		      semimajor_axis = -All.G * sink[i].mass * sink[j].mass / 2 / (-e_grav + e_kin);
		    }

		  iter++;
		}
	    }

	if(index > -1 && semimajor_axis > 0 && semimajor_axis < All.SinkHsml * hubble_param)
	  {
	    for(k = 0; k < 3; k++)
	      {
		sink[i].pos[k] =
		  (sink[i].mass * sink[i].pos[k] + sink[index].mass * sink[index].pos[k]) / (sink[i].mass +
											     sink[index].
											     mass);
		sink[i].vel[k] = vel_cm[k];
	      }

	    sink[i].mass += sink[index].mass;

	    if(ThisTask == sink[i].task)
	      {
		P[NumPart - NumSinks + sink[i].index].Mass = sink[i].mass;

		for(k = 0; k < 3; k++)
		  {
		    P[NumPart - NumSinks + sink[i].index].Pos[k] = sink[i].pos[k];
		    P[NumPart - NumSinks + sink[i].index].Vel[k] = sink[i].vel[k];
		  }

		printf("Sink merger: %g M_sun\n",
		       sink[i].mass / hubble_param * All.UnitMass_in_g / SOLAR_MASS);
		fflush(stdout);
	      }

	    if(ThisTask == sink[index].task)
	      {
		for(k = sink[index].index; k < NumSinks - 1; k++)
		  P[NumPart - NumSinks + k] = P[NumPart - NumSinks + k + 1];

		num_update++;

		NumSinks--;
		NumPart--;
		NumForceUpdate--;
	      }

	    for(k = index; k < All.TotNumSinks - 1; k++)
	      {
		if(sink[k + 1].task == sink[index].task)
		  sink[k + 1].index -= 1;

		sink[k] = sink[k + 1];
	      }

	    All.TotNumSinks--;
	    All.TotNumPart--;
	    GlobNumForceUpdate--;

	    break;
	  }
      }



  /* Sink Accretion */

  if(All.TotNumSinks > 0)
    {
      double mass_array[All.TotNumSinks], cm_array[All.TotNumSinks][3], mom_array[All.TotNumSinks][3];
      double glob_mass_array[All.TotNumSinks], glob_cm_array[All.TotNumSinks][3],
	glob_mom_array[All.TotNumSinks][3];

      for(i = 0; i < All.TotNumSinks; i++)
	{
	  mass_array[i] = 0;

	  for(j = 0; j < 3; j++)
	    cm_array[i][j] = mom_array[i][j] = 0;
	}

      for(i = num_acc = 0; i < N_gas; i++)
	if(TimeBinActive[P[i].TimeBin])
	  {
	    iter = min_alpha = semimajor_axis = 0;
	    index = -1;

	    for(j = 0; j < All.TotNumSinks; j++)
	      if(sink[j].active)
		{
		  dx = NGB_PERIODIC_LONG_X(P[i].Pos[0] - sink[j].pos[0]);
		  dy = NGB_PERIODIC_LONG_Y(P[i].Pos[1] - sink[j].pos[1]);
		  dz = NGB_PERIODIC_LONG_Z(P[i].Pos[2] - sink[j].pos[2]);

		  r = sqrt(dx * dx + dy * dy + dz * dz);

		  if(r < All.SinkHsml / a * hubble_param)
		    {
		      h = DMAX(SphP[i].Hsml, All.ForceSoftening[5]);

		      if(r > h)
			wp = 1 / r;
		      else
			{
			  u = r / h;

			  if(u < 0.5)
			    wp = -1.0 / h * (-2.8 + u * u * (5.333333333333 + u * u * (6.4 * u - 9.6)));
			  else
			    wp =
			      -1.0 / h * (-3.2 + 0.066666666667 / u +
					  u * u * (10.666666666667 +
						   u * (-16.0 + u * (9.6 - 2.133333333333 * u))));
			}

		      e_grav = All.G * P[i].Mass * sink[j].mass * wp / a;

		      for(k = e_kin = 0; k < 3; k++)
			{
			  vel_cm[k] =
			    (P[i].Mass * P[i].Vel[k] + sink[j].mass * sink[j].vel[k]) / (P[i].Mass +
											 sink[j].mass);

			  e_kin +=
			    0.5 * P[i].Mass * (P[i].Vel[k] - vel_cm[k]) * (P[i].Vel[k] - vel_cm[k]) / a2;
			  e_kin +=
			    0.5 * sink[j].mass * (sink[j].vel[k] - vel_cm[k]) * (sink[j].vel[k] -
										 vel_cm[k]) / a2;
			}

		      alpha = e_kin / e_grav;

		      if(iter == 0 || alpha < min_alpha)
			{
			  index = j;

			  min_alpha = alpha;

			  semimajor_axis = -All.G * P[i].Mass * sink[j].mass / 2 / (-e_grav + e_kin);
			}

		      iter++;
		    }
		}

	    //if(index > -1 && semimajor_axis > 0 && semimajor_axis < All.SinkHsml * hubble_param)
	    if(index > -1)
	      {
		num_acc++;
		num_update++;

		mass_array[index] += P[i].Mass;

		for(j = 0; j < 3; j++)
		  {
		    cm_array[index][j] += P[i].Mass * P[i].Pos[j];
		    mom_array[index][j] += P[i].Mass * P[i].Vel[j];
		  }

		P[i] = P[N_gas - 1];
		SphP[i] = SphP[N_gas - 1];

		N_gas--;
		NumPart--;
		NumForceUpdate--;

		i--;
	      }
	  }

      if(num_acc > 0 && NumPart - N_gas > 0)
	memmove(P + N_gas, P + N_gas + num_acc, (NumPart - N_gas) * sizeof(struct particle_data));

      MPI_Allreduce(&num_acc, &num_acc_glob, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

      if(num_acc_glob)
	{
	  MPI_Allreduce(mass_array, glob_mass_array, All.TotNumSinks, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	  MPI_Allreduce(cm_array, glob_cm_array, 3 * All.TotNumSinks, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	  MPI_Allreduce(mom_array, glob_mom_array, 3 * All.TotNumSinks, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

	  for(i = 0; i < All.TotNumSinks; i++)
	    {
	      if(ThisTask == sink[i].task && glob_mass_array[i] > 0)
		{
		  index = NumPart - NumSinks + sink[i].index;

		  for(j = 0; j < 3; j++)
		    {
		      P[index].Pos[j] =
			(P[index].Mass * P[index].Pos[j] + glob_cm_array[i][j]) / (P[index].Mass +
										   glob_mass_array[i]);
		      P[index].Vel[j] =
			(P[index].Mass * P[index].Vel[j] + glob_mom_array[i][j]) / (P[index].Mass +
										    glob_mass_array[i]);
		    }

		  P[index].Mass += glob_mass_array[i];

		  printf("Sink accretion: new mass = %g M_sun\n",
			 P[index].Mass / hubble_param * All.UnitMass_in_g / SOLAR_MASS);
		  fflush(stdout);
		}
	    }

	  All.TotN_gas -= num_acc_glob;
	  All.TotNumPart -= num_acc_glob;
	  GlobNumForceUpdate -= num_acc_glob;
	}

      free(sink);
    }



  /* Sink Creation */

  for(i = local.nh_max = sink_index = 0; i < N_gas; i++)
    if(TimeBinActive[P[i].TimeBin])
      {
	for(j = iter = r2_min = 0; j < All.TotNumSinks; j++)
	  {
	    dx = NGB_PERIODIC_LONG_X(P[i].Pos[0] - sink[j].pos[0]);
	    dy = NGB_PERIODIC_LONG_Y(P[i].Pos[1] - sink[j].pos[1]);
	    dz = NGB_PERIODIC_LONG_Z(P[i].Pos[2] - sink[j].pos[2]);

	    r2 = (dx * dx + dy * dy + dz * dz) * a2 / hubble_param2;

	    if(iter == 0 || r2 < r2_min)
	      r2_min = r2;
	  }

	if(All.TotNumSinks == 0 || r2_min > All.SinkHsml * All.SinkHsml)
	  {
	    nh =
	      HYDROGEN_MASSFRAC * SphP[i].d.Density * All.UnitDensity_in_cgs * a3inv * hubble_param2 /
	      PROTONMASS;

	    if(nh > local.nh_max)
	      {
		local.nh_max = nh;
		sink_index = i;
	      }
	  }
      }

  local.task = ThisTask;

  MPI_Allreduce(&local, &global, 1, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);

  //if((All.TotNumSinks == 0 && global.nh_max > All.SinkDensThresh) || (All.TotNumSinks > 0 && global.nh_max > 10 * All.SinkDensThresh))
  if(global.nh_max > All.SinkDensThresh)
    {
      if(ThisTask == global.task)
	for(i = 0; i < 3; i++)
	  sink_pos[i] = P[sink_index].Pos[i];

      MPI_Bcast(sink_pos, 3, MPI_DOUBLE, global.task, MPI_COMM_WORLD);

      for(i = count = 0; i < N_gas; i++)
	{
	  dx = NGB_PERIODIC_LONG_X(P[i].Pos[0] - sink_pos[0]);
	  dy = NGB_PERIODIC_LONG_Y(P[i].Pos[1] - sink_pos[1]);
	  dz = NGB_PERIODIC_LONG_Z(P[i].Pos[2] - sink_pos[2]);

	  r2 = (dx * dx + dy * dy + dz * dz) * a2 / hubble_param2;

	  if(r2 < All.SinkHsml * All.SinkHsml)
	    count++;
	}

      MPI_Allgather(&count, 1, MPI_INT, count_list, 1, MPI_INT, MPI_COMM_WORLD);

      for(i = count_global = 0; i < NTask; i++)
	count_global += count_list[i];

      if(count_global)
	{
	  if(!(sink_send = (struct sink_list *) malloc(count_global * sizeof(struct sink_list))))
	    terminate("Failed to allocate memory!\n");

	  if(!(sink = (struct sink_list *) malloc(count_global * sizeof(struct sink_list))))
	    terminate("Failed to allocate memory!\n");

	  for(i = 1, prev = prev_list[0] = prev_list_bytes[0] = 0, count_list_bytes[0] =
	      count_list[0] * sizeof(struct sink_list); i < NTask; i++)
	    {
	      prev += count_list[i - 1];
	      prev_list[i] = prev;

	      count_list_bytes[i] = count_list[i] * sizeof(struct sink_list);
	      prev_list_bytes[i] = prev_list[i] * sizeof(struct sink_list);
	    }

	  for(i = 0, index = prev_list[ThisTask]; i < N_gas; i++)
	    {
	      dx = NGB_PERIODIC_LONG_X(P[i].Pos[0] - sink_pos[0]);
	      dy = NGB_PERIODIC_LONG_Y(P[i].Pos[1] - sink_pos[1]);
	      dz = NGB_PERIODIC_LONG_Z(P[i].Pos[2] - sink_pos[2]);

	      r2 = (dx * dx + dy * dy + dz * dz) * a2 / hubble_param2;

	      if(r2 < All.SinkHsml * All.SinkHsml)
		{
		  sink_send[index].mass = P[i].Mass;
		  sink_send[index].soft = SphP[i].Hsml;
#ifndef FB_BAROTROPIC_EOS
		  sink_send[index].utherm = SphP[i].Entropy * pow(SphP[i].d.Density * a3inv, GAMMA_MINUS1) / GAMMA_MINUS1;
#else
		  sink_send[index].utherm = get_energy(SphP[i].d.Density);
#endif				 
#ifdef MAGNETIC
		  sink_send[index].rho = SphP[i].d.Density;
#endif

		  for(j = 0; j < 3; j++)
		    {
		      sink_send[index].pos[j] = P[i].Pos[j];
		      sink_send[index].vel[j] = P[i].Vel[j];
#ifdef MAGNETIC
              sink_send[index].bfld[j] = SphP[i].BPred[j];
#endif
		    }

		  index++;
		}
	    }

	  MPI_Allgatherv(&sink_send[prev_list[ThisTask]], count * sizeof(struct sink_list), MPI_BYTE, sink,
			 count_list_bytes, prev_list_bytes, MPI_BYTE, MPI_COMM_WORLD);

	  for(i = pos_cm[0] = pos_cm[1] = pos_cm[2] = vel_cm[0] = vel_cm[1] = vel_cm[2] = mass_cm = 0;
	      i < count_global; i++)
	    {
	      for(j = 0; j < 3; j++)
		{
		  pos_cm[j] += sink[i].mass * sink[i].pos[j];
		  vel_cm[j] += sink[i].mass * sink[i].vel[j];
		}

	      mass_cm += sink[i].mass;
	    }

	  for(i = 0; i < 3; i++)
	    {
	      pos_cm[i] /= mass_cm;
	      vel_cm[i] /= mass_cm;
	    }


#ifndef MAGNETIC
	  for(i = e_grav = e_kin = e_therm = divv = 0; i < count_global; i++)
#else
	  for(i = e_grav = e_kin = e_therm = e_mag = divv = 0; i < count_global; i++)
#endif
	    {
	      if(count_global > 500)
		e_grav = 3.0 / 5.0 * All.G * mass_cm * mass_cm / (All.SinkHsml * hubble_param);
	      else
		for(j = i; j < count_global; j++)
		  {
		    dx = NGB_PERIODIC_LONG_X(sink[i].pos[0] - sink[j].pos[0]);
		    dy = NGB_PERIODIC_LONG_Y(sink[i].pos[1] - sink[j].pos[1]);
		    dz = NGB_PERIODIC_LONG_Z(sink[i].pos[2] - sink[j].pos[2]);

		    r = sqrt(dx * dx + dy * dy + dz * dz);

		    h = DMAX(sink[i].soft, sink[j].soft);

		    if(r > h)
		      wp = 1 / r;
		    else
		      {
			u = r / h;

			if(u < 0.5)
			  wp = -1.0 / h * (-2.8 + u * u * (5.333333333333 + u * u * (6.4 * u - 9.6)));
			else
			  wp =
			    -1.0 / h * (-3.2 + 0.066666666667 / u +
					u * u * (10.666666666667 +
						 u * (-16.0 + u * (9.6 - 2.133333333333 * u))));
		      }

		    e_grav += All.G * sink[i].mass * sink[j].mass * wp / a;
		  }

	      for(j = 0; j < 3; j++)
             {
		        e_kin +=
		         0.5 * sink[i].mass * (sink[i].vel[j] - vel_cm[j]) * (sink[i].vel[j] - vel_cm[j]) / a2;
#ifdef MAGNETIC
	          e_mag += 0.5 * sink[i].mass * (sink[i].bfld[j]*sink[i].bfld[j]) / (sink[i].rho*mu0);
#endif
             }

	      e_therm += sink[i].utherm * sink[i].mass;

	      dx = NGB_PERIODIC_LONG_X(sink[i].pos[0] - pos_cm[0]);
	      dy = NGB_PERIODIC_LONG_Y(sink[i].pos[1] - pos_cm[1]);
	      dz = NGB_PERIODIC_LONG_Z(sink[i].pos[2] - pos_cm[2]);

	      r = sqrt(dx * dx + dy * dy + dz * dz);

	      if(r > 0)
		divv +=
		  sink[i].mass * ((sink[i].vel[0] - vel_cm[0]) * dx + (sink[i].vel[1] - vel_cm[1]) * dy +
				  (sink[i].vel[2] - vel_cm[2]) * dz) / r;
	    }

	  if(ThisTask == 0)
#ifndef MAGNETIC
	    printf("Sink Check: %d particles, egrav = %g, ekin = %g, etherm = %g, divv = %g\n", count_global, e_grav, e_kin,
		   e_therm, divv);
#else
	    printf("Sink Check: %d particles, egrav = %g, ekin = %g, etherm = %g, emag = %g, divv = %g\n", count_global, e_grav, e_kin,
		   e_therm, e_mag, divv);
#endif

	  //if(e_grav > e_kin && divv < 0)
	  //if(divv < 0)
#ifndef MAGNETIC
	  if(e_grav > 2 * e_therm && e_grav > e_kin + e_therm && divv < 0)
#else
	  if(e_grav > (2 * e_therm + e_mag) && e_grav > e_kin + e_therm + e_mag && divv < 0)		  
#endif	
    {
	      for(i = num_acc = numforceupdate = 0; i < N_gas; i++)
		{
		  dx = NGB_PERIODIC_LONG_X(P[i].Pos[0] - sink_pos[0]);
		  dy = NGB_PERIODIC_LONG_Y(P[i].Pos[1] - sink_pos[1]);
		  dz = NGB_PERIODIC_LONG_Z(P[i].Pos[2] - sink_pos[2]);

		  r2 = (dx * dx + dy * dy + dz * dz) * a2 / hubble_param2;

		  if(r2 < All.SinkHsml * All.SinkHsml)
		    {
		      num_acc++;
		      num_update++;

		      if(TimeBinActive[P[i].TimeBin])
			{
			  numforceupdate++;
			  NumForceUpdate--;
			}

		      if(ThisTask == global.task && i == sink_index)
			{
			  sink_index = -1;
			  psave = P[i];
			}

		      P[i] = P[N_gas - 1];
		      SphP[i] = SphP[N_gas - 1];

		      if(ThisTask == global.task && N_gas - 1 == sink_index)
			sink_index = i;

		      N_gas--;
		      NumPart--;

		      i--;
		    }
		}

	      if(num_acc > 0 && NumPart - N_gas > 0)
		memmove(P + N_gas, P + N_gas + num_acc, (NumPart - N_gas) * sizeof(struct particle_data));

	      MPI_Allreduce(&num_acc, &num_acc_glob, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

	      if(num_acc_glob)
		{
		  MPI_Allreduce(&numforceupdate, &globnumforceupdate, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

		  All.TotN_gas -= num_acc_glob;
		  All.TotNumPart -= num_acc_glob;
		  GlobNumForceUpdate -= globnumforceupdate;

		  if(ThisTask == global.task)
		    {
		      NumSinks++;
		      NumPart++;
		      NumForceUpdate++;

		      P[NumPart - 1] = psave;

		      for(i = 0; i < 3; i++)
			{
			  P[NumPart - 1].Pos[i] = pos_cm[i];
			  P[NumPart - 1].Vel[i] = vel_cm[i];
			}

		      P[NumPart - 1].Mass = mass_cm;
		      P[NumPart - 1].Type = 5;
		    }

		  if(All.TotNumSinks == 0)
		    All.MassTable[5] = 0;

		  All.TotNumSinks++;
		  All.TotNumPart++;
		  GlobNumForceUpdate++;

		  if(ThisTask == 0)
		    {
		      printf("Created sink #%d at %g, %g, %g with %d particles for a total of %g M_sun\n",
			     All.TotNumSinks - 1, pos_cm[0], pos_cm[1], pos_cm[2], num_acc_glob,
			     mass_cm / hubble_param * All.UnitMass_in_g / SOLAR_MASS);
		      fflush(stdout);
		    }
		}
	    }

	  free(sink_send);
	  free(sink);
	}
    }

  MPI_Allreduce(&num_update, &num_update_glob, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  if(num_update_glob)
    {
      All.DoDynamicUpdate = 0;
      domain_Decomposition();

      force_treebuild(NumPart, NULL);
    }

  if(ThisTask == 0)
    {
      printf("Done with sinks (%d actions)...\n", num_update_glob);
      fflush(stdout);
    }

  CPU_Step[CPU_BLACKHOLES] += measure_time();
}
#endif
