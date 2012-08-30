#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <gsl/gsl_math.h>

#include "allvars.h"
#include "proto.h"

/*! \file snIa_heating.c
 *  \brief routines for heating by type Ia supernovae
 */

#if defined(SNIA_HEATING)


static struct snheatingdata_in
{
  MyDouble Pos[3];
  MyFloat Density;
  MyFloat Energy;
  MyFloat Hsml;
  int NodeList[NODELISTLENGTH];
}
 *SnheatingdataIn, *SnheatingdataGet;


static double hubble_a, ascale, a3inv;



void snIa_heating(void)
{
  int i, j, k;
  int ndone_flag, ndone;
  int ngrp, sendTask, recvTask, place, nexport, nimport, dummy;
  double dt;
  MPI_Status status;

  if(ThisTask == 0)
    {
      printf("Beginning snIa heating\n");
      fflush(stdout);
    }

  CPU_Step[CPU_MISC] += measure_time();

  if(All.ComovingIntegrationOn)
    {
      ascale = All.Time;
      a3inv = 1.0 / (ascale * ascale * ascale);
      hubble_a = hubble_function(All.Time);
    }
  else
    hubble_a = ascale = a3inv = 1;



  /* allocate buffers to arrange communication */

  Ngblist = (int *) mymalloc("Ngblist", NumPart * sizeof(int));

  All.BunchSize =
    (int) ((All.BufferSize * 1024 * 1024) / (sizeof(struct data_index) + sizeof(struct data_nodelist) +
					     2 * sizeof(struct snheatingdata_in)));
  DataIndexTable =
    (struct data_index *) mymalloc("DataIndexTable", All.BunchSize * sizeof(struct data_index));
  DataNodeList =
    (struct data_nodelist *) mymalloc("DataNodeList", All.BunchSize * sizeof(struct data_nodelist));



  /** Let's first spread the feedback energy */

  i = FirstActiveParticle;	/* first particle for this task */

  do
    {
      for(j = 0; j < NTask; j++)
	{
	  Send_count[j] = 0;
	  Exportflag[j] = -1;
	}

      /* do local particles and prepare export list */

      for(nexport = 0; i >= 0; i = NextActiveParticle[i])
	if(P[i].Type == 4)
	  if(snIaheating_evaluate(i, 0, &nexport, Send_count) < 0)
	    break;

#ifdef MYSORT
      mysort_dataindex(DataIndexTable, nexport, sizeof(struct data_index), data_index_compare);
#else
      qsort(DataIndexTable, nexport, sizeof(struct data_index), data_index_compare);
#endif

      MPI_Alltoall(Send_count, 1, MPI_INT, Recv_count, 1, MPI_INT, MPI_COMM_WORLD);

      for(j = 0, nimport = 0, Recv_offset[0] = 0, Send_offset[0] = 0; j < NTask; j++)
	{
	  nimport += Recv_count[j];

	  if(j > 0)
	    {
	      Send_offset[j] = Send_offset[j - 1] + Send_count[j - 1];
	      Recv_offset[j] = Recv_offset[j - 1] + Recv_count[j - 1];
	    }
	}

      SnheatingdataGet =
	(struct snheatingdata_in *) mymalloc("SnheatingdataGet", nimport * sizeof(struct snheatingdata_in));
      SnheatingdataIn =
	(struct snheatingdata_in *) mymalloc("SnheatingdataIn", nexport * sizeof(struct snheatingdata_in));

      for(j = 0; j < nexport; j++)
	{
	  place = DataIndexTable[j].Index;

	  for(k = 0; k < 3; k++)
	    SnheatingdataIn[j].Pos[k] = P[place].Pos[k];

	  SnheatingdataIn[j].Hsml = PPP[place].Hsml;
	  SnheatingdataIn[j].Density = P[place].DensAroundStar;

	  dt = (P[place].TimeBin ? (1 << P[place].TimeBin) : 0) * All.Timebase_interval / hubble_a;

	  SnheatingdataIn[j].Energy = All.SnIaHeatingRate * dt * P[place].Mass;

	  memcpy(SnheatingdataIn[j].NodeList,
		 DataNodeList[DataIndexTable[j].IndexGet].NodeList, NODELISTLENGTH * sizeof(int));
	}


      /* exchange particle data */
      for(ngrp = 1; ngrp < (1 << PTask); ngrp++)
	{
	  sendTask = ThisTask;
	  recvTask = ThisTask ^ ngrp;

	  if(recvTask < NTask)
	    {
	      if(Send_count[recvTask] > 0 || Recv_count[recvTask] > 0)
		{
		  /* get the particles */
		  MPI_Sendrecv(&SnheatingdataIn[Send_offset[recvTask]],
			       Send_count[recvTask] * sizeof(struct snheatingdata_in), MPI_BYTE,
			       recvTask, TAG_DENS_A,
			       &SnheatingdataGet[Recv_offset[recvTask]],
			       Recv_count[recvTask] * sizeof(struct snheatingdata_in), MPI_BYTE,
			       recvTask, TAG_DENS_A, MPI_COMM_WORLD, &status);
		}
	    }
	}

      myfree(SnheatingdataIn);

      /* now do the particles that were sent to us */

      for(j = 0; j < nimport; j++)
	snIaheating_evaluate(j, 1, &dummy, &dummy);

      if(i < 0)
	ndone_flag = 1;
      else
	ndone_flag = 0;

      MPI_Allreduce(&ndone_flag, &ndone, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

      myfree(SnheatingdataGet);
    }
  while(ndone < NTask);



  myfree(DataNodeList);
  myfree(DataIndexTable);
  myfree(Ngblist);


  CPU_Step[CPU_COOLINGSFR] += measure_time();
}




int snIaheating_evaluate(int target, int mode, int *nexport, int *nsend_local)
{
  int startnode, numngb, j, n, listindex = 0;
  MyFloat *pos, h_i, dt, rho;
  double dx, dy, dz, h_i2, r2, r, u, hinv, hinv3, wk, energy, egy;

  if(mode == 0)
    {
      pos = P[target].Pos;
      rho = P[target].DensAroundStar;

      dt = (P[target].TimeBin ? (1 << P[target].TimeBin) : 0) * All.Timebase_interval / hubble_a;

      energy = All.SnIaHeatingRate * dt * P[target].Mass;

      h_i = PPP[target].Hsml;
    }
  else
    {
      pos = SnheatingdataGet[target].Pos;
      rho = SnheatingdataGet[target].Density;
      energy = SnheatingdataGet[target].Energy;
      h_i = SnheatingdataGet[target].Hsml;
    }

  /* initialize variables before SPH loop is started */
  h_i2 = h_i * h_i;

  /* Now start the actual SPH computation for this particle */
  if(mode == 0)
    {
      startnode = All.MaxPart;	/* root node */
    }
  else
    {
      startnode = SnheatingdataGet[target].NodeList[0];
      startnode = Nodes[startnode].u.d.nextnode;	/* open it */
    }

  while(startnode >= 0)
    {
      while(startnode >= 0)
	{
	  numngb = ngb_treefind_variable(pos, h_i, target, &startnode, mode, nexport, nsend_local);

	  if(numngb < 0)
	    return -1;

	  for(n = 0; n < numngb; n++)
	    {
	      j = Ngblist[n];

	      if(P[j].Type == 0 && P[j].Mass > 0)
		{
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

		  if(r2 < h_i2)
		    {
		      r = sqrt(r2);
		      hinv = 1 / h_i;
#ifndef  TWODIMS
		      hinv3 = hinv * hinv * hinv;
#else
		      hinv3 = hinv * hinv / boxSize_Z;
#endif

		      u = r * hinv;

		      if(u < 0.5)
			wk = hinv3 * (KERNEL_COEFF_1 + KERNEL_COEFF_2 * (u - 1) * u * u);
		      else
			wk = hinv3 * KERNEL_COEFF_5 * (1.0 - u) * (1.0 - u) * (1.0 - u);

		      egy = P[j].Mass * wk / rho * energy;

		      SphP[j].Entropy += egy / P[j].Mass * GAMMA_MINUS1 /
			pow(SphP[j].d.Density * a3inv, GAMMA_MINUS1);

		    }
		}
	    }
	}

      if(mode == 1)
	{
	  listindex++;
	  if(listindex < NODELISTLENGTH)
	    {
	      startnode = SnheatingdataGet[target].NodeList[listindex];
	      if(startnode >= 0)
		startnode = Nodes[startnode].u.d.nextnode;	/* open it */
	    }
	}
    }


  return 0;
}





#endif
