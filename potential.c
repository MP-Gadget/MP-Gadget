#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>


#include "allvars.h"
#include "proto.h"


/*! \file potential.c
 *  \brief Computation of the gravitational potential of particles
 */

#if defined(COMPUTE_POTENTIAL_ENERGY) || defined(OUTPUTPOTENTIAL)

/*! This function computes the gravitational potential for ALL the particles.
 *  First, the (short-range) tree potential is computed, and then, if needed,
 *  the long range PM potential is added.
 */
void compute_potential(void)
{
  int i;

#ifndef NOGRAVITY
  int j, k, ret, sendTask, recvTask;
  int ndone, ndone_flag, dummy;
  int ngrp, place, nexport, nimport;
  double fac;
  MPI_Status status;
  double r2;

  if(All.ComovingIntegrationOn)
    set_softenings();

  if(ThisTask == 0)
    {
      printf("Start computation of potential for all particles...\n");
      fflush(stdout);
    }

  CPU_Step[CPU_MISC] += walltime_measure(WALL_MISC);

  /* allocate buffers to arrange communication */
  All.BunchSize =
    (int) ((All.BufferSize * 1024 * 1024) / (sizeof(struct data_index) + sizeof(struct data_nodelist) +
					     sizeof(struct gravdata_in) + sizeof(struct potdata_out) +
					     sizemax(sizeof(struct gravdata_in),
						     sizeof(struct potdata_out))));
  DataIndexTable =
    (struct data_index *) mymalloc("DataIndexTable", All.BunchSize * sizeof(struct data_index));
  DataNodeList =
    (struct data_nodelist *) mymalloc("DataNodeList", All.BunchSize * sizeof(struct data_nodelist));

#ifndef SUBFIND_RESHUFFLE_AND_POTENTIAL
  for(i = 0; i < NumPart; i++)
    if(P[i].Ti_current != All.Ti_Current)
      drift_particle(i, All.Ti_Current);
#endif
  i = 0;			/* beginn with this index */

  do
    {
      for(j = 0; j < NTask; j++)
	{
	  Send_count[j] = 0;
	  Exportflag[j] = -1;
	}

      /* do local particles and prepare export list */
      for(nexport = 0; i < NumPart; i++)
	{
#ifndef PMGRID
	  ret = force_treeevaluate_potential(i, 0, &nexport, Send_count);
#else
	  ret = force_treeevaluate_potential_shortrange(i, 0, &nexport, Send_count);
#endif
	  if(ret < 0)
	    break;		/* export buffer has filled up */
	}

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

      GravDataGet = (struct gravdata_in *) mymalloc("GravDataGet", nimport * sizeof(struct gravdata_in));
      GravDataIn = (struct gravdata_in *) mymalloc("GravDataIn", nexport * sizeof(struct gravdata_in));

      /* prepare particle data for export */
      for(j = 0; j < nexport; j++)
	{
	  place = DataIndexTable[j].Index;

	  for(k = 0; k < 3; k++)
	    GravDataIn[j].Pos[k] = P[place].Pos[k];

#ifdef UNEQUALSOFTENINGS
	  GravDataIn[j].Type = P[place].Type;
#ifdef ADAPTIVE_GRAVSOFT_FORGAS
	  if(P[place].Type == 0)
	    GravDataIn[j].Soft = SPHP(place).Hsml;
#endif
#endif
	  GravDataIn[j].OldAcc = P[place].OldAcc;

	  for(k = 0; k < NODELISTLENGTH; k++)
	    GravDataIn[j].NodeList[k] = DataNodeList[DataIndexTable[j].IndexGet].NodeList[k];
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
		  MPI_Sendrecv(&GravDataIn[Send_offset[recvTask]],
			       Send_count[recvTask] * sizeof(struct gravdata_in), MPI_BYTE,
			       recvTask, TAG_POTENTIAL_A,
			       &GravDataGet[Recv_offset[recvTask]],
			       Recv_count[recvTask] * sizeof(struct gravdata_in), MPI_BYTE,
			       recvTask, TAG_POTENTIAL_A, MPI_COMM_WORLD, &status);
		}
	    }
	}

      myfree(GravDataIn);
      PotDataResult = (struct potdata_out *) mymalloc("PotDataResult", nimport * sizeof(struct potdata_out));
      PotDataOut = (struct potdata_out *) mymalloc("PotDataOut", nexport * sizeof(struct potdata_out));


      /* now do the particles that were sent to us */
      for(j = 0; j < nimport; j++)
	{
#ifndef PMGRID
	  force_treeevaluate_potential(j, 1, &dummy, &dummy);
#else
	  force_treeevaluate_potential_shortrange(j, 1, &dummy, &dummy);
#endif
	}

      if(i >= NumPart)
	ndone_flag = 1;
      else
	ndone_flag = 0;

      MPI_Allreduce(&ndone_flag, &ndone, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

      /* get the result */
      for(ngrp = 1; ngrp < (1 << PTask); ngrp++)
	{
	  sendTask = ThisTask;
	  recvTask = ThisTask ^ ngrp;
	  if(recvTask < NTask)
	    {
	      if(Send_count[recvTask] > 0 || Recv_count[recvTask] > 0)
		{
		  /* send the results */
		  MPI_Sendrecv(&PotDataResult[Recv_offset[recvTask]],
			       Recv_count[recvTask] * sizeof(struct potdata_out),
			       MPI_BYTE, recvTask, TAG_POTENTIAL_B,
			       &PotDataOut[Send_offset[recvTask]],
			       Send_count[recvTask] * sizeof(struct potdata_out),
			       MPI_BYTE, recvTask, TAG_POTENTIAL_B, MPI_COMM_WORLD, &status);
		}
	    }

	}

      /* add the results to the local particles */
      for(j = 0; j < nexport; j++)
	{
	  place = DataIndexTable[j].Index;

	  P[place].p.dPotential += PotDataOut[j].Potential;
	}

      myfree(PotDataOut);
      myfree(PotDataResult);
      myfree(GravDataGet);
    }
  while(ndone < NTask);

  myfree(DataNodeList);
  myfree(DataIndexTable);

  /* add correction to exclude self-potential */

  for(i = 0; i < NumPart; i++)
    {
#ifdef FLTROUNDOFFREDUCTION
      P[i].p.Potential = FLT(P[i].p.dPotential);
#endif
      /* remove self-potential */
      P[i].p.Potential += P[i].Mass / All.SofteningTable[P[i].Type];

      if(All.ComovingIntegrationOn)
	if(All.PeriodicBoundariesOn)
	  P[i].p.Potential -= 2.8372975 * pow(P[i].Mass, 2.0 / 3) *
	    pow(All.Omega0 * 3 * All.Hubble * All.Hubble / (8 * M_PI * All.G), 1.0 / 3);
    }


  /* multiply with the gravitational constant */

  for(i = 0; i < NumPart; i++)
    P[i].p.Potential *= All.G;


#ifdef PMGRID

#ifdef PERIODIC
  pmpotential_periodic();
#ifdef PLACEHIGHRESREGION
  i = pmpotential_nonperiodic(1);
  if(i == 1)			/* this is returned if a particle lied outside allowed range */
    {
      pm_init_regionsize();
      pm_setup_nonperiodic_kernel();
      i = pmpotential_nonperiodic(1);	/* try again */
    }
  if(i == 1)
    endrun(88686);
#endif
#else
  i = pmpotential_nonperiodic(0);
  if(i == 1)			/* this is returned if a particle lied outside allowed range */
    {
      pm_init_regionsize();
      pm_setup_nonperiodic_kernel();
      i = pmpotential_nonperiodic(0);	/* try again */
    }
  if(i == 1)
    endrun(88687);
#ifdef PLACEHIGHRESREGION
  i = pmpotential_nonperiodic(1);
  if(i == 1)			/* this is returned if a particle lied outside allowed range */
    {
      pm_init_regionsize();

      i = pmpotential_nonperiodic(1);
    }
  if(i != 0)
    endrun(88688);
#endif
#endif

#endif



  if(All.ComovingIntegrationOn)
    {
#ifndef PERIODIC
      fac = -0.5 * All.Omega0 * All.Hubble * All.Hubble;

      for(i = 0; i < NumPart; i++)
	{
	  for(k = 0, r2 = 0; k < 3; k++)
	    r2 += P[i].Pos[k] * P[i].Pos[k];

	  P[i].p.Potential += fac * r2;
	}
#endif
    }
  else
    {
      fac = -0.5 * All.OmegaLambda * All.Hubble * All.Hubble;
      if(fac != 0)
	{
	  for(i = 0; i < NumPart; i++)
	    {
	      for(k = 0, r2 = 0; k < 3; k++)
		r2 += P[i].Pos[k] * P[i].Pos[k];

	      P[i].p.Potential += fac * r2;
	    }
	}
    }


  if(ThisTask == 0)
    {
      printf("potential done.\n");
      fflush(stdout);
    }


#else
  for(i = 0; i < NumPart; i++)
    P[i].Potential = 0;
#endif

  CPU_Step[CPU_POTENTIAL] += walltime_measure(WALL_POTENTIAL);
}


#endif
