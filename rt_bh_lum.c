#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <gsl/gsl_math.h>

#include "allvars.h"
#include "proto.h"


#ifdef RADTRANSFER
#if defined(EDDINGTON_TENSOR_BH) && defined(BLACK_HOLES)

static struct bhdata_in
{
  MyDouble Pos[3], Density, Mass, BH_Mdot;
  MyFloat Hsml;
  int NodeList[NODELISTLENGTH];
}
 *BHDataIn, *BHDataGet;

#define  eV_to_erg    1.60184e-12

void bh_lum(void)
{
  int j;
  int i, dummy;
  int ngrp, sendTask, recvTask, place, nexport, nimport, ndone, ndone_flag;

  /* clear Je in all gas particles */

  for(j = 0; j < N_gas; j++)
    if(P[j].Type == 0)
      for(i = 0; i < N_BINS; i++)
      {
	SphP[j].Je[i] = 0;
      }

  /* allocate buffers to arrange communication */

  Ngblist = (int *) mymalloc("Ngblist", NumPart * sizeof(int));

  All.BunchSize =
    (int) ((All.BufferSize * 1024 * 1024) /
	   (sizeof(struct data_index) + sizeof(struct data_nodelist) + 2 * sizeof(struct bhdata_in)));

  DataIndexTable =
    (struct data_index *) mymalloc("DataIndexTable", All.BunchSize * sizeof(struct data_index));
  DataNodeList =
    (struct data_nodelist *) mymalloc("DataNodeList", All.BunchSize * sizeof(struct data_nodelist));

  i = FirstActiveParticle;	/* beginn with this index */

  do
    {
      for(j = 0; j < NTask; j++)
	{
	  Send_count[j] = 0;
	  Exportflag[j] = -1;
	}

      /* do local particles and prepare export list */
      for(nexport = 0; i >= 0; i = NextActiveParticle[i])
	{
	  if(P[i].Type == 5)
	    {
	      if(bh_lum_evaluate(i, 0, &nexport, Send_count) < 0)
		break;
	    }
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

      BHDataGet = (struct bhdata_in *) mymalloc("BHDataGet", nimport * sizeof(struct bhdata_in));
      BHDataIn = (struct bhdata_in *) mymalloc("BHDataIn", nexport * sizeof(struct bhdata_in));

      /* prepare particle data for export */
      for(j = 0; j < nexport; j++)
	{
	  place = DataIndexTable[j].Index;

	  BHDataIn[j].Pos[0] = P[place].Pos[0];
	  BHDataIn[j].Pos[1] = P[place].Pos[1];
	  BHDataIn[j].Pos[2] = P[place].Pos[2];
	  BHDataIn[j].Hsml = PPP[place].Hsml;
	  BHDataIn[j].Density = P[place].b1.BH_Density;
	  BHDataIn[j].Mass = P[place].Mass;
	  BHDataIn[j].BH_Mdot = P[place].BH_Mdot;

	  memcpy(BHDataIn[j].NodeList,
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
		  MPI_Sendrecv(&BHDataIn[Send_offset[recvTask]],
			       Send_count[recvTask] * sizeof(struct bhdata_in), MPI_BYTE,
			       recvTask, TAG_DENS_A,
			       &BHDataGet[Recv_offset[recvTask]],
			       Recv_count[recvTask] * sizeof(struct bhdata_in), MPI_BYTE,
			       recvTask, TAG_DENS_A, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}
	    }
	}

      myfree(BHDataIn);


      /* now do the particles that were sent to us */

      for(j = 0; j < nimport; j++)
	bh_lum_evaluate(j, 1, &dummy, &dummy);

      /* check whether this is the last iteration */
      if(i < 0)
	ndone_flag = 1;
      else
	ndone_flag = 0;

      MPI_Allreduce(&ndone_flag, &ndone, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

      myfree(BHDataGet);
    }
  while(ndone < NTask);

  myfree(DataNodeList);
  myfree(DataIndexTable);
  myfree(Ngblist);
}


int bh_lum_evaluate(int target, int mode, int *nexport, int *nsend_local)
{
  int j, n, numngb;
  int startnode, listindex = 0;
  double h, hinv, h2, mass_j, weight, hinv3;
  double wk, mass, mdot, density, lum, hv;
  double dx, dy, dz, r, r2, u, a3inv;
  MyDouble *pos;

#ifdef PERIODIC
  double boxsize, boxhalf;

  boxsize = All.BoxSize;
  boxhalf = 0.5 * All.BoxSize;
#endif

  if(All.ComovingIntegrationOn)
    a3inv = 1.0 / (All.Time * All.Time * All.Time);
  else
    a3inv = 1.0;

  if(mode == 0)
    {
      pos = P[target].Pos;
      h = PPP[target].Hsml;
      density = P[target].b1.BH_Density;
      mdot = P[target].BH_Mdot;
      mass = P[target].Mass;
    }
  else
    {
      pos = BHDataGet[target].Pos;
      h = BHDataGet[target].Hsml;
      density = BHDataGet[target].Density;
      mass = BHDataGet[target].Mass;
      mdot = BHDataGet[target].BH_Mdot;
    }

  h2 = h * h;
  hinv = 1.0 / h;
  hinv3 = hinv * hinv * hinv;

  hv = 13.6 * eV_to_erg / All.UnitEnergy_in_cgs * All.HubbleParam;
  lum = 0.0001 * mdot * pow(C / All.UnitVelocity_in_cm_per_s, 2) / hv;

  if(mode == 0)
    {
      startnode = All.MaxPart;	/* root node */
    }
  else
    {
      startnode = BHDataGet[target].NodeList[0];
      startnode = Nodes[startnode].u.d.nextnode;	/* open it */
    }

  while(startnode >= 0)
    {
      while(startnode >= 0)
	{
	  numngb = ngb_treefind_variable(pos, h, target, &startnode, mode, nexport, nsend_local);

	  if(numngb < 0)
	    return -1;

	  for(n = 0; n < numngb; n++)
	    {
	      j = Ngblist[n];

	      dx = pos[0] - P[j].Pos[0];
	      dy = pos[1] - P[j].Pos[1];
	      dz = pos[2] - P[j].Pos[2];
#ifdef PERIODIC
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
	      r = sqrt(r2);

	      if(r2 < h2)
		{
		  u = r * hinv;

		  if(u < 0.5)
		    wk = hinv3 * (KERNEL_COEFF_1 + KERNEL_COEFF_2 * (u - 1) * u * u);
		  else
		    wk = hinv3 * KERNEL_COEFF_5 * (1.0 - u) * (1.0 - u) * (1.0 - u);

		}
	      else
		wk = 0;

	      mass_j = P[j].Mass;

	      weight = mass_j * wk / density;

	      SphP[j].Je[0] += lum * weight / HYDROGEN_MASSFRAC / mass_j * (PROTONMASS / SOLAR_MASS);
	      SphP[j].Je[1] +=
		lum * weight / (1.0 - HYDROGEN_MASSFRAC) / mass_j * (4.0 * PROTONMASS / SOLAR_MASS) / 100.0;
	    }
	}

      if(mode == 1)
	{
	  listindex++;
	  if(listindex < NODELISTLENGTH)
	    {
	      startnode = BHDataGet[target].NodeList[listindex];
	      if(startnode >= 0)
		startnode = Nodes[startnode].u.d.nextnode;	/* open it */
	    }
	}
    }

  return 0;
}

#endif
#endif
