#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <gsl/gsl_math.h>

#include "allvars.h"
#include "proto.h"

#ifdef VORONOI

#include "voronoi.h"


struct vorodata_in
{
  MyDouble Pos[3];
  MyFloat Hsml;
  int Origin;
  int NodeList[NODELISTLENGTH];
}
 *VoroDataIn, *VoroDataGet;



struct data_primexch_compare
{
  int rank, task, index;
}
 *SortPrimExch, *SortPrimExch2;




void voronoi_setup_exchange(void)
{
  int listp;
  struct indexexch
  {
    int task, index;
  } *tmpIndexExch, *IndexExch;
  int i, j, p, task, off, count;
  int ngrp, sendTask, recvTask, place;


  for(j = 0; j < NTask; j++)
    Mesh_Send_count[j] = 0;

  for(p = 0; p < N_gas; p++)
    {
      listp = List_P[p].firstexport;
      while(listp >= 0)
	{
	  if(ListExports[listp].origin != ThisTask)
	    {
	      Mesh_Send_count[ListExports[listp].origin]++;
	    }
	  listp = ListExports[listp].nextexport;
	}
    }

  MPI_Alltoall(Mesh_Send_count, 1, MPI_INT, Mesh_Recv_count, 1, MPI_INT, MPI_COMM_WORLD);

  for(j = 0, Mesh_nimport = 0, Mesh_nexport = 0, Mesh_Recv_offset[0] = 0, Mesh_Send_offset[0] = 0; j < NTask;
      j++)
    {
      Mesh_nimport += Mesh_Recv_count[j];
      Mesh_nexport += Mesh_Send_count[j];

      if(j > 0)
	{
	  Mesh_Send_offset[j] = Mesh_Send_offset[j - 1] + Mesh_Send_count[j - 1];
	  Mesh_Recv_offset[j] = Mesh_Recv_offset[j - 1] + Mesh_Recv_count[j - 1];
	}
    }

  IndexExch = (struct indexexch *) mymalloc("IndexExch", Mesh_nimport * sizeof(struct indexexch));
  tmpIndexExch = (struct indexexch *) mymalloc("tmpIndexExch", Mesh_nexport * sizeof(struct indexexch));

  /* prepare data for export */
  for(j = 0; j < NTask; j++)
    Mesh_Send_count[j] = 0;

  for(p = 0; p < N_gas; p++)
    {
      listp = List_P[p].firstexport;
      while(listp >= 0)
	{
	  if((task = ListExports[listp].origin) != ThisTask)
	    {
	      place = ListExports[listp].index;
	      off = Mesh_Send_offset[task] + Mesh_Send_count[task]++;

	      tmpIndexExch[off].task = ThisTask;
	      tmpIndexExch[off].index = place;
	    }
	  listp = ListExports[listp].nextexport;
	}
    }

  /* exchange data */
  for(ngrp = 1; ngrp < (1 << PTask); ngrp++)
    {
      sendTask = ThisTask;
      recvTask = ThisTask ^ ngrp;

      if(recvTask < NTask)
	{
	  if(Mesh_Send_count[recvTask] > 0 || Mesh_Recv_count[recvTask] > 0)
	    {
	      /* get the particles */
	      MPI_Sendrecv(&tmpIndexExch[Mesh_Send_offset[recvTask]], Mesh_Send_count[recvTask]
			   * sizeof(struct indexexch), MPI_BYTE, recvTask, TAG_DENS_A,
			   &IndexExch[Mesh_Recv_offset[recvTask]],
			   Mesh_Recv_count[recvTask] * sizeof(struct indexexch), MPI_BYTE, recvTask,
			   TAG_DENS_A, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	    }
	}
    }

  myfree(tmpIndexExch);

  /* now we need to associate the imported data with the points stored in the DP[] array */

  SortPrimExch = (struct data_primexch_compare *) mymalloc("SortPrimExch", Mesh_nimport
							   * sizeof(struct data_primexch_compare));

  for(i = 0; i < Mesh_nimport; i++)
    {
      SortPrimExch[i].rank = i;
      SortPrimExch[i].task = IndexExch[i].task;
      SortPrimExch[i].index = IndexExch[i].index;
    }

  /* let sort the data according to task and index */
  qsort(SortPrimExch, Mesh_nimport, sizeof(struct data_primexch_compare), compare_primexch);

  SortPrimExch2 =
    (struct data_primexch_compare *) mymalloc("SortPrimExch2", Ndp * sizeof(struct data_primexch_compare));

  for(i = 0, count = 0; i < Ndp; i++)
    {
      if(DP[i].task != ThisTask)
	{
	  SortPrimExch2[count].rank = i;
	  SortPrimExch2[count].task = DP[i].task;
	  SortPrimExch2[count].index = DP[i].index;
	  count++;
	}
    }

  /* let sort according to task and index */
  qsort(SortPrimExch2, count, sizeof(struct data_primexch_compare), compare_primexch);

  /* count can be larger than nimport because a foreigh particle can appear
     multiple times on the local domain, due to periodicity */

  for(i = 0, j = 0; i < count; i++)
    {
      if(SortPrimExch2[i].task != SortPrimExch[j].task || SortPrimExch2[i].index != SortPrimExch[j].index)
	j++;

      if(j >= Mesh_nimport)
	terminate("j >= Mesh_nimport");

      DP[SortPrimExch2[i].rank].index = SortPrimExch[j].rank;	/* note: this change is now permanent and available for next exchange */
    }

  myfree(SortPrimExch2);
  myfree(SortPrimExch);
  myfree(IndexExch);
}


void voronoi_exchange_ghost_variables(void)
{
  int listp;
  struct primexch *tmpPrimExch;
  int j, p, task, off;
  int ngrp, sendTask, recvTask, place;


  PrimExch = (struct primexch *) mymalloc("PrimExch", Mesh_nimport * sizeof(struct primexch));
#ifdef VORONOI_MESHRELAX
  GradExch = (struct grad_data *) mymalloc("GradExch", Mesh_nimport * sizeof(struct grad_data));
#endif
  tmpPrimExch = (struct primexch *) mymalloc("tmpPrimExch", Mesh_nexport * sizeof(struct primexch));


  /* prepare data for export */
  for(j = 0; j < NTask; j++)
    Mesh_Send_count[j] = 0;

  for(p = 0; p < N_gas; p++)
    {
      if(P[p].Type == 0)
	{
	  listp = List_P[p].firstexport;
	  
	  while(listp >= 0)
	    {
	      if((task = ListExports[listp].origin) != ThisTask)
		{
		  place = ListExports[listp].index;
		  off = Mesh_Send_offset[task] + Mesh_Send_count[task]++;
		  
		  tmpPrimExch[off].Pressure = SphP[place].Pressure;
		  tmpPrimExch[off].Mass = P[place].Mass;
		  tmpPrimExch[off].Density = SphP[place].d.Density;
		  tmpPrimExch[off].Entropy = SphP[place].Entropy;
		  tmpPrimExch[off].Volume = SphP[place].Volume;
#ifdef VORONOI_SHAPESCHEME
		  tmpPrimExch[off].W = SphP[place].W;
#endif
		  for(j = 0; j < 3; j++)
		    {
		      tmpPrimExch[off].VelPred[j] = SphP[place].VelPred[j];
		      tmpPrimExch[off].Center[j] = SphP[place].Center[j];
		    }
		}
	  listp = ListExports[listp].nextexport;
	    }
	}
    }


  /* exchange data */
  for(ngrp = 1; ngrp < (1 << PTask); ngrp++)
    {
      sendTask = ThisTask;
      recvTask = ThisTask ^ ngrp;

      if(recvTask < NTask)
	{
	  if(Mesh_Send_count[recvTask] > 0 || Mesh_Recv_count[recvTask] > 0)
	    {
	      /* get the particles */
	      MPI_Sendrecv(&tmpPrimExch[Mesh_Send_offset[recvTask]], Mesh_Send_count[recvTask]
			   * sizeof(struct primexch), MPI_BYTE, recvTask, TAG_DENS_A,
			   &PrimExch[Mesh_Recv_offset[recvTask]],
			   Mesh_Recv_count[recvTask] * sizeof(struct primexch), MPI_BYTE, recvTask,
			   TAG_DENS_A, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	    }
	}
    }

  myfree(tmpPrimExch);
}


#ifdef VORONOI_MESHRELAX

void voronoi_exchange_gradients(void)
{
  int listp;
  struct grad_data *tmpGradExch;
  struct primexch *tmpPrimExch;
  int j, p, task, off;
  int ngrp, sendTask, recvTask, place;

  tmpPrimExch = (struct primexch *) mymalloc("tmpPrimExch", Mesh_nexport * sizeof(struct primexch));
  tmpGradExch = (struct grad_data *) mymalloc("tmpGradExch", Mesh_nexport * sizeof(struct grad_data));

  /* prepare data for export */
  for(j = 0; j < NTask; j++)
    Mesh_Send_count[j] = 0;

  for(p = 0; p < N_gas; p++)
    {
      listp = List_P[p].firstexport;
      while(listp >= 0)
	{
	  if((task = ListExports[listp].origin) != ThisTask)
	    {
	      place = ListExports[listp].index;
	      off = Mesh_Send_offset[task] + Mesh_Send_count[task]++;

	      tmpPrimExch[off].Pressure = SphP[place].Pressure;
	      tmpPrimExch[off].Mass = P[place].Mass;
	      tmpPrimExch[off].Density = SphP[place].d.Density;
	      tmpPrimExch[off].Entropy = SphP[place].Entropy;

	      for(j = 0; j < 3; j++)
		{
		  tmpPrimExch[off].VelPred[j] = SphP[place].VelPred[j];
		  tmpPrimExch[off].HydroAccel[j] = SphP[place].a.HydroAccel[j];
		  tmpPrimExch[off].Center[j] = SphP[place].Center[j];
		}

	      tmpGradExch[off] = Grad[place];
	    }
	  listp = ListExports[listp].nextexport;
	}
    }


  /* exchange data */
  for(ngrp = 1; ngrp < (1 << PTask); ngrp++)
    {
      sendTask = ThisTask;
      recvTask = ThisTask ^ ngrp;

      if(recvTask < NTask)
	{
	  if(Mesh_Send_count[recvTask] > 0 || Mesh_Recv_count[recvTask] > 0)
	    {
	      /* exchange the data */
	      MPI_Sendrecv(&tmpPrimExch[Mesh_Send_offset[recvTask]], Mesh_Send_count[recvTask]
			   * sizeof(struct primexch), MPI_BYTE, recvTask, TAG_DENS_A,
			   &PrimExch[Mesh_Recv_offset[recvTask]],
			   Mesh_Recv_count[recvTask] * sizeof(struct primexch), MPI_BYTE, recvTask,
			   TAG_DENS_A, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

	      MPI_Sendrecv(&tmpGradExch[Mesh_Send_offset[recvTask]], Mesh_Send_count[recvTask]
			   * sizeof(struct grad_data), MPI_BYTE, recvTask, TAG_HYDRO_A,
			   &GradExch[Mesh_Recv_offset[recvTask]],
			   Mesh_Recv_count[recvTask] * sizeof(struct grad_data), MPI_BYTE, recvTask,
			   TAG_HYDRO_A, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	    }
	}
    }


  myfree(tmpGradExch);
  myfree(tmpPrimExch);

  /* note: because the sequence is the same as before, we don't have to do the sorts again */
}
#endif






int compare_primexch(const void *a, const void *b)
{
  if(((struct data_primexch_compare *) a)->task < ((struct data_primexch_compare *) b)->task)
    return -1;

  if(((struct data_primexch_compare *) a)->task > ((struct data_primexch_compare *) b)->task)
    return +1;

  if(((struct data_primexch_compare *) a)->index < ((struct data_primexch_compare *) b)->index)
    return -1;

  if(((struct data_primexch_compare *) a)->index > ((struct data_primexch_compare *) b)->index)
    return +1;

  return 0;
}

#endif
