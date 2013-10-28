#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "allvars.h"
#include "proto.h"


#ifdef SUBFIND

#include "fof.h"
#include "subfind.h"



void subfind_potential_compute(int num, struct unbind_data *d, int phase, double weakly_bound_limit)
{
  int i, j, k, sendTask, recvTask;
  int ndone, ndone_flag, dummy;
  int ngrp, place, nexport, nimport;
  double atime;

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


  i = 0;			/* beginn with this index */

  do
    {
      for(j = 0; j < NTask; j++)
	{
	  Send_count[j] = 0;
	  Exportflag[j] = -1;
	}

      /* do local particles and prepare export list */
      for(nexport = 0; i < num; i++)
	{
	  if(phase == 1)
	    if(P[d[i].index].v.DM_BindingEnergy <= weakly_bound_limit)
	      continue;

	  if(subfind_force_treeevaluate_potential(d[i].index, 0, &nexport, Send_count) < 0)
	    break;
	}

      qsort(DataIndexTable, nexport, sizeof(struct data_index), data_index_compare);

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

	  for(k = 0; k < NODELISTLENGTH; k++)
	    GravDataIn[j].NodeList[k] = DataNodeList[DataIndexTable[j].IndexGet].NodeList[k];

#ifdef UNEQUALSOFTENINGS
	  GravDataIn[j].Type = P[place].Type;
#ifdef ADAPTIVE_GRAVSOFT_FORGAS
	  if(P[place].Type == 0)
	    GravDataIn[j].Soft = SPHP(place).Hsml;
#endif
#endif
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
			       recvTask, TAG_POTENTIAL_A, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}
	    }
	}

      myfree(GravDataIn);
      PotDataResult = (struct potdata_out *) mymalloc("PotDataResult", nimport * sizeof(struct potdata_out));
      PotDataOut = (struct potdata_out *) mymalloc("PotDataOut", nexport * sizeof(struct potdata_out));


      /* now do the particles that were sent to us */
      for(j = 0; j < nimport; j++)
	subfind_force_treeevaluate_potential(j, 1, &dummy, &dummy);


      if(i >= num)
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
			       MPI_BYTE, recvTask, TAG_POTENTIAL_B, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}
	    }

	}

      /* add the results to the local particles */
      for(j = 0; j < nexport; j++)
	{
	  place = DataIndexTable[j].Index;

	  P[place].u.DM_Potential += PotDataOut[j].Potential;
	}

      myfree(PotDataOut);
      myfree(PotDataResult);
      myfree(GravDataGet);
    }
  while(ndone < NTask);


  if(All.ComovingIntegrationOn)
    atime = All.Time;
  else
    atime = 1;

  for(i = 0; i < num; i++)
    {
      if(phase == 1)
	if(P[d[i].index].v.DM_BindingEnergy <= weakly_bound_limit)
	  continue;

      P[d[i].index].u.DM_Potential += P[d[i].index].Mass / All.SofteningTable[P[d[i].index].Type];
      P[d[i].index].u.DM_Potential *= All.G / atime;

      if(All.TotN_sph > 0 && (FOF_SECONDARY_LINK_TYPES & 1) == 0 && 
	 (FOF_PRIMARY_LINK_TYPES & 1) == 0 && All.OmegaBaryon > 0)
  	P[d[i].index].u.DM_Potential *= All.Omega0 / (All.Omega0 - All.OmegaBaryon);
    }

  myfree(DataNodeList);
  myfree(DataIndexTable);
}

#endif
