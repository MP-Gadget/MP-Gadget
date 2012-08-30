#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <gsl/gsl_math.h>


#include "allvars.h"
#include "proto.h"


#ifdef RADTRANSFER


/*structures for eddington tensor*/
struct eddingtondata_in
{
  int NodeList[NODELISTLENGTH];
  MyDouble Pos[3];
  MyFloat Hsml;
}
 *EddingtonDataIn, *EddingtonDataGet;


struct eddingtondata_out
{
  MyFloat ET[6];
}
 *EddingtonDataResult, *EddingtonDataOut;

/* eddington tensor computation and arrangement of particle communication*/
void eddington(void)
{
  int i, j, k, ngrp, dummy;
  int sendTask, recvTask, nexport, nimport, place, ndone, ndone_flag;
  MyLongDouble trace;

  /* allocate buffers to arrange communication */

  All.BunchSize =
    (int) ((All.BufferSize * 1024 * 1024) / (sizeof(struct data_index) + sizeof(struct data_nodelist) +
					     sizeof(struct eddingtondata_in) +
					     sizeof(struct eddingtondata_out) +
					     sizemax(sizeof(struct eddingtondata_in),
						     sizeof(struct eddingtondata_out))));
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
	if(P[i].Type == 0)
	  {
	    if(eddington_treeevaluate(i, 0, &nexport, Send_count) < 0)
	      break;
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

      EddingtonDataGet =
	(struct eddingtondata_in *) mymalloc("EddingtonDataGet", nimport * sizeof(struct eddingtondata_in));
      EddingtonDataIn =
	(struct eddingtondata_in *) mymalloc("EddingtonDataIn", nexport * sizeof(struct eddingtondata_in));

      /* prepare particle data for export */

      for(j = 0; j < nexport; j++)
	{
	  place = DataIndexTable[j].Index;

	  for(k = 0; k < 3; k++)
	    EddingtonDataIn[j].Pos[k] = P[place].Pos[k];

	  EddingtonDataIn[j].Hsml = PPP[place].Hsml;

	  memcpy(EddingtonDataIn[j].NodeList,
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
		  MPI_Sendrecv(&EddingtonDataIn[Send_offset[recvTask]],
			       Send_count[recvTask] * sizeof(struct eddingtondata_in), MPI_BYTE,
			       recvTask, TAG_HYDRO_A,
			       &EddingtonDataGet[Recv_offset[recvTask]],
			       Recv_count[recvTask] * sizeof(struct eddingtondata_in), MPI_BYTE,
			       recvTask, TAG_HYDRO_A, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}
	    }
	}

      myfree(EddingtonDataIn);
      EddingtonDataResult =
	(struct eddingtondata_out *) mymalloc("EddingtonDataResult",
					      nimport * sizeof(struct eddingtondata_out));
      EddingtonDataOut =
	(struct eddingtondata_out *) mymalloc("EddingtonDataOut", nexport * sizeof(struct eddingtondata_out));



      /* now do the particles that were sent to us */
      for(j = 0; j < nimport; j++)
	{
	  eddington_treeevaluate(j, 1, &dummy, &dummy);
	}

      /* check whether this is the last iteration */
      if(i < 0)
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
		  MPI_Sendrecv(&EddingtonDataResult[Recv_offset[recvTask]],
			       Recv_count[recvTask] * sizeof(struct eddingtondata_out),
			       MPI_BYTE, recvTask, TAG_HYDRO_B,
			       &EddingtonDataOut[Send_offset[recvTask]],
			       Send_count[recvTask] * sizeof(struct eddingtondata_out),
			       MPI_BYTE, recvTask, TAG_HYDRO_B, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}
	    }
	}

      /* add the result to the local particles */
      for(j = 0; j < nexport; j++)
	{
	  place = DataIndexTable[j].Index;
	  for(k = 0; k < 6; k++)
	    SphP[place].ET[k] += EddingtonDataOut[j].ET[k];
	}

      myfree(EddingtonDataOut);
      myfree(EddingtonDataResult);
      myfree(EddingtonDataGet);

    }
  while(ndone < NTask);

  myfree(DataNodeList);
  myfree(DataIndexTable);

  /* do final operations divide by the trace */
  for(i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i])
    if(P[i].Type == 0)
      {
	trace = SphP[i].ET[0] + SphP[i].ET[1] + SphP[i].ET[2];

	if(trace)
	  for(k = 0; k < 6; k++)
	    SphP[i].ET[k] /= trace;
	else
	  {
	    for(k = 0; k < 6; k++)
	      SphP[i].ET[k] = 0.0;
	    SphP[i].ET[0] = SphP[i].ET[1] = SphP[i].ET[2] = 0.333;
	  }
      }

}


/*! This routine computes the eddington tensor ET for a given local
 *  particle, or for a particle in the communication buffer. Depending on
 *  the value of TypeOfOpeningCriterion, either the geometrical BH
 *  cell-opening criterion, or the `relative' opening criterion is used.
 */
int eddington_treeevaluate(int target, int mode, int *nexport, int *nsend_local)
{
  struct NODE *nop = 0;
  int k, no, nodesinlist, nexport_save, ninteractions, task, listindex = 0, ptype;
  double r4, r2, r4_invfac, dx = 0, dy = 0, dz = 0, stellar_mass = 0, h;
  double pos_x, pos_y, pos_z;
  MyLongDouble ET[6] = { 0, 0, 0, 0, 0, 0 };

#ifdef ADAPTIVE_GRAVSOFT_FORGAS
  double soft = 0;
#endif
#ifdef PERIODIC
  double boxsize, boxhalf;

  boxsize = All.BoxSize;
  boxhalf = 0.5 * All.BoxSize;
#endif
  nexport_save = *nexport;

  ninteractions = 0;
  nodesinlist = 0;
  ptype = 0;			/* we only deal with gas particles */

  if(mode == 0)
    {
      pos_x = P[target].Pos[0];
      pos_y = P[target].Pos[1];
      pos_z = P[target].Pos[2];
#ifdef ADAPTIVE_GRAVSOFT_FORGAS
      soft = P[target].Hsml;
#endif
    }
  else				/* mode=1 */
    {
      pos_x = EddingtonDataGet[target].Pos[0];
      pos_y = EddingtonDataGet[target].Pos[1];
      pos_z = EddingtonDataGet[target].Pos[2];
#ifdef ADAPTIVE_GRAVSOFT_FORGAS
      soft = EddingtonDataGet[target].Hsml;
#endif
    }

#ifndef UNEQUALSOFTENINGS
  h = All.ForceSoftening[ptype];
#endif

  if(mode == 0)
    {
      no = All.MaxPart;		/* root node */
    }
  else
    {
      nodesinlist++;
      no = EddingtonDataGet[target].NodeList[0];
      no = Nodes[no].u.d.nextnode;	/* open it */
    }

  while(no >= 0)
    {
      while(no >= 0)
	{
	  if(no < All.MaxPart)	/* single particle */
	    {
	      /* the index of the node is the index of the particle */
	      /* observe the sign */

#ifdef EDDINGTON_TENSOR_STARS
	      if(P[no].Type == 4 || P[no].Type == 3 || P[no].Type == 2)
		{
		  dx = P[no].Pos[0] - pos_x;
		  dy = P[no].Pos[1] - pos_y;
		  dz = P[no].Pos[2] - pos_z;
		  stellar_mass = P[no].Mass;
		}
	      else
		{
		  dx = 0;
		  dy = 0;
		  dz = 0;
		  stellar_mass = 0;
		}
#endif

#if defined(SFR) && defined(EDDINGTON_TENSOR_SFR)
	      if(P[no].Type == 0)
		{
		  double a;

		  if(All.ComovingIntegrationOn == 1)
		    a = All.Time;
		  else
		    a = 1.0;

		  if((SphP[no].d.Density / (a * a * a)) >= All.PhysDensThresh)
		    {
		      dx = P[no].Pos[0] - pos_x;
		      dy = P[no].Pos[1] - pos_y;
		      dz = P[no].Pos[2] - pos_z;
		      stellar_mass = P[no].Mass;
		    }
		}
	      else
		{
		  dx = 0;
		  dy = 0;
		  dz = 0;
		  stellar_mass = 0;
		}
#endif

#if defined(EDDINGTON_TENSOR_GAS)
	      if(P[no].Type == 0)
		{
		  dx = P[no].Pos[0] - pos_x;
		  dy = P[no].Pos[1] - pos_y;
		  dz = P[no].Pos[2] - pos_z;
		  stellar_mass = P[no].Mass;
		}
	      else
		{
		  dx = 0;
		  dy = 0;
		  dz = 0;
		  stellar_mass = 0;
		}
#endif

#if defined(BLACK_HOLES) && defined(EDDINGTON_TENSOR_BH)
	      if(P[no].Type == 5)
		{
		  dx = P[no].Pos[0] - pos_x;
		  dy = P[no].Pos[1] - pos_y;
		  dz = P[no].Pos[2] - pos_z;
		  stellar_mass = P[no].Mass;
		}
	      else
		{
		  dx = 0;
		  dy = 0;
		  dz = 0;
		  stellar_mass = 0;
		}
#endif
	    }

	  else			/* not a single particle */
	    {
	      if(no >= All.MaxPart + MaxNodes)	/* pseudo particle */
		{
		  if(mode == 0)
		    {
		      if(Exportflag[task = DomainTask[no - (All.MaxPart + MaxNodes)]] != target)
			{
			  Exportflag[task] = target;
			  Exportnodecount[task] = NODELISTLENGTH;
			}

		      if(Exportnodecount[task] == NODELISTLENGTH)
			{
			  if(*nexport >= All.BunchSize)
			    {
			      /* out of buffer space. Need to discard work for this particle and interrupt */
			      *nexport = nexport_save;
			      if(nexport_save == 0)
				endrun(17998);	/* in this case, the buffer is too small to process even a single particle */
			      for(task = 0; task < NTask; task++)
				nsend_local[task] = 0;
			      for(no = 0; no < nexport_save; no++)
				nsend_local[DataIndexTable[no].Task]++;
			      return -1;
			    }
			  Exportnodecount[task] = 0;
			  Exportindex[task] = *nexport;
			  DataIndexTable[*nexport].Task = task;
			  DataIndexTable[*nexport].Index = target;
			  DataIndexTable[*nexport].IndexGet = *nexport;
			  *nexport = *nexport + 1;
			  nsend_local[task]++;
			}

		      DataNodeList[Exportindex[task]].NodeList[Exportnodecount[task]++] =
			DomainNodeIndex[no - (All.MaxPart + MaxNodes)];

		      if(Exportnodecount[task] < NODELISTLENGTH)
			DataNodeList[Exportindex[task]].NodeList[Exportnodecount[task]] = -1;
		    }
		  no = Nextnode[no - MaxNodes];
		  continue;
		}

	      nop = &Nodes[no];

	      if(mode == 1)
		{
		  if(nop->u.d.bitflags & (1 << BITFLAG_TOPLEVEL))	/* we reached a top-level node again, which means that we are done with the branch */
		    {
		      no = -1;
		      continue;
		    }
		}

	      stellar_mass = nop->stellar_mass;

	      if(!(nop->u.d.bitflags & (1 << BITFLAG_MULTIPLEPARTICLES)))
		{
		  /* open cell */
		  if(stellar_mass)
		    {
		      no = nop->u.d.nextnode;
		      continue;
		    }
		}

	      dx = nop->stellar_s[0] - pos_x;
	      dy = nop->stellar_s[1] - pos_y;
	      dz = nop->stellar_s[2] - pos_z;
	    }

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

	  r2 = (dx * dx) + (dy * dy) + (dz * dz);
	  r4 = r2 * r2;

	  if(no < All.MaxPart)
	    {
#ifdef UNEQUALSOFTENINGS
#ifdef ADAPTIVE_GRAVSOFT_FORGAS
	      if(ptype == 0)
		h = soft;
	      else
		h = All.ForceSoftening[ptype];

	      if(P[no].Type == 0)
		{
		  if(h < PPP[no].Hsml)
		    h = PPP[no].Hsml;
		}
	      else
		{
		  if(h < All.ForceSoftening[P[no].Type])
		    h = All.ForceSoftening[P[no].Type];
		}
#else
	      h = All.ForceSoftening[ptype];
	      if(h < All.ForceSoftening[P[no].Type])
		h = All.ForceSoftening[P[no].Type];
#endif
#endif
	      no = Nextnode[no];
	    }
	  else			/* we have an  internal node. Need to check opening criterion */
	    {
	      if(All.ErrTolTheta)	/* check Barnes-Hut opening criterion */
		{
		  if(nop->len * nop->len > r2 * All.ErrTolTheta * All.ErrTolTheta)
		    {
		      /* open cell */
		      no = nop->u.d.nextnode;
		      continue;
		    }
		}
	      else
		{
		  /* check in addition whether we lie inside the cell */

		  if(fabs(nop->center[0] - pos_x) < 0.60 * nop->len)
		    {
		      if(fabs(nop->center[1] - pos_y) < 0.60 * nop->len)
			{
			  if(fabs(nop->center[2] - pos_z) < 0.60 * nop->len)
			    {
			      no = nop->u.d.nextnode;
			      continue;
			    }
			}
		    }
		}
#ifdef UNEQUALSOFTENINGS
#ifndef ADAPTIVE_GRAVSOFT_FORGAS
	      h = All.ForceSoftening[ptype];
	      if(h < All.ForceSoftening[extract_max_softening_type(nop->u.d.bitflags)])
		{
		  h = All.ForceSoftening[extract_max_softening_type(nop->u.d.bitflags)];
		  if(r2 < h * h)
		    {
		      if(maskout_different_softening_flag(nop->u.d.bitflags))	/* signals that there are particles of different softening in the node */
			{
			  no = nop->u.d.nextnode;
			  continue;
			}
		    }
		}
#else
	      if(ptype == 0)
		h = soft;
	      else
		h = All.ForceSoftening[ptype];

	      if(h < nop->maxsoft)
		{
		  h = nop->maxsoft;
		  if(r2 < h * h)
		    {
		      no = nop->u.d.nextnode;
		      continue;
		    }
		}
#endif
#endif
	      no = nop->u.d.sibling;	/* ok, node can be used */

	    }

	  r4_invfac = 1.0 / (r4 + h * h * h * h);

	  ET[0] += stellar_mass * dx * dx * r4_invfac;
	  ET[1] += stellar_mass * dy * dy * r4_invfac;
	  ET[2] += stellar_mass * dz * dz * r4_invfac;
	  ET[3] += stellar_mass * dx * dy * r4_invfac;
	  ET[4] += stellar_mass * dy * dz * r4_invfac;
	  ET[5] += stellar_mass * dz * dx * r4_invfac;

	  if(stellar_mass > 0)
	    ninteractions++;
	}
      if(mode == 1)
	{
	  listindex++;
	  if(listindex < NODELISTLENGTH)
	    {
	      no = EddingtonDataGet[target].NodeList[listindex];
	      if(no >= 0)
		{
		  nodesinlist++;
		  no = Nodes[no].u.d.nextnode;	/* open it */
		}
	    }
	}
    }

  /* store result at the proper place */
  if(mode == 0)
    {
      for(k = 0; k < 6; k++)
	SphP[target].ET[k] = ET[k];
    }
  else
    {
      for(k = 0; k < 6; k++)
	EddingtonDataResult[target].ET[k] = ET[k];
      *nexport = nodesinlist;
    }


  return ninteractions;
}


#endif
