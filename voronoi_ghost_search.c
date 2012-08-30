#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <gsl/gsl_math.h>

#ifdef VORONOI

#include "allvars.h"
#include "proto.h"
#include "voronoi.h"

#ifndef ALTERNATIVE_GHOST_SEARCH

static struct vorodata_in
{
  MyDouble Pos[3];
  MyDouble RefPos[3];
  MyFloat MaxDist;
  int Origin;
  int NodeList[NODELISTLENGTH];
#ifdef EXTENDED_GHOST_SEARCH
  unsigned char BitFlagList[NODELISTLENGTH];
#endif
} *VoroDataGet, *VoroDataIn;

static struct vorodata_out
{
  int Count;			/* counts how many have been found */
} *VoroDataResult, *VoroDataOut;

static struct data_nodelist_special
{
  int NodeList[NODELISTLENGTH];
#ifdef EXTENDED_GHOST_SEARCH
  unsigned char BitFlagList[NODELISTLENGTH];
#endif
} *DataNodeListSpecial;

static point *DP_Buffer;

static int MaxN_DP_Buffer, N_DP_Buffer;

static int NadditionalPoints;

static int *send_count_new;

int voronoi_ghost_search(void)
{
  int i, j, k, q, ndone, ndone_flag, dummy;
  int ngrp, sendTask, recvTask, place, nexport, nimport;

  NadditionalPoints = 0;

  /* allocate buffers to arrange communication */

  send_count_new = (int *) mymalloc_movable(&send_count_new, "send_count_new", NTask * sizeof(int));

  All.BunchSize = (int) ((All.BufferSize * 1024 * 1024) / (sizeof(struct data_index)
							   + sizeof(struct data_nodelist) +
							   2 * sizeof(struct vorodata_in)));
  DataIndexTable =
    (struct data_index *) mymalloc_movable(&DataIndexTable, "DataIndexTable",
					   All.BunchSize * sizeof(struct data_index));
  DataNodeListSpecial =
    (struct data_nodelist_special *) mymalloc_movable(&DataNodeListSpecial, "DataNodeListSpecial",
						      All.BunchSize * sizeof(struct data_nodelist_special));

  i = 0;

  do
    {
      for(j = 0; j < NTask; j++)
	{
	  Send_count[j] = 0;
	  Exportflag[j] = -1;
	}

      /* do local particles and prepare export list */

      for(nexport = 0; i < Ndt; i++)
	if((DTF[i] & 2) == 0)	/* DT that is not flagged as tested ok */
	  {
	    DTF[i] |= 2;	/* if we find a particle, need to clear this flag again! */

	    if(DT[i].t[0] < 0)	/* deleted ? */
	      continue;

	    if(DT[i].p[0] == DPinfinity || DT[i].p[1] == DPinfinity || DT[i].p[2] == DPinfinity)
	      continue;

#ifndef TWODIMS
	    if(DT[i].p[3] == DPinfinity)
	      continue;
#endif

	    for(j = 0, q = -1; j < (NUMDIMS + 1); j++)
	      {
		if(DP[DT[i].p[j]].task == ThisTask)
		  if(DP[DT[i].p[j]].index >= 0 && DP[DT[i].p[j]].index < N_gas)
		    {
		      if(TimeBinActive[P[DP[DT[i].p[j]].index].TimeBin])
			{
			  q = DP[DT[i].p[j]].index;
			  break;
			}
		    }
	      }

	    if(q == -1)		/* this triangle does not have a local point. No need to test it */
	      continue;

	    if(voronoi_ghost_search_evaluate(i, 0, q, &nexport, Send_count) < 0)
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

      VoroDataGet = (struct vorodata_in *) mymalloc_movable(&VoroDataGet, "VoroDataGet",
							    nimport * sizeof(struct vorodata_in));
      VoroDataIn = (struct vorodata_in *) mymalloc_movable(&VoroDataIn, "VoroDataIn",
							   nexport * sizeof(struct vorodata_in));

      /* prepare particle data for export */
      for(j = 0; j < nexport; j++)
	{
	  place = DataIndexTable[j].Index;

	  for(k = 0, q = -1; k < (NUMDIMS + 1); k++)
	    {
	      if(DP[DT[place].p[k]].task == ThisTask)
		if(DP[DT[place].p[k]].index >= 0 && DP[DT[place].p[k]].index < N_gas)
		  {
		    if(TimeBinActive[P[DP[DT[place].p[k]].index].TimeBin])
		      {
			q = DT[place].p[k];
			break;
		      }
		  }
	    }

	  if(q == -1)
	    terminate("q=-1");

	  VoroDataIn[j].Pos[0] = DTC[place].cx;
	  VoroDataIn[j].Pos[1] = DTC[place].cy;
	  VoroDataIn[j].Pos[2] = DTC[place].cz;

	  VoroDataIn[j].RefPos[0] = DP[q].x;
	  VoroDataIn[j].RefPos[1] = DP[q].y;
	  VoroDataIn[j].RefPos[2] = DP[q].z;

	  VoroDataIn[j].Origin = ThisTask;

	  VoroDataIn[j].MaxDist = SphP[DP[q].index].Hsml;

	  memcpy(VoroDataIn[j].NodeList, DataNodeListSpecial[DataIndexTable[j].IndexGet].NodeList,
		 NODELISTLENGTH * sizeof(int));
#ifdef EXTENDED_GHOST_SEARCH
	  memcpy(VoroDataIn[j].BitFlagList, DataNodeListSpecial[DataIndexTable[j].IndexGet].BitFlagList,
		 NODELISTLENGTH * sizeof(unsigned char));
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
		  MPI_Sendrecv(&VoroDataIn[Send_offset[recvTask]], Send_count[recvTask]
			       * sizeof(struct vorodata_in), MPI_BYTE, recvTask, TAG_DENS_A,
			       &VoroDataGet[Recv_offset[recvTask]],
			       Recv_count[recvTask] * sizeof(struct vorodata_in), MPI_BYTE, recvTask,
			       TAG_DENS_A, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}
	    }
	}

      myfree(VoroDataIn);
      VoroDataResult = (struct vorodata_out *) mymalloc_movable(&VoroDataResult, "VoroDataResult",
								nimport * sizeof(struct vorodata_out));
      VoroDataOut = (struct vorodata_out *) mymalloc_movable(&VoroDataOut, "VoroDataOut",
							     nexport * sizeof(struct vorodata_out));

      MaxN_DP_Buffer = Indi.AllocFacN_DP_Buffer;

      N_DP_Buffer = 0;

      DP_Buffer = (point *) mymalloc_movable(&DP_Buffer, "DP_Buffer", MaxN_DP_Buffer * sizeof(point));

      for(j = 0; j < NTask; j++)
	send_count_new[j] = 0;

      /* now do the particles that were sent to us */

      for(j = 0; j < nimport; j++)
	voronoi_ghost_search_evaluate(j, 1, 0, &dummy, &dummy);

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
		  MPI_Sendrecv(&VoroDataResult[Recv_offset[recvTask]], Recv_count[recvTask]
			       * sizeof(struct vorodata_out), MPI_BYTE, recvTask, TAG_DENS_B,
			       &VoroDataOut[Send_offset[recvTask]],
			       Send_count[recvTask] * sizeof(struct vorodata_out), MPI_BYTE, recvTask,
			       TAG_DENS_B, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}
	    }

	}

      /* add the result to the local particles */
      for(j = 0; j < nexport; j++)
	{
	  place = DataIndexTable[j].Index;

	  if(VoroDataOut[j].Count > 0)
	    DTF[place] -= (DTF[place] & 2);	/* for this triangle we found a further particle */
	}

      memcpy(Send_count, send_count_new, NTask * sizeof(int));

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

      while(nimport + Ndp > MaxNdp)
	{
	  Indi.AllocFacNdp *= ALLOC_INCREASE_FACTOR;
	  MaxNdp = Indi.AllocFacNdp;
#ifdef VERBOSE
	  printf("Task=%d: increase memory allocation, MaxNdp=%d Indi.AllocFacNdp=%g\n",
		 ThisTask, MaxNdp, Indi.AllocFacNdp);
#endif
	  DP -= 5;
	  DP = myrealloc_movable(DP, (MaxNdp + 5) * sizeof(point));
	  DP += 5;

	  if(nimport + Ndp > MaxNdp && N_gas == 0)
	    terminate("nimport + Ndp > MaxNdp");
	}

      /* get the delaunay points */
      for(ngrp = 1; ngrp < (1 << PTask); ngrp++)
	{
	  sendTask = ThisTask;
	  recvTask = ThisTask ^ ngrp;

	  if(recvTask < NTask)
	    {
	      if(Send_count[recvTask] > 0 || Recv_count[recvTask] > 0)
		{
		  /* get the particles */
		  MPI_Sendrecv(&DP_Buffer[Send_offset[recvTask]], Send_count[recvTask] * sizeof(point),
			       MPI_BYTE, recvTask, TAG_DENS_B,
			       &DP[Ndp + Recv_offset[recvTask]], Recv_count[recvTask] * sizeof(point),
			       MPI_BYTE, recvTask, TAG_DENS_B, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}
	    }
	}

      Ndp += nimport;
      NadditionalPoints += nimport;

      if(N_DP_Buffer > Largest_N_DP_Buffer)
	Largest_N_DP_Buffer = N_DP_Buffer;

      myfree(DP_Buffer);
      myfree(VoroDataOut);
      myfree(VoroDataResult);
      myfree(VoroDataGet);

      if(i >= Ndt)
	ndone_flag = 1;
      else
	ndone_flag = 0;

      MPI_Allreduce(&ndone_flag, &ndone, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    }
  while(ndone < NTask);

  myfree(DataNodeListSpecial);
  myfree(DataIndexTable);
  myfree(send_count_new);

  return NadditionalPoints;
}

int voronoi_ghost_search_evaluate(int target, int mode, int q, int *nexport, int *nsend_local)
{
  int origin, bitflags;
  int startnode, numngb, numngb_inbox, listindex = 0;
  double h, dx, dy, dz, maxdist;
  MyDouble pos[3], refpos[3];

  if(mode == 0)
    {
      pos[0] = DTC[target].cx;
      pos[1] = DTC[target].cy;
      pos[2] = DTC[target].cz;
      refpos[0] = P[q].Pos[0];
      refpos[1] = P[q].Pos[1];
      refpos[2] = P[q].Pos[2];
      maxdist = SphP[q].Hsml;
      origin = ThisTask;
    }
  else
    {
      /* note: we do not use a pointer here to VoroDataGet[target].Pos, because VoroDataGet may be moved in a realloc operation */
      pos[0] = VoroDataGet[target].Pos[0];
      pos[1] = VoroDataGet[target].Pos[1];
      pos[2] = VoroDataGet[target].Pos[2];
      refpos[0] = VoroDataGet[target].RefPos[0];
      refpos[1] = VoroDataGet[target].RefPos[1];
      refpos[2] = VoroDataGet[target].RefPos[2];
      maxdist = VoroDataGet[target].MaxDist;
      origin = VoroDataGet[target].Origin;
    }

  dx = refpos[0] - pos[0];
  dy = refpos[1] - pos[1];
  dz = refpos[2] - pos[2];

  h = 1.00001 * sqrt(dx * dx + dy * dy + dz * dz);

  if(mode == 0)
    {
      if(maxdist < 2 * h)
	{
	  DTF[target] -= (DTF[target] & 2);	/* since we restrict the search radius, we are not guaranteed to search the full circumcircle of the triangle */
	}

      startnode = All.MaxPart;	/* root node */
      bitflags = 0;
    }
  else
    {
      startnode = VoroDataGet[target].NodeList[0];
#ifdef EXTENDED_GHOST_SEARCH
      bitflags = VoroDataGet[target].BitFlagList[0];
#else
      bitflags = 0;
#endif
      startnode = Nodes[startnode].u.d.nextnode;	/* open it */
    }

  numngb = 0;

  while(startnode >= 0)
    {
      while(startnode >= 0)
	{
	  numngb_inbox = ngb_treefind_ghost_search(pos, refpos, h, maxdist, target, origin, &startnode,
						   bitflags, mode, nexport, nsend_local);

	  if(numngb_inbox < 0)
	    return -1;

	  numngb += numngb_inbox;
	}

      if(mode == 1)
	{
	  listindex++;
	  if(listindex < NODELISTLENGTH)
	    {
	      startnode = VoroDataGet[target].NodeList[listindex];
#ifdef EXTENDED_GHOST_SEARCH
	      bitflags = VoroDataGet[target].BitFlagList[listindex];
#endif
	      if(startnode >= 0)
		startnode = Nodes[startnode].u.d.nextnode;	/* open it */
	    }
	}
    }

  if(mode == 0)
    {
      if(numngb)
	DTF[target] -= (DTF[target] & 2);
    }
  else
    {
      VoroDataResult[target].Count = numngb;
    }

  return 0;
}


#ifdef EXTENDED_GHOST_SEARCH	/* this allowes for mirrored images in a full 3x3 grid in terms of the principal domain */

int ngb_treefind_ghost_search(MyDouble searchcenter[3], MyDouble refpos[3],
			      MyFloat hsml, MyFloat maxdist, int target, int origin, int *startnode,
			      int bitflags, int mode, int *nexport, int *nsend_local)
{
  int numngb, no, p, task, nexport_save, ndp_save, nadditionalpoints_save;
  int image_flag;
  struct NODE *current;
  MyDouble x, y, z, dx, dy, dz, dist, dist2, distref, distref2;
  int listp;
  double dx_ref, dy_ref, dz_ref, mindistance, thisdistance;
  double min_x = 0, min_y = 0, min_z = 0;
  int min_p = 0, min_imageflag = 0;
  double offx, offy, offz;

  nadditionalpoints_save = NadditionalPoints;
  ndp_save = Ndp;
  nexport_save = *nexport;

  numngb = 0;
  mindistance = 1.0e30;

  int repx, repy, repz;
  int repx_A, repy_A, repz_A;
  int repx_B, repy_B, repz_B;
  int xbits;
  int ybits;
  int zbits;
  int count;

  if(mode == 0)
    {
      repx_A = -1;
      repx_B = 1;
      repy_A = -1;
      repy_B = 1;
      repz_A = -1;
      repz_B = 1;
    }
  else
    {
      zbits = (bitflags / 9);
      ybits = (bitflags - zbits * 9) / 3;
      xbits = bitflags - zbits * 9 - ybits * 3;

      if(xbits == 1)
	repx_A = repx_B = -1;
      else if(xbits == 2)
	repx_A = repx_B = 1;
      else
	repx_A = repx_B = 0;

      if(ybits == 1)
	repy_A = repy_B = -1;
      else if(ybits == 2)
	repy_A = repy_B = 1;
      else
	repy_A = repy_B = 0;

      if(zbits == 1)
	repz_A = repz_B = -1;
      else if(zbits == 2)
	repz_A = repz_B = 1;
      else
	repz_A = repz_B = 0;
    }

  for(repx = repx_A; repx <= repx_B; repx++)
    for(repy = repy_A; repy <= repy_B; repy++)
      for(repz = repz_A; repz <= repz_B; repz++)
	{
	  offx = offy = offz = 0;

	  image_flag = 0;	/* for each coordinate there are three possibilities.
				   We encodee them to basis three, i.e. x*3^0 + y*3^1 + z*3^2
				 */

	  if(repx == -1)
	    {
#if defined(PERIODIC) && !defined(REFLECTIVE_X)
	      offx = boxSize_X;
#endif
	      image_flag += 1;
	    }
	  else if(repx == 1)
	    {
#if defined(PERIODIC) && !defined(REFLECTIVE_X)
	      offx = -boxSize_X;
#else
	      offx = 2 * boxSize_X;
#endif
	      image_flag += 2;
	    }

	  if(repy == -1)
	    {
#if defined(PERIODIC) && !defined(REFLECTIVE_Y)
	      offy = boxSize_Y;
#endif
	      image_flag += 1 * 3;
	    }
	  else if(repy == 1)
	    {
#if defined(PERIODIC) && !defined(REFLECTIVE_Y)
	      offy = -boxSize_Y;
#else
	      offy = 2 * boxSize_Y;
#endif
	      image_flag += 2 * 3;
	    }

	  if(repz == -1)
	    {
#if defined(PERIODIC) && !defined(REFLECTIVE_Z)
	      offz = boxSize_Y;
#endif
	      image_flag += 1 * 9;
	    }
	  else if(repz == 1)
	    {
#if defined(PERIODIC) && !defined(REFLECTIVE_Z)
	      offz = -boxSize_Z;
#else
	      offz = 2 * boxSize_Z;
#endif
	      image_flag += 2 * 9;
	    }

	  if(mode == 1)
	    if(bitflags != image_flag)
	      {
		printf("bitflags=%d image_flag=%d xbits=%d ybits=%d zbits=%d  \n", bitflags, image_flag,
		       xbits, ybits, zbits);
		terminate("problem");
	      }

	  no = *startnode;
	  count = 0;

	  while(no >= 0)
	    {
	      count++;
	      if(no < All.MaxPart)	/* single particle */
		{
		  p = no;
		  no = Nextnode[no];

		  if(P[p].Type > 0)
		    continue;

		  if(P[p].Ti_current != All.Ti_Current)
		    drift_particle(p, All.Ti_Current);

		  dist = hsml;

		  x = P[p].Pos[0];
		  y = P[p].Pos[1];
		  z = P[p].Pos[2];

#if defined(REFLECTIVE_X)
		  if(repx != 0)
		    x = -x;
#endif
#if defined(REFLECTIVE_Y)
		  if(repy != 0)
		    y = -y;
#endif
#if defined(REFLECTIVE_Z)
		  if(repz != 0)
		    z = -z;
#endif
		  x += offx;
		  y += offy;
		  z += offz;

		  dx = x - searchcenter[0];
		  dy = y - searchcenter[1];
		  dz = z - searchcenter[2];

		  if(dx * dx + dy * dy + dz * dz > dist * dist)
		    continue;

		  dx_ref = x - refpos[0];
		  dy_ref = y - refpos[1];
		  dz_ref = z - refpos[2];

		  if((thisdistance = dx_ref * dx_ref + dy_ref * dy_ref + dz_ref * dz_ref) > maxdist * maxdist)
		    continue;

		  /* now we need to check whether this particle has already been sent to
		     the requesting cpu for this particular image shift */

		  if(thisdistance >= mindistance)
		    continue;

		  if(List_P[p].firstexport >= 0)
		    {
		      if(ListExports[List_P[p].currentexport].origin != origin)
			{
			  listp = List_P[p].firstexport;
			  while(listp >= 0)
			    {
			      if(ListExports[listp].origin == origin)
				{
				  List_P[p].currentexport = listp;
				  break;
				}

			      listp = ListExports[listp].nextexport;
			    }

			  if(listp >= 0)
			    if((ListExports[listp].image_bits & (1 << image_flag)))	/* already in list */
			      continue;
			}
		      else
			{
			  if((ListExports[List_P[p].currentexport].image_bits & (1 << image_flag)))	/* already in list */
			    continue;
			}
		    }

		  /* here we have found a new closest particle that has not been inserted yet */

		  numngb = 1;
		  mindistance = thisdistance;
		  min_p = p;
		  min_imageflag = image_flag;
		  min_x = x;
		  min_y = y;
		  min_z = z;
		}
	      else
		{
		  if(no >= All.MaxPart + MaxNodes)	/* pseudo particle */
		    {
		      if(mode == 1)
			terminate("mode == 1");

		      if(target >= 0)	/* if no target is given, export will not occur */
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
				  Ndp = ndp_save;
				  NadditionalPoints = nadditionalpoints_save;
				  *nexport = nexport_save;
				  if(nexport_save == 0)
				    terminate("nexport_save == 0");	/* in this case, the buffer is too small to process even a single particle */
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

			  DataNodeListSpecial[Exportindex[task]].BitFlagList[Exportnodecount[task]] =
			    image_flag;
			  DataNodeListSpecial[Exportindex[task]].NodeList[Exportnodecount[task]++] =
			    DomainNodeIndex[no - (All.MaxPart + MaxNodes)];

			  if(Exportnodecount[task] < NODELISTLENGTH)
			    DataNodeListSpecial[Exportindex[task]].NodeList[Exportnodecount[task]] = -1;
			}

		      no = Nextnode[no - MaxNodes];
		      continue;
		    }

		  current = &Nodes[no];

		  if(mode == 1)
		    {
		      if(current->u.d.bitflags & (1 << BITFLAG_TOPLEVEL))	/* we reached a top-level node again, which means that we are done with the branch */
			{
			  break;
			}
		    }

		  if(current->Ti_current != All.Ti_Current)
		    force_drift_node(no, All.Ti_Current);

		  no = current->u.d.sibling;	/* in case the node can be discarded */

#ifdef FORCE_EQUAL_TIMESTEPS
		  if(mode == 0 && repx == 0 && repy == 0 && repz == 0)	/* this can only yield local particles, which are all already present in case the mesh is created for all particles */
		    if(!(current->u.d.bitflags & (1 << BITFLAG_TOPLEVEL)))
		      continue;
#endif

		  /* test the node against the search sphere */
		  dist = hsml + 0.5 * current->len;

		  /* test the node against the maximum search distance from the reference point */
		  distref = maxdist + 0.5 * current->len;

		  x = current->center[0];
#if defined(REFLECTIVE_X)
		  if(repx != 0)
		    x = -x;
#endif
		  x += offx;
		  if((dx = fabs(x - searchcenter[0])) > dist)
		    continue;
		  if((dx_ref = fabs(x - refpos[0])) > distref)
		    continue;

		  y = current->center[1];
#if defined(REFLECTIVE_Y)
		  if(repy != 0)
		    y = -y;
#endif
		  y += offy;
		  if((dy = fabs(y - searchcenter[1])) > dist)
		    continue;
		  if((dy_ref = fabs(y - refpos[1])) > distref)
		    continue;

		  z = current->center[2];
#if defined(REFLECTIVE_Z)
		  if(repz != 0)
		    z = -z;
#endif
		  z += offz;
		  if((dz = fabs(z - searchcenter[2])) > dist)
		    continue;
		  if((dz_ref = fabs(z - refpos[2])) > distref)
		    continue;

		  /* now test against the minimal sphere enclosing everything */
		  dist2 = dist + FACT1 * current->len;
		  if(dx * dx + dy * dy + dz * dz > dist2 * dist2)
		    continue;

		  /* now test against the minimal sphere enclosing everything */
		  distref2 = distref + FACT1 * current->len;
		  if(dx_ref * dx_ref + dy_ref * dy_ref + dz_ref * dz_ref > distref2 * distref2)
		    continue;

		  no = current->u.d.nextnode;	/* ok, we need to open the node */
		}
	    }
	}

  *startnode = -1;

  if(numngb)
    {
      p = min_p;

      image_flag = min_imageflag;

      if(List_P[p].firstexport >= 0)
	{
	  if(ListExports[List_P[p].currentexport].origin != origin)
	    {
	      listp = List_P[p].firstexport;
	      while(listp >= 0)
		{
		  if(ListExports[listp].origin == origin)
		    {
		      List_P[p].currentexport = listp;
		      break;
		    }

		  if(ListExports[listp].nextexport < 0)
		    {
		      if(Ninlist >= MaxNinlist)
			{
			  Indi.AllocFacNinlist *= ALLOC_INCREASE_FACTOR;
			  MaxNinlist = Indi.AllocFacNinlist;
#ifdef VERBOSE
			  printf
			    ("Task=%d: increase memory allocation, MaxNinlist=%d Indi.AllocFacNinlist=%g\n",
			     ThisTask, MaxNinlist, Indi.AllocFacNinlist);
#endif
			  ListExports =
			    myrealloc_movable(ListExports, MaxNinlist * sizeof(struct list_export_data));

			  if(Ninlist >= MaxNinlist)
			    terminate("Ninlist >= MaxNinlist");
			}

		      List_P[p].currentexport = Ninlist++;
		      ListExports[List_P[p].currentexport].image_bits = 0;
		      ListExports[List_P[p].currentexport].nextexport = -1;
		      ListExports[List_P[p].currentexport].origin = origin;
		      ListExports[List_P[p].currentexport].index = p;
		      ListExports[listp].nextexport = List_P[p].currentexport;
		      break;
		    }
		  listp = ListExports[listp].nextexport;
		}
	    }
	}
      else
	{
	  /* here we have a local particle that hasn't been made part of the mesh */

	  if(Ninlist >= MaxNinlist)
	    {
	      Indi.AllocFacNinlist *= ALLOC_INCREASE_FACTOR;
	      MaxNinlist = Indi.AllocFacNinlist;
#ifdef VERBOSE
	      printf("Task=%d: increase memory allocation, MaxNinlist=%d Indi.AllocFacNinlist=%g\n",
		     ThisTask, MaxNinlist, Indi.AllocFacNinlist);
#endif
	      ListExports = myrealloc_movable(ListExports, MaxNinlist * sizeof(struct list_export_data));

	      if(Ninlist >= MaxNinlist)
		terminate("Ninlist >= MaxNinlist");
	    }

	  List_P[p].currentexport = List_P[p].firstexport = Ninlist++;
	  ListExports[List_P[p].currentexport].image_bits = 0;
	  ListExports[List_P[p].currentexport].nextexport = -1;
	  ListExports[List_P[p].currentexport].origin = origin;
	  ListExports[List_P[p].currentexport].index = p;
	}

      if((ListExports[List_P[p].currentexport].image_bits & (1 << image_flag)))
	terminate("this should not happen");

      ListExports[List_P[p].currentexport].image_bits |= (1 << image_flag);

      /* add the particle to the ones that need to be exported */

      if(origin == ThisTask)
	{
	  if(mode == 1)
	    terminate("mode==1: how can this be?");

	  if(Ndp >= MaxNdp)
	    {
	      Indi.AllocFacNdp *= ALLOC_INCREASE_FACTOR;
	      MaxNdp = Indi.AllocFacNdp;
#ifdef VERBOSE
	      printf("Task=%d: increase memory allocation, MaxNdp=%d Indi.AllocFacNdp=%g\n",
		     ThisTask, MaxNdp, Indi.AllocFacNdp);
#endif
	      DP -= 5;
	      DP = myrealloc_movable(DP, (MaxNdp + 5) * sizeof(point));
	      DP += 5;

	      if(Ndp >= MaxNdp)
		terminate("Ndp >= MaxNdp");
	    }

	  DP[Ndp].x = min_x;
	  DP[Ndp].y = min_y;
	  DP[Ndp].z = min_z;
	  DP[Ndp].task = ThisTask;
	  DP[Ndp].ID = P[p].ID;
	  if(image_flag)
	    DP[Ndp].index = p + N_gas;	/* this is a replicated/mirrored local point */
	  else
	    DP[Ndp].index = p;	/* this is actually a local point that wasn't made part of the mesh yet */
	  if(TimeBinActive[P[p].TimeBin])
	    DP[Ndp].inactiveflag = -1;
	  else
	    DP[Ndp].inactiveflag = p;

#if defined(REFLECTIVE_X) || defined(REFLECTIVE_Y) || defined(REFLECTIVE_Z)
	  DP[Ndp].image_flags = (1 << image_flag);
#endif
	  Ndp++;
	  NadditionalPoints++;
	}
      else
	{
	  if(mode == 0)
	    terminate("mode == 0: how can this be?");

	  if(N_DP_Buffer >= MaxN_DP_Buffer)
	    {
	      Indi.AllocFacN_DP_Buffer *= ALLOC_INCREASE_FACTOR;
	      MaxN_DP_Buffer = Indi.AllocFacN_DP_Buffer;
#ifdef VERBOSE
	      printf
		("Task=%d: increase memory allocation, MaxN_DP_Buffer=%d Indi.AllocFacN_DP_Buffer=%g\n",
		 ThisTask, MaxN_DP_Buffer, Indi.AllocFacN_DP_Buffer);
#endif
	      DP_Buffer = (point *) myrealloc_movable(DP_Buffer, MaxN_DP_Buffer * sizeof(point));

	      if(N_DP_Buffer >= MaxN_DP_Buffer)
		terminate("(N_DP_Buffer >= MaxN_DP_Buffer");
	    }

	  DP_Buffer[N_DP_Buffer].x = min_x;
	  DP_Buffer[N_DP_Buffer].y = min_y;
	  DP_Buffer[N_DP_Buffer].z = min_z;
	  DP_Buffer[N_DP_Buffer].ID = P[p].ID;
	  DP_Buffer[N_DP_Buffer].task = ThisTask;
	  DP_Buffer[N_DP_Buffer].index = p;
	  if(TimeBinActive[P[p].TimeBin])
	    DP_Buffer[N_DP_Buffer].inactiveflag = -1;
	  else
	    DP_Buffer[N_DP_Buffer].inactiveflag = p;
#if defined(REFLECTIVE_X) || defined(REFLECTIVE_Y) || defined(REFLECTIVE_Z)
	  DP_Buffer[N_DP_Buffer].image_flags = (1 << image_flag);
#endif
	  send_count_new[origin]++;
	  N_DP_Buffer++;
	}
    }

  return numngb;
}

#else

int ngb_treefind_ghost_search(MyDouble searchcenter[3], MyDouble refpos[3], MyFloat hsml, MyFloat maxdist,
			      int target, int origin, int *startnode, int bitflags, int mode, int *nexport,
			      int *nsend_local)
{
  int numngb, no, p, task, nexport_save, ndp_save, nadditionalpoints_save;
  int image_flag;
  struct NODE *current;
  MyDouble x, y, z, dx, dy, dz, dist, dist2, distref, distref2;
  int listp;
  double dx_ref, dy_ref, dz_ref, mindistance, thisdistance, maxdistSquared, hsmlSquared;
  double min_x = 0, min_y = 0, min_z = 0;
  int min_p = 0, min_imageflag = 0;
  double offx, offy, offz;

  nadditionalpoints_save = NadditionalPoints;
  ndp_save = Ndp;
  nexport_save = *nexport;

  numngb = 0;
  mindistance = 1.0e30;

  int repx, repy, repz;
  int count;

  no = *startnode;
  count = 0;

  maxdistSquared = maxdist * maxdist;
  hsmlSquared = hsml * hsml;



  while(no >= 0)
    {
      count++;
      if(no < All.MaxPart)	/* single particle */
	{
	  p = no;
	  no = Nextnode[no];

	  if(P[p].Type > 0)
	    continue;

	  if(P[p].Ti_current != All.Ti_Current)
	    drift_particle(p, All.Ti_Current);

	  offx = offy = offz = 0;
	  repx = repy = repz = 0;

	  image_flag = 0;	/* for each coordinates there are three possibilities. We
				   encode them to basis three, i.e. x*3^0 + y*3^1 + z*3^2 */

#if !defined(REFLECTIVE_X)
	  if(P[p].Pos[0] - refpos[0] < -boxHalf_X)
	    {
	      offx = boxSize_X;
	      image_flag += 1;
	    }
	  else if(P[p].Pos[0] - refpos[0] > boxHalf_X)
	    {
	      offx = -boxSize_X;
	      image_flag += 2;
	    }
#endif

#if !defined(REFLECTIVE_Y)
	  if(P[p].Pos[1] - refpos[1] < -boxHalf_Y)
	    {
	      offy = boxSize_Y;
	      image_flag += 1 * 3;
	    }
	  else if(P[p].Pos[1] - refpos[1] > boxHalf_Y)
	    {
	      offy = -boxSize_Y;
	      image_flag += 2 * 3;
	    }
#endif

#if !defined(REFLECTIVE_Z)
	  if(P[p].Pos[2] - refpos[2] < -boxHalf_Z)
	    {
	      offz = boxSize_Z;
	      image_flag += 1 * 9;
	    }
	  else if(P[p].Pos[2] - refpos[2] > boxHalf_Z)
	    {
	      offz = -boxSize_Z;
	      image_flag += 2 * 9;
	    }
#endif

	  int image_flag_periodic_bnds = image_flag;

#if defined(REFLECTIVE_X)
	  for(repx = -1; repx <= 1; repx++, offx = 0)
#endif
#if defined(REFLECTIVE_Y)
	    for(repy = -1; repy <= 1; repy++, offy = 0)
#endif
#if defined(REFLECTIVE_Z)
	      for(repz = -1; repz <= 1; repz++, offz = 0)
#endif
		{

		  image_flag = image_flag_periodic_bnds;

		  x = P[p].Pos[0];
		  y = P[p].Pos[1];
		  z = P[p].Pos[2];

#if defined(REFLECTIVE_X)
		  if(repx == 1)
		    {
		      offx = 2 * boxSize_X;
		      image_flag += 2;
		    }
		  else if(repx == -1)
		    {
		      image_flag += 1;
		    }
		  if(repx != 0)
		    x = -x;
#endif

#if  defined(REFLECTIVE_Y)
		  if(repy == 1)
		    {
		      offy = 2 * boxSize_Y;
		      image_flag += 2 * 3;
		    }
		  else if(repy == -1)
		    {
		      image_flag += 1 * 3;
		    }
		  if(repy != 0)
		    y = -y;
#endif
#if  defined(REFLECTIVE_Z)
		  if(repz == 1)
		    {
		      offz = 2 * boxSize_Z;
		      image_flag += 2 * 9;
		    }
		  else if(repz == -1)
		    {
		      image_flag += 1 * 9;
		    }
		  if(repz != 0)
		    z = -z;
#endif

		  x += offx;
		  y += offy;
		  z += offz;

		  dx_ref = x - refpos[0];
		  dy_ref = y - refpos[1];
		  dz_ref = z - refpos[2];

		  if((thisdistance = dx_ref * dx_ref + dy_ref * dy_ref + dz_ref * dz_ref) > maxdistSquared)
		    continue;

		  dx = x - searchcenter[0];
		  dy = y - searchcenter[1];
		  dz = z - searchcenter[2];

		  if(dx * dx + dy * dy + dz * dz > hsmlSquared)
		    continue;

		  if(x < -0.5 * boxSize_X || x > 1.5 * boxSize_X || y < -0.5 * boxSize_Y
		     || y > 1.5 * boxSize_Y)
		    continue;

		  /* now we need to check whether this particle has already been sent to
		     the requesting cpu for this particular image shift */

		  if(thisdistance >= mindistance)
		    continue;

		  if(List_P[p].firstexport >= 0)
		    {
		      if(ListExports[List_P[p].currentexport].origin != origin)
			{
			  listp = List_P[p].firstexport;
			  while(listp >= 0)
			    {
			      if(ListExports[listp].origin == origin)
				{
				  List_P[p].currentexport = listp;
				  break;
				}

			      listp = ListExports[listp].nextexport;
			    }

			  if(listp >= 0)
			    if((ListExports[listp].image_bits & (1 << image_flag)))	/* already in list */
			      continue;
			}
		      else
			{
			  if((ListExports[List_P[p].currentexport].image_bits & (1 << image_flag)))	/* already in list */
			    continue;
			}
		    }

		  /* here we have found a new closest particle that has not been inserted yet */

		  numngb = 1;
		  mindistance = thisdistance;
		  min_p = p;
		  min_imageflag = image_flag;
		  min_x = x;
		  min_y = y;
		  min_z = z;

		  maxdistSquared = thisdistance;
		  maxdist = sqrt(thisdistance);
		}
	}
      else
	{
	  if(no >= All.MaxPart + MaxNodes)	/* pseudo particle */
	    {
	      if(mode == 1)
		terminate("mode == 1");

	      if(target >= 0)	/* if no target is given, export will not occur */
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
			  Ndp = ndp_save;
			  NadditionalPoints = nadditionalpoints_save;
			  *nexport = nexport_save;
			  if(nexport_save == 0)
			    terminate("nexport_save == 0");	/* in this case, the buffer is too small to process even a single particle */
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

		  DataNodeListSpecial[Exportindex[task]].NodeList[Exportnodecount[task]++]
		    = DomainNodeIndex[no - (All.MaxPart + MaxNodes)];

		  if(Exportnodecount[task] < NODELISTLENGTH)
		    DataNodeListSpecial[Exportindex[task]].NodeList[Exportnodecount[task]] = -1;
		}

	      no = Nextnode[no - MaxNodes];
	      continue;
	    }

	  current = &Nodes[no];

	  if(mode == 1)
	    {
	      if(current->u.d.bitflags & (1 << BITFLAG_TOPLEVEL))	/* we reached a top-level node again, which means that we are done with the branch */
		{
		  break;
		}
	    }

	  if(current->Ti_current != All.Ti_Current)
	    force_drift_node(no, All.Ti_Current);

	  no = current->u.d.sibling;	/* in case the node can be discarded */

	  /* test the node against the search sphere */
	  dist = hsml + 0.5 * current->len;

	  /* test the node against the maximum search distance from the reference point */
	  distref = maxdist + 0.5 * current->len;

	  int flag = 0;

	  x = current->center[0];
#if !defined(REFLECTIVE_X)
	  if(x - refpos[0] < -boxHalf_X)
	    {
	      x += boxSize_X;
	      flag = 1;
	    }
	  else if(x - refpos[0] > boxHalf_X)
	    {
	      x -= boxSize_X;
	      flag = 1;
	    }
#else
	  if(fabs(-x - refpos[0]) < fabs(x - refpos[0]))
	    {
	      x = -x;
	      flag = 1;
	    }
	  else if(fabs(2 * boxSize_X - x - refpos[0]) < fabs(x - refpos[0]))
	    {
	      x = 2 * boxSize_X - x;
	      flag = 1;
	    }
	  else
	    {
	      if(fabs(-x - searchcenter[0]) <= dist && fabs(-x - refpos[0]) <= distref)
		flag = 1;
	      if(fabs(2 * boxSize_X - x - searchcenter[0]) <= dist &&
		 fabs(2 * boxSize_X - x - refpos[0]) <= distref)
		flag = 1;
	    }
#endif
	  if((dx = fabs(x - searchcenter[0])) > dist)
	    continue;
	  if((dx_ref = fabs(x - refpos[0])) > distref)
	    continue;

	  y = current->center[1];
#if !defined(REFLECTIVE_Y)
	  if(y - refpos[1] < -boxHalf_Y)
	    {
	      y += boxSize_Y;
	      flag = 1;
	    }
	  else if(y - refpos[1] > boxHalf_Y)
	    {
	      y -= boxSize_Y;
	      flag = 1;
	    }
#else
	  if(fabs(-y - refpos[1]) < fabs(y - refpos[1]))
	    {
	      y = -y;
	      flag = 1;
	    }
	  else if(fabs(2 * boxSize_Y - y - refpos[1]) < fabs(y - refpos[1]))
	    {
	      y = 2 * boxSize_Y - y;
	      flag = 1;
	    }
	  else
	    {
	      if(fabs(-y - searchcenter[1]) <= dist && fabs(-y - refpos[1]) <= distref)
		flag = 1;
	      if(fabs(2 * boxSize_Y - y - searchcenter[1]) <= dist &&
		 fabs(2 * boxSize_Y - y - refpos[1]) <= distref)
		flag = 1;
	    }
#endif
	  if((dy = fabs(y - searchcenter[1])) > dist)
	    continue;
	  if((dy_ref = fabs(y - refpos[1])) > distref)
	    continue;

	  z = current->center[2];
#if !defined(REFLECTIVE_Z)
	  if(z - refpos[2] < -boxHalf_Z)
	    {
	      z += boxSize_X;
	      flag = 1;
	    }
	  else if(z - refpos[2] > boxHalf_Z)
	    {
	      z -= boxSize_Z;
	      flag = 1;
	    }
#else
	  if(fabs(-z - refpos[2]) < fabs(z - refpos[2]))
	    {
	      z = -z;
	      flag = 1;
	    }
	  else if(fabs(2 * boxSize_Z - z - refpos[2]) < fabs(z - refpos[2]))
	    {
	      z = 2 * boxSize_Z - z;
	      flag = 1;
	    }
	  else
	    {
	      if(fabs(-z - searchcenter[2]) <= dist && fabs(-z - refpos[2]) <= distref)
		flag = 1;
	      if(fabs(2 * boxSize_Z - z - searchcenter[2]) <= dist &&
		 fabs(2 * boxSize_Z - z - refpos[2]) <= distref)
		flag = 1;
	    }
#endif
	  if((dz = fabs(z - searchcenter[2])) > dist)
	    continue;
	  if((dz_ref = fabs(z - refpos[2])) > distref)
	    continue;

#ifdef FORCE_EQUAL_TIMESTEPS
	  if(mode == 0 && flag == 0)	/* this can only yield local particles, which are all already present in case the mesh is created for all particles  */
	    if(!(current->u.d.bitflags & (1 << BITFLAG_TOPLEVEL)))
	      continue;
#endif

	  /* now test against the minimal sphere enclosing everything */
	  distref2 = distref + FACT1 * current->len;
	  if(dx_ref * dx_ref + dy_ref * dy_ref + dz_ref * dz_ref > distref2 * distref2)
	    continue;

	  /* now test against the minimal sphere enclosing everything */
	  dist2 = dist + FACT1 * current->len;
	  if(dx * dx + dy * dy + dz * dz > dist2 * dist2)
	    continue;

	  no = current->u.d.nextnode;	/* ok, we need to open the node */
	}
    }

  *startnode = -1;

  if(numngb)
    {
      p = min_p;

      image_flag = min_imageflag;

      if(List_P[p].firstexport >= 0)
	{
	  if(ListExports[List_P[p].currentexport].origin != origin)
	    {
	      listp = List_P[p].firstexport;
	      while(listp >= 0)
		{
		  if(ListExports[listp].origin == origin)
		    {
		      List_P[p].currentexport = listp;
		      break;
		    }

		  if(ListExports[listp].nextexport < 0)
		    {
		      if(Ninlist >= MaxNinlist)
			{
			  Indi.AllocFacNinlist *= ALLOC_INCREASE_FACTOR;
			  MaxNinlist = Indi.AllocFacNinlist;
#ifdef VERBOSE
			  printf
			    ("Task=%d: increase memory allocation, MaxNinlist=%d Indi.AllocFacNinlist=%g\n",
			     ThisTask, MaxNinlist, Indi.AllocFacNinlist);
#endif
			  ListExports =
			    myrealloc_movable(ListExports, MaxNinlist * sizeof(struct list_export_data));

			  if(Ninlist >= MaxNinlist)
			    terminate("Ninlist >= MaxNinlist");
			}

		      List_P[p].currentexport = Ninlist++;
		      ListExports[List_P[p].currentexport].image_bits = 0;
		      ListExports[List_P[p].currentexport].nextexport = -1;
		      ListExports[List_P[p].currentexport].origin = origin;
		      ListExports[List_P[p].currentexport].index = p;
		      ListExports[listp].nextexport = List_P[p].currentexport;
		      break;
		    }
		  listp = ListExports[listp].nextexport;
		}
	    }
	}
      else
	{
	  /* here we have a local particle that hasn't been made part of the mesh */

	  if(Ninlist >= MaxNinlist)
	    {
	      Indi.AllocFacNinlist *= ALLOC_INCREASE_FACTOR;
	      MaxNinlist = Indi.AllocFacNinlist;
#ifdef VERBOSE
	      printf("Task=%d: increase memory allocation, MaxNinlist=%d Indi.AllocFacNinlist=%g\n",
		     ThisTask, MaxNinlist, Indi.AllocFacNinlist);
#endif
	      ListExports = myrealloc_movable(ListExports, MaxNinlist * sizeof(struct list_export_data));

	      if(Ninlist >= MaxNinlist)
		terminate("Ninlist >= MaxNinlist");
	    }

	  List_P[p].currentexport = List_P[p].firstexport = Ninlist++;
	  ListExports[List_P[p].currentexport].image_bits = 0;
	  ListExports[List_P[p].currentexport].nextexport = -1;
	  ListExports[List_P[p].currentexport].origin = origin;
	  ListExports[List_P[p].currentexport].index = p;
	}

      if((ListExports[List_P[p].currentexport].image_bits & (1 << image_flag)))
	terminate("this should not happen");

      ListExports[List_P[p].currentexport].image_bits |= (1 << image_flag);

      /* add the particle to the ones that need to be exported */

      if(origin == ThisTask)
	{
	  if(mode == 1)
	    terminate("mode==1: how can this be?");

	  if(Ndp >= MaxNdp)
	    {
	      Indi.AllocFacNdp *= ALLOC_INCREASE_FACTOR;
	      MaxNdp = Indi.AllocFacNdp;
#ifdef VERBOSE
	      printf("Task=%d: increase memory allocation, MaxNdp=%d Indi.AllocFacNdp=%g\n",
		     ThisTask, MaxNdp, Indi.AllocFacNdp);
#endif
	      DP -= 5;
	      DP = myrealloc_movable(DP, (MaxNdp + 5) * sizeof(point));
	      DP += 5;

	      if(Ndp >= MaxNdp)
		terminate("Ndp >= MaxNdp");
	    }

	  DP[Ndp].x = min_x;
	  DP[Ndp].y = min_y;
	  DP[Ndp].z = min_z;
	  DP[Ndp].task = ThisTask;
	  DP[Ndp].ID = P[p].ID;
	  if(image_flag)
	    DP[Ndp].index = p + N_gas;	/* this is a replicated/mirrored local point */
	  else
	    DP[Ndp].index = p;	/* this is actually a local point that wasn't made part of the mesh yet */
	  if(TimeBinActive[P[p].TimeBin])
	    DP[Ndp].inactiveflag = -1;
	  else
	    DP[Ndp].inactiveflag = p;
#if defined(REFLECTIVE_X) || defined(REFLECTIVE_Y) || defined(REFLECTIVE_Z)
	  DP[Ndp].image_flags = (1 << image_flag);
#endif
	  Ndp++;
	  NadditionalPoints++;
	}
      else
	{
	  if(mode == 0)
	    terminate("mode == 0: how can this be?");

	  if(N_DP_Buffer >= MaxN_DP_Buffer)
	    {
	      Indi.AllocFacN_DP_Buffer *= ALLOC_INCREASE_FACTOR;
	      MaxN_DP_Buffer = Indi.AllocFacN_DP_Buffer;
#ifdef VERBOSE
	      printf
		("Task=%d: increase memory allocation, MaxN_DP_Buffer=%d Indi.AllocFacN_DP_Buffer=%g\n",
		 ThisTask, MaxN_DP_Buffer, Indi.AllocFacN_DP_Buffer);
#endif
	      DP_Buffer = (point *) myrealloc_movable(DP_Buffer, MaxN_DP_Buffer * sizeof(point));

	      if(N_DP_Buffer >= MaxN_DP_Buffer)
		terminate("(N_DP_Buffer >= MaxN_DP_Buffer");
	    }

	  DP_Buffer[N_DP_Buffer].x = min_x;
	  DP_Buffer[N_DP_Buffer].y = min_y;
	  DP_Buffer[N_DP_Buffer].z = min_z;
	  DP_Buffer[N_DP_Buffer].ID = P[p].ID;
	  DP_Buffer[N_DP_Buffer].task = ThisTask;
	  DP_Buffer[N_DP_Buffer].index = p;
	  if(TimeBinActive[P[p].TimeBin])
	    DP_Buffer[N_DP_Buffer].inactiveflag = -1;
	  else
	    DP_Buffer[N_DP_Buffer].inactiveflag = p;
#if defined(REFLECTIVE_X) || defined(REFLECTIVE_Y) || defined(REFLECTIVE_Z)
	  DP_Buffer[N_DP_Buffer].image_flags = (1 << image_flag);
#endif
	  send_count_new[origin]++;
	  N_DP_Buffer++;
	}
    }

  return numngb;
}

#endif

int count_undecided_tetras(void)
{
  int i, count;

  for(i = 0, count = 0; i < Ndt; i++)
    if((DTF[i] & 2) == 0)
      count++;

  return count;
}

#endif


#endif
