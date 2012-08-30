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

#ifdef ALTERNATIVE_GHOST_SEARCH

static struct vorodata_in
{
  MyDouble Pos[3];
  MyFloat Hsml;
  int Origin;
  int ID;
  int NodeList[NODELISTLENGTH];
} *VoroDataGet, *VoroDataIn;

static point *DP_Buffer;

static int MaxN_DP_Buffer, N_DP_Buffer;

static int NadditionalPoints;

#ifndef ONEDIMS
int voronoi_ghost_search_alternative(void)
{
  int i, j, ndone, ndone_flag, dummy;

  int ngrp, sendTask, recvTask, place, nexport, nimport;

  NadditionalPoints = 0;

  /* allocate buffers to arrange communication */

  All.BunchSize = (int) ((All.BufferSize * 1024 * 1024) / (sizeof(struct data_index)
							   + sizeof(struct data_nodelist) +
							   2 * sizeof(struct vorodata_in)));
  DataIndexTable =
    (struct data_index *) mymalloc_movable(&DataIndexTable, "DataIndexTable",
					   All.BunchSize * sizeof(struct data_index));
  DataNodeList =
    (struct data_nodelist *) mymalloc_movable(&DataNodeList, "DataNodeList",
					      All.BunchSize * sizeof(struct data_nodelist));

  i = FirstActiveParticle;

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
	    if(SphP[i].Hsml / HSML_INCREASE_FACTOR < SphP[i].MaxDelaunayRadius)
	      {
		if(voronoi_exchange_evaluate(i, 0, &nexport, Send_count) < 0)
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

      VoroDataGet =
	(struct vorodata_in *) mymalloc_movable(&VoroDataGet, "VoroDataGet",
						nimport * sizeof(struct vorodata_in));
      VoroDataIn =
	(struct vorodata_in *) mymalloc_movable(&VoroDataIn, "VoroDataIn",
						nexport * sizeof(struct vorodata_in));

      /* prepare particle data for export */
      for(j = 0; j < nexport; j++)
	{
	  place = DataIndexTable[j].Index;

	  VoroDataIn[j].Pos[0] = P[place].Pos[0];
	  VoroDataIn[j].Pos[1] = P[place].Pos[1];
	  VoroDataIn[j].Pos[2] = P[place].Pos[2];
	  VoroDataIn[j].Origin = ThisTask;
	  VoroDataIn[j].ID = P[place].ID;
	  VoroDataIn[j].Hsml = SphP[place].Hsml;

	  memcpy(VoroDataIn[j].NodeList, DataNodeList[DataIndexTable[j].IndexGet].NodeList, NODELISTLENGTH
		 * sizeof(int));
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

      MaxN_DP_Buffer = Indi.AllocFacN_DP_Buffer;

      N_DP_Buffer = 0;

      DP_Buffer = (point *) mymalloc_movable(&DP_Buffer, "DP_Buffer", MaxN_DP_Buffer * sizeof(point));

      for(j = 0; j < NTask; j++)
	Send_count[j] = 0;

      /* now do the particles that were sent to us */

      for(j = 0; j < nimport; j++)
	voronoi_exchange_evaluate(j, 1, &dummy, Send_count);

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
      myfree(VoroDataGet);

      if(i < 0)
	ndone_flag = 1;
      else
	ndone_flag = 0;

      MPI_Allreduce(&ndone_flag, &ndone, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    }
  while(ndone < NTask);

  myfree(DataNodeList);
  myfree(DataIndexTable);

  return NadditionalPoints;
}
#endif

int voronoi_exchange_evaluate(int target, int mode, int *nexport, int *nsend_local)
{
  int origin;

  int startnode, numngb, numngb_inbox, listindex = 0, id;

  double h;

  MyDouble pos[3];

  if(mode == 0)
    {
      pos[0] = P[target].Pos[0];
      pos[1] = P[target].Pos[1];
      pos[2] = P[target].Pos[2];
      h = SphP[target].Hsml;
      h *= HSML_INCREASE_FACTOR;
      origin = ThisTask;
      id = P[target].ID;
    }
  else
    {
      /* note: we do not use a pointer here to VoroDataGet[target].Pos, because VoroDataGet may be moved in a realloc operation */
      pos[0] = VoroDataGet[target].Pos[0];
      pos[1] = VoroDataGet[target].Pos[1];
      pos[2] = VoroDataGet[target].Pos[2];
      h = VoroDataGet[target].Hsml;
      origin = VoroDataGet[target].Origin;
      id = VoroDataGet[target].ID;
    }

  if(h > boxHalf_X || h > boxHalf_Y || h > boxHalf_Z)
    {
      h = boxHalf_X;
      /*
         printf("too big Hsml: target=%d h=%g\n", target, h);

         endrun(6998642); */
    }

  if(mode == 0)
    {
      startnode = All.MaxPart;	/* root node */
    }
  else
    {
      startnode = VoroDataGet[target].NodeList[0];
      startnode = Nodes[startnode].u.d.nextnode;	/* open it */
    }

  numngb = 0;

  while(startnode >= 0)
    {
      while(startnode >= 0)
	{
	  numngb_inbox = ngb_treefind_voronoi(pos, 1.01 * h, target, origin, &startnode, mode, nexport,
					      nsend_local, id);

	  if(numngb_inbox < 0)
	    return -1;
	}

      if(mode == 1)
	{
	  listindex++;
	  if(listindex < NODELISTLENGTH)
	    {
	      startnode = VoroDataGet[target].NodeList[listindex];
	      if(startnode >= 0)
		startnode = Nodes[startnode].u.d.nextnode;	/* open it */
	    }
	}
    }

  return 0;
}

int ngb_treefind_voronoi(MyDouble searchcenter[3], MyFloat hsml, int target, int origin, int *startnode,
			 int mode, int *nexport, int *nsend_local, int id)
{
  int numngb, no, p, task, nexport_save, ndp_save, nadditionalpoints_save;

  int image_flag;

  struct NODE *current;

  MyDouble dx, dy, dz, dist, dist2;

  int listp;

  nadditionalpoints_save = NadditionalPoints;
  ndp_save = Ndp;
  nexport_save = *nexport;

  numngb = 0;
  no = *startnode;

  while(no >= 0)
    {
      if(no < All.MaxPart)	/* single particle */
	{
	  p = no;
	  no = Nextnode[no];

	  if(P[p].Type > 0)
	    continue;

	  if(P[p].Ti_current != All.Ti_Current)
	    drift_particle(p, All.Ti_Current);

	  dist = hsml;

	  image_flag = 0;	/* for each coordinates there are three possibilities. We
				   encode them to basis three, i.e. x*3^0 + y*3^1 + z*3^2 */

	  dx = P[p].Pos[0] - searchcenter[0];
#if defined(PERIODIC) && !defined(REFLECTIVE_X)
	  if(dx < -boxHalf_X)
	    {
	      dx += boxSize_X;
	      image_flag += 1;
	    }
	  else if(dx > boxHalf_X)
	    {
	      dx -= boxSize_X;
	      image_flag += 2;
	    }
#endif
	  if(fabs(dx) > dist)
	    continue;

	  dy = P[p].Pos[1] - searchcenter[1];
#if defined(PERIODIC) && !defined(REFLECTIVE_Y)
	  if(dy < -boxHalf_Y)
	    {
	      dy += boxSize_Y;
	      image_flag += 1 * 3;
	    }
	  else if(dy > boxHalf_Y)
	    {
	      dy -= boxSize_Y;
	      image_flag += 2 * 3;
	    }
#endif
	  if(fabs(dy) > dist)
	    continue;

	  dz = P[p].Pos[2] - searchcenter[2];
#if defined(PERIODIC) && !defined(REFLECTIVE_Z)
	  if(dz < -boxHalf_Z)
	    {
	      dz += boxSize_Z;
	      image_flag += 1 * 9;
	    }
	  else if(dz > boxHalf_Z)
	    {
	      dz -= boxSize_Z;
	      image_flag += 2 * 9;
	    }
#endif
	  if(fabs(dz) > dist)
	    continue;

	  if(dx * dx + dy * dy + dz * dz > dist * dist)
	    continue;

	  /* now we need to check whether this particle has already been sent to
	     the requesting cpu for this particular image shift */

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

	  if(!(ListExports[List_P[p].currentexport].image_bits & (1 << image_flag)))
	    {
	      ListExports[List_P[p].currentexport].image_bits |= (1 << image_flag);

	      /* add the particle to the ones that need to be exported */

	      if(origin == ThisTask)
		{
		  if(Ndp >= MaxNdp)
		    {
		      Indi.AllocFacNdp *= ALLOC_INCREASE_FACTOR;
		      MaxNdp = Indi.AllocFacNdp;

		      printf("Task=%d: increase memory allocation, MaxNdp=%d Indi.AllocFacNdp=%g\n",
			     ThisTask, MaxNdp, Indi.AllocFacNdp);

		      DP -= 5;
		      DP = myrealloc_movable(DP, (MaxNdp + 5) * sizeof(point));
		      DP += 5;

		      if(Ndp >= MaxNdp)
			terminate("Ndp >= MaxNdp");
		    }

		  DP[Ndp].x = dx + searchcenter[0];
		  DP[Ndp].y = dy + searchcenter[1];
		  DP[Ndp].z = dz + searchcenter[2];
		  DP[Ndp].task = ThisTask;
		  DP[Ndp].ID = P[p].ID;
		  if(image_flag)
		    DP[Ndp].index = p + N_gas;	/* this is a replicated/mirrored local point */
		  else
		    DP[Ndp].index = p;	/* this is actually a local point that wasn't part of the mesh yet */
		  if(TimeBinActive[P[p].TimeBin])
		    DP[Ndp].inactiveflag = -1;
		  else
		    DP[Ndp].inactiveflag = 0;

#if defined(REFLECTIVE_X) || defined(REFLECTIVE_Y) || defined(REFLECTIVE_Z)
		  DP[Ndp].image_flags = (1 << image_flag);
#endif
		  Ndp++;
		  NadditionalPoints++;
		}
	      else
		{
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

		  DP_Buffer[N_DP_Buffer].x = dx + searchcenter[0];
		  DP_Buffer[N_DP_Buffer].y = dy + searchcenter[1];
		  DP_Buffer[N_DP_Buffer].z = dz + searchcenter[2];
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
		  nsend_local[origin]++;
		  N_DP_Buffer++;
		}
	    }

#if defined(REFLECTIVE_X) || defined(REFLECTIVE_Y) || defined(REFLECTIVE_Z)
	  int repx = 0, repy = 0, repz = 0, image_flag_refl;

	  double dxx, dyy, dzz;

#if defined(REFLECTIVE_X)
	  for(repx = -1; repx <= 1; repx++)
#endif
#if defined(REFLECTIVE_Y)
	    for(repy = -1; repy <= 1; repy++)
#endif
#if defined(REFLECTIVE_Z)
	      for(repz = -1; repz <= 1; repz++)
#endif
		{
		  image_flag_refl = image_flag;

		  if(repx == -1)
		    {
		      dxx = -(P[p].Pos[0] + searchcenter[0]);
		      image_flag_refl += 1;
		    }
		  else if(repx == 1)
		    {
		      dxx = 2 * boxSize_X - (P[p].Pos[0] + searchcenter[0]);
		      image_flag_refl += 2;
		    }
		  else
		    dxx = dx;

		  if(repy == -1)
		    {
		      dyy = -(P[p].Pos[1] + searchcenter[1]);
		      image_flag_refl += (1) * 3;
		    }
		  else if(repy == 1)
		    {
		      dyy = 2 * boxSize_Y - (P[p].Pos[1] + searchcenter[1]);
		      image_flag_refl += (2) * 3;
		    }
		  else
		    dyy = dy;

		  if(repz == -1)
		    {
		      dzz = -(P[p].Pos[2] + searchcenter[2]);
		      image_flag_refl += (1) * 9;
		    }
		  else if(repz == 1)
		    {
		      dzz = 2 * boxSize_Z - (P[p].Pos[2] + searchcenter[2]);
		      image_flag_refl += (2) * 9;
		    }
		  else
		    dzz = dz;

		  if(repx != 0 || repy != 0 || repz != 0)
		    if(dxx * dxx + dyy * dyy + dzz * dzz <= dist * dist)
		      {
			if(!(ListExports[List_P[p].currentexport].image_bits & (1 << image_flag_refl)))
			  {
			    ListExports[List_P[p].currentexport].image_bits |= (1 << image_flag_refl);

			    /* add the particle to the ones that need to be exported */

			    if(origin == ThisTask)
			      {
				if(Ndp >= MaxNdp)
				  {
				    Indi.AllocFacNdp *= ALLOC_INCREASE_FACTOR;
				    MaxNdp = Indi.AllocFacNdp;
#ifdef VERBOSE
				    printf
				      ("Task=%d: increase memory allocation, MaxNdp=%d Indi.AllocFacNdp=%g\n",
				       ThisTask, MaxNdp, Indi.AllocFacNdp);
#endif
				    DP -= 5;
				    DP = myrealloc_movable(DP, (MaxNdp + 5) * sizeof(point));
				    DP += 5;

				    if(Ndp >= MaxNdp)
				      terminate("Ndp >= MaxNdp");
				  }

				DP[Ndp].x = dxx + searchcenter[0];
				DP[Ndp].y = dyy + searchcenter[1];
				DP[Ndp].z = dzz + searchcenter[2];
				DP[Ndp].ID = P[p].ID;
				DP[Ndp].task = ThisTask;
				DP[Ndp].index = p + N_gas;	/* this is a mirrored local point */
				if(TimeBinActive[P[p].TimeBin])
				  DP[Ndp].inactiveflag = -1;
				else
				  DP[Ndp].inactiveflag = 0;
#if defined(REFLECTIVE_X) || defined(REFLECTIVE_Y) || defined(REFLECTIVE_Z)
				DP[Ndp].image_flags = (1 << image_flag_refl);
#endif
				Ndp++;
				NadditionalPoints++;
			      }
			    else
			      {
				if(N_DP_Buffer >= MaxN_DP_Buffer)
				  {
				    Indi.AllocFacN_DP_Buffer *= ALLOC_INCREASE_FACTOR;
				    MaxN_DP_Buffer = Indi.AllocFacN_DP_Buffer;
#ifdef VERBOSE
				    printf
				      ("Task=%d: increase memory allocation, MaxN_DP_Buffer=%d Indi.AllocFacN_DP_Buffer=%g\n",
				       ThisTask, MaxN_DP_Buffer, Indi.AllocFacN_DP_Buffer);
#endif
				    DP_Buffer =
				      (point *) myrealloc_movable(DP_Buffer, MaxN_DP_Buffer * sizeof(point));

				    if(N_DP_Buffer >= MaxN_DP_Buffer)
				      terminate("(N_DP_Buffer >= MaxN_DP_Buffer");
				  }

				DP_Buffer[N_DP_Buffer].x = dxx + searchcenter[0];
				DP_Buffer[N_DP_Buffer].y = dyy + searchcenter[1];
				DP_Buffer[N_DP_Buffer].z = dzz + searchcenter[2];
				DP_Buffer[N_DP_Buffer].task = ThisTask;
				DP_Buffer[N_DP_Buffer].index = p;
				DP_Buffer[N_DP_Buffer].ID = P[p].ID;
				if(TimeBinActive[P[p].TimeBin])
				  DP_Buffer[N_DP_Buffer].inactiveflag = -1;
				else
				  DP_Buffer[N_DP_Buffer].inactiveflag = p;
#if defined(REFLECTIVE_X) || defined(REFLECTIVE_Y) || defined(REFLECTIVE_Z)
				DP_Buffer[N_DP_Buffer].image_flags = (1 << image_flag_refl);
#endif
				nsend_local[origin]++;

				N_DP_Buffer++;
			      }
			  }
		      }
		}
#endif
	}
      else
	{
	  if(no >= All.MaxPart + MaxNodes)	/* pseudo particle */
	    {
	      if(mode == 1)
		endrun(12312);

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
			    endrun(13004);	/* in this case, the buffer is too small to process even a single particle */
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

		  DataNodeList[Exportindex[task]].NodeList[Exportnodecount[task]++] = DomainNodeIndex[no
												      -
												      (All.MaxPart
												       +
												       MaxNodes)];

		  if(Exportnodecount[task] < NODELISTLENGTH)
		    DataNodeList[Exportindex[task]].NodeList[Exportnodecount[task]] = -1;
		}

	      no = Nextnode[no - MaxNodes];
	      continue;
	    }

	  current = &Nodes[no];

	  if(mode == 1)
	    {
	      if(current->u.d.bitflags & (1 << BITFLAG_TOPLEVEL))	/* we reached a top-level node again, which means that we are done with the branch */
		{
		  *startnode = -1;
		  return numngb;
		}
	    }

	  if(current->Ti_current != All.Ti_Current)
	    force_drift_node(no, All.Ti_Current);

	  if(!(current->u.d.bitflags & (1 << BITFLAG_MULTIPLEPARTICLES)))
	    {
	      if(current->u.d.mass)	/* open cell */
		{
		  no = current->u.d.nextnode;
		  continue;
		}
	    }

	  no = current->u.d.sibling;	/* in case the node can be discarded */

	  dist = hsml + 0.5 * current->len;

	  dx = current->center[0] - searchcenter[0];
#if defined(PERIODIC) && !defined(REFLECTIVE_X)
	  if(dx < -boxHalf_X)
	    {
	      dx += boxSize_X;
	    }
	  else if(dx > boxHalf_X)
	    {
	      dx -= boxSize_X;
	    }
#endif
	  if((dx = fabs(dx)) > dist)
	    continue;

	  dy = current->center[1] - searchcenter[1];
#if defined(PERIODIC) && !defined(REFLECTIVE_Y)
	  if(dy < -boxHalf_Y)
	    {
	      dy += boxSize_Y;
	    }
	  else if(dy > boxHalf_Y)
	    {
	      dy -= boxSize_Y;
	    }
#endif
	  if((dy = fabs(dy)) > dist)
	    continue;

	  dz = current->center[2] - searchcenter[2];
#if defined(PERIODIC) && !defined(REFLECTIVE_Z)
	  if(dz < -boxHalf_Z)
	    {
	      dz += boxSize_Z;
	    }
	  else if(dz > boxHalf_Z)
	    {
	      dz -= boxSize_Z;
	    }
#endif
	  if((dz = fabs(dz)) > dist)
	    continue;

	  /* now test against the minimal sphere enclosing everything */
	  dist2 = dist + FACT1 * current->len;
	  if(dx * dx + dy * dy + dz * dz > dist2 * dist2)
	    continue;

	  no = current->u.d.nextnode;	/* ok, we need to open the node */
	}
    }

  *startnode = -1;
  return numngb;
}

#endif

#endif

