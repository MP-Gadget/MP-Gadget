#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <gsl/gsl_math.h>

#include "allvars.h"
#include "proto.h"

#ifdef SUBFIND
#include "subfind.h"

/*! Structure for communication during the density computation. Holds data that is sent to other processors.
 */
static struct nearestdata_in
{
  MyDouble Pos[3];
  MyIDType ID;
  MyFloat Hsml;
  MyFloat Density;
  MyFloat Dist[2];
  int Count;
  int64_t Index[2];
  int NodeList[NODELISTLENGTH];
}
 *NearestDataIn, *NearestDataGet;


static struct nearestdata_out
{
  MyFloat Dist[2];
  int64_t Index[2];
  int Count;
}
 *NearestDataResult, *NearestDataOut;


void subfind_find_nearesttwo(void)
{
  int i, j, k, l, ndone, ndone_flag, dummy;
  int ngrp, sendTask, recvTask, place, nexport, nimport, maxexport;

  if(ThisTask == 0)
    printf("Start finding nearest two (%d particles on task=%d)\n", NumPartGroup, ThisTask);

  /* allocate buffers to arrange communication */

  Ngblist = (int *) mymalloc("Ngblist", NumPartGroup * sizeof(int));
  Dist2list = (double *) mymalloc("Dist2list", NumPartGroup * sizeof(double));

  All.BunchSize =
    (int) ((All.BufferSize * 1024 * 1024) / (sizeof(struct data_index) + sizeof(struct data_nodelist) +
					     sizeof(struct nearestdata_in) + sizeof(struct nearestdata_out) +
					     sizemax(sizeof(struct nearestdata_in),
						     sizeof(struct nearestdata_out))));
  DataIndexTable =
    (struct data_index *) mymalloc("DataIndexTable", All.BunchSize * sizeof(struct data_index));
  DataNodeList =
    (struct data_nodelist *) mymalloc("DataNodeList", All.BunchSize * sizeof(struct data_nodelist));
  maxexport = All.BunchSize - 2 * NTopleaves / NODELISTLENGTH;


  for(i = 0; i < NumPartGroup; i++)
    NgbLoc[i].count = 0;

  /* we will repeat the whole thing for those particles where we didn't find enough neighbours */

  i = 0;			/* begin with this index */

  do
    {
      for(j = 0; j < NTask; j++)
	{
	  Send_count[j] = 0;
	  Exportflag[j] = -1;
	}

      /* do local particles and prepare export list */

      for(nexport = 0; i < NumPartGroup; i++)
	{
	  if(subfind_nearesttwo_evaluate(i, 0, &nexport, Send_count) < 0)
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

      NearestDataGet =
	(struct nearestdata_in *) mymalloc("NearestDataGet", nimport * sizeof(struct nearestdata_in));
      NearestDataIn =
	(struct nearestdata_in *) mymalloc("NearestDataIn", nexport * sizeof(struct nearestdata_in));


      /* prepare particle data for export */
      for(j = 0; j < nexport; j++)
	{
	  place = DataIndexTable[j].Index;

	  NearestDataIn[j].Pos[0] = P[place].Pos[0];
	  NearestDataIn[j].Pos[1] = P[place].Pos[1];
	  NearestDataIn[j].Pos[2] = P[place].Pos[2];
	  NearestDataIn[j].Hsml = P[place].DM_Hsml;
	  NearestDataIn[j].ID = P[place].ID;
	  NearestDataIn[j].Density = P[place].u.DM_Density;
	  NearestDataIn[j].Count = NgbLoc[place].count;
	  for(k = 0; k < NgbLoc[place].count; k++)
	    {
	      NearestDataIn[j].Dist[k] = R2Loc[place].dist[k];
	      NearestDataIn[j].Index[k] = NgbLoc[place].index[k];
	    }

	  memcpy(NearestDataIn[j].NodeList,
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
		  MPI_Sendrecv(&NearestDataIn[Send_offset[recvTask]],
			       Send_count[recvTask] * sizeof(struct nearestdata_in), MPI_BYTE,
			       recvTask, TAG_DENS_A,
			       &NearestDataGet[Recv_offset[recvTask]],
			       Recv_count[recvTask] * sizeof(struct nearestdata_in), MPI_BYTE,
			       recvTask, TAG_DENS_A, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}
	    }
	}

      myfree(NearestDataIn);
      NearestDataResult =
	(struct nearestdata_out *) mymalloc("NearestDataResult", nimport * sizeof(struct nearestdata_out));
      NearestDataOut =
	(struct nearestdata_out *) mymalloc("NearestDataOut", nexport * sizeof(struct nearestdata_out));


      /* now do the particles that were sent to us */
      for(j = 0; j < nimport; j++)
	subfind_nearesttwo_evaluate(j, 1, &dummy, &dummy);

      if(i >= NumPartGroup)
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
		  MPI_Sendrecv(&NearestDataResult[Recv_offset[recvTask]],
			       Recv_count[recvTask] * sizeof(struct nearestdata_out),
			       MPI_BYTE, recvTask, TAG_DENS_B,
			       &NearestDataOut[Send_offset[recvTask]],
			       Send_count[recvTask] * sizeof(struct nearestdata_out),
			       MPI_BYTE, recvTask, TAG_DENS_B, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}
	    }
	}


      /* add the result to the local particles */
      for(j = 0; j < nexport; j++)
	{
	  place = DataIndexTable[j].Index;

	  for(k = 0; k < NearestDataOut[j].Count; k++)
	    {
	      if(NgbLoc[place].count >= 1)
		if(NgbLoc[place].index[0] == NearestDataOut[j].Index[k])
		  continue;

	      if(NgbLoc[place].count == 2)
		if(NgbLoc[place].index[1] == NearestDataOut[j].Index[k])
		  continue;

	      if(NgbLoc[place].count < 2)
		{
		  l = NgbLoc[place].count;
		  NgbLoc[place].count++;
		}
	      else
		{
		  if(R2Loc[place].dist[0] > R2Loc[place].dist[1])
		    l = 0;
		  else
		    l = 1;

		  if(NearestDataOut[j].Dist[k] >= R2Loc[place].dist[l])
		    continue;
		}

	      R2Loc[place].dist[l] = NearestDataOut[j].Dist[k];
	      NgbLoc[place].index[l] = NearestDataOut[j].Index[k];


	      if(NgbLoc[place].count == 2)
		if(NgbLoc[place].index[0] == NgbLoc[place].index[1])
		  {
		    /*
		       printf("taaa=%d i=%d  task_0=%d index_0=%d  task_1=%d index_1=%d\n",
		       ThisTask, place,
		       NgbLoc[place].task[0], NgbLoc[place].index[0],
		       NgbLoc[place].task[1], NgbLoc[place].index[1]);

		       printf
		       ("NearestDataOut[j].Count=%d  l=%d k=%d task_0=%d index_0=%d  task_1=%d index_1=%d\n",
		       NearestDataOut[j].Count, l, k, NearestDataOut[j].Task[0], NearestDataOut[j].Index[0],
		       NearestDataOut[j].Task[1], NearestDataOut[j].Index[1]);
		     */

		    endrun(112);
		  }
	    }
	}

      myfree(NearestDataOut);
      myfree(NearestDataResult);
      myfree(NearestDataGet);
    }
  while(ndone < NTask);

  myfree(DataNodeList);
  myfree(DataIndexTable);

  myfree(Dist2list);
  myfree(Ngblist);
}


/*! This function represents the core of the SPH density computation. The
 *  target particle may either be local, or reside in the communication
 *  buffer.
 */
int subfind_nearesttwo_evaluate(int target, int mode, int *nexport, int *nsend_local)
{
  int j, k, n, count;
  MyIDType ID;
  int64_t index[2];
  double dist[2];
  int startnode, numngb, numngb_inbox, listindex = 0;
  double hmax;
  double h, h2, hinv, hinv3;
  double r2, density;
  MyDouble *pos;

  numngb = 0;

  if(mode == 0)
    {
      ID = P[target].ID;
      density = P[target].u.DM_Density;
      pos = P[target].Pos;
      h = P[target].DM_Hsml;
      count = NgbLoc[target].count;
      for(k = 0; k < count; k++)
	{
	  dist[k] = R2Loc[target].dist[k];
	  index[k] = NgbLoc[target].index[k];
	}
    }
  else
    {
      ID = NearestDataGet[target].ID;
      density = NearestDataGet[target].Density;
      pos = NearestDataGet[target].Pos;
      h = NearestDataGet[target].Hsml;
      count = NearestDataGet[target].Count;
      for(k = 0; k < count; k++)
	{
	  dist[k] = NearestDataGet[target].Dist[k];
	  index[k] = NearestDataGet[target].Index[k];
	}
    }

  h2 = h * h;
  hinv = 1.0 / h;
  hinv3 = hinv * hinv * hinv;


  if(mode == 0)
    {
      startnode = All.MaxPart;	/* root node */
    }
  else
    {
      startnode = NearestDataGet[target].NodeList[0];
      startnode = Nodes[startnode].u.d.nextnode;	/* open it */
    }

  numngb = 0;

  if(count == 2)
    if(index[0] == index[1])
      {
	/*      printf("T-A-S-K=%d target=%d mode=%d  task_0=%d index_0=%d  task_1=%d index_1=%d\n",
	   ThisTask, target, mode, task[0], index[0], task[1], index[1]);
	 */
	endrun(1232);
      }

  while(startnode >= 0)
    {
      while(startnode >= 0)
	{
	  numngb_inbox =
	    subfind_ngb_treefind_nearesttwo(pos, h, target, &startnode, mode, &hmax, nexport, nsend_local);

	  if(numngb_inbox < 0)
	    return -1;

	  for(n = 0; n < numngb_inbox; n++)
	    {
	      j = Ngblist[n];
	      r2 = Dist2list[n];

	      if(P[j].ID != ID)	/* exclude the self-particle */
		{
		  if(P[j].u.DM_Density > density)	/* we only look at neighbours that are denser */
		    {
		      if(count < 2)
			{
			  dist[count] = r2;
			  index[count] = (((int64_t) ThisTask) << 32) + j;
			  count++;
			}
		      else
			{
			  if(dist[0] > dist[1])
			    k = 0;
			  else
			    k = 1;

			  if(r2 < dist[k])
			    {
			      dist[k] = r2;
			      index[k] = (((int64_t) ThisTask) << 32) + j;
			    }
			}
		    }

		  numngb++;
		}
	    }
	}

      if(mode == 1)
	{
	  listindex++;
	  if(listindex < NODELISTLENGTH)
	    {
	      startnode = NearestDataGet[target].NodeList[listindex];
	      if(startnode >= 0)
		startnode = Nodes[startnode].u.d.nextnode;	/* open it */
	    }
	}
    }

  if(mode == 0)
    {
      NgbLoc[target].count = count;
      for(k = 0; k < count; k++)
	{
	  R2Loc[target].dist[k] = dist[k];
	  NgbLoc[target].index[k] = index[k];
	}

      if(count == 2)
	if(NgbLoc[target].index[0] == NgbLoc[target].index[1])
	  {
	    /*
	       printf("TASK=%d target=%d  task_0=%d index_0=%d  task_1=%d index_1=%d\n",
	       ThisTask, target,
	       NgbLoc[target].task[0], NgbLoc[target].index[0],
	       NgbLoc[target].task[1], NgbLoc[target].index[1]);
	     */
	    endrun(2);
	  }
    }
  else
    {
      NearestDataResult[target].Count = count;
      for(k = 0; k < count; k++)
	{
	  NearestDataResult[target].Dist[k] = dist[k];
	  NearestDataResult[target].Index[k] = index[k];
	}

      if(count == 2)
	if(NearestDataResult[target].Index[0] == NearestDataResult[target].Index[1])
	  {
	    /*
	       printf("TASK!!=%d target=%d  task_0=%d index_0=%d  task_1=%d index_1=%d\n",
	       ThisTask, target,
	       NearestDataResult[target].Task[0], NearestDataResult[target].Index[0],
	       NearestDataResult[target].Task[1], NearestDataResult[target].Index[1]);
	     */
	    endrun(22);
	  }
    }

  return 0;
}




int subfind_ngb_treefind_nearesttwo(MyDouble searchcenter[3], double hsml, int target, int *startnode,
				    int mode, double *hmax, int *nexport, int *nsend_local)
{
  int numngb, no, p, task, nexport_save, exported = 0;
  struct NODE *current;
  double dx, dy, dz, dist, r2;

#ifdef PERIODIC
  double xtmp;
#endif

  nexport_save = *nexport;

  *hmax = 0;
  numngb = 0;
  no = *startnode;

  while(no >= 0)
    {
      if(no < All.MaxPart)	/* single particle */
	{
	  p = no;
	  no = Nextnode[no];

#ifdef DENSITY_SPLIT_BY_TYPE
	  if(!((1 << P[p].Type) & (DENSITY_SPLIT_BY_TYPE)))
	    continue;
#else
	  if(!((1 << P[p].Type) & (FOF_PRIMARY_LINK_TYPES)))
	    continue;
#endif

	  dist = hsml;
	  dx = NGB_PERIODIC_LONG_X(P[p].Pos[0] - searchcenter[0]);
	  if(dx > dist)
	    continue;
	  dy = NGB_PERIODIC_LONG_Y(P[p].Pos[1] - searchcenter[1]);
	  if(dy > dist)
	    continue;
	  dz = NGB_PERIODIC_LONG_Z(P[p].Pos[2] - searchcenter[2]);
	  if(dz > dist)
	    continue;
	  if((r2 = (dx * dx + dy * dy + dz * dz)) > dist * dist)
	    continue;

	  Dist2list[numngb] = r2;
	  Ngblist[numngb++] = p;
	}
      else
	{
	  if(no >= All.MaxPart + MaxNodes)	/* pseudo particle */
	    {
	      if(mode == 1)
		endrun(12312);

	      if(target >= 0)	/* if no target is given, export will not occur */
		{
		  exported = 1;
		  if(Exportflag[task = DomainTask[no - (All.MaxPart + MaxNodes)]] != target)
		    {
		      Exportflag[task] = target;
		      Exportnodecount[task] = NODELISTLENGTH;
		    }

		  if(Exportnodecount[task] == NODELISTLENGTH)
		    {
		      if(*nexport >= All.BunchSize)
			{
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

		  DataNodeList[Exportindex[task]].NodeList[Exportnodecount[task]++] =
		    DomainNodeIndex[no - (All.MaxPart + MaxNodes)];

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

	  no = current->u.d.sibling;	/* in case the node can be discarded */

	  dist = hsml + 0.5 * current->len;;
	  dx = NGB_PERIODIC_LONG_X(current->center[0] - searchcenter[0]);
	  if(dx > dist)
	    continue;
	  dy = NGB_PERIODIC_LONG_Y(current->center[1] - searchcenter[1]);
	  if(dy > dist)
	    continue;
	  dz = NGB_PERIODIC_LONG_Z(current->center[2] - searchcenter[2]);
	  if(dz > dist)
	    continue;
	  /* now test against the minimal sphere enclosing everything */
	  dist += FACT1 * current->len;
	  if(dx * dx + dy * dy + dz * dz > dist * dist)
	    continue;

	  no = current->u.d.nextnode;	/* ok, we need to open the node */
	}
    }

  *startnode = -1;
  return numngb;
}









#endif
