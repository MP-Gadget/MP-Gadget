#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <gsl/gsl_math.h>
#include <sys/stat.h>
#include <sys/types.h>


#include "allvars.h"
#include "proto.h"


#ifdef SUBFIND

#include "fof.h"
#include "subfind.h"


/*! Structure for communication during the density computation. Holds data that is sent to other processors.
 */
static struct contamdata_in
{
  MyDouble Pos[3];
  MyFloat R200;
  int NodeList[NODELISTLENGTH];
}
 *ContamIn, *ContamGet;


static struct contamdata_out
{
  double ContaminationMass;
  int ContaminationLen;
}
 *ContamResult, *ContamOut;



void subfind_contamination(void)
{
  int i, j, ndone, ndone_flag, dummy, count;
  int ngrp, sendTask, recvTask, place, nexport, nimport;
  struct unbind_data *d;

  d = (struct unbind_data *) mymalloc("d", NumPart * sizeof(struct unbind_data));

  for(i = 0, count = 0; i < NumPart; i++)
#ifdef DENSITY_SPLIT_BY_TYPE
    if(!((1 << P[i].Type) & (DENSITY_SPLIT_BY_TYPE)))
#else
    if(!((1 << P[i].Type) & (FOF_PRIMARY_LINK_TYPES)))
#endif
      d[count++].index = i;

  force_treebuild(count, d);	/* construct tree only with boundary particles */

  myfree(d);


  /* allocate buffers to arrange communication */

  All.BunchSize =
    (int) ((All.BufferSize * 1024 * 1024) / (sizeof(struct data_index) + sizeof(struct data_nodelist) +
					     sizeof(struct contamdata_in) + sizeof(struct contamdata_out) +
					     sizemax(sizeof(struct contamdata_in),
						     sizeof(struct contamdata_out))));
  DataIndexTable =
    (struct data_index *) mymalloc("DataIndexTable", All.BunchSize * sizeof(struct data_index));
  DataNodeList =
    (struct data_nodelist *) mymalloc("DataNodeList", All.BunchSize * sizeof(struct data_nodelist));


  /* we will repeat the whole thing for those groups where we didn't converge to a SO radius yet */

  i = 0;			/* begin with this index */

  do
    {
      for(j = 0; j < NTask; j++)
	{
	  Send_count[j] = 0;
	  Exportflag[j] = -1;
	}

      /* do local particles and prepare export list */

      for(nexport = 0; i < Ngroups; i++)
	{
	  if(Group[i].R_Mean200 > 0)
	    {
	      if(subfind_contamination_evaluate(i, 0, &nexport, Send_count) < 0)
		break;
	    }
	  else
	    {
	      Group[i].ContaminationLen = 0;
	      Group[i].ContaminationMass = 0;
	    }
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

      ContamGet = (struct contamdata_in *) mymalloc("ContamGet", nimport * sizeof(struct contamdata_in));
      ContamIn = (struct contamdata_in *) mymalloc("ContamIn", nexport * sizeof(struct contamdata_in));

      /* prepare particle data for export */
      for(j = 0; j < nexport; j++)
	{
	  place = DataIndexTable[j].Index;

	  ContamIn[j].Pos[0] = Group[place].Pos[0];
	  ContamIn[j].Pos[1] = Group[place].Pos[1];
	  ContamIn[j].Pos[2] = Group[place].Pos[2];
	  ContamIn[j].R200 = Group[place].R_Mean200;

	  memcpy(ContamIn[j].NodeList,
		 DataNodeList[DataIndexTable[j].IndexGet].NodeList, NODELISTLENGTH * sizeof(int));
	}

      /* exchange data */
      for(ngrp = 1; ngrp < (1 << PTask); ngrp++)
	{
	  sendTask = ThisTask;
	  recvTask = ThisTask ^ ngrp;

	  if(recvTask < NTask)
	    {
	      if(Send_count[recvTask] > 0 || Recv_count[recvTask] > 0)
		{
		  /* get the data */
		  MPI_Sendrecv(&ContamIn[Send_offset[recvTask]],
			       Send_count[recvTask] * sizeof(struct contamdata_in), MPI_BYTE,
			       recvTask, TAG_DENS_A,
			       &ContamGet[Recv_offset[recvTask]],
			       Recv_count[recvTask] * sizeof(struct contamdata_in), MPI_BYTE,
			       recvTask, TAG_DENS_A, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}
	    }
	}

      myfree(ContamIn);
      ContamResult =
	(struct contamdata_out *) mymalloc("ContamResult", nimport * sizeof(struct contamdata_out));
      ContamOut = (struct contamdata_out *) mymalloc("ContamOut", nexport * sizeof(struct contamdata_out));


      /* now do the locations that were sent to us */
      for(j = 0; j < nimport; j++)
	subfind_contamination_evaluate(j, 1, &dummy, &dummy);

      if(i >= Ngroups)
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
		  MPI_Sendrecv(&ContamResult[Recv_offset[recvTask]],
			       Recv_count[recvTask] * sizeof(struct contamdata_out),
			       MPI_BYTE, recvTask, TAG_DENS_B,
			       &ContamOut[Send_offset[recvTask]],
			       Send_count[recvTask] * sizeof(struct contamdata_out),
			       MPI_BYTE, recvTask, TAG_DENS_B, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}
	    }
	}

      /* add the result to the local particles */
      for(j = 0; j < nexport; j++)
	{
	  place = DataIndexTable[j].Index;
	  Group[place].ContaminationLen += ContamOut[j].ContaminationLen;
	  Group[place].ContaminationMass += ContamOut[j].ContaminationMass;
	}

      myfree(ContamOut);
      myfree(ContamResult);
      myfree(ContamGet);
    }
  while(ndone < NTask);


  myfree(DataNodeList);
  myfree(DataIndexTable);
}


/*! This function represents the core of the SPH density computation. The
 *  target particle may either be local, or reside in the communication
 *  buffer.
 */
int subfind_contamination_evaluate(int target, int mode, int *nexport, int *nsend_local)
{
  int startnode, listindex = 0;
  double h, mass, masssum;
  int count, countsum;
  MyDouble *pos;

  masssum = 0;
  countsum = 0;

  if(mode == 0)
    {
      pos = Group[target].Pos;
      h = Group[target].R_Mean200;
    }
  else
    {
      pos = ContamGet[target].Pos;
      h = ContamGet[target].R200;
    }

  if(mode == 0)
    {
      startnode = All.MaxPart;	/* root node */
    }
  else
    {
      startnode = ContamGet[target].NodeList[0];
      startnode = Nodes[startnode].u.d.nextnode;	/* open it */
    }

  while(startnode >= 0)
    {
      while(startnode >= 0)
	{
	  count =
	    subfind_contamination_treefind(pos, h, target, &startnode, mode, nexport, nsend_local, &mass);

	  if(count < 0)
	    return -1;

	  masssum += mass;
	  countsum += count;
	}

      if(mode == 1)
	{
	  listindex++;
	  if(listindex < NODELISTLENGTH)
	    {
	      startnode = ContamGet[target].NodeList[listindex];
	      if(startnode >= 0)
		startnode = Nodes[startnode].u.d.nextnode;	/* open it */
	    }
	}
    }

  if(mode == 0)
    {
      Group[target].ContaminationMass = masssum;
      Group[target].ContaminationLen = countsum;
    }
  else
    {
      ContamResult[target].ContaminationMass = masssum;
      ContamResult[target].ContaminationLen = countsum;
    }

  return 0;
}


int subfind_contamination_treefind(MyDouble searchcenter[3], MyFloat hsml, int target, int *startnode,
				   int mode, int *nexport, int *nsend_local, double *Mass)
{
  int no, p, task, nexport_save;
  struct NODE *current;
  double mass;
  int count;
  MyDouble dx, dy, dz, dist, r2;

  nexport_save = *nexport;

  mass = 0;
  count = 0;

  no = *startnode;

  while(no >= 0)
    {
      if(no < All.MaxPart)	/* single particle */
	{
	  p = no;
	  no = Nextnode[no];

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
	  if(dx * dx + dy * dy + dz * dz > dist * dist)
	    continue;

	  mass += P[p].Mass;
	  count++;
	}
      else
	{
	  if(no >= All.MaxPart + MaxNodes)	/* pseudo particle */
	    {
	      if(mode == 1)
		endrun(12312);

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
			  *nexport = nexport_save;
			  if(nexport_save == 0)
			    endrun(13005);	/* in this case, the buffer is too small to process even a single particle */
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
		  *Mass = mass;
		  return count;
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
	  if((r2 = (dx * dx + dy * dy + dz * dz)) > dist * dist)
	    continue;

	  no = current->u.d.nextnode;	/* ok, we need to open the node */
	}
    }

  *startnode = -1;

  *Mass = mass;
  return count;
}




#endif
