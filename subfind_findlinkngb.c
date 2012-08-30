
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include <gsl/gsl_math.h>


#ifdef SUBFIND

#include "allvars.h"
#include "proto.h"
#include "subfind.h"


/*! Structure for communication during the density computation. Holds data that is sent to other processors.
 */
static struct linkngbdata_in
{
  MyDouble Pos[3];
  MyFloat DM_Hsml;
  int NodeList[NODELISTLENGTH];
}
 *LinkngbDataIn, *LinkngbDataGet;


static struct linkngbdata_out
{
  int Ngb;
}
 *LinkngbDataResult, *LinkngbDataOut;



void subfind_find_linkngb(void)
{
  long long ntot;
  int i, j, ndone, ndone_flag, npleft, dummy, iter = 0, save_DesNumNgb;
  MyFloat *Left, *Right;
  char *Todo;
  int ngrp, sendTask, recvTask, place, nexport, nimport;
  double dmax1, dmax2, t0, t1;


  if(ThisTask == 0)
    printf("Start find_linkngb (%d particles on task=%d)\n", NumPartGroup, ThisTask);

  save_DesNumNgb = All.DesNumNgb;
  All.DesNumNgb = All.DesLinkNgb;	/* for simplicity, reset this value */


  /* allocate buffers to arrange communication */

  Ngblist = (int *) mymalloc("Ngblist", NumPartGroup * sizeof(int));
  Dist2list = (double *) mymalloc("Dist2list", NumPartGroup * sizeof(double));

  All.BunchSize =
    (int) ((All.BufferSize * 1024 * 1024) / (sizeof(struct data_index) + sizeof(struct data_nodelist) +
					     sizeof(struct linkngbdata_in) + sizeof(struct linkngbdata_out) +
					     sizemax(sizeof(struct linkngbdata_in),
						     sizeof(struct linkngbdata_out))));
  DataIndexTable =
    (struct data_index *) mymalloc("DataIndexTable", All.BunchSize * sizeof(struct data_index));
  DataNodeList =
    (struct data_nodelist *) mymalloc("DataNodeList", All.BunchSize * sizeof(struct data_nodelist));

  Left = mymalloc("Left", sizeof(MyFloat) * NumPartGroup);
  Right = mymalloc("Right", sizeof(MyFloat) * NumPartGroup);
  Todo = mymalloc("Todo", sizeof(char) * NumPartGroup);

  for(i = 0; i < NumPartGroup; i++)
    {
      Left[i] = Right[i] = 0;
      Todo[i] = 1;
    }

  /* we will repeat the whole thing for those particles where we didn't find enough neighbours */
  do
    {
      t0 = second();

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
	      if(Todo[i])
		{
		  if(subfind_linkngb_evaluate(i, 0, &nexport, Send_count) < 0)
		    break;
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

	  LinkngbDataGet =
	    (struct linkngbdata_in *) mymalloc("	  LinkngbDataGet",
					       nimport * sizeof(struct linkngbdata_in));
	  LinkngbDataIn =
	    (struct linkngbdata_in *) mymalloc("	  LinkngbDataIn",
					       nexport * sizeof(struct linkngbdata_in));

	  /* prepare particle data for export */
	  for(j = 0; j < nexport; j++)
	    {
	      place = DataIndexTable[j].Index;

	      LinkngbDataIn[j].Pos[0] = P[place].Pos[0];
	      LinkngbDataIn[j].Pos[1] = P[place].Pos[1];
	      LinkngbDataIn[j].Pos[2] = P[place].Pos[2];
	      LinkngbDataIn[j].DM_Hsml = P[place].DM_Hsml;

	      memcpy(LinkngbDataIn[j].NodeList,
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
		      MPI_Sendrecv(&LinkngbDataIn[Send_offset[recvTask]],
				   Send_count[recvTask] * sizeof(struct linkngbdata_in), MPI_BYTE,
				   recvTask, TAG_DENS_A,
				   &LinkngbDataGet[Recv_offset[recvTask]],
				   Recv_count[recvTask] * sizeof(struct linkngbdata_in), MPI_BYTE,
				   recvTask, TAG_DENS_A, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		    }
		}
	    }

	  myfree(LinkngbDataIn);
	  LinkngbDataResult =
	    (struct linkngbdata_out *) mymalloc("	  LinkngbDataResult",
						nimport * sizeof(struct linkngbdata_out));
	  LinkngbDataOut =
	    (struct linkngbdata_out *) mymalloc("	  LinkngbDataOut",
						nexport * sizeof(struct linkngbdata_out));


	  /* now do the particles that were sent to us */
	  for(j = 0; j < nimport; j++)
	    subfind_linkngb_evaluate(j, 1, &dummy, &dummy);

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
		      MPI_Sendrecv(&LinkngbDataResult[Recv_offset[recvTask]],
				   Recv_count[recvTask] * sizeof(struct linkngbdata_out),
				   MPI_BYTE, recvTask, TAG_DENS_B,
				   &LinkngbDataOut[Send_offset[recvTask]],
				   Send_count[recvTask] * sizeof(struct linkngbdata_out),
				   MPI_BYTE, recvTask, TAG_DENS_B, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		    }
		}
	    }

	  /* add the result to the local particles */
	  for(j = 0; j < nexport; j++)
	    {
	      place = DataIndexTable[j].Index;

	      P[place].DM_NumNgb += LinkngbDataOut[j].Ngb;
	    }


	  myfree(LinkngbDataOut);
	  myfree(LinkngbDataResult);
	  myfree(LinkngbDataGet);
	}
      while(ndone < NTask);

      /* do final operations on results */
      for(i = 0, npleft = 0; i < NumPartGroup; i++)
	{
	  /* now check whether we had enough neighbours */
	  if(Todo[i])
	    {
	      if(P[i].DM_NumNgb != All.DesLinkNgb &&
		 ((Right[i] - Left[i]) > 1.0e-3 * Left[i] || Left[i] == 0 || Right[i] == 0))
		{
		  /* need to redo this particle */
		  npleft++;

		  if(P[i].DM_NumNgb < All.DesLinkNgb)
		    Left[i] = DMAX(P[i].DM_Hsml, Left[i]);
		  else
		    {
		      if(Right[i] != 0)
			{
			  if(P[i].DM_Hsml < Right[i])
			    Right[i] = P[i].DM_Hsml;
			}
		      else
			Right[i] = P[i].DM_Hsml;
		    }

		  if(iter >= MAXITER - 10)
		    {
		      printf
			("i=%d task=%d ID=%d DM_Hsml=%g Left=%g Right=%g Ngbs=%g Right-Left=%g\n   pos=(%g|%g|%g)\n",
			 i, ThisTask, (int) P[i].ID, P[i].DM_Hsml, Left[i], Right[i],
			 (double) P[i].DM_NumNgb, Right[i] - Left[i], P[i].Pos[0], P[i].Pos[1], P[i].Pos[2]);
		      fflush(stdout);
		    }

		  if(Right[i] > 0 && Left[i] > 0)
		    P[i].DM_Hsml = pow(0.5 * (pow(Left[i], 3) + pow(Right[i], 3)), 1.0 / 3);
		  else
		    {
		      if(Right[i] == 0 && Left[i] == 0)
			endrun(8189);	/* can't occur */

		      if(Right[i] == 0 && Left[i] > 0)
			P[i].DM_Hsml *= 1.26;

		      if(Right[i] > 0 && Left[i] == 0)
			P[i].DM_Hsml /= 1.26;
		    }
		}
	      else
		Todo[i] = 0;
	    }
	}


      sumup_large_ints(1, &npleft, &ntot);

      t1 = second();

      if(ntot > 0)
	{
	  iter++;

	  if(iter > 0 && ThisTask == 0)
	    {
	      printf("find linkngb iteration %d: need to repeat for %d%09d particles. (took %g sec)\n", iter,
		     (int) (ntot / 1000000000), (int) (ntot % 1000000000), timediff(t0, t1));
	      fflush(stdout);
	    }

	  if(iter > MAXITER)
	    {
	      printf("failed to converge in neighbour iteration in density()\n");
	      fflush(stdout);
	      endrun(1155);
	    }
	}
    }
  while(ntot > 0);

  myfree(Todo);
  myfree(Right);
  myfree(Left);

  myfree(DataNodeList);
  myfree(DataIndexTable);

  myfree(Dist2list);
  myfree(Ngblist);

  All.DesNumNgb = save_DesNumNgb;	/* restore it */
}


/*! This function represents the core of the SPH density computation. The
 *  target particle may either be local, or reside in the communication
 *  buffer.
 */
int subfind_linkngb_evaluate(int target, int mode, int *nexport, int *nsend_local)
{
  int startnode, numngb, ngbs, listindex = 0;
  double hmax;
  double h, h2, hinv, hinv3;
  MyDouble *pos;

  numngb = 0;

  if(mode == 0)
    {
      pos = P[target].Pos;
      h = P[target].DM_Hsml;
    }
  else
    {
      pos = LinkngbDataGet[target].Pos;
      h = LinkngbDataGet[target].DM_Hsml;
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
      startnode = LinkngbDataGet[target].NodeList[0];
      startnode = Nodes[startnode].u.d.nextnode;	/* open it */
    }

  numngb = 0;

  while(startnode >= 0)
    {
      while(startnode >= 0)
	{
	  ngbs = subfind_ngb_treefind_linkngb(pos, h, target, &startnode, mode, &hmax, nexport, nsend_local);

	  if(ngbs < 0)
	    return -1;

	  if(mode == 0 && hmax > 0)
	    P[target].DM_Hsml = hmax;

	  numngb += ngbs;
	}

      if(mode == 1)
	{
	  listindex++;
	  if(listindex < NODELISTLENGTH)
	    {
	      startnode = LinkngbDataGet[target].NodeList[listindex];
	      if(startnode >= 0)
		startnode = Nodes[startnode].u.d.nextnode;	/* open it */
	    }
	}
    }

  if(mode == 0)
    {
      P[target].DM_NumNgb = numngb;
    }
  else
    {
      LinkngbDataResult[target].Ngb = numngb;
    }

  return 0;
}




int subfind_ngb_treefind_linkngb(MyDouble searchcenter[3], double hsml, int target, int *startnode, int mode,
				 double *hmax, int *nexport, int *nsend_local)
{
  int numngb, i, no, p, task, nexport_save, exported = 0;
  struct NODE *current;
  double dx, dy, dz, dist, r2;

#ifdef PERIODIC
  MyDouble xtmp;
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
#else
	  if(!((1 << P[p].Type) & (FOF_PRIMARY_LINK_TYPES)))
#endif
	    continue;

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


  if(mode == 0)			/* local particle */
    if(exported == 0)		/* completely local */
      if(numngb >= All.DesNumNgb)
	{
	  R2list = mymalloc("	  R2list", sizeof(struct r2data) * numngb);
	  for(i = 0; i < numngb; i++)
	    {
	      R2list[i].index = Ngblist[i];
	      R2list[i].r2 = Dist2list[i];
	    }

	  qsort(R2list, numngb, sizeof(struct r2data), subfind_ngb_compare_dist);

	  *hmax = sqrt(R2list[All.DesNumNgb - 1].r2);
	  numngb = All.DesNumNgb;

	  for(i = 0; i < numngb; i++)
	    {
	      Ngblist[i] = R2list[i].index;
	      Dist2list[i] = R2list[i].r2;
	    }

	  myfree(R2list);
	}


  *startnode = -1;
  return numngb;
}

#ifdef DENSITY_SPLIT_BY_TYPE
/*! This routine finds all neighbours `j' that can interact with
 *  \f$ r_{ij} < h_i \f$  OR if  \f$ r_{ij} < h_j \f$.
 */
int subfind_ngb_treefind_linkpairs(MyDouble searchcenter[3], double hsml, int target, int *startnode,
				   int mode, double *hmax, int *nexport, int *nsend_local)
{
  int numngb, i, no, p, task, nexport_save, exported = 0;
  struct NODE *current;
  double dx, dy, dz, dist, r2, dmax1, dmax2;

#ifdef PERIODIC
  MyDouble xtmp;
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
#else
	  if(!((1 << P[p].Type) & (FOF_PRIMARY_LINK_TYPES)))
#endif
	    continue;

	  dist = DMAX(P[p].DM_Hsml, hsml);
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

	  dist = DMAX(Extnodes[no].hmax, hsml) + 0.5 * current->len;
	  no = current->u.d.sibling;	/* in case the node can be discarded */
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


  if(mode == 0)			/* local particle */
    if(exported == 0)		/* completely local */
      if(numngb >= All.DesNumNgb)
	{
	  R2list = mymalloc("	  R2list", sizeof(struct r2data) * numngb);
	  for(i = 0; i < numngb; i++)
	    {
	      R2list[i].index = Ngblist[i];
	      R2list[i].r2 = Dist2list[i];
	    }

	  qsort(R2list, numngb, sizeof(struct r2data), subfind_ngb_compare_dist);

	  *hmax = sqrt(R2list[All.DesNumNgb - 1].r2);
	  numngb = All.DesNumNgb;

	  for(i = 0; i < numngb; i++)
	    {
	      Ngblist[i] = R2list[i].index;
	      Dist2list[i] = R2list[i].r2;
	    }

	  myfree(R2list);
	}


  *startnode = -1;
  return numngb;
}
#endif

int subfind_ngb_compare_dist(const void *a, const void *b)
{
  if(((struct r2data *) a)->r2 < (((struct r2data *) b)->r2))
    return -1;

  if(((struct r2data *) a)->r2 > (((struct r2data *) b)->r2))
    return +1;

  return 0;
}




#endif
