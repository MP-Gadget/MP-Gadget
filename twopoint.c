#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include <gsl/gsl_rng.h>

#include "allvars.h"
#include "proto.h"

/*! \file twopoint.c
 *  \brief computes the two-point mass correlation function on the fly
 */

/* Note: This routine will only work correctly for particles of equal mass ! */


#define BINS_TP  40		/* number of bins used */
#define ALPHA  -1.0		/* slope used in randomly selecting radii around target particles */

#ifndef FRACTION_TP
#define FRACTION_TP  0.2
#endif /* fraction of particles selected for sphere
          placement. Will be scaled with total
          particle number so that a fixed value
          should give roughly the same noise level
          in the meaurement, indpendent of
          simulation size */

struct twopointdata_in
{
  MyDouble Pos[3];
  MyFloat Rs;
  int NodeList[NODELISTLENGTH];
}
 *TwoPointDataIn, *TwoPointDataGet;


#define SQUARE_IT(x) ((x)*(x))


static int64_t Count[BINS_TP], Count_bak[BINS_TP];
static int64_t CountSpheres[BINS_TP];
static double Xi[BINS_TP];
static double Rbin[BINS_TP];

static double R0, R1;		/* inner and outer radius for correlation function determination */

static double logR0;
static double binfac;
static double PartMass;

static MyFloat *RsList;



/*  This function computes the two-point function.
 */
void twopoint(void)
{
  int i, j, k, bin, n;
  double p, rs, vol, scaled_frac;
  int ndone, ndone_flag, dummy, nexport, nimport, place;
  int64_t *countbuf;
  int sendTask, recvTask, ngrp;
  double tstart, tend;
  double mass, masstot;
  void *state_buffer;


  if(ThisTask == 0)
    {
      printf("begin two-point correlation function...\n");
      fflush(stdout);
    }

  tstart = second();


  /* set inner and outer radius for the bins that are used for the correlation function estimate */
  R0 = All.SofteningTable[1];	/* we assume that type=1 is the primary type */
  R1 = All.BoxSize / 2;


  scaled_frac = FRACTION_TP * 1.0e7 / All.TotNumPart;

  logR0 = log(R0);
  binfac = BINS_TP / (log(R1) - log(R0));


  for(i = 0, mass = 0; i < NumPart; i++)
    mass += P[i].Mass;

  MPI_Allreduce(&mass, &masstot, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  PartMass = masstot / All.TotNumPart;


  for(i = 0; i < BINS_TP; i++)
    {
      Count[i] = 0;
      CountSpheres[i] = 0;
    }

  /* allocate buffers to arrange communication */

  RsList = (MyFloat *) mymalloc("RsList", NumPart * sizeof(MyFloat));

  All.BunchSize =
    (int) ((All.BufferSize * 1024 * 1024) / (sizeof(struct data_index) + sizeof(struct data_nodelist) +
					     2 * sizeof(struct twopointdata_in)));
  DataIndexTable =
    (struct data_index *) mymalloc("DataIndexTable", All.BunchSize * sizeof(struct data_index));
  DataNodeList =
    (struct data_nodelist *) mymalloc("DataNodeList", All.BunchSize * sizeof(struct data_nodelist));




  state_buffer = mymalloc("state_buffer", gsl_rng_size(random_generator));

  memcpy(state_buffer, gsl_rng_state(random_generator), gsl_rng_size(random_generator));

  gsl_rng_set(random_generator, P[0].ID + ThisTask);	/* seed things with first particle ID to make sure we are
							   different on each CPU */


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
	if(gsl_rng_uniform(random_generator) < scaled_frac)
	  {
	    p = gsl_rng_uniform(random_generator);

	    rs = pow(pow(R0, ALPHA) + p * (pow(R1, ALPHA) - pow(R0, ALPHA)), 1 / ALPHA);

	    bin = (log(rs) - logR0) * binfac;

	    rs = exp((bin + 1) / binfac + logR0);

	    RsList[i] = rs;

	    if(twopoint_count_local(i, 0, &nexport, Send_count) < 0)
	      break;

	    for(j = 0; j <= bin; j++)
	      CountSpheres[j]++;
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

      TwoPointDataGet =
	(struct twopointdata_in *) mymalloc("TwoPointDataGet", nimport * sizeof(struct twopointdata_in));
      TwoPointDataIn =
	(struct twopointdata_in *) mymalloc("TwoPointDataIn", nexport * sizeof(struct twopointdata_in));

      for(j = 0; j < nexport; j++)
	{
	  place = DataIndexTable[j].Index;

	  for(k = 0; k < 3; k++)
	    TwoPointDataIn[j].Pos[k] = P[place].Pos[k];

	  TwoPointDataIn[j].Rs = RsList[place];

	  memcpy(TwoPointDataIn[j].NodeList,
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
		  MPI_Sendrecv(&TwoPointDataIn[Send_offset[recvTask]],
			       Send_count[recvTask] * sizeof(struct twopointdata_in), MPI_BYTE,
			       recvTask, TAG_HYDRO_A,
			       &TwoPointDataGet[Recv_offset[recvTask]],
			       Recv_count[recvTask] * sizeof(struct twopointdata_in), MPI_BYTE,
			       recvTask, TAG_HYDRO_A, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}
	    }
	}

      myfree(TwoPointDataIn);


      for(j = 0; j < nimport; j++)
	twopoint_count_local(j, 1, &dummy, &dummy);

      if(i >= NumPart)
	ndone_flag = 1;
      else
	ndone_flag = 0;

      MPI_Allreduce(&ndone_flag, &ndone, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

      myfree(TwoPointDataGet);
    }
  while(ndone < NTask);

  memcpy(gsl_rng_state(random_generator), state_buffer, gsl_rng_size(random_generator));
  myfree(state_buffer);

  myfree(DataNodeList);
  myfree(DataIndexTable);

  myfree(RsList);


  /* Now compute the actual correlation function */

  countbuf = mymalloc("countbuf", NTask * BINS_TP * sizeof(int64_t));

  MPI_Allgather(Count, BINS_TP * sizeof(int64_t), MPI_BYTE,
		countbuf, BINS_TP * sizeof(int64_t), MPI_BYTE, MPI_COMM_WORLD);

  for(i = 0; i < BINS_TP; i++)
    {
      Count[i] = 0;
      for(n = 0; n < NTask; n++)
	Count[i] += countbuf[n * BINS_TP + i];
    }

  MPI_Allgather(CountSpheres, BINS_TP * sizeof(int64_t), MPI_BYTE,
		countbuf, BINS_TP * sizeof(int64_t), MPI_BYTE, MPI_COMM_WORLD);

  for(i = 0; i < BINS_TP; i++)
    {
      CountSpheres[i] = 0;
      for(n = 0; n < NTask; n++)
	CountSpheres[i] += countbuf[n * BINS_TP + i];
    }

  myfree(countbuf);


  for(i = 0; i < BINS_TP; i++)
    {
      vol = 4 * M_PI / 3.0 * (pow(exp((i + 1.0) / binfac + logR0), 3)
			      - pow(exp((i + 0.0) / binfac + logR0), 3));

      if(CountSpheres[i] > 0)
	Xi[i] = -1 + Count[i] / ((double) CountSpheres[i]) / (All.TotNumPart / pow(All.BoxSize, 3)) / vol;
      else
	Xi[i] = 0;

      Rbin[i] = exp((i + 0.5) / binfac + logR0);
    }

  twopoint_save();

  tend = second();

  if(ThisTask == 0)
    {
      printf("end two-point: Took=%g seconds.\n", timediff(tstart, tend));
      fflush(stdout);
    }
}




void twopoint_save(void)
{
  FILE *fd;
  char buf[500];
  int i;

  if(ThisTask == 0)
    {
      sprintf(buf, "%s/correl_%03d.txt", All.OutputDir, RestartSnapNum);

      if(!(fd = fopen(buf, "w")))
	{
	  printf("can't open file `%s`\n", buf);
	  endrun(1323);
	}

      fprintf(fd, "%g\n", All.Time);
      i = BINS_TP;
      fprintf(fd, "%d\n", i);

      for(i = 0; i < BINS_TP; i++)
	fprintf(fd, "%g %g %g %g\n", Rbin[i], Xi[i], (double) Count[i], (double) CountSpheres[i]);

      fclose(fd);
    }
}




/*! This function counts the pairs in a sphere
 */
int twopoint_count_local(int target, int mode, int *nexport, int *nsend_local)
{
  int startnode, listindex = 0;
  MyDouble *pos;
  MyFloat rs;

  if(mode == 0)
    {
      pos = P[target].Pos;
      rs = RsList[target];
      memcpy(Count_bak, Count, sizeof(int64_t) * BINS_TP);
    }
  else
    {
      pos = TwoPointDataGet[target].Pos;
      rs = TwoPointDataGet[target].Rs;
    }


  /* Now start the actual tree-walk for this particle */

  if(mode == 0)
    {
      startnode = All.MaxPart;	/* root node */
    }
  else
    {
      startnode = TwoPointDataGet[target].NodeList[0];
      startnode = Nodes[startnode].u.d.nextnode;	/* open it */
    }

  while(startnode >= 0)
    {
      while(startnode >= 0)
	{
	  if(twopoint_ngb_treefind_variable(pos, rs, target, &startnode, mode, nexport, nsend_local) < 0)
	    {
	      /* in this case restore the count-table */
	      memcpy(Count, Count_bak, sizeof(int64_t) * BINS_TP);
	      return -1;
	    }
	}

      if(mode == 1)
	{
	  listindex++;
	  if(listindex < NODELISTLENGTH)
	    {
	      startnode = TwoPointDataGet[target].NodeList[listindex];
	      if(startnode >= 0)
		startnode = Nodes[startnode].u.d.nextnode;	/* open it */
	    }
	}
    }

  return 0;
}





/*! This function finds all particles within the radius "rsearch",
 *  and counts them in the bins used for the two-point correlation function.
 */
int twopoint_ngb_treefind_variable(MyDouble searchcenter[3], MyFloat rsearch, int target, int *startnode,
				   int mode, int *nexport, int *nsend_local)
{
  double r2, r, ri, ro;
  int no, p, bin, task, bin2, nexport_save;
  struct NODE *current;
  MyDouble dx, dy, dz, dist;

  nexport_save = *nexport;

  no = *startnode;

  while(no >= 0)
    {
      if(no < All.MaxPart)	/* single particle */
	{
	  p = no;
	  no = Nextnode[no];

	  dx = NGB_PERIODIC_LONG_X(P[p].Pos[0] - searchcenter[0]);
	  dy = NGB_PERIODIC_LONG_Y(P[p].Pos[1] - searchcenter[1]);
	  dz = NGB_PERIODIC_LONG_Z(P[p].Pos[2] - searchcenter[2]);

	  r2 = dx * dx + dy * dy + dz * dz;

	  if(r2 >= R0 * R0 && r2 < R1 * R1)
	    {
	      if(r2 < rsearch * rsearch)
		{
		  bin = (log(sqrt(r2)) - logR0) * binfac;
		  if(bin < BINS_TP)
		    Count[bin]++;
		}
	    }
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
		  return 0;
		}
	    }

	  no = current->u.d.sibling;	/* make skipping the branch the default */

	  dist = rsearch + 0.5 * current->len;;
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
	  if((r2 = dx * dx + dy * dy + dz * dz) > dist * dist)
	    continue;

	  r = sqrt(r2);

	  ri = r - FACT2 * current->len;
	  ro = r + FACT2 * current->len;

	  if(ri >= R0 && ro < R1)
	    {
	      if(ro < rsearch)
		{
		  bin = (log(ri) - logR0) * binfac;
		  bin2 = (log(ro) - logR0) * binfac;
		  if(bin == bin2)
		    {
		      if(mode == 1)
			{
			  if((current->u.d.bitflags & (1 << BITFLAG_TOPLEVEL)))
			    continue;
			}
		      Count[bin] += current->u.d.mass / PartMass;
		      continue;
		    }
		}
	    }

	  no = current->u.d.nextnode;	/* ok, we need to open the node */
	}
    }

  *startnode = -1;
  return 0;
}
