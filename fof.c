#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <gsl/gsl_math.h>
#include <inttypes.h>

#include "allvars.h"
#include "proto.h"
#include "domain.h"

#ifdef HAVE_HDF5
#include <hdf5.h>
#endif

/*! \file fof.c
 *  \brief parallel FoF group finder
 */

#ifdef FOF
#include "fof.h"
#ifdef SUBFIND
#include "subfind.h"
#endif


int Ngroups, TotNgroups;
int64_t TotNids;

struct group_properties *Group;



static struct fofdata_in
{
  MyDouble Pos[3];
  MyFloat Hsml;
  MyIDType MinID;
  MyIDType MinIDTask;
  int NodeList[NODELISTLENGTH];
}
 *FoFDataIn, *FoFDataGet;

static struct fofdata_out
{
  MyFloat Distance;
  MyIDType MinID;
  MyIDType MinIDTask;
}
 *FoFDataResult, *FoFDataOut;


static struct fof_particle_list
{
  MyIDType MinID;
  MyIDType MinIDTask;
  int Pindex;
}
 *FOF_PList;

static struct fof_group_list
{
  MyIDType MinID;
  MyIDType MinIDTask;
  int LocCount;
  int ExtCount;
#ifdef DENSITY_SPLIT_BY_TYPE
  int LocDMCount;
  int ExtDMCount;
#endif
  int GrNr;
}
 *FOF_GList;

static struct id_list
{
  MyIDType ID;
  unsigned int GrNr;
}
 *ID_list;


static double LinkL;
static int NgroupsExt, Nids;


static MyIDType *Head, *Len, *Next, *Tail, *MinID, *MinIDTask;
static char *NonlocalFlag;


static float *fof_nearest_distance;
static float *fof_nearest_hsml;


void fof_fof(int num)
{
  int i, ndm, start, lenloc, largestgroup, n;
  double mass, masstot, rhodm, t0, t1;
  struct unbind_data *d;
  int64_t ndmtot;


#ifdef SUBFIND_DENSITY_AND_POTENTIAL
  if(ThisTask == 0)
    printf("\nStarting SUBFIND_DENSITY_AND_POTENTIAL...\n");

  subfind(num);
  strcat(All.SnapshotFileBase, "_rho_and_pot");
  savepositions(RestartSnapNum);
  if(ThisTask == 0)
    printf("\nSUBFIND_DENSITY_AND_POTENTIAL done.\n");
  endrun(0);
#endif

#ifdef SUBFIND_READ_FOF
  read_fof(num);
#endif


#ifdef SUBFIND_RESHUFFLE_CATALOGUE
  force_treefree();

  read_subfind_ids();

  if(All.TotN_sph > 0)
    {
      if(ThisTask == 0)
	printf("\nThe option SUBFIND_RESHUFFLE_CATALOGUE does not work with gas particles yet\n");
      endrun(0);
    }

  t0 = second();
  //  parallel_sort(P, NumPart, sizeof(struct particle_data), io_compare_P_GrNr_ID);
  parallel_sort_special_P_GrNr_ID();
  t1 = second();
  if(ThisTask == 0)
    printf("Ordering of particle-data took = %g sec\n", timediff(t0, t1));

  strcat(All.SnapshotFileBase, "_subidorder");
  savepositions(RestartSnapNum);
  endrun(0);
#endif


  if(ThisTask == 0)
    {
      printf("\nBegin to compute FoF group catalogues...  (presently allocated=%g MB)\n",
	     AllocatedBytes / (1024.0 * 1024.0));
      fflush(stdout);
    }

  CPU_Step[CPU_MISC] += measure_time();

  domain_Decomposition();

#ifdef ONLY_PRODUCE_HSML_FILES
  subfind(num);
  endrun(0);
#endif

  for(i = 0, ndm = 0, mass = 0; i < NumPart; i++)
#ifdef DENSITY_SPLIT_BY_TYPE
    {
      if(((1 << P[i].Type) & (FOF_PRIMARY_LINK_TYPES)))
	ndm++;
      if(((1 << P[i].Type) & (DENSITY_SPLIT_BY_TYPE)))
	mass += P[i].Mass;
    }
#else
    if(((1 << P[i].Type) & (FOF_PRIMARY_LINK_TYPES)))
      {
	ndm++;
	mass += P[i].Mass;
      }
#endif

  sumup_large_ints(1, &ndm, &ndmtot);
  MPI_Allreduce(&mass, &masstot, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

#ifdef DENSITY_SPLIT_BY_TYPE
  rhodm = All.Omega0 * 3 * All.Hubble * All.Hubble / (8 * M_PI * All.G);
#else
  rhodm = (All.Omega0 - All.OmegaBaryon) * 3 * All.Hubble * All.Hubble / (8 * M_PI * All.G);
#endif

  LinkL = LINKLENGTH * pow(masstot / ndmtot / rhodm, 1.0 / 3);

  if(ThisTask == 0)
    {
      printf("\nComoving linking length: %g    ", LinkL);
      printf("(presently allocated=%g MB)\n", AllocatedBytes / (1024.0 * 1024.0));
      fflush(stdout);
    }

  FOF_PList =
    (struct fof_particle_list *) mymalloc("FOF_PList", NumPart *
					  sizemax(sizeof(struct fof_particle_list), 3 * sizeof(MyIDType)));

  MinID = (MyIDType *) FOF_PList;
  MinIDTask = MinID + NumPart;
  Head = MinIDTask + NumPart;
  Len = (MyIDType *) mymalloc("Len", NumPart * sizeof(MyIDType));
  Next = (MyIDType *) mymalloc("Next", NumPart * sizeof(MyIDType));
  Tail = (MyIDType *) mymalloc("Tail", NumPart * sizeof(MyIDType));

  CPU_Step[CPU_FOF] += measure_time();

  if(ThisTask == 0)
    printf("Tree construction.\n");

  force_treeallocate((int) (All.TreeAllocFactor * All.MaxPart) + NTopnodes, All.MaxPart);


  /* build index list of particles of selected primary species */
  d = (struct unbind_data *) mymalloc("d", NumPart * sizeof(struct unbind_data));
  for(i = 0, n = 0; i < NumPart; i++)
    if(((1 << P[i].Type) & (FOF_PRIMARY_LINK_TYPES)))
      d[n++].index = i;

  force_treebuild(n, d);

  myfree(d);


  for(i = 0; i < NumPart; i++)
    {
      Head[i] = Tail[i] = i;
      Len[i] = 1;
      Next[i] = -1;
      MinID[i] = P[i].ID;
      MinIDTask[i] = ThisTask;
    }


  t0 = second();

  fof_find_groups();

  t1 = second();
  if(ThisTask == 0)
    printf("group finding took = %g sec\n", timediff(t0, t1));


  t0 = second();

  fof_find_nearest_dmparticle();

  t1 = second();
  if(ThisTask == 0)
    printf("attaching gas and star particles to nearest dm particles took = %g sec\n", timediff(t0, t1));


  t0 = second();

  for(i = 0; i < NumPart; i++)
    {
      Next[i] = MinID[Head[i]];
      Tail[i] = MinIDTask[Head[i]];

      if(Tail[i] >= NTask)	/* it appears that the Intel C 9.1 on Itanium2 produces incorrect code if
				   this if-statemet is omitted. Apparently, the compiler then joins the two loops,
				   but this is here not permitted because storage for FOF_PList actually overlaps
				   (on purpose) with MinID/MinIDTask/Head */
	{
	  printf("oh no: ThisTask=%d i=%d Head[i]=%d  NumPart=%d MinIDTask[Head[i]]=%d\n",
		 ThisTask, i, (int) Head[i], NumPart, (int) MinIDTask[Head[i]]);
	  fflush(stdout);
	  endrun(8812);
	}
    }

  for(i = 0; i < NumPart; i++)
    {
      FOF_PList[i].MinID = Next[i];
      FOF_PList[i].MinIDTask = Tail[i];
      FOF_PList[i].Pindex = i;
    }

  force_treefree();

  myfree(Tail);
  myfree(Next);
  myfree(Len);

  FOF_GList = (struct fof_group_list *) mymalloc("FOF_GList", sizeof(struct fof_group_list) * NumPart);

  fof_compile_catalogue();

  t1 = second();
  if(ThisTask == 0)
    printf("compiling local group data and catalogue took = %g sec\n", timediff(t0, t1));


  MPI_Allreduce(&Ngroups, &TotNgroups, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  sumup_large_ints(1, &Nids, &TotNids);

  if(TotNgroups > 0)
    {
      int largestloc = 0;

      for(i = 0; i < NgroupsExt; i++)
	if(FOF_GList[i].LocCount + FOF_GList[i].ExtCount > largestloc)
	  largestloc = FOF_GList[i].LocCount + FOF_GList[i].ExtCount;
      MPI_Allreduce(&largestloc, &largestgroup, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    }
  else
    largestgroup = 0;

  if(ThisTask == 0)
    {
      printf("\nTotal number of groups with at least %d particles: %d\n", FOF_GROUP_MIN_LEN, TotNgroups);
      if(TotNgroups > 0)
	{
	  printf("Largest group has %d particles.\n", largestgroup);
	  printf("Total number of particles in groups: %d%09d\n\n",
		 (int) (TotNids / 1000000000), (int) (TotNids % 1000000000));
	}
    }

  t0 = second();

  Group =
    (struct group_properties *) mymalloc("Group", sizeof(struct group_properties) *
					 IMAX(NgroupsExt, TotNgroups / NTask + 1));

  if(ThisTask == 0)
    {
      printf("group properties are now allocated.. (presently allocated=%g MB)\n",
	     AllocatedBytes / (1024.0 * 1024.0));
      fflush(stdout);
    }

  for(i = 0, start = 0; i < NgroupsExt; i++)
    {
      while(FOF_PList[start].MinID < FOF_GList[i].MinID)
	{
	  start++;
	  if(start > NumPart)
	    endrun(78);
	}

      if(FOF_PList[start].MinID != FOF_GList[i].MinID)
	endrun(123);

      for(lenloc = 0; start + lenloc < NumPart;)
	if(FOF_PList[start + lenloc].MinID == FOF_GList[i].MinID)
	  lenloc++;
	else
	  break;

      Group[i].MinID = FOF_GList[i].MinID;
      Group[i].MinIDTask = FOF_GList[i].MinIDTask;

      fof_compute_group_properties(i, start, lenloc);

      start += lenloc;
    }

  fof_exchange_group_data();

  fof_finish_group_properties();

  t1 = second();
  if(ThisTask == 0)
    printf("computation of group properties took = %g sec\n", timediff(t0, t1));

#ifdef BLACK_HOLES
  if(num < 0)
    fof_make_black_holes();
#endif

#ifdef BUBBLES
  if(num < 0)
    find_CM_of_biggest_group();
#endif

#ifdef MULTI_BUBBLES
  if(num < 0)
    multi_bubbles();
#endif

  CPU_Step[CPU_FOF] += measure_time();

  if(num >= 0)
    {
      fof_save_groups(num);
#ifdef SUBFIND
      subfind(num);
#endif
    }

  myfree(Group);

  myfree(FOF_GList);
  myfree(FOF_PList);

  if(ThisTask == 0)
    {
      printf("Finished computing FoF groups.  (presently allocated=%g MB)\n\n",
	     AllocatedBytes / (1024.0 * 1024.0));
      fflush(stdout);
    }


  CPU_Step[CPU_FOF] += measure_time();

#ifdef SUBFIND
  domain_Decomposition();
#endif

  force_treeallocate((int) (All.TreeAllocFactor * All.MaxPart) + NTopnodes, All.MaxPart);

  if(ThisTask == 0)
    printf("Tree construction.\n");
  force_treebuild(NumPart, NULL);

  TreeReconstructFlag = 0;
}



void fof_find_groups(void)
{
  int i, j, ndone_flag, link_count, dummy, nprocessed;
  int ndone, ngrp, sendTask, recvTask, place, nexport, nimport, link_across;
  int npart, marked;
  int64_t totmarked, totnpart;
  int64_t link_across_tot, ntot;
  MyIDType *MinIDOld;
  char *FoFDataOut, *FoFDataResult, *MarkedFlag, *ChangedFlag;
  double t0, t1;

  if(ThisTask == 0)
    {
      printf("\nStart linking particles (presently allocated=%g MB)\n", AllocatedBytes / (1024.0 * 1024.0));
      fflush(stdout);
    }


  /* allocate buffers to arrange communication */

  Ngblist = (int *) mymalloc("Ngblist", NumPart * sizeof(int));

  All.BunchSize =
    (int) ((All.BufferSize * 1024 * 1024) / (sizeof(struct data_index) + sizeof(struct data_nodelist) +
					     2 * sizeof(struct fofdata_in)));
  DataIndexTable =
    (struct data_index *) mymalloc("DataIndexTable", All.BunchSize * sizeof(struct data_index));
  DataNodeList =
    (struct data_nodelist *) mymalloc("DataNodeList", All.BunchSize * sizeof(struct data_nodelist));

  NonlocalFlag = (char *) mymalloc("NonlocalFlag", NumPart * sizeof(char));
  MarkedFlag = (char *) mymalloc("MarkedFlag", NumPart * sizeof(char));
  ChangedFlag = (char *) mymalloc("ChangedFlag", NumPart * sizeof(char));
  MinIDOld = (MyIDType *) mymalloc("MinIDOld", NumPart * sizeof(MyIDType));

  t0 = second();

  /* first, link only among local particles */
  for(i = 0, marked = 0, npart = 0; i < NumPart; i++)
    {
      if(((1 << P[i].Type) & (FOF_PRIMARY_LINK_TYPES)))
	{
	  fof_find_dmparticles_evaluate(i, -1, &dummy, &dummy);

	  npart++;

	  if(NonlocalFlag[i])
	    marked++;
	}
    }


  sumup_large_ints(1, &marked, &totmarked);
  sumup_large_ints(1, &npart, &totnpart);

  t1 = second();


  if(ThisTask == 0)
    {
      printf
	("links on local processor done (took %g sec).\nMarked=%d%09d out of the %d%09d primaries which are linked\n",
	 timediff(t0, t1),
	 (int) (totmarked / 1000000000), (int) (totmarked % 1000000000),
	 (int) (totnpart / 1000000000), (int) (totnpart % 1000000000));

      printf("\nlinking across processors (presently allocated=%g MB) \n",
	     AllocatedBytes / (1024.0 * 1024.0));
      fflush(stdout);
    }

  for(i = 0; i < NumPart; i++)
    {
      MinIDOld[i] = MinID[Head[i]];
      MarkedFlag[i] = 1;
    }

  do
    {
      t0 = second();

      for(i = 0; i < NumPart; i++)
	{
	  ChangedFlag[i] = MarkedFlag[i];
	  MarkedFlag[i] = 0;
	}

      i = 0;			/* begin with this index */
      link_across = 0;
      nprocessed = 0;

      do
	{
	  for(j = 0; j < NTask; j++)
	    {
	      Send_count[j] = 0;
	      Exportflag[j] = -1;
	    }

	  /* do local particles and prepare export list */
	  for(nexport = 0; i < NumPart; i++)
	    {
	      if(((1 << P[i].Type) & (FOF_PRIMARY_LINK_TYPES)))
		{
		  if(NonlocalFlag[i] && ChangedFlag[i])
		    {
		      if(fof_find_dmparticles_evaluate(i, 0, &nexport, Send_count) < 0)
			break;

		      nprocessed++;
		    }
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

	  FoFDataGet = (struct fofdata_in *) mymalloc("FoFDataGet", nimport * sizeof(struct fofdata_in));
	  FoFDataIn = (struct fofdata_in *) mymalloc("FoFDataIn", nexport * sizeof(struct fofdata_in));


	  /* prepare particle data for export */
	  for(j = 0; j < nexport; j++)
	    {
	      place = DataIndexTable[j].Index;

	      FoFDataIn[j].Pos[0] = P[place].Pos[0];
	      FoFDataIn[j].Pos[1] = P[place].Pos[1];
	      FoFDataIn[j].Pos[2] = P[place].Pos[2];
	      FoFDataIn[j].MinID = MinID[Head[place]];
	      FoFDataIn[j].MinIDTask = MinIDTask[Head[place]];

	      memcpy(FoFDataIn[j].NodeList,
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
		      MPI_Sendrecv(&FoFDataIn[Send_offset[recvTask]],
				   Send_count[recvTask] * sizeof(struct fofdata_in), MPI_BYTE,
				   recvTask, TAG_DENS_A,
				   &FoFDataGet[Recv_offset[recvTask]],
				   Recv_count[recvTask] * sizeof(struct fofdata_in), MPI_BYTE,
				   recvTask, TAG_DENS_A, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		    }
		}
	    }

	  myfree(FoFDataIn);
	  FoFDataResult = (char *) mymalloc("FoFDataResult", nimport * sizeof(char));
	  FoFDataOut = (char *) mymalloc("FoFDataOut", nexport * sizeof(char));

	  /* now do the particles that were sent to us */

	  for(j = 0; j < nimport; j++)
	    {
	      link_count = fof_find_dmparticles_evaluate(j, 1, &dummy, &dummy);
	      link_across += link_count;
	      if(link_count)
		FoFDataResult[j] = 1;
	      else
		FoFDataResult[j] = 0;
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
		      /* get the particles */
		      MPI_Sendrecv(&FoFDataResult[Recv_offset[recvTask]],
				   Recv_count[recvTask] * sizeof(char),
				   MPI_BYTE, recvTask, TAG_DENS_B,
				   &FoFDataOut[Send_offset[recvTask]],
				   Send_count[recvTask] * sizeof(char),
				   MPI_BYTE, recvTask, TAG_DENS_B, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		    }
		}
	    }

	  /* need to mark the particle if it induced a link */
	  for(j = 0; j < nexport; j++)
	    {
	      place = DataIndexTable[j].Index;
	      if(FoFDataOut[j])
		MarkedFlag[place] = 1;
	    }

	  myfree(FoFDataOut);
	  myfree(FoFDataResult);
	  myfree(FoFDataGet);

	  if(i >= NumPart)
	    ndone_flag = 1;
	  else
	    ndone_flag = 0;

	  MPI_Allreduce(&ndone_flag, &ndone, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
	}
      while(ndone < NTask);


      sumup_large_ints(1, &link_across, &link_across_tot);
      sumup_large_ints(1, &nprocessed, &ntot);

      t1 = second();

      if(ThisTask == 0)
	{
	  printf("have done %d%09d cross links (processed %d%09d, took %g sec)\n",
		 (int) (link_across_tot / 1000000000), (int) (link_across_tot % 1000000000),
		 (int) (ntot / 1000000000), (int) (ntot % 1000000000), timediff(t0, t1));
	  fflush(stdout);
	}


      /* let's check out which particles have changed their MinID */
      for(i = 0; i < NumPart; i++)
	if(NonlocalFlag[i])
	  {
	    if(MinID[Head[i]] != MinIDOld[i])
	      MarkedFlag[i] = 1;

	    MinIDOld[i] = MinID[Head[i]];
	  }

    }
  while(link_across_tot > 0);

  myfree(MinIDOld);
  myfree(ChangedFlag);
  myfree(MarkedFlag);
  myfree(NonlocalFlag);

  myfree(DataNodeList);
  myfree(DataIndexTable);
  myfree(Ngblist);

  if(ThisTask == 0)
    {
      printf("Local groups found.\n\n");
      fflush(stdout);
    }
}


int fof_find_dmparticles_evaluate(int target, int mode, int *nexport, int *nsend_local)
{
  int j, n, links, p, s, ss, listindex = 0;
  int startnode, numngb_inbox;
  MyDouble *pos;

  links = 0;

  if(mode == 0 || mode == -1)
    pos = P[target].Pos;
  else
    pos = FoFDataGet[target].Pos;

  if(mode == 0 || mode == -1)
    {
      startnode = All.MaxPart;	/* root node */
    }
  else
    {
      startnode = FoFDataGet[target].NodeList[0];
      startnode = Nodes[startnode].u.d.nextnode;	/* open it */
    }

  while(startnode >= 0)
    {
      while(startnode >= 0)
	{
	  if(mode == -1)
	    *nexport = 0;

	  numngb_inbox = ngb_treefind_fof_primary(pos, LinkL, target, &startnode, mode, nexport, nsend_local);

	  if(numngb_inbox < 0)
	    return -1;

	  if(mode == -1)
	    {
	      if(*nexport == 0)
		NonlocalFlag[target] = 0;
	      else
		NonlocalFlag[target] = 1;
	    }

	  for(n = 0; n < numngb_inbox; n++)
	    {
	      j = Ngblist[n];

	      if(mode == 0 || mode == -1)
		{
		  if(Head[target] != Head[j])	/* only if not yet linked */
		    {

		      if(mode == 0)
			endrun(87654);

		      if(Len[Head[target]] > Len[Head[j]])	/* p group is longer */
			{
			  p = target;
			  s = j;
			}
		      else
			{
			  p = j;
			  s = target;
			}
		      Next[Tail[Head[p]]] = Head[s];

		      Tail[Head[p]] = Tail[Head[s]];

		      Len[Head[p]] += Len[Head[s]];

		      ss = Head[s];
		      do
			Head[ss] = Head[p];
		      while((ss = Next[ss]) >= 0);

		      if(MinID[Head[s]] < MinID[Head[p]])
			{
			  MinID[Head[p]] = MinID[Head[s]];
			  MinIDTask[Head[p]] = MinIDTask[Head[s]];
			}
		    }
		}
	      else		/* mode is 1 */
		{
		  if(MinID[Head[j]] > FoFDataGet[target].MinID)
		    {
		      MinID[Head[j]] = FoFDataGet[target].MinID;
		      MinIDTask[Head[j]] = FoFDataGet[target].MinIDTask;
		      links++;
		    }
		}
	    }
	}

      if(mode == 1)
	{
	  listindex++;
	  if(listindex < NODELISTLENGTH)
	    {
	      startnode = FoFDataGet[target].NodeList[listindex];
	      if(startnode >= 0)
		startnode = Nodes[startnode].u.d.nextnode;	/* open it */
	    }
	}
    }

  return links;
}



void fof_compile_catalogue(void)
{
  int i, j, start, nimport, ngrp, sendTask, recvTask;
  struct fof_group_list *get_FOF_GList;

  /* sort according to MinID */
  qsort(FOF_PList, NumPart, sizeof(struct fof_particle_list), fof_compare_FOF_PList_MinID);

  for(i = 0; i < NumPart; i++)
    {
      FOF_GList[i].MinID = FOF_PList[i].MinID;
      FOF_GList[i].MinIDTask = FOF_PList[i].MinIDTask;
      if(FOF_GList[i].MinIDTask == ThisTask)
	{
	  FOF_GList[i].LocCount = 1;
	  FOF_GList[i].ExtCount = 0;
#ifdef DENSITY_SPLIT_BY_TYPE
	  if(((1 << P[FOF_PList[i].Pindex].Type) & (FOF_PRIMARY_LINK_TYPES)))
	    FOF_GList[i].LocDMCount = 1;
	  else
	    FOF_GList[i].LocDMCount = 0;
	  FOF_GList[i].ExtDMCount = 0;
#endif
	}
      else
	{
	  FOF_GList[i].LocCount = 0;
	  FOF_GList[i].ExtCount = 1;
#ifdef DENSITY_SPLIT_BY_TYPE
	  FOF_GList[i].LocDMCount = 0;
	  if(((1 << P[FOF_PList[i].Pindex].Type) & (FOF_PRIMARY_LINK_TYPES)))
	    FOF_GList[i].ExtDMCount = 1;
	  else
	    FOF_GList[i].ExtDMCount = 0;
#endif
	}
    }

  /* eliminate duplicates in FOF_GList with respect to MinID */

  if(NumPart)
    NgroupsExt = 1;
  else
    NgroupsExt = 0;

  for(i = 1, start = 0; i < NumPart; i++)
    {
      if(FOF_GList[i].MinID == FOF_GList[start].MinID)
	{
	  FOF_GList[start].LocCount += FOF_GList[i].LocCount;
	  FOF_GList[start].ExtCount += FOF_GList[i].ExtCount;
#ifdef DENSITY_SPLIT_BY_TYPE
	  FOF_GList[start].LocDMCount += FOF_GList[i].LocDMCount;
	  FOF_GList[start].ExtDMCount += FOF_GList[i].ExtDMCount;
#endif
	}
      else
	{
	  start = NgroupsExt;
	  FOF_GList[start] = FOF_GList[i];
	  NgroupsExt++;
	}
    }


  /* sort the remaining ones according to task */
  qsort(FOF_GList, NgroupsExt, sizeof(struct fof_group_list), fof_compare_FOF_GList_MinIDTask);

  /* count how many we have of each task */
  for(i = 0; i < NTask; i++)
    Send_count[i] = 0;
  for(i = 0; i < NgroupsExt; i++)
    Send_count[FOF_GList[i].MinIDTask]++;

  MPI_Alltoall(Send_count, 1, MPI_INT, Recv_count, 1, MPI_INT, MPI_COMM_WORLD);

  for(j = 0, nimport = 0, Recv_offset[0] = 0, Send_offset[0] = 0; j < NTask; j++)
    {
      if(j == ThisTask)		/* we will not exchange the ones that are local */
	Recv_count[j] = 0;
      nimport += Recv_count[j];

      if(j > 0)
	{
	  Send_offset[j] = Send_offset[j - 1] + Send_count[j - 1];
	  Recv_offset[j] = Recv_offset[j - 1] + Recv_count[j - 1];
	}
    }

  get_FOF_GList =
    (struct fof_group_list *) mymalloc("get_FOF_GList", nimport * sizeof(struct fof_group_list));

  for(ngrp = 1; ngrp < (1 << PTask); ngrp++)
    {
      sendTask = ThisTask;
      recvTask = ThisTask ^ ngrp;

      if(recvTask < NTask)
	{
	  if(Send_count[recvTask] > 0 || Recv_count[recvTask] > 0)
	    {
	      /* get the group info */
	      MPI_Sendrecv(&FOF_GList[Send_offset[recvTask]],
			   Send_count[recvTask] * sizeof(struct fof_group_list), MPI_BYTE,
			   recvTask, TAG_DENS_A,
			   &get_FOF_GList[Recv_offset[recvTask]],
			   Recv_count[recvTask] * sizeof(struct fof_group_list), MPI_BYTE,
			   recvTask, TAG_DENS_A, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	    }
	}
    }

  for(i = 0; i < nimport; i++)
    get_FOF_GList[i].MinIDTask = i;


  /* sort the groups according to MinID */
  qsort(FOF_GList, NgroupsExt, sizeof(struct fof_group_list), fof_compare_FOF_GList_MinID);
  qsort(get_FOF_GList, nimport, sizeof(struct fof_group_list), fof_compare_FOF_GList_MinID);

  /* merge the imported ones with the local ones */
  for(i = 0, start = 0; i < nimport; i++)
    {
      while(FOF_GList[start].MinID < get_FOF_GList[i].MinID)
	{
	  start++;
	  if(start >= NgroupsExt)
	    endrun(7973);
	}

      if(get_FOF_GList[i].LocCount != 0)
	endrun(123);

      if(FOF_GList[start].MinIDTask != ThisTask)
	endrun(124);

      FOF_GList[start].ExtCount += get_FOF_GList[i].ExtCount;
#ifdef DENSITY_SPLIT_BY_TYPE
      FOF_GList[start].ExtDMCount += get_FOF_GList[i].ExtDMCount;
#endif
    }

  /* copy the size information back into the list, to inform the others */
  for(i = 0, start = 0; i < nimport; i++)
    {
      while(FOF_GList[start].MinID < get_FOF_GList[i].MinID)
	{
	  start++;
	  if(start >= NgroupsExt)
	    endrun(797831);
	}

      get_FOF_GList[i].ExtCount = FOF_GList[start].ExtCount;
      get_FOF_GList[i].LocCount = FOF_GList[start].LocCount;
#ifdef DENSITY_SPLIT_BY_TYPE
      get_FOF_GList[i].ExtDMCount = FOF_GList[start].ExtDMCount;
      get_FOF_GList[i].LocDMCount = FOF_GList[start].LocDMCount;
#endif
    }

  /* sort the imported/exported list according to MinIDTask */
  qsort(get_FOF_GList, nimport, sizeof(struct fof_group_list), fof_compare_FOF_GList_MinIDTask);
  qsort(FOF_GList, NgroupsExt, sizeof(struct fof_group_list), fof_compare_FOF_GList_MinIDTask);


  for(i = 0; i < nimport; i++)
    get_FOF_GList[i].MinIDTask = ThisTask;

  for(ngrp = 1; ngrp < (1 << PTask); ngrp++)
    {
      sendTask = ThisTask;
      recvTask = ThisTask ^ ngrp;

      if(recvTask < NTask)
	{
	  if(Send_count[recvTask] > 0 || Recv_count[recvTask] > 0)
	    {
	      /* get the group info */
	      MPI_Sendrecv(&get_FOF_GList[Recv_offset[recvTask]],
			   Recv_count[recvTask] * sizeof(struct fof_group_list), MPI_BYTE,
			   recvTask, TAG_DENS_A,
			   &FOF_GList[Send_offset[recvTask]],
			   Send_count[recvTask] * sizeof(struct fof_group_list), MPI_BYTE,
			   recvTask, TAG_DENS_A, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	    }
	}
    }

  myfree(get_FOF_GList);

  /* eliminate all groups that are too small, and count local groups */
  for(i = 0, Ngroups = 0, Nids = 0; i < NgroupsExt; i++)
    {
#ifdef DENSITY_SPLIT_BY_TYPE
      if(FOF_GList[i].LocDMCount + FOF_GList[i].ExtDMCount < FOF_GROUP_MIN_LEN)
#else
      if(FOF_GList[i].LocCount + FOF_GList[i].ExtCount < FOF_GROUP_MIN_LEN)
#endif
	{
	  FOF_GList[i] = FOF_GList[NgroupsExt - 1];
	  NgroupsExt--;
	  i--;
	}
      else
	{
	  if(FOF_GList[i].MinIDTask == ThisTask)
	    {
	      Ngroups++;
	      Nids += FOF_GList[i].LocCount + FOF_GList[i].ExtCount;
	    }
	}
    }

  /* sort the group list according to MinID */
  qsort(FOF_GList, NgroupsExt, sizeof(struct fof_group_list), fof_compare_FOF_GList_MinID);
}



void fof_compute_group_properties(int gr, int start, int len)
{
  int j, k, index;
  double xyz[3];

  Group[gr].Len = 0;
  Group[gr].Mass = 0;
#ifdef SFR
  Group[gr].Sfr = 0;
#endif
#ifdef BLACK_HOLES
  Group[gr].BH_Mass = 0;
  Group[gr].BH_Mdot = 0;
  Group[gr].index_maxdens = Group[gr].task_maxdens = -1;
  Group[gr].MaxDens = 0;
#endif

  for(k = 0; k < 3; k++)
    {
      Group[gr].CM[k] = 0;
      Group[gr].Vel[k] = 0;
      Group[gr].FirstPos[k] = P[FOF_PList[start].Pindex].Pos[k];
    }

  for(k = 0; k < 6; k++)
    {
      Group[gr].LenType[k] = 0;
      Group[gr].MassType[k] = 0;
    }

  for(k = 0; k < len; k++)
    {
      index = FOF_PList[start + k].Pindex;

      Group[gr].Len++;
      Group[gr].Mass += P[index].Mass;
      Group[gr].LenType[P[index].Type]++;
      Group[gr].MassType[P[index].Type] += P[index].Mass;


#ifdef SFR
      if(P[index].Type == 0)
	Group[gr].Sfr += SPHP(index).Sfr;
#endif
#ifdef BLACK_HOLES
      if(P[index].Type == 5)
	{
	  Group[gr].BH_Mdot += BHP(index).Mdot;
	  Group[gr].BH_Mass += BHP(index).Mass;
	}
      if(P[index].Type == 0)
	{
#ifdef WINDS
        /* make bh in non wind gas on bh wind*/
	  if(SPHP(index).DelayTime <= 0)
#endif
	    if(SPHP(index).d.Density > Group[gr].MaxDens)
	      {
		Group[gr].MaxDens = SPHP(index).d.Density;
		Group[gr].index_maxdens = index;
		Group[gr].task_maxdens = ThisTask;
	      }
	}
#endif

      for(j = 0; j < 3; j++)
	{
	  xyz[j] = P[index].Pos[j];
#ifdef PERIODIC
	  xyz[j] = fof_periodic(xyz[j] - Group[gr].FirstPos[j]);
#endif
	  Group[gr].CM[j] += P[index].Mass * xyz[j];
	  Group[gr].Vel[j] += P[index].Mass * P[index].Vel[j];
	}
    }
}


void fof_exchange_group_data(void)
{
  struct group_properties *get_Group;
  int i, j, ngrp, sendTask, recvTask, nimport, start;
  double xyz[3];

  /* sort the groups according to task */
  qsort(Group, NgroupsExt, sizeof(struct group_properties), fof_compare_Group_MinIDTask);

  /* count how many we have of each task */
  for(i = 0; i < NTask; i++)
    Send_count[i] = 0;
  for(i = 0; i < NgroupsExt; i++)
    Send_count[FOF_GList[i].MinIDTask]++;

  MPI_Alltoall(Send_count, 1, MPI_INT, Recv_count, 1, MPI_INT, MPI_COMM_WORLD);

  for(j = 0, nimport = 0, Recv_offset[0] = 0, Send_offset[0] = 0; j < NTask; j++)
    {
      if(j == ThisTask)		/* we will not exchange the ones that are local */
	Recv_count[j] = 0;
      nimport += Recv_count[j];

      if(j > 0)
	{
	  Send_offset[j] = Send_offset[j - 1] + Send_count[j - 1];
	  Recv_offset[j] = Recv_offset[j - 1] + Recv_count[j - 1];
	}
    }

  get_Group = (struct group_properties *) mymalloc("get_Group", sizeof(struct group_properties) * nimport);

  for(ngrp = 1; ngrp < (1 << PTask); ngrp++)
    {
      sendTask = ThisTask;
      recvTask = ThisTask ^ ngrp;

      if(recvTask < NTask)
	{
	  if(Send_count[recvTask] > 0 || Recv_count[recvTask] > 0)
	    {
	      /* get the group data */
	      MPI_Sendrecv(&Group[Send_offset[recvTask]],
			   Send_count[recvTask] * sizeof(struct group_properties), MPI_BYTE,
			   recvTask, TAG_DENS_A,
			   &get_Group[Recv_offset[recvTask]],
			   Recv_count[recvTask] * sizeof(struct group_properties), MPI_BYTE,
			   recvTask, TAG_DENS_A, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	    }
	}
    }

  /* sort the groups again according to MinID */
  qsort(Group, NgroupsExt, sizeof(struct group_properties), fof_compare_Group_MinID);
  qsort(get_Group, nimport, sizeof(struct group_properties), fof_compare_Group_MinID);

  /* now add in the partial imported group data to the main ones */
  for(i = 0, start = 0; i < nimport; i++)
    {
      while(Group[start].MinID < get_Group[i].MinID)
	{
	  start++;
	  if(start >= NgroupsExt)
	    endrun(797890);
	}

      Group[start].Len += get_Group[i].Len;
      Group[start].Mass += get_Group[i].Mass;

      for(j = 0; j < 6; j++)
	{
	  Group[start].LenType[j] += get_Group[i].LenType[j];
	  Group[start].MassType[j] += get_Group[i].MassType[j];
	}

#ifdef SFR
      Group[start].Sfr += get_Group[i].Sfr;
#endif
#ifdef BLACK_HOLES
      Group[start].BH_Mdot += get_Group[i].BH_Mdot;
      Group[start].BH_Mass += get_Group[i].BH_Mass;
      if(get_Group[i].MaxDens > Group[start].MaxDens)
	{
	  Group[start].MaxDens = get_Group[i].MaxDens;
	  Group[start].index_maxdens = get_Group[i].index_maxdens;
	  Group[start].task_maxdens = get_Group[i].task_maxdens;
	}
#endif

      for(j = 0; j < 3; j++)
	{
	  xyz[j] = get_Group[i].CM[j] / get_Group[i].Mass + get_Group[i].FirstPos[j];
#ifdef PERIODIC
	  xyz[j] = fof_periodic(xyz[j] - Group[start].FirstPos[j]);
#endif
	  Group[start].CM[j] += get_Group[i].Mass * xyz[j];
	  Group[start].Vel[j] += get_Group[i].Vel[j];
	}
    }

  myfree(get_Group);
}

void fof_finish_group_properties(void)
{
  double cm[3];
  int i, j, ngr;

  for(i = 0; i < NgroupsExt; i++)
    {
      if(Group[i].MinIDTask == ThisTask)
	{
	  for(j = 0; j < 3; j++)
	    {
	      Group[i].Vel[j] /= Group[i].Mass;

	      cm[j] = Group[i].CM[j] / Group[i].Mass;
#ifdef PERIODIC
	      cm[j] = fof_periodic_wrap(cm[j] + Group[i].FirstPos[j]);
#endif
	      Group[i].CM[j] = cm[j];
	    }
	}
    }

  /* eliminate the non-local groups */
  for(i = 0, ngr = NgroupsExt; i < ngr; i++)
    {
      if(Group[i].MinIDTask != ThisTask)
	{
	  Group[i] = Group[ngr - 1];
	  i--;
	  ngr--;
	}
    }

  if(ngr != Ngroups)
    endrun(876889);

  qsort(Group, Ngroups, sizeof(struct group_properties), fof_compare_Group_MinID);
}



void fof_save_groups(int num)
{
  int i, j, start, lenloc, nprocgroup, masterTask, groupTask, ngr, totlen;
  int64_t totNids;
  char buf[500];
  double t0, t1;

  if(ThisTask == 0)
    {
      printf("start global sorting of group catalogues\n");
      fflush(stdout);
    }

  t0 = second();

  /* assign group numbers (at this point, both Group and FOF_GList are sorted by MinID) */
  for(i = 0; i < NgroupsExt; i++)
    {
      FOF_GList[i].LocCount += FOF_GList[i].ExtCount;	/* total length */
      FOF_GList[i].ExtCount = ThisTask;	/* original task */
#ifdef DENSITY_SPLIT_BY_TYPE
      FOF_GList[i].LocDMCount += FOF_GList[i].ExtDMCount;	/* total length */
      FOF_GList[i].ExtDMCount = ThisTask;	/* not longer needed/used (hopefully) */
#endif
    }

  parallel_sort(FOF_GList, NgroupsExt, sizeof(struct fof_group_list),
		fof_compare_FOF_GList_LocCountTaskDiffMinID);

  for(i = 0, ngr = 0; i < NgroupsExt; i++)
    {
      if(FOF_GList[i].ExtCount == FOF_GList[i].MinIDTask)
	ngr++;

      FOF_GList[i].GrNr = ngr;
    }

  MPI_Allgather(&ngr, 1, MPI_INT, Send_count, 1, MPI_INT, MPI_COMM_WORLD);
  for(j = 1, Send_offset[0] = 0; j < NTask; j++)
    Send_offset[j] = Send_offset[j - 1] + Send_count[j - 1];

  for(i = 0; i < NgroupsExt; i++)
    FOF_GList[i].GrNr += Send_offset[ThisTask];


  MPI_Allreduce(&ngr, &i, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  if(i != TotNgroups)
    {
      printf("i=%d\n", i);
      endrun(123123);
    }

  /* bring the group list back into the original order */
  parallel_sort(FOF_GList, NgroupsExt, sizeof(struct fof_group_list), fof_compare_FOF_GList_ExtCountMinID);

  /* Assign the group numbers to the group properties array */
  for(i = 0, start = 0; i < Ngroups; i++)
    {
      while(FOF_GList[start].MinID < Group[i].MinID)
	{
	  start++;
	  if(start >= NgroupsExt)
	    endrun(7297890);
	}
      Group[i].GrNr = FOF_GList[start].GrNr;
    }

  /* sort the groups according to group-number */
  parallel_sort(Group, Ngroups, sizeof(struct group_properties), fof_compare_Group_GrNr);

  /* fill in the offset-values */
  for(i = 0, totlen = 0; i < Ngroups; i++)
    {
      if(i > 0)
	Group[i].Offset = Group[i - 1].Offset + Group[i - 1].Len;
      else
	Group[i].Offset = 0;
      totlen += Group[i].Len;
    }

  MPI_Allgather(&totlen, 1, MPI_INT, Send_count, 1, MPI_INT, MPI_COMM_WORLD);
  unsigned int *uoffset = mymalloc("uoffset", NTask * sizeof(unsigned int));

  for(j = 1, uoffset[0] = 0; j < NTask; j++)
    uoffset[j] = uoffset[j - 1] + Send_count[j - 1];

  for(i = 0; i < Ngroups; i++)
    Group[i].Offset += uoffset[ThisTask];

  myfree(uoffset);

  /* prepare list of ids with assigned group numbers */

  ID_list = mymalloc("ID_list", sizeof(struct id_list) * NumPart);

#ifdef SUBFIND
  for(i = 0; i < NumPart; i++)
    P[i].GrNr = TotNgroups + 1;	/* will mark particles that are not in any group */
#endif

  for(i = 0, start = 0, Nids = 0; i < NgroupsExt; i++)
    {
      while(FOF_PList[start].MinID < FOF_GList[i].MinID)
	{
	  start++;
	  if(start > NumPart)
	    endrun(78);
	}

      if(FOF_PList[start].MinID != FOF_GList[i].MinID)
	endrun(1313);

      for(lenloc = 0; start + lenloc < NumPart;)
	if(FOF_PList[start + lenloc].MinID == FOF_GList[i].MinID)
	  {
	    ID_list[Nids].GrNr = FOF_GList[i].GrNr;
	    ID_list[Nids].ID = P[FOF_PList[start + lenloc].Pindex].ID;
#ifdef SUBFIND
	    P[FOF_PList[start + lenloc].Pindex].GrNr = FOF_GList[i].GrNr;
#endif
	    Nids++;
	    lenloc++;
	  }
	else
	  break;

      start += lenloc;
    }

  sumup_large_ints(1, &Nids, &totNids);

  MPI_Allgather(&Nids, 1, MPI_INT, Send_count, 1, MPI_INT, MPI_COMM_WORLD);
  for(j = 1, Send_offset[0] = 0; j < NTask; j++)
    Send_offset[j] = Send_offset[j - 1] + Send_count[j - 1];


  if(totNids != TotNids)
    {
      printf("Task=%d Nids=%d totNids=%d TotNids=%d\n", ThisTask, Nids, (int) totNids, (int) TotNids);
      endrun(12);
    }

  /* sort the particle IDs according to group-number */

  parallel_sort(ID_list, Nids, sizeof(struct id_list), fof_compare_ID_list_GrNrID);

  t1 = second();
  if(ThisTask == 0)
    {
      printf("Group catalogues globally sorted. took = %g sec\n", timediff(t0, t1));
      printf("starting saving of group catalogue\n");
      fflush(stdout);
    }

  t0 = second();

  if(ThisTask == 0)
    {
      sprintf(buf, "%s/groups_%03d", All.OutputDir, num);
      mkdir(buf, 02755);
    }
  MPI_Barrier(MPI_COMM_WORLD);


  if(NTask < All.NumFilesWrittenInParallel)
    {
      printf
	("Fatal error.\nNumber of processors must be a smaller or equal than `NumFilesWrittenInParallel'.\n");
      endrun(241931);
    }

  nprocgroup = NTask / All.NumFilesWrittenInParallel;
  if((NTask % All.NumFilesWrittenInParallel))
    nprocgroup++;
  masterTask = (ThisTask / nprocgroup) * nprocgroup;
  for(groupTask = 0; groupTask < nprocgroup; groupTask++)
    {
      if(ThisTask == (masterTask + groupTask))	/* ok, it's this processor's turn */
	fof_save_local_catalogue(num);
      MPI_Barrier(MPI_COMM_WORLD);	/* wait inside the group */
    }

  myfree(ID_list);

  t1 = second();

  if(ThisTask == 0)
    {
      printf("Group catalogues saved. took = %g sec\n", timediff(t0, t1));
      fflush(stdout);
    }
}



void fof_save_local_catalogue(int num)
{
  FILE *fd;
  float *mass, *cm, *vel;
  char fname[500];
  int i, j, *len;
  MyIDType *ids;

  sprintf(fname, "%s/groups_%03d/%s_%03d.%d", All.OutputDir, num, "group_tab", num, ThisTask);
  if(!(fd = fopen(fname, "w")))
    {
      printf("can't open file `%s`\n", fname);
      endrun(1183);
    }

  my_fwrite(&Ngroups, sizeof(int), 1, fd);
  my_fwrite(&TotNgroups, sizeof(int), 1, fd);
  my_fwrite(&Nids, sizeof(int), 1, fd);
  my_fwrite(&TotNids, sizeof(int64_t), 1, fd);
  my_fwrite(&NTask, sizeof(int), 1, fd);

  /* group len */
  len = mymalloc("len", Ngroups * sizeof(int));
  for(i = 0; i < Ngroups; i++)
    len[i] = Group[i].Len;
  my_fwrite(len, Ngroups, sizeof(int), fd);
  myfree(len);

  /* offset into id-list */
  len = mymalloc("len", Ngroups * sizeof(int));
  for(i = 0; i < Ngroups; i++)
    len[i] = Group[i].Offset;
  my_fwrite(len, Ngroups, sizeof(int), fd);
  myfree(len);

  /* mass */
  mass = mymalloc("mass", Ngroups * sizeof(float));
  for(i = 0; i < Ngroups; i++)
    mass[i] = Group[i].Mass;
  my_fwrite(mass, Ngroups, sizeof(float), fd);
  myfree(mass);

  /* CM */
  cm = mymalloc("cm", Ngroups * 3 * sizeof(float));
  for(i = 0; i < Ngroups; i++)
    for(j = 0; j < 3; j++)
      cm[i * 3 + j] = Group[i].CM[j];
  my_fwrite(cm, Ngroups, 3 * sizeof(float), fd);
  myfree(cm);

  /* vel */
  vel = mymalloc("vel", Ngroups * 3 * sizeof(float));
  for(i = 0; i < Ngroups; i++)
    for(j = 0; j < 3; j++)
      vel[i * 3 + j] = Group[i].Vel[j];
  my_fwrite(vel, Ngroups, 3 * sizeof(float), fd);
  myfree(vel);

  /* group len for each type */
  len = mymalloc("len", Ngroups * 6 * sizeof(int));
  for(i = 0; i < Ngroups; i++)
    for(j = 0; j < 6; j++)
      len[i * 6 + j] = Group[i].LenType[j];
  my_fwrite(len, Ngroups, 6 * sizeof(int), fd);
  myfree(len);

  /* group mass for each type */
  mass = mymalloc("mass", Ngroups * 6 * sizeof(float));
  for(i = 0; i < Ngroups; i++)
    for(j = 0; j < 6; j++)
      mass[i * 6 + j] = Group[i].MassType[j];
  my_fwrite(mass, Ngroups, 6 * sizeof(float), fd);
  myfree(mass);

#ifdef SFR
  /* sfr */
  mass = mymalloc("mass", Ngroups * sizeof(float));
  for(i = 0; i < Ngroups; i++)
    mass[i] = Group[i].Sfr;
  my_fwrite(mass, Ngroups, sizeof(float), fd);
  myfree(mass);
#endif

#ifdef BLACK_HOLES
  /* BH_Mass */
  mass = mymalloc("mass", Ngroups * sizeof(float));
  for(i = 0; i < Ngroups; i++)
    mass[i] = Group[i].BH_Mass;
  my_fwrite(mass, Ngroups, sizeof(float), fd);
  myfree(mass);

  /* BH_Mdot */
  mass = mymalloc("mass", Ngroups * sizeof(float));
  for(i = 0; i < Ngroups; i++)
    mass[i] = Group[i].BH_Mdot;
  my_fwrite(mass, Ngroups, sizeof(float), fd);
  myfree(mass);
#endif

  fclose(fd);


  ids = (MyIDType *) ID_list;
  for(i = 0; i < Nids; i++)
    ids[i] = ID_list[i].ID;

  sprintf(fname, "%s/groups_%03d/%s_%03d.%d", All.OutputDir, num, "group_ids", num, ThisTask);
  if(!(fd = fopen(fname, "w")))
    {
      printf("can't open file `%s`\n", fname);
      endrun(1184);
    }

  my_fwrite(&Ngroups, sizeof(int), 1, fd);
  my_fwrite(&TotNgroups, sizeof(int), 1, fd);
  my_fwrite(&Nids, sizeof(int), 1, fd);
  my_fwrite(&TotNids, sizeof(int64_t), 1, fd);
  my_fwrite(&NTask, sizeof(int), 1, fd);
  my_fwrite(&Send_offset[ThisTask], sizeof(int), 1, fd);	/* this is the number of IDs in previous files */
  my_fwrite(ids, sizeof(MyIDType), Nids, fd);
  fclose(fd);
}


void fof_find_nearest_dmparticle(void)
{
  int i, j, n, dummy;
  int64_t ntot;
  int ndone, ndone_flag, ngrp, sendTask, recvTask, place, nexport, nimport, npleft, iter;

  if(ThisTask == 0)
    {
      printf("Start finding nearest dm-particle (presently allocated=%g MB)\n",
	     AllocatedBytes / (1024.0 * 1024.0));
      fflush(stdout);
    }

  fof_nearest_distance = (float *) mymalloc("fof_nearest_distance", sizeof(float) * NumPart);
  fof_nearest_hsml = (float *) mymalloc("fof_nearest_hsml", sizeof(float) * NumPart);

  for(n = 0; n < NumPart; n++)
    {
      if(((1 << P[n].Type) & (FOF_SECONDARY_LINK_TYPES)))
	{
	  fof_nearest_distance[n] = 1.0e30;
	  fof_nearest_hsml[n] = 0.1 * LinkL;
	}
    }

  /* allocate buffers to arrange communication */

  Ngblist = (int *) mymalloc("Ngblist", NumPart * sizeof(int));

  All.BunchSize =
    (int) ((All.BufferSize * 1024 * 1024) / (sizeof(struct data_index) + sizeof(struct data_nodelist) +
					     sizeof(struct fofdata_in) + sizeof(struct fofdata_out) +
					     sizemax(sizeof(struct fofdata_in), sizeof(struct fofdata_out))));
  DataIndexTable =
    (struct data_index *) mymalloc("DataIndexTable", All.BunchSize * sizeof(struct data_index));
  DataNodeList =
    (struct data_nodelist *) mymalloc("DataNodeList", All.BunchSize * sizeof(struct data_nodelist));


  iter = 0;
  /* we will repeat the whole thing for those particles where we didn't find enough neighbours */
  if(ThisTask == 0)
    {
      printf("fof-nearest iteration started\n");
      fflush(stdout);
    }

  do
    {
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
	    if(((1 << P[i].Type) & (FOF_SECONDARY_LINK_TYPES)))
	      {
		if(fof_nearest_distance[i] > 1.0e29)
		  {
		    if(fof_find_nearest_dmparticle_evaluate(i, 0, &nexport, Send_count) < 0)
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

	  FoFDataGet = (struct fofdata_in *) mymalloc("FoFDataGet", nimport * sizeof(struct fofdata_in));
	  FoFDataIn = (struct fofdata_in *) mymalloc("FoFDataIn", nexport * sizeof(struct fofdata_in));

	  if(ThisTask == 0)
	    {
	      printf("still finding nearest... (presently allocated=%g MB)\n",
		     AllocatedBytes / (1024.0 * 1024.0));
	      fflush(stdout);
	    }

	  for(j = 0; j < nexport; j++)
	    {
	      place = DataIndexTable[j].Index;

	      FoFDataIn[j].Pos[0] = P[place].Pos[0];
	      FoFDataIn[j].Pos[1] = P[place].Pos[1];
	      FoFDataIn[j].Pos[2] = P[place].Pos[2];
	      FoFDataIn[j].Hsml = fof_nearest_hsml[place];

	      memcpy(FoFDataIn[j].NodeList,
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
		      MPI_Sendrecv(&FoFDataIn[Send_offset[recvTask]],
				   Send_count[recvTask] * sizeof(struct fofdata_in), MPI_BYTE,
				   recvTask, TAG_DENS_A,
				   &FoFDataGet[Recv_offset[recvTask]],
				   Recv_count[recvTask] * sizeof(struct fofdata_in), MPI_BYTE,
				   recvTask, TAG_DENS_A, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		    }
		}
	    }

	  myfree(FoFDataIn);
	  FoFDataResult =
	    (struct fofdata_out *) mymalloc("FoFDataResult", nimport * sizeof(struct fofdata_out));
	  FoFDataOut = (struct fofdata_out *) mymalloc("FoFDataOut", nexport * sizeof(struct fofdata_out));

	  for(j = 0; j < nimport; j++)
	    {
	      fof_find_nearest_dmparticle_evaluate(j, 1, &dummy, &dummy);
	    }

	  for(ngrp = 1; ngrp < (1 << PTask); ngrp++)
	    {
	      sendTask = ThisTask;
	      recvTask = ThisTask ^ ngrp;
	      if(recvTask < NTask)
		{
		  if(Send_count[recvTask] > 0 || Recv_count[recvTask] > 0)
		    {
		      /* send the results */
		      MPI_Sendrecv(&FoFDataResult[Recv_offset[recvTask]],
				   Recv_count[recvTask] * sizeof(struct fofdata_out),
				   MPI_BYTE, recvTask, TAG_DENS_B,
				   &FoFDataOut[Send_offset[recvTask]],
				   Send_count[recvTask] * sizeof(struct fofdata_out),
				   MPI_BYTE, recvTask, TAG_DENS_B, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		    }
		}

	    }

	  for(j = 0; j < nexport; j++)
	    {
	      place = DataIndexTable[j].Index;

	      if(FoFDataOut[j].Distance < fof_nearest_distance[place])
		{
		  fof_nearest_distance[place] = FoFDataOut[j].Distance;
		  MinID[place] = FoFDataOut[j].MinID;
		  MinIDTask[place] = FoFDataOut[j].MinIDTask;
		}
	    }

	  if(i >= NumPart)
	    ndone_flag = 1;
	  else
	    ndone_flag = 0;

	  MPI_Allreduce(&ndone_flag, &ndone, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

	  myfree(FoFDataOut);
	  myfree(FoFDataResult);
	  myfree(FoFDataGet);
	}
      while(ndone < NTask);

      /* do final operations on results */
      for(i = 0, npleft = 0; i < NumPart; i++)
	{
	  if(((1 << P[i].Type) & (FOF_SECONDARY_LINK_TYPES)))
	    {
	      if(fof_nearest_distance[i] > 1.0e29)
		{
                  if(fof_nearest_hsml[i] < 4 * LinkL)  /* we only search out to a maximum distance */
                    {
                      /* need to redo this particle */
                      npleft++;
                      fof_nearest_hsml[i] *= 2.0;
                      if(iter >= MAXITER - 10)
                        {
                          printf("i=%d task=%d ID=%llu Hsml=%g  pos=(%g|%g|%g)\n",
                                 i, ThisTask, P[i].ID, fof_nearest_hsml[i],
                                 P[i].Pos[0], P[i].Pos[1], P[i].Pos[2]);
                          fflush(stdout);
                        }
                    }
                  else
                    {
                      fof_nearest_distance[i] = 0;  /* we not continue to search for this particle */
                    }
		}
	    }
	}

      sumup_large_ints(1, &npleft, &ntot);
      if(ntot > 0)
	{
	  iter++;
	  if(iter > 0 && ThisTask == 0)
	    {
              printf("fof-nearest iteration %d: need to repeat for %010ld particles.\n", iter, ntot);
	      fflush(stdout);
	    }

	  if(iter > MAXITER)
	    {
	      printf("failed to converge in fof-nearest\n");
	      fflush(stdout);
	      endrun(1159);
	    }
	}
    }
  while(ntot > 0);

  myfree(DataNodeList);
  myfree(DataIndexTable);
  myfree(Ngblist);

  myfree(fof_nearest_hsml);
  myfree(fof_nearest_distance);

  if(ThisTask == 0)
    {
      printf("done finding nearest dm-particle\n");
      fflush(stdout);
    }
}


int fof_find_nearest_dmparticle_evaluate(int target, int mode, int *nexport, int *nsend_local)
{
  int j, n, index, listindex = 0;
  int startnode, numngb_inbox;
  double h, r2max;
  double dx, dy, dz, r2;
  MyDouble *pos;

  if(mode == 0)
    {
      pos = P[target].Pos;
      h = fof_nearest_hsml[target];
    }
  else
    {
      pos = FoFDataGet[target].Pos;
      h = FoFDataGet[target].Hsml;
    }

  index = -1;
  r2max = 1.0e30;

  if(mode == 0)
    {
      startnode = All.MaxPart;	/* root node */
    }
  else
    {
      startnode = FoFDataGet[target].NodeList[0];
      startnode = Nodes[startnode].u.d.nextnode;	/* open it */
    }

  while(startnode >= 0)
    {
      while(startnode >= 0)
	{
	  numngb_inbox = ngb_treefind_fof_nearest(pos, h, target, &startnode, mode, nexport, nsend_local);

	  if(numngb_inbox < 0)
	    return -1;

	  for(n = 0; n < numngb_inbox; n++)
	    {
	      j = Ngblist[n];
	      dx = pos[0] - P[j].Pos[0];
	      dy = pos[1] - P[j].Pos[1];
	      dz = pos[2] - P[j].Pos[2];
#ifdef PERIODIC			/*  now find the closest image in the given box size  */
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
	      if(r2 < r2max && r2 < h * h)
		{
		  index = j;
		  r2max = r2;
		}
	    }
	}

      if(mode == 1)
	{
	  listindex++;
	  if(listindex < NODELISTLENGTH)
	    {
	      startnode = FoFDataGet[target].NodeList[listindex];
	      if(startnode >= 0)
		startnode = Nodes[startnode].u.d.nextnode;	/* open it */
	    }
	}
    }


  if(mode == 0)
    {
      if(index >= 0)
	{
	  fof_nearest_distance[target] = sqrt(r2max);
	  MinID[target] = MinID[Head[index]];
	  MinIDTask[target] = MinIDTask[Head[index]];
	}
    }
  else
    {
      if(index >= 0)
	{
	  FoFDataResult[target].Distance = sqrt(r2max);
	  FoFDataResult[target].MinID = MinID[Head[index]];
	  FoFDataResult[target].MinIDTask = MinIDTask[Head[index]];
	}
      else
	FoFDataResult[target].Distance = 2.0e30;
    }
  return 0;
}




#ifdef BLACK_HOLES

void fof_make_black_holes(void)
{
  int i, j, n, ntot;
  int nexport, nimport, sendTask, recvTask, level;
  int *import_indices, *export_indices;
  double massDMpart;

  if(All.MassTable[1] > 0)
    massDMpart = All.MassTable[1];
  else {
    endrun(991234569); /* deprecate massDMpart in paramfile*/
  }

  for(n = 0; n < NTask; n++)
    Send_count[n] = 0;

  for(i = 0; i < Ngroups; i++)
    {
      if(Group[i].LenType[1] * massDMpart >=
	 (All.Omega0 - All.OmegaBaryon) / All.Omega0 * All.MinFoFMassForNewSeed)
	if(Group[i].LenType[5] == 0)
	  {
	    if(Group[i].index_maxdens >= 0)
	      Send_count[Group[i].task_maxdens]++;
	  }
    }

  MPI_Alltoall(Send_count, 1, MPI_INT, Recv_count, 1, MPI_INT, MPI_COMM_WORLD);

  for(j = 0, nimport = nexport = 0, Recv_offset[0] = 0, Send_offset[0] = 0; j < NTask; j++)
    {
      nexport += Send_count[j];
      nimport += Recv_count[j];

      if(j > 0)
	{
	  Send_offset[j] = Send_offset[j - 1] + Send_count[j - 1];
	  Recv_offset[j] = Recv_offset[j - 1] + Recv_count[j - 1];
	}
    }

  import_indices = mymalloc("import_indices", nimport * sizeof(int));
  export_indices = mymalloc("export_indices", nexport * sizeof(int));

  for(n = 0; n < NTask; n++)
    Send_count[n] = 0;

  for(i = 0; i < Ngroups; i++)
    {
      if(Group[i].LenType[1] * massDMpart >=
	 (All.Omega0 - All.OmegaBaryon) / All.Omega0 * All.MinFoFMassForNewSeed)
	if(Group[i].LenType[5] == 0)
	  {
	    if(Group[i].index_maxdens >= 0)
	      export_indices[Send_offset[Group[i].task_maxdens] +
			     Send_count[Group[i].task_maxdens]++] = Group[i].index_maxdens;
	  }
    }

  memcpy(&import_indices[Recv_offset[ThisTask]], &export_indices[Send_offset[ThisTask]],
	 Send_count[ThisTask] * sizeof(int));

  for(level = 1; level < (1 << PTask); level++)
    {
      sendTask = ThisTask;
      recvTask = ThisTask ^ level;

      if(recvTask < NTask)
	MPI_Sendrecv(&export_indices[Send_offset[recvTask]],
		     Send_count[recvTask] * sizeof(int),
		     MPI_BYTE, recvTask, TAG_FOF_E,
		     &import_indices[Recv_offset[recvTask]],
		     Recv_count[recvTask] * sizeof(int),
		     MPI_BYTE, recvTask, TAG_FOF_E, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

  MPI_Allreduce(&nimport, &ntot, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  if(ThisTask == 0)
    {
      printf("\nMaking %d new black hole particles\n\n", ntot);
      fflush(stdout);
    }

  All.TotN_bh += ntot;

  for(n = 0; n < nimport; n++)
    {
        blackhole_make_one(import_indices[n]);
    }

  All.TotN_sph -= ntot;
  myfree(export_indices);
  myfree(import_indices);
}



#endif


#ifdef BUBBLES

void find_CM_of_biggest_group(void)
{
  int i, rootcpu;
  struct group_properties BiggestGroup;

  parallel_sort(Group, Ngroups, sizeof(struct group_properties), fof_compare_Group_Len);

  /* the biggest group will now be the first group on the first cpu that has any groups */
  MPI_Allgather(&Ngroups, 1, MPI_INT, Send_count, 1, MPI_INT, MPI_COMM_WORLD);

  for(rootcpu = 0; Send_count[rootcpu] == 0 && rootcpu < NTask - 1; rootcpu++);

  if(rootcpu == ThisTask)
    BiggestGroup = Group[0];

  /* bring groups back into original order */
  parallel_sort(Group, Ngroups, sizeof(struct group_properties), fof_compare_Group_MinIDTask_MinID);

  MPI_Bcast(&BiggestGroup, sizeof(struct group_properties), MPI_BYTE, rootcpu, MPI_COMM_WORLD);

  All.BiggestGroupLen = BiggestGroup.Len;

  for(i = 0; i < 3; i++)
    All.BiggestGroupCM[i] = BiggestGroup.CM[i];

  All.BiggestGroupMass = BiggestGroup.Mass;

  if(ThisTask == 0)
    {
      printf("Biggest group length has %d particles.\n", All.BiggestGroupLen);
      printf("CM of biggest group is: (%g|%g|%g)\n", All.BiggestGroupCM[0], All.BiggestGroupCM[1],
	     All.BiggestGroupCM[2]);
      printf("Mass of biggest group is: %g\n", All.BiggestGroupMass);
    }
}

#endif



#ifdef MULTI_BUBBLES
void multi_bubbles(void)
{
  double phi, theta;
  double dx, dy, dz, rr, r2, dE;
  double E_bubble, totE_bubble, hubble_a = 0.0;
  double BubbleDistance, BubbleRadius, BubbleEnergy;
  MyFloat Mass_bubble, totMass_bubble;
  MyFloat pos[3];
  int numngb, tot_numngb, startnode, numngb_inbox;
  int n, i, j, k, l, dummy;
  int nheat, tot_nheat;
  int eff_nheat, tot_eff_nheat;
  double *GroupMassType_common, *GroupMassType_dum;
  float *GroupCM_common_x, *GroupCM_dum_x;
  float *GroupCM_common_y, *GroupCM_dum_y;
  float *GroupCM_common_z, *GroupCM_dum_z;
  int logical;
  int *nn_heat, *disp;
  double massDMpart;

  if(All.MassTable[1] > 0)
    massDMpart = All.MassTable[1];
  else
    massDMpart = All.massDMpart;


  if(All.ComovingIntegrationOn)
    {
      hubble_a = hubble_function(All.Time) / All.Hubble;
    }

  nheat = tot_nheat = 0;
  eff_nheat = tot_eff_nheat = 0;

  logical = 0;

  for(k = 0; k < Ngroups; k++)
    {
      if(massDMpart > 0)
	{
	  if(Group[k].LenType[1] * massDMpart >=
	     (All.Omega0 - All.OmegaBaryon) / All.Omega0 * All.MinFoFMassForNewSeed)
	    nheat++;
	}
      else
	{
	  printf("The DM particles mass is zero! I will stop.\n");
	  endrun(0);
	}

    }

  MPI_Allreduce(&nheat, &tot_nheat, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  if(ThisTask == 0)
    printf("The total number of clusters to heat is: %d\n", tot_nheat);


  nn_heat = mymalloc("nn_heat", NTask * sizeof(int));
  disp = mymalloc("disp", NTask * sizeof(int));

  MPI_Allgather(&nheat, 1, MPI_INT, nn_heat, 1, MPI_INT, MPI_COMM_WORLD);

  for(k = 1, disp[0] = 0; k < NTask; k++)
    disp[k] = disp[k - 1] + nn_heat[k - 1];


  if(tot_nheat > 0)
    {
      GroupMassType_common = mymalloc("GroupMassType_common", tot_nheat * sizeof(double));
      GroupMassType_dum = mymalloc("GroupMassType_dum", nheat * sizeof(double));

      GroupCM_common_x = mymalloc("GroupCM_common_x", tot_nheat * sizeof(float));
      GroupCM_dum_x = mymalloc("GroupCM_dum_x", nheat * sizeof(float));

      GroupCM_common_y = mymalloc("GroupCM_common_y", tot_nheat * sizeof(float));
      GroupCM_dum_y = mymalloc("GroupCM_dum_y", nheat * sizeof(float));

      GroupCM_common_z = mymalloc("GroupCM_common_z", tot_nheat * sizeof(float));
      GroupCM_dum_z = mymalloc("GroupCM_dum_z", nheat * sizeof(float));


      for(k = 0, i = 0; k < Ngroups; k++)
	{
	  if(massDMpart > 0)
	    {
	      if(Group[k].LenType[1] * massDMpart >=
		 (All.Omega0 - All.OmegaBaryon) / All.Omega0 * All.MinFoFMassForNewSeed)
		{
		  GroupCM_dum_x[i] = Group[k].CM[0];
		  GroupCM_dum_y[i] = Group[k].CM[1];
		  GroupCM_dum_z[i] = Group[k].CM[2];

		  GroupMassType_dum[i] = Group[k].Mass;

		  i++;
		}
	    }
	  else
	    {
	      printf("The DM particles mass is zero! I will stop.\n");
	      endrun(0);
	    }
	}

      MPI_Allgatherv(GroupMassType_dum, nheat, MPI_DOUBLE, GroupMassType_common, nn_heat, disp, MPI_DOUBLE,
		     MPI_COMM_WORLD);

      MPI_Allgatherv(GroupCM_dum_x, nheat, MPI_FLOAT, GroupCM_common_x, nn_heat, disp, MPI_FLOAT,
		     MPI_COMM_WORLD);

      MPI_Allgatherv(GroupCM_dum_y, nheat, MPI_FLOAT, GroupCM_common_y, nn_heat, disp, MPI_FLOAT,
		     MPI_COMM_WORLD);

      MPI_Allgatherv(GroupCM_dum_z, nheat, MPI_FLOAT, GroupCM_common_z, nn_heat, disp, MPI_FLOAT,
		     MPI_COMM_WORLD);


      for(l = 0; l < tot_nheat; l++)
	{
	  if(All.ComovingIntegrationOn > 0)
	    {
	      BubbleDistance =
		All.BubbleDistance * 1. / All.Time * pow(GroupMassType_common[l] / All.ClusterMass200,
							 1. / 3.) / pow(hubble_a, 2. / 3.);

	      BubbleRadius =
		All.BubbleRadius * 1. / All.Time * pow(GroupMassType_common[l] / All.ClusterMass200,
						       1. / 3.) / pow(hubble_a, 2. / 3.);

	      BubbleEnergy =
		All.BubbleEnergy * pow(GroupMassType_common[l] / All.ClusterMass200, 5. / 3.) * pow(hubble_a,
												    2. / 3.);

	      phi = theta = rr = 0.0;

	      phi = 2 * M_PI * get_random_number(0);
	      theta = acos(2 * get_random_number(0) - 1);
	      rr = pow(get_random_number(0), 1. / 3.) * BubbleDistance;

	      pos[0] = pos[1] = pos[2] = 0.0;

	      pos[0] = sin(theta) * cos(phi);
	      pos[1] = sin(theta) * sin(phi);
	      pos[2] = cos(theta);

	      for(k = 0; k < 3; k++)
		pos[k] *= rr;

	      pos[0] += GroupCM_common_x[l];
	      pos[1] += GroupCM_common_y[l];
	      pos[2] += GroupCM_common_z[l];


	      /* First, let's see how many particles are in the bubble */

	      Ngblist = (int *) mymalloc("Ngblist", NumPart * sizeof(int));

	      numngb = 0;
	      E_bubble = 0.;
	      Mass_bubble = 0.;

	      startnode = All.MaxPart;
	      do
		{
		  numngb_inbox = ngb_treefind_variable(pos, BubbleRadius, -1, &startnode, 0, &dummy, &dummy);

		  for(n = 0; n < numngb_inbox; n++)
		    {
		      j = Ngblist[n];
		      dx = pos[0] - P[j].Pos[0];
		      dy = pos[1] - P[j].Pos[1];
		      dz = pos[2] - P[j].Pos[2];

#ifdef PERIODIC			/*  now find the closest image in the given box size  */
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

		      if(r2 < BubbleRadius * BubbleRadius)
			{
			  numngb++;

			  if(All.ComovingIntegrationOn)
			    E_bubble +=
			      SPHP(j).Entropy * P[j].Mass * pow(SPHP(j).d.Density / pow(All.Time, 3),
								GAMMA_MINUS1) / GAMMA_MINUS1;
			  else
			    E_bubble +=
			      SPHP(j).Entropy * P[j].Mass * pow(SPHP(j).d.Density,
								GAMMA_MINUS1) / GAMMA_MINUS1;

			  Mass_bubble += P[j].Mass;

			}
		    }
		}
	      while(startnode >= 0);


	      tot_numngb = totE_bubble = totMass_bubble = 0.0;

	      MPI_Allreduce(&numngb, &tot_numngb, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
	      MPI_Allreduce(&E_bubble, &totE_bubble, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	      MPI_Allreduce(&Mass_bubble, &totMass_bubble, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);


	      if(tot_numngb == 0)
		{
		  tot_numngb = 1;
		  totMass_bubble = 1;
		  totE_bubble = 1;
		  logical = 0;
		}
	      else
		{
		  logical = 1;
		  eff_nheat++;
		}

	      totE_bubble *= All.UnitEnergy_in_cgs;


	      if(ThisTask == 0)
		{
		  if(logical == 1)
		    {
		      printf("%g, %g, %g, %g, %d, %g, %g, %g\n", GroupMassType_common[l], GroupCM_common_x[l],
			     GroupCM_common_y[l], GroupCM_common_z[l], tot_numngb, BubbleRadius, BubbleEnergy,
			     (BubbleEnergy + totE_bubble) / totE_bubble);

		    }
		  fflush(stdout);
		}

	      /* now find particles in Bubble again, and inject energy */

	      startnode = All.MaxPart;

	      do
		{
		  numngb_inbox = ngb_treefind_variable(pos, BubbleRadius, -1, &startnode, 0, &dummy, &dummy);

		  for(n = 0; n < numngb_inbox; n++)
		    {
		      j = Ngblist[n];
		      dx = pos[0] - P[j].Pos[0];
		      dy = pos[1] - P[j].Pos[1];
		      dz = pos[2] - P[j].Pos[2];

#ifdef PERIODIC			/*  now find the closest image in the given box size  */
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

		      if(r2 < BubbleRadius * BubbleRadius)
			{
			  /* with sf on gas particles have mass that is not fixed */
			  /* energy we want to inject in this particle */

			  if(logical == 1)
			    dE = ((BubbleEnergy / All.UnitEnergy_in_cgs) / totMass_bubble) * P[j].Mass;
			  else
			    dE = 0;


			  if(All.ComovingIntegrationOn)
			    SPHP(j).Entropy +=
			      GAMMA_MINUS1 * dE / P[j].Mass / pow(SPHP(j).d.Density / pow(All.Time, 3),
								  GAMMA_MINUS1);
			  else
			    SPHP(j).Entropy +=
			      GAMMA_MINUS1 * dE / P[j].Mass / pow(SPHP(j).d.Density, GAMMA_MINUS1);
			  if(dE > 0 && P[j].ID > 0)
			    P[j].ID = -P[j].ID;

			}
		    }
		}
	      while(startnode >= 0);

	      myfree(Ngblist);

	    }
	}

      myfree(GroupCM_dum_z);
      myfree(GroupCM_common_z);
      myfree(GroupCM_dum_y);
      myfree(GroupCM_common_y);
      myfree(GroupCM_dum_x);
      myfree(GroupCM_common_x);
      myfree(GroupMassType_dum);
      myfree(GroupMassType_common);

      MPI_Allreduce(&eff_nheat, &tot_eff_nheat, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

      if(ThisTask == 0)
	printf("The total effective! number of clusters to heat is: %d\n", eff_nheat);

    }
  else
    {
      printf("There are no clusters to heat. I will stop.\n");
      endrun(0);
    }


  myfree(disp);
  myfree(nn_heat);

}

#endif



double fof_periodic(double x)
{
  if(x >= 0.5 * All.BoxSize)
    x -= All.BoxSize;
  if(x < -0.5 * All.BoxSize)
    x += All.BoxSize;
  return x;
}


double fof_periodic_wrap(double x)
{
  while(x >= All.BoxSize)
    x -= All.BoxSize;
  while(x < 0)
    x += All.BoxSize;
  return x;
}


int fof_compare_FOF_PList_MinID(const void *a, const void *b)
{
  if(((struct fof_particle_list *) a)->MinID < ((struct fof_particle_list *) b)->MinID)
    return -1;

  if(((struct fof_particle_list *) a)->MinID > ((struct fof_particle_list *) b)->MinID)
    return +1;

  return 0;
}

int fof_compare_FOF_GList_MinID(const void *a, const void *b)
{
  if(((struct fof_group_list *) a)->MinID < ((struct fof_group_list *) b)->MinID)
    return -1;

  if(((struct fof_group_list *) a)->MinID > ((struct fof_group_list *) b)->MinID)
    return +1;

  return 0;
}

int fof_compare_FOF_GList_MinIDTask(const void *a, const void *b)
{
  if(((struct fof_group_list *) a)->MinIDTask < ((struct fof_group_list *) b)->MinIDTask)
    return -1;

  if(((struct fof_group_list *) a)->MinIDTask > ((struct fof_group_list *) b)->MinIDTask)
    return +1;

  return 0;
}

int fof_compare_FOF_GList_LocCountTaskDiffMinID(const void *a, const void *b)
{
  if(((struct fof_group_list *) a)->LocCount > ((struct fof_group_list *) b)->LocCount)
    return -1;

  if(((struct fof_group_list *) a)->LocCount < ((struct fof_group_list *) b)->LocCount)
    return +1;

  if(((struct fof_group_list *) a)->MinID < ((struct fof_group_list *) b)->MinID)
    return -1;

  if(((struct fof_group_list *) a)->MinID > ((struct fof_group_list *) b)->MinID)
    return +1;

  if(labs(((struct fof_group_list *) a)->ExtCount - ((struct fof_group_list *) a)->MinIDTask) <
     labs(((struct fof_group_list *) b)->ExtCount - ((struct fof_group_list *) b)->MinIDTask))
    return -1;

  if(labs(((struct fof_group_list *) a)->ExtCount - ((struct fof_group_list *) a)->MinIDTask) >
     labs(((struct fof_group_list *) b)->ExtCount - ((struct fof_group_list *) b)->MinIDTask))
    return +1;

  return 0;
}

int fof_compare_FOF_GList_ExtCountMinID(const void *a, const void *b)
{
  if(((struct fof_group_list *) a)->ExtCount < ((struct fof_group_list *) b)->ExtCount)
    return -1;

  if(((struct fof_group_list *) a)->ExtCount > ((struct fof_group_list *) b)->ExtCount)
    return +1;

  if(((struct fof_group_list *) a)->MinID < ((struct fof_group_list *) b)->MinID)
    return -1;

  if(((struct fof_group_list *) a)->MinID > ((struct fof_group_list *) b)->MinID)
    return +1;

  return 0;
}

int fof_compare_Group_MinID(const void *a, const void *b)
{
  if(((struct group_properties *) a)->MinID < ((struct group_properties *) b)->MinID)
    return -1;

  if(((struct group_properties *) a)->MinID > ((struct group_properties *) b)->MinID)
    return +1;

  return 0;
}

int fof_compare_Group_GrNr(const void *a, const void *b)
{
  if(((struct group_properties *) a)->GrNr < ((struct group_properties *) b)->GrNr)
    return -1;

  if(((struct group_properties *) a)->GrNr > ((struct group_properties *) b)->GrNr)
    return +1;

  return 0;
}

int fof_compare_Group_MinIDTask(const void *a, const void *b)
{
  if(((struct group_properties *) a)->MinIDTask < ((struct group_properties *) b)->MinIDTask)
    return -1;

  if(((struct group_properties *) a)->MinIDTask > ((struct group_properties *) b)->MinIDTask)
    return +1;

  return 0;
}

int fof_compare_Group_MinIDTask_MinID(const void *a, const void *b)
{
  if(((struct group_properties *) a)->MinIDTask < ((struct group_properties *) b)->MinIDTask)
    return -1;

  if(((struct group_properties *) a)->MinIDTask > ((struct group_properties *) b)->MinIDTask)
    return +1;

  if(((struct group_properties *) a)->MinID < ((struct group_properties *) b)->MinID)
    return -1;

  if(((struct group_properties *) a)->MinID > ((struct group_properties *) b)->MinID)
    return +1;

  return 0;
}


int fof_compare_Group_Len(const void *a, const void *b)
{
  if(((struct group_properties *) a)->Len > ((struct group_properties *) b)->Len)
    return -1;

  if(((struct group_properties *) a)->Len < ((struct group_properties *) b)->Len)
    return +1;

  return 0;
}



int fof_compare_ID_list_GrNrID(const void *a, const void *b)
{
  if(((struct id_list *) a)->GrNr < ((struct id_list *) b)->GrNr)
    return -1;

  if(((struct id_list *) a)->GrNr > ((struct id_list *) b)->GrNr)
    return +1;

  if(((struct id_list *) a)->ID < ((struct id_list *) b)->ID)
    return -1;

  if(((struct id_list *) a)->ID > ((struct id_list *) b)->ID)
    return +1;

  return 0;
}








#ifdef SUBFIND_READ_FOF		/* read already existing FOF instead of recomputing it */

void read_fof(int num)
{
  FILE *fd;
  double t0, t1;
  char fname[500];
  float *mass, *cm;
  int i, j, ntask, *len, count;
  MyIDType *ids;
  int *list_of_ngroups, *list_of_nids, *list_of_allgrouplen;
  int *recvoffset;
  int grnr, ngrp, sendTask, recvTask;
  int nprocgroup, masterTask, groupTask, nid_previous;
  int fof_compare_P_SubNr(const void *a, const void *b);


  if(ThisTask == 0)
    {
      printf("\nTrying to read preexisting FoF group catalogues...  (presently allocated=%g MB)\n",
	     AllocatedBytes / (1024.0 * 1024.0));
      fflush(stdout);
    }


  domain_Decomposition();

  /* start reading of group catalogue */

  if(ThisTask == 0)
    {
      sprintf(fname, "%s/groups_%03d/%s_%03d.%d", All.OutputDir, num, "group_tab", num, 0);
      if(!(fd = fopen(fname, "r")))
	{
	  printf("can't read file `%s`\n", fname);
	  endrun(11831);
	}

      my_fread(&Ngroups, sizeof(int), 1, fd);
      my_fread(&TotNgroups, sizeof(int), 1, fd);
      my_fread(&Nids, sizeof(int), 1, fd);
      my_fread(&TotNids, sizeof(int64_t), 1, fd);
      my_fread(&ntask, sizeof(int), 1, fd);
      fclose(fd);
    }

  MPI_Bcast(&ntask, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&TotNgroups, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&TotNids, sizeof(int64_t), MPI_BYTE, 0, MPI_COMM_WORLD);

  t0 = second();

  if(NTask != ntask)
    {
      if(ThisTask == 0)
	printf
	  ("number of files (%d) in group catalogues does not match MPI-Tasks, I'm working around this.\n",
	   ntask);

      Group =
	(struct group_properties *) mymalloc("Group",
					     sizeof(struct group_properties) * (TotNgroups / NTask + NTask));

      ID_list = mymalloc("ID_list", (TotNids / NTask + NTask) * sizeof(struct id_list));


      int filenr, target, ngroups, nids, nsend, stored;

      int *ngroup_to_get = mymalloc("ngroup_to_get", NTask * sizeof(NTask));
      int *nids_to_get = mymalloc("nids_to_get", NTask * sizeof(NTask));
      int *ngroup_obtained = mymalloc("ngroup_obtained", NTask * sizeof(NTask));
      int *nids_obtained = mymalloc("nids_obtained", NTask * sizeof(NTask));

      for(i = 0; i < NTask; i++)
	ngroup_obtained[i] = nids_obtained[i] = 0;

      for(i = 0; i < NTask - 1; i++)
	ngroup_to_get[i] = TotNgroups / NTask;
      ngroup_to_get[NTask - 1] = TotNgroups - (NTask - 1) * (TotNgroups / NTask);

      for(i = 0; i < NTask - 1; i++)
	nids_to_get[i] = (int) (TotNids / NTask);
      nids_to_get[NTask - 1] = (int) (TotNids - (NTask - 1) * (TotNids / NTask));

      Ngroups = ngroup_to_get[ThisTask];
      Nids = nids_to_get[ThisTask];



      if(ThisTask == 0)
	{
	  for(filenr = 0; filenr < ntask; filenr++)
	    {
	      sprintf(fname, "%s/groups_%03d/%s_%03d.%d", All.OutputDir, num, "group_tab", num, filenr);
	      if(!(fd = fopen(fname, "r")))
		{
		  printf("can't read file `%s`\n", fname);
		  endrun(11831);
		}

	      printf("reading '%s'\n", fname);
	      fflush(stdout);

	      my_fread(&ngroups, sizeof(int), 1, fd);
	      my_fread(&TotNgroups, sizeof(int), 1, fd);
	      my_fread(&nids, sizeof(int), 1, fd);
	      my_fread(&TotNids, sizeof(int64_t), 1, fd);
	      my_fread(&ntask, sizeof(int), 1, fd);

	      struct group_properties *tmpGroup =
		(struct group_properties *) mymalloc("tmpGroup", sizeof(struct group_properties) * ngroups);

	      /* group len */
	      len = mymalloc("len", ngroups * sizeof(int));
	      my_fread(len, ngroups, sizeof(int), fd);
	      for(i = 0; i < ngroups; i++)
		tmpGroup[i].Len = len[i];
	      myfree(len);

	      /* offset into id-list */
	      len = mymalloc("len", ngroups * sizeof(int));
	      my_fread(len, ngroups, sizeof(int), fd);
	      for(i = 0; i < ngroups; i++)
		tmpGroup[i].Offset = len[i];
	      myfree(len);

	      /* mass */
	      mass = mymalloc("mass", ngroups * sizeof(float));
	      my_fread(mass, ngroups, sizeof(float), fd);
	      for(i = 0; i < ngroups; i++)
		tmpGroup[i].Mass = mass[i];
	      myfree(mass);

	      /* CM */
	      cm = mymalloc("cm", ngroups * 3 * sizeof(float));
	      my_fread(cm, ngroups, 3 * sizeof(float), fd);
	      for(i = 0; i < ngroups; i++)
		for(j = 0; j < 3; j++)
		  tmpGroup[i].CM[j] = cm[i * 3 + j];
	      myfree(cm);

	      fclose(fd);

	      target = 0;
	      stored = 0;
	      while(ngroups > 0)
		{
		  while(ngroup_to_get[target] == 0)
		    target++;

		  if(ngroups > ngroup_to_get[target])
		    nsend = ngroup_to_get[target];
		  else
		    nsend = ngroups;

		  if(target == 0)
		    memcpy(&Group[ngroup_obtained[target]], &tmpGroup[stored],
			   nsend * sizeof(struct group_properties));
		  else
		    {
		      MPI_Send(&nsend, 1, MPI_INT, target, TAG_N, MPI_COMM_WORLD);
		      MPI_Send(&tmpGroup[stored], nsend * sizeof(struct group_properties), MPI_BYTE,
			       target, TAG_PDATA, MPI_COMM_WORLD);
		    }

		  ngroup_to_get[target] -= nsend;
		  ngroup_obtained[target] += nsend;
		  ngroups -= nsend;
		  stored += nsend;
		}

	      myfree(tmpGroup);
	    }



	      /**** now ids ****/
	  for(filenr = 0; filenr < ntask; filenr++)
	    {
	      sprintf(fname, "%s/groups_%03d/%s_%03d.%d", All.OutputDir, num, "group_ids", num, filenr);
	      if(!(fd = fopen(fname, "r")))
		{
		  printf("can't read file `%s`\n", fname);
		  endrun(1184132);
		}

	      printf("reading '%s'\n", fname);
	      fflush(stdout);

	      my_fread(&ngroups, sizeof(int), 1, fd);
	      my_fread(&TotNgroups, sizeof(int), 1, fd);
	      my_fread(&nids, sizeof(int), 1, fd);
	      my_fread(&TotNids, sizeof(int64_t), 1, fd);
	      my_fread(&ntask, sizeof(int), 1, fd);
	      my_fread(&nid_previous, sizeof(int), 1, fd);	/* this is the number of IDs in previous files */


	      struct id_list *tmpID_list = mymalloc("tmpID_list", nids * sizeof(struct id_list));

	      ids = mymalloc("ids", nids * sizeof(MyIDType));

	      my_fread(ids, sizeof(MyIDType), nids, fd);

	      for(i = 0; i < nids; i++)
		tmpID_list[i].ID = ids[i];

	      myfree(ids);

	      fclose(fd);

	      target = 0;
	      stored = 0;
	      while(nids > 0)
		{
		  while(nids_to_get[target] == 0)
		    target++;

		  if(nids > nids_to_get[target])
		    nsend = nids_to_get[target];
		  else
		    nsend = nids;

		  if(target == 0)
		    memcpy(&ID_list[nids_obtained[target]], &tmpID_list[stored],
			   nsend * sizeof(struct id_list));
		  else
		    {
		      MPI_Send(&nsend, 1, MPI_INT, target, TAG_HEADER, MPI_COMM_WORLD);
		      MPI_Send(&tmpID_list[stored], nsend * sizeof(struct id_list), MPI_BYTE,
			       target, TAG_SPHDATA, MPI_COMM_WORLD);
		    }

		  nids_to_get[target] -= nsend;
		  nids_obtained[target] += nsend;
		  nids -= nsend;
		  stored += nsend;
		}

	      myfree(tmpID_list);
	    }
	}
      else
	{
	  while(ngroup_to_get[ThisTask])
	    {
	      MPI_Recv(&nsend, 1, MPI_INT, 0, TAG_N, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	      MPI_Recv(&Group[ngroup_obtained[ThisTask]], nsend * sizeof(struct group_properties), MPI_BYTE,
		       0, TAG_PDATA, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

	      ngroup_to_get[ThisTask] -= nsend;
	      ngroup_obtained[ThisTask] += nsend;
	    }

	  while(nids_to_get[ThisTask])
	    {
	      MPI_Recv(&nsend, 1, MPI_INT, 0, TAG_HEADER, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	      MPI_Recv(&ID_list[nids_obtained[ThisTask]], nsend * sizeof(struct id_list), MPI_BYTE,
		       0, TAG_SPHDATA, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

	      nids_to_get[ThisTask] -= nsend;
	      nids_obtained[ThisTask] += nsend;
	    }
	}

      myfree(nids_obtained);
      myfree(ngroup_obtained);
      myfree(nids_to_get);
      myfree(ngroup_to_get);
    }

  else
    {
      /* read routine can constinue in parallel */

      nprocgroup = NTask / All.NumFilesWrittenInParallel;
      if((NTask % All.NumFilesWrittenInParallel))
	nprocgroup++;
      masterTask = (ThisTask / nprocgroup) * nprocgroup;
      for(groupTask = 0; groupTask < nprocgroup; groupTask++)
	{
	  if(ThisTask == (masterTask + groupTask))	/* ok, it's this processor's turn */
	    {

	      sprintf(fname, "%s/groups_%03d/%s_%03d.%d", All.OutputDir, num, "group_tab", num, ThisTask);
	      if(!(fd = fopen(fname, "r")))
		{
		  printf("can't read file `%s`\n", fname);
		  endrun(11831);
		}

	      printf("reading '%s'\n", fname);
	      fflush(stdout);

	      my_fread(&Ngroups, sizeof(int), 1, fd);
	      my_fread(&TotNgroups, sizeof(int), 1, fd);
	      my_fread(&Nids, sizeof(int), 1, fd);
	      my_fread(&TotNids, sizeof(int64_t), 1, fd);
	      my_fread(&ntask, sizeof(int), 1, fd);
	      if(NTask != ntask)
		{
		  if(ThisTask == 0)
		    printf("number of files in group catalogues needs to match MPI-Tasks\n");
		  endrun(0);
		}

	      if(ThisTask == 0)
		printf("TotNgroups=%d\n", TotNgroups);

	      Group = (struct group_properties *) mymalloc("Group", sizeof(struct group_properties) *
							   IMAX(Ngroups, TotNgroups / NTask + 1));

	      /* group len */
	      len = mymalloc("len", Ngroups * sizeof(int));
	      my_fread(len, Ngroups, sizeof(int), fd);
	      for(i = 0; i < Ngroups; i++)
		Group[i].Len = len[i];
	      myfree(len);

	      /* offset into id-list */
	      len = mymalloc("len", Ngroups * sizeof(int));
	      my_fread(len, Ngroups, sizeof(int), fd);
	      for(i = 0; i < Ngroups; i++)
		Group[i].Offset = len[i];
	      myfree(len);

	      /* mass */
	      mass = mymalloc("mass", Ngroups * sizeof(float));
	      my_fread(mass, Ngroups, sizeof(float), fd);
	      for(i = 0; i < Ngroups; i++)
		Group[i].Mass = mass[i];
	      myfree(mass);

	      /* CM */
	      cm = mymalloc("cm", Ngroups * 3 * sizeof(float));
	      my_fread(cm, Ngroups, 3 * sizeof(float), fd);
	      for(i = 0; i < Ngroups; i++)
		for(j = 0; j < 3; j++)
		  Group[i].CM[j] = cm[i * 3 + j];
	      myfree(cm);

	      fclose(fd);


	      printf("reading '%s'\n", fname);
	      fflush(stdout);


	      sprintf(fname, "%s/groups_%03d/%s_%03d.%d", All.OutputDir, num, "group_ids", num, ThisTask);
	      if(!(fd = fopen(fname, "r")))
		{
		  printf("can't read file `%s`\n", fname);
		  endrun(1184132);
		}

	      printf("reading '%s'\n", fname);
	      fflush(stdout);

	      my_fread(&Ngroups, sizeof(int), 1, fd);
	      my_fread(&TotNgroups, sizeof(int), 1, fd);
	      my_fread(&Nids, sizeof(int), 1, fd);
	      my_fread(&TotNids, sizeof(int64_t), 1, fd);
	      my_fread(&ntask, sizeof(int), 1, fd);
	      my_fread(&nid_previous, sizeof(int), 1, fd);	/* this is the number of IDs in previous files */

	      ID_list = mymalloc("ID_list", Nids * sizeof(struct id_list));
	      ids = mymalloc("ids", Nids * sizeof(MyIDType));

	      my_fread(ids, sizeof(MyIDType), Nids, fd);

	      for(i = 0; i < Nids; i++)
		ID_list[i].ID = ids[i];

	      myfree(ids);

	      fclose(fd);
	    }

	  MPI_Barrier(MPI_COMM_WORLD);	/* wait inside the group */
	}
    }

  t1 = second();
  if(ThisTask == 0)
    printf("reading  took %g sec\n", timediff(t0, t1));



  t0 = second();

  /* now need to assign group numbers */


  list_of_ngroups = mymalloc("list_of_ngroups", NTask * sizeof(int));
  list_of_nids = mymalloc("list_of_nids", NTask * sizeof(int));

  MPI_Allgather(&Ngroups, 1, MPI_INT, list_of_ngroups, 1, MPI_INT, MPI_COMM_WORLD);
  MPI_Allgather(&Nids, 1, MPI_INT, list_of_nids, 1, MPI_INT, MPI_COMM_WORLD);

  list_of_allgrouplen = mymalloc("list_of_allgrouplen", TotNgroups * sizeof(int));

  recvoffset = mymalloc("recvoffset", NTask * sizeof(int));
  for(i = 1, recvoffset[0] = 0; i < NTask; i++)
    recvoffset[i] = recvoffset[i - 1] + list_of_ngroups[i - 1];
  len = mymalloc("len", Ngroups * sizeof(int));
  for(i = 0; i < Ngroups; i++)
    len[i] = Group[i].Len;
  MPI_Allgatherv(len, Ngroups, MPI_INT, list_of_allgrouplen, list_of_ngroups, recvoffset, MPI_INT,
		 MPI_COMM_WORLD);
  myfree(len);
  myfree(recvoffset);

  /* do a check */
  int64_t totlen;

  for(i = 0, totlen = 0; i < TotNgroups; i++)
    totlen += list_of_allgrouplen[i];
  if(totlen != TotNids)
    endrun(8881);


  for(i = 0, count = 0; i < ThisTask; i++)
    count += list_of_ngroups[i];

  for(i = 0; i < Ngroups; i++)
    Group[i].GrNr = 1 + count + i;

  /* fix Group.Offset (may have overflown) */
  if(Ngroups > 0)
    for(i = 0, count = 0, Group[0].Offset = 0; i < ThisTask; i++)
      for(j = 0; j < list_of_ngroups[i]; j++)
	Group[0].Offset += list_of_allgrouplen[count++];

  for(i = 1; i < Ngroups; i++)
    Group[i].Offset = Group[i - 1].Offset + Group[i - 1].Len;


  int64_t *idoffset = mymalloc("idoffset", NTask * sizeof(int64_t));

  for(i = 1, idoffset[0] = 0; i < NTask; i++)
    idoffset[i] = idoffset[i - 1] + list_of_nids[i - 1];

  count = 0;

  for(i = 0, grnr = 1, totlen = 0; i < TotNgroups; totlen += list_of_allgrouplen[i++], grnr++)
    {
      if(totlen + list_of_allgrouplen[i] - 1 >= idoffset[ThisTask] && totlen < idoffset[ThisTask] + Nids)
	{
	  for(j = 0; j < list_of_allgrouplen[i]; j++)
	    {
	      if((totlen + j) >= idoffset[ThisTask] && (totlen + j) < (idoffset[ThisTask] + Nids))
		{
		  ID_list[(totlen + j) - idoffset[ThisTask]].GrNr = grnr;
		  count++;
		}
	    }
	}
    }
  if(count != Nids)
    endrun(1231);

  myfree(idoffset);

  t1 = second();
  if(ThisTask == 0)
    printf("assigning took %g sec\n", timediff(t0, t1));



  t0 = second();

  for(i = 0; i < NumPart; i++)
    P[i].SubNr = i;

  qsort(P, NumPart, sizeof(struct particle_data), io_compare_P_ID);
  qsort(ID_list, Nids, sizeof(struct id_list), fof_compare_ID_list_ID);

  for(i = 0; i < NumPart; i++)
    P[i].GrNr = TotNgroups + 1;	/* will mark particles that are not in any group */

  t1 = second();
  if(ThisTask == 0)
    printf("sorting took %g sec\n", timediff(t0, t1));


  static struct id_list *recv_ID_list;

  t0 = second();

  int matches = 0;

  /* exchange  data */
  for(ngrp = 0; ngrp < (1 << PTask); ngrp++)
    {
      sendTask = ThisTask;
      recvTask = ThisTask ^ ngrp;

      if(recvTask < NTask)
	{
	  if(list_of_nids[sendTask] > 0 || list_of_nids[recvTask] > 0)
	    {
	      if(ngrp == 0)
		{
		  recv_ID_list = ID_list;
		}
	      else
		{
		  recv_ID_list = mymalloc("recv_ID_list", list_of_nids[recvTask] * sizeof(struct id_list));

		  /* get the particles */
		  MPI_Sendrecv(ID_list, Nids * sizeof(struct id_list), MPI_BYTE, recvTask, TAG_DENS_A,
			       recv_ID_list, list_of_nids[recvTask] * sizeof(struct id_list), MPI_BYTE,
			       recvTask, TAG_DENS_A, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}

	      for(i = 0, j = 0; i < list_of_nids[recvTask]; i++)
		{
		  while(j < NumPart - 1 && P[j].ID < recv_ID_list[i].ID)
		    j++;

		  if(recv_ID_list[i].ID == P[j].ID)
		    {
		      P[j].GrNr = recv_ID_list[i].GrNr;
		      matches++;
		    }
		}

	      if(ngrp != 0)
		myfree(recv_ID_list);
	    }
	}
    }

  sumup_large_ints(1, &matches, &totlen);
  if(totlen != TotNids)
    endrun(543);

  t1 = second();
  if(ThisTask == 0)
    printf("assigning GrNr to P[] took %g sec\n", timediff(t0, t1));

  MPI_Barrier(MPI_COMM_WORLD);

  myfree(list_of_allgrouplen);
  myfree(list_of_nids);
  myfree(list_of_ngroups);

  /* restore peano-hilbert order */
  qsort(P, NumPart, sizeof(struct particle_data), fof_compare_P_SubNr);

  subfind(num);
  endrun(0);

}



int fof_compare_ID_list_ID(const void *a, const void *b)
{
  if(((struct id_list *) a)->ID < ((struct id_list *) b)->ID)
    return -1;

  if(((struct id_list *) a)->ID > ((struct id_list *) b)->ID)
    return +1;

  return 0;
}


int fof_compare_P_SubNr(const void *a, const void *b)
{
  if(((struct particle_data *) a)->SubNr < (((struct particle_data *) b)->SubNr))
    return -1;

  if(((struct particle_data *) a)->SubNr > (((struct particle_data *) b)->SubNr))
    return +1;

  return 0;
}

#endif /* of SUBFIND_READ_FOF */

#endif /* of FOF */
