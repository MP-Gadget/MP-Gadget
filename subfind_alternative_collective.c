#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "allvars.h"
#include "proto.h"
#include "domain.h"

#ifdef SUBFIND
#ifdef SUBFIND_ALTERNATIVE_COLLECTIVE
#include "subfind.h"
#include "fof.h"



#define MASK ((((long long)1)<< 32)-1)
#define HIGHBIT (1 << 30)





static long long **Head, **Next, **Tail;
static int **Len;
static int LocalLen;
static int *count_cand, max_candidates, loc_count_cand;
static int TotalGroupLen;
static int *List_NumPartGroup, *Offset_NumPartGroup, *List_LocalLen;

static int **targettask, **submark, **origintask;

static struct cand_dat
{
  long long head;
  long long rank;
  int len;
  int nsub;
  int subnr, parent;
  int bound_length;
}
**candidates, *loc_candidates;

int subfind_mark_independent_ones(void);


static struct unbind_data *ud, **ud_list;


static struct sort_density_data
{
  MyFloat density;
  int ngbcount;
  long long index;		/* this will store the task in the upper word */
  long long ngb_index1, ngb_index2;
}
 *sd;

void subfind_process_group_collectively(int num)
{
  long long p;
  int len, totgrouplen1, totgrouplen2;
  int parent, totcand, nremaining;
  int max_length;
  int countall;
  int i, j, k, nr, grindex = 0, nsubs = 0, subnr;
  int tot_count_leaves;
  int task;
  double SubMass, SubPos[3], SubVel[3], SubCM[3], SubVelDisp, SubVmax, SubVmaxRad, SubSpin[3], SubHalfMass,
    SubMassTab[6];
  struct cand_dat *tmp_candidates = 0;
  MyIDType SubMostBoundID;
  double t0, t1, tt0, tt1, ttt0, ttt1;


  ttt0 = second();

  if(ThisTask == 0)
    printf("\ncollectively doing halo %d (alternative subfind_collective)\n", GrNr);

  for(i = 0, NumPartGroup = 0; i < NumPart; i++)
    if(P[i].GrNr == GrNr)
      NumPartGroup++;

  MPI_Allreduce(&NumPartGroup, &totgrouplen1, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  if(ThisTask == ((GrNr - 1) % NTask))
    {
      for(grindex = 0; grindex < Ngroups; grindex++)
	if(Group[grindex].GrNr == GrNr)
	  break;
      if(grindex >= Ngroups)
	endrun(8);
      totgrouplen2 = Group[grindex].Len;
    }

  MPI_Bcast(&totgrouplen2, 1, MPI_INT, (GrNr - 1) % NTask, MPI_COMM_WORLD);

  if(totgrouplen1 != totgrouplen2)
    endrun(9);			/* inconsistency */

  TotalGroupLen = totgrouplen1;



  /* distribute this halo among the processors */
  t0 = second();

  All.DoDynamicUpdate = 0;

  domain_free_trick();

  domain_Decomposition();

  t1 = second();
  if(ThisTask == 0)
    printf("coldomain_Decomposition() took %g sec  (presently allocated=%g MB)\n", timediff(t0, t1),
	   AllocatedBytes / (1024.0 * 1024.0));


  for(i = 0; i < NumPart; i++)
    P[i].origindex = i;

  qsort(P, NumPart, sizeof(struct particle_data), subfind_compare_P_GrNrGrNr);

  /* now we have the particles of the group at the beginning, but SPH particles are not aligned. 
     They can however be accessed via SphP[P[i].origindex] */

  for(i = 0, NumPartGroup = 0; i < NumPart; i++)
    if(P[i].GrNr == GrNr)
      NumPartGroup++;

  subfind_loctree_copyExtent();	/* this will make sure that all the serial trees start from the same root node geometry */

  /* construct a tree for the halo */
  t0 = second();
  force_treebuild(NumPartGroup, NULL);
  t1 = second();
  if(ThisTask == 0)
    printf("force_treebuild() took %g sec (presently allocated=%g MB)\n", timediff(t0, t1),
	   AllocatedBytes / (1024.0 * 1024.0));


  /* determine the radius that encloses a certain number of link particles */
  t0 = second();
  subfind_find_linkngb();
  t1 = second();
  if(ThisTask == 0)
    printf("find_linkngb() took %g sec\n", timediff(t0, t1));

  List_NumPartGroup = mymalloc("List_NumPartGroup", NTask * sizeof(int));
  Offset_NumPartGroup = mymalloc("Offset_NumPartGroup", NTask * sizeof(int));

  MPI_Allgather(&NumPartGroup, 1, MPI_INT, List_NumPartGroup, 1, MPI_INT, MPI_COMM_WORLD);

  for(task = 1, Offset_NumPartGroup[0] = 0; task < NTask; task++)
    Offset_NumPartGroup[task] = Offset_NumPartGroup[task - 1] + List_NumPartGroup[task - 1];


  max_candidates = (TotalGroupLen / NTask / 50);

  if(ThisTask == 0)
    {
      /* allocate a list to store subhalo candidates */

      count_cand = mymalloc("count_cand", NTask * sizeof(int));
      candidates = mymalloc("candidates", NTask * sizeof(struct cand_dat *));

      for(i = 0; i < NTask; i++)
	{
	  count_cand[i] = 0;
	  candidates[i] = mymalloc("	  candidates[i]", max_candidates * sizeof(struct cand_dat));
	}

      /* allocate and initialize global link list on CPU0 */
      Next = mymalloc("Next", NTask * sizeof(long long *));
      Len = mymalloc("Len", NTask * sizeof(int *));

      for(i = 0; i < NTask; i++)
	{
	  Next[i] = mymalloc("	  Next[i]", List_NumPartGroup[i] * sizeof(long long));
	  Len[i] = mymalloc("	  Len[i]", List_NumPartGroup[i] * sizeof(int));
	}

      Head = mymalloc("Head", NTask * sizeof(long long *));
      Tail = mymalloc("Tail", NTask * sizeof(long long *));

      for(i = 0; i < NTask; i++)
	{
	  Head[i] = mymalloc("	  Head[i]", List_NumPartGroup[i] * sizeof(long long));
	  Tail[i] = mymalloc("	  Tail[i]", List_NumPartGroup[i] * sizeof(long long));
	}
    }

  if(ThisTask == 0)
    {
      printf("Link-list allocated on CPU 0: presently allocated there=%g MB\n",
	     AllocatedBytes / (1024.0 * 1024.0));
      fflush(stdout);
    }



  if(ThisTask == 0)
    sd = mymalloc("sd", TotalGroupLen * sizeof(struct sort_density_data));
  else
    sd = mymalloc("sd", NumPartGroup * sizeof(struct sort_density_data));

  /* determine the indices of the nearest two denser neighbours within the link region */
  t0 = second();
  NgbLoc = mymalloc("NgbLoc", NumPartGroup * sizeof(struct nearest_ngb_data));
  R2Loc = mymalloc("R2Loc", NumPartGroup * sizeof(struct nearest_r2_data));

  if(ThisTask == 0)
    {
      printf("sd-global allocated on CPU 0: presently allocated there=%g MB\n",
	     AllocatedBytes / (1024.0 * 1024.0));
      fflush(stdout);
    }

  subfind_find_nearesttwo();
  t1 = second();
  if(ThisTask == 0)
    printf("find_nearesttwo() took %g sec (presently allocated=%g MB)\n", timediff(t0, t1),
	   AllocatedBytes / (1024.0 * 1024.0));

  for(i = 0; i < NumPartGroup; i++)
    {
      sd[i].density = P[i].u.DM_Density;
      sd[i].ngbcount = NgbLoc[i].count;
      sd[i].index = (((long long) ThisTask) << 32) + i;
      sd[i].ngb_index1 = NgbLoc[i].index[0];
      sd[i].ngb_index2 = NgbLoc[i].index[1];
    }
  myfree(R2Loc);
  myfree(NgbLoc);

  /* sort the densities */
  t0 = second();
  parallel_sort(sd, NumPartGroup, sizeof(struct sort_density_data), subfind_compare_densities);
  t1 = second();
  if(ThisTask == 0)
    printf("parallel sort of densities done. took %g sec\n", timediff(t0, t1));

  t0 = second();
  /* now get all the other sd's from other CPUs */
  if(ThisTask == 0)
    {
      for(task = 1; task < NTask; task++)
	{
	  MPI_Send(&task, 1, MPI_INT, task, TAG_N, MPI_COMM_WORLD);
	  MPI_Recv(&sd[Offset_NumPartGroup[task]], List_NumPartGroup[task] * sizeof(struct sort_density_data),
		   MPI_BYTE, task, TAG_PDATA, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}
    }
  else
    {
      MPI_Recv(&i, 1, MPI_INT, 0, TAG_N, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Send(sd, NumPartGroup * sizeof(struct sort_density_data), MPI_BYTE, 0, TAG_PDATA, MPI_COMM_WORLD);
    }
  t1 = second();
  if(ThisTask == 0)
    printf("global sd list assembled on task 0 took %g sec\n", timediff(t0, t1));


  MPI_Barrier(MPI_COMM_WORLD);


  if(ThisTask == 0)
    {
      t0 = second();
      for(i = 0; i < NTask; i++)
	{
	  for(j = 0; j < List_NumPartGroup[i]; j++)
	    {
	      Head[i][j] = Next[i][j] = Tail[i][j] = -1;
	      Len[i][j] = 0;
	    }
	}
      t1 = second();
      if(ThisTask == 0)
	printf("initializing link list took %g sec\n", timediff(t0, t1));
    }


  if(ThisTask == 0)
    {
      printf("before col_find_candidates on CPU 0: presently allocated there=%g MB\n",
	     AllocatedBytes / (1024.0 * 1024.0));
      fflush(stdout);
    }


  if(ThisTask == 0)
    {
      t0 = second();

      subfind_col_find_candidates(TotalGroupLen);

      t1 = second();

      printf("finding of candidates on task 0 took %g sec\n", timediff(t0, t1));
    }

  myfree(sd);



  /* establish total number of candidates */
  if(ThisTask == 0)
    for(i = 0, totcand = 0; i < NTask; i++)
      totcand += count_cand[i];

  MPI_Bcast(&totcand, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if(ThisTask == 0)
    printf("\ntotal number of subhalo candidates=%d\n", totcand);

  nremaining = totcand;

  if(ThisTask == 0)
    {
      for(i = NTask - 1; i >= 0; i--)
	{
	  myfree(Tail[i]);
	  myfree(Head[i]);
	}

      myfree(Tail);
      myfree(Head);
    }

  if(ThisTask == 0)
    {
      for(task = 0; task < NTask; task++)
	{
	  for(i = 0; i < List_NumPartGroup[task]; i++)
	    Len[task][i] = -1;

	  for(i = 0; i < count_cand[task]; i++)
	    candidates[task][i].parent = 0;
	}
    }

  do
    {
      /* Let's see which candidates can be unbound independent from each other.
         We identify them with those candidates that have no embedded other candidate */

      t0 = second();

      if(ThisTask == 0)
	{
	  tmp_candidates = mymalloc("	  tmp_candidates", totcand * sizeof(struct cand_dat));

	  for(task = 0, j = 0; task < NTask; task++)
	    {
	      for(i = 0; i < count_cand[task]; i++)
		tmp_candidates[j++] = candidates[task][i];
	    }

	  for(k = 0; k < totcand; k++)
	    {
	      tmp_candidates[k].nsub = k;
	      tmp_candidates[k].subnr = k;
	    }

	  qsort(tmp_candidates, totcand, sizeof(struct cand_dat), subfind_compare_candidates_rank);

	  for(k = 0; k < totcand; k++)
	    {
	      if(tmp_candidates[k].parent >= 0)
		{
		  tmp_candidates[k].parent = 0;

		  for(j = k + 1; j < totcand; j++)
		    {
		      if(tmp_candidates[j].rank > tmp_candidates[k].rank + tmp_candidates[k].len)
			break;

		      if(tmp_candidates[j].parent < 0)	/* ignore these */
			continue;

		      if(tmp_candidates[k].rank + tmp_candidates[k].len >=
			 tmp_candidates[j].rank + tmp_candidates[j].len)
			{
			  tmp_candidates[k].parent++;	/* we here count the number of subhalos that are enclosed */
			}
		      else
			{
			  printf("k=%d|%d has rank=%d and len=%d.  j=%d has rank=%d and len=%d\n",
				 k, totcand,
				 (int) tmp_candidates[k].rank, (int) tmp_candidates[k].len,
				 j, (int) tmp_candidates[j].rank, (int) tmp_candidates[j].len);
			  endrun(8812313);
			}
		    }
		}
	    }
	  qsort(tmp_candidates, totcand, sizeof(struct cand_dat), subfind_compare_candidates_subnr);

	  for(task = 0, j = 0; task < NTask; task++)
	    {
	      for(i = 0; i < count_cand[task]; i++)
		candidates[task][i] = tmp_candidates[j++];
	    }

	  myfree(tmp_candidates);


	  for(task = 0, tot_count_leaves = 0, max_length = 0; task < NTask; task++)
	    for(i = 0; i < count_cand[task]; i++)
	      if(candidates[task][i].parent == 0)
		{
		  if(candidates[task][i].len > max_length)
		    max_length = candidates[task][i].len;

		  if(candidates[task][i].len > 0.15 * All.TotNumPart / NTask)	/* seems large, let's rather do it collectively */
		    {
		      candidates[task][i].parent++;	/* this will ensure that it is not considered in this round */
		    }
		  else
		    {
		      tot_count_leaves++;
		    }
		}
	}

      MPI_Bcast(&tot_count_leaves, 1, MPI_INT, 0, MPI_COMM_WORLD);
      MPI_Bcast(&max_length, 1, MPI_INT, 0, MPI_COMM_WORLD);

      t1 = second();
      if(ThisTask == 0)
	printf
	  ("\nnumber of subhalo candidates that can be done independently=%d.\n(Largest size is %d, finding them took %g sec)\n",
	   tot_count_leaves, max_length, timediff(t0, t1));

      if(tot_count_leaves <= 0 * NTask)	/* if there are only a few left, let's do them collectively */
	{
	  if(ThisTask == 0)
	    printf("too few, I do the rest of %d collectively\n\n", nremaining);
	  break;
	}

      if(max_length > 0.8 * All.TotNumPart / NTask)	/* seems large, let's rather do it collectively */
	{
	  if(ThisTask == 0)
	    printf("too big candidates, I do the rest collectively\n\n");
	  break;
	}

      nremaining -= tot_count_leaves;

      for(i = 0; i < NumPart; i++)
	{
	  P[i].origintask = P[i].targettask = ThisTask;
	  P[i].submark = HIGHBIT;
	}

      if(ThisTask == 0)
	{
	  t0 = second();

	  origintask = mymalloc("	  origintask", NTask * sizeof(int *));
	  targettask = mymalloc("	  targettask", NTask * sizeof(int *));
	  submark = mymalloc("	  submark", NTask * sizeof(int *));

	  for(task = 0; task < NTask; task++)
	    {
	      origintask[task] =
		mymalloc("	      origintask[task]", List_NumPartGroup[task] * sizeof(int));
	      targettask[task] =
		mymalloc("	      targettask[task]", List_NumPartGroup[task] * sizeof(int));
	      submark[task] = mymalloc("	      submark[task]", List_NumPartGroup[task] * sizeof(int));
	    }

	  printf("before submark on CPU 0: presently allocated there=%g MB\n",
		 AllocatedBytes / (1024.0 * 1024.0));
	  fflush(stdout);

	  for(task = 0; task < NTask; task++)
	    {
	      for(i = 0; i < List_NumPartGroup[task]; i++)
		{
		  submark[task][i] = HIGHBIT;
		  origintask[task][i] = targettask[task][i] = task;

		  if(Len[task][i] >= 0)	/* this means this particle is already bound to a substructure */
		    origintask[task][i] |= HIGHBIT;
		}
	    }


	  /* we now mark the particles that are in subhalo candidates that can be processed independently in parallel */

	  nsubs = subfind_mark_independent_ones();

	  t1 = second();

	  printf("particles are marked (took %g)\n", timediff(t0, t1));
	  fflush(stdout);
	}

      if(ThisTask == 0)
	{
	  t0 = second();

	  for(i = 0; i < NumPartGroup; i++)
	    {
	      P[i].origintask = origintask[0][i];
	      P[i].targettask = targettask[0][i];
	      P[i].submark = submark[0][i];
	    }

	  for(task = 1; task < NTask; task++)
	    {
	      MPI_Send(origintask[task], List_NumPartGroup[task], MPI_INT, task, TAG_FOF_C, MPI_COMM_WORLD);
	      MPI_Send(targettask[task], List_NumPartGroup[task], MPI_INT, task, TAG_FOF_A, MPI_COMM_WORLD);
	      MPI_Send(submark[task], List_NumPartGroup[task], MPI_INT, task, TAG_FOF_B, MPI_COMM_WORLD);
	    }

	  t1 = second();

	  printf("distributing marks (took %g)\n", timediff(t0, t1));
	  fflush(stdout);

	  for(task = NTask - 1; task >= 0; task--)
	    {
	      myfree(submark[task]);
	      myfree(targettask[task]);
	      myfree(origintask[task]);
	    }

	  myfree(submark);
	  myfree(targettask);
	  myfree(origintask);
	}
      else
	{
	  int *origin, *target, *mark;

	  origin = mymalloc("	  origin", NumPartGroup * sizeof(int));
	  target = mymalloc("	  target", NumPartGroup * sizeof(int));
	  mark = mymalloc("	  mark", NumPartGroup * sizeof(int));

	  MPI_Recv(origin, NumPartGroup, MPI_INT, 0, TAG_FOF_C, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	  MPI_Recv(target, NumPartGroup, MPI_INT, 0, TAG_FOF_A, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	  MPI_Recv(mark, NumPartGroup, MPI_INT, 0, TAG_FOF_B, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

	  for(i = 0; i < NumPartGroup; i++)
	    {
	      P[i].origintask = origin[i];
	      P[i].targettask = target[i];
	      P[i].submark = mark[i];
	    }

	  myfree(mark);
	  myfree(target);
	  myfree(origin);
	}


      t0 = second();
      subfind_distribute_particles(1);	/* assemble the particles on individual processors */
      t1 = second();

      if(ThisTask == 0)
	{
	  printf
	    ("independent subhalos are assembled on individual CPUs for unbinding (%g sec, (presently allocated=%g MB)\n",
	     timediff(t0, t1), AllocatedBytes / (1024.0 * 1024.0));
	  fflush(stdout);
	}

      qsort(P, NumPart, sizeof(struct particle_data), subfind_compare_P_submark);	/* groups particles of the same canidate together */

      MPI_Barrier(MPI_COMM_WORLD);
      t0 = second();

      if(ThisTask == 0)
	{
	  loc_count_cand = count_cand[0];
	  loc_candidates = candidates[0];
	  for(task = 1; task < NTask; task++)
	    {
	      MPI_Send(&count_cand[task], 1, MPI_INT, task, TAG_N, MPI_COMM_WORLD);
	      MPI_Send(candidates[task], count_cand[task] * sizeof(struct cand_dat),
		       MPI_BYTE, task, TAG_FOF_C, MPI_COMM_WORLD);
	    }
	}
      else
	{
	  MPI_Recv(&loc_count_cand, 1, MPI_INT, 0, TAG_N, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	  loc_candidates = mymalloc("	  loc_candidates", loc_count_cand * sizeof(struct cand_dat));
	  MPI_Recv(loc_candidates, loc_count_cand * sizeof(struct cand_dat), MPI_BYTE, 0, TAG_FOF_C,
		   MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}

      subfind_unbind_independent_ones(loc_count_cand);

      if(ThisTask == 0)
	{
	  count_cand[0] = loc_count_cand;

	  for(task = 1; task < NTask; task++)
	    {
	      MPI_Send(&task, 1, MPI_INT, task, TAG_N, MPI_COMM_WORLD);
	      MPI_Recv(&count_cand[task], 1, MPI_INT, task, TAG_FOF_E, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	      MPI_Recv(candidates[task], count_cand[task] * sizeof(struct cand_dat),
		       MPI_BYTE, task, TAG_FOF_D, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	    }
	}
      else
	{
	  MPI_Recv(&i, 1, MPI_INT, 0, TAG_N, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	  MPI_Send(&loc_count_cand, 1, MPI_INT, 0, TAG_FOF_E, MPI_COMM_WORLD);
	  MPI_Send(loc_candidates, loc_count_cand * sizeof(struct cand_dat), MPI_BYTE, 0, TAG_FOF_D,
		   MPI_COMM_WORLD);
	  myfree(loc_candidates);
	}

      MPI_Barrier(MPI_COMM_WORLD);
      t1 = second();

      if(ThisTask == 0)
	{
	  printf("unbinding of independent ones took %g sec\n", timediff(t0, t1));
	  fflush(stdout);
	}

      for(i = 0; i < NumPart; i++)
	P[i].origintask &= (HIGHBIT - 1);	/* clear high bit if set */

      t0 = second();
      subfind_distribute_particles(2);	/* bring them back to their original processor */
      t1 = second();


      if(ThisTask == 0)
	printf("particles have returned to their original processor (%g sec, presently allocated %g MB)\n",
	       timediff(t0, t1), AllocatedBytes / (1024.0 * 1024.0));

      /* reestablish the original order */
      qsort(P, NumPart, sizeof(struct particle_data), subfind_compare_P_GrNrGrNr);


      /* now mark the bound particles */
      if(ThisTask == 0)
	{
	  for(i = 0; i < NumPartGroup; i++)
	    if(P[i].submark >= 0 && P[i].submark < nsubs)
	      Len[0][i] = P[i].submark;	/* we use this to flag bound parts of substructures */

	  for(task = 1; task < NTask; task++)
	    {
	      MPI_Send(&task, 1, MPI_INT, task, TAG_N, MPI_COMM_WORLD);

	      int *submark = mymalloc("	      int *submark", List_NumPartGroup[task] * sizeof(int));

	      MPI_Recv(submark, List_NumPartGroup[task], MPI_INT, task, TAG_FOF_E, MPI_COMM_WORLD,
		       MPI_STATUS_IGNORE);

	      for(i = 0; i < List_NumPartGroup[task]; i++)
		if(submark[i] >= 0 && submark[i] < nsubs)
		  Len[task][i] = submark[i];

	      myfree(submark);
	    }
	}
      else
	{
	  int *submark = mymalloc("	  int *submark", NumPartGroup * sizeof(int));

	  for(i = 0; i < NumPartGroup; i++)
	    submark[i] = P[i].submark;

	  MPI_Recv(&i, 1, MPI_INT, 0, TAG_N, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

	  MPI_Send(submark, NumPartGroup, MPI_INT, 0, TAG_FOF_E, MPI_COMM_WORLD);

	  myfree(submark);
	}

    }
  while(tot_count_leaves > 0);


  force_treebuild(NumPartGroup, NULL);	/* re construct the tree for the collective part */



  /**** now we do the collective unbinding of the subhalo candidates that contain other subhalo candidates ****/


  ud = mymalloc("ud", NumPartGroup * sizeof(struct unbind_data));

  t0 = second();

  if(ThisTask == 0)
    {
      List_LocalLen = mymalloc("List_LocalLen", NTask * sizeof(int));
      ud_list = mymalloc("ud_list", NTask * sizeof(struct unbind_data *));

      for(i = 0; i < NTask; i++)
	ud_list[i] = mymalloc("	ud_list[i]", List_NumPartGroup[i] * sizeof(struct unbind_data));

      for(task = 0, nr = 0; task < NTask; task++)
	{
	  for(k = 0; k < count_cand[task]; k++)
	    {
	      len = candidates[task][k].len;
	      nsubs = candidates[task][k].nsub;
	      parent = candidates[task][k].parent;	/* this is here actually the daughter count */

	      if(parent >= 0)
		{
		  printf("collective unbinding of nr=%d (%d) of length=%d ... ", nr, nremaining, (int) len);
		  fflush(stdout);

		  nr++;

		  for(i = 0; i < NTask; i++)
		    List_LocalLen[i] = 0;

		  for(i = 0, p = candidates[task][k].head; i < candidates[task][k].len; i++)
		    {
		      subfind_distlinklist_add_particle(p);
		      if(p < 0)
			{
			  printf("Bummer i=%d \n", i);
			  endrun(123);

			}
		      p = subfind_distlinklist_get_next(p);
		    }

		  /* inform the others */

		  for(i = 1; i < NTask; i++)
		    {
		      MPI_Send(&List_LocalLen[i], 1, MPI_INT, i, TAG_N, MPI_COMM_WORLD);
		      MPI_Send(ud_list[i], List_LocalLen[i] * sizeof(struct unbind_data),
			       MPI_BYTE, i, TAG_FOF_A, MPI_COMM_WORLD);
		    }
		  LocalLen = List_LocalLen[0];
		  memcpy(ud, ud_list[0], LocalLen * sizeof(struct unbind_data));

		  tt0 = second();

		  LocalLen = subfind_col_unbind(ud, LocalLen);
		  MPI_Allreduce(&LocalLen, &len, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

		  tt1 = second();

		  if(ThisTask == 0)
		    {
		      printf("took %g sec\n", timediff(tt0, tt1));
		      fflush(stdout);
		    }

		  if(len >= All.DesLinkNgb)
		    {
		      /* ok, we found a substructure */

		      List_LocalLen[0] = LocalLen;
		      memcpy(ud_list[0], ud, LocalLen * sizeof(struct unbind_data));

		      for(i = 1; i < NTask; i++)
			{
			  MPI_Recv(&List_LocalLen[i], 1, MPI_INT, i, TAG_HEADER, MPI_COMM_WORLD,
				   MPI_STATUS_IGNORE);
			  MPI_Recv(ud_list[i], List_LocalLen[i] * sizeof(struct unbind_data), MPI_BYTE, i,
				   TAG_FOF_B, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			}

		      for(i = 0; i < NTask; i++)
			for(j = 0; j < List_LocalLen[i]; j++)
			  Len[i][ud_list[i][j].index] = nsubs;	/* we use this to flag the substructures */

		      candidates[task][k].bound_length = len;
		    }
		  else
		    {
		      candidates[task][k].bound_length = 0;
		    }
		}
	    }
	}

      parent = -1;
      /* inform the others that we are done */

      for(i = 1; i < NTask; i++)
	MPI_Send(&parent, 1, MPI_INT, i, TAG_N, MPI_COMM_WORLD);

      for(i = NTask - 1; i >= 0; i--)
	myfree(ud_list[i]);

      myfree(ud_list);

      myfree(List_LocalLen);
    }
  else
    {
      do
	{
	  MPI_Recv(&LocalLen, 1, MPI_INT, 0, TAG_N, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

	  if(LocalLen >= 0)
	    {
	      MPI_Recv(ud, LocalLen * sizeof(struct unbind_data),
		       MPI_BYTE, 0, TAG_FOF_A, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

	      LocalLen = subfind_col_unbind(ud, LocalLen);

	      MPI_Allreduce(&LocalLen, &len, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

	      if(len >= All.DesLinkNgb)
		{
		  MPI_Send(&LocalLen, 1, MPI_INT, 0, TAG_HEADER, MPI_COMM_WORLD);
		  MPI_Send(ud, LocalLen * sizeof(struct unbind_data), MPI_BYTE, 0, TAG_FOF_B, MPI_COMM_WORLD);
		}
	    }
	}
      while(LocalLen >= 0);
    }

  t1 = second();
  if(ThisTask == 0)
    printf("\nthe collective unbinding of remaining halos took %g sec\n", timediff(t0, t1));



  /********************/



  if(ThisTask == 0)
    {
      for(task = 0, countall = 0; task < NTask; task++)
	for(k = 0; k < count_cand[task]; k++)
	  if(candidates[task][k].bound_length >= All.DesLinkNgb)
	    countall++;

      printf("\nfound %d bound substructures in FoF group of length %d\n", countall, totgrouplen1);
      fflush(stdout);
    }

  MPI_Bcast(&countall, 1, MPI_INT, 0, MPI_COMM_WORLD);



  /* now determine the parent subhalo for each candidate */

  t0 = second();


  if(ThisTask == 0)
    {
      tmp_candidates = mymalloc("tmp_candidates", totcand * sizeof(struct cand_dat));

      for(task = 0, j = 0; task < NTask; task++)
	{
	  for(i = 0; i < count_cand[task]; i++)
	    tmp_candidates[j++] = candidates[task][i];
	}

      qsort(tmp_candidates, totcand, sizeof(struct cand_dat), subfind_compare_candidates_boundlength);

      for(k = 0; k < totcand; k++)
	{
	  tmp_candidates[k].subnr = k;
	  tmp_candidates[k].parent = 0;
	}

      qsort(tmp_candidates, totcand, sizeof(struct cand_dat), subfind_compare_candidates_rank);

      for(k = 0; k < totcand; k++)
	{
	  for(j = k + 1; j < totcand; j++)
	    {
	      if(tmp_candidates[j].rank > tmp_candidates[k].rank + tmp_candidates[k].len)
		break;

	      if(tmp_candidates[k].rank + tmp_candidates[k].len >=
		 tmp_candidates[j].rank + tmp_candidates[j].len)
		{
		  if(tmp_candidates[k].bound_length >= All.DesLinkNgb)
		    tmp_candidates[j].parent = tmp_candidates[k].subnr;
		}
	      else
		{
		  printf("k=%d|%d has rank=%d and len=%d.  j=%d has rank=%d and len=%d bound=%d\n",
			 k, countall, (int) tmp_candidates[k].rank, (int) tmp_candidates[k].len,
			 (int) tmp_candidates[k].bound_length, (int) tmp_candidates[j].rank,
			 (int) tmp_candidates[j].len, (int) tmp_candidates[j].bound_length);
		  endrun(1212313);
		}
	    }
	}

      qsort(tmp_candidates, totcand, sizeof(struct cand_dat), subfind_compare_candidates_subnr);


      for(task = 0, j = 0; task < NTask; task++)
	{
	  for(i = 0; i < count_cand[task]; i++)
	    candidates[task][i] = tmp_candidates[j++];
	}

      myfree(tmp_candidates);
    }

  t1 = second();
  if(ThisTask == 0)
    printf("determination of parent subhalo took %g sec (presently allocated %g MB)\n", timediff(t0, t1),
	   AllocatedBytes / (1024.0 * 1024.0));


  /* Now let's save  some properties of the substructures */
  if(ThisTask == ((GrNr - 1) % NTask))
    {
      Group[grindex].Nsubs = countall;
    }


  /************** determine properties ***************/



  t0 = second();

  if(ThisTask == 0)
    {
      List_LocalLen = mymalloc("List_LocalLen", NTask * sizeof(int));
      ud_list = mymalloc("ud_list", NTask * sizeof(struct unbind_data *));

      for(i = 0; i < NTask; i++)
	ud_list[i] = mymalloc("	ud_list[i]", List_NumPartGroup[i] * sizeof(struct unbind_data));


      for(task = 0, subnr = 0; task < NTask; task++)
	{
	  /*
	     printf("task=%d...", task); fflush(stdout);
	   */

	  /*      double ttt0 = second();
	   */

	  for(k = 0; k < count_cand[task]; k++)
	    {
	      len = candidates[task][k].bound_length;
	      nsubs = candidates[task][k].nsub;
	      parent = candidates[task][k].parent;

	      if(len > 0)
		{
		  for(i = 0; i < NTask; i++)
		    List_LocalLen[i] = 0;

		  for(i = 0, p = candidates[task][k].head; i < candidates[task][k].len; i++)
		    {
		      subfind_distlinklist_add_bound_particles(p, nsubs);
		      p = subfind_distlinklist_get_next(p);
		    }

		  /* inform the others */

		  for(i = 1; i < NTask; i++)
		    {
		      MPI_Send(&List_LocalLen[i], 1, MPI_INT, i, TAG_N, MPI_COMM_WORLD);
		      MPI_Send(&len, 1, MPI_INT, i, TAG_HMAX, MPI_COMM_WORLD);
		      MPI_Send(&parent, 1, MPI_INT, i, TAG_SPHDATA, MPI_COMM_WORLD);
		      MPI_Send(ud_list[i], List_LocalLen[i] * sizeof(struct unbind_data),
			       MPI_BYTE, i, TAG_FOF_A, MPI_COMM_WORLD);
		    }
		  LocalLen = List_LocalLen[0];
		  memcpy(ud, ud_list[0], LocalLen * sizeof(struct unbind_data));

		  tt0 = second();
		  subfind_col_determine_sub_halo_properties(ud, LocalLen, &SubMass,
							    &SubPos[0], &SubVel[0], &SubCM[0], &SubVelDisp,
							    &SubVmax, &SubVmaxRad, &SubSpin[0],
							    &SubMostBoundID, &SubHalfMass, &SubMassTab[0]);
		  tt1 = second();

		  if(ThisTask == 0 && timediff(tt0, tt1) > 10.0)
		    {
		      printf("  determining properties of halo of len=%d took %g sec\n", len,
			     timediff(tt0, tt1));
		      fflush(stdout);
		    }

		  /* we have filled into ud the binding energy and the particle ID return */

		  if(((GrNr - 1) % NTask) == ThisTask)
		    {
		      if(Nsubgroups >= MaxNsubgroups)
			endrun(899);

		      if(subnr == 0)
			{
			  for(j = 0; j < 3; j++)
			    Group[grindex].Pos[j] = SubPos[j];
			}

		      SubGroup[Nsubgroups].Len = len;
		      if(subnr == 0)
			SubGroup[Nsubgroups].Offset = Group[grindex].Offset;
		      else
			SubGroup[Nsubgroups].Offset =
			  SubGroup[Nsubgroups - 1].Offset + SubGroup[Nsubgroups - 1].Len;
		      SubGroup[Nsubgroups].GrNr = GrNr - 1;
		      SubGroup[Nsubgroups].SubNr = subnr;
		      SubGroup[Nsubgroups].SubParent = parent;
		      SubGroup[Nsubgroups].Mass = SubMass;
		      SubGroup[Nsubgroups].SubMostBoundID = SubMostBoundID;
		      SubGroup[Nsubgroups].SubVelDisp = SubVelDisp;
		      SubGroup[Nsubgroups].SubVmax = SubVmax;
		      SubGroup[Nsubgroups].SubVmaxRad = SubVmaxRad;
		      SubGroup[Nsubgroups].SubHalfMass = SubHalfMass;

		      for(j = 0; j < 3; j++)
			{
			  SubGroup[Nsubgroups].Pos[j] = SubPos[j];
			  SubGroup[Nsubgroups].Vel[j] = SubVel[j];
			  SubGroup[Nsubgroups].CM[j] = SubCM[j];
			  SubGroup[Nsubgroups].Spin[j] = SubSpin[j];
			}

#ifdef SAVE_MASS_TAB
		      for(j = 0; j < 6; j++)
			SubGroup[Nsubgroups].MassTab[j] = SubMassTab[j];
#endif

		      Nsubgroups++;
		    }

		  /* Let's now assign the subgroup number */

		  for(i = 0; i < LocalLen; i++)
		    {
		      P[ud[i].index].SubNr = subnr;
		    }

		  subnr++;
		}
	    }

	  /*
	     double ttt1 = second();
	     printf("  took %g sec\n", timediff(ttt0, ttt1));
	     fflush(stdout);
	   */
	}

      parent = -1;
      /* inform the others that we are done */

      for(i = 1; i < NTask; i++)
	MPI_Send(&parent, 1, MPI_INT, i, TAG_N, MPI_COMM_WORLD);

      for(i = NTask - 1; i >= 0; i--)
	myfree(ud_list[i]);

      myfree(ud_list);

      myfree(List_LocalLen);
    }
  else
    {
      subnr = 0;
      do
	{
	  MPI_Recv(&LocalLen, 1, MPI_INT, 0, TAG_N, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

	  if(LocalLen >= 0)
	    {
	      MPI_Recv(&len, 1, MPI_INT, 0, TAG_HMAX, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	      MPI_Recv(&parent, 1, MPI_INT, 0, TAG_SPHDATA, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

	      MPI_Recv(ud, LocalLen * sizeof(struct unbind_data),
		       MPI_BYTE, 0, TAG_FOF_A, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

	      subfind_col_determine_sub_halo_properties(ud, LocalLen, &SubMass,
							&SubPos[0], &SubVel[0], &SubCM[0], &SubVelDisp,
							&SubVmax, &SubVmaxRad, &SubSpin[0], &SubMostBoundID,
							&SubHalfMass, &SubMassTab[0]);
	      if(((GrNr - 1) % NTask) == ThisTask)
		{
		  if(Nsubgroups >= MaxNsubgroups)
		    endrun(899);

		  if(subnr == 0)
		    {
		      for(j = 0; j < 3; j++)
			Group[grindex].Pos[j] = SubPos[j];
		    }

		  SubGroup[Nsubgroups].Len = len;
		  if(subnr == 0)
		    SubGroup[Nsubgroups].Offset = Group[grindex].Offset;
		  else
		    SubGroup[Nsubgroups].Offset =
		      SubGroup[Nsubgroups - 1].Offset + SubGroup[Nsubgroups - 1].Len;
		  SubGroup[Nsubgroups].GrNr = GrNr - 1;
		  SubGroup[Nsubgroups].SubNr = subnr;
		  SubGroup[Nsubgroups].SubParent = parent;
		  SubGroup[Nsubgroups].Mass = SubMass;
		  SubGroup[Nsubgroups].SubMostBoundID = SubMostBoundID;
		  SubGroup[Nsubgroups].SubVelDisp = SubVelDisp;
		  SubGroup[Nsubgroups].SubVmax = SubVmax;
		  SubGroup[Nsubgroups].SubVmaxRad = SubVmaxRad;
		  SubGroup[Nsubgroups].SubHalfMass = SubHalfMass;

		  for(j = 0; j < 3; j++)
		    {
		      SubGroup[Nsubgroups].Pos[j] = SubPos[j];
		      SubGroup[Nsubgroups].Vel[j] = SubVel[j];
		      SubGroup[Nsubgroups].CM[j] = SubCM[j];
		      SubGroup[Nsubgroups].Spin[j] = SubSpin[j];
		    }

#ifdef SAVE_MASS_TAB
		  for(j = 0; j < 6; j++)
		    SubGroup[Nsubgroups].MassTab[j] = SubMassTab[j];
#endif

		  Nsubgroups++;
		}

	      /* Let's now assign the subgroup number */

	      for(i = 0; i < LocalLen; i++)
		{
		  P[ud[i].index].SubNr = subnr;
		}

	      subnr++;
	    }
	}
      while(LocalLen >= 0);
    }

  t1 = second();
  if(ThisTask == 0)
    printf("determining substructure properties took %g sec (presently allocated %g MB)\n", timediff(t0, t1),
	   AllocatedBytes / (1024.0 * 1024.0));

  myfree(ud);


  if(ThisTask == 0)
    {
      for(i = NTask - 1; i >= 0; i--)
	{
	  myfree(Len[i]);
	  myfree(Next[i]);
	}

      myfree(Len);
      myfree(Next);

      for(i = NTask - 1; i >= 0; i--)
	myfree(candidates[i]);

      myfree(candidates);
      myfree(count_cand);
    }

  myfree(Offset_NumPartGroup);
  myfree(List_NumPartGroup);

  force_treefree();
  domain_free();
  domain_allocate_trick();

  qsort(P, NumPart, sizeof(struct particle_data), subfind_compare_P_origindex);	/* reorder them such that the gas particles match again */

  for(i = 0; i < NumPart; i++)
    if(P[i].origindex != i)
      endrun(7777);


  ttt1 = second();

  if(ThisTask == 0)
    printf("\ncollective processing of halo done. Took in total %g sec\n", timediff(ttt0, ttt1));

}




void subfind_unbind_independent_ones(int count_cand)
{
  int i, j, k, len, nsubs;

  /*subfind_loctree_treeallocate(All.TreeAllocFactor * NumPart, NumPart); */

  R2list = mymalloc("R2list", NumPart * sizeof(struct r2data));
  ud = mymalloc("ud", NumPart * sizeof(struct unbind_data));

  qsort(loc_candidates, count_cand, sizeof(struct cand_dat), subfind_compare_candidates_nsubs);

  for(k = 0, i = 0; k < count_cand; k++)
    if(loc_candidates[k].parent == 0)
      {
	while(P[i].submark < loc_candidates[k].nsub)
	  {
	    i++;
	    if(i >= NumPart)
	      endrun(13213);
	  }

	if(P[i].submark >= 0 && P[i].submark < HIGHBIT)
	  {
	    len = 0;
	    nsubs = P[i].submark;

	    if(nsubs != loc_candidates[k].nsub)
	      {
		printf("TASK=%d i=%d k=%d nsubs=%d loc_candidates[k].nsub=%d\n",
		       ThisTask, i, k, nsubs, loc_candidates[k].nsub);
		endrun(13199);
	      }

	    while(i < NumPart)
	      {
		if(P[i].submark == nsubs)
		  {
		    P[i].submark = HIGHBIT;
		    if((P[i].origintask & HIGHBIT) == 0)
		      {
			ud[len].index = i;
			len++;
		      }
		    i++;
		  }
		else
		  break;
	      }

	    len = subfind_unbind(ud, len);

	    if(len >= All.DesLinkNgb)
	      {
		/* ok, we found a substructure */
		loc_candidates[k].bound_length = len;

		for(j = 0; j < len; j++)
		  P[ud[j].index].submark = nsubs;	/* we use this to flag the substructures */
	      }
	    else
	      loc_candidates[k].bound_length = 0;
	  }

	loc_candidates[k].parent = -1;
      }

  myfree(ud);
  myfree(R2list);
}





int subfind_mark_independent_ones(void)
{
  int nsubs, task, k, i, len, parent;
  long long p;

  nsubs = 0;

  for(task = 0; task < NTask; task++)
    {
      for(k = 0; k < count_cand[task]; k++)
	{
	  len = candidates[task][k].len;
	  parent = candidates[task][k].parent;	/* this is here actually the daughter count */

	  if(parent == 0)
	    {
	      for(i = 0, p = candidates[task][k].head; i < len; i++)
		{
		  subfind_distlinklist_mark_particle(p, task, nsubs);

		  p = subfind_distlinklist_get_next(p);
		}
	    }

	  nsubs++;
	}
    }
  return nsubs;
}









int subfind_col_unbind(struct unbind_data *d, int num)
{
  int iter = 0;
  int i, j, p, part_index, minindex, task;
  int unbound, totunbound, numleft, mincpu;
  int *npart, *offset, *nbu_count, count_bound_unbound, phaseflag;
  double s[3], dx[3], ddxx, v[3], dv[3], sloc[3], vloc[3], pos[3];
  double vel_to_phys, H_of_a, atime;
  MyFloat minpot, *potlist;
  double boxsize, boxhalf;
  double mass, massloc, t0, t1;
  double *bnd_energy, energy_limit, energy_limit_local, weakly_bound_limit_local, weakly_bound_limit = 0;
  double mbs, glob_mbs;

  boxsize = All.BoxSize;
  boxhalf = 0.5 * All.BoxSize;

  if(All.ComovingIntegrationOn)
    {
      vel_to_phys = 1.0 / All.Time;
      H_of_a = hubble_function(All.Time);
      atime = All.Time;
    }
  else
    {
      H_of_a = 0;
      atime = vel_to_phys = 1;
    }

  phaseflag = 0;		/* this means we will recompute the potential for all particles */


  mbs = AllocatedBytes / (1024.0 * 1024.0);
  MPI_Allreduce(&mbs, &glob_mbs, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

  if(ThisTask == 0)
    printf("maximum alloacted %g MB\n", glob_mbs);

  do
    {
      t0 = second();

      force_treebuild(num, d);

      /* let's compute the potential energy */

      subfind_potential_compute(num, d, phaseflag, weakly_bound_limit);

      if(phaseflag == 0)
	{
	  potlist = (MyFloat *) mymalloc("	  potlist", NTask * sizeof(MyFloat));

	  for(i = 0, minindex = -1, minpot = 1.0e30; i < num; i++)
	    {
	      if(P[d[i].index].u.DM_Potential < minpot || minindex == -1)
		{
		  minpot = P[d[i].index].u.DM_Potential;
		  minindex = d[i].index;
		}
	    }

	  MPI_Allgather(&minpot, sizeof(MyFloat), MPI_BYTE, potlist, sizeof(MyFloat), MPI_BYTE,
			MPI_COMM_WORLD);

	  for(i = 0, mincpu = -1, minpot = 1.0e30; i < NTask; i++)
	    if(potlist[i] < minpot)
	      {
		mincpu = i;
		minpot = potlist[i];
	      }

	  if(mincpu < 0)
	    endrun(112);

	  myfree(potlist);

	  if(ThisTask == mincpu)
	    {
	      for(j = 0; j < 3; j++)
		pos[j] = P[minindex].Pos[j];
	    }

	  MPI_Bcast(&pos[0], 3, MPI_DOUBLE, mincpu, MPI_COMM_WORLD);
	  /* pos[] now holds the position of minimum potential */
	  /* we take that as the center */
	}

      /* let's get bulk velocity and the center-of-mass */

      for(j = 0; j < 3; j++)
	sloc[j] = vloc[j] = 0;

      for(i = 0, massloc = 0; i < num; i++)
	{
	  part_index = d[i].index;

	  for(j = 0; j < 3; j++)
	    {
#ifdef PERIODIC
	      ddxx = NEAREST(P[part_index].Pos[j] - pos[j]);
#else
	      ddxx = P[part_index].Pos[j] - pos[j];
#endif
	      sloc[j] += P[part_index].Mass * ddxx;
	      vloc[j] += P[part_index].Mass * P[part_index].Vel[j];
	    }
	  massloc += P[part_index].Mass;
	}

      MPI_Allreduce(sloc, s, 3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(vloc, v, 3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(&massloc, &mass, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

      for(j = 0; j < 3; j++)
	{
	  s[j] /= mass;		/* center of mass */
	  v[j] /= mass;

	  s[j] += pos[j];

#ifdef PERIODIC
	  while(s[j] < 0)
	    s[j] += boxsize;
	  while(s[j] >= boxsize)
	    s[j] -= boxsize;
#endif
	}

      bnd_energy = mymalloc("bnd_energy", num * sizeof(double));

      for(i = 0; i < num; i++)
	{
	  part_index = d[i].index;

	  for(j = 0; j < 3; j++)
	    {
	      dv[j] = vel_to_phys * (P[part_index].Vel[j] - v[j]);
#ifdef PERIODIC
	      dx[j] = atime * NEAREST(P[part_index].Pos[j] - s[j]);
#else
	      dx[j] = atime * (P[part_index].Pos[j] - s[j]);
#endif
	      dv[j] += H_of_a * dx[j];
	    }

	  P[part_index].v.DM_BindingEnergy =
	    P[part_index].u.DM_Potential + 0.5 * (dv[0] * dv[0] + dv[1] * dv[1] + dv[2] * dv[2]);

#ifdef DENSITY_SPLIT_BY_TYPE
	  if(P[part_index].Type == 0)
	    P[part_index].v.DM_BindingEnergy += P[part_index].w.int_energy;
#endif
	  bnd_energy[i] = P[part_index].v.DM_BindingEnergy;
	}

      parallel_sort(bnd_energy, num, sizeof(double), subfind_compare_binding_energy);


      npart = mymalloc("npart", NTask * sizeof(int));
      nbu_count = mymalloc("nbu_count", NTask * sizeof(int));
      offset = mymalloc("offset", NTask * sizeof(int));

      MPI_Allgather(&num, 1, MPI_INT, npart, 1, MPI_INT, MPI_COMM_WORLD);
      MPI_Allreduce(&num, &numleft, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

      for(i = 1, offset[0] = 0; i < NTask; i++)
	offset[i] = offset[i - 1] + npart[i - 1];

      j = (int) (0.25 * numleft);	/* index of limiting energy value */

      task = 0;
      while(j >= npart[task])
	{
	  j -= npart[task];
	  task++;
	}

      if(ThisTask == task)
	energy_limit_local = bnd_energy[j];
      else
	energy_limit_local = 1.0e30;

      MPI_Allreduce(&energy_limit_local, &energy_limit, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);

      for(i = 0, count_bound_unbound = 0; i < num; i++)
	{
	  if(bnd_energy[i] > 0)
	    count_bound_unbound++;
	  else
	    count_bound_unbound--;
	}

      MPI_Allgather(&count_bound_unbound, 1, MPI_INT, nbu_count, 1, MPI_INT, MPI_COMM_WORLD);

      for(i = 0, count_bound_unbound = 0; i < ThisTask; i++)
	count_bound_unbound += nbu_count[i];

      for(i = 0; i < num - 1; i++)
	{
	  if(bnd_energy[i] > 0)
	    count_bound_unbound++;
	  else
	    count_bound_unbound--;
	  if(count_bound_unbound <= 0)
	    break;
	}

      if(num > 0 && count_bound_unbound <= 0)
	weakly_bound_limit_local = bnd_energy[i];
      else
	weakly_bound_limit_local = -1.0e30;

      MPI_Allreduce(&weakly_bound_limit_local, &weakly_bound_limit, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

      for(i = 0, unbound = 0; i < num; i++)
	{
	  p = d[i].index;

	  if(P[p].v.DM_BindingEnergy > 0 && P[p].v.DM_BindingEnergy > energy_limit)
	    {
	      unbound++;

	      d[i] = d[num - 1];
	      num--;
	      i--;
	    }
	}

      myfree(offset);
      myfree(nbu_count);
      myfree(npart);
      myfree(bnd_energy);

      MPI_Allreduce(&unbound, &totunbound, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(&num, &numleft, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

      t1 = second();

      /*
         if(ThisTask == 0)
         printf("iter=%d phaseflag=%d unbound=%d numleft=%d  (took %g sec)\n", iter, phaseflag, totunbound,
         numleft, timediff(t0, t1));
       */

      if(phaseflag == 0)
	{
	  if(totunbound > 0)
	    phaseflag = 1;
	}
      else
	{
	  if(totunbound == 0)
	    {
	      phaseflag = 0;	/* this will make us repeat everything once more for all particles */
	      totunbound = 1;
	    }
	}

      iter++;
    }
  while(totunbound > 0 && numleft >= All.DesLinkNgb);

  return num;
}




void subfind_col_determine_sub_halo_properties(struct unbind_data *d, int num, double *totmass,
					       double *pos, double *vel, double *cm, double *veldisp,
					       double *vmax, double *vmaxrad, double *spin,
					       MyIDType * mostboundid, double *halfmassrad, double *mass_tab)
{
  int i, j, part_index, *npart, numtot, nhalf, offset;
  MyIDType mbid;
  double s[3], sloc[3], v[3], vloc[3], max, vel_to_phys, H_of_a, atime;
  double lx, ly, lz, dv[3], dx[3], disp;
  double loclx, locly, loclz, locdisp;
  double boxhalf, boxsize, ddxx;
  sort_r2list *loc_rr_list;
  int minindex, mincpu;
  double mass, mass_tab_loc[6], maxrad, massloc, *masslist;
  MyFloat minpot, *potlist;


  boxsize = All.BoxSize;
  boxhalf = 0.5 * boxsize;

  if(All.ComovingIntegrationOn)
    {
      vel_to_phys = 1.0 / All.Time;
      H_of_a = hubble_function(All.Time);
      atime = All.Time;
    }
  else
    {
      H_of_a = 0;
      atime = vel_to_phys = 1;
    }

  potlist = (MyFloat *) mymalloc("potlist", NTask * sizeof(MyFloat));

  for(i = 0, minindex = -1, minpot = 1.0e30; i < num; i++)
    {
      if(P[d[i].index].u.DM_Potential < minpot || minindex == -1)
	{
	  minpot = P[d[i].index].u.DM_Potential;
	  minindex = d[i].index;
	}
    }

  MPI_Allgather(&minpot, sizeof(MyFloat), MPI_BYTE, potlist, sizeof(MyFloat), MPI_BYTE, MPI_COMM_WORLD);

  for(i = 0, mincpu = -1, minpot = 1.0e30; i < NTask; i++)
    if(potlist[i] < minpot)
      {
	mincpu = i;
	minpot = potlist[i];
      }

  if(mincpu < 0)
    {
      printf("ta=%d num=%d\n", ThisTask, num);
      endrun(121);
    }

  if(ThisTask == mincpu)
    {
      for(j = 0; j < 3; j++)
	s[j] = P[minindex].Pos[j];
    }

  MPI_Bcast(&s[0], 3, MPI_DOUBLE, mincpu, MPI_COMM_WORLD);

  /* s[] now holds the position of minimum potential */
  /* we take that as the center */
  for(j = 0; j < 3; j++)
    pos[j] = s[j];


  /* the ID of the most bound particle, we take the minimum binding energy */
  for(i = 0, minindex = -1, minpot = 1.0e30; i < num; i++)
    {
      if(P[d[i].index].v.DM_BindingEnergy < minpot || minindex == -1)
	{
	  minpot = P[d[i].index].v.DM_BindingEnergy;
	  minindex = d[i].index;
	}
    }

  MPI_Allgather(&minpot, sizeof(MyFloat), MPI_BYTE, potlist, sizeof(MyFloat), MPI_BYTE, MPI_COMM_WORLD);

  for(i = 0, minpot = 1.0e30; i < NTask; i++)
    if(potlist[i] < minpot)
      {
	mincpu = i;
	minpot = potlist[i];
      }

  if(ThisTask == mincpu)
    mbid = P[minindex].ID;

  MPI_Bcast(&mbid, sizeof(mbid), MPI_BYTE, mincpu, MPI_COMM_WORLD);

  myfree(potlist);

  *mostboundid = mbid;

  /* let's get bulk velocity and the center-of-mass */

  for(j = 0; j < 3; j++)
    sloc[j] = vloc[j] = 0;

  for(j = 0; j < 6; j++)
    mass_tab_loc[j] = 0;

  for(i = 0, massloc = 0; i < num; i++)
    {
      part_index = d[i].index;

      for(j = 0; j < 3; j++)
	{
#ifdef PERIODIC
	  ddxx = NEAREST(P[part_index].Pos[j] - pos[j]);
#else
	  ddxx = P[part_index].Pos[j] - pos[j];
#endif
	  sloc[j] += P[part_index].Mass * ddxx;
	  vloc[j] += P[part_index].Mass * P[part_index].Vel[j];
	}
      massloc += P[part_index].Mass;

      mass_tab_loc[P[part_index].Type] += P[part_index].Mass;
    }

  MPI_Allreduce(sloc, s, 3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(vloc, v, 3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&massloc, &mass, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(mass_tab_loc, mass_tab, 6, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  *totmass = mass;

  for(j = 0; j < 3; j++)
    {
      s[j] /= mass;		/* center of mass */
      v[j] /= mass;

      vel[j] = vel_to_phys * v[j];

      s[j] += pos[j];

#ifdef PERIODIC
      while(s[j] < 0)
	s[j] += boxsize;
      while(s[j] >= boxsize)
	s[j] -= boxsize;
#endif

      cm[j] = s[j];
    }


  locdisp = loclx = locly = loclz = 0;

  loc_rr_list = mymalloc("loc_rr_list", sizeof(sort_r2list) * num);

  for(i = 0, massloc = 0; i < num; i++)
    {
      part_index = d[i].index;

      loc_rr_list[i].r = 0;
      loc_rr_list[i].mass = P[part_index].Mass;

      for(j = 0; j < 3; j++)
	{
#ifdef PERIODIC
	  ddxx = NEAREST(P[part_index].Pos[j] - s[j]);
#else
	  ddxx = P[part_index].Pos[j] - s[j];
#endif
	  dx[j] = atime * ddxx;
	  dv[j] = vel_to_phys * (P[part_index].Vel[j] - v[j]);
	  dv[j] += H_of_a * dx[j];

	  locdisp += P[part_index].Mass * dv[j] * dv[j];
	  /* for rotation curve computation, take minimum of potential as center */
#ifdef PERIODIC
	  ddxx = NEAREST(P[part_index].Pos[j] - pos[j]);
#else
	  ddxx = P[part_index].Pos[j] - pos[j];
#endif
	  ddxx = atime * ddxx;
	  loc_rr_list[i].r += ddxx * ddxx;
	}

      loclx += P[part_index].Mass * (dx[1] * dv[2] - dx[2] * dv[1]);
      locly += P[part_index].Mass * (dx[2] * dv[0] - dx[0] * dv[2]);
      loclz += P[part_index].Mass * (dx[0] * dv[1] - dx[1] * dv[0]);

      loc_rr_list[i].r = sqrt(loc_rr_list[i].r);
    }

  MPI_Allreduce(&loclx, &lx, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&locly, &ly, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&loclz, &lz, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&locdisp, &disp, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  *veldisp = sqrt(disp / (3 * mass));	/* convert to 1d velocity dispersion */

  spin[0] = lx / mass;
  spin[1] = ly / mass;
  spin[2] = lz / mass;


  npart = (int *) mymalloc("npart", NTask * sizeof(int));

  MPI_Allgather(&num, 1, MPI_INT, npart, 1, MPI_INT, MPI_COMM_WORLD);

  for(i = 0, numtot = 0; i < NTask; i++)
    numtot += npart[i];

  parallel_sort(loc_rr_list, num, sizeof(sort_r2list), subfind_compare_dist_rotcurve);

  nhalf = numtot / 2;
  mincpu = 0;

  while(nhalf >= npart[mincpu])
    {
      nhalf -= npart[mincpu];
      mincpu++;
    }

  if(ThisTask == mincpu)
    *halfmassrad = loc_rr_list[nhalf].r;

  MPI_Bcast(halfmassrad, sizeof(double), MPI_BYTE, mincpu, MPI_COMM_WORLD);


  /* compute cumulative mass */

  masslist = (double *) mymalloc("masslist", NTask * sizeof(double));

  for(i = 0, massloc = 0; i < num; i++)
    massloc += loc_rr_list[i].mass;

  MPI_Allgather(&massloc, 1, MPI_DOUBLE, masslist, 1, MPI_DOUBLE, MPI_COMM_WORLD);

  for(i = 1; i < NTask; i++)
    masslist[i] += masslist[i - 1];

  for(i = 1; i < num; i++)
    loc_rr_list[i].mass += loc_rr_list[i - 1].mass;

  if(ThisTask > 0)
    for(i = 0; i < num; i++)
      loc_rr_list[i].mass += masslist[ThisTask - 1];

  for(i = 0, offset = 0; i < ThisTask; i++)
    offset += npart[i];

  for(i = num - 1, max = 0, maxrad = 0; i + offset > 5 && i >= 0; i--)
    if(loc_rr_list[i].mass / loc_rr_list[i].r > max)
      {
	max = loc_rr_list[i].mass / loc_rr_list[i].r;
	maxrad = loc_rr_list[i].r;
      }

  MPI_Allreduce(&max, vmax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

  if(max < *vmax)
    maxrad = 0;

  *vmax = sqrt(All.G * (*vmax));

  MPI_Allreduce(&maxrad, &max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

  *vmaxrad = max;


  myfree(masslist);
  myfree(npart);
  myfree(loc_rr_list);
}





void subfind_distlinklist_add_particle(long long index)
{
  int task, i;

  task = (index >> 32);
  i = (index & MASK);

  if(Len[task][i] < 0)		/* consider only particles not already in substructures */
    {
      ud_list[task][List_LocalLen[task]].index = i;
      List_LocalLen[task]++;
    }
}

void subfind_distlinklist_mark_particle(long long index, int target, int mark)
{
  int task, i;

  task = (index >> 32);
  i = (index & MASK);

  if(submark[task][i] != HIGHBIT)
    {
      printf("Task=%d i=%d submark[task][i]=%d?\n", task, i, submark[task][i]);
      endrun(131);
    }

  targettask[task][i] = target;
  submark[task][i] = mark;
}


void subfind_distlinklist_add_bound_particles(long long index, int nsub)
{
  int task, i;

  task = (index >> 32);
  i = (index & MASK);

  if(Len[task][i] == nsub)	/* consider only particles not already in substructures */
    {
      ud_list[task][List_LocalLen[task]].index = i;
      List_LocalLen[task]++;
    }
}


void subfind_col_find_candidates(int totgrouplen)
{
  int ngbcount, retcode, len_attach;
  int i, k, len, task, off;
  long long prev, tail, tail_attach, tmp, next, index;
  long long p, ss, head, head_attach, ngb_index1, ngb_index2, rank;
  double t0, t1;

  /* now find the subhalo candidates by building up link lists from high density to low density */

  for(task = 0; task < NTask; task++)
    {
      /*      printf("begin task=%d...", task); fflush(stdout); */
      t0 = second();

      for(k = 0; k < List_NumPartGroup[task]; k++)
	{
	  /*      if((k % 1000) == 0)
	     printf("k=%d|%d\n", k,  List_NumPartGroup[task]);
	   */

	  off = Offset_NumPartGroup[task] + k;

	  ngbcount = sd[off].ngbcount;
	  ngb_index1 = sd[off].ngb_index1;
	  ngb_index2 = sd[off].ngb_index2;

	  switch (ngbcount)	/* treat the different possible cases */
	    {
	    case 0:		/* this appears to be a lonely maximum -> new group */
	      subfind_distlinklist_set_all(sd[off].index, sd[off].index, sd[off].index, 1, -1);
	      break;

	    case 1:		/* the particle is attached to exactly one group */
	      head = subfind_distlinklist_get_head(ngb_index1);

	      if(head == -1)
		{
		  printf("We have a problem!  head=%d/%d for k=%d on task=%d\n",
			 (int) (head >> 32), (int) head, k, task);
		  fflush(stdout);
		  endrun(13123);
		}

	      retcode = subfind_distlinklist_get_tail_set_tail_increaselen(head, &tail, sd[off].index);

	      if(!(retcode & 1))
		subfind_distlinklist_set_headandnext(sd[off].index, head, -1);
	      if(!(retcode & 2))
		subfind_distlinklist_set_next(tail, sd[off].index);
	      break;

	    case 2:		/* the particle merges two groups together */
	      if((ngb_index1 >> 32) == (ngb_index2 >> 32))
		{
		  subfind_distlinklist_get_two_heads(ngb_index1, ngb_index2, &head, &head_attach);
		}
	      else
		{
		  head = subfind_distlinklist_get_head(ngb_index1);
		  head_attach = subfind_distlinklist_get_head(ngb_index2);
		}

	      if(head == -1 || head_attach == -1)
		{
		  printf("We have a problem!  head=%d/%d head_attach=%d/%d for k=%d on task=%d\n",
			 (int) (head >> 32), (int) head,
			 (int) (head_attach >> 32), (int) head_attach, k, task);
		  fflush(stdout);
		  endrun(13123);
		}

	      if(head != head_attach)
		{
		  subfind_distlinklist_get_tailandlen(head, &tail, &len);
		  subfind_distlinklist_get_tailandlen(head_attach, &tail_attach, &len_attach);

		  if(len_attach > len)	/* other group is longer, swap them */
		    {
		      tmp = head;
		      head = head_attach;
		      head_attach = tmp;
		      tmp = tail;
		      tail = tail_attach;
		      tail_attach = tmp;
		      tmp = len;
		      len = len_attach;
		      len_attach = tmp;
		    }

		  /* only in case the attached group is long enough we bother to register it 
		     as a subhalo candidate */

		  if(len_attach >= All.DesLinkNgb)
		    {
		      if(count_cand[task] < max_candidates)
			{
			  candidates[task][count_cand[task]].len = len_attach;
			  candidates[task][count_cand[task]].head = head_attach;
			  count_cand[task]++;
			}
		      else
			endrun(87);
		    }

		  /* now join the two groups */
		  subfind_distlinklist_set_tailandlen(head, tail_attach, len + len_attach);
		  subfind_distlinklist_set_next(tail, head_attach);

		  ss = head_attach;
		  do
		    {
		      ss = subfind_distlinklist_set_head_get_next(ss, head);
		    }
		  while(ss >= 0);
		}

	      /* finally, attach the particle to 'head' */
	      retcode = subfind_distlinklist_get_tail_set_tail_increaselen(head, &tail, sd[off].index);

	      if(!(retcode & 1))
		subfind_distlinklist_set_headandnext(sd[off].index, head, -1);
	      if(!(retcode & 2))
		subfind_distlinklist_set_next(tail, sd[off].index);
	      break;
	    }
	}

      t1 = second();
      /*
         printf("%g sec\n", timediff(t0, t1)); fflush(stdout);
       */
    }

  printf("identification of primary candidates finished\n");

  /* add the full thing as a subhalo candidate */

  t0 = second();

  for(task = 0, head = -1, prev = -1; task < NTask; task++)
    {
      for(i = 0; i < List_NumPartGroup[task]; i++)
	{
	  index = (((long long) task) << 32) + i;

	  if(Head[task][i] == index)
	    {
	      subfind_distlinklist_get_tailandlen(Head[task][i], &tail, &len);
	      next = subfind_distlinklist_get_next(tail);
	      if(next == -1)
		{
		  if(prev < 0)
		    head = index;

		  if(prev >= 0)
		    subfind_distlinklist_set_next(prev, index);

		  prev = tail;
		}
	    }
	}
    }

  if(count_cand[NTask - 1] < max_candidates)
    {
      candidates[NTask - 1][count_cand[NTask - 1]].len = totgrouplen;
      candidates[NTask - 1][count_cand[NTask - 1]].head = head;
      count_cand[NTask - 1]++;
    }
  else
    endrun(123123);

  t1 = second();
  printf("adding background as candidate finished. (%g sec)\n", timediff(t0, t1));
  fflush(stdout);


  t0 = second();

  /* go through the whole chain once to establish a rank order. For the rank we use Len[] */

  task = (head >> 32);

  p = head;
  rank = 0;

  while(p >= 0)
    {
      p = subfind_distlinklist_setrank_and_get_next(p, &rank);
    }

  /* for each candidate, we now pull out the rank of its head */
  for(task = 0; task < NTask; task++)
    {
      for(k = 0; k < count_cand[task]; k++)
	candidates[task][k].rank = subfind_distlinklist_get_rank(candidates[task][k].head);
    }

  t1 = second();

  printf("establishing of rank order finished (%g sec)\n", timediff(t0, t1));
  fflush(stdout);

  if(((int) rank) != totgrouplen)
    {
      printf("mismatch\n");
      endrun(0);
    }

}





long long subfind_distlinklist_setrank_and_get_next(long long index, long long *rank)
{
  int task, i;
  long long next;

  task = (index >> 32);
  i = (index & MASK);

  Len[task][i] = *rank;
  *rank = *rank + 1;
  next = Next[task][i];

  return next;
}


long long subfind_distlinklist_set_head_get_next(long long index, long long head)
{
  int task, i;
  long long next;

  task = (index >> 32);
  i = (index & MASK);

  Head[task][i] = head;
  next = Next[task][i];

  return next;
}




void subfind_distlinklist_set_next(long long index, long long next)
{
  int task, i;

  task = (index >> 32);
  i = (index & MASK);

  Next[task][i] = next;
}


long long subfind_distlinklist_get_next(long long index)
{
  int task, i;
  long long next;

  task = (index >> 32);
  i = (index & MASK);

  next = Next[task][i];

  return next;
}

long long subfind_distlinklist_get_rank(long long index)
{
  int task, i;
  long long rank;

  task = (index >> 32);
  i = (index & MASK);

  rank = Len[task][i];

  return rank;
}



long long subfind_distlinklist_get_head(long long index)
{
  int task, i;
  long long head;

  task = (index >> 32);
  i = (index & MASK);

  head = Head[task][i];

  return head;
}

void subfind_distlinklist_get_two_heads(long long ngb_index1, long long ngb_index2,
					long long *head, long long *head_attach)
{
  int task, i1, i2;

  task = (ngb_index1 >> 32);
  i1 = (ngb_index1 & MASK);
  i2 = (ngb_index2 & MASK);

  *head = Head[task][i1];
  *head_attach = Head[task][i2];
}



void subfind_distlinklist_set_headandnext(long long index, long long head, long long next)
{
  int task, i;

  task = (index >> 32);
  i = (index & MASK);

  Head[task][i] = head;
  Next[task][i] = next;
}

int subfind_distlinklist_get_tail_set_tail_increaselen(long long index, long long *tail, long long newtail)
{
  int task, i, task_newtail, i_newtail, task_oldtail, i_oldtail, retcode;
  long long oldtail;

  task = (index >> 32);
  i = (index & MASK);

  retcode = 0;

  oldtail = Tail[task][i];
  Tail[task][i] = newtail;
  Len[task][i]++;
  *tail = oldtail;

  task_newtail = (newtail >> 32);
  i_newtail = (newtail & MASK);
  Head[task_newtail][i_newtail] = index;
  Next[task_newtail][i_newtail] = -1;
  retcode |= 1;

  task_oldtail = (oldtail >> 32);
  i_oldtail = (oldtail & MASK);
  Next[task_oldtail][i_oldtail] = newtail;
  retcode |= 2;

  return retcode;
}



void subfind_distlinklist_set_tailandlen(long long index, long long tail, int len)
{
  int task, i;

  task = (index >> 32);
  i = (index & MASK);

  Tail[task][i] = tail;
  Len[task][i] = len;
}




void subfind_distlinklist_get_tailandlen(long long index, long long *tail, int *len)
{
  int task, i;

  task = (index >> 32);
  i = (index & MASK);

  *tail = Tail[task][i];
  *len = Len[task][i];
}


void subfind_distlinklist_set_all(long long index, long long head, long long tail, int len, long long next)
{
  int task, i;

  task = (index >> 32);
  i = (index & MASK);

  Head[task][i] = head;
  Tail[task][i] = tail;
  Len[task][i] = len;
  Next[task][i] = next;
}



int subfind_compare_P_GrNrGrNr(const void *a, const void *b)
{
  if(abs(((struct particle_data *) a)->GrNr - GrNr) < abs(((struct particle_data *) b)->GrNr - GrNr))
    return -1;

  if(abs(((struct particle_data *) a)->GrNr - GrNr) > abs(((struct particle_data *) b)->GrNr - GrNr))
    return +1;

  if(((struct particle_data *) a)->Key < ((struct particle_data *) b)->Key)
    return -1;

  if(((struct particle_data *) a)->Key > ((struct particle_data *) b)->Key)
    return +1;

  if(((struct particle_data *) a)->ID < ((struct particle_data *) b)->ID)
    return -1;

  if(((struct particle_data *) a)->ID > ((struct particle_data *) b)->ID)
    return +1;

  return 0;
}

int subfind_compare_P_submark(const void *a, const void *b)
{
  if(((struct particle_data *) a)->submark < ((struct particle_data *) b)->submark)
    return -1;

  if(((struct particle_data *) a)->submark > ((struct particle_data *) b)->submark)
    return +1;

  if(((struct particle_data *) a)->Key < ((struct particle_data *) b)->Key)
    return -1;

  if(((struct particle_data *) a)->Key > ((struct particle_data *) b)->Key)
    return +1;

  return 0;
}


int subfind_compare_candidates_subnr(const void *a, const void *b)
{
  if(((struct cand_dat *) a)->subnr < ((struct cand_dat *) b)->subnr)
    return -1;

  if(((struct cand_dat *) a)->subnr > ((struct cand_dat *) b)->subnr)
    return +1;

  return 0;
}

int subfind_compare_candidates_nsubs(const void *a, const void *b)
{
  if(((struct cand_dat *) a)->nsub < ((struct cand_dat *) b)->nsub)
    return -1;

  if(((struct cand_dat *) a)->nsub > ((struct cand_dat *) b)->nsub)
    return +1;

  return 0;
}

int subfind_compare_candidates_boundlength(const void *a, const void *b)
{
  if(((struct cand_dat *) a)->bound_length > ((struct cand_dat *) b)->bound_length)
    return -1;

  if(((struct cand_dat *) a)->bound_length < ((struct cand_dat *) b)->bound_length)
    return +1;

  if(((struct cand_dat *) a)->rank < ((struct cand_dat *) b)->rank)
    return -1;

  if(((struct cand_dat *) a)->rank > ((struct cand_dat *) b)->rank)
    return +1;

  return 0;
}

int subfind_compare_candidates_rank(const void *a, const void *b)
{
  if(((struct cand_dat *) a)->rank < ((struct cand_dat *) b)->rank)
    return -1;

  if(((struct cand_dat *) a)->rank > ((struct cand_dat *) b)->rank)
    return +1;

  if(((struct cand_dat *) a)->len > ((struct cand_dat *) b)->len)
    return -1;

  if(((struct cand_dat *) a)->len < ((struct cand_dat *) b)->len)
    return +1;

  return 0;
}




int subfind_compare_dist_rotcurve(const void *a, const void *b)
{
  if(((sort_r2list *) a)->r < ((sort_r2list *) b)->r)
    return -1;

  if(((sort_r2list *) a)->r > ((sort_r2list *) b)->r)
    return +1;

  return 0;
}

int subfind_compare_binding_energy(const void *a, const void *b)
{
  if(*((double *) a) > *((double *) b))
    return -1;

  if(*((double *) a) < *((double *) b))
    return +1;

  return 0;
}


int subfind_compare_densities(const void *a, const void *b)	/* largest density first */
{
  if(((struct sort_density_data *) a)->density > (((struct sort_density_data *) b)->density))
    return -1;

  if(((struct sort_density_data *) a)->density < (((struct sort_density_data *) b)->density))
    return +1;

  return 0;
}


#endif
#endif
