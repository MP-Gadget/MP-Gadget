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
#ifndef SUBFIND_ALTERNATIVE_COLLECTIVE

#include "subfind.h"
#include "fof.h"


#define TAG_POLLING_DONE         201
#define TAG_SET_ALL              202
#define TAG_GET_NGB_INDICES      204
#define TAG_GET_TAILANDLEN       205
#define TAG_GET_TAILANDLEN_DATA  206
#define TAG_SET_TAILANDLEN       207
#define TAG_SET_HEADANDNEXT      209
#define TAG_SETHEADGETNEXT_DATA  210
#define TAG_SET_NEXT             211
#define TAG_SETHEADGETNEXT       213
#define TAG_GET_NEXT             215
#define TAG_GET_NEXT_DATA        216
#define TAG_GET_HEAD             217
#define TAG_GET_HEAD_DATA        218
#define TAG_ADD_PARTICLE         219
#define TAG_ADDBOUND             220
#define TAG_NID                  222
#define TAG_NID_DATA             223
#define TAG_SETRANK              224
#define TAG_SETRANK_OUT          226
#define TAG_GET_RANK             227
#define TAG_GET_RANK_DATA        228
#define TAG_MARK_PARTICLE        229
#define TAG_SET_NEWTAIL          230
#define TAG_GET_OLDTAIL          231
#define TAG_GET_TWOHEADS         232
#define TAG_GET_TWOHEADS_DATA    233


#define MASK ((((int64_t)1)<< 32)-1)
#define HIGHBIT (1 << 30)





static int64_t *Head, *Next, *Tail;
static int *Len;
static int LocalLen;
static int count_cand, max_candidates;

static struct cand_dat
{
  int64_t head;
  int64_t rank;
  int len;
  int nsub;
  int subnr;
  int parent;
  int daughtercount;
  int bound_length;
}
 *candidates;



static struct unbind_data *ud;


static struct sort_density_data
{
  MyFloat density;
  int ngbcount;
  int64_t index;		/* this will store the task in the upper word */
  int64_t ngb_index1, ngb_index2;
}
 *sd;


void subfind_unbind_independent_ones(int count_cand)
{
  int i, j, k, len, nsubs;

  /*subfind_loctree_treeallocate(All.TreeAllocFactor * NumPart, NumPart); */

  R2list = mymalloc("R2list", NumPart * sizeof(struct r2data));
  ud = mymalloc("ud", NumPart * sizeof(struct unbind_data));

  qsort(candidates, count_cand, sizeof(struct cand_dat), subfind_compare_candidates_nsubs);

  for(k = 0, i = 0; k < count_cand; k++)
    if(candidates[k].daughtercount == 0)
      {
	while(P[i].submark < candidates[k].nsub)
	  {
	    i++;
	    if(i >= NumPart)
	      endrun(13213);
	  }

	if(P[i].submark >= 0 && P[i].submark < HIGHBIT)
	  {
	    len = 0;
	    nsubs = P[i].submark;

	    if(nsubs != candidates[k].nsub)
	      {
		printf("TASK=%d i=%d k=%d nsubs=%d candidates[k].nsub=%d\n",
		       ThisTask, i, k, nsubs, candidates[k].nsub);
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
		candidates[k].bound_length = len;

		for(j = 0; j < len; j++)
		  P[ud[j].index].submark = nsubs;	/* we use this to flag the substructures */
	      }
	    else
	      candidates[k].bound_length = 0;
	  }
      }

  myfree(ud);
  myfree(R2list);
}

void subfind_process_group_collectively(int num)
{
  int64_t p;
  int len, totgrouplen1, totgrouplen2;
  int ncand, daughtercount, parent, totcand, nremaining;
  int max_loc_length, max_length;
  int count, countall, *countlist, *offset;
  int i, j, k, nr, grindex = 0, nsubs, subnr;
  int count_leaves, tot_count_leaves;
  int master;
  double SubMass, SubPos[3], SubVel[3], SubCM[3], SubVelDisp, SubVmax, SubVmaxRad, SubSpin[3], SubHalfMass,
    SubMassTab[6];
  struct cand_dat *tmp_candidates = 0;
  MyIDType SubMostBoundID;
  double t0, t1, tt0, tt1;


  if(ThisTask == 0)
    printf("\ncollectively doing halo %d\n", GrNr);

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
     They can however be accessed via SPHP(P[i).origindex] */

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

#ifndef SUBFIND_COLLECTIVE_STAGE2
  /* determine the radius that encloses a certain number of link particles */
  t0 = second();
  subfind_find_linkngb();
  t1 = second();
  if(ThisTask == 0)
    printf("find_linkngb() took %g sec\n", timediff(t0, t1));

  sd = mymalloc("sd", NumPartGroup * sizeof(struct sort_density_data));

  /* determine the indices of the nearest two denser neighbours within the link region */
  t0 = second();
  NgbLoc = mymalloc("NgbLoc", NumPartGroup * sizeof(struct nearest_ngb_data));
  R2Loc = mymalloc("R2Loc", NumPartGroup * sizeof(struct nearest_r2_data));
  subfind_find_nearesttwo();
  t1 = second();
  if(ThisTask == 0)
    printf("find_nearesttwo() took %g sec (presently allocated=%g MB)\n", timediff(t0, t1),
	   AllocatedBytes / (1024.0 * 1024.0));

  for(i = 0; i < NumPartGroup; i++)
    {
      sd[i].density = P[i].u.DM_Density;
      sd[i].ngbcount = NgbLoc[i].count;
      sd[i].index = (((int64_t) ThisTask) << 32) + i;
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
#endif

#ifdef SUBFIND_COLLECTIVE_STAGE1
  subfind_col_save_candidates_task(totgrouplen1, num);

  myfree(sd);

  force_treefree();
  domain_free();
  domain_allocate_trick();

  qsort(P, NumPart, sizeof(struct particle_data), subfind_compare_P_origindex);	/* reorder them such that the gas particles match again */

  return;
#endif


  /* allocate and initialize distributed link list */
  Head = mymalloc("Head", NumPartGroup * sizeof(int64_t));
  Next = mymalloc("Next", NumPartGroup * sizeof(int64_t));
  Tail = mymalloc("Tail", NumPartGroup * sizeof(int64_t));
  Len = mymalloc("Len", NumPartGroup * sizeof(int));

  for(i = 0; i < NumPartGroup; i++)
    {
      Head[i] = Next[i] = Tail[i] = -1;
      Len[i] = 0;
    }


  /* allocate a list to store subhalo candidates */
  max_candidates = (NumPartGroup / 100);
  candidates = mymalloc("candidates", max_candidates * sizeof(struct cand_dat));
  count_cand = 0;

#ifdef SUBFIND_COLLECTIVE_STAGE2

  subfind_col_load_candidates(num);

#else

  subfind_col_find_candidates(totgrouplen1);

#endif

  /* establish total number of candidates */
  MPI_Allreduce(&count_cand, &totcand, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  if(ThisTask == 0)
    printf("\ntotal number of subhalo candidates=%d\n", totcand);

  nremaining = totcand;

  for(i = 0; i < NumPartGroup; i++)
    Tail[i] = -1;

  for(i = 0; i < count_cand; i++)
    candidates[i].daughtercount = 0;

  do
    {
      /* Let's see which candidates can be unbound independent from each other.
         We identify them with those candidates that have no embedded other candidate */
      t0 = second();
      if(ThisTask == 0)
	tmp_candidates = mymalloc("	tmp_candidates", totcand * sizeof(struct cand_dat));

      count = count_cand;
      count *= sizeof(struct cand_dat);

      countlist = mymalloc("countlist", NTask * sizeof(int));
      offset = mymalloc("offset", NTask * sizeof(int));

      MPI_Allgather(&count, 1, MPI_INT, countlist, 1, MPI_INT, MPI_COMM_WORLD);

      for(i = 1, offset[0] = 0; i < NTask; i++)
	offset[i] = offset[i - 1] + countlist[i - 1];

      MPI_Gatherv(candidates, countlist[ThisTask], MPI_BYTE,
		  tmp_candidates, countlist, offset, MPI_BYTE, 0, MPI_COMM_WORLD);

      if(ThisTask == 0)
	{
	  for(k = 0; k < totcand; k++)
	    {
	      tmp_candidates[k].nsub = k;
	      tmp_candidates[k].subnr = k;
	    }

	  qsort(tmp_candidates, totcand, sizeof(struct cand_dat), subfind_compare_candidates_rank);

	  for(k = 0; k < totcand; k++)
	    {
	      if(tmp_candidates[k].daughtercount >= 0)
		{
		  tmp_candidates[k].daughtercount = 0;

		  for(j = k + 1; j < totcand; j++)
		    {
		      if(tmp_candidates[j].rank > tmp_candidates[k].rank + tmp_candidates[k].len)
			break;

		      if(tmp_candidates[j].daughtercount < 0)	/* ignore these */
			continue;

		      if(tmp_candidates[k].rank + tmp_candidates[k].len >=
			 tmp_candidates[j].rank + tmp_candidates[j].len)
			{
			  tmp_candidates[k].daughtercount ++;	/* we here count the number of subhalos that are enclosed */
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
	}

      MPI_Scatterv(tmp_candidates, countlist, offset, MPI_BYTE,
		   candidates, countlist[ThisTask], MPI_BYTE, 0, MPI_COMM_WORLD);


      myfree(offset);
      myfree(countlist);

      if(ThisTask == 0)
	myfree(tmp_candidates);


      for(i = 0, count_leaves = 0, max_loc_length = 0; i < count_cand; i++)
	if(candidates[i].daughtercount == 0)
	  {
	    if(candidates[i].len > max_loc_length)
	      max_loc_length = candidates[i].len;

	    if(candidates[i].len > 0.15 * All.TotNumPart / NTask)	/* seems large, let's rather do it collectively */
	      {
		candidates[i].daughtercount ++;	/* this will ensure that it is not considered in this round */
	      }
	    else
	      {
		count_leaves++;
	      }
	  }

      MPI_Allreduce(&count_leaves, &tot_count_leaves, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(&max_loc_length, &max_length, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

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
	  if(i < NumPartGroup)
	    if(Tail[i] >= 0)	/* this means this particle is already bound to a substructure */
	      P[i].origintask |= HIGHBIT;
	}

      /* we now mark the particles that are in subhalo candidates that can be processed independently in parallel */
      nsubs = 0;
      t0 = second();
      for(master = 0; master < NTask; master++)
	{
	  ncand = count_cand;

	  MPI_Bcast(&ncand, sizeof(ncand), MPI_BYTE, master, MPI_COMM_WORLD);

	  for(k = 0; k < ncand; k++)
	    {
	      if(ThisTask == master)
		{
		  len = candidates[k].len;
		  daughtercount = candidates[k].daughtercount ;	/* this is here actually the daughter count */
		}

	      MPI_Bcast(&len, sizeof(len), MPI_BYTE, master, MPI_COMM_WORLD);
	      MPI_Bcast(&daughtercount, sizeof(daughtercount), MPI_BYTE, master, MPI_COMM_WORLD);
	      MPI_Barrier(MPI_COMM_WORLD);

	      if(daughtercount == 0)
		{
		  if(ThisTask != master)
		    subfind_poll_for_requests();
		  else
		    {
		      for(i = 0, p = candidates[k].head; i < candidates[k].len; i++)
			{
			  subfind_distlinklist_mark_particle(p, master, nsubs);

			  if(p < 0)
			    {
			      printf("Bummer i=%d \n", i);
			      endrun(128);
			    }
			  p = subfind_distlinklist_get_next(p);
			}

		      /* now tell the others to stop polling */
		      for(i = 0; i < NTask; i++)
			if(i != ThisTask)
			  MPI_Send(&i, 1, MPI_INT, i, TAG_POLLING_DONE, MPI_COMM_WORLD);
		    }

		  MPI_Barrier(MPI_COMM_WORLD);
		}

	      nsubs++;
	    }
	}
      t1 = second();
      if(ThisTask == 0)
	{
	  printf("particles are marked (took %g)\n", timediff(t0, t1));
	  fflush(stdout);
	}

      t0 = second();
      subfind_exchange(0, 1); /* assemble the particles on individual processors */
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

      subfind_unbind_independent_ones(count_cand);

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
      subfind_exchange(1, 1);    /* bring them back to their original processor */
      t1 = second();


      if(ThisTask == 0)
	printf("particles have returned to their original processor (%g sec, presently allocated %g MB)\n",
	       timediff(t0, t1), AllocatedBytes / (1024.0 * 1024.0));

      /* reestablish the original order */
      qsort(P, NumPart, sizeof(struct particle_data), subfind_compare_P_GrNrGrNr);


      /* now mark the bound particles */
      for(i = 0; i < NumPartGroup; i++)
	if(P[i].submark >= 0 && P[i].submark < nsubs)
	  Tail[i] = P[i].submark;	/* we use this to flag bound parts of substructures */

      for(i = 0; i < count_cand; i++)
	if(candidates[i].daughtercount == 0)
	  candidates[i].daughtercount = -1;
    }
  while(tot_count_leaves > 0);





  force_treebuild(NumPartGroup, NULL);	/* re construct the tree for the collective part */


  /**** now we do the collective unbinding of the subhalo candidates that contain other subhalo candidates ****/
  ud = mymalloc("ud", NumPartGroup * sizeof(struct unbind_data));

  t0 = second();
  for(master = 0, nr = 0; master < NTask; master++)
    {
      ncand = count_cand;

      MPI_Bcast(&ncand, sizeof(ncand), MPI_BYTE, master, MPI_COMM_WORLD);

      for(k = 0; k < ncand; k++)
	{
	  if(ThisTask == master)
	    {
	      len = candidates[k].len;
	      nsubs = candidates[k].nsub;
	      daughtercount = candidates[k].daughtercount;	/* this is here actually the daughter count */
	    }

	  MPI_Bcast(&daughtercount, sizeof(daughtercount), MPI_BYTE, master, MPI_COMM_WORLD);
	  MPI_Barrier(MPI_COMM_WORLD);

	  if(daughtercount >= 0)
	    {
	      MPI_Bcast(&len, sizeof(len), MPI_BYTE, master, MPI_COMM_WORLD);
	      MPI_Bcast(&nsubs, sizeof(nsubs), MPI_BYTE, master, MPI_COMM_WORLD);

	      if(ThisTask == 0)
		{
		  printf("collective unbinding of nr=%d (%d) of length=%d ... ", nr, nremaining, (int) len);
		  fflush(stdout);
		}

	      nr++;

	      LocalLen = 0;

	      tt0 = second();

	      if(ThisTask != master)
		subfind_poll_for_requests();
	      else
		{
		  for(i = 0, p = candidates[k].head; i < candidates[k].len; i++)
		    {
		      subfind_distlinklist_add_particle(p);
		      if(p < 0)
			{
			  printf("Bummer i=%d \n", i);
			  endrun(123);

			}
		      p = subfind_distlinklist_get_next(p);
		    }

		  /* now tell the others to stop polling */
		  for(i = 0; i < NTask; i++)
		    if(i != ThisTask)
		      MPI_Send(&i, 1, MPI_INT, i, TAG_POLLING_DONE, MPI_COMM_WORLD);
		}

	      LocalLen = subfind_col_unbind(ud, LocalLen);

	      tt1 = second();
	      if(ThisTask == 0)
		{
		  printf("took %g sec\n", timediff(tt0, tt1));
		  fflush(stdout);
		}

	      MPI_Allreduce(&LocalLen, &len, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

	      if(len >= All.DesLinkNgb)
		{
		  /* ok, we found a substructure */
#ifdef VERBOSE
		  if(ThisTask == 0)
		    printf("substructure of len=%d found\n", (int) len);
#endif
		  for(i = 0; i < LocalLen; i++)
		    Tail[ud[i].index] = nsubs;	/* we use this to flag the substructures */

		  if(ThisTask == master)
		    {
		      candidates[k].bound_length = len;
		    }
		}
	      else
		{
		  if(ThisTask == master)
		    {
		      candidates[k].bound_length = 0;
		    }
		}
	    }
	}
    }
  t1 = second();
  if(ThisTask == 0)
    printf("\nthe collective unbinding of remaining halos took %g sec\n", timediff(t0, t1));


  for(k = 0, count = 0; k < count_cand; k++)
    if(candidates[k].bound_length >= All.DesLinkNgb)
      {
	if(candidates[k].len < All.DesLinkNgb)
	  {
	    printf("candidates[k=%d].len=%d bound=%d\n", k, candidates[k].len, candidates[k].bound_length);
	    endrun(77);
	  }
	count++;
      }

  MPI_Allreduce(&count, &countall, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  if(ThisTask == 0)
    {
      printf("\nfound %d bound substructures in FoF group of length %d\n", countall, totgrouplen1);
      fflush(stdout);
    }



  /* now determine the parent subhalo for each candidate */
  t0 = second();
  parallel_sort(candidates, count_cand, sizeof(struct cand_dat), subfind_compare_candidates_boundlength);

  if(ThisTask == 0)
    tmp_candidates = mymalloc("tmp_candidates", totcand * sizeof(struct cand_dat));

  count = count_cand;
  count *= sizeof(struct cand_dat);

  countlist = mymalloc("countlist", NTask * sizeof(int));
  offset = mymalloc("offset", NTask * sizeof(int));

  MPI_Allgather(&count, 1, MPI_INT, countlist, 1, MPI_INT, MPI_COMM_WORLD);

  for(i = 1, offset[0] = 0; i < NTask; i++)
    offset[i] = offset[i - 1] + countlist[i - 1];

  MPI_Gatherv(candidates, countlist[ThisTask], MPI_BYTE,
	      tmp_candidates, countlist, offset, MPI_BYTE, 0, MPI_COMM_WORLD);

  if(ThisTask == 0)
    {
      for(k = 0; k < totcand; k++)
	{
	  tmp_candidates[k].subnr = k;
	  tmp_candidates[k].parent = -1;
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
    }

  MPI_Scatterv(tmp_candidates, countlist, offset, MPI_BYTE,
	       candidates, countlist[ThisTask], MPI_BYTE, 0, MPI_COMM_WORLD);


  myfree(offset);
  myfree(countlist);

  if(ThisTask == 0)
    myfree(tmp_candidates);

  t1 = second();
  if(ThisTask == 0)
    printf("determination of parent subhalo took %g sec (presently allocated %g MB)\n", timediff(t0, t1),
	   AllocatedBytes / (1024.0 * 1024.0));


  /* Now let's save  some properties of the substructures */
  if(ThisTask == ((GrNr - 1) % NTask))
    {
      Group[grindex].Nsubs = countall;
    }

  t0 = second();
  for(master = 0, subnr = 0; master < NTask; master++)
    {
      ncand = count_cand;
      MPI_Bcast(&ncand, sizeof(int), MPI_INT, master, MPI_COMM_WORLD);

      for(k = 0; k < ncand; k++)
	{
	  if(ThisTask == master)
	    {
	      len = candidates[k].bound_length;
	      nsubs = candidates[k].nsub;
	      parent = candidates[k].parent;
	    }

	  MPI_Bcast(&len, sizeof(len), MPI_BYTE, master, MPI_COMM_WORLD);
	  MPI_Barrier(MPI_COMM_WORLD);

	  if(len > 0)
	    {
	      MPI_Bcast(&nsubs, sizeof(nsubs), MPI_BYTE, master, MPI_COMM_WORLD);
	      MPI_Bcast(&parent, sizeof(parent), MPI_BYTE, master, MPI_COMM_WORLD);

	      LocalLen = 0;

	      if(ThisTask != master)
		subfind_poll_for_requests();
	      else
		{
		  for(i = 0, p = candidates[k].head; i < candidates[k].len; i++)
		    {
		      subfind_distlinklist_add_bound_particles(p, nsubs);
		      p = subfind_distlinklist_get_next(p);
		    }

		  /* now tell the others to stop polling */
		  for(i = 0; i < NTask; i++)
		    if(i != ThisTask)
		      MPI_Send(&i, 1, MPI_INT, i, TAG_POLLING_DONE, MPI_COMM_WORLD);
		}

	      MPI_Barrier(MPI_COMM_WORLD);

	      tt0 = second();
	      subfind_col_determine_sub_halo_properties(ud, LocalLen, &SubMass,
							&SubPos[0], &SubVel[0], &SubCM[0], &SubVelDisp,
							&SubVmax, &SubVmaxRad, &SubSpin[0], &SubMostBoundID,
							&SubHalfMass, &SubMassTab[0]);
	      tt1 = second();

	      if(ThisTask == 0 && timediff(tt0, tt1) > 10.0)
		{
		  printf("  determining properties of halo of len=%d took %g sec\n", len, timediff(tt0, tt1));
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
    }
  t1 = second();
  if(ThisTask == 0)
    printf("determining substructure properties took %g sec (presently allocated %g MB)\n", timediff(t0, t1),
	   AllocatedBytes / (1024.0 * 1024.0));

  myfree(ud);
  myfree(candidates);
  myfree(Len);
  myfree(Tail);
  myfree(Next);
  myfree(Head);
#ifndef SUBFIND_COLLECTIVE_STAGE2
  myfree(sd);
#endif

  force_treefree();
  domain_free();
  domain_allocate_trick();

  qsort(P, NumPart, sizeof(struct particle_data), subfind_compare_P_origindex);	/* reorder them such that the gas particles match again */

  for(i = 0; i < NumPart; i++)
    if(P[i].origindex != i)
      endrun(7777);
}


#if defined(SUBFIND_COLLECTIVE_STAGE1) || defined(SUBFIND_COLLECTIVE_STAGE2)
void subfind_col_save_candidates_task(int totgrouplen, int num)
{
  FILE *fd;
  char fname[500], buf[500];
  double t0, t1;
  int nprocgroup, masterTask, groupTask;

  /* start writing of sd field */

  if(ThisTask == 0)
    {
      sprintf(buf, "%s/aux_%03d", All.OutputDir, num);
      mkdir(buf, 02755);
    }
  MPI_Barrier(MPI_COMM_WORLD);

  t0 = second();

  nprocgroup = NTask / All.NumFilesWrittenInParallel;
  if((NTask % All.NumFilesWrittenInParallel))
    nprocgroup++;
  masterTask = (ThisTask / nprocgroup) * nprocgroup;
  for(groupTask = 0; groupTask < nprocgroup; groupTask++)
    {
      if(ThisTask == (masterTask + groupTask))	/* ok, it's this processor's turn */
	{
	  sprintf(fname, "%s/aux_%03d/%s_%03d_%d.%d", All.OutputDir, num, "aux_col_sd", num, GrNr, ThisTask);
	  if(!(fd = fopen(fname, "w")))
	    {
	      printf("can't write file `%s`\n", fname);
	      endrun(118312);
	    }

	  printf("writing '%s'\n", fname);
	  fflush(stdout);

	  my_fwrite(&NTask, sizeof(int), 1, fd);
	  my_fwrite(&NumPartGroup, sizeof(int), 1, fd);
	  my_fwrite(&totgrouplen, sizeof(int), 1, fd);
	  my_fwrite(sd, NumPartGroup, sizeof(struct sort_density_data), fd);

	  fclose(fd);
	}

      MPI_Barrier(MPI_COMM_WORLD);	/* wait inside the group */
    }

  t1 = second();
  if(ThisTask == 0)
    printf("writing took %g sec\n", timediff(t0, t1));

}

void subfind_col_load_candidates(int num)
{
  FILE *fd;
  char fname[500];
  double t0, t1;
  int nprocgroup, masterTask, groupTask;
  int numPartGroup;

  /* start loading link list */

  t0 = second();

  nprocgroup = NTask / All.NumFilesWrittenInParallel;
  if((NTask % All.NumFilesWrittenInParallel))
    nprocgroup++;
  masterTask = (ThisTask / nprocgroup) * nprocgroup;
  for(groupTask = 0; groupTask < nprocgroup; groupTask++)
    {
      if(ThisTask == (masterTask + groupTask))	/* ok, it's this processor's turn */
	{
	  sprintf(fname, "%s/aux_%03d/%s_%03d_%d.%d",
		  All.OutputDir, num, "aux_col_list", num, GrNr, ThisTask);
	  if(!(fd = fopen(fname, "r")))
	    {
	      printf("can't read file `%s`\n", fname);
	      endrun(118319);
	    }

	  printf("reading '%s'\n", fname);
	  fflush(stdout);

	  my_fread(&numPartGroup, sizeof(int), 1, fd);
	  if(numPartGroup != NumPartGroup)
	    {
	      printf("Task=%d NumPartGroup mismatch, expected %d, obtained %d\n",
		     ThisTask, NumPartGroup, numPartGroup);
	    }

	  my_fread(Head, NumPartGroup, sizeof(int64_t), fd);
	  my_fread(Next, NumPartGroup, sizeof(int64_t), fd);
	  my_fread(Tail, NumPartGroup, sizeof(int64_t), fd);
	  my_fread(Len, NumPartGroup, sizeof(int), fd);

	  my_fread(&count_cand, 1, sizeof(int), fd);
	  if(count_cand > max_candidates)
	    {
	      printf("task=%d count_cand=%d max_candidates=%d\n", ThisTask, count_cand, max_candidates);
	      endrun(84);
	    }
	  my_fread(candidates, count_cand, sizeof(struct cand_dat), fd);

	  fclose(fd);
	}

      MPI_Barrier(MPI_COMM_WORLD);	/* wait inside the group */
    }

  t1 = second();
  if(ThisTask == 0)
    printf("reading took %g sec\n", timediff(t0, t1));
}
#endif


void subfind_col_find_candidates(int totgrouplen)
{
  int ngbcount, retcode, len_attach;
  int i, k, len, master;
  int64_t prev, tail, tail_attach, tmp, next, index;
  int64_t p, ss, head, head_attach, ngb_index1, ngb_index2, rank;
  double t0, t1, tt0, tt1;

  if(ThisTask == 0)
    {
      printf("building distributed linked list. (presently allocated %g MB)\n",
	     AllocatedBytes / (1024.0 * 1024.0));
      fflush(stdout);
    }

  /* now find the subhalo candidates by building up link lists from high density to low density */
  t0 = second();
  for(master = 0; master < NTask; master++)
    {
      tt0 = second();
      if(ThisTask != master)
	subfind_poll_for_requests();
      else
	{
	  for(k = 0; k < NumPartGroup; k++)
	    {
	      ngbcount = sd[k].ngbcount;
	      ngb_index1 = sd[k].ngb_index1;
	      ngb_index2 = sd[k].ngb_index2;

	      switch (ngbcount)	/* treat the different possible cases */
		{
		case 0:	/* this appears to be a lonely maximum -> new group */
		  subfind_distlinklist_set_all(sd[k].index, sd[k].index, sd[k].index, 1, -1);
		  break;

		case 1:	/* the particle is attached to exactly one group */
		  head = subfind_distlinklist_get_head(ngb_index1);

		  if(head == -1)
		    {
		      printf("We have a problem!  head=%d/%d for k=%d on task=%d\n",
			     (int) (head >> 32), (int) head, k, ThisTask);
		      fflush(stdout);
		      endrun(13123);
		    }

		  retcode = subfind_distlinklist_get_tail_set_tail_increaselen(head, &tail, sd[k].index);

		  if(!(retcode & 1))
		    subfind_distlinklist_set_headandnext(sd[k].index, head, -1);
		  if(!(retcode & 2))
		    subfind_distlinklist_set_next(tail, sd[k].index);
		  break;

		case 2:	/* the particle merges two groups together */
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
			     (int) (head_attach >> 32), (int) head_attach, k, ThisTask);
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
			  if(count_cand < max_candidates)
			    {
			      candidates[count_cand].len = len_attach;
			      candidates[count_cand].head = head_attach;
			      count_cand++;
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
		  retcode = subfind_distlinklist_get_tail_set_tail_increaselen(head, &tail, sd[k].index);

		  if(!(retcode & 1))
		    subfind_distlinklist_set_headandnext(sd[k].index, head, -1);
		  if(!(retcode & 2))
		    subfind_distlinklist_set_next(tail, sd[k].index);
		  break;
		}
	    }

	  fflush(stdout);

	  /* now tell the others to stop polling */
	  for(k = 0; k < NTask; k++)
	    if(k != ThisTask)
	      MPI_Send(&k, 1, MPI_INT, k, TAG_POLLING_DONE, MPI_COMM_WORLD);
	}

      MPI_Barrier(MPI_COMM_WORLD);
      tt1 = second();
      if(ThisTask == 0)
	{
	  printf("  ma=%d/%d took %g sec\n", master, NTask, timediff(tt0, tt1));
	  fflush(stdout);
	}
    }
  t1 = second();
  if(ThisTask == 0)
    printf("identification of primary candidates took %g sec\n", timediff(t0, t1));

  /* add the full thing as a subhalo candidate */
  t0 = second();
  for(master = 0, head = -1, prev = -1; master < NTask; master++)
    {
      if(ThisTask != master)
	subfind_poll_for_requests();
      else
	{
	  for(i = 0; i < NumPartGroup; i++)
	    {
	      index = (((int64_t) ThisTask) << 32) + i;

	      if(Head[i] == index)
		{
		  subfind_distlinklist_get_tailandlen(Head[i], &tail, &len);
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

	  /* now tell the others to stop polling */
	  for(k = 0; k < NTask; k++)
	    if(k != ThisTask)
	      MPI_Send(&k, 1, MPI_INT, k, TAG_POLLING_DONE, MPI_COMM_WORLD);
	}

      MPI_Barrier(MPI_COMM_WORLD);
      MPI_Bcast(&head, sizeof(head), MPI_BYTE, master, MPI_COMM_WORLD);
      MPI_Bcast(&prev, sizeof(prev), MPI_BYTE, master, MPI_COMM_WORLD);
    }

  if(ThisTask == NTask - 1)
    {
      if(count_cand < max_candidates)
	{
	  candidates[count_cand].len = totgrouplen;
	  candidates[count_cand].head = head;
	  count_cand++;
	}
      else
	endrun(123123);
    }
  t1 = second();
  if(ThisTask == 0)
    printf("adding background as candidate took %g sec\n", timediff(t0, t1));

  /* go through the whole chain once to establish a rank order. For the rank we use Len[] */
  t0 = second();

  master = (head >> 32);

  if(ThisTask != master)
    subfind_poll_for_requests();
  else
    {
      p = head;
      rank = 0;

      while(p >= 0)
	{
	  p = subfind_distlinklist_setrank_and_get_next(p, &rank);
	}

      /* now tell the others to stop polling */
      for(i = 0; i < NTask; i++)
	if(i != master)
	  MPI_Send(&i, 1, MPI_INT, i, TAG_POLLING_DONE, MPI_COMM_WORLD);
    }

  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Bcast(&rank, sizeof(rank), MPI_BYTE, master, MPI_COMM_WORLD);	/* just for testing */

  /* for each candidate, we now pull out the rank of its head */
  for(master = 0; master < NTask; master++)
    {
      if(ThisTask != master)
	subfind_poll_for_requests();
      else
	{
	  for(k = 0; k < count_cand; k++)
	    candidates[k].rank = subfind_distlinklist_get_rank(candidates[k].head);

	  /* now tell the others to stop polling */
	  for(i = 0; i < NTask; i++)
	    if(i != ThisTask)
	      MPI_Send(&i, 1, MPI_INT, i, TAG_POLLING_DONE, MPI_COMM_WORLD);
	}
    }
  MPI_Barrier(MPI_COMM_WORLD);

  t1 = second();
  if(ThisTask == 0)
    printf("establishing of rank order took %g sec  (p=%d, grouplen=%d) presently allocated %g MB\n",
	   timediff(t0, t1), (int) rank, totgrouplen, AllocatedBytes / (1024.0 * 1024.0));

  if(((int) rank) != totgrouplen)
    {
      printf("mismatch\n");
      endrun(0);
    }

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
      vel_to_phys = atime = 1;
      H_of_a = 0;
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
      vel_to_phys = atime = 1;
      H_of_a = 0;
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



void subfind_poll_for_requests(void)
{
  int index, nsub, source, tag, ibuf[3], target, submark, task;
  int64_t head, next, rank, buf[5];
  int64_t oldtail, newtail;
  int task_newtail, i_newtail, task_oldtail, i_oldtail;
  MPI_Status status;

  do
    {
      MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

      source = status.MPI_SOURCE;
      tag = status.MPI_TAG;

      /* MPI_Get_count(&status, MPI_BYTE, &count); */
      switch (tag)
	{
	case TAG_GET_TWOHEADS:
	  MPI_Recv(ibuf, 2, MPI_INT, source, TAG_GET_TWOHEADS, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	  buf[0] = Head[ibuf[0]];
	  buf[1] = Head[ibuf[1]];
	  MPI_Send(buf, 2 * sizeof(int64_t), MPI_BYTE, source, TAG_GET_TWOHEADS_DATA, MPI_COMM_WORLD);
	  break;
	case TAG_SET_NEWTAIL:
	  MPI_Recv(buf, 2 * sizeof(int64_t), MPI_BYTE, source, TAG_SET_NEWTAIL, MPI_COMM_WORLD,
		   MPI_STATUS_IGNORE);
	  index = buf[0];
	  newtail = buf[1];
	  oldtail = Tail[index];	/* return old tail */
	  Tail[index] = newtail;
	  Len[index]++;

	  task_newtail = (newtail >> 32);
	  if(task_newtail == ThisTask)
	    {
	      i_newtail = (newtail & MASK);
	      Head[i_newtail] = (((int64_t) ThisTask) << 32) + index;
	      Next[i_newtail] = -1;
	    }
	  task_oldtail = (oldtail >> 32);
	  if(task_oldtail == ThisTask)
	    {
	      i_oldtail = (oldtail & MASK);
	      Next[i_oldtail] = newtail;
	    }

	  buf[0] = oldtail;
	  MPI_Send(buf, 1 * sizeof(int64_t), MPI_BYTE, source, TAG_GET_OLDTAIL, MPI_COMM_WORLD);
	  break;
	case TAG_SET_ALL:
	  MPI_Recv(buf, 5 * sizeof(int64_t), MPI_BYTE, source, TAG_SET_ALL, MPI_COMM_WORLD,
		   MPI_STATUS_IGNORE);
	  index = buf[0];
	  Head[index] = buf[1];
	  Tail[index] = buf[2];
	  Len[index] = buf[3];
	  Next[index] = buf[4];
	  break;
	case TAG_GET_TAILANDLEN:
	  MPI_Recv(&index, 1, MPI_INT, source, tag, MPI_COMM_WORLD, &status);
	  buf[0] = Tail[index];
	  buf[1] = Len[index];
	  MPI_Send(buf, 2 * sizeof(int64_t), MPI_BYTE, source, TAG_GET_TAILANDLEN_DATA, MPI_COMM_WORLD);
	  break;
	case TAG_SET_TAILANDLEN:
	  MPI_Recv(buf, 3 * sizeof(int64_t), MPI_BYTE, source, TAG_SET_TAILANDLEN, MPI_COMM_WORLD,
		   MPI_STATUS_IGNORE);
	  index = buf[0];
	  Tail[index] = buf[1];
	  Len[index] = buf[2];
	  break;
	case TAG_SET_HEADANDNEXT:
	  MPI_Recv(buf, 3 * sizeof(int64_t), MPI_BYTE, source, TAG_SET_HEADANDNEXT, MPI_COMM_WORLD,
		   MPI_STATUS_IGNORE);
	  index = buf[0];
	  Head[index] = buf[1];
	  Next[index] = buf[2];
	  break;
	case TAG_SET_NEXT:
	  MPI_Recv(buf, 2 * sizeof(int64_t), MPI_BYTE, source, TAG_SET_NEXT, MPI_COMM_WORLD,
		   MPI_STATUS_IGNORE);
	  index = buf[0];
	  Next[index] = buf[1];
	  break;
	case TAG_SETHEADGETNEXT:
	  MPI_Recv(buf, 2 * sizeof(int64_t), MPI_BYTE, source, TAG_SETHEADGETNEXT, MPI_COMM_WORLD,
		   MPI_STATUS_IGNORE);
	  index = buf[0];
	  head = buf[1];
	  do
	    {
	      Head[index] = head;
	      next = Next[index];
	      task = (next >> 32);
	      index = (next & MASK);
	    }
	  while(next >= 0 && task == ThisTask);
	  MPI_Send(&next, 1 * sizeof(int64_t), MPI_BYTE, source, TAG_SETHEADGETNEXT_DATA, MPI_COMM_WORLD);
	  break;
	case TAG_GET_NEXT:
	  MPI_Recv(&index, 1, MPI_INT, source, tag, MPI_COMM_WORLD, &status);
	  MPI_Send(&Next[index], 1 * sizeof(int64_t), MPI_BYTE, source, TAG_GET_NEXT_DATA, MPI_COMM_WORLD);
	  break;
	case TAG_GET_HEAD:
	  MPI_Recv(&index, 1, MPI_INT, source, tag, MPI_COMM_WORLD, &status);
	  MPI_Send(&Head[index], 1 * sizeof(int64_t), MPI_BYTE, source, TAG_GET_HEAD_DATA, MPI_COMM_WORLD);
	  break;
	case TAG_ADD_PARTICLE:
	  MPI_Recv(&index, 1, MPI_INT, source, tag, MPI_COMM_WORLD, &status);
	  if(Tail[index] < 0)	/* consider only particles not already in substructures */
	    {
	      ud[LocalLen].index = index;
	      if(index >= NumPartGroup)
		{
		  printf("What: index=%d NumPartGroup=%d\n", index, NumPartGroup);
		  endrun(199);
		}
	      LocalLen++;
	    }
	  break;
	case TAG_MARK_PARTICLE:
	  MPI_Recv(ibuf, 3, MPI_INT, source, TAG_MARK_PARTICLE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	  index = ibuf[0];
	  target = ibuf[1];
	  submark = ibuf[2];

	  if(P[index].submark != HIGHBIT)
	    {
	      printf("TasK=%d i=%d P[i].submark=%d?\n", ThisTask, index, P[index].submark);
	      endrun(132);
	    }

	  P[index].targettask = target;
	  P[index].submark = submark;
	  break;
	case TAG_ADDBOUND:
	  MPI_Recv(ibuf, 2, MPI_INT, source, TAG_ADDBOUND, MPI_COMM_WORLD, &status);
	  index = ibuf[0];
	  nsub = ibuf[1];
	  if(Tail[index] == nsub)	/* consider only particles in this substructure */
	    {
	      ud[LocalLen].index = index;
	      LocalLen++;
	    }
	  break;
	case TAG_SETRANK:
	  MPI_Recv(buf, 2 * sizeof(int64_t), MPI_BYTE, source, TAG_SETRANK, MPI_COMM_WORLD,
		   MPI_STATUS_IGNORE);
	  index = buf[0];
	  rank = buf[1];
	  do
	    {
	      Len[index] = rank++;
	      next = Next[index];
	      if(next < 0)
		break;
	      index = (next & MASK);
	    }
	  while((next >> 32) == ThisTask);
	  buf[0] = next;
	  buf[1] = rank;
	  MPI_Send(buf, 2 * sizeof(int64_t), MPI_BYTE, source, TAG_SETRANK_OUT, MPI_COMM_WORLD);
	  break;
	case TAG_GET_RANK:
	  MPI_Recv(&index, 1, MPI_INT, source, tag, MPI_COMM_WORLD, &status);
	  rank = Len[index];
	  MPI_Send(&rank, 1 * sizeof(int64_t), MPI_BYTE, source, TAG_GET_RANK_DATA, MPI_COMM_WORLD);
	  break;

	case TAG_POLLING_DONE:
	  MPI_Recv(&index, 1, MPI_INT, source, tag, MPI_COMM_WORLD, &status);
	  break;

	default:
	  endrun(1213);
	  break;
	}

    }
  while(tag != TAG_POLLING_DONE);

}


int64_t subfind_distlinklist_setrank_and_get_next(int64_t index, int64_t *rank)
{
  int task, i;
  int64_t next;
  int64_t buf[2];

  task = (index >> 32);
  i = (index & MASK);

  if(ThisTask == task)
    {
      Len[i] = *rank;
      *rank = *rank + 1;
      next = Next[i];
    }
  else
    {
      buf[0] = i;
      buf[1] = *rank;

      MPI_Send(buf, 2 * sizeof(int64_t), MPI_BYTE, task, TAG_SETRANK, MPI_COMM_WORLD);
      MPI_Recv(buf, 2 * sizeof(int64_t), MPI_BYTE, task, TAG_SETRANK_OUT, MPI_COMM_WORLD,
	       MPI_STATUS_IGNORE);
      next = buf[0];
      *rank = buf[1];
    }
  return next;
}


int64_t subfind_distlinklist_set_head_get_next(int64_t index, int64_t head)
{
  int task, i;
  int64_t buf[2];
  int64_t next;

  task = (index >> 32);
  i = (index & MASK);

  if(ThisTask == task)
    {
      Head[i] = head;
      next = Next[i];
    }
  else
    {
      buf[0] = i;
      buf[1] = head;
      MPI_Send(buf, 2 * sizeof(int64_t), MPI_BYTE, task, TAG_SETHEADGETNEXT, MPI_COMM_WORLD);
      MPI_Recv(&next, 1 * sizeof(int64_t), MPI_BYTE, task, TAG_SETHEADGETNEXT_DATA, MPI_COMM_WORLD,
	       MPI_STATUS_IGNORE);
    }

  return next;
}




void subfind_distlinklist_set_next(int64_t index, int64_t next)
{
  int task, i;
  int64_t buf[2];

  task = (index >> 32);
  i = (index & MASK);

  if(ThisTask == task)
    {
      Next[i] = next;
    }
  else
    {
      buf[0] = i;
      buf[1] = next;
      MPI_Send(buf, 2 * sizeof(int64_t), MPI_BYTE, task, TAG_SET_NEXT, MPI_COMM_WORLD);
    }
}

void subfind_distlinklist_add_particle(int64_t index)
{
  int task, i;

  task = (index >> 32);
  i = (index & MASK);

  if(ThisTask == task)
    {
      if(Tail[i] < 0)		/* consider only particles not already in substructures */
	{
	  ud[LocalLen].index = i;
	  if(i >= NumPartGroup)
	    {
	      printf("What: index=%d NumPartGroup=%d\n", i, NumPartGroup);
	      endrun(299);
	    }

	  LocalLen++;
	}
    }
  else
    {
      MPI_Send(&i, 1, MPI_INT, task, TAG_ADD_PARTICLE, MPI_COMM_WORLD);
    }
}

void subfind_distlinklist_mark_particle(int64_t index, int target, int submark)
{
  int task, i, ibuf[3];

  task = (index >> 32);
  i = (index & MASK);

  if(ThisTask == task)
    {
      if(P[i].submark != HIGHBIT)
	{
	  printf("Tas=%d i=%d P[i].submark=%d?\n", ThisTask, i, P[i].submark);
	  endrun(131);
	}

      P[i].targettask = target;
      P[i].submark = submark;
    }
  else
    {
      ibuf[0] = i;
      ibuf[1] = target;
      ibuf[2] = submark;
      MPI_Send(ibuf, 3, MPI_INT, task, TAG_MARK_PARTICLE, MPI_COMM_WORLD);
    }
}


void subfind_distlinklist_add_bound_particles(int64_t index, int nsub)
{
  int task, i, ibuf[2];

  task = (index >> 32);
  i = (index & MASK);

  if(ThisTask == task)
    {
      if(Tail[i] == nsub)	/* consider only particles not already in substructures */
	{
	  ud[LocalLen].index = i;
	  LocalLen++;
	}
    }
  else
    {
      ibuf[0] = i;
      ibuf[1] = nsub;
      MPI_Send(ibuf, 2, MPI_INT, task, TAG_ADDBOUND, MPI_COMM_WORLD);
    }
}


int64_t subfind_distlinklist_get_next(int64_t index)
{
  int task, i;
  int64_t next;

  task = (index >> 32);
  i = (index & MASK);

  if(ThisTask == task)
    {
      next = Next[i];
    }
  else
    {
      MPI_Send(&i, 1, MPI_INT, task, TAG_GET_NEXT, MPI_COMM_WORLD);
      MPI_Recv(&next, 1 * sizeof(int64_t), MPI_BYTE, task, TAG_GET_NEXT_DATA, MPI_COMM_WORLD,
	       MPI_STATUS_IGNORE);
    }

  return next;
}

int64_t subfind_distlinklist_get_rank(int64_t index)
{
  int task, i;
  int64_t rank;

  task = (index >> 32);
  i = (index & MASK);

  if(ThisTask == task)
    {
      rank = Len[i];
    }
  else
    {
      MPI_Send(&i, 1, MPI_INT, task, TAG_GET_RANK, MPI_COMM_WORLD);
      MPI_Recv(&rank, 1 * sizeof(int64_t), MPI_BYTE, task, TAG_GET_RANK_DATA, MPI_COMM_WORLD,
	       MPI_STATUS_IGNORE);
    }

  return rank;
}



int64_t subfind_distlinklist_get_head(int64_t index)
{
  int task, i;
  int64_t head;

  task = (index >> 32);
  i = (index & MASK);

  if(ThisTask == task)
    {
      head = Head[i];
    }
  else
    {
      MPI_Send(&i, 1, MPI_INT, task, TAG_GET_HEAD, MPI_COMM_WORLD);
      MPI_Recv(&head, 1 * sizeof(int64_t), MPI_BYTE, task, TAG_GET_HEAD_DATA, MPI_COMM_WORLD,
	       MPI_STATUS_IGNORE);
    }

  return head;
}

void subfind_distlinklist_get_two_heads(int64_t ngb_index1, int64_t ngb_index2,
					int64_t *head, int64_t *head_attach)
{
  int task, i1, i2, ibuf[2];
  int64_t buf[2];

  task = (ngb_index1 >> 32);
  i1 = (ngb_index1 & MASK);
  i2 = (ngb_index2 & MASK);

  if(ThisTask == task)
    {
      *head = Head[i1];
      *head_attach = Head[i2];
    }
  else
    {
      ibuf[0] = i1;
      ibuf[1] = i2;
      MPI_Send(ibuf, 2, MPI_INT, task, TAG_GET_TWOHEADS, MPI_COMM_WORLD);
      MPI_Recv(buf, 2 * sizeof(int64_t), MPI_BYTE, task, TAG_GET_TWOHEADS_DATA, MPI_COMM_WORLD,
	       MPI_STATUS_IGNORE);
      *head = buf[0];
      *head_attach = buf[1];
    }
}



void subfind_distlinklist_set_headandnext(int64_t index, int64_t head, int64_t next)
{
  int task, i;
  int64_t buf[3];

  task = (index >> 32);
  i = (index & MASK);

  if(ThisTask == task)
    {
      Head[i] = head;
      Next[i] = next;
    }
  else
    {
      buf[0] = i;
      buf[1] = head;
      buf[2] = next;
      MPI_Send(buf, 3 * sizeof(int64_t), MPI_BYTE, task, TAG_SET_HEADANDNEXT, MPI_COMM_WORLD);
    }
}

int subfind_distlinklist_get_tail_set_tail_increaselen(int64_t index, int64_t *tail, int64_t newtail)
{
  int task, i, task_newtail, i_newtail, task_oldtail, i_oldtail, retcode;
  int64_t oldtail;
  int64_t buf[2];

  task = (index >> 32);
  i = (index & MASK);

  retcode = 0;

  if(ThisTask == task)
    {
      oldtail = Tail[i];
      Tail[i] = newtail;
      Len[i]++;
      *tail = oldtail;

      task_newtail = (newtail >> 32);
      if(task_newtail == ThisTask)
	{
	  i_newtail = (newtail & MASK);
	  Head[i_newtail] = index;
	  Next[i_newtail] = -1;
	  retcode |= 1;
	}
      task_oldtail = (oldtail >> 32);
      if(task_oldtail == ThisTask)
	{
	  i_oldtail = (oldtail & MASK);
	  Next[i_oldtail] = newtail;
	  retcode |= 2;
	}
    }
  else
    {
      buf[0] = i;
      buf[1] = newtail;
      MPI_Send(buf, 2 * sizeof(int64_t), MPI_BYTE, task, TAG_SET_NEWTAIL, MPI_COMM_WORLD);
      MPI_Recv(&oldtail, 1 * sizeof(int64_t), MPI_BYTE, task, TAG_GET_OLDTAIL, MPI_COMM_WORLD,
	       MPI_STATUS_IGNORE);
      *tail = oldtail;

      if((newtail >> 32) == task)
	retcode |= 1;
      if((oldtail >> 32) == task)
	retcode |= 2;
    }

  return retcode;
}



void subfind_distlinklist_set_tailandlen(int64_t index, int64_t tail, int len)
{
  int task, i;
  int64_t buf[3];

  task = (index >> 32);
  i = (index & MASK);

  if(ThisTask == task)
    {
      Tail[i] = tail;
      Len[i] = len;
    }
  else
    {
      buf[0] = i;
      buf[1] = tail;
      buf[2] = len;
      MPI_Send(buf, 3 * sizeof(int64_t), MPI_BYTE, task, TAG_SET_TAILANDLEN, MPI_COMM_WORLD);
    }
}




void subfind_distlinklist_get_tailandlen(int64_t index, int64_t *tail, int *len)
{
  int task, i;
  int64_t buf[2];

  task = (index >> 32);
  i = (index & MASK);

  if(ThisTask == task)
    {
      *tail = Tail[i];
      *len = Len[i];
    }
  else
    {
      MPI_Send(&i, 1, MPI_INT, task, TAG_GET_TAILANDLEN, MPI_COMM_WORLD);
      MPI_Recv(buf, 2 * sizeof(int64_t), MPI_BYTE, task, TAG_GET_TAILANDLEN_DATA, MPI_COMM_WORLD,
	       MPI_STATUS_IGNORE);
      *tail = buf[0];
      *len = buf[1];
    }
}


void subfind_distlinklist_set_all(int64_t index, int64_t head, int64_t tail, int len, int64_t next)
{
  int task, i;
  int64_t buf[5];

  task = (index >> 32);
  i = (index & MASK);

  if(ThisTask == task)
    {
      Head[i] = head;
      Tail[i] = tail;
      Len[i] = len;
      Next[i] = next;
    }
  else
    {
      buf[0] = i;
      buf[1] = head;
      buf[2] = tail;
      buf[3] = len;
      buf[4] = next;
      MPI_Send(buf, 5 * sizeof(int64_t), MPI_BYTE, task, TAG_SET_ALL, MPI_COMM_WORLD);
    }
}




int subfind_compare_P_GrNrGrNr(const void *a, const void *b)
{
  if(abs(((struct particle_data *) a)->GrNr - GrNr) < abs(((struct particle_data *) b)->GrNr - GrNr))
    return -1;

  if(abs(((struct particle_data *) a)->GrNr - GrNr) > abs(((struct particle_data *) b)->GrNr - GrNr))
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
