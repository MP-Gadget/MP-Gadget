#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>


#include "allvars.h"
#include "proto.h"
#include "domain.h"


/*! \file domain.c
 *  \brief code for domain decomposition
 *
 *  This file contains the code for the domain decomposition of the
 *  simulation volume.  The domains are constructed from disjoint subsets
 *  of the leaves of a fiducial top-level tree that covers the full
 *  simulation volume. Domain boundaries hence run along tree-node
 *  divisions of a fiducial global BH tree. As a result of this method, the
 *  tree force are in principle strictly independent of the way the domains
 *  are cut. The domain decomposition can be carried out for an arbitrary
 *  number of CPUs. Individual domains are not cubical, but spatially
 *  coherent since the leaves are traversed in a Peano-Hilbert order and
 *  individual domains form segments along this order.  This also ensures
 *  that each domain has a small surface to volume ratio, which minimizes
 *  communication.
 */


#define REDUC_FAC      0.98


/*! toGo[task*NTask + partner] gives the number of particles in task 'task'
 *  that have to go to task 'partner'
 */
static int *toGo, *toGoSph;
static int *toGet, *toGetSph;
static int *list_NumPart;
static int *list_N_gas;
static int *list_load;
static int *list_loadsph;
static double *list_work;
static double *list_speedfac;
static double *list_cadj_cpu;
static double *list_cadj_cost;

#ifdef LT_STELLAREVOLUTION
static int *toGoStars, *toGetStars, *list_N_stars, *list_loadstars;
#endif

static struct local_topnode_data
{
  peanokey Size;		/*!< number of Peano-Hilbert mesh-cells represented by top-level node */
  peanokey StartKey;		/*!< first Peano-Hilbert key in top-level node */
  long long Count;		/*!< counts the number of particles in this top-level node */
  double Cost;
  int Daughter;			/*!< index of first daughter cell (out of 8) of top-level node */
  int Leaf;			/*!< if the node is a leaf, this gives its number when all leaves are traversed in Peano-Hilbert order */
  int Parent;
  int PIndex;			/*!< first particle in node */
}
 *topNodes;			/*!< points to the root node of the top-level tree */

static struct peano_hilbert_data
{
  peanokey key;
  int index;
}
 *mp;




static void domain_insertnode(struct local_topnode_data *treeA, struct local_topnode_data *treeB, int noA,
			      int noB);
static void domain_add_cost(struct local_topnode_data *treeA, int noA, long long count, double cost);



static float *domainWork;	/*!< a table that gives the total "work" due to the particles stored by each processor */
static int *domainCount;	/*!< a table that gives the total number of particles held by each processor */
static int *domainCountSph;	/*!< a table that gives the total number of SPH particles held by each processor */

#ifdef LT_STELLAREVOLUTION
static int *domainCountStars;
#endif

static int domain_allocated_flag = 0;

static int maxLoad, maxLoadsph;

#ifdef LT_STELLAREVOLUTION
static int maxLoadstars;
#endif

static double totgravcost, totpartcount, gravcost;



/*! This is the main routine for the domain decomposition.  It acts as a
 *  driver routine that allocates various temporary buffers, maps the
 *  particles back onto the periodic box if needed, and then does the
 *  domain decomposition, and a final Peano-Hilbert order of all particles
 *  as a tuning measure.
 */
void domain_Decomposition(void)
{
  int i, ret, retsum;
  size_t bytes, all_bytes;
  double t0, t1;

#ifdef PMGRID
  if(All.PM_Ti_endstep == All.Ti_Current)
    {
      All.NumForcesSinceLastDomainDecomp = (long long) (1 + All.TotNumPart * All.TreeDomainUpdateFrequency);
      /* to make sure that we do a domain decomposition before the PM-force is evaluated.
         this is needed to make sure that the particles are wrapped into the box */
    }
#endif

  /* Check whether it is really time for a new domain decomposition */
  if(All.NumForcesSinceLastDomainDecomp >= All.TotNumPart * All.TreeDomainUpdateFrequency
     || All.DoDynamicUpdate == 0)
    {
      CPU_Step[CPU_MISC] += measure_time();

      for(i = 0; i < NumPart; i++)
	if(P[i].Ti_current != All.Ti_Current)
	  drift_particle(i, All.Ti_Current);

      force_treefree();
      domain_free();

#if defined(SFR) || defined(BLACK_HOLES)
      rearrange_particle_sequence();
#endif

#ifdef PERIODIC
      do_box_wrapping();	/* map the particles back onto the box */
#endif
      All.NumForcesSinceLastDomainDecomp = 0;
      TreeReconstructFlag = 1;	/* ensures that new tree will be constructed */

      if(ThisTask == 0)
	{
	  printf("domain decomposition... (presently allocated=%g MB)\n", AllocatedBytes / (1024.0 * 1024.0));
	  fflush(stdout);
	}

      t0 = second();

      do
	{
	  domain_allocate();

	  all_bytes = 0;

	  Key = (peanokey *) mymalloc("domain_key", bytes = (sizeof(peanokey) * All.MaxPart));
	  all_bytes += bytes;

	  toGo = (int *) mymalloc("toGo", bytes = (sizeof(int) * NTask));
	  all_bytes += bytes;
	  toGoSph = (int *) mymalloc("toGoSph", bytes = (sizeof(int) * NTask));
	  all_bytes += bytes;
	  toGet = (int *) mymalloc("toGet", bytes = (sizeof(int) * NTask));
	  all_bytes += bytes;
	  toGetSph = (int *) mymalloc("toGetSph", bytes = (sizeof(int) * NTask));
	  all_bytes += bytes;
	  list_NumPart = (int *) mymalloc("list_NumPart", bytes = (sizeof(int) * NTask));
	  all_bytes += bytes;
	  list_N_gas = (int *) mymalloc("list_N_gas", bytes = (sizeof(int) * NTask));
	  all_bytes += bytes;
	  list_cadj_cpu = (double *) mymalloc("list_cadj_cpu", bytes = (sizeof(double) * NTask));
	  all_bytes += bytes;
	  list_cadj_cost = (double *) mymalloc("list_cadj_cost", bytes = (sizeof(double) * NTask));
	  all_bytes += bytes;
	  list_load = (int *) mymalloc("list_load", bytes = (sizeof(int) * NTask));
	  all_bytes += bytes;
	  list_loadsph = (int *) mymalloc("list_loadsph", bytes = (sizeof(int) * NTask));
	  all_bytes += bytes;
	  list_work = (double *) mymalloc("list_work", bytes = (sizeof(double) * NTask));
	  all_bytes += bytes;
	  list_speedfac = (double *) mymalloc("list_speedfac", bytes = (sizeof(double) * NTask));
	  all_bytes += bytes;
	  domainWork = (float *) mymalloc("domainWork", bytes = (MaxTopNodes * sizeof(float)));
	  all_bytes += bytes;
	  domainCount = (int *) mymalloc("domainCount", bytes = (MaxTopNodes * sizeof(int)));
	  all_bytes += bytes;
	  domainCountSph = (int *) mymalloc("domainCountSph", bytes = (MaxTopNodes * sizeof(int)));
	  all_bytes += bytes;

#ifdef LT_STELLAREVOLUTION
	  toGoStars = (int *) mymalloc("toGoStars", bytes = (sizeof(int) * NTask));
	  all_bytes += bytes;
	  toGetStars = (int *) mymalloc("toGetStars", bytes = (sizeof(int) * NTask));
	  all_bytes += bytes;
	  list_N_stars = (int *) mymalloc("list_N_stars", bytes = (sizeof(int) * NTask));
	  all_bytes += bytes;
	  list_loadstars = (int *) mymalloc("list_loadstars", bytes = (sizeof(int) * NTask));
	  all_bytes += bytes;
	  domainCountStars = (int *) mymalloc("domainCountStars", bytes = (MaxTopNodes * sizeof(int)));
	  all_bytes += bytes;
#endif

	  topNodes = (struct local_topnode_data *) mymalloc("topNodes", bytes =
							    (MaxTopNodes *
							     sizeof(struct local_topnode_data)));
	  all_bytes += bytes;

	  if(ThisTask == 0)
	    {
	      printf
		("use of %g MB of temporary storage for domain decomposition... (presently allocated=%g MB)\n",
		 all_bytes / (1024.0 * 1024.0), AllocatedBytes / (1024.0 * 1024.0));
	      fflush(stdout);
	    }

	  maxLoad = (int) (All.MaxPart * REDUC_FAC);
	  maxLoadsph = (int) (All.MaxPartSph * REDUC_FAC);
#ifdef LT_STELLAREVOLUTION
	  maxLoadstars = (int) (All.MaxPartMet * REDUC_FAC);
#endif

	  report_memory_usage(&HighMark_domain, "DOMAIN");

	  ret = domain_decompose();

	  /* copy what we need for the topnodes */
	  for(i = 0; i < NTopnodes; i++)
	    {
	      TopNodes[i].StartKey = topNodes[i].StartKey;
	      TopNodes[i].Size = topNodes[i].Size;
	      TopNodes[i].Daughter = topNodes[i].Daughter;
	      TopNodes[i].Leaf = topNodes[i].Leaf;
	    }

	  myfree(topNodes);
#ifdef LT_STELLAREVOLUTION
	  myfree(domainCountStars);
	  myfree(list_loadstars);
	  myfree(list_N_stars);
	  myfree(toGetStars);
	  myfree(toGoStars);
#endif
	  myfree(domainCountSph);
	  myfree(domainCount);
	  myfree(domainWork);
	  myfree(list_speedfac);
	  myfree(list_work);
	  myfree(list_loadsph);
	  myfree(list_load);
	  myfree(list_cadj_cost);
	  myfree(list_cadj_cpu);
	  myfree(list_N_gas);
	  myfree(list_NumPart);
	  myfree(toGetSph);
	  myfree(toGet);
	  myfree(toGoSph);
	  myfree(toGo);


	  MPI_Allreduce(&ret, &retsum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
	  if(retsum)
	    {
	      myfree(Key);
	      domain_free();
	      if(ThisTask == 0)
		printf("Increasing TopNodeAllocFactor=%g  ", All.TopNodeAllocFactor);

	      All.TopNodeAllocFactor *= 1.3;

	      if(ThisTask == 0)
		{
		  printf("new value=%g\n", All.TopNodeAllocFactor);
		  fflush(stdout);
		}

	      if(All.TopNodeAllocFactor > 1000)
		{
		  if(ThisTask == 0)
		    printf("something seems to be going seriously wrong here. Stopping.\n");
		  fflush(stdout);
		  endrun(781);
		}
	    }
	}
      while(retsum);

      t1 = second();

      if(ThisTask == 0)
	{
	  printf("domain decomposition done. (took %g sec)\n", timediff(t0, t1));
	  fflush(stdout);
	}

      CPU_Step[CPU_DOMAIN] += measure_time();

      for(i = 0; i < NumPart; i++)
#ifdef LT_STELLAREVOLUTION
	{
#endif
	  if(P[i].Type > 5 || P[i].Type < 0)
	    {
	      printf("task=%d:  P[i=%d].Type=%d\n", ThisTask, i, P[i].Type);
	      endrun(111111);
	    }
#ifdef LT_STELLAREVOLUTION
	  if((P[i].Type == 4) && (MetP[P[i].MetID].PID != i))
	    {
	      printf("task=%d:  error in cross-indexes for star-particle %d ID %u\n", ThisTask, i, P[i].ID);
	      fflush(stdout);
	      endrun(111112);
	    }

	}
#endif

#ifdef PEANOHILBERT
#ifdef SUBFIND
      if(GrNr < 0)		/* we don't do it when SUBFIND is executed for a certain group */
#endif
	peano_hilbert_order();

      CPU_Step[CPU_PEANO] += measure_time();
#endif
      myfree(Key);

      memmove(TopNodes + NTopnodes, DomainTask, NTopnodes * sizeof(int));

      TopNodes = (struct topnode_data *) myrealloc(TopNodes, bytes =
						   (NTopnodes * sizeof(struct topnode_data) +
						    NTopnodes * sizeof(int)));
      if(ThisTask == 0)
	printf("Freed %g MByte in top-level domain structure\n",
	       (MaxTopNodes - NTopnodes) * sizeof(struct topnode_data) / (1024.0 * 1024.0));

      DomainTask = (int *) (TopNodes + NTopnodes);

      force_treeallocate((int) (All.TreeAllocFactor * All.MaxPart) + NTopnodes, All.MaxPart);

      reconstruct_timebins();
    }

}

/*! This function allocates all the stuff that will be required for the tree-construction/walk later on */
void domain_allocate(void)
{
  size_t bytes, all_bytes = 0;

  MaxTopNodes = (int) (All.TopNodeAllocFactor * All.MaxPart + 1);

  DomainStartList = (int *) mymalloc("DomainStartList", bytes = (NTask * MULTIPLEDOMAINS * sizeof(int)));
  all_bytes += bytes;

  DomainEndList = (int *) mymalloc("DomainEndList", bytes = (NTask * MULTIPLEDOMAINS * sizeof(int)));
  all_bytes += bytes;

  TopNodes = (struct topnode_data *) mymalloc("TopNodes", bytes =
					      (MaxTopNodes * sizeof(struct topnode_data) +
					       MaxTopNodes * sizeof(int)));
  all_bytes += bytes;

  DomainTask = (int *) (TopNodes + MaxTopNodes);

  if(ThisTask == 0)
    printf("Allocated %g MByte for top-level domain structure\n", all_bytes / (1024.0 * 1024.0));

  domain_allocated_flag = 1;
}

void domain_free(void)
{
  if(domain_allocated_flag)
    {
      myfree(TopNodes);
      myfree(DomainEndList);
      myfree(DomainStartList);
      domain_allocated_flag = 0;
    }
}

static struct topnode_data *save_TopNodes;
static int *save_DomainStartList, *save_DomainEndList;

void domain_free_trick(void)
{
  if(domain_allocated_flag)
    {
      save_TopNodes = TopNodes;
      save_DomainEndList = DomainEndList;
      save_DomainStartList = DomainStartList;
      domain_allocated_flag = 0;
    }
  else
    endrun(131231);
}

void domain_allocate_trick(void)
{
  domain_allocated_flag = 1;

  TopNodes = save_TopNodes;
  DomainEndList = save_DomainEndList;
  DomainStartList = save_DomainStartList;
}




double domain_particle_costfactor(int i)
{
  if(P[i].TimeBin)
    return (1.0 + P[i].GravCost) / (1 << P[i].TimeBin);
  else
    return (1.0 + P[i].GravCost) / TIMEBASE;
}


/*! This function carries out the actual domain decomposition for all
 *  particle types. It will try to balance the work-load for each domain,
 *  as estimated based on the P[i]-GravCost values.  The decomposition will
 *  respect the maximum allowed memory-imbalance given by the value of
 *  PartAllocFactor.
 */
int domain_decompose(void)
{
  int i, no, status;
  long long sumtogo, sumload;
  int maxload;
  double sumwork, sumcpu, sumcost, maxwork, costfac, cadj_SpeedFac;

#ifdef CPUSPEEDADJUSTMENT
  double min_load, sum_speedfac;
#endif

  for(i = 0; i < 6; i++)
    NtypeLocal[i] = 0;

  for(i = 0, gravcost = 0; i < NumPart; i++)
    {
#ifdef SUBFIND
      if(GrNr >= 0 && P[i].GrNr != GrNr)
	continue;
#endif
      NtypeLocal[P[i].Type]++;
      costfac = domain_particle_costfactor(i);

      gravcost += costfac;
      All.Cadj_Cost += costfac;
    }
  /* because Ntype[] is of type `long long', we cannot do a simple
   * MPI_Allreduce() to sum the total particle numbers 
   */
  sumup_large_ints(6, NtypeLocal, Ntype);

  for(i = 0, totpartcount = 0; i < 6; i++)
    totpartcount += Ntype[i];


  MPI_Allreduce(&gravcost, &totgravcost, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  All.Cadj_Cpu *= 0.9;
  All.Cadj_Cost *= 0.9;

  MPI_Allgather(&All.Cadj_Cpu, 1, MPI_DOUBLE, list_cadj_cpu, 1, MPI_DOUBLE, MPI_COMM_WORLD);
  MPI_Allgather(&All.Cadj_Cost, 1, MPI_DOUBLE, list_cadj_cost, 1, MPI_DOUBLE, MPI_COMM_WORLD);

#ifdef CPUSPEEDADJUSTMENT
  MPI_Allreduce(&All.Cadj_Cost, &min_load, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
  if(min_load > 0)
    {
      cadj_SpeedFac = All.Cadj_Cpu / All.Cadj_Cost;

      MPI_Allreduce(&cadj_SpeedFac, &sum_speedfac, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      cadj_SpeedFac /= (sum_speedfac / NTask);
    }
  else
    cadj_SpeedFac = 1;

  MPI_Allgather(&cadj_SpeedFac, 1, MPI_DOUBLE, list_speedfac, 1, MPI_DOUBLE, MPI_COMM_WORLD);
#else
  cadj_SpeedFac = 1;
  MPI_Allgather(&cadj_SpeedFac, 1, MPI_DOUBLE, list_speedfac, 1, MPI_DOUBLE, MPI_COMM_WORLD);
#endif

#ifndef UNEQUALSOFTENINGS
  for(i = 0; i < 6; i++)
    if(Ntype[i] > 0)
      break;

  int j;

  for(j = i + 1; j < 6; j++)
    {
      if(Ntype[j] > 0)
	if(All.SofteningTable[j] != All.SofteningTable[i])
	  {
	    if(ThisTask == 0)
	      {
		fprintf(stdout, "Code was not compiled with UNEQUALSOFTENINGS, but some of the\n");
		fprintf(stdout, "softening lengths are unequal nevertheless.\n");
		fprintf(stdout, "This is not allowed.\n");
	      }
	    endrun(0);
	  }
    }
#endif


  /* determine global dimensions of domain grid */
  domain_findExtent();

  if(domain_determineTopTree())
    return 1;


  /* find the split of the domain grid */
  domain_findSplit_work_balanced(MULTIPLEDOMAINS * NTask, NTopleaves);
  domain_assign_load_or_work_balanced(1);


  status = domain_check_memory_bound();

  if(status != 0)		/* the optimum balanced solution violates memory constraint, let's try something different */
    {
      if(ThisTask == 0)
	printf
	  ("Note: the domain decomposition is suboptimum because the ceiling for memory-imbalance is reached\n");

      domain_findSplit_load_balanced(MULTIPLEDOMAINS * NTask, NTopleaves);
      domain_assign_load_or_work_balanced(0);

      status = domain_check_memory_bound();

      if(status != 0)
	{
	  if(ThisTask == 0)
	    printf("No domain decomposition that stays within memory bounds is possible.\n");
	  endrun(0);
	}
    }


  if(ThisTask == 0)
    {
      sumload = maxload = 0;
      sumwork = sumcpu = sumcost = maxwork = 0;
      for(i = 0; i < NTask; i++)
	{
	  sumload += list_load[i];
	  sumwork += list_speedfac[i] * list_work[i];
	  sumcpu += list_cadj_cpu[i];
	  sumcost += list_cadj_cost[i];

	  if(list_load[i] > maxload)
	    maxload = list_load[i];

	  if(list_speedfac[i] * list_work[i] > maxwork)
	    maxwork = list_speedfac[i] * list_work[i];
	}

      printf("work-load balance=%g   memory-balance=%g\n",
	     maxwork / (sumwork / NTask), maxload / (((double) sumload) / NTask));

#ifdef VERBOSE
      printf("Speedfac:\n");
      for(i = 0; i < NTask; i++)
	{
	  printf("Speedfac [%3d]  speedfac=%8.4f  work=%8.4f   load=%8.4f   cpu=%8.4f   cost=%8.4f \n", i,
		 list_speedfac[i], list_speedfac[i] * list_work[i] / (sumwork / NTask),
		 list_load[i] / (((double) sumload) / NTask), list_cadj_cpu[i] / (sumcpu / NTask),
		 list_cadj_cost[i] / (sumcost / NTask));
	}
#endif
    }

  /* flag the particles that need to be exported */

  for(i = 0; i < NumPart; i++)
    {
#ifdef SUBFIND
      if(GrNr >= 0 && P[i].GrNr != GrNr)
	continue;
#endif

      no = 0;

      while(topNodes[no].Daughter >= 0)
	no = topNodes[no].Daughter + (Key[i] - topNodes[no].StartKey) / (topNodes[no].Size / 8);

      no = topNodes[no].Leaf;

      if(DomainTask[no] != ThisTask)
	P[i].Type |= 32;
    }


  int iter = 0, ret;
  size_t exchange_limit;

  do
    {
      exchange_limit = FreeBytes - NTask * (24 * sizeof(int) + 16 * sizeof(MPI_Request));

      if(exchange_limit <= 0)
	{
	  printf("task=%d: exchange_limit=%d\n", ThisTask, (int) exchange_limit);
	  endrun(1223);
	}

      /* determine for each cpu how many particles have to be shifted to other cpus */
      ret = domain_countToGo(exchange_limit);

      for(i = 0, sumtogo = 0; i < NTask; i++)
	sumtogo += toGo[i];

      sumup_longs(1, &sumtogo, &sumtogo);

      if(ThisTask == 0)
	{
	  printf("iter=%d exchange of %d%09d particles\n", iter,
		 (int) (sumtogo / 1000000000), (int) (sumtogo % 1000000000));
	  fflush(stdout);
	}

      domain_exchange();

      iter++;
    }
  while(ret > 0);

  return 0;
}


int domain_check_memory_bound(void)
{
  int ta, m, i;
  int load, sphload, max_load, max_sphload;
  double work;

#ifdef LT_STELLAREVOLUTION
  int starsload, max_starsload;
#endif

  max_load = max_sphload = 0;
#ifdef LT_STELLAREVOLUTION
  max_starsload = 0;
#endif

  for(ta = 0; ta < NTask; ta++)
    {
      load = sphload = 0;
      work = 0;
#ifdef LT_STELLAREVOLUTION
      starsload = 0;
#endif

      for(m = 0; m < MULTIPLEDOMAINS; m++)
	for(i = DomainStartList[ta * MULTIPLEDOMAINS + m]; i <= DomainEndList[ta * MULTIPLEDOMAINS + m]; i++)
	  {
	    load += domainCount[i];
	    sphload += domainCountSph[i];
	    work += domainWork[i];
#ifdef LT_STELLAREVOLUTION
	    starsload += domainCountStars[i];
#endif
	  }

      list_load[ta] = load;
      list_loadsph[ta] = sphload;
      list_work[ta] = work;
#ifdef LT_STELLAREVOLUTION
      list_loadstars[ta] = starsload;
#endif

      if(load > max_load)
	max_load = load;
      if(sphload > max_sphload)
	max_sphload = sphload;
#ifdef LT_STELLAREVOLUTION
      if(starsload > max_starsload)
	max_starsload = starsload;
#endif
    }

#ifdef SUBFIND
  if(GrNr >= 0)
    {
      load = max_load;
      sphload = max_sphload;
#ifdef LT_STELLAREVOLUTION
      starsload = max_starsload;
#endif

      for(i = 0; i < NumPart; i++)
	{
	  if(P[i].GrNr != GrNr)
	    {
	      load++;
	      if(P[i].Type == 0)
		sphload++;
#ifdef LT_STELLAREVOLUTION
	      if(P[i].Type == 4)
		starsload++;
#endif
	    }
	}
      MPI_Allreduce(&load, &max_load, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
      MPI_Allreduce(&sphload, &max_sphload, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
#ifdef LT_STELLAREVOLUTION
      MPI_Allreduce(&starsload, &max_starsload, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
#endif
    }
#endif

  if(max_load > maxLoad)
    {
      if(ThisTask == 0)
	{
	  printf("desired memory imbalance=%g  (limit=%d, needed=%d)\n",
		 (max_load * All.PartAllocFactor) / maxLoad, maxLoad, max_load);
	}

      return 1;
    }

  if(max_sphload > maxLoadsph)
    {
      if(ThisTask == 0)
	{
	  printf("desired memory imbalance=%g  (SPH) (limit=%d, needed=%d)\n",
		 (max_sphload * All.PartAllocFactor) / maxLoadsph, maxLoadsph, max_sphload);
	}

      return 1;
    }

#ifdef LT_STELLAREVOLUTION
  if(max_starsload > maxLoadstars)
    {
      if(ThisTask == 0)
	{
	  printf("desired memory imbalance=%g  (STARS) (limit=%d, needed=%d)\n",
		 (max_starsload * All.PartAllocFactor) / maxLoadstars, maxLoadstars, max_starsload);
	}

      return 1;
    }
#endif
  return 0;
}

#ifndef NO_ISEND_IRECV_IN_DOMAIN

void domain_exchange(void)
{
  int count_togo = 0, count_togo_sph = 0, count_get = 0, count_get_sph = 0;
  int *count, *count_sph, *offset, *offset_sph;
  int *count_recv, *count_recv_sph, *offset_recv, *offset_recv_sph;
  int i, n, ngrp, no, target, n_requests;
  struct particle_data *partBuf;
  struct sph_particle_data *sphBuf;
  peanokey *keyBuf;
  MPI_Request *requests;

  requests = (MPI_Request *) mymalloc("requests", 16 * NTask * sizeof(MPI_Request));
  count = (int *) mymalloc("count", NTask * sizeof(int));
  count_sph = (int *) mymalloc("count_sph", NTask * sizeof(int));
  offset = (int *) mymalloc("offset", NTask * sizeof(int));
  offset_sph = (int *) mymalloc("offset_sph", NTask * sizeof(int));

  count_recv = (int *) mymalloc("count_recv", NTask * sizeof(int));
  count_recv_sph = (int *) mymalloc("count_recv_sph", NTask * sizeof(int));
  offset_recv = (int *) mymalloc("offset_recv", NTask * sizeof(int));
  offset_recv_sph = (int *) mymalloc("offset_recv_sph", NTask * sizeof(int));

#ifdef LT_STELLAREVOLUTION
  int count_togo_stars = 0, count_get_stars = 0;

  int *count_stars, *offset_stars;

  int *count_recv_stars, *offset_recv_stars;

  struct met_particle_data *metBuf;

  count_stars = (int *) mymalloc("count_stars", NTask * sizeof(int));
  offset_stars = (int *) mymalloc("offset_stars", NTask * sizeof(int));
  count_recv_stars = (int *) mymalloc("count_recv_stars", NTask * sizeof(int));
  offset_recv_stars = (int *) mymalloc("offset_recv_stars", NTask * sizeof(int));
#endif

  for(i = 1, offset_sph[0] = 0; i < NTask; i++)
    offset_sph[i] = offset_sph[i - 1] + toGoSph[i - 1];

#ifndef LT_STELLAREVOLUTION
  offset[0] = offset_sph[NTask - 1] + toGoSph[NTask - 1];

  for(i = 1; i < NTask; i++)
    offset[i] = offset[i - 1] + (toGo[i - 1] - toGoSph[i - 1]);
#else
  offset_stars[0] = offset_sph[NTask - 1] + toGoSph[NTask - 1];
  for(i = 1; i < NTask; i++)
    offset_stars[i] = offset_stars[i - 1] + toGoStars[i - 1];

  offset[0] = offset_stars[NTask - 1] + toGoStars[NTask - 1];
  for(i = 1; i < NTask; i++)
    offset[i] = offset[i - 1] + (toGo[i - 1] - toGoSph[i - 1] - toGoStars[i - 1]);
#endif

  for(i = 0; i < NTask; i++)
    {
      count_togo += toGo[i];
      count_togo_sph += toGoSph[i];

      count_get += toGet[i];
      count_get_sph += toGetSph[i];

#ifdef LT_STELLAREVOLUTION
      count_togo_stars += toGoStars[i];
      count_get_stars += toGetStars[i];
#endif
    }

  partBuf = (struct particle_data *) mymalloc("partBuf", count_togo * sizeof(struct particle_data));
  sphBuf = (struct sph_particle_data *) mymalloc("sphBuf", count_togo_sph * sizeof(struct sph_particle_data));
  keyBuf = (peanokey *) mymalloc("keyBuf", count_togo * sizeof(peanokey));

  for(i = 0; i < NTask; i++)
    count[i] = count_sph[i] = 0;

#ifdef LT_STELLAREVOLUTION
  metBuf =
    (struct met_particle_data *) mymalloc("metBuf", count_togo_stars * sizeof(struct met_particle_data));
  for(i = 0; i < NTask; i++)
    count_stars[i] = 0;
#endif

  for(n = 0; n < NumPart; n++)
    {
      if((P[n].Type & (32 + 16)) == (32 + 16))
	{
	  P[n].Type &= 15;

	  no = 0;

	  while(topNodes[no].Daughter >= 0)
	    no = topNodes[no].Daughter + (Key[n] - topNodes[no].StartKey) / (topNodes[no].Size / 8);

	  no = topNodes[no].Leaf;

	  target = DomainTask[no];

	  if(P[n].Type == 0)
	    {
	      partBuf[offset_sph[target] + count_sph[target]] = P[n];
	      keyBuf[offset_sph[target] + count_sph[target]] = Key[n];
	      sphBuf[offset_sph[target] + count_sph[target]] = SphP[n];
	      count_sph[target]++;
	    }
#ifdef LT_STELLAREVOLUTION
	  else if(P[n].Type == 4)
	    {
	      partBuf[offset_stars[target] + count_stars[target]] = P[n];
	      keyBuf[offset_stars[target] + count_stars[target]] = Key[n];
	      metBuf[offset_stars[target] - offset_stars[0] + count_stars[target]] = MetP[P[n].MetID];
	      count_stars[target]++;
	    }
#endif
	  else
	    {
	      partBuf[offset[target] + count[target]] = P[n];
	      keyBuf[offset[target] + count[target]] = Key[n];
	      count[target]++;
	    }


	  if(P[n].Type == 0)
	    {
	      P[n] = P[N_gas - 1];
	      P[N_gas - 1] = P[NumPart - 1];
	      Key[n] = Key[N_gas - 1];
	      Key[N_gas - 1] = Key[NumPart - 1];
#ifdef LT_STELLAREVOLUTION
	      if((P[N_gas - 1].Type & 15) == 4)
		MetP[P[N_gas - 1].MetID].PID = N_gas - 1;
#endif
	      SphP[n] = SphP[N_gas - 1];

	      NumPart--;
	      N_gas--;
	      n--;
	    }
#ifdef LT_STELLAREVOLUTION
	  else if(P[n].Type == 4)
	    {
	      MetP[P[n].MetID] = MetP[N_stars - 1];
	      P[MetP[N_stars - 1].PID].MetID = P[n].MetID;

	      if(n < NumPart - 1)
		{
		  P[n] = P[NumPart - 1];
		  Key[n] = Key[NumPart - 1];
		  if((P[n].Type & 15) == 4)
		    MetP[P[n].MetID].PID = n;
		}

	      NumPart--;
	      N_stars--;
	      n--;
	    }
#endif
	  else
	    {
	      P[n] = P[NumPart - 1];
	      Key[n] = Key[NumPart - 1];
#ifdef LT_STELLAREVOLUTION
	      if((P[n].Type & 15) == 4)
		MetP[P[n].MetID].PID = n;
#endif
	      NumPart--;
	      n--;
	    }
	}
    }

#ifndef LT_STELLAREVOLUTION
  if(count_get_sph)
    {
      memmove(P + N_gas + count_get_sph, P + N_gas, (NumPart - N_gas) * sizeof(struct particle_data));
      memmove(Key + N_gas + count_get_sph, Key + N_gas, (NumPart - N_gas) * sizeof(peanokey));
    }
#else
  if(count_get_sph + count_get_stars)
    {
      memmove(P + N_gas + count_get_sph + count_get_stars, P + N_gas,
	      (NumPart - N_gas) * sizeof(struct particle_data));
      memmove(Key + N_gas + count_get_sph + count_get_stars, Key + N_gas,
	      (NumPart - N_gas) * sizeof(peanokey));
      for(n = 0; n < N_stars; n++)
	{
	  MetP[n].PID += count_get_sph + count_get_stars;
	  if(P[MetP[n].PID].MetID != n)
	    {
	      printf("[Task %d] some serious error in adjusting the memory before particle exchange\n",
		     ThisTask);
	      fflush(stdout);
	      endrun(991000);
	    }
	}
    }
#endif

  for(i = 0; i < NTask; i++)
    {
      count_recv_sph[i] = toGetSph[i];
      count_recv[i] = toGet[i] - toGetSph[i];
#ifdef LT_STELLAREVOLUTION
      count_recv_stars[i] = toGetStars[i];
      count_recv[i] -= toGetStars[i];
#endif
    }

  for(i = 1, offset_recv_sph[0] = N_gas; i < NTask; i++)
    offset_recv_sph[i] = offset_recv_sph[i - 1] + count_recv_sph[i - 1];

#ifndef LT_STELLAREVOLUTION
  offset_recv[0] = NumPart + count_get_sph;

  for(i = 1; i < NTask; i++)
    offset_recv[i] = offset_recv[i - 1] + count_recv[i - 1];
#else
  for(i = 1, offset_recv_stars[0] = N_gas + count_get_sph; i < NTask; i++)
    offset_recv_stars[i] = offset_recv_stars[i - 1] + count_recv_stars[i - 1];

  offset_recv[0] = NumPart + count_get_sph + count_get_stars;

  for(i = 1; i < NTask; i++)
    offset_recv[i] = offset_recv[i - 1] + count_recv[i - 1];
#endif
  n_requests = 0;

  for(ngrp = 1; ngrp < (1 << PTask); ngrp++)
    {
      target = ThisTask ^ ngrp;

      if(target < NTask)
	{
	  if(count_recv_sph[target] > 0)
	    {
	      MPI_Irecv(P + offset_recv_sph[target], count_recv_sph[target] * sizeof(struct particle_data),
			MPI_BYTE, target, TAG_PDATA_SPH, MPI_COMM_WORLD, &requests[n_requests++]);

	      MPI_Irecv(Key + offset_recv_sph[target], count_recv_sph[target] * sizeof(peanokey),
			MPI_BYTE, target, TAG_KEY_SPH, MPI_COMM_WORLD, &requests[n_requests++]);

	      MPI_Irecv(SphP + offset_recv_sph[target],
			count_recv_sph[target] * sizeof(struct sph_particle_data), MPI_BYTE, target,
			TAG_SPHDATA, MPI_COMM_WORLD, &requests[n_requests++]);
	    }

#ifdef LT_STELLAREVOLUTION
	  if(count_recv_stars[target] > 0)
	    {
	      MPI_Irecv(P + offset_recv_stars[target],
			count_recv_stars[target] * sizeof(struct particle_data), MPI_BYTE, target,
			TAG_PDATA_STARS, MPI_COMM_WORLD, &requests[n_requests++]);

	      MPI_Irecv(Key + offset_recv_stars[target], count_recv_stars[target] * sizeof(peanokey),
			MPI_BYTE, target, TAG_KEY_STARS, MPI_COMM_WORLD, &requests[n_requests++]);

	      MPI_Irecv(MetP + N_stars + offset_recv_stars[target] - offset_recv_sph[NTask - 1] -
			count_recv_sph[NTask - 1],
			count_recv_stars[target] * sizeof(struct met_particle_data), MPI_BYTE, target,
			TAG_METDATA, MPI_COMM_WORLD, &requests[n_requests++]);
	    }
#endif
	  if(count_recv[target] > 0)
	    {
	      MPI_Irecv(P + offset_recv[target], count_recv[target] * sizeof(struct particle_data),
			MPI_BYTE, target, TAG_PDATA, MPI_COMM_WORLD, &requests[n_requests++]);

	      MPI_Irecv(Key + offset_recv[target], count_recv[target] * sizeof(peanokey),
			MPI_BYTE, target, TAG_KEY, MPI_COMM_WORLD, &requests[n_requests++]);
	    }
	}
    }


  MPI_Barrier(MPI_COMM_WORLD);	/* not really necessary, but this will guarantee that all receives are
				   posted before the sends, which helps the stability of MPI on 
				   bluegene, and perhaps some mpich1-clusters */

  for(ngrp = 1; ngrp < (1 << PTask); ngrp++)
    {
      target = ThisTask ^ ngrp;

      if(target < NTask)
	{
	  if(count_sph[target] > 0)
	    {
	      MPI_Isend(partBuf + offset_sph[target], count_sph[target] * sizeof(struct particle_data),
			MPI_BYTE, target, TAG_PDATA_SPH, MPI_COMM_WORLD, &requests[n_requests++]);

	      MPI_Isend(keyBuf + offset_sph[target], count_sph[target] * sizeof(peanokey),
			MPI_BYTE, target, TAG_KEY_SPH, MPI_COMM_WORLD, &requests[n_requests++]);

	      MPI_Isend(sphBuf + offset_sph[target], count_sph[target] * sizeof(struct sph_particle_data),
			MPI_BYTE, target, TAG_SPHDATA, MPI_COMM_WORLD, &requests[n_requests++]);
	    }

#ifdef LT_STELLAREVOLUTION
	  if(count_stars[target] > 0)
	    {
	      MPI_Isend(partBuf + offset_stars[target], count_stars[target] * sizeof(struct particle_data),
			MPI_BYTE, target, TAG_PDATA_STARS, MPI_COMM_WORLD, &requests[n_requests++]);

	      MPI_Isend(keyBuf + offset_stars[target], count_stars[target] * sizeof(peanokey),
			MPI_BYTE, target, TAG_KEY_STARS, MPI_COMM_WORLD, &requests[n_requests++]);

	      MPI_Isend(metBuf + offset_stars[target] - offset_sph[NTask - 1] - count_sph[NTask - 1],
			count_stars[target] * sizeof(struct met_particle_data), MPI_BYTE, target, TAG_METDATA,
			MPI_COMM_WORLD, &requests[n_requests++]);
	    }
#endif

	  if(count[target] > 0)
	    {
	      MPI_Isend(partBuf + offset[target], count[target] * sizeof(struct particle_data),
			MPI_BYTE, target, TAG_PDATA, MPI_COMM_WORLD, &requests[n_requests++]);

	      MPI_Isend(keyBuf + offset[target], count[target] * sizeof(peanokey),
			MPI_BYTE, target, TAG_KEY, MPI_COMM_WORLD, &requests[n_requests++]);
	    }
	}
    }

  MPI_Waitall(n_requests, requests, MPI_STATUSES_IGNORE);

  NumPart += count_get;
  N_gas += count_get_sph;
#ifdef LT_STELLAREVOLUTION
  for(i = 0; i < count_get_stars; i++)
    {
      if(P[offset_recv_stars[0] + i].Type != 4)
	printf("haloa oh!!!\n");
      P[offset_recv_stars[0] + i].MetID = N_stars + i;
      MetP[N_stars + i].PID = offset_recv_stars[0] + i;
    }
  N_stars += count_get_stars;
  for(i = 0; i < N_stars; i++)
    if(MetP[i].PID >= NumPart || P[MetP[i].PID].Type != 4)
      endrun(987654);
#endif

  if(NumPart > All.MaxPart)
    {
      printf("Task=%d NumPart=%d All.MaxPart=%d\n", ThisTask, NumPart, All.MaxPart);
      endrun(787878);
    }

  if(N_gas > All.MaxPartSph)
    endrun(787879);

#ifdef LT_STELLAREVOLUTION
  if(N_stars > All.MaxPartMet)
    endrun(787880);

  myfree(metBuf);
#endif

  myfree(keyBuf);
  myfree(sphBuf);
  myfree(partBuf);

#ifdef LT_STELLAREVOLUTION
  myfree(offset_recv_stars);
  myfree(count_recv_stars);
  myfree(offset_stars);
  myfree(count_stars);
#endif

  myfree(offset_recv_sph);
  myfree(offset_recv);
  myfree(count_recv_sph);
  myfree(count_recv);

  myfree(offset_sph);
  myfree(offset);
  myfree(count_sph);
  myfree(count);
  myfree(requests);
}

#else

void domain_exchange(void)
{
  int count_togo = 0, count_togo_sph = 0, count_get = 0, count_get_sph = 0;
  int *count, *count_sph, *offset, *offset_sph;
  int *count_recv, *count_recv_sph, *offset_recv, *offset_recv_sph;
  int i, n, ngrp, no, target;
  struct particle_data *partBuf;
  struct sph_particle_data *sphBuf;
  peanokey *keyBuf;

  count = (int *) mymalloc("count", NTask * sizeof(int));
  count_sph = (int *) mymalloc("count_sph", NTask * sizeof(int));
  offset = (int *) mymalloc("offset", NTask * sizeof(int));
  offset_sph = (int *) mymalloc("offset_sph", NTask * sizeof(int));

  count_recv = (int *) mymalloc("count_recv", NTask * sizeof(int));
  count_recv_sph = (int *) mymalloc("count_recv_sph", NTask * sizeof(int));
  offset_recv = (int *) mymalloc("offset_recv", NTask * sizeof(int));
  offset_recv_sph = (int *) mymalloc("offset_recv_sph", NTask * sizeof(int));

#ifdef LT_STELLAREVOLUTION
  int count_togo_stars, count_get_stars;
  int *count_stars, *offset_stars;
  int *count_recv_stars, *ffset_recv_stars;
  struct met_particle_data *metBuf;

  count_stars = (int *) mymalloc("count_stars", NTask * sizeof(int));
  offset_stars = (int *) mymalloc("offset_stars", NTask * sizeof(int));
  count_recv_stars = (int *) mymalloc("count_recv_stars", NTask * sizeof(int));
  offset_recv_stars = (int *) mymalloc("offset_recv_stars", NTask * sizeof(int));
#endif

  for(i = 1, offset_sph[0] = 0; i < NTask; i++)
    offset_sph[i] = offset_sph[i - 1] + toGoSph[i - 1];

#ifndef LT_STELLAREVOLUTION
  offset[0] = offset_sph[NTask - 1] + toGoSph[NTask - 1];

  for(i = 1; i < NTask; i++)
    offset[i] = offset[i - 1] + (toGo[i - 1] - toGoSph[i - 1]);
#else
  offset_stars[0] = offset_sph[NTask - 1] + toGoSph[NTask - 1];
  for(i = 1, offset_stars[0] = 0; i < NTask; i++)
    offset_stars[i] = offset_stars[i - 1] + toGoStars[i - 1];

  offset[0] = offset_stars[NTask - 1] + toGoStars[NTask - 1];
  for(i = 1; i < NTask; i++)
    offset[i] = offset[i - 1] + (toGo[i - 1] - toGoSph[i - 1] - toGoStars[i - 1]);
#endif

  for(i = 0; i < NTask; i++)
    {
      count_togo += toGo[i];
      count_togo_sph += toGoSph[i];

      count_get += toGet[i];
      count_get_sph += toGetSph[i];
#ifdef LT_STELLAREVOLUTION
      count_togo_stars += toGoStars[i];
      count_get_stars += toGetStars[i];
#endif
    }

  partBuf = (struct particle_data *) mymalloc("partBuf", count_togo * sizeof(struct particle_data));
  sphBuf = (struct sph_particle_data *) mymalloc("sphBuf", count_togo_sph * sizeof(struct sph_particle_data));
  keyBuf = (peanokey *) mymalloc("keyBuf", count_togo * sizeof(peanokey));

  for(i = 0; i < NTask; i++)
    count[i] = count_sph[i] = 0;

#ifdef LT_STELLAREVOLUTION
  metBuf =
    (struct met_particle_data *) mymalloc("metBuf", count_togo_stars * sizeof(struct met_particle_data));
  for(i = 0; i < NTask; i++)
    count_stars[i] = 0;
#endif

  for(n = 0; n < NumPart; n++)
    {
      if((P[n].Type & (32 + 16)) == (32 + 16))
	{
	  P[n].Type &= 15;

	  no = 0;

	  while(topNodes[no].Daughter >= 0)
	    no = topNodes[no].Daughter + (Key[n] - topNodes[no].StartKey) / (topNodes[no].Size / 8);

	  no = topNodes[no].Leaf;

	  target = DomainTask[no];

	  if(P[n].Type == 0)
	    {
	      partBuf[offset_sph[target] + count_sph[target]] = P[n];
	      keyBuf[offset_sph[target] + count_sph[target]] = Key[n];
	      sphBuf[offset_sph[target] + count_sph[target]] = SphP[n];
	      count_sph[target]++;
	    }
#ifdef LT_STELLAREVOLUTION
	  else if(P[n].Type == 4)
	    {
	      partBuf[offset_star[target] + count_stars[target]] = P[n];
	      keyBuf[offset_star[target] + count_stars[target]] = Key[n];
	      metBuf[offset_stars[target] - ofsset count_stars[target]] = MetP[P[n].MetID];
	      count_stars[target]++;
	    }
#endif
	  else
	    {
	      partBuf[offset[target] + count[target]] = P[n];
	      keyBuf[offset[target] + count[target]] = Key[n];
	      count[target]++;
	    }

	  if(P[n].Type == 0)
	    {
	      P[n] = P[N_gas - 1];
	      P[N_gas - 1] = P[NumPart - 1];
	      Key[n] = Key[N_gas - 1];
	      Key[N_gas - 1] = Key[NumPart - 1];
#ifdef LT_STELLAREVOLUTION
	      if(P[N_gas - 1].Type == 4)
		MetP[P[N_gas - 1].MetID].PID = N_gas - 1;
#endif
	      SphP[n] = SphP[N_gas - 1];

	      NumPart--;
	      N_gas--;
	      n--;
	    }
#ifdef LT_STELLAREVOLUTION
	  else if(P[n].Type == 4)
	    {
	      P[n] = P[NumPart - 1];
	      Key[n] = Key[N_gas - 1];
	      if(P[n].Type == 4)
		MetP[P[n].MetID].PID = n;
	      else
		{
		  MetP[n] = MetP[N_stars - 1];
		  P[MetP[n].PID].MetID = n;
		}

	      NumPart--;
	      N_stars--;
	      n--;
	    }
#endif
	  else
	    {
	      P[n] = P[NumPart - 1];
	      Key[n] = Key[NumPart - 1];
#ifdef LT_STELLAREVOLUTION
	      if(P[n].Type == 4)
		MetP[P[n].MetID].PID = n;
#endif
	      NumPart--;
	      n--;
	    }
	}
    }

#ifndef LT_STELLAREVOLUTION
  if(count_get_sph)
    {
      memmove(P + N_gas + count_get_sph, P + N_gas, (NumPart - N_gas) * sizeof(struct particle_data));
      memmove(Key + N_gas + count_get_sph, Key + N_gas, (NumPart - N_gas) * sizeof(peanokey));
    }
#else
  if(count_get_sph + count_get_stars)
    {
      memmove(P + N_gas + count_get_sph + count_get_stars, P + N_gas,
	      (NumPart - N_gas) * sizeof(struct particle_data));
      memmove(Key + N_gas + count_get_sph + count_get_stars, Key + N_gas,
	      (NumPart - N_gas) * sizeof(peanokey));
      for(n = 0; n < N_stars; n++)
	{
	  MetP[n].PID += count_get_sph + count_get_stars;
	  if(P[MetP[n].PID].MetID != n)
	    {
	      printf("[Task %d] some serious error in adjusting the memory before particle exchange\n",
		     ThisTask);
	      fflush(stdout);
	      endrun(991000);
	    }
	}
    }
#endif

  for(i = 0; i < NTask; i++)
    {
      count_recv_sph[i] = toGetSph[i];
      count_recv[i] = toGet[i] - toGetSph[i];
#ifdef LT_STELLAREVOLUTION
      count_recv_stars[i] = toGetStars[i];
      count_recv[i] -= toGetStars[i];
#endif
    }

  for(i = 1, offset_recv_sph[0] = N_gas; i < NTask; i++)
    offset_recv_sph[i] = offset_recv_sph[i - 1] + count_recv_sph[i - 1];

#ifndef LT_STELLAREVOLUTION
  offset_recv[0] = NumPart + count_get_sph;

  for(i = 1; i < NTask; i++)
    offset_recv[i] = offset_recv[i - 1] + count_recv[i - 1];
#else
  for(i = 1, offset_recv_stars[0] = N_gas + count_recv_sph; i < NTask; i++)
    offset_recv_stars[i] = offset_recv_stars[i - 1] + count_recv_stars[i - 1];

  offset_recv[0] = NumPart + count_get_sph + count_get_stars;

  for(i = 1; i < NTask; i++)
    offset_recv[i] = offset_recv[i - 1] + count_recv[i - 1];
#endif

  for(ngrp = 1; ngrp < (1 << PTask); ngrp++)
    {
      target = ThisTask ^ ngrp;

      if(target < NTask)
	{
	  if(count_sph[target] > 0 || count_recv_sph[target] > 0)
	    {
	      MPI_Sendrecv(partBuf + offset_sph[target], count_sph[target] * sizeof(struct particle_data),
			   MPI_BYTE, target, TAG_PDATA_SPH,
			   P + offset_recv_sph[target], count_recv_sph[target] * sizeof(struct particle_data),
			   MPI_BYTE, target, TAG_PDATA_SPH, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

	      MPI_Sendrecv(sphBuf + offset_sph[target], count_sph[target] * sizeof(struct sph_particle_data),
			   MPI_BYTE, target, TAG_SPHDATA,
			   SphP + offset_recv_sph[target],
			   count_recv_sph[target] * sizeof(struct sph_particle_data), MPI_BYTE, target,
			   TAG_SPHDATA, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

	      MPI_Sendrecv(keyBuf + offset_sph[target], count_sph[target] * sizeof(peanokey),
			   MPI_BYTE, target, TAG_KEY_SPH,
			   Key + offset_recv_sph[target], count_recv_sph[target] * sizeof(peanokey),
			   MPI_BYTE, target, TAG_KEY_SPH, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	    }

#ifdef LT_STELLAREVOLUTION
	  if(count_stars[target] > 0 || count_recv_stars[target] > 0)
	    {
	      MPI_Sendrecv(partBuf + offset_stars[target], count_stars[target] * sizeof(struct particle_data),
			   MPI_BYTE, target, TAG_PDATA_STARS,
			   P + offset_recv_stars[target],
			   count_recv_stars[target] * sizeof(struct particle_data), MPI_BYTE, target,
			   TAG_PDATA_STARS, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

	      MPI_Sendrecv(metBuf + offset_stars[target] - offset_sph[NTask - 1],
			   count_stars[target] * sizeof(struct met_particle_data), MPI_BYTE, target,
			   TAG_METDATA,
			   MetP + offset_recv_stars[target] - offset_recv_sph[NTask - 1] -
			   count_recv_sph[NTask - 1],
			   count_recv_stars[target] * sizeof(struct met_particle_data), MPI_BYTE, target,
			   TAG_SPHDATA, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

	      MPI_Sendrecv(keyBuf + offset_stars[target], count_stars[target] * sizeof(peanokey),
			   MPI_BYTE, target, TAG_KEY_STAR,
			   Key + offset_recv_stars[target], count_recv_stars[target] * sizeof(peanokey),
			   MPI_BYTE, target, TAG_KEY_STAR, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	    }

#endif
	  if(count[target] > 0 || count_recv[target] > 0)
	    {
	      MPI_Sendrecv(partBuf + offset[target], count[target] * sizeof(struct particle_data),
			   MPI_BYTE, target, TAG_PDATA,
			   P + offset_recv[target], count_recv[target] * sizeof(struct particle_data),
			   MPI_BYTE, target, TAG_PDATA, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

	      MPI_Sendrecv(keyBuf + offset[target], count[target] * sizeof(peanokey),
			   MPI_BYTE, target, TAG_KEY,
			   Key + offset_recv[target], count_recv[target] * sizeof(peanokey),
			   MPI_BYTE, target, TAG_KEY, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	    }
	}
    }

  NumPart += count_get;
  N_gas += count_get_sph;
#ifdef LT_STELLAREVOLUTION
  for(i = 0; i < count_get_stars; i++)
    {
      if(P[offset_recv_stars[0] + i].Type != 4)
	printf("haloa oh!!!\n");
      P[offset_recv_stars[0] + i].MetID = N_stars + i;
      MetP[N_stars + i].PID = offset_recv_stars[0] + i;
    }
  N_stars += count_get_stars;
#endif
  if(NumPart > All.MaxPart)
    {
      printf("TASK=%d NumPart=%d All.MaxPart=%d\n", ThisTask, NumPart, All.MaxPart);
      endrun(787878);
    }

  if(N_gas > All.MaxPartSph)
    endrun(787879);

#ifdef LT_STELLAREVOLUTION
  if(N_stars > All.MaxPartMet)
    endrun(787880);

  myfree(metBuf);
#endif

  myfree(keyBuf);
  myfree(sphBuf);
  myfree(partBuf);

#ifdef LT_STELLAREVOLUTION
  myfree(offset_recv_stars);
  myfree(count_recv_stars);
  myfree(offset_stars);
  myfree(count_stars);
#endif

  myfree(offset_recv_sph);
  myfree(offset_recv);
  myfree(count_recv_sph);
  myfree(count_recv);

  myfree(offset_sph);
  myfree(offset);
  myfree(count_sph);
  myfree(count);
}

#endif





void domain_findSplit_work_balanced(int ncpu, int ndomain)
{
  int i, start, end;
  double work, workavg, work_before, workavg_before;

  for(i = 0, work = 0; i < ndomain; i++)
    work += domainWork[i];

  workavg = work / ncpu;

  work_before = workavg_before = 0;

  start = 0;

  for(i = 0; i < ncpu; i++)
    {
      work = 0;
      end = start;

      work += domainWork[end] / list_speedfac[i % NTask];

      while((work + work_before < workavg + workavg_before) || (i == ncpu - 1 && end < ndomain - 1))
	{
	  if((ndomain - end) > (ncpu - i))
	    end++;
	  else
	    break;

	  work += domainWork[end] / list_speedfac[i % NTask];
	}

      DomainStartList[i] = start;
      DomainEndList[i] = end;

      work_before += work;
      workavg_before += workavg;
      start = end + 1;
    }
}


static struct domain_loadorigin_data
{
  double load;
  int origin;
}
 *domain;

static struct domain_segments_data
{
  int task, start, end;
}
 *domainAssign;



int domain_sort_loadorigin(const void *a, const void *b)
{
  if(((struct domain_loadorigin_data *) a)->load < (((struct domain_loadorigin_data *) b)->load))
    return -1;

  if(((struct domain_loadorigin_data *) a)->load > (((struct domain_loadorigin_data *) b)->load))
    return +1;

  return 0;
}

int domain_sort_segments(const void *a, const void *b)
{
  if(((struct domain_segments_data *) a)->task < (((struct domain_segments_data *) b)->task))
    return -1;

  if(((struct domain_segments_data *) a)->task > (((struct domain_segments_data *) b)->task))
    return +1;

  return 0;
}


void domain_assign_load_or_work_balanced(int mode)
{
  int i, n, ndomains, *target;

  domainAssign =
    (struct domain_segments_data *) mymalloc("domainAssign",
					     MULTIPLEDOMAINS * NTask * sizeof(struct domain_segments_data));
  domain =
    (struct domain_loadorigin_data *) mymalloc("domain",
					       MULTIPLEDOMAINS * NTask *
					       sizeof(struct domain_loadorigin_data));
  target = (int *) mymalloc("target", MULTIPLEDOMAINS * NTask * sizeof(int));

  for(n = 0; n < MULTIPLEDOMAINS * NTask; n++)
    domainAssign[n].task = n;

  ndomains = MULTIPLEDOMAINS * NTask;

  while(ndomains > NTask)
    {
      for(i = 0; i < ndomains; i++)
	{
	  domain[i].load = 0;
	  domain[i].origin = i;
	}

      for(n = 0; n < MULTIPLEDOMAINS * NTask; n++)
	{
	  for(i = DomainStartList[n]; i <= DomainEndList[n]; i++)
	    if(mode == 1)
	      domain[domainAssign[n].task].load += domainCount[i];
	    else
	      domain[domainAssign[n].task].load += domainWork[i];
	}

      qsort(domain, ndomains, sizeof(struct domain_loadorigin_data), domain_sort_loadorigin);

      for(i = 0; i < ndomains / 2; i++)
	{
	  target[domain[i].origin] = i;
	  target[domain[ndomains - 1 - i].origin] = i;
	}

      for(n = 0; n < MULTIPLEDOMAINS * NTask; n++)
	domainAssign[n].task = target[domainAssign[n].task];

      ndomains /= 2;
    }

  for(n = 0; n < MULTIPLEDOMAINS * NTask; n++)
    {
      domainAssign[n].start = DomainStartList[n];
      domainAssign[n].end = DomainEndList[n];
    }

  qsort(domainAssign, MULTIPLEDOMAINS * NTask, sizeof(struct domain_segments_data), domain_sort_segments);

  for(n = 0; n < MULTIPLEDOMAINS * NTask; n++)
    {
      DomainStartList[n] = domainAssign[n].start;
      DomainEndList[n] = domainAssign[n].end;

      for(i = DomainStartList[n]; i <= DomainEndList[n]; i++)
	DomainTask[i] = domainAssign[n].task;
    }

  myfree(target);
  myfree(domain);
  myfree(domainAssign);
}



void domain_findSplit_load_balanced(int ncpu, int ndomain)
{
  int i, start, end;
  double load, loadavg, load_before, loadavg_before;

  for(i = 0, load = 0; i < ndomain; i++)
    load += domainCount[i];

  loadavg = load / ncpu;

  load_before = loadavg_before = 0;

  start = 0;

  for(i = 0; i < ncpu; i++)
    {
      load = 0;
      end = start;

      load += domainCount[end];

      while((load + load_before < loadavg + loadavg_before) || (i == ncpu - 1 && end < ndomain - 1))
	{
	  if((ndomain - end) > (ncpu - i))
	    end++;
	  else
	    break;

	  load += domainCount[end];
	}

      DomainStartList[i] = start;
      DomainEndList[i] = end;

      load_before += load;
      loadavg_before += loadavg;
      start = end + 1;
    }
}







/*! This function determines how many particles that are currently stored
 *  on the local CPU have to be moved off according to the domain
 *  decomposition.
 */
int domain_countToGo(size_t nlimit)
{
  int n, no, ret, retsum;
  size_t package;

  for(n = 0; n < NTask; n++)
    {
      toGo[n] = 0;
      toGoSph[n] = 0;
#ifdef LT_STELLAREVOLUTION
      toGoStars[n] = 0;
#endif
    }

  package = (sizeof(struct particle_data) + sizeof(struct sph_particle_data) + sizeof(peanokey));
  if(package >= nlimit)
    endrun(212);


  for(n = 0; n < NumPart && package < nlimit; n++)
    {
#ifdef SUBFIND
      if(GrNr >= 0 && P[n].GrNr != GrNr)
	continue;
#endif
      if(P[n].Type & 32)
	{
	  no = 0;

	  while(topNodes[no].Daughter >= 0)
	    no = topNodes[no].Daughter + (Key[n] - topNodes[no].StartKey) / (topNodes[no].Size / 8);

	  no = topNodes[no].Leaf;

	  if(DomainTask[no] != ThisTask)
	    {
	      toGo[DomainTask[no]] += 1;
	      nlimit -= sizeof(struct particle_data) + sizeof(peanokey);

	      if((P[n].Type & 15) == 0)
		{
		  toGoSph[DomainTask[no]] += 1;
		  nlimit -= sizeof(struct sph_particle_data);
		}
#ifdef LT_STELLAREVOLUTION
	      if((P[n].Type & 15) == 4)
		{
		  toGoStars[DomainTask[no]] += 1;
		  nlimit -= sizeof(struct met_particle_data);
		}
#endif
	      P[n].Type |= 16;	/* flag this particle for export */
	    }
	}
    }

  MPI_Alltoall(toGo, 1, MPI_INT, toGet, 1, MPI_INT, MPI_COMM_WORLD);
  MPI_Alltoall(toGoSph, 1, MPI_INT, toGetSph, 1, MPI_INT, MPI_COMM_WORLD);
#ifdef LT_STELLAREVOLUTION
  MPI_Alltoall(toGoStars, 1, MPI_INT, toGetStars, 1, MPI_INT, MPI_COMM_WORLD);
#endif

  if(package >= nlimit)
    ret = 1;
  else
    ret = 0;

  MPI_Allreduce(&ret, &retsum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  if(retsum)
    {
      /* in this case, we are not guaranteed that the temporary state after
         the partial exchange will actually observe the particle limits on all
         processors... we need to test this explicitly and rework the exchange
         such that this is guaranteed. This is actually a rather non-trivial
         constraint. */

      MPI_Allgather(&NumPart, 1, MPI_INT, list_NumPart, 1, MPI_INT, MPI_COMM_WORLD);
      MPI_Allgather(&N_gas, 1, MPI_INT, list_N_gas, 1, MPI_INT, MPI_COMM_WORLD);
#ifdef LT_STELLAREVOLUTION
      MPI_Allgather(&N_stars, 1, MPI_INT, list_N_stars, 1, MPI_INT, MPI_COMM_WORLD);
#endif

      int flag, flagsum, ntoomany, ta, i, target;
      int count_togo, count_toget, count_togo_sph, count_toget_sph;

#ifdef LT_STELLAREVOLUTION
      int ntoomanystars, count_togo_stars, count_toget_stars;
#endif

      do
	{
	  flagsum = 0;

	  do
	    {
	      flag = 0;

	      for(ta = 0; ta < NTask; ta++)
		{
		  if(ta == ThisTask)
		    {
		      count_togo = count_toget = 0;
		      count_togo_sph = count_toget_sph = 0;
#ifdef LT_STELLAREVOLUTION
		      count_togo_stars = count_toget_stars = 0;
#endif
		      for(i = 0; i < NTask; i++)
			{
			  count_togo += toGo[i];
			  count_toget += toGet[i];
			  count_togo_sph += toGoSph[i];
			  count_toget_sph += toGetSph[i];
#ifdef LT_STELLAREVOLUTION
			  count_togo_stars += toGoStars[i];
			  count_toget_stars += toGetStars[i];
#endif
			}
		    }
		  MPI_Bcast(&count_togo, 1, MPI_INT, ta, MPI_COMM_WORLD);
		  MPI_Bcast(&count_toget, 1, MPI_INT, ta, MPI_COMM_WORLD);
		  MPI_Bcast(&count_togo_sph, 1, MPI_INT, ta, MPI_COMM_WORLD);
		  MPI_Bcast(&count_toget_sph, 1, MPI_INT, ta, MPI_COMM_WORLD);
#ifdef LT_STELLAREVOLUTION
		  MPI_Bcast(&count_togo_stars, 1, MPI_INT, ta, MPI_COMM_WORLD);
		  MPI_Bcast(&count_toget_stars, 1, MPI_INT, ta, MPI_COMM_WORLD);

		  if(((ntoomany = list_N_gas[ta] + count_toget_sph - count_togo_sph - All.MaxPartSph) > 0) ||
		     ((ntoomanystars =
		       list_N_stars[ta] + count_toget_stars - count_togo_stars - All.MaxPartMet) > 0))
#else
		  if((ntoomany = list_N_gas[ta] + count_toget_sph - count_togo_sph - All.MaxPartSph) > 0)
#endif
		    {
		      if(ThisTask == 0)
			{
#ifdef LT_STELLAREVOLUTION
			  if(ntoomany)
			    {
#endif
			      printf
				("exchange needs to be modified because I can't receive %d SPH-particles on task=%d\n",
				 ntoomany, ta);
			      if(flagsum > 25)
				printf("list_N_gas[ta=%d]=%d  count_toget_sph=%d count_togo_sph=%d\n",
				       ta, list_N_gas[ta], count_toget_sph, count_togo_sph);
			      fflush(stdout);
#ifdef LT_STELLAREVOLUTION
			    }
			  if(ntoomanystars)
			    {
			      printf
				("exchange needs to be modified because I can't receive %d STAR-particles on task=%d\n",
				 ntoomany, ta);
			      if(flagsum > 25)
				printf("list_N_stars[ta=%d]=%d  count_toget_stars=%d count_togo_stars=%d\n",
				       ta, list_N_stars[ta], count_toget_stars, count_togo_stars);
			      fflush(stdout);
			    }
#endif
			}

		      flag = 1;
		      i = flagsum % NTask;
#ifndef LT_STELLAREVOLUTION
		      while(ntoomany)
#else
		      while(ntoomany || ntoomanystars)
#endif
			{
			  if(i == ThisTask)
			    {
			      if(toGoSph[ta] > 0)
#ifdef LT_STELLAREVOLUTION
				if(ntoomany)
#endif
				  {
				    toGoSph[ta]--;
				    count_toget_sph--;
				    count_toget--;
				    ntoomany--;
				  }
#ifdef LT_STELLAREVOLUTION
			      if(toGoStars[ta] > 0 && ntoomanystars)
				{
				  toGoStars[ta]--;
				  count_toget_stars--;
				  count_toget--;
				  ntoomanystars--;
				}
#endif
			    }

			  MPI_Bcast(&ntoomany, 1, MPI_INT, i, MPI_COMM_WORLD);
			  MPI_Bcast(&count_toget, 1, MPI_INT, i, MPI_COMM_WORLD);
			  MPI_Bcast(&count_toget_sph, 1, MPI_INT, i, MPI_COMM_WORLD);
#ifdef LT_STELLAREVOLUTION
			  MPI_Bcast(&count_toget_stars, 1, MPI_INT, i, MPI_COMM_WORLD);
#endif
			  i++;
			  if(i >= NTask)
			    i = 0;
			}
		    }

		  if((ntoomany = list_NumPart[ta] + count_toget - count_togo - All.MaxPart) > 0)
		    {
		      if(ThisTask == 0)
			{
			  printf
			    ("exchange needs to be modified because I can't receive %d particles on task=%d\n",
			     ntoomany, ta);
			  if(flagsum > 25)
			    printf("list_NumPart[ta=%d]=%d  count_toget=%d count_togo=%d\n",
				   ta, list_NumPart[ta], count_toget, count_togo);
			  fflush(stdout);
			}

		      flag = 1;
		      i = flagsum % NTask;
		      while(ntoomany)
			{
			  if(i == ThisTask)
			    {
			      if(toGo[ta] > 0)
				{
				  toGo[ta]--;
				  count_toget--;
				  ntoomany--;
				}
			    }

			  MPI_Bcast(&ntoomany, 1, MPI_INT, i, MPI_COMM_WORLD);
			  MPI_Bcast(&count_toget, 1, MPI_INT, i, MPI_COMM_WORLD);

			  i++;
			  if(i >= NTask)
			    i = 0;
			}
		    }
		}
	      flagsum += flag;

	      if(ThisTask == 0)
		{
		  printf("flagsum = %d\n", flagsum);
		  fflush(stdout);
		  if(flagsum > 100)
		    endrun(1013);
		}
	    }
	  while(flag);

	  if(flagsum)
	    {
	      int *local_toGo, *local_toGoSph;

	      local_toGo = (int *)mymalloc("	      local_toGo", NTask * sizeof(int));
	      local_toGoSph = (int *)mymalloc("	      local_toGoSph", NTask * sizeof(int));

#ifdef LT_STELLAREVOLUTION
	      int *local_toGoStars;

	      local_toGoStars = (int *)mymalloc("	      local_toGoStars", NTask * sizeof(int));
#endif

	      for(n = 0; n < NTask; n++)
		{
		  local_toGo[n] = 0;
		  local_toGoSph[n] = 0;
#ifdef LT_STELLAREVOLUTION
		  local_toGoStars[n] = 0;
#endif
		}

	      for(n = 0; n < NumPart; n++)
		{
		  if(P[n].Type & 32)
		    {
		      P[n].Type &= (15 + 32);	/* clear 16 */

		      no = 0;

		      while(topNodes[no].Daughter >= 0)
			no =
			  topNodes[no].Daughter + (Key[n] - topNodes[no].StartKey) / (topNodes[no].Size / 8);

		      no = topNodes[no].Leaf;

		      target = DomainTask[no];

		      if((P[n].Type & 15) == 0)
			{
			  if(local_toGoSph[target] < toGoSph[target] && local_toGo[target] < toGo[target])
			    {
			      local_toGo[target] += 1;
			      local_toGoSph[target] += 1;
			      P[n].Type |= 16;
			    }
			}
#ifdef LT_STELLAREVOLUTION
		      else if((P[n].Type & 15) == 4)
			{
			  if(local_toGoStars[target] < toGoStars[target] && local_toGo[target] < toGo[target])
			    {
			      local_toGo[target] += 1;
			      local_toGoStars[target] += 1;
			      P[n].Type |= 16;
			    }
			}
#endif
		      else
			{
			  if(local_toGo[target] < toGo[target])
			    {
			      local_toGo[target] += 1;
			      P[n].Type |= 16;
			    }
			}
		    }
		}

	      for(n = 0; n < NTask; n++)
		{
		  toGo[n] = local_toGo[n];
		  toGoSph[n] = local_toGoSph[n];
#ifdef LT_STELLAREVOLUTION
		  toGoStars[n] = local_toGoStars[n];
#endif
		}

	      MPI_Alltoall(toGo, 1, MPI_INT, toGet, 1, MPI_INT, MPI_COMM_WORLD);
	      MPI_Alltoall(toGoSph, 1, MPI_INT, toGetSph, 1, MPI_INT, MPI_COMM_WORLD);
#ifdef LT_STELLAREVOLUTION
	      MPI_Alltoall(toGoStars, 1, MPI_INT, toGetStars, 1, MPI_INT, MPI_COMM_WORLD);
	      myfree(local_toGoStars);
#endif

	      myfree(local_toGoSph);
	      myfree(local_toGo);
	    }
	}
      while(flagsum);

      return 1;
    }
  else
    return 0;
}






/*! This function walks the global top tree in order to establish the
 *  number of leaves it has. These leaves are distributed to different
 *  processors.
 */
void domain_walktoptree(int no)
{
  int i;

  if(topNodes[no].Daughter == -1)
    {
      topNodes[no].Leaf = NTopleaves;
      NTopleaves++;
    }
  else
    {
      for(i = 0; i < 8; i++)
	domain_walktoptree(topNodes[no].Daughter + i);
    }
}


int domain_compare_key(const void *a, const void *b)
{
  if(((struct peano_hilbert_data *) a)->key < (((struct peano_hilbert_data *) b)->key))
    return -1;

  if(((struct peano_hilbert_data *) a)->key > (((struct peano_hilbert_data *) b)->key))
    return +1;

  return 0;
}


int domain_check_for_local_refine(int i, double countlimit, double costlimit)
{
  int j, p, sub, flag = 0;

#ifdef DENSITY_INDEPENDENT_SPH_DEBUG
  costlimit = -1;
  countlimit = -1;
#endif
  if(topNodes[i].Parent >= 0)
    {
      if(topNodes[i].Count > 0.8 * topNodes[topNodes[i].Parent].Count ||
	 topNodes[i].Cost > 0.8 * topNodes[topNodes[i].Parent].Cost)
	flag = 1;
    }

  if((topNodes[i].Count > countlimit || topNodes[i].Cost > costlimit || flag == 1) && topNodes[i].Size >= 8)
    {
      if(topNodes[i].Size >= 8)
	{
	  if((NTopnodes + 8) <= MaxTopNodes)
	    {
	      topNodes[i].Daughter = NTopnodes;

	      for(j = 0; j < 8; j++)
		{
		  sub = topNodes[i].Daughter + j;
		  topNodes[sub].Daughter = -1;
		  topNodes[sub].Parent = i;
		  topNodes[sub].Size = (topNodes[i].Size >> 3);
		  topNodes[sub].StartKey = topNodes[i].StartKey + j * topNodes[sub].Size;
		  topNodes[sub].PIndex = topNodes[i].PIndex;
		  topNodes[sub].Count = 0;
		  topNodes[sub].Cost = 0;

		}

	      NTopnodes += 8;

	      sub = topNodes[i].Daughter;

	      for(p = topNodes[i].PIndex, j = 0; p < topNodes[i].PIndex + topNodes[i].Count; p++)
		{
		  if(j < 7)
		    while(mp[p].key >= topNodes[sub + 1].StartKey)
		      {
			j++;
			sub++;
			topNodes[sub].PIndex = p;
			if(j >= 7)
			  break;
		      }

		  topNodes[sub].Cost += domain_particle_costfactor(mp[p].index);
		  topNodes[sub].Count++;
		}

	      for(j = 0; j < 8; j++)
		{
		  sub = topNodes[i].Daughter + j;

#ifdef DENSITY_INDEPENDENT_SPH_DEBUG
          if(topNodes[sub].Count > All.TotNumPart / 
                  (TOPNODEFACTOR * NTask * NTask))
#endif
		  if(domain_check_for_local_refine(sub, countlimit, costlimit))
		    return 1;
		}
	    }
	  else
	    return 1;
	}
    }

  return 0;
}


int domain_recursively_combine_topTree(int start, int ncpu)
{
  int i, nleft, nright, errflag = 0, imax1, imax2;
  int recvTask, ntopnodes_import;
  int master_left, master_right;
  struct local_topnode_data *topNodes_import = 0, *topNodes_temp;

  nleft = ncpu / 2;
  nright = ncpu - nleft;

  if(ncpu > 2)
    {
      errflag += domain_recursively_combine_topTree(start, nleft);
      errflag += domain_recursively_combine_topTree(start + nleft, nright);
    }

  if(ncpu >= 2)
    {
      master_left = start;
      master_right = start + nleft;
      if(master_left == master_right)
	endrun(123);

      if(ThisTask == master_left || ThisTask == master_right)
	{
	  if(ThisTask == master_left)
	    recvTask = master_right;
	  else
	    recvTask = master_left;

	  /* inform each other about the length of the trees */
	  MPI_Sendrecv(&NTopnodes, 1, MPI_INT, recvTask, TAG_GRAV_A,
		       &ntopnodes_import, 1, MPI_INT, recvTask, TAG_GRAV_A, MPI_COMM_WORLD,
		       MPI_STATUS_IGNORE);


	  topNodes_import =
	    (struct local_topnode_data *) mymalloc("topNodes_import",
						   IMAX(ntopnodes_import,
							NTopnodes) * sizeof(struct local_topnode_data));

	  /* exchange the trees */
	  MPI_Sendrecv(topNodes,
		       NTopnodes * sizeof(struct local_topnode_data), MPI_BYTE,
		       recvTask, TAG_GRAV_B,
		       topNodes_import,
		       ntopnodes_import * sizeof(struct local_topnode_data), MPI_BYTE,
		       recvTask, TAG_GRAV_B, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}

      if(ThisTask == master_left)
	{
	  for(recvTask = master_left + 1; recvTask < master_left + nleft; recvTask++)
	    {
	      MPI_Send(&ntopnodes_import, 1, MPI_INT, recvTask, TAG_GRAV_A, MPI_COMM_WORLD);
	      MPI_Send(topNodes_import,
		       ntopnodes_import * sizeof(struct local_topnode_data), MPI_BYTE,
		       recvTask, TAG_GRAV_B, MPI_COMM_WORLD);
	    }
	}

      if(ThisTask == master_right)
	{
	  for(recvTask = master_right + 1; recvTask < master_right + nright; recvTask++)
	    {
	      MPI_Send(&ntopnodes_import, 1, MPI_INT, recvTask, TAG_GRAV_A, MPI_COMM_WORLD);
	      MPI_Send(topNodes_import,
		       ntopnodes_import * sizeof(struct local_topnode_data), MPI_BYTE,
		       recvTask, TAG_GRAV_B, MPI_COMM_WORLD);
	    }
	}

      if(ThisTask > master_left && ThisTask < master_left + nleft)
	{
	  MPI_Recv(&ntopnodes_import, 1, MPI_INT, master_left, TAG_GRAV_A, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

	  topNodes_import =
	    (struct local_topnode_data *) mymalloc("topNodes_import",
						   IMAX(ntopnodes_import,
							NTopnodes) * sizeof(struct local_topnode_data));

	  MPI_Recv(topNodes_import,
		   ntopnodes_import * sizeof(struct local_topnode_data), MPI_BYTE,
		   master_left, TAG_GRAV_B, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

	}


      if(ThisTask > master_right && ThisTask < master_right + nright)
	{
	  MPI_Recv(&ntopnodes_import, 1, MPI_INT, master_right, TAG_GRAV_A, MPI_COMM_WORLD,
		   MPI_STATUS_IGNORE);

	  topNodes_import =
	    (struct local_topnode_data *) mymalloc("topNodes_import",
						   IMAX(ntopnodes_import,
							NTopnodes) * sizeof(struct local_topnode_data));

	  MPI_Recv(topNodes_import,
		   ntopnodes_import * sizeof(struct local_topnode_data), MPI_BYTE,
		   master_right, TAG_GRAV_B, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}

      if(ThisTask >= master_left && ThisTask < master_left + nleft)
	{
	  /* swap the two trees so that result will be equal on all cpus */

	  topNodes_temp =
	    (struct local_topnode_data *) mymalloc("topNodes_temp",
						   NTopnodes * sizeof(struct local_topnode_data));
	  memcpy(topNodes_temp, topNodes, NTopnodes * sizeof(struct local_topnode_data));
	  memcpy(topNodes, topNodes_import, ntopnodes_import * sizeof(struct local_topnode_data));
	  memcpy(topNodes_import, topNodes_temp, NTopnodes * sizeof(struct local_topnode_data));
	  myfree(topNodes_temp);
	  i = NTopnodes;
	  NTopnodes = ntopnodes_import;
	  ntopnodes_import = i;
	}

      if(ThisTask >= start && ThisTask < start + ncpu)
	{
	  if(errflag == 0)
	    {
	      if((NTopnodes + ntopnodes_import) <= MaxTopNodes)
		{
		  domain_insertnode(topNodes, topNodes_import, 0, 0);
		}
	      else
		{
		  errflag += 1;
		}
	    }

	  myfree(topNodes_import);
	}
    }

  return errflag;
}


#ifdef ALT_QSORT
#define KEY_TYPE struct peano_hilbert_data
#define KEY_BASE_TYPE peanokey
#define KEY_GETVAL(pk) ((pk)->key)
#define KEY_COPY(pk1,pk2)       \
  {                               \
    (pk2)->key = (pk1)->key;      \
    (pk2)->index = (pk1)->index;  \
  }
#define QSORT qsort_domain
#include "myqsort.h"
#endif

/*! This function constructs the global top-level tree node that is used
 *  for the domain decomposition. This is done by considering the string of
 *  Peano-Hilbert keys for all particles, which is recursively chopped off
 *  in pieces of eight segments until each segment holds at most a certain
 *  number of particles.
 */
int domain_determineTopTree(void)
{
  int i, count, j, sub, ngrp, imax1, imax2;
  int recvTask, sendTask, ntopnodes_import, errflag, errsum;
  struct local_topnode_data *topNodes_import, *topNodes_temp;
  double costlimit, countlimit;
  MPI_Status status;

  mp = (struct peano_hilbert_data *) mymalloc("mp", sizeof(struct peano_hilbert_data) * NumPart);

  for(i = 0, count = 0; i < NumPart; i++)
    {
#ifdef SUBFIND
      if(GrNr >= 0 && P[i].GrNr != GrNr)
	continue;
#endif

      mp[count].key = Key[i] = peano_hilbert_key((int) ((P[i].Pos[0] - DomainCorner[0]) * DomainFac),
						 (int) ((P[i].Pos[1] - DomainCorner[1]) * DomainFac),
						 (int) ((P[i].Pos[2] - DomainCorner[2]) * DomainFac),
						 BITS_PER_DIMENSION);
#ifdef SUBFIND_ALTERNATIVE_COLLECTIVE
      P[i].Key = Key[i];
#endif
      mp[count].index = i;
      count++;
    }

#ifdef SUBFIND
  if(GrNr >= 0 && count != NumPartGroup)
    endrun(1222);
#endif

#ifdef MYSORT
  mysort_domain(mp, count, sizeof(struct peano_hilbert_data));
#else
#ifndef POWER6
  qsort(mp, count, sizeof(struct peano_hilbert_data), domain_compare_key);
#else
  qsort_domain(mp, count);
#endif
#endif

  NTopnodes = 1;
  topNodes[0].Daughter = -1;
  topNodes[0].Parent = -1;
  topNodes[0].Size = PEANOCELLS;
  topNodes[0].StartKey = 0;
  topNodes[0].PIndex = 0;
  topNodes[0].Count = count;
  topNodes[0].Cost = gravcost;

  costlimit = totgravcost / (TOPNODEFACTOR * MULTIPLEDOMAINS * NTask);
  countlimit = totpartcount / (TOPNODEFACTOR * MULTIPLEDOMAINS * NTask);

  errflag = domain_check_for_local_refine(0, countlimit, costlimit);

  myfree(mp);

  MPI_Allreduce(&errflag, &errsum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  if(errsum)
    {
      if(ThisTask == 0)
	printf
	  ("We are out of Topnodes. We'll try to repeat with a higher value than All.TopNodeAllocFactor=%g\n",
	   All.TopNodeAllocFactor);
      fflush(stdout);

      return errsum;
    }


  /* we now need to exchange tree parts and combine them as needed */

  if(NTask == (1 << PTask))	/* the following algoritm only works for power of 2 */
    {
      for(ngrp = 1, errflag = 0; ngrp < (1 << PTask); ngrp <<= 1)
	{
	  sendTask = ThisTask;
	  recvTask = ThisTask ^ ngrp;

	  if(recvTask < NTask)
	    {
	      /* inform each other about the length of the trees */
	      MPI_Sendrecv(&NTopnodes, 1, MPI_INT, recvTask, TAG_GRAV_A,
			   &ntopnodes_import, 1, MPI_INT, recvTask, TAG_GRAV_A, MPI_COMM_WORLD, &status);


	      topNodes_import =
		(struct local_topnode_data *) mymalloc("topNodes_import",
						       IMAX(ntopnodes_import,
							    NTopnodes) * sizeof(struct local_topnode_data));

	      /* exchange the trees */
	      MPI_Sendrecv(topNodes,
			   NTopnodes * sizeof(struct local_topnode_data), MPI_BYTE,
			   recvTask, TAG_GRAV_B,
			   topNodes_import,
			   ntopnodes_import * sizeof(struct local_topnode_data), MPI_BYTE,
			   recvTask, TAG_GRAV_B, MPI_COMM_WORLD, &status);

	      if(sendTask > recvTask)	/* swap the two trees so that result will be equal on all cpus */
		{
		  topNodes_temp =
		    (struct local_topnode_data *) mymalloc("topNodes_temp",
							   NTopnodes * sizeof(struct local_topnode_data));
		  memcpy(topNodes_temp, topNodes, NTopnodes * sizeof(struct local_topnode_data));
		  memcpy(topNodes, topNodes_import, ntopnodes_import * sizeof(struct local_topnode_data));
		  memcpy(topNodes_import, topNodes_temp, NTopnodes * sizeof(struct local_topnode_data));
		  myfree(topNodes_temp);
		  i = NTopnodes;
		  NTopnodes = ntopnodes_import;
		  ntopnodes_import = i;
		}


	      if(errflag == 0)
		{
		  if((NTopnodes + ntopnodes_import) <= MaxTopNodes)
		    {
		      domain_insertnode(topNodes, topNodes_import, 0, 0);
		    }
		  else
		    {
		      errflag = 1;
		    }
		}

	      myfree(topNodes_import);
	    }
	}
    }
  else
    {
      errflag = domain_recursively_combine_topTree(0, NTask);
    }

  MPI_Allreduce(&errflag, &errsum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  if(errsum)
    {
      if(ThisTask == 0)
	printf("can't combine trees due to lack of storage. Will try again.\n");
      return errsum;
    }

  /* now let's see whether we should still append more nodes, based on the estimated cumulative cost/count in each cell */


#ifndef DENSITY_INDEPENDENT_SPH_DEBUG
  if(ThisTask == 0)
    printf("Before=%d\n", NTopnodes);

  for(i = 0, errflag = 0; i < NTopnodes; i++)
    {
      if(topNodes[i].Daughter < 0)
	if(topNodes[i].Count > countlimit || topNodes[i].Cost > costlimit)	/* ok, let's add nodes if we can */
	  if(topNodes[i].Size > 1)
	    {
	      if((NTopnodes + 8) <= MaxTopNodes)
		{
		  topNodes[i].Daughter = NTopnodes;

		  for(j = 0; j < 8; j++)
		    {
		      sub = topNodes[i].Daughter + j;
		      topNodes[sub].Size = (topNodes[i].Size >> 3);
		      topNodes[sub].Count = topNodes[i].Count / 8;
		      topNodes[sub].Cost = topNodes[i].Cost / 8;
		      topNodes[sub].Daughter = -1;
		      topNodes[sub].Parent = i;
		      topNodes[sub].StartKey = topNodes[i].StartKey + j * topNodes[sub].Size;
		    }

		  NTopnodes += 8;
		}
	      else
		{
		  errflag = 1;
		  break;
		}
	    }
    }

  MPI_Allreduce(&errflag, &errsum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  if(errsum)
    return errsum;

  if(ThisTask == 0)
    printf("After=%d\n", NTopnodes);
#endif
  /* count toplevel leaves */
  domain_sumCost();

  if(NTopleaves < MULTIPLEDOMAINS * NTask)
    endrun(112);

  return 0;
}



void domain_sumCost(void)
{
  int i, n, no;
  float *local_domainWork;
  int *local_domainCount;
  int *local_domainCountSph;

  local_domainWork = (float *) mymalloc("local_domainWork", NTopnodes * sizeof(float));
  local_domainCount = (int *) mymalloc("local_domainCount", NTopnodes * sizeof(int));
  local_domainCountSph = (int *) mymalloc("local_domainCountSph", NTopnodes * sizeof(int));

#ifdef LT_STELLAREVOLUTION
  int *local_domainCountStars;

  local_domainCountStars = (int *) mymalloc("local_domainCountStars", NTopnodes * sizeof(int));
#endif

  NTopleaves = 0;
  domain_walktoptree(0);

  for(i = 0; i < NTopleaves; i++)
    {
      local_domainWork[i] = 0;
      local_domainCount[i] = 0;
      local_domainCountSph[i] = 0;
#ifdef LT_STELLAREVOLUTION
      local_domainCountStars[i] = 0;
#endif
    }

  if(ThisTask == 0)
    printf("NTopleaves= %d  NTopnodes=%d (space for %d)\n", NTopleaves, NTopnodes, MaxTopNodes);

  for(n = 0; n < NumPart; n++)
    {
#ifdef SUBFIND
      if(GrNr >= 0 && P[n].GrNr != GrNr)
	continue;
#endif

      no = 0;

      while(topNodes[no].Daughter >= 0)
	no = topNodes[no].Daughter + (Key[n] - topNodes[no].StartKey) / (topNodes[no].Size >> 3);

      no = topNodes[no].Leaf;

      local_domainWork[no] += (float) domain_particle_costfactor(n);

      local_domainCount[no] += 1;
      if(P[n].Type == 0)
	local_domainCountSph[no] += 1;
#ifdef LT_STELLAREVOLUTION
      if(P[n].Type == 4)
	local_domainCountStars[no] += 1;
#endif
    }

  MPI_Allreduce(local_domainWork, domainWork, NTopleaves, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(local_domainCount, domainCount, NTopleaves, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(local_domainCountSph, domainCountSph, NTopleaves, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
#ifdef LT_STELLAREVOLUTION
  MPI_Allreduce(local_domainCountStars, domainCountStars, NTopleaves, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  myfree(local_domainCountStars);
#endif
  myfree(local_domainCountSph);
  myfree(local_domainCount);
  myfree(local_domainWork);
}


/*! This routine finds the extent of the global domain grid.
 */
void domain_findExtent(void)
{
  int i, j;
  double len, xmin[3], xmax[3], xmin_glob[3], xmax_glob[3];

  /* determine local extension */
  for(j = 0; j < 3; j++)
    {
      xmin[j] = MAX_REAL_NUMBER;
      xmax[j] = -MAX_REAL_NUMBER;
    }

  for(i = 0; i < NumPart; i++)
    {
#ifdef SUBFIND
      if(GrNr >= 0 && P[i].GrNr != GrNr)
	continue;
#endif

      for(j = 0; j < 3; j++)
	{
	  if(xmin[j] > P[i].Pos[j])
	    xmin[j] = P[i].Pos[j];

	  if(xmax[j] < P[i].Pos[j])
	    xmax[j] = P[i].Pos[j];
	}
    }

  MPI_Allreduce(xmin, xmin_glob, 3, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(xmax, xmax_glob, 3, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

  len = 0;
  for(j = 0; j < 3; j++)
    if(xmax_glob[j] - xmin_glob[j] > len)
      len = xmax_glob[j] - xmin_glob[j];

  len *= 1.001;

  for(j = 0; j < 3; j++)
    {
      DomainCenter[j] = 0.5 * (xmin_glob[j] + xmax_glob[j]);
      DomainCorner[j] = 0.5 * (xmin_glob[j] + xmax_glob[j]) - 0.5 * len;
    }

  DomainLen = len;
  DomainFac = 1.0 / len * (((peanokey) 1) << (BITS_PER_DIMENSION));
}




void domain_add_cost(struct local_topnode_data *treeA, int noA, long long count, double cost)
{
  int i, sub;
  long long countA, countB;

  countB = count / 8;
  countA = count - 7 * countB;

  cost = cost / 8;

  for(i = 0; i < 8; i++)
    {
      sub = treeA[noA].Daughter + i;

      if(i == 0)
	count = countA;
      else
	count = countB;

      treeA[sub].Count += count;
      treeA[sub].Cost += cost;

      if(treeA[sub].Daughter >= 0)
	domain_add_cost(treeA, sub, count, cost);
    }
}


void domain_insertnode(struct local_topnode_data *treeA, struct local_topnode_data *treeB, int noA, int noB)
{
  int j, sub;
  long long count, countA, countB;
  double cost, costA, costB;

  if(treeB[noB].Size < treeA[noA].Size)
    {
      if(treeA[noA].Daughter < 0)
	{
	  if((NTopnodes + 8) <= MaxTopNodes)
	    {
	      count = treeA[noA].Count - treeB[treeB[noB].Parent].Count;
	      countB = count / 8;
	      countA = count - 7 * countB;

	      cost = treeA[noA].Cost - treeB[treeB[noB].Parent].Cost;
	      costB = cost / 8;
	      costA = cost - 7 * costB;

	      treeA[noA].Daughter = NTopnodes;
	      for(j = 0; j < 8; j++)
		{
		  if(j == 0)
		    {
		      count = countA;
		      cost = costA;
		    }
		  else
		    {
		      count = countB;
		      cost = costB;
		    }

		  sub = treeA[noA].Daughter + j;
		  topNodes[sub].Size = (treeA[noA].Size >> 3);
		  topNodes[sub].Count = count;
		  topNodes[sub].Cost = cost;
		  topNodes[sub].Daughter = -1;
		  topNodes[sub].Parent = noA;
		  topNodes[sub].StartKey = treeA[noA].StartKey + j * treeA[sub].Size;
		}
	      NTopnodes += 8;
	    }
	  else
	    endrun(88);
	}

      sub = treeA[noA].Daughter + (treeB[noB].StartKey - treeA[noA].StartKey) / (treeA[noA].Size >> 3);
      domain_insertnode(treeA, treeB, sub, noB);
    }
  else if(treeB[noB].Size == treeA[noA].Size)
    {
      treeA[noA].Count += treeB[noB].Count;
      treeA[noA].Cost += treeB[noB].Cost;

      if(treeB[noB].Daughter >= 0)
	{
	  for(j = 0; j < 8; j++)
	    {
	      sub = treeB[noB].Daughter + j;
	      domain_insertnode(treeA, treeB, noA, sub);
	    }
	}
      else
	{
	  if(treeA[noA].Daughter >= 0)
	    domain_add_cost(treeA, noA, treeB[noB].Count, treeB[noB].Cost);
	}
    }
  else
    endrun(89);
}



static void msort_domain_with_tmp(struct peano_hilbert_data *b, size_t n, struct peano_hilbert_data *t)
{
  struct peano_hilbert_data *tmp;
  struct peano_hilbert_data *b1, *b2;
  size_t n1, n2;

  if(n <= 1)
    return;

  n1 = n / 2;
  n2 = n - n1;
  b1 = b;
  b2 = b + n1;

  msort_domain_with_tmp(b1, n1, t);
  msort_domain_with_tmp(b2, n2, t);

  tmp = t;

  while(n1 > 0 && n2 > 0)
    {
      if(b1->key <= b2->key)
	{
	  --n1;
	  *tmp++ = *b1++;
	}
      else
	{
	  --n2;
	  *tmp++ = *b2++;
	}
    }

  if(n1 > 0)
    memcpy(tmp, b1, n1 * sizeof(struct peano_hilbert_data));

  memcpy(b, t, (n - n2) * sizeof(struct peano_hilbert_data));
}

void mysort_domain(void *b, size_t n, size_t s)
{
  const size_t size = n * s;
  struct peano_hilbert_data *tmp;

  tmp = (struct peano_hilbert_data *) mymalloc("tmp", size);

  msort_domain_with_tmp((struct peano_hilbert_data *) b, n, tmp);

  myfree(tmp);
}
