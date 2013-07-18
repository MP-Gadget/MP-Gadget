#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <gsl/gsl_rng.h>

#ifdef SUBFIND

#include "fof.h"

#include "allvars.h"
#include "proto.h"
#include "domain.h"
#include "subfind.h"

static struct id_list
{
  MyIDType ID;
  int GrNr;
  int SubNr;
  float BindingEgy;

#ifdef SUBFIND_SAVE_PARTICLELISTS
  float Pos[3];
  float Vel[3];
  int Type;
#ifdef STELLARAGE
  float Mass;
  float StellarAge;
#endif
#endif
}
 *ID_list;

static int Nids;


void subfind(int num)
{
  double t0, t1, tstart, tend;
  int i, gr, nlocid, offset, limit, ncount;

#ifdef DENSITY_SPLIT_BY_TYPE
  struct unbind_data *d;
  int j, n, count[6], countall[6];
  double a3inv;
#endif

  if(ThisTask == 0)
    printf("\nWe now execute a parallel version of SUBFIND.\n");

  tstart = second();

#ifdef DENSITY_SPLIT_BY_TYPE
  if(All.ComovingIntegrationOn)
    a3inv = 1 / (All.Time * All.Time * All.Time);
  else
    a3inv = 1;

  for(j = 0; j < 6; j++)
    count[j] = 0;

  /* let's count number of particles of selected species */
  for(i = 0; i < NumPart; i++)
    count[P[i].Type]++;

  MPI_Allreduce(count, countall, 6, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  /* do first loop: basically just defining the hsml for different species */
  for(j = 0; j < 6; j++)
    {
      if((1 << j) & (DENSITY_SPLIT_BY_TYPE))
	{
#ifdef BLACK_HOLES
	  if(j == 5)
	    countall[j] = 0;	/* this will prevent that the black holes are treated separately */
#endif

	  force_treeallocate((int) (All.TreeAllocFactor * All.MaxPart) + NTopnodes, All.MaxPart);

	  if(countall[j] > All.DesNumNgb)
	    {
	      /* build index list of particles of selectes species */
	      d = (struct unbind_data *) mymalloc("	      d", count[j] * sizeof(struct unbind_data));
	      for(i = 0, n = 0; i < NumPart; i++)
		if(P[i].Type == j)
		  d[n++].index = i;

	      t0 = second();
	      if(ThisTask == 0)
		printf("Tree construction for species %d (%d).\n", j, countall[j]);

	      CPU_Step[CPU_FOF] += measure_time();

	      force_treebuild(count[j], d);

	      myfree(d);

	      t1 = second();
	      if(ThisTask == 0)
		printf("tree build for species %d took %g sec\n", j, timediff(t0, t1));
	    }
	  else
	    {
	      t0 = second();
	      if(ThisTask == 0)
		printf("Tree construction.\n");

	      CPU_Step[CPU_FOF] += measure_time();

	      force_treebuild(NumPart, NULL);

	      t1 = second();
	      if(ThisTask == 0)
		printf("tree build took %g sec\n", timediff(t0, t1));
	    }


	  /* let's determine the local densities */
	  t0 = second();
	  subfind_setup_smoothinglengths(j);
	  subfind_density(j);
	  t1 = second();
	  if(ThisTask == 0)
	    printf("density and smoothing length for species %d took %g sec\n", j, timediff(t0, t1));

	  force_treefree();

	  /* let's save density contribution of own species */
	  for(i = 0; i < NumPart; i++)
	    if(P[i].Type == j)
	      P[i].w.density_sum = P[i].u.DM_Density;

	}
    }

  /* do second loop: now calculate all density contributions */
  for(j = 0; j < 6; j++)
    {
      if((1 << j) & (DENSITY_SPLIT_BY_TYPE))
	{
	  force_treeallocate((int) (All.TreeAllocFactor * All.MaxPart) + NTopnodes, All.MaxPart);

	  /* build index list of particles of selectes species */
	  d = (struct unbind_data *) mymalloc("	  d", count[j] * sizeof(struct unbind_data));
	  for(i = 0, n = 0; i < NumPart; i++)
	    if(P[i].Type == j)
	      d[n++].index = i;

	  t0 = second();
	  if(ThisTask == 0)
	    printf("Tree construction for species %d (%d).\n", j, countall[j]);

	  CPU_Step[CPU_FOF] += measure_time();

	  force_treebuild(count[j], d);

	  myfree(d);

	  t1 = second();
	  if(ThisTask == 0)
	    printf("tree build for species %d took %g sec\n", j, timediff(t0, t1));

	  /* let's determine the local densities */
	  t0 = second();
	  for(i = 0; i < 6; i++)
	    if((1 << i) & (DENSITY_SPLIT_BY_TYPE))
	      if(j != i)
		{
		  if(countall[i] > All.DesNumNgb)
		    {
		      if(ThisTask == 0)
			printf("calculating density contribution of species %d to species %d\n", j, i);
		      subfind_density(-(i + 1));
		    }
		}
	  t1 = second();
	  if(ThisTask == 0)
	    printf("density() of species %d took %g sec\n", j, timediff(t0, t1));

	  force_treefree();

	  /* let's sum up density contribution */
	  for(i = 0; i < NumPart; i++)
	    if((1 << P[i].Type) & (DENSITY_SPLIT_BY_TYPE))
	      if(j != P[i].Type)
		if(countall[P[i].Type] > All.DesNumNgb)
		  P[i].w.density_sum += P[i].u.DM_Density;
	}
    }


  for(i = 0; i < NumPart; i++)
    {
      P[i].u.DM_Density = P[i].w.density_sum;

      if(P[i].Type == 0)
	P[i].w.int_energy = DMAX(All.MinEgySpec,
				 SphP[i].Entropy / GAMMA_MINUS1 * pow(SphP[i].d.Density * a3inv,
								      GAMMA_MINUS1));
      else
	P[i].w.int_energy = 0;

    }
#else
  force_treeallocate((int) (All.TreeAllocFactor * All.MaxPart) + NTopnodes, All.MaxPart);

  t0 = second();
  if(ThisTask == 0)
    printf("Tree construction.\n");

  CPU_Step[CPU_FOF] += measure_time();

  force_treebuild(NumPart, NULL);

  t1 = second();
  if(ThisTask == 0)
    printf("tree build took %g sec\n", timediff(t0, t1));


  /* let's determine the local dark matter densities */
  t0 = second();
  subfind_setup_smoothinglengths();
  subfind_density();
  t1 = second();
  if(ThisTask == 0)
    printf("dark matter density() took %g sec\n", timediff(t0, t1));

  force_treefree();
#endif /* DENSITY_SPLIT_BY_TYPE */

#ifndef SUBFIND_DENSITY_AND_POTENTIAL
  if(DumpFlag)
    {
      /* let's save the densities to a file (for making images) */
      t0 = second();
      subfind_save_densities(num);
      t1 = second();
      if(ThisTask == 0)
	printf("saving densities took %g sec\n", timediff(t0, t1));
    }
#endif
#ifdef ONLY_PRODUCE_HSML_FILES
  return;
#endif

  /* count how many groups we have that should be done collectively */
  limit = 0.6 * All.TotNumPart / NTask;


  for(i = 0, ncount = 0; i < Ngroups; i++)
    if(Group[i].Len >= limit)
      ncount++;
  MPI_Allreduce(&ncount, &Ncollective, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  if(ThisTask == 0)
    {
      printf("\nNumber of FOF halos treated with collective SubFind code = %d\n", Ncollective);
      printf("(the adopted size-limit for the collective algorithm was %d particles.)\n", limit);
      printf("the other %d FOF halos are treated in parallel with serial code\n\n", TotNgroups - Ncollective);
    }

  /*  to decide on which task a group should be:
   *  if   GrNr <= Ncollective:  collective groupfinding.
   *  the task where the group info is put is TaskNr = (GrNr - 1) % NTask
   */

  /* now we distribute the particles such that small groups are assigned in
   *  total to certain CPUs, and big groups are left where they are 
   */

  t0 = second();

  for(i = 0; i < NumPart; i++)
    {
      P[i].origintask2 = ThisTask;

      if(P[i].GrNr > Ncollective && P[i].GrNr <= TotNgroups)	/* particle is in small group */
	P[i].targettask = (P[i].GrNr - 1) % NTask;
      else
	P[i].targettask = ThisTask;
    }

  subfind_exchange();		/* distributes gas particles as well if needed */

  t1 = second();
  if(ThisTask == 0)
    printf("subfind_exchange()() took %g sec\n", timediff(t0, t1));

  subfind_distribute_groups();

  qsort(Group, Ngroups, sizeof(struct group_properties), fof_compare_Group_GrNr);


  for(i = 0; i < NumPart; i++)
    if(P[i].GrNr > Ncollective && P[i].GrNr <= TotNgroups)
      if(((P[i].GrNr - 1) % NTask) != ThisTask)
	{
	  printf("i=%d %d task=%d\n", i, P[i].GrNr, ThisTask);
	  endrun(87);
	}

  /* lets estimate the maximum number of substructures we need to store on the local CPU */
  for(i = 0, nlocid = 0; i < Ngroups; i++)
    nlocid += Group[i].Len;

  MaxNsubgroups = nlocid / All.DesLinkNgb;	/* this is a quite conservative upper limit */
  Nsubgroups = 0;
  SubGroup =
    (struct subgroup_properties *) mymalloc("SubGroup", MaxNsubgroups * sizeof(struct subgroup_properties));

  for(i = 0; i < NumPart; i++)
    P[i].SubNr = (1 << 30);	/* default */

  /* we begin by applying the collective version of subfind to distributed groups */
  t0 = second();
  for(GrNr = 1; GrNr <= Ncollective; GrNr++)
    subfind_process_group_collectively(num);
  t1 = second();
  if(ThisTask == 0)
    printf("processing of collective halos took %g sec\n", timediff(t0, t1));

#ifdef SUBFIND_COLLECTIVE_STAGE1
  if(ThisTask == 0)
    printf("stage 1 ended\n");
  endrun(0);
#endif

  for(i = 0; i < NumPart; i++)
    {
      P[i].origindex = i;
      P[i].origintask = ThisTask;
    }

  t0 = second();
  qsort(P, NumPart, sizeof(struct particle_data), subfind_compare_P_GrNr_DM_Density);
  t1 = second();
  if(ThisTask == 0)
    printf("sort of local particles()() took %g sec\n", timediff(t0, t1));


  /* now we have the particles of groups consecutively, but SPH particles are
     not aligned. They can however be accessed via SphP[P[i].originindex] */


  /* let's count how many local particles we have in small groups */
  for(i = 0, nlocid = 0; i < NumPart; i++)
    if(P[i].GrNr > Ncollective && P[i].GrNr <= Ngroups)	/* particle is in small group */
      nlocid++;

  if(ThisTask == 0)
    printf("contructing tree for serial subfind of local groups\n");

  subfind_loctree_treeallocate((int) (All.TreeAllocFactor * All.MaxPart) + NTopnodes, All.MaxPart);

  if(ThisTask == 0)
    printf("Start to do local groups with serial subfind algorithm\n");

  t0 = second();

  /* we now apply a serial version of subfind to the local groups */
  for(gr = 0, offset = 0; gr < Ngroups; gr++)
    {
      if(Group[gr].GrNr > Ncollective)
	{
	  if(((Group[gr].GrNr - 1) % NTask) == ThisTask)
	    offset = subfind_process_group_serial(gr, offset);
	}
    }

  MPI_Barrier(MPI_COMM_WORLD);

  t1 = second();
  if(ThisTask == 0)
    printf("\nprocessing of local groups took took %g sec\n\n", timediff(t0, t1));


  subfind_loctree_treefree();


  /* bringing back particles in original positions, such that gas particles are aligned */
  t0 = second();
  qsort(P, NumPart, sizeof(struct particle_data), subfind_compare_P_origindex);
  t1 = second();
  if(ThisTask == 0)
    printf("unsorting of local particles()() took %g sec\n", timediff(t0, t1));


  GrNr = -1;			/* to ensure that domain decomposition acts normally again */

  /* now determine the remaining spherical overdensity values for the non-local groups */

  domain_free_trick();

  CPU_Step[CPU_FOF] += measure_time();


#ifdef DENSITY_SPLIT_BY_TYPE
  printf("Task %d: testing particles ...\n", ThisTask);
  for(i = 0; i < NumPart; i++)
    {
      if(P[i].origintask != ThisTask)
	printf("Task %d: Holding particle of task %d !\n", ThisTask, P[i].origintask);
      if(P[i].origindex != i)
	printf("Task %d: Particles is in wrong position (is=%d, was=%d) !\n", ThisTask, i, P[i].origindex);
    }
#endif



  t0 = second();

  for(i = 0; i < NumPart; i++)
    P[i].targettask = P[i].origintask2;

  subfind_exchange();		/* distributes gas particles as well if needed */

  t1 = second();
  if(ThisTask == 0)
    printf("subfind_exchange() (for return to original CPU)  took %g sec\n", timediff(t0, t1));



  All.DoDynamicUpdate = 0;
  domain_Decomposition();

  force_treebuild(NumPart, NULL);


  /* compute spherical overdensities for FOF groups */
  t0 = second();

  subfind_overdensity();

  t1 = second();
  if(ThisTask == 0)
    printf("determining spherical overdensity masses took %g sec\n", timediff(t0, t1));


  /* determine which halos are contaminated by boundary particles */
  t0 = second();

  subfind_contamination();

  t1 = second();
  if(ThisTask == 0)
    printf("determining contamination of halos took %g sec\n", timediff(t0, t1));


  force_treefree();
  domain_free();

  domain_allocate_trick();

  /* now assemble final output */
  subfind_save_final(num);

  tend = second();

  if(ThisTask == 0)
    printf("\nFinished with SUBFIND.  (total time=%g sec)\n\n", timediff(tstart, tend));

  myfree(SubGroup);

  CPU_Step[CPU_FOF] += measure_time();
}




void subfind_save_final(int num)
{
  int i, j, totsubs, masterTask, groupTask, nprocgroup;
  char buf[1000];
  double t0, t1;

  /* prepare list of ids with assigned group numbers */

  parallel_sort(Group, Ngroups, sizeof(struct group_properties), fof_compare_Group_GrNr);
  parallel_sort(SubGroup, Nsubgroups, sizeof(struct subgroup_properties),
		subfind_compare_SubGroup_GrNr_SubNr);

  ID_list = mymalloc("ID_list", sizeof(struct id_list) * NumPart);

  for(i = 0, Nids = 0; i < NumPart; i++)
    {
      if(P[i].GrNr <= TotNgroups)
	{
	  ID_list[Nids].GrNr = P[i].GrNr;
	  ID_list[Nids].SubNr = P[i].SubNr;
	  ID_list[Nids].BindingEgy = P[i].v.DM_BindingEnergy;
	  ID_list[Nids].ID = P[i].ID;
#ifdef SUBFIND_SAVE_PARTICLELISTS
	  for(j = 0; j < 3; j++)
	    {
	      ID_list[Nids].Pos[j] = P[i].Pos[j];
	      ID_list[Nids].Vel[j] = P[i].Vel[j];
	    }
	  ID_list[Nids].Type = P[i].Type;
#ifdef STELLARAGE
	  ID_list[Nids].Mass = P[i].Mass;
	  if(P[i].Type == 4)
	    ID_list[Nids].StellarAge = P[i].StellarAge;
	  else
	    ID_list[Nids].StellarAge = 0;
#endif
#endif
	  Nids++;
	}
    }

  parallel_sort(ID_list, Nids, sizeof(struct id_list), subfind_compare_ID_list);


  MPI_Allreduce(&Nsubgroups, &TotNsubgroups, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);


  /* fill in the FirstSub-values */
  for(i = 0, totsubs = 0; i < Ngroups; i++)
    {
      if(i > 0)
	Group[i].FirstSub = Group[i - 1].FirstSub + Group[i - 1].Nsubs;
      else
	Group[i].FirstSub = 0;
      totsubs += Group[i].Nsubs;
    }

  MPI_Allgather(&totsubs, 1, MPI_INT, Send_count, 1, MPI_INT, MPI_COMM_WORLD);
  for(j = 1, Send_offset[0] = 0; j < NTask; j++)
    Send_offset[j] = Send_offset[j - 1] + Send_count[j - 1];

  for(i = 0; i < Ngroups; i++)
    Group[i].FirstSub += Send_offset[ThisTask];




  MPI_Allgather(&Nids, 1, MPI_INT, Send_count, 1, MPI_INT, MPI_COMM_WORLD);
  for(j = 1, Send_offset[0] = 0; j < NTask; j++)
    Send_offset[j] = Send_offset[j - 1] + Send_count[j - 1];

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

  t0 = second();

  nprocgroup = NTask / All.NumFilesWrittenInParallel;
  if((NTask % All.NumFilesWrittenInParallel))
    nprocgroup++;
  masterTask = (ThisTask / nprocgroup) * nprocgroup;
  for(groupTask = 0; groupTask < nprocgroup; groupTask++)
    {
      if(ThisTask == (masterTask + groupTask))	/* ok, it's this processor's turn */
	subfind_save_local_catalogue(num);
      MPI_Barrier(MPI_COMM_WORLD);	/* wait inside the group */
    }

  t1 = second();

  if(ThisTask == 0)
    {
      printf("Subgroup catalogues saved. took = %g sec\n", timediff(t0, t1));
      fflush(stdout);
    }

  myfree(ID_list);
}


void subfind_save_local_catalogue(int num)
{
  FILE *fd;
  char buf[500], fname[500];
  float *mass, *pos, *vel, *spin;

#ifdef SAVE_MASS_TAB
  float *masstab;
#endif
  int i, j, *len;
  MyIDType *ids;

  sprintf(fname, "%s/groups_%03d/%s_%03d.%d", All.OutputDir, num, "subhalo_tab", num, ThisTask);
  strcpy(buf, fname);
  if(!(fd = fopen(buf, "w")))
    {
      printf("can't open file `%s`\n", buf);
      endrun(1183);
    }

  my_fwrite(&Ngroups, sizeof(int), 1, fd);
  my_fwrite(&TotNgroups, sizeof(int), 1, fd);
  my_fwrite(&Nids, sizeof(int), 1, fd);
  my_fwrite(&TotNids, sizeof(long long), 1, fd);
  my_fwrite(&NTask, sizeof(int), 1, fd);
  my_fwrite(&Nsubgroups, sizeof(int), 1, fd);
  my_fwrite(&TotNsubgroups, sizeof(int), 1, fd);

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

  /* location (potential minimum) */
  pos = mymalloc("pos", Ngroups * 3 * sizeof(float));
  for(i = 0; i < Ngroups; i++)
    for(j = 0; j < 3; j++)
      pos[i * 3 + j] = Group[i].Pos[j];
  my_fwrite(pos, Ngroups, 3 * sizeof(float), fd);
  myfree(pos);

  /* M_Mean200 */
  mass = mymalloc("mass", Ngroups * sizeof(float));
  for(i = 0; i < Ngroups; i++)
    mass[i] = Group[i].M_Mean200;
  my_fwrite(mass, Ngroups, sizeof(float), fd);
  myfree(mass);

  /* R_Mean200 */
  mass = mymalloc("mass", Ngroups * sizeof(float));
  for(i = 0; i < Ngroups; i++)
    mass[i] = Group[i].R_Mean200;
  my_fwrite(mass, Ngroups, sizeof(float), fd);
  myfree(mass);

  /* M_Crit200 */
  mass = mymalloc("mass", Ngroups * sizeof(float));
  for(i = 0; i < Ngroups; i++)
    mass[i] = Group[i].M_Crit200;
  my_fwrite(mass, Ngroups, sizeof(float), fd);
  myfree(mass);

  /* R_Crit200 */
  mass = mymalloc("mass", Ngroups * sizeof(float));
  for(i = 0; i < Ngroups; i++)
    mass[i] = Group[i].R_Crit200;
  my_fwrite(mass, Ngroups, sizeof(float), fd);
  myfree(mass);

  /* M_TopHat200 */
  mass = mymalloc("mass", Ngroups * sizeof(float));
  for(i = 0; i < Ngroups; i++)
    mass[i] = Group[i].M_TopHat200;
  my_fwrite(mass, Ngroups, sizeof(float), fd);
  myfree(mass);

  /* R_TopHat200 */
  mass = mymalloc("mass", Ngroups * sizeof(float));
  for(i = 0; i < Ngroups; i++)
    mass[i] = Group[i].R_TopHat200;
  my_fwrite(mass, Ngroups, sizeof(float), fd);
  myfree(mass);

#ifdef SO_VEL_DISPERSIONS
  /* VelDisp_Mean200 */
  mass = mymalloc("mass", Ngroups * sizeof(float));
  for(i = 0; i < Ngroups; i++)
    mass[i] = Group[i].VelDisp_Mean200;
  my_fwrite(mass, Ngroups, sizeof(float), fd);
  myfree(mass);

  /* VelDisp_Crit200 */
  mass = mymalloc("mass", Ngroups * sizeof(float));
  for(i = 0; i < Ngroups; i++)
    mass[i] = Group[i].VelDisp_Crit200;
  my_fwrite(mass, Ngroups, sizeof(float), fd);
  myfree(mass);

  /* VelDisp_TopHat200 */
  mass = mymalloc("mass", Ngroups * sizeof(float));
  for(i = 0; i < Ngroups; i++)
    mass[i] = Group[i].VelDisp_TopHat200;
  my_fwrite(mass, Ngroups, sizeof(float), fd);
  myfree(mass);
#endif

  /* contamination particle count */
  len = mymalloc("len", Ngroups * sizeof(int));
  for(i = 0; i < Ngroups; i++)
    len[i] = Group[i].ContaminationLen;
  my_fwrite(len, Ngroups, sizeof(int), fd);
  myfree(len);

  /* contamination mass */
  mass = mymalloc("mass", Ngroups * sizeof(float));
  for(i = 0; i < Ngroups; i++)
    mass[i] = Group[i].ContaminationMass;
  my_fwrite(mass, Ngroups, sizeof(float), fd);
  myfree(mass);

  /* number of substructures in FOF group  */
  len = mymalloc("len", Ngroups * sizeof(int));
  for(i = 0; i < Ngroups; i++)
    len[i] = Group[i].Nsubs;
  my_fwrite(len, Ngroups, sizeof(int), fd);
  myfree(len);

  /* first substructure in FOF group  */
  len = mymalloc("len", Ngroups * sizeof(int));
  for(i = 0; i < Ngroups; i++)
    len[i] = Group[i].FirstSub;
  my_fwrite(len, Ngroups, sizeof(int), fd);
  myfree(len);

  /* ------------------------------ */

  /* Len of substructure  */
  len = mymalloc("len", Nsubgroups * sizeof(int));
  for(i = 0; i < Nsubgroups; i++)
    len[i] = SubGroup[i].Len;
  my_fwrite(len, Nsubgroups, sizeof(int), fd);
  myfree(len);

  /* offset of substructure  */
  len = mymalloc("len", Nsubgroups * sizeof(int));
  for(i = 0; i < Nsubgroups; i++)
    len[i] = SubGroup[i].Offset;
  my_fwrite(len, Nsubgroups, sizeof(int), fd);
  myfree(len);

  /* parent of substructure  */
  len = mymalloc("len", Nsubgroups * sizeof(int));
  for(i = 0; i < Nsubgroups; i++)
    len[i] = SubGroup[i].SubParent;
  my_fwrite(len, Nsubgroups, sizeof(int), fd);
  myfree(len);

  /* Mass of substructure  */
  mass = mymalloc("mass", Nsubgroups * sizeof(float));
  for(i = 0; i < Nsubgroups; i++)
    mass[i] = SubGroup[i].Mass;
  my_fwrite(mass, Nsubgroups, sizeof(float), fd);
  myfree(mass);

  /* Pos of substructure  */
  pos = mymalloc("pos", Nsubgroups * 3 * sizeof(float));
  for(i = 0; i < Nsubgroups; i++)
    for(j = 0; j < 3; j++)
      pos[i * 3 + j] = SubGroup[i].Pos[j];
  my_fwrite(pos, Nsubgroups, 3 * sizeof(float), fd);
  myfree(pos);

  /* Vel of substructure  */
  vel = mymalloc("vel", Nsubgroups * 3 * sizeof(float));
  for(i = 0; i < Nsubgroups; i++)
    for(j = 0; j < 3; j++)
      vel[i * 3 + j] = SubGroup[i].Vel[j];
  my_fwrite(vel, Nsubgroups, 3 * sizeof(float), fd);
  myfree(vel);

  /* Center of mass of substructure  */
  pos = mymalloc("pos", Nsubgroups * 3 * sizeof(float));
  for(i = 0; i < Nsubgroups; i++)
    for(j = 0; j < 3; j++)
      pos[i * 3 + j] = SubGroup[i].CM[j];
  my_fwrite(pos, Nsubgroups, 3 * sizeof(float), fd);
  myfree(pos);

  /* Spin of substructure  */
  spin = mymalloc("spin", Nsubgroups * 3 * sizeof(float));
  for(i = 0; i < Nsubgroups; i++)
    for(j = 0; j < 3; j++)
      spin[i * 3 + j] = SubGroup[i].Spin[j];
  my_fwrite(spin, Nsubgroups, 3 * sizeof(float), fd);
  myfree(spin);

  /* velocity dispesion  */
  mass = mymalloc("mass", Nsubgroups * sizeof(float));
  for(i = 0; i < Nsubgroups; i++)
    mass[i] = SubGroup[i].SubVelDisp;
  my_fwrite(mass, Nsubgroups, sizeof(float), fd);
  myfree(mass);

  /* maximum circular velocity  */
  mass = mymalloc("mass", Nsubgroups * sizeof(float));
  for(i = 0; i < Nsubgroups; i++)
    mass[i] = SubGroup[i].SubVmax;
  my_fwrite(mass, Nsubgroups, sizeof(float), fd);
  myfree(mass);

  /* radius of maximum circular velocity  */
  mass = mymalloc("mass", Nsubgroups * sizeof(float));
  for(i = 0; i < Nsubgroups; i++)
    mass[i] = SubGroup[i].SubVmaxRad;
  my_fwrite(mass, Nsubgroups, sizeof(float), fd);
  myfree(mass);

  /* radius of half the mass  */
  mass = mymalloc("mass", Nsubgroups * sizeof(float));
  for(i = 0; i < Nsubgroups; i++)
    mass[i] = SubGroup[i].SubHalfMass;
  my_fwrite(mass, Nsubgroups, sizeof(float), fd);
  myfree(mass);

  /* ID of most bound particle  */
  ids = mymalloc("ids", Nsubgroups * sizeof(MyIDType));
  for(i = 0; i < Nsubgroups; i++)
    ids[i] = SubGroup[i].SubMostBoundID;
  my_fwrite(ids, Nsubgroups, sizeof(MyIDType), fd);
  myfree(ids);

  /* GrNr of substructure  */
  len = mymalloc("len", Nsubgroups * sizeof(int));
  for(i = 0; i < Nsubgroups; i++)
    len[i] = SubGroup[i].GrNr;
  my_fwrite(len, Nsubgroups, sizeof(int), fd);
  myfree(len);

  /* Masstab of substructure  */
#ifdef SAVE_MASS_TAB
  masstab = mymalloc("masstab", Nsubgroups * 6 * sizeof(float));
  for(i = 0; i < Nsubgroups; i++)
    for(j = 0; j < 6; j++)
      masstab[i * 6 + j] = SubGroup[i].MassTab[j];
  my_fwrite(masstab, Nsubgroups, 6 * sizeof(float), fd);
  myfree(masstab);
#endif

  fclose(fd);


#ifdef SUBFIND_SAVE_PARTICLELISTS
  sprintf(buf, "%s/groups_%03d/%s_%03d.%d", All.OutputDir, num, "subhalo_posvel", num, ThisTask);
  if(!(fd = fopen(buf, "w")))
    {
      printf("can't open file `%s`\n", buf);
      endrun(1184);
    }

  my_fwrite(&Ngroups, sizeof(int), 1, fd);
  my_fwrite(&TotNgroups, sizeof(int), 1, fd);
  my_fwrite(&Nids, sizeof(int), 1, fd);
  my_fwrite(&TotNids, sizeof(long long), 1, fd);
  my_fwrite(&NTask, sizeof(int), 1, fd);
  my_fwrite(&Send_offset[ThisTask], sizeof(int), 1, fd);
  my_fwrite(&All.Time, sizeof(double), 1, fd);

  float *posdata;
  double a3inv;
  char *types;

  if(All.ComovingIntegrationOn)
    a3inv = 1.0 / (All.Time * All.Time * All.Time);
  else
    a3inv = 1.0;

  posdata = (float *) mymalloc("posdata", 3 * Nids * sizeof(float));

  for(i = 0; i < Nids; i++)
    for(j = 0; j < 3; j++)
      posdata[i * 3 + j] = ID_list[i].Pos[j];

  my_fwrite(posdata, 3 * sizeof(float), Nids, fd);

  for(i = 0; i < Nids; i++)
    for(j = 0; j < 3; j++)
      posdata[i * 3 + j] = ID_list[i].Vel[j] * sqrt(a3inv);

  my_fwrite(posdata, 3 * sizeof(float), Nids, fd);

  types = (char *) posdata;

  for(i = 0; i < Nids; i++)
    types[i] = ID_list[i].Type;

  my_fwrite(types, sizeof(char), Nids, fd);


#ifdef STELLARAGE
  for(i = 0; i < Nids; i++)
    posdata[i] = ID_list[i].Mass;

  my_fwrite(posdata, sizeof(float), Nids, fd);

  for(i = 0; i < Nids; i++)
    posdata[i] = ID_list[i].StellarAge;

  my_fwrite(posdata, sizeof(float), Nids, fd);
#endif


  myfree(posdata);

  fclose(fd);
#endif



  ids = (MyIDType *) ID_list;

  for(i = 0; i < Nids; i++)
    ids[i] = ID_list[i].ID;

  sprintf(buf, "%s/groups_%03d/%s_%03d.%d", All.OutputDir, num, "subhalo_ids", num, ThisTask);
  if(!(fd = fopen(buf, "w")))
    {
      printf("can't open file `%s`\n", buf);
      endrun(1184);
    }

  my_fwrite(&Ngroups, sizeof(int), 1, fd);
  my_fwrite(&TotNgroups, sizeof(int), 1, fd);
  my_fwrite(&Nids, sizeof(int), 1, fd);
  my_fwrite(&TotNids, sizeof(long long), 1, fd);
  my_fwrite(&NTask, sizeof(int), 1, fd);
  my_fwrite(&Send_offset[ThisTask], sizeof(int), 1, fd);
  my_fwrite(ids, sizeof(MyIDType), Nids, fd);

  fclose(fd);
}

int subfind_compare_ID_list(const void *a, const void *b)
{
  if(((struct id_list *) a)->GrNr < ((struct id_list *) b)->GrNr)
    return -1;

  if(((struct id_list *) a)->GrNr > ((struct id_list *) b)->GrNr)
    return +1;

  if(((struct id_list *) a)->SubNr < ((struct id_list *) b)->SubNr)
    return -1;

  if(((struct id_list *) a)->SubNr > ((struct id_list *) b)->SubNr)
    return +1;

  if(((struct id_list *) a)->BindingEgy < ((struct id_list *) b)->BindingEgy)
    return -1;

  if(((struct id_list *) a)->BindingEgy > ((struct id_list *) b)->BindingEgy)
    return +1;

  return 0;
}

int subfind_compare_SubGroup_GrNr_SubNr(const void *a, const void *b)
{
  if(((struct subgroup_properties *) a)->GrNr < ((struct subgroup_properties *) b)->GrNr)
    return -1;

  if(((struct subgroup_properties *) a)->GrNr > ((struct subgroup_properties *) b)->GrNr)
    return +1;

  if(((struct subgroup_properties *) a)->SubNr < ((struct subgroup_properties *) b)->SubNr)
    return -1;

  if(((struct subgroup_properties *) a)->SubNr > ((struct subgroup_properties *) b)->SubNr)
    return +1;

  return 0;
}


int subfind_compare_P_GrNr_DM_Density(const void *a, const void *b)
{
  if(((struct particle_data *) a)->GrNr < (((struct particle_data *) b)->GrNr))
    return -1;

  if(((struct particle_data *) a)->GrNr > (((struct particle_data *) b)->GrNr))
    return +1;

  if(((struct particle_data *) a)->u.DM_Density > (((struct particle_data *) b)->u.DM_Density))
    return -1;

  if(((struct particle_data *) a)->u.DM_Density < (((struct particle_data *) b)->u.DM_Density))
    return +1;

  return 0;
}


int subfind_compare_P_origindex(const void *a, const void *b)
{
  if(((struct particle_data *) a)->origindex < (((struct particle_data *) b)->origindex))
    return -1;

  if(((struct particle_data *) a)->origindex > (((struct particle_data *) b)->origindex))
    return +1;

  return 0;
}


#endif
