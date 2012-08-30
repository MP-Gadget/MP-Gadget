#include <mpi.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef INVARIANCETEST
#define DO_NOT_REDEFINE_MPI_COMM_WORLD
#endif

#include "allvars.h"
#include "proto.h"


#ifdef INVARIANCETEST

#define ACC 1.0e-25

#define NMAX_CHECK  10000	/* bunch size for communication */

#define MAXPRINT    1000	/* maximum number of differences printed out */

struct combined_data
{
  struct particle_data P;
  struct sph_particle_data SphP;

  int Task;
  int Index;

} *PP, *PP1, *PP2;



int cmp_partitions_compare_id(const void *a, const void *b);
int compare_particles(int ncheck, int count);


void compare_partitions(void)
{
  int n, count = 0, *numpart_list;
  int i, nskip, cpu1, cpu2, ncheck, nstill_to_check, nget, nobtained1, nobtained2, numPart1, numPart2;

  MPI_Barrier(MPI_COMM_WORLD);
  fflush(stdout);

  if(World_ThisTask == 0)
    printf("\nStart COMPARE of particle data between partitions\n\n");
  fflush(stdout);
  MPI_Barrier(MPI_COMM_WORLD);


  PP = (struct combined_data *) mymalloc("PP", NumPart * sizeof(struct combined_data));


  for(n = 0; n < NumPart; n++)
    {
      PP[n].Task = World_ThisTask;
      PP[n].Index = n;

      PP[n].P = P[n];

      if(P[n].Type == 0)
	PP[n].SphP = SphP[n];
    }


  parallel_sort(PP, NumPart, sizeof(struct combined_data), cmp_partitions_compare_id);


  numpart_list = (int *) mymalloc("numpart_list", sizeof(int) * World_NTask);

  MPI_Allgather(&NumPart, 1, MPI_INT, numpart_list, 1, MPI_INT, MPI_COMM_WORLD);

  nskip = 0;

  if(ThisTask == 0 && World_ThisTask == 0)
    {
      for(n = 0, numPart1 = 0; n < INVARIANCETEST_SIZE1; n++)
	numPart1 += numpart_list[n];

      for(n = INVARIANCETEST_SIZE1, numPart2 = 0; n < INVARIANCETEST_SIZE1 + INVARIANCETEST_SIZE2; n++)
	numPart2 += numpart_list[n];

      if(numPart1 != numPart2)
	{
	  printf("particle numbers %d and %d are not equal!\n", numPart1, numPart2);
	  MPI_Abort(MPI_COMM_WORLD, 1);
	  exit(0);
	}

      nstill_to_check = numPart1;

      cpu1 = 0;
      cpu2 = INVARIANCETEST_SIZE1;

      PP1 = (struct combined_data *) mymalloc("PP1", NMAX_CHECK * sizeof(struct combined_data));
      PP2 = (struct combined_data *) mymalloc("PP2", NMAX_CHECK * sizeof(struct combined_data));

      while(nstill_to_check > 0)
	{
	  if((ncheck = NMAX_CHECK) > nstill_to_check)
	    ncheck = nstill_to_check;

	  nobtained1 = nobtained2 = 0;

	  while(nobtained1 < ncheck)
	    {
	      if(numpart_list[cpu1] < (ncheck - nobtained1))
		nget = numpart_list[cpu1];
	      else
		nget = ncheck - nobtained1;

	      if(cpu1 == 0)
		{
		  memcpy(&PP1[nobtained1], &PP[nskip], nget * sizeof(struct combined_data));
		  nskip += nget;
		}
	      else
		{
		  MPI_Send(&nget, 1, MPI_INT, cpu1, TAG_N, MPI_COMM_WORLD);
		  MPI_Recv(&PP1[nobtained1], nget * sizeof(struct combined_data), MPI_BYTE, cpu1, TAG_PDATA,
			   MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}

	      nobtained1 += nget;
	      numpart_list[cpu1] -= nget;
	      if(numpart_list[cpu1] == 0)
		cpu1++;
	    }

	  while(nobtained2 < ncheck)
	    {
	      if(numpart_list[cpu2] < (ncheck - nobtained2))
		nget = numpart_list[cpu2];
	      else
		nget = ncheck - nobtained2;

	      MPI_Send(&nget, 1, MPI_INT, cpu2, TAG_N, MPI_COMM_WORLD);
	      MPI_Recv(&PP2[nobtained2], nget * sizeof(struct combined_data), MPI_BYTE, cpu2, TAG_PDATA,
		       MPI_COMM_WORLD, MPI_STATUS_IGNORE);

	      nobtained2 += nget;
	      numpart_list[cpu2] -= nget;
	      if(numpart_list[cpu2] == 0)
		cpu2++;
	    }

	  count += compare_particles(ncheck, count);	/* now do check */

	  nstill_to_check -= ncheck;
	}

      myfree(PP2);
      myfree(PP1);

      for(i = 1; i < World_NTask; i++)
	{
	  nget = 0;
	  MPI_Send(&nget, 1, MPI_INT, i, TAG_N, MPI_COMM_WORLD);	/* signals stop for polling */
	}
    }
  else
    {
      do
	{
	  MPI_Recv(&nget, 1, MPI_INT, 0, TAG_N, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	  if(nget)
	    {
	      MPI_Send(&PP[nskip], nget * sizeof(struct combined_data), MPI_BYTE, 0, TAG_PDATA,
		       MPI_COMM_WORLD);
	      nskip += nget;
	    }
	}
      while(nget);
    }

  myfree(numpart_list);
  myfree(PP);


  MPI_Bcast(&count, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if(count > 0)
    {
      if(World_ThisTask == 0)
	printf("DIFFERENCES! %d \n", count);
      fflush(stdout);

      if(count > 200)
	endrun(0);
    }
}



int compare_particles(int ncheck, int count)
{
  int n, j, flag;
  double a1, a2, da, delta[3];

  for(n = 0; n < ncheck; n++)
    {
      /* compare Ti_current */
      if(PP1[n].P.Ti_current != PP2[n].P.Ti_current)
	{
	  if(count < MAXPRINT)
	    printf("STEP=%d  id1=%d id2=%d  ti_current=%d ti_current=%d\n",
		   All.NumCurrentTiStep, PP1[n].P.ID, PP2[n].P.ID, PP1[n].P.Ti_current, PP2[n].P.Ti_current);
	  count++;
	}


      /* compare accelerations */
      for(j = 0, a1 = a2 = da = 0; j < 3; j++)
	{
	  a1 += PP1[n].P.g.GravAccel[j] * PP1[n].P.g.GravAccel[j];
	  a2 += PP2[n].P.g.GravAccel[j] * PP2[n].P.g.GravAccel[j];
	  delta[j] = (PP1[n].P.g.GravAccel[j] - PP2[n].P.g.GravAccel[j]);
	  da += delta[j] * delta[j];
	}
      a1 = sqrt(a1);
      a2 = sqrt(a2);
      da = sqrt(da);

      if(da > ACC * a1)
	{
	  if(count < MAXPRINT)
	    printf("STEP=%d  id1=%d id2=%d  a1=%g a2=%g  da/a1=%g\n",
		   All.NumCurrentTiStep, PP1[n].P.ID, PP2[n].P.ID, a1, a2, da / a1);
	  count++;
	}



      /* compare coordinates */
      for(j = 0, a1 = a2 = da = 0, flag = 0; j < 3; j++)
	{
	  a1 += PP1[n].P.Pos[j] * PP1[n].P.Pos[j];
	  a2 += PP2[n].P.Pos[j] * PP2[n].P.Pos[j];

	  if(PP1[n].P.Pos[j] != PP2[n].P.Pos[j])
	    flag += 1;

	  delta[j] = (PP1[n].P.Pos[j] - PP2[n].P.Pos[j]);
	  da += delta[j] * delta[j];
	}
      a1 = sqrt(a1);
      a2 = sqrt(a2);
      da = sqrt(da);

      if(flag)
	{
	  if(count < MAXPRINT)
	    printf("STEP=%d  id1=%d id2=%d  type=%d pos1=%g pos2=%g  dpos/pos1=%g\n",
		   All.NumCurrentTiStep, PP1[n].P.ID, PP2[n].P.ID, PP1[n].P.Type, a1, a2, da / a1);
	  count++;
	}

      /* compare velocities */
      for(j = 0, a1 = a2 = da = 0; j < 3; j++)
	{
	  a1 += PP1[n].P.Vel[j] * PP1[n].P.Vel[j];
	  a2 += PP2[n].P.Vel[j] * PP2[n].P.Vel[j];

	  if(PP1[n].P.Vel[j] != PP2[n].P.Vel[j])
	    flag += 1;

	  delta[j] = (PP1[n].P.Vel[j] - PP2[n].P.Vel[j]);
	  da += delta[j] * delta[j];
	}
      a1 = sqrt(a1);
      a2 = sqrt(a2);
      da = sqrt(da);

      if(flag)
	{
	  if(count < MAXPRINT)
	    printf("STEP=%d  id1=%d id2=%d  type=%d vel1=%g vel2=%g  dvel/vel1=%g\n",
		   All.NumCurrentTiStep, PP1[n].P.ID, PP2[n].P.ID, PP1[n].P.Type, a1, a2, da / a1);
	  count++;
	}


      /* compare masses */

      a1 = PP1[n].P.Mass;
      a2 = PP2[n].P.Mass;
      da = fabs(a1 - a2);

      if(a1 != a2)
	{
	  if(count < MAXPRINT)
	    printf("STEP=%d  id1=%d id2=%d  mass1=%g mass2=%g  da/da1=%g\n",
		   All.NumCurrentTiStep, PP1[n].P.ID, PP2[n].P.ID, a1, a2, da / a1);
	  count++;
	}


#if defined(CS_MODEL) && defined(CS_FEEDBACK)
      /* compare hothsml */
      a1 = PP1[n].P.EnergySN;
      a2 = PP2[n].P.EnergySN;
      da = fabs(a1 - a2);
      if(da > ACC * fabs(a1))
	{
	  if(count < MAXPRINT)
	    printf("STEP=%d  id1=%d id2=%d  SN_energy1=%g SN_energy2=%g  da/da1=%g\n",
		   All.NumCurrentTiStep, PP1[n].P.ID, PP2[n].P.ID, a1, a2, da / a1);
	  count++;
	}
#endif

      if(PP1[n].P.Type == 0)	/* do separate checks for gas particles */
	{
	  /* compare densities */
	  a1 = PP1[n].SphP.d.Density;
	  a2 = PP2[n].SphP.d.Density;
	  da = fabs(a1 - a2);
	  if(da > ACC * fabs(a1))
	    {
	      if(count < MAXPRINT)
		printf("STEP=%d  id1=%d id2=%d  dens1=%g dens2=%g  da/da1=%g\n",
		       All.NumCurrentTiStep, PP1[n].P.ID, PP2[n].P.ID, a1, a2, da / a1);
	      count++;
	    }


	  /* compare acceleration */
	  for(j = 0, a1 = a2 = da = 0; j < 3; j++)
	    {
	      a1 += PP1[n].SphP.a.HydroAccel[j] * PP1[n].SphP.a.HydroAccel[j];
	      a2 += PP2[n].SphP.a.HydroAccel[j] * PP2[n].SphP.a.HydroAccel[j];

	      delta[j] = (PP1[n].SphP.a.HydroAccel[j] - PP2[n].SphP.a.HydroAccel[j]);
	      da += delta[j] * delta[j];
	    }
	  a1 = sqrt(a1);
	  a2 = sqrt(a2);
	  da = sqrt(da);

	  if(da > ACC * a1)
	    {
	      if(count < MAXPRINT)
		printf("STEP=%d  id1=%d id2=%d  hyaccel1=%g hydacc2=%g  da/da1=%g\n",
		       All.NumCurrentTiStep, PP1[n].P.ID, PP2[n].P.ID, a1, a2, da / a1);
	      count++;
	    }


	  /* compare entropies */
	  a1 = PP1[n].SphP.Entropy;
	  a2 = PP2[n].SphP.Entropy;
	  da = fabs(a1 - a2);
	  if(da > ACC * fabs(a1))
	    {
	      if(count < MAXPRINT)
		printf("STEP=%d  id1=%d id2=%d  entr1=%g entr2=%g  da/da1=%g\n",
		       All.NumCurrentTiStep, PP1[n].P.ID, PP2[n].P.ID, a1, a2, da / a1);
	      count++;
	    }


	  /* compare rate of change of entropies */
	  a1 = fabs(PP1[n].SphP.e.DtEntropy);
	  a2 = fabs(PP2[n].SphP.e.DtEntropy);
	  da = fabs(a1 - a2);

	  if(da > ACC * fabs(a1))
	    {
	      if(ThisTask == 0)
		printf("STEP=%d  id1=%d id2=%d  dtEntr1=%g dtEntr2=%g  da/da1=%g\n",
		       All.NumCurrentTiStep, PP1[n].P.ID, PP2[n].P.ID, a1, a2, da / a1);
	      count++;
	    }


	  /* compare smoothing lengths */
	  a1 = PP1[n].PPP.Hsml;
	  a2 = PP2[n].PPP.Hsml;
	  da = fabs(a1 - a2);

	  if(da > ACC * a1)
	    {
	      if(count < MAXPRINT)
		printf("STEP=%d  id1=%d id2=%d  hsml=%g hsml=%g  da/da1=%g\n",
		       All.NumCurrentTiStep, PP1[n].P.ID, PP2[n].P.ID, a1, a2, da / a1);
	      count++;
	    }




	  /* compare predicted velocities */
	  for(j = 0, a1 = a2 = da = 0; j < 3; j++)
	    {
	      a1 += PP1[n].SphP.VelPred[j] * PP1[n].SphP.VelPred[j];
	      a2 += PP2[n].SphP.VelPred[j] * PP2[n].SphP.VelPred[j];

	      if(PP1[n].SphP.VelPred[j] != PP2[n].SphP.VelPred[j])
		flag += 1;

	      delta[j] = (PP1[n].SphP.VelPred[j] - PP2[n].SphP.VelPred[j]);
	      da += delta[j] * delta[j];
	    }
	  a1 = sqrt(a1);
	  a2 = sqrt(a2);
	  da = sqrt(da);

	  if(flag)
	    {
	      if(count < MAXPRINT)
		printf("STEP=%d  id1=%d id2=%d  type=%d velpred1=%g velpred2=%g  dvelpred/velpred1=%g\n",
		       All.NumCurrentTiStep, PP1[n].P.ID, PP2[n].P.ID, PP1[n].P.Type, a1, a2, da / a1);
	      count++;
	    }

	  /* compare velocity divergence */
	  a1 = PP1[n].SphP.v.DivVel;
	  a2 = PP2[n].SphP.v.DivVel;
	  da = fabs(a1 - a2);

	  if(da > ACC * fabs(a1))
	    {
	      if(count < MAXPRINT)
		printf("STEP=%d  id1=%d id2=%d divv=%g divv=%g  da/da1=%g\n",
		       All.NumCurrentTiStep, PP1[n].P.ID, PP2[n].P.ID, a1, a2, da / a1);
	      count++;
	    }

#if defined(CS_MODEL) && defined(CS_FEEDBACK)
	  /* compare averaged entropies */
	  a1 = PP1[n].SphP.ea.EntropyAvg;
	  a2 = PP2[n].SphP.ea.EntropyAvg;
	  da = fabs(a1 - a2);
	  if(da > ACC * fabs(a1))
	    {
	      if(count < MAXPRINT)
		printf("STEP=%d  id1=%d id2=%d  entravg1=%g entravg2=%g  da/da1=%g\n",
		       All.NumCurrentTiStep, PP1[n].P.ID, PP2[n].P.ID, a1, a2, da / a1);
	      count++;
	    }

	  /* compare averaged densities */
	  a1 = PP1[n].SphP.da.DensityAvg;
	  a2 = PP2[n].SphP.da.DensityAvg;
	  da = fabs(a1 - a2);
	  if(da > ACC * fabs(a1))
	    {
	      if(count < MAXPRINT)
		printf("STEP=%d  id1=%d id2=%d  densavg1=%g densavg2=%g  da/da1=%g\n",
		       All.NumCurrentTiStep, PP1[n].P.ID, PP2[n].P.ID, a1, a2, da / a1);
	      count++;
	    }

	  /* compare hothsml */
	  a1 = PP1[n].SphP.HotHsml;
	  a2 = PP2[n].SphP.HotHsml;
	  da = fabs(a1 - a2);
	  if(da > ACC * fabs(a1))
	    {
	      if(count < MAXPRINT)
		printf("STEP=%d  id1=%d id2=%d  hohsml1=%g hothsml2=%g  da/da1=%g\n",
		       All.NumCurrentTiStep, PP1[n].P.ID, PP2[n].P.ID, a1, a2, da / a1);
	      count++;
	    }
#endif
	}

    }

  return count;
}


/*! This is a comparison kernel for a sort routine, which is used to group
 *  particles that are going to be exported to the same CPU.
 */
int cmp_partitions_compare_id(const void *a, const void *b)
{
  if(((struct combined_data *) a)->P.ID < (((struct combined_data *) b)->P.ID))
    return -1;
  if(((struct combined_data *) a)->P.ID > (((struct combined_data *) b)->P.ID))
    return +1;
  return 0;
}


#endif
