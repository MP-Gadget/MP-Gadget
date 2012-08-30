#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <gsl/gsl_math.h>


#include "allvars.h"
#include "proto.h"

#ifndef DEBUG
#define NDEBUG
#endif
#include <assert.h>

#ifdef RADTRANSFER
#ifndef CG

#define NSTEP 1
#define MAX_ITER 100
#define ACCURACY 1.e-4


/*structures for radtransfer*/
struct radtransferdata_in
{
  int NodeList[NODELISTLENGTH];
  MyDouble Pos[3];
  MyFloat Hsml, kappa, n_gamma;
  MyFloat ET[6];
  MyFloat Mass, Density;
}
 *RadTransferDataIn, *RadTransferDataGet;

struct radtransferdata_out
{
  MyFloat fac1, fac2;
}
 *RadTransferDataResult, *RadTransferDataOut;

static double c_light, dt, a3inv, timestep, hubble_a;
static double *fac1, *fac2;
static double sigma, alpha, nH;

void radtransfer(void)
{
  int i, j;

  radtransfer_mean();

  if(All.Time != All.TimeBegin)
    for(i = 0; i < NSTEP; i++)
      {
	/*  the actual time-step we need to do */
	timestep = (All.Radiation_Ti_endstep - All.Radiation_Ti_begstep) * All.Timebase_interval;

	if(All.ComovingIntegrationOn)
	  {
	    a3inv = 1 / (All.Time * All.Time * All.Time);
	    hubble_a = hubble_function(All.Time);
	    /* in comoving case, timestep is dloga at this point. Convert to dt */
	    timestep /= hubble_a;
	  }
	else
	  {
	    a3inv = hubble_a = 1.0;
	  }

	dt = timestep / NSTEP;

	c_light = C / All.UnitVelocity_in_cm_per_s;

	if(ThisTask == 0)
	  {
	    printf("%s %i\n", "the step is ", i);
	    printf("%s %g\n", "c is ", c_light);
	    printf("%s %g\n", "dt is ", dt);
	    fflush(stdout);
	  }

	for(j = 0; j < N_gas; j++)
	  if(P[j].Type == 0)
	    {
	      SphP[j].n_gamma_old = SphP[j].n_gamma;
	      //SphP[j].n_gamma += dt * SphP[j].Je;
	      //printf("%g %g %g\n", dt * SphP[j].Je, SphP[j].n_gamma, SphP[j].n_gamma_old);
	    }

	radtransfer_begin();

	radtransfer_update_chemistry();

	radtransfer_mean();

      }
}

/* this function loops over all particles and computes the mean intensity for the particles
that do not need to be exported to other processors. the particle that need to be exportes
are then exported and their thei mean intensity is calculated as well */
void radtransfer_begin(void)
{
  int i, j, k, ngrp, dummy, ndone, ndone_flag;
  int sendTask, recvTask, nexport, nimport, place, iter = 0;
  double residue, maxresidue;
  double n_gamma_old, delta_n_gamma, FFac1, FFac2, a3inv;

  /* allocate buffers to arrange communication */

  Ngblist = (int *) mymalloc("Ngblist", NumPart * sizeof(int));

  fac1 = (double *) mymalloc("fac1", N_gas * sizeof(double));
  fac2 = (double *) mymalloc("fac2", N_gas * sizeof(double));

  alpha = 2.59e-13 * All.UnitTime_in_s / All.UnitLength_in_cm / All.UnitLength_in_cm / All.UnitLength_in_cm;
  sigma = 6.3e-18 / All.UnitLength_in_cm / All.UnitLength_in_cm;

  All.BunchSize =
    (int) ((All.BufferSize * 1024 * 1024) / (sizeof(struct data_index) + sizeof(struct data_nodelist) +
					     sizeof(struct radtransferdata_in) +
					     sizeof(struct radtransferdata_out) +
					     sizemax(sizeof(struct radtransferdata_in),
						     sizeof(struct radtransferdata_out))));
  DataIndexTable =
    (struct data_index *) mymalloc("DataIndexTable", All.BunchSize * sizeof(struct data_index));
  DataNodeList =
    (struct data_nodelist *) mymalloc("DataNodeList", All.BunchSize * sizeof(struct data_nodelist));

  if(All.ComovingIntegrationOn)
    {
      a3inv = 1 / (All.Time * All.Time * All.Time);
    }
  else
    {
      a3inv = 1.0;
    }

  do				/* Jacobi iteration */
    {

      i = 0;

      do			/* communication loop */
	{

	  for(j = 0; j < NTask; j++)
	    {
	      Send_count[j] = 0;
	      Exportflag[j] = -1;
	    }

	  /* do local particles and prepare export list */
	  for(nexport = 0; i < N_gas; i++)
	    if(P[i].Type == 0)
	      if(radtransfer_evaluate(i, 0, &nexport, Send_count) < 0)
		break;

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

	  RadTransferDataGet =
	    (struct radtransferdata_in *) mymalloc(nimport * sizeof(struct radtransferdata_in));
	  RadTransferDataIn =
	    (struct radtransferdata_in *) mymalloc(nexport * sizeof(struct radtransferdata_in));

	  /* prepare particle data for export */
	  for(j = 0; j < nexport; j++)
	    {
	      place = DataIndexTable[j].Index;
	      for(k = 0; k < 3; k++)
		{
		  RadTransferDataIn[j].Pos[k] = P[place].Pos[k];
		  RadTransferDataIn[j].ET[k] = SphP[place].ET[k];
		  RadTransferDataIn[j].ET[k + 3] = SphP[place].ET[k + 3];
		}

	      nH = (SphP[j].d.Density * a3inv) / (PROTONMASS / All.UnitMass_in_g * All.HubbleParam);

	      RadTransferDataIn[j].Hsml = PPP[place].Hsml;
	      RadTransferDataIn[j].n_gamma = SphP[place].n_gamma;
	      RadTransferDataIn[j].kappa = SphP[place].nHI * nH * sigma;
	      RadTransferDataIn[j].Mass = P[place].Mass;
	      RadTransferDataIn[j].Density = SphP[place].d.Density;

	      memcpy(RadTransferDataIn[j].NodeList,
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
		      MPI_Sendrecv(&RadTransferDataIn[Send_offset[recvTask]],
				   Send_count[recvTask] * sizeof(struct radtransferdata_in), MPI_BYTE,
				   recvTask, TAG_HYDRO_A,
				   &RadTransferDataGet[Recv_offset[recvTask]],
				   Recv_count[recvTask] * sizeof(struct radtransferdata_in), MPI_BYTE,
				   recvTask, TAG_HYDRO_A, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		    }
		}
	    }

	  myfree(RadTransferDataIn);
	  RadTransferDataResult =
	    (struct radtransferdata_out *) mymalloc(nimport * sizeof(struct radtransferdata_out));
	  RadTransferDataOut =
	    (struct radtransferdata_out *) mymalloc(nexport * sizeof(struct radtransferdata_out));

	  /* now do the particles that were sent to us */
	  for(j = 0; j < nimport; j++)
	    radtransfer_evaluate(j, 1, &dummy, &dummy);

	  /* check whether this is the last iteration */
	  if(i < N_gas)
	    ndone_flag = 0;
	  else
	    ndone_flag = 1;

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
		      MPI_Sendrecv(&RadTransferDataResult[Recv_offset[recvTask]],
				   Recv_count[recvTask] * sizeof(struct radtransferdata_out),
				   MPI_BYTE, recvTask, TAG_HYDRO_B,
				   &RadTransferDataOut[Send_offset[recvTask]],
				   Send_count[recvTask] * sizeof(struct radtransferdata_out),
				   MPI_BYTE, recvTask, TAG_HYDRO_B, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		    }
		}
	    }

	  /* add the result to the local particles */
	  for(j = 0; j < nexport; j++)
	    {
	      place = DataIndexTable[j].Index;
	      fac1[place] += RadTransferDataOut[j].fac1;
	      fac2[place] += RadTransferDataOut[j].fac2;
	    }

	  myfree(RadTransferDataOut);
	  myfree(RadTransferDataResult);
	  myfree(RadTransferDataGet);


	}
      while(ndone < NTask);

      /* do final operations on particles */
      for(i = 0, residue = 0; i < N_gas; i++)
	if(P[i].Type == 0)
	  {
	    nH = (SphP[i].d.Density * a3inv) / (PROTONMASS / All.UnitMass_in_g * All.HubbleParam);
	    FFac2 = fac2[i] + SphP[i].Je;
	    FFac1 = fac1[i] + c_light * SphP[i].nHI * nH * sigma;

	    n_gamma_old = SphP[i].n_gamma;

	    SphP[i].n_gamma = (SphP[i].n_gamma_old + dt * FFac2) / (1.0 + dt * FFac1);

	    if(SphP[i].n_gamma < 0)
	      {
		printf("!!! negative intensity !!! \n ENDING \n");
		printf("%g %g %g %g %g %g\n", SphP[i].n_gamma, SphP[i].n_gamma_old, FFac2, FFac1, fac2[i],
		       fac1[i]);
		fflush(stdout);
		endrun(123);
	      }

	    delta_n_gamma = SphP[i].n_gamma - n_gamma_old;

	    if(SphP[i].n_gamma != 0)
	      if(fabs(delta_n_gamma) / SphP[i].n_gamma > residue)
		residue = fabs(delta_n_gamma) / SphP[i].n_gamma;
	  }

      MPI_Allreduce(&residue, &maxresidue, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

      if(ThisTask == 0)
	{
	  printf("%s %g\n", "the residue is ", maxresidue);
	  fflush(stdout);
	}

      iter++;

    }
  while(maxresidue > ACCURACY);

  myfree(DataNodeList);
  myfree(DataIndexTable);
  myfree(fac2);
  myfree(fac1);
  myfree(Ngblist);

}

/* this function evaluates the coefficients  D1 and P1 needed for the Jacobi method integration */
int radtransfer_evaluate(int target, int mode, int *nexport, int *nsend_local)
{
  int startnode, numngb, listindex = 0;
  int j, n, k;
  MyFloat *ET_aux, ET_j[6], ET_i[6], ET_ij[6];
  MyFloat kappa_i, kappa_j, kappa_ij, n_gamma_j, n_gamma_i;
  MyDouble *pos;
  MyFloat mass_j, mass_i, rho_j, rho_i, mass, rho;
  MyFloat Factor1 = 0, Factor2 = 0, a, a3inv;

  double dx, dy, dz;
  double h_j, hinv, hinv4, h_i;
  double dwk_i, dwk_j, dwk;
  double r, r2, r3, u;
  double fac;

  if(All.ComovingIntegrationOn)
    {
      a3inv = 1 / (All.Time * All.Time * All.Time);
      a = All.Time;
    }
  else
    {
      a = 1.0;
      a3inv = 1.0;
    }

  nH = (SphP[target].d.Density * a3inv) / (PROTONMASS / All.UnitMass_in_g * All.HubbleParam);

  if(mode == 0)
    {
      ET_aux = SphP[target].ET;
      pos = P[target].Pos;
      h_i = PPP[target].Hsml;
      kappa_i = SphP[target].nHI * nH * sigma;
      n_gamma_i = SphP[target].n_gamma;
      mass_i = P[target].Mass;
      rho_i = SphP[target].d.Density;
    }
  else
    {
      ET_aux = RadTransferDataGet[target].ET;
      pos = RadTransferDataGet[target].Pos;
      h_i = RadTransferDataGet[target].Hsml;
      kappa_i = RadTransferDataGet[target].kappa;
      n_gamma_i = RadTransferDataGet[target].n_gamma;
      mass_i = RadTransferDataGet[target].Mass;
      rho_i = RadTransferDataGet[target].Density;
    }

#ifdef RADTRANSFER_MODIFY_EDDINGTON_TENSOR
  /*modify Eddington tensor */
  ET_i[0] = 2 * ET_aux[0] - 0.5 * ET_aux[1] - 0.5 * ET_aux[2];
  ET_i[1] = 2 * ET_aux[1] - 0.5 * ET_aux[2] - 0.5 * ET_aux[0];
  ET_i[2] = 2 * ET_aux[2] - 0.5 * ET_aux[0] - 0.5 * ET_aux[1];

  for(k = 3; k < 6; k++)
    ET_i[k] = 2.5 * ET_aux[k];
#else
  for(k = 0; k < 6; k++)
    ET_i[k] = ET_aux[k];
#endif

  if(mode == 0)
    {
      startnode = All.MaxPart;	/* root node */
    }
  else
    {
      startnode = RadTransferDataGet[target].NodeList[0];
      startnode = Nodes[startnode].u.d.nextnode;	/* open node */
    }

  while(startnode >= 0)
    {
      while(startnode >= 0)
	{
	  numngb = ngb_treefind_pairs(pos, h_i, target, &startnode, mode, nexport, nsend_local);

	  if(numngb < 0)
	    return -1;

	  for(n = 0; n < numngb; n++)
	    {
	      j = Ngblist[n];
	      dx = P[j].Pos[0] - pos[0];
	      dy = P[j].Pos[1] - pos[1];
	      dz = P[j].Pos[2] - pos[2];
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
	      r = sqrt(r2);
	      r3 = r2 * r;
	      h_j = PPP[j].Hsml;

	      if(r > 0 && (r < h_i || r < h_j))
		{
		  mass_j = P[j].Mass;
		  rho_j = SphP[j].d.Density;
		  n_gamma_j = SphP[j].n_gamma;
		  kappa_j = SphP[j].nHI * nH * sigma;

#ifdef RADTRANSFER_MODIFY_EDDINGTON_TENSOR
		  ET_aux = SphP[j].ET;

		  /*modify Eddington tensor */
		  ET_j[0] = 2 * ET_aux[0] - 0.5 * ET_aux[1] - 0.5 * ET_aux[2];
		  ET_j[1] = 2 * ET_aux[1] - 0.5 * ET_aux[2] - 0.5 * ET_aux[0];
		  ET_j[2] = 2 * ET_aux[2] - 0.5 * ET_aux[0] - 0.5 * ET_aux[1];

		  for(k = 3; k < 6; k++)
		    ET_j[k] = 2.5 * ET_aux[k];
#else
		  for(k = 0; k < 6; k++)
		    ET_j[k] = SphP[j].ET[k];
#endif

		  /* compute the kernel */
		  if(r < h_i)
		    {
		      hinv = 1.0 / h_i;
		      hinv4 = hinv * hinv * hinv * hinv;
		      u = r * hinv;

		      if(u < 0.5)
			dwk_i = hinv4 * u * (KERNEL_COEFF_3 * u - KERNEL_COEFF_4);
		      else
			dwk_i = hinv4 * KERNEL_COEFF_6 * (1.0 - u) * (1.0 - u);
		    }
		  else
		    dwk_i = 0;

		  if(r < h_j)
		    {
		      hinv = 1.0 / h_j;
		      hinv4 = hinv * hinv * hinv * hinv;
		      u = r * hinv;

		      if(u < 0.5)
			dwk_j = hinv4 * u * (KERNEL_COEFF_3 * u - KERNEL_COEFF_4);
		      else
			dwk_j = hinv4 * KERNEL_COEFF_6 * (1.0 - u) * (1.0 - u);
		    }
		  else
		    dwk_j = 0;

		  /* symmetrize */
		  dwk = 0.5 * (dwk_i + dwk_j);
		  kappa_ij = 0.5 * (1 / kappa_i + 1 / kappa_j);
		  mass = 0.5 * (mass + mass_i);
		  rho = 0.5 * (rho + rho_i);
		  for(k = 0; k < 6; k++)
		    ET_ij[k] = 0.5 * (ET_i[k] + ET_j[k]);

		  double tensor = (ET_ij[0] * dx * dx + ET_ij[1] * dy * dy + ET_ij[2] * dz * dz
				   + 2.0 * ET_ij[3] * dx * dy + 2.0 * ET_ij[4] * dy * dz +
				   2.0 * ET_ij[5] * dz * dx);

		  if(tensor > 0)
		    {
		      fac = -2.0 * dt * (c_light / a / a) * (mass / rho) * kappa_ij * dwk / r3 * tensor;

		      Factor1 += fac;

		      Factor2 += fac * n_gamma_j;
		    }
		}
	    }
	}

      if(mode == 1)
	{
	  listindex++;
	  if(listindex < NODELISTLENGTH)
	    {
	      startnode = RadTransferDataGet[target].NodeList[listindex];
	      if(startnode >= 0)
		startnode = Nodes[startnode].u.d.nextnode;	/* open it */
	    }
	}
    }

  if(mode == 0)
    {
      fac1[target] = Factor1;
      fac2[target] = Factor2;
    }
  else
    {
      RadTransferDataResult[target].fac1 = Factor1;
      RadTransferDataResult[target].fac2 = Factor2;
    }

  return 0;
}

/* this function calculates the total radiation intensity */
void radtransfer_mean(void)
{
  int i;
  double n_gamma, n_gamma_all = 0;

  for(i = 0, n_gamma = 0; i < N_gas; i++)
    if(P[i].Type == 0)
      n_gamma += SphP[i].n_gamma;

  MPI_Allreduce(&n_gamma, &n_gamma_all, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  if(ThisTask == 0)
    {
      printf("n_gamma_all is: %g\n", n_gamma_all);
      fflush(stdout);
    }

}

/* this function sets up simple initial conditions for a single source in a uniform field of gas with constant density*/
void radtransfer_set_simple_inits(void)
{
  int i;

  for(i = 0; i < N_gas; i++)
    if(P[i].Type == 0)
      {
	/* in code units */
	SphP[i].nHII = 0.0;
	SphP[i].nHI = 1 - SphP[i].nHII;
	SphP[i].n_elec = SphP[i].nHII;
	SphP[i].n_gamma = 0.0;
      }
}

/* produces a simple output - particle positions x, y and z, overintensity and neutral fraction*/
void radtransfer_simple_output(void)
{
  char buf[100];
  int i;

  sprintf(buf, "%s%s%i%s%f%s", All.OutputDir, "radtransfer_", ThisTask, "_", All.Time, ".txt");
  FdRadtransfer = fopen(buf, "wa");
  for(i = 0; i < N_gas; i++)
    if(P[i].Type == 0)
      fprintf(FdRadtransfer, "%f %f %f %g %f \n", P[i].Pos[0], P[i].Pos[1], P[i].Pos[2], SphP[i].n_gamma,
	      SphP[i].nHII);

  fclose(FdRadtransfer);
}

/* solves the recombination equation and updates the HI and HII abundances */
void radtransfer_update_chemistry(void)
{
  int i, j, n;
  double inter_dt, a3inv;

  n = 1;
  inter_dt = dt / FLT(n);

  if(All.ComovingIntegrationOn)
    {
      a3inv = 1 / (All.Time * All.Time * All.Time);
    }
  else
    {
      a3inv = 1.0;
    }

  alpha = 2.59e-13 * All.UnitTime_in_s / All.UnitLength_in_cm / All.UnitLength_in_cm / All.UnitLength_in_cm;
  sigma = 6.3e-18 / All.UnitLength_in_cm / All.UnitLength_in_cm;

  /* begin substepping */
  for(j = 0; j < n; j++)
    for(i = 0; i < N_gas; i++)
      if(P[i].Type == 0)
	{
	  nH = (SphP[i].d.Density * a3inv) / (PROTONMASS / All.UnitMass_in_g * All.HubbleParam);

	  /* number of photons should be positive */
	  if(SphP[i].n_gamma < 0)
	    {
	      printf("NEGATIVE n_gamma: %g %d %d \n", SphP[i].n_gamma, i, ThisTask);
	      endrun(111);
	    }

	  /* semi-implicit scheme */
	  SphP[i].nHII = (SphP[i].nHII + inter_dt * c_light * sigma * nH * SphP[i].n_gamma) /
	    (1.0 + inter_dt * c_light * sigma * nH * SphP[i].n_gamma +
	     inter_dt * alpha * nH * SphP[i].n_elec);

	  /* fraction should be between 0 and 1 */
	  if(SphP[i].nHII < 0 || SphP[i].nHII > 1)
	    {
	      printf("WRONG nHI: %g %d %d \n", SphP[i].nHII, i, ThisTask);
	      endrun(222);
	    }

	  SphP[i].nHI = 1 - SphP[i].nHII;
	  SphP[i].n_elec = SphP[i].nHII;
	}

}

#endif
#endif
