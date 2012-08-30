#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <gsl/gsl_math.h>

#include "allvars.h"
#include "proto.h"

#ifdef RADTRANSFER
#ifdef CG

#define MAX_ITER 1000
#define ACCURACY 1.0e-2
#define EPSILON 1.0e-5

/*structures for radtransfer*/
struct radtransferdata_in
{
  int NodeList[NODELISTLENGTH];
  MyDouble Pos[3];
  MyFloat Hsml;
  MyFloat ET[6];
  double Kappa, Lambda;
  MyFloat Mass, Density;
}
 *RadTransferDataIn, *RadTransferDataGet;

struct radtransferdata_out
{
  double Out, Sum;
}
 *RadTransferDataResult, *RadTransferDataOut;

static double *XVec;
static double *QVec, *DVec, *Residue, *Zvec;
static double *Kappa, *Lambda, *Diag, *Diag2;
static double c_light, dt, a3inv, hubble_a;

void radtransfer(void)
{
  int i, j, iter;
  double alpha_cg, beta, delta_old, delta_new, sum, old_sum, min_diag, glob_min_diag, max_diag, glob_max_diag;
  double nH, nHe;
  double rel, res, maxrel, glob_maxrel;
  double DQ;
  
  c_light = C / All.UnitVelocity_in_cm_per_s;

  for(i = 0; i < N_BINS; i++)
    {
      XVec = (double *) mymalloc("XVec", N_gas * sizeof(double));
      QVec = (double *) mymalloc("QVec", N_gas * sizeof(double));
      DVec = (double *) mymalloc("DVec", N_gas * sizeof(double));
      Residue = (double *) mymalloc("Residue", N_gas * sizeof(double));
      Kappa = (double *) mymalloc("Kappa", N_gas * sizeof(double));
      Lambda = (double *) mymalloc("Lambda", N_gas * sizeof(double));
      Diag = (double *) mymalloc("Diag", N_gas * sizeof(double));
      Zvec = (double *) mymalloc("Zvec", N_gas * sizeof(double));
      Diag2 = (double *) mymalloc("Diag2", N_gas * sizeof(double));
      
      /*  the actual time-step we need to do */
      dt = (All.Radiation_Ti_endstep - All.Radiation_Ti_begstep) * All.Timebase_interval;

      if(All.ComovingIntegrationOn)
	{
	  a3inv = 1 / (All.Time * All.Time * All.Time);
	  hubble_a = hubble_function(All.Time);
	  /* in comoving case, timestep is dloga at this point. Convert to dt */
	  dt /= hubble_a;
	}
      else
	{
	  a3inv = hubble_a = 1.0;
	}
      
      /* initialization for the CG method */
      
      for(j = 0; j < N_gas; j++)
	if(P[j].Type == 0)
	  {
	    XVec[j] = SphP[j].n_gamma[i];
	    
	    nH = (HYDROGEN_MASSFRAC * SphP[j].d.Density) / (PROTONMASS / All.UnitMass_in_g * All.HubbleParam);
	    nHe = ((1.0 - HYDROGEN_MASSFRAC) * SphP[j].d.Density) / (2.0 * PROTONMASS / All.UnitMass_in_g * All.HubbleParam);
	    
	    Kappa[j] = a3inv * ((SphP[j].nHI + 1.0e-8) * nH * rt_sigma_HI[i] + 
				(SphP[j].nHeI + 1.0e-8) * nHe * rt_sigma_HeI[i] + 
				(SphP[j].nHeII + 1.0e-8) * nHe * rt_sigma_HeII[i]);
	    
	    if(All.ComovingIntegrationOn)
	      Kappa[j] *= All.Time;
	    
#ifdef RADTRANSFER_FLUXLIMITER
	    /* now calculate flux limiter */
	    
	    if(SphP[j].n_gamma[i] > 0)
	      {
		double R = sqrt(SphP[j].Grad_ngamma[0][i] * SphP[j].Grad_ngamma[0][i] +
				SphP[j].Grad_ngamma[1][i] * SphP[j].Grad_ngamma[1][i] +
				SphP[j].Grad_ngamma[2][i] * SphP[j].Grad_ngamma[2][i]) / (SphP[j].n_gamma[i] * Kappa[j]);
		
		if(All.ComovingIntegrationOn)
		  R /= All.Time;
		
		R *= 0.1;
		
		Lambda[j] = (1 + R) / (1 + R + R * R);
		if(Lambda[j] < 1e-100)
		  Lambda[j] = 0;
	      }
	    else
	      Lambda[j] = 1.0;
#endif
	    
	    /* add the source term */
	    SphP[j].n_gamma[i] += dt * SphP[j].Je[i] * P[j].Mass * nH;
	  }
      
      radtransfer_matrix_multiply(XVec, Residue, Diag);
      
      /* Let's take the diagonal matrix elements as Jacobi preconditioner */
      
      for(j = 0, min_diag = MAX_REAL_NUMBER, max_diag = -MAX_REAL_NUMBER; j < N_gas; j++)
	if(P[j].Type == 0)
	  {
	    Residue[j] = SphP[j].n_gamma[i] - Residue[j];
	    
	    /* note: in principle we would have to substract the w_ii term, but this is always zero */
	    if(Diag[j] < min_diag)
	      min_diag = Diag[j];
	    if(Diag[j] > max_diag)
	      max_diag = Diag[j];
	    
	    Zvec[j] = Residue[j] / Diag[j];
	    DVec[j] = Zvec[j];
	  }
      
      MPI_Allreduce(&min_diag, &glob_min_diag, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
      MPI_Allreduce(&max_diag, &glob_max_diag, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
      
      delta_new = radtransfer_vector_multiply(Zvec, Residue);
      delta_old = delta_new;
      
      old_sum = radtransfer_vector_sum(XVec);
      
      if(ThisTask == 0)
	printf("\nBegin CG iteration\nold |x|=%g, min-diagonal=%g, max-diagonal=%g\n", old_sum,
	       glob_min_diag, glob_max_diag);
      
      
      /* begin the CG method iteration */
      iter = 0;
      
      do
	{
	  radtransfer_matrix_multiply(DVec, QVec, Diag2);
	  
	  DQ = radtransfer_vector_multiply(DVec, QVec);
	  if(DQ == 0)
	    alpha_cg = 0;
	  else
	    alpha_cg = delta_new / DQ;
	  
	  
	  for(j = 0, maxrel = 0; j < N_gas; j++)
	    {
	      XVec[j] += alpha_cg * DVec[j];
	      Residue[j] -= alpha_cg * QVec[j];
	      
	      Zvec[j] = Residue[j] / Diag[j];
	      
	      rel = fabs(alpha_cg * DVec[j]) / (XVec[j] + 1.0e-10);
	      if(rel > maxrel)
		maxrel = rel;
	    }
	  
	  delta_old = delta_new;
	  delta_new = radtransfer_vector_multiply(Zvec, Residue);
	  
	  sum = radtransfer_vector_sum(XVec);
	  res = radtransfer_vector_sum(Residue);
	  
	  if(delta_old)
	    beta = delta_new / delta_old;
	  else
	    beta = 0;
	  
	  for(j = 0; j < N_gas; j++)
	    DVec[j] = Zvec[j] + beta * DVec[j];
	  
	  MPI_Allreduce(&maxrel, &glob_maxrel, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
	  
	  if(ThisTask == 0)
	    {
	      printf("radtransfer: iter=%3d  |res|/|x|=%12.6g  maxrel=%12.6g  |x|=%12.6g | res|=%12.6g\n",
		     iter, res / sum, glob_maxrel, sum, res);
	      fflush(stdout);
	    }
	  
	  iter++;
	}
      while((res > ACCURACY * sum && iter < MAX_ITER) || iter < 2);
      
      if(ThisTask == 0)
	printf("\n");
      
      /* update the intensity */
      for(j = 0; j < N_gas; j++)
	if(P[j].Type == 0)
	  {
	    if(XVec[j] < 0)
	      XVec[j] = 0;
#ifdef RT_RAD_PRESSURE
	    SphP[j].dn_gamma[i] = (XVec[j] - SphP[j].n_gamma[i]) / P[j].Mass;
#endif
	    
	    SphP[j].n_gamma[i] = XVec[j];
	  }
      
      myfree(Diag2);
      myfree(Zvec);
      myfree(Diag);
      myfree(Lambda);
      myfree(Kappa);
      myfree(Residue);
      myfree(DVec);
      myfree(QVec);
      myfree(XVec);
    }
  
}

/* internal product of two vectors */
double radtransfer_vector_multiply(double *a, double *b)
{
  int i;
  double sum, sumall;

  for(i = 0, sum = 0; i < N_gas; i++)
    if(P[i].Type == 0)
      sum += a[i] * b[i];

  MPI_Allreduce(&sum, &sumall, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  return sumall;
}


double radtransfer_vector_sum(double *a)
{
  int i;
  double sum, sumall;

  for(i = 0, sum = 0; i < N_gas; i++)
    if(P[i].Type == 0)
      sum += fabs(a[i]);

  MPI_Allreduce(&sum, &sumall, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  return sumall;
}


/* this function computes the vector b(out) given the vector x(in) such as Ax = b, where A is a matrix */
void radtransfer_matrix_multiply(double *in, double *out, double *sum)
{
  int i, j, k, ngrp, dummy, ndone, ndone_flag;
  int sendTask, recvTask, nexport, nimport, place;
  double a, dt;

  /* allocate buffers to arrange communication */

  Ngblist = (int *) mymalloc("Ngblist", NumPart * sizeof(int));

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

  dt = (All.Radiation_Ti_endstep - All.Radiation_Ti_begstep) * All.Timebase_interval;
  
  if(All.ComovingIntegrationOn)
    {
      a = All.Time;
      /* in comoving case, timestep is dloga at this point. Convert to dt */
      dt /= hubble_function(All.Time);
    }
  else
    {
      a = 1.0;
    }
  
  i = 0;

  do				/* communication loop */
    {

      for(j = 0; j < NTask; j++)
	{
	  Send_count[j] = 0;
	  Exportflag[j] = -1;
	}

      /* do local particles and prepare export list */
      for(nexport = 0; i < N_gas; i++)
	{
	  if(P[i].Type == 0)
	    if(radtransfer_evaluate(i, 0, in, out, sum, &nexport, Send_count) < 0)
	      break;
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

      RadTransferDataGet =
	(struct radtransferdata_in *) mymalloc("RadTransferDataGet",
					       nimport * sizeof(struct radtransferdata_in));
      RadTransferDataIn =
	(struct radtransferdata_in *) mymalloc("RadTransferDataIn",
					       nexport * sizeof(struct radtransferdata_in));

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
	  RadTransferDataIn[j].Hsml = PPP[place].Hsml;
	  RadTransferDataIn[j].Kappa = Kappa[place];
	  RadTransferDataIn[j].Lambda = Lambda[place];
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
			       recvTask, TAG_RT_A,
			       &RadTransferDataGet[Recv_offset[recvTask]],
			       Recv_count[recvTask] * sizeof(struct radtransferdata_in), MPI_BYTE,
			       recvTask, TAG_RT_A, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}
	    }
	}

      myfree(RadTransferDataIn);
      RadTransferDataResult =
	(struct radtransferdata_out *) mymalloc("RadTransferDataResult",
						nimport * sizeof(struct radtransferdata_out));
      RadTransferDataOut =
	(struct radtransferdata_out *) mymalloc("RadTransferDataOut",
						nexport * sizeof(struct radtransferdata_out));

      /* now do the particles that were sent to us */
      for(j = 0; j < nimport; j++)
	radtransfer_evaluate(j, 1, in, out, sum, &dummy, &dummy);

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
			       MPI_BYTE, recvTask, TAG_RT_B,
			       &RadTransferDataOut[Send_offset[recvTask]],
			       Send_count[recvTask] * sizeof(struct radtransferdata_out),
			       MPI_BYTE, recvTask, TAG_RT_B, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}
	    }
	}

      /* add the result to the local particles */
      for(j = 0; j < nexport; j++)
	{
	  place = DataIndexTable[j].Index;
	  out[place] += RadTransferDataOut[j].Out;
	  sum[place] += RadTransferDataOut[j].Sum;
	}

      myfree(RadTransferDataOut);
      myfree(RadTransferDataResult);
      myfree(RadTransferDataGet);

    }
  while(ndone < NTask);

  /* do final operations on results */
  for(i = 0; i < N_gas; i++)
    if(P[i].Type == 0)
      {
	/* divide c_light by a to get comoving speed of light (because kappa is comoving) */
	if((1 + dt * c_light / a * Kappa[i] + sum[i]) < 0)
	  {
	    printf("1 + sum + rate= %g   sum=%g rate=%g i =%d\n",
		   1 + dt * c_light / a * Kappa[i] + sum[i], sum[i], dt * c_light / a * Kappa[i], i);
	    endrun(11111111);
	  }

	sum[i] += 1.0 + dt * c_light / a * Kappa[i];

	out[i] += in[i] * sum[i];
      }

  myfree(DataNodeList);
  myfree(DataIndexTable);
  myfree(Ngblist);
}

/* this function evaluates parts of the matrix A */
int radtransfer_evaluate(int target, int mode, double *in, double *out, double *sum, int *nexport,
			 int *nsend_local)
{
  int startnode, numngb, listindex = 0;
  int j, n, k;
  MyFloat *ET_aux, ET_j[6], ET_i[6], ET_ij[6];
  MyFloat kappa_i, kappa_j, kappa_ij;

#ifdef RADTRANSFER_FLUXLIMITER
  MyFloat lambda_i, lambda_j;
#endif
  MyDouble *pos;
  MyFloat mass, mass_i, rho, rho_i;
  double sum_out = 0, sum_w = 0, fac = 0;

  double dx, dy, dz;
  double h_j, hinv, hinv4, h_i;
  double dwk_i, dwk_j, dwk;
  double r, r2, r3, u, a, dt;

#ifdef PERIODIC
  double boxsize, boxhalf;

  boxsize = All.BoxSize;
  boxhalf = 0.5 * All.BoxSize;
#endif

  dt = (All.Radiation_Ti_endstep - All.Radiation_Ti_begstep) * All.Timebase_interval;
  
  if(All.ComovingIntegrationOn)
    {
      a = All.Time;
      /* in comoving case, timestep is dloga at this point. Convert to dt */
      dt /= hubble_function(All.Time);
    }
  else
    {
      a = 1.0;
    }

  if(mode == 0)
    {
      ET_aux = SphP[target].ET;
      pos = P[target].Pos;
      h_i = PPP[target].Hsml;
      kappa_i = Kappa[target];
#ifdef RADTRANSFER_FLUXLIMITER
      lambda_i = Lambda[target];
#endif
      mass_i = P[target].Mass;
      rho_i = SphP[target].d.Density;
    }
  else
    {
      ET_aux = RadTransferDataGet[target].ET;
      pos = RadTransferDataGet[target].Pos;
      h_i = RadTransferDataGet[target].Hsml;
      kappa_i = RadTransferDataGet[target].Kappa;
#ifdef RADTRANSFER_FLUXLIMITER
      lambda_i = RadTransferDataGet[target].Lambda;
#endif
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
      startnode = All.MaxPart;
    }
  else
    {
      startnode = RadTransferDataGet[target].NodeList[0];
      startnode = Nodes[startnode].u.d.nextnode;
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
	      {
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
		r = sqrt(r2);
		r3 = r2 * r;
		h_j = PPP[j].Hsml;

		if(r > 0 && (r < h_i || r < h_j))
		  {
		    mass = P[j].Mass;
		    rho = SphP[j].d.Density;
		    kappa_j = Kappa[j];
#ifdef RADTRANSFER_FLUXLIMITER
		    lambda_j = Lambda[j];
#endif

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

		    for(k = 0; k < 6; k++)
		      ET_ij[k] = 0.5 * (ET_i[k] + ET_j[k]);

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

		    kappa_ij = 0.5 * (1 / kappa_i + 1 / kappa_j);
		    dwk = 0.5 * (dwk_i + dwk_j);
		    mass = 0.5 * (mass + mass_i);
		    rho = 0.5 * (rho + rho_i);

		    double tensor = (ET_ij[0] * dx * dx + ET_ij[1] * dy * dy + ET_ij[2] * dz * dz
				     + 2.0 * ET_ij[3] * dx * dy + 2.0 * ET_ij[4] * dy * dz +
				     2.0 * ET_ij[5] * dz * dx);

		    if(tensor > 0)
		      {
			fac = -2.0 * dt * c_light / a * (mass / rho) * kappa_ij * dwk / r3 * tensor;

#ifdef RADTRANSFER_FLUXLIMITER
			fac *= 0.5 * (lambda_i + lambda_j);
#endif

			sum_out -= fac * in[j];

			sum_w += fac;
		      }
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
		startnode = Nodes[startnode].u.d.nextnode;
	    }
	}
    }

  if(mode == 0)
    {
      out[target] = sum_out;
      sum[target] = sum_w;
    }
  else
    {
      RadTransferDataResult[target].Out = sum_out;
      RadTransferDataResult[target].Sum = sum_w;
    }

  return 0;
}


/* this function sets up simple initial conditions for a single source in a uniform field of gas with constant density*/
void radtransfer_set_simple_inits(void)
{
  int i, j;
  
  for(i = 0; i < NumPart; i++)
    if(P[i].Type == 0)
      {
	/* in code units */
	SphP[i].nHII = 1.0e-8;
	SphP[i].nHI = 1.0 - SphP[i].nHII;
	SphP[i].n_elec = SphP[i].nHII;

	for(j = 0; j < N_BINS; j++)
	  {
	    SphP[i].n_gamma[j] = 0.0;
#ifdef RT_RAD_PRESSURE
	    SphP[i].dn_gamma[j] = 0.0;
#endif
	  }

	SphP[i].nHeIII = 0.0;
	SphP[i].nHeII = 1e-8;
	SphP[i].nHeI = 1.0 - SphP[i].nHeII - SphP[i].nHeIII;
	SphP[i].n_elec +=
	  (SphP[i].nHeII + 2.0 * SphP[i].nHeIII) * (1.0 - HYDROGEN_MASSFRAC) / 4.0 / HYDROGEN_MASSFRAC;
      }
}



void radtransfer_update_chemistry(void)
{
  int i, j;
  double nH, temp, molecular_weight;
  double nHII;
  double dt, dtime, a3inv;
  double A, B, CC;
  double x, y;
  double n_gamma;
  double alpha_HII, gamma_HI, alpha_HeII, alpha_HeIII, gamma_HeI, gamma_HeII; 
  double nHe, nHeII, nHeIII;
  double D, E, F, G, J;
  double total_nHI, total_V, total_nHI_all, total_V_all, total_nHeI, total_nHeI_all;
      
  total_nHI = total_V = total_nHeI = 0;

  dt = (All.Radiation_Ti_endstep - All.Radiation_Ti_begstep) * All.Timebase_interval;
  
  if(All.ComovingIntegrationOn)
    {
      dtime = dt / hubble_function(All.Time);
      a3inv = 1.0 / All.Time / All.Time / All.Time;
    }
  else
    {
      dtime = dt;
      a3inv = 1.0;
    }

  c_light = C / All.UnitVelocity_in_cm_per_s;

  for(i = 0; i < N_gas; i++)
    if(P[i].Type == 0)
      {
	nH = (HYDROGEN_MASSFRAC * SphP[i].d.Density * a3inv) / (PROTONMASS / All.UnitMass_in_g * All.HubbleParam);
	
	nHe = ((1.0 - HYDROGEN_MASSFRAC) * SphP[i].d.Density * a3inv) / (4.0 * PROTONMASS / All.UnitMass_in_g * All.HubbleParam);   
	
	molecular_weight = 4 / (1 + 3 * HYDROGEN_MASSFRAC + 4 * HYDROGEN_MASSFRAC * SphP[i].n_elec);
	
	temp = (SphP[i].Entropy + SphP[i].e.DtEntropy * dt) * pow(SphP[i].d.Density * a3inv, GAMMA_MINUS1) *
	  (molecular_weight * PROTONMASS / All.UnitMass_in_g * All.HubbleParam) /
	  (BOLTZMANN / All.UnitEnergy_in_cgs * All.HubbleParam);
	
	/* collisional ionization rate */
	gamma_HI = 5.85e-11 * pow(temp, 0.5) * exp(-157809.1 / temp) / (1.0 + pow(temp / 1e5, 0.5));
	gamma_HI *= All.UnitTime_in_s / All.UnitLength_in_cm / All.UnitLength_in_cm / All.UnitLength_in_cm;
	gamma_HI *= All.HubbleParam * All.HubbleParam;
	
	/* alpha_B recombination coefficient */
	alpha_HII = 2.59e-13 * pow(temp / 1e4, -0.7);
	alpha_HII *= All.UnitTime_in_s / All.UnitLength_in_cm / All.UnitLength_in_cm / All.UnitLength_in_cm;
	alpha_HII *= All.HubbleParam * All.HubbleParam;
	
	/* collisional ionization rate */
	gamma_HeI = 2.38e-11 * pow(temp, 0.5) * exp(-285335.4 / temp) / (1.0 + pow(temp / 1e5, 0.5));
	gamma_HeI *= All.UnitTime_in_s / All.UnitLength_in_cm / All.UnitLength_in_cm / All.UnitLength_in_cm;
	gamma_HeI *= All.HubbleParam * All.HubbleParam;
	
	gamma_HeII = 5.68e-12 * pow(temp, 0.5) * exp(-631515 / temp) / (1.0 + pow(temp / 1e5, 0.5));
	gamma_HeII *= All.UnitTime_in_s / All.UnitLength_in_cm / All.UnitLength_in_cm / All.UnitLength_in_cm;
	gamma_HeII *= All.HubbleParam * All.HubbleParam;
	
	/* alpha_B recombination coefficient */
	alpha_HeII = 1.5e-10 * pow(temp, -0.6353);
	alpha_HeII *= All.UnitTime_in_s / All.UnitLength_in_cm / All.UnitLength_in_cm / All.UnitLength_in_cm;
	alpha_HeII *= All.HubbleParam * All.HubbleParam;
	
	alpha_HeIII = 3.36e-10 * pow(temp, -0.5) * pow(temp / 1e3, -0.2) / (1.0 + pow(temp / 1e6, 0.7));
	alpha_HeIII *= All.UnitTime_in_s / All.UnitLength_in_cm / All.UnitLength_in_cm / All.UnitLength_in_cm;
	alpha_HeIII *= All.HubbleParam * All.HubbleParam;
	
	for(j = 0; j < N_BINS; j++)
	  {
	    x = SphP[i].nHI * nH * rt_sigma_HI[j] / 
	      (SphP[i].nHI * nH * rt_sigma_HI[j] + SphP[i].nHeI * nHe * rt_sigma_HeI[j] + SphP[i].nHeII * nHe * rt_sigma_HeII[j]);
	    
	    y = SphP[i].nHeI * nHe * rt_sigma_HeI[j] / 
	      (SphP[i].nHI * nH * rt_sigma_HI[j] + SphP[i].nHeI * nHe * rt_sigma_HeI[j] + SphP[i].nHeII * nHe * rt_sigma_HeII[j]);
	    
	    n_gamma = SphP[i].n_gamma[j] / P[i].Mass * a3inv;

	    /* number of photons should be positive */
	    if(n_gamma < 0)
	      {
		printf("NEGATIVE n_gamma: %g %d %d \n", n_gamma, i, ThisTask);
		endrun(111);
	      }

	    A = dtime * gamma_HI * nH;
	    B = dtime * c_light * rt_sigma_HI[j] * x;
	    CC = dtime * alpha_HII* nH;
	    
	    /* semi-implicit scheme for ionization */
	    nHII =  (SphP[i].nHII + B * n_gamma + A * SphP[i].n_elec) /
	      (1.0 + B * n_gamma + CC * SphP[i].n_elec + A * SphP[i].n_elec);

	    if(nHII < 0 || nHII > 1)
	      {
		printf("ERROR nHII %g\n", nHII);
		endrun(333);
	      }
	    
	    SphP[i].n_elec = nHII;
	    SphP[i].n_elec += SphP[i].nHeII * (1.0 - HYDROGEN_MASSFRAC) / (4.0 * HYDROGEN_MASSFRAC);
	    SphP[i].n_elec +=2.0 * SphP[i].nHeIII * (1.0 - HYDROGEN_MASSFRAC) / (4.0 * HYDROGEN_MASSFRAC);
	    
	    SphP[i].nHII = nHII;
	    
	    SphP[i].nHI = 1.0 - SphP[i].nHII;
	    
	    D = dtime * gamma_HeII * nH;
	    E = dtime * alpha_HeIII * nH;
	    F = dtime * gamma_HeI * nH;
	    G = dtime * c_light * rt_sigma_HeI[j] * y;
	    J = dtime * alpha_HeII * nH;
	    
	    nHeII =
	      SphP[i].nHeII + F * SphP[i].n_elec + G * n_gamma -
	      ((F * SphP[i].n_elec + G * n_gamma - E * SphP[i].n_elec) / (1.0 +
									     E * SphP[i].n_elec) *
	       SphP[i].nHeIII);
	    
	    nHeII /= 1.0 + F * SphP[i].n_elec + G * n_gamma + D * SphP[i].n_elec + J * SphP[i].n_elec +
	      ((F * SphP[i].n_elec + G * n_gamma - E * SphP[i].n_elec) / (1.0 +
									     E * SphP[i].n_elec) * D *
	       SphP[i].n_elec);
	    
	    if(nHeII < 0 || nHeII > 1)
	      {
		printf("ERROR neHII %g\n", nHeII);
		endrun(333);
	      }
	    
	    nHeIII = (SphP[i].nHeIII + D * nHeII * SphP[i].n_elec) / (1.0 + E * SphP[i].n_elec);
	    
	    if(nHeIII < 0 || nHeIII > 1)
	      {
		printf("ERROR nHeIII %g\n", nHeIII);
		endrun(222);
	      }
	    
	    SphP[i].n_elec = SphP[i].nHII;
	    SphP[i].n_elec += nHeII * (1.0 - HYDROGEN_MASSFRAC) / (4.0 * HYDROGEN_MASSFRAC);
	    SphP[i].n_elec += 2.0 * nHeIII * (1.0 - HYDROGEN_MASSFRAC) / (4.0 * HYDROGEN_MASSFRAC);
	    
	    SphP[i].nHeII = nHeII;
	    SphP[i].nHeIII = nHeIII;
	    
	    SphP[i].nHeI = 1.0 - SphP[i].nHeII - SphP[i].nHeIII;
	    
	    if(SphP[i].nHeI < 0 || SphP[i].nHeI > 1)
	      {
		printf("ERROR nHeI %g\n", SphP[i].nHeI);
		endrun(444);
	      }
	  }	    

	total_nHI += SphP[i].nHI * P[i].Mass / (SphP[i].d.Density * a3inv);
	total_V += P[i].Mass / (SphP[i].d.Density * a3inv);
	total_nHeI += SphP[i].nHeI * P[i].Mass / (SphP[i].d.Density * a3inv);
	
      }
  
  MPI_Allreduce(&total_nHI, &total_nHI_all, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&total_V, &total_V_all, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&total_nHeI, &total_nHeI_all, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  
  /* output the input of photon density in physical units */
  if(ThisTask == 0)
    {
      fprintf(FdRad, "%g %g ", All.Time, total_nHI_all / total_V_all);
      fprintf(FdRad, "%g\n", total_nHeI_all / total_V_all);
      fflush(FdRad);
    }
}


void rt_get_sigma(void)
{
  int i;
  double E, dE;

  dE = (end_E - start_E) / N_BINS;

  for(i = 0; i < N_BINS; i++)
    {
      E = start_E + (i + 0.5) * dE;

      rt_sigma_HeI[i] = rt_sigma_HeII[i] = 0.0;

      rt_sigma_HI[i] = 6.3e-18 * pow(13.6/E,3) / All.UnitLength_in_cm / All.UnitLength_in_cm * All.HubbleParam * All.HubbleParam;
      if(E > 24.6)
        rt_sigma_HeI[i] = 7.83e-18 * pow(24.6/E,3) / All.UnitLength_in_cm / All.UnitLength_in_cm * All.HubbleParam * All.HubbleParam;
      if(E > 54.4)
        rt_sigma_HeI[i] = 1.58e-18 * pow(54.4/E,3) / All.UnitLength_in_cm / All.UnitLength_in_cm * All.HubbleParam * All.HubbleParam;
    }                                                                                                                                     
}

void rt_get_lum_stars(void)
{
  int j;
  double temp;
  double kT, BB;
  double nu, d_nu;
  double R_solar;
  double eV_to_erg = 1.60184e-12;
  double start_nu, end_nu;
  
  R_solar = 6.955e10;

  start_nu = start_E * eV_to_erg / PLANCK;
  end_nu = end_E * eV_to_erg / PLANCK;

  d_nu = (end_nu - start_nu) / (float)N_BINS;
  
  temp = 50700.0;
  
  kT = temp * BOLTZMANN;

  for(j = 0; j < N_BINS; j++)
    {
      nu = start_nu + (j + 0.5) * d_nu;
      
      BB =  2.0 * nu * nu * nu * PLANCK / 
	(exp(nu * PLANCK / kT) - 1.0) / C / C;
      
      lum[j] = BB / nu / PLANCK * d_nu;
      lum[j] *= 4.0 * M_PI * 100.0 * R_solar * R_solar;
      lum[j] *= All.UnitTime_in_s / All.HubbleParam;
    }
  
}

void rt_get_lum_gas(int target, double *je)
{
  int j;
  double temp, molecular_weight;
  double kT, BB;
  double nu, d_nu, R;
  double R_solar;
  double eV_to_erg = 1.60184e-12;
  double start_nu, end_nu;
  double u_cooling, u_BB;
  double dt, dtime;

  R_solar = 6.955e10;

  start_nu = start_E * eV_to_erg / PLANCK;
  end_nu = end_E * eV_to_erg / PLANCK;

  d_nu = (end_nu - start_nu) / (float)N_BINS;
  
  dt = (All.Radiation_Ti_endstep - All.Radiation_Ti_begstep) * All.Timebase_interval;

  if(All.ComovingIntegrationOn)
    {
      dtime = dt / hubble_function(All.Time);
      a3inv = 1.0 / All.Time / All.Time / All.Time;
    }
  else
    {
      dtime = dt;
      a3inv = 1.0;
    }

  R = pow(P[target].Mass / SphP[target].d.Density * a3inv, 1./3.);

  molecular_weight = 4 / (1 + 3 * HYDROGEN_MASSFRAC + 4 * HYDROGEN_MASSFRAC * SphP[target].n_elec);
  
  temp = (SphP[target].Entropy + SphP[target].e.DtEntropy * dt) * pow(SphP[target].d.Density * a3inv, GAMMA_MINUS1) *
    (molecular_weight * PROTONMASS / All.UnitMass_in_g * All.HubbleParam) /
    (BOLTZMANN / All.UnitEnergy_in_cgs * All.HubbleParam);
	
  kT = temp * BOLTZMANN;
  
  u_cooling = radtransfer_cooling_photoheating(target, dtime) / dtime;
  u_BB = 0.0;
  
  for(j = 0; j < N_BINS; j++)
    {
      nu = start_nu + (j + 0.5) * d_nu;
      
      BB =  2.0 * nu * nu * nu * PLANCK / 
	(exp(nu * PLANCK / kT) - 1.0) / C / C;
      
      je[j] = 4.0 * M_PI * BB / nu / PLANCK * d_nu * R * R;
      je[j] *= All.UnitTime_in_s / All.HubbleParam;
      
      u_BB += je[j] * nu;
    }
  
  u_BB /= P[target].Mass;
  
  for(j = 0; j < N_BINS; j++)
    je[j] *= u_cooling / u_BB;
}

#endif
#endif
