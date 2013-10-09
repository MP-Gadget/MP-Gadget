#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <gsl/gsl_math.h>

#include "allvars.h"
#include "proto.h"

/*! \file conduction.c
*  \brief Computes conduction bases on an implicit diffusion solver
*
*/

#ifdef CONDUCTION

#define MAX_COND_ITER 25
#define COND_ITER_ACCURACY 1.0e-4

static struct conductiondata_in
{
  MyDouble Pos[3];
  MyFloat Hsml;
  MyFloat Density;
  MyFloat Kappa;
  int NodeList[NODELISTLENGTH];
}
 *ConductionDataIn, *ConductionDataGet;


static struct conductiondata_out
{
  MyFloat Out;
  MyFloat Sum;
}
 *ConductionDataResult, *ConductionDataOut;


static double *Energy, *EnergyOld;
static double *Residual, *DVec, *QVec;
static double *Kappa;


/* we will use the conjugate gradient method to compute a solution
   of the implicitly formulate diffusion equation */
/* Note: the conduction equation we solve is really formulated with u instead of T, i.e.
   the factor (gamma-1)*mu*mp/k_B that converts from T to u is implicitely absorbed in a
   redefinition of kappa */

void conduction(void)
{
  int i, iter;
  double delta0, delta1, alpha, beta, a3inv, dt, rel_change, loc_max_rel_change, glob_max_rel_change;
  double sumnew, sumold, sumtransfer, sumnew_tot, sumold_tot, sumtransfer_tot;

#ifdef CONDUCTION_SATURATION
  double electron_free_path, temp_scale_length;
#endif

  if(ThisTask == 0)
    {
      printf("Start thermal conduction...\n");
      fflush(stdout);
    }

  Energy = (double *) mymalloc("Energy", N_gas * sizeof(double));
  EnergyOld = (double *) mymalloc("EnergyOld", N_gas * sizeof(double));
  Residual = (double *) mymalloc("Residual", N_gas * sizeof(double));
  DVec = (double *) mymalloc("DVec", N_gas * sizeof(double));
  QVec = (double *) mymalloc("QVec", N_gas * sizeof(double));

  Kappa = (double *) mymalloc("Kappa", N_gas * sizeof(double));


  if(All.ComovingIntegrationOn)
    a3inv = 1 / (All.Time * All.Time * All.Time);
  else
    a3inv = 1.0;


  dt = (All.Conduction_Ti_endstep - All.Conduction_Ti_begstep) * All.Timebase_interval;

  if(All.ComovingIntegrationOn)
    dt *= All.Time / hubble_function(All.Time);

  if(ThisTask == 0)
    {
      printf("dt=%g\n", dt);
    }

  /* First, let's compute the thermal energies per unit mass and
     conductivities for all particles */

  for(i = 0; i < N_gas; i++)
    {
      if(P[i].Type == 0)
	{
	  /* this gives the thermal energy per unit mass for particle i */
	  Energy[i] = EnergyOld[i] = SphP[i].Entropy *
	    pow(SphP[i].d.Density * a3inv, GAMMA_MINUS1) / GAMMA_MINUS1;

#ifdef CONDUCTION_CONSTANT
	  Kappa[i] = All.ConductionCoeff;
#else
	  Kappa[i] = All.ConductionCoeff * pow(EnergyOld[i], 2.5);
#ifdef CONDUCTION_SATURATION
	  electron_free_path =
	    All.ElectronFreePathFactor * Energy[i] * Energy[i] / (SphP[i].d.Density * a3inv);
	  temp_scale_length =
	    All.Time * fabs(SphP[i].Entropy) / sqrt(SphP[i].GradEntr[0] * SphP[i].GradEntr[0] +
						    SphP[i].GradEntr[1] * SphP[i].GradEntr[1] +
						    SphP[i].GradEntr[2] * SphP[i].GradEntr[2]);
	  Kappa[i] /= (1 + 4.2 * electron_free_path / temp_scale_length);
#endif
#endif

#ifdef SFR
	  if(SphP[i].d.Density * a3inv >= All.PhysDensThresh)
	    Kappa[i] = 0;
#endif

	  /* we'll factor the timestep into the conductivities, for simplicity */
	  Kappa[i] *= dt;
	}
    }



  /* Let's start the Conjugate Gradient Algorithm */

  /* Initialization */

  conduction_matrix_multiply(EnergyOld, Residual);

  for(i = 0; i < N_gas; i++)
    {
      if(P[i].Type == 0)
	{
	  Residual[i] = EnergyOld[i] - Residual[i];
	  DVec[i] = Residual[i];
	}
    }

  delta1 = conduction_vector_multiply(Residual, Residual);
  delta0 = delta1;

  iter = 0;			/* iteration counter */
  glob_max_rel_change = 1 + COND_ITER_ACCURACY;	/* to make sure that we enter the iteration */

  while(iter < MAX_COND_ITER && glob_max_rel_change > COND_ITER_ACCURACY && delta1 > 0)
    {
      conduction_matrix_multiply(DVec, QVec);

      alpha = delta1 / conduction_vector_multiply(DVec, QVec);

      for(i = 0, loc_max_rel_change = 0; i < N_gas; i++)
	{
	  Energy[i] += alpha * DVec[i];
	  Residual[i] -= alpha * QVec[i];

	  rel_change = alpha * DVec[i] / Energy[i];
	  if(loc_max_rel_change < rel_change)
	    loc_max_rel_change = rel_change;
	}

      delta0 = delta1;
      delta1 = conduction_vector_multiply(Residual, Residual);

      beta = delta1 / delta0;

      for(i = 0; i < N_gas; i++)
	DVec[i] = Residual[i] + beta * DVec[i];

      iter++;


      MPI_Allreduce(&loc_max_rel_change, &glob_max_rel_change, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

      if(ThisTask == 0)
	{
	  printf("conduction iter=%d  delta1=%g delta1/delta0=%g  max-rel-change=%g\n",
		 iter, delta1, delta1 / delta0, glob_max_rel_change);
	  fflush(stdout);
	}
    }


  /* Now we have the solution vector in Energy[] */
  /* assign it to the entropies, and update the pressure */

  for(i = 0, sumnew = sumold = sumtransfer = 0; i < N_gas; i++)
    {
      if(P[i].Type == 0)
	{
	  sumnew += P[i].Mass * Energy[i];
	  sumold += P[i].Mass * EnergyOld[i];
	  sumtransfer += P[i].Mass * fabs(Energy[i] - EnergyOld[i]);

	  SphP[i].Entropy = Energy[i] / pow(SphP[i].d.Density * a3inv, GAMMA_MINUS1) * GAMMA_MINUS1;
	}
    }

  MPI_Allreduce(&sumnew, &sumnew_tot, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&sumold, &sumold_tot, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&sumtransfer, &sumtransfer_tot, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  if(ThisTask == 0)
    {
      printf("\nconduction finished. energy_before=%g energy_after=%g  rel-change=%g rel-transfer=%g\n\n",
	     sumold_tot, sumnew_tot, (sumnew_tot - sumold_tot) / sumold_tot, sumtransfer_tot / sumold_tot);
      fflush(stdout);
    }

  myfree(Kappa);
  myfree(QVec);
  myfree(DVec);
  myfree(Residual);
  myfree(EnergyOld);
  myfree(Energy);
}


double conduction_vector_multiply(double *a, double *b)
{
  int i;
  double sum, sumall;

  for(i = 0, sum = 0; i < N_gas; i++)
    if(P[i].Type == 0)
      sum += a[i] * b[i];

  MPI_Allreduce(&sum, &sumall, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  return sumall;
}


void conduction_matrix_multiply(double *in, double *out)
{
  int i, j, k, ngrp, ndone, ndone_flag, dummy;
  int sendTask, recvTask, nexport, nimport, place;
  double *sum;


  /* allocate buffers to arrange communication */

  Ngblist = (int *) mymalloc("Ngblist", NumPart * sizeof(int));

  All.BunchSize =
    (int) ((All.BufferSize * 1024 * 1024) / (sizeof(struct data_index) + sizeof(struct data_nodelist) +
					     sizeof(struct conductiondata_in) +
					     sizeof(struct conductiondata_out) +
					     sizemax(sizeof(struct conductiondata_in),
						     sizeof(struct conductiondata_out))));
  DataIndexTable =
    (struct data_index *) mymalloc("DataIndexTable", All.BunchSize * sizeof(struct data_index));
  DataNodeList =
    (struct data_nodelist *) mymalloc("DataNodeList", All.BunchSize * sizeof(struct data_nodelist));


  sum = (double *) mymalloc("sum", N_gas * sizeof(double));


  i = 0;			/* need to go over all gas particles */

  do
    {
      for(j = 0; j < NTask; j++)
	{
	  Send_count[j] = 0;
	  Exportflag[j] = -1;
	}

      /* do local particles and prepare export list */
      for(nexport = 0; i < N_gas; i++)
	if(P[i].Type == 0)
	  {
	    if(conduction_evaluate(i, 0, in, out, sum, &nexport, Send_count) < 0)
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

      ConductionDataGet =
	(struct conductiondata_in *) mymalloc("ConductionDataGet",
					      nimport * sizeof(struct conductiondata_in));
      ConductionDataIn =
	(struct conductiondata_in *) mymalloc("ConductionDataIn", nexport * sizeof(struct conductiondata_in));

      /* prepare particle data for export */

      for(j = 0; j < nexport; j++)
	{
	  place = DataIndexTable[j].Index;

	  for(k = 0; k < 3; k++)
	    ConductionDataIn[j].Pos[k] = P[place].Pos[k];

	  ConductionDataIn[j].Hsml = P[place].Hsml;
	  ConductionDataIn[j].Density = SphP[place].d.Density;
	  ConductionDataIn[j].Kappa = Kappa[place];

	  memcpy(ConductionDataIn[j].NodeList,
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
		  MPI_Sendrecv(&ConductionDataIn[Send_offset[recvTask]],
			       Send_count[recvTask] * sizeof(struct conductiondata_in), MPI_BYTE,
			       recvTask, TAG_HYDRO_A,
			       &ConductionDataGet[Recv_offset[recvTask]],
			       Recv_count[recvTask] * sizeof(struct conductiondata_in), MPI_BYTE,
			       recvTask, TAG_HYDRO_A, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}
	    }
	}

      myfree(ConductionDataIn);
      ConductionDataResult =
	(struct conductiondata_out *) mymalloc("ConductionDataResult",
					       nimport * sizeof(struct conductiondata_out));
      ConductionDataOut =
	(struct conductiondata_out *) mymalloc("ConductionDataOut",
					       nexport * sizeof(struct conductiondata_out));


      /* now do the particles that were sent to us */
      for(j = 0; j < nimport; j++)
	conduction_evaluate(j, 1, in, out, sum, &dummy, &dummy);

      if(i >= N_gas)
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
		  MPI_Sendrecv(&ConductionDataResult[Recv_offset[recvTask]],
			       Recv_count[recvTask] * sizeof(struct conductiondata_out),
			       MPI_BYTE, recvTask, TAG_HYDRO_B,
			       &ConductionDataOut[Send_offset[recvTask]],
			       Send_count[recvTask] * sizeof(struct conductiondata_out),
			       MPI_BYTE, recvTask, TAG_HYDRO_B, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}
	    }
	}

      /* add the result to the local particles */
      for(j = 0; j < nexport; j++)
	{
	  place = DataIndexTable[j].Index;

	  out[place] += ConductionDataOut[j].Out;
	  sum[place] += ConductionDataOut[j].Sum;
	}

      myfree(ConductionDataOut);
      myfree(ConductionDataResult);
      myfree(ConductionDataGet);
    }
  while(ndone < NTask);


  /* do final operations  */

  for(i = 0; i < N_gas; i++)
    if(P[i].Type == 0)
      {
	out[i] += in[i] * (1 + sum[i]);
      }

  myfree(sum);

  myfree(DataNodeList);
  myfree(DataIndexTable);
  myfree(Ngblist);
}





int conduction_evaluate(int target, int mode, double *in, double *out, double *sum,
			int *nexport, int *nsend_local)
{
  int startnode, numngb, listindex = 0;
  int j, n;
  MyDouble *pos;
  MyFloat h_i, rho;
  double dx, dy, dz;
  double h_i2, hinv_i, hinv4_i, hinv_j, hinv4_j;
  double dwk_i, h_j, dwk_j, dwk;
  double r, r2, u, Kappa_i, kappa_mean, w, out_sum, w_sum;


  if(mode == 0)
    {
      pos = P[target].Pos;
      h_i = P[target].Hsml;
      rho = SphP[target].d.Density;
      Kappa_i = Kappa[target];
    }
  else
    {
      pos = ConductionDataGet[target].Pos;
      h_i = ConductionDataGet[target].Hsml;
      rho = ConductionDataGet[target].Density;
      Kappa_i = ConductionDataGet[target].Kappa;
    }

  h_i2 = h_i * h_i;
  hinv_i = 1.0 / h_i;
#ifndef  TWODIMS
  hinv4_i = hinv_i * hinv_i * hinv_i * hinv_i;
#else
  hinv4_i = hinv_i * hinv_i * hinv_i / boxSize_Z;
#endif

  /* initialize variables before SPH loop is started */
  out_sum = 0;
  w_sum = 0;

  /* Now start the actual SPH computation for this particle */

  if(mode == 0)
    {
      startnode = All.MaxPart;	/* root node */
    }
  else
    {
      startnode = ConductionDataGet[target].NodeList[0];
      startnode = Nodes[startnode].u.d.nextnode;	/* open it */
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

	      h_j = P[j].Hsml;
	      if(r2 < h_i2 || r2 < h_j * h_j)
		{
		  r = sqrt(r2);
		  if(r > 0)
		    {
		      if(r2 < h_i2)
			{
			  u = r * hinv_i;
			  if(u < 0.5)
			    dwk_i = hinv4_i * u * (KERNEL_COEFF_3 * u - KERNEL_COEFF_4);
			  else
			    dwk_i = hinv4_i * KERNEL_COEFF_6 * (1.0 - u) * (1.0 - u);
			}
		      else
			dwk_i = 0;

		      if(r2 < h_j * h_j)
			{
			  hinv_j = 1.0 / h_j;
#ifndef  TWODIMS
			  hinv4_j = hinv_j * hinv_j * hinv_j * hinv_j;
#else
			  hinv4_j = hinv_j * hinv_j * hinv_j / boxSize_Z;
#endif
			  u = r * hinv_j;
			  if(u < 0.5)
			    dwk_j = hinv4_j * u * (KERNEL_COEFF_3 * u - KERNEL_COEFF_4);
			  else
			    dwk_j = hinv4_j * KERNEL_COEFF_6 * (1.0 - u) * (1.0 - u);
			}
		      else
			dwk_j = 0;


		      /* conduction equation kernel */
		      if((Kappa_i + Kappa[j]) > 0)
			{
			  kappa_mean = 2 * (Kappa_i * Kappa[j]) / (Kappa_i + Kappa[j]);
			  dwk = 0.5 * (dwk_i + dwk_j);

			  w = 2.0 * P[j].Mass / (rho * SphP[j].d.Density) * kappa_mean * (-dwk) / r;

			  out_sum += (-w * in[j]);
			  w_sum += w;
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
	      startnode = ConductionDataGet[target].NodeList[listindex];
	      if(startnode >= 0)
		startnode = Nodes[startnode].u.d.nextnode;	/* open it */
	    }
	}
    }


  /* Now collect the result at the right place */
  if(mode == 0)
    {
      out[target] = out_sum;
      sum[target] = w_sum;
    }
  else
    {
      ConductionDataResult[target].Out = out_sum;
      ConductionDataResult[target].Sum = w_sum;
    }

  return 0;
}


#endif
