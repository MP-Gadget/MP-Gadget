#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <gsl/gsl_math.h>

#include "allvars.h"
#include "proto.h"

/*! \file cosmic_rays_diffusion.c
*  \brief Computes cosmic ray diffusion with an implicit solver
*
*/

#ifdef CR_DIFFUSION

#include "cosmic_rays.h"

#define MAX_CR_DIFFUSION_ITER 25
#define CR_DIFFUSION_ITER_ACCURACY 1.0e-4

#define m_p (PROTONMASS * All.HubbleParam / All.UnitMass_in_g)

static struct crdiffusiondata_in
{
  MyDouble Pos[3];
  MyFloat Hsml;
  MyFloat Density;
  MyFloat CR_E0_Kappa[NUMCRPOP];
  MyFloat CR_n0_Kappa[NUMCRPOP];
  int NodeList[NODELISTLENGTH];
}
 *CR_DiffusionDataIn, *CR_DiffusionDataGet;


static struct crdiffusiondata_out
{
  MyFloat CR_E0_Out;
  MyFloat CR_E0_Sum;
  MyFloat CR_n0_Out;
  MyFloat CR_n0_Sum;
}
 *CR_DiffusionDataResult, *CR_DiffusionDataOut;


static double *CR_E0[NUMCRPOP], *CR_E0_Old[NUMCRPOP];
static double *CR_E0_Residual, *CR_E0_DVec, *CR_E0_QVec;
static double *CR_E0_Kappa[NUMCRPOP];
static double *CR_n0[NUMCRPOP], *CR_n0_Old[NUMCRPOP];
static double *CR_n0_Residual, *CR_n0_DVec, *CR_n0_QVec;
static double *CR_n0_Kappa[NUMCRPOP];


/* we will use the conjugate gradient method to compute a solution
   of the implicitly formulate diffusion equation */

/* Note: the conduction equation we solve is really formulated with u instead
   of T, i.e.  the factor (gamma-1)*mu*mp/k_B that converts from T to u is
   implicitely absorbed in a redefinition of kappa */


void cosmic_ray_diffusion(void)
{
  int i, iter;
  double CR_E0_delta0, CR_E0_delta1, CR_E0_alpha, CR_E0_beta;
  double CR_n0_delta0, CR_n0_delta1, CR_n0_alpha, CR_n0_beta;
  double a3inv, dt, rel_change, loc_max_rel_change, glob_max_rel_change;
  double sumnew, sumold, sumtransfer, sumnew_tot, sumold_tot, sumtransfer_tot;
  double kappa, kappa_egy, CR_q_i, cr_efac_i;
  double meanKineticEnergy, qmeanKin;
  int CRpop;

  if(ThisTask == 0)
    {
      printf("Start cosmic ray diffusion...\n");
      fflush(stdout);
    }

  for(CRpop = 0; CRpop < NUMCRPOP; CRpop++)
    {
      CR_E0[CRpop] = (double *) mymalloc("CR_E0[CRpop]", N_gas * sizeof(double));
      CR_E0_Old[CRpop] = (double *) mymalloc("CR_E0_Old[CRpop]", N_gas * sizeof(double));
      CR_E0_Kappa[CRpop] = (double *) mymalloc("CR_E0_Kappa[CRpop]", N_gas * sizeof(double));
    }

  CR_E0_Residual = (double *) mymalloc("CR_E0_Residual", N_gas * sizeof(double));
  CR_E0_DVec = (double *) mymalloc("CR_E0_DVec", N_gas * sizeof(double));
  CR_E0_QVec = (double *) mymalloc("CR_E0_QVec", N_gas * sizeof(double));

  for(CRpop = 0; CRpop < NUMCRPOP; CRpop++)
    {
      CR_n0[CRpop] = (double *) mymalloc("CR_n0[CRpop]", N_gas * sizeof(double));
      CR_n0_Old[CRpop] = (double *) mymalloc("CR_n0_Old[CRpop]", N_gas * sizeof(double));
      CR_n0_Kappa[CRpop] = (double *) mymalloc("CR_n0_Kappa[CRpop]", N_gas * sizeof(double));
    }

  CR_n0_Residual = (double *) mymalloc("CR_n0_Residual", N_gas * sizeof(double));
  CR_n0_DVec = (double *) mymalloc("CR_n0_DVec", N_gas * sizeof(double));
  CR_n0_QVec = (double *) mymalloc("CR_n0_QVec", N_gas * sizeof(double));

  if(All.ComovingIntegrationOn)
    a3inv = 1 / (All.Time * All.Time * All.Time);
  else
    a3inv = 1.0;


  dt = (All.CR_Diffusion_Ti_endstep - All.CR_Diffusion_Ti_begstep) * All.Timebase_interval;

  if(All.ComovingIntegrationOn)
    dt *= All.Time / hubble_function(All.Time);

  if(ThisTask == 0)
    {
      printf("dt=%g\n", dt);
    }

  /* First, let's compute the diffusivities for each particle */

  for(i = 0; i < N_gas; i++)
    {
      if(P[i].Type == 0)
	{
	  for(CRpop = 0; CRpop < NUMCRPOP; CRpop++)
	    {
	      CR_E0[CRpop][i] = CR_E0_Old[CRpop][i] = SphP[i].CR_E0[CRpop];
	      CR_n0[CRpop][i] = CR_n0_Old[CRpop][i] = SphP[i].CR_n0[CRpop];

	      kappa = All.CR_DiffusionCoeff;

	      if(All.CR_DiffusionDensScaling != 0.0)
		kappa *=
		  pow(SphP[i].d.Density * a3inv / All.CR_DiffusionDensZero, All.CR_DiffusionDensScaling);

	      if(All.CR_DiffusionEntropyScaling != 0.0)
		kappa *= pow(SphP[i].Entropy / All.CR_DiffusionEntropyZero, All.CR_DiffusionEntropyScaling);

	      if(SphP[i].CR_E0[CRpop] > 0)
		{
		  CR_q_i = SphP[i].CR_q0[CRpop] * pow(SphP[i].d.Density * a3inv, 0.33333);

		  cr_efac_i =
		    CR_Tab_MeanEnergy(CR_q_i, All.CR_Alpha[CRpop] - 0.3333) /
		    CR_Tab_MeanEnergy(CR_q_i, All.CR_Alpha[CRpop]);

		  kappa *= (All.CR_Alpha[CRpop] - 1) / (All.CR_Alpha[CRpop] - 1.33333) * pow(CR_q_i, 0.3333);

		  kappa_egy = kappa * cr_efac_i;
		}
	      else
		kappa_egy = kappa;

	      CR_E0_Kappa[CRpop][i] = kappa_egy;
	      CR_n0_Kappa[CRpop][i] = kappa;

	      /* we'll factor the timestep into the diffusivities, for simplicity */
	      CR_E0_Kappa[CRpop][i] *= dt;
	      CR_n0_Kappa[CRpop][i] *= dt;
	    }
	}
    }



  /* Let's start the Conjugate Gradient Algorithm */

  /* Initialization */
  for(CRpop = 0; CRpop < NUMCRPOP; CRpop++)
    {
      cosmic_ray_diffusion_matrix_multiply(CR_E0_Old[CRpop], CR_E0_Residual, CR_n0_Old[CRpop], CR_n0_Residual,
					   CRpop);

      for(i = 0; i < N_gas; i++)
	{
	  if(P[i].Type == 0)
	    {
	      CR_E0_Residual[i] = CR_E0_Old[CRpop][i] - CR_E0_Residual[i];
	      CR_E0_DVec[i] = CR_E0_Residual[i];

	      CR_n0_Residual[i] = CR_n0_Old[CRpop][i] - CR_n0_Residual[i];
	      CR_n0_DVec[i] = CR_n0_Residual[i];
	    }
	}

      CR_E0_delta1 = cosmic_ray_diffusion_vector_multiply(CR_E0_Residual, CR_E0_Residual);
      CR_E0_delta0 = CR_E0_delta1;

      CR_n0_delta1 = cosmic_ray_diffusion_vector_multiply(CR_n0_Residual, CR_n0_Residual);
      CR_n0_delta0 = CR_n0_delta1;

      iter = 0;			/* iteration counter */
      glob_max_rel_change = 1 + CR_DIFFUSION_ITER_ACCURACY;	/* to make sure that we enter the iteration */


      while(iter < MAX_CR_DIFFUSION_ITER && glob_max_rel_change > CR_DIFFUSION_ITER_ACCURACY
	    && CR_E0_delta1 > 0)
	{
	  cosmic_ray_diffusion_matrix_multiply(CR_E0_DVec, CR_E0_QVec, CR_n0_DVec, CR_n0_QVec, CRpop);

	  CR_E0_alpha = CR_E0_delta1 / cosmic_ray_diffusion_vector_multiply(CR_E0_DVec, CR_E0_QVec);
	  CR_n0_alpha = CR_n0_delta1 / cosmic_ray_diffusion_vector_multiply(CR_n0_DVec, CR_n0_QVec);

	  for(i = 0, loc_max_rel_change = 0; i < N_gas; i++)
	    {
	      CR_E0[CRpop][i] += CR_E0_alpha * CR_E0_DVec[i];
	      CR_E0_Residual[i] -= CR_E0_alpha * CR_E0_QVec[i];
	      CR_n0[CRpop][i] += CR_n0_alpha * CR_n0_DVec[i];
	      CR_n0_Residual[i] -= CR_n0_alpha * CR_n0_QVec[i];

	      if(CR_E0[CRpop][i] > 0)
		{
		  rel_change = CR_E0_alpha * CR_E0_DVec[i] / CR_E0[CRpop][i];
		  if(loc_max_rel_change < rel_change)
		    loc_max_rel_change = rel_change;
		}
	      if(CR_n0[CRpop][i] > 0)
		{
		  rel_change = CR_n0_alpha * CR_n0_DVec[i] / CR_n0[CRpop][i];
		  if(loc_max_rel_change < rel_change)
		    loc_max_rel_change = rel_change;
		}
	    }

	  CR_E0_delta0 = CR_E0_delta1;
	  CR_E0_delta1 = cosmic_ray_diffusion_vector_multiply(CR_E0_Residual, CR_E0_Residual);
	  CR_n0_delta0 = CR_n0_delta1;
	  CR_n0_delta1 = cosmic_ray_diffusion_vector_multiply(CR_n0_Residual, CR_n0_Residual);

	  CR_E0_beta = CR_E0_delta1 / CR_E0_delta0;
	  CR_n0_beta = CR_n0_delta1 / CR_n0_delta0;

	  for(i = 0; i < N_gas; i++)
	    {
	      CR_E0_DVec[i] = CR_E0_Residual[i] + CR_E0_beta * CR_E0_DVec[i];
	      CR_n0_DVec[i] = CR_n0_Residual[i] + CR_n0_beta * CR_n0_DVec[i];
	    }

	  iter++;

	  MPI_Allreduce(&loc_max_rel_change, &glob_max_rel_change, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

	  if(ThisTask == 0)
	    {
	      printf
		("cosmic ray diffusion iter=%d  CR_E0_delta1=%g delta1/delta0=%g CR_n0_delta1=%g delta1/delta0=%g  max-rel-change=%g\n",
		 iter, CR_E0_delta1, CR_E0_delta1 / CR_E0_delta0, CR_n0_delta1, CR_n0_delta1 / CR_n0_delta0,
		 glob_max_rel_change);
	      fflush(stdout);
	    }
	}



      /* Now we have the solution vectors */
      /* set the new cosmic ray prorperties */


      for(i = 0, sumnew = sumold = sumtransfer = 0; i < N_gas; i++)
	{
	  if(P[i].Type == 0)
	    {
	      SphP[i].CR_E0[CRpop] = CR_E0[CRpop][i];
	      SphP[i].CR_n0[CRpop] = CR_n0[CRpop][i];
	      SphP[i].CR_DeltaE[CRpop] = 0;
	      SphP[i].CR_DeltaN[CRpop] = 0;

	      if(SphP[i].CR_n0[CRpop] > 1.0e-12 && SphP[i].CR_E0[CRpop] > 0)
		{
		  meanKineticEnergy = SphP[i].CR_E0[CRpop] * m_p / SphP[i].CR_n0[CRpop];

		  qmeanKin = CR_q_from_mean_kinetic_energy(meanKineticEnergy, CRpop);

		  SphP[i].CR_q0[CRpop] = qmeanKin * pow(SphP[i].d.Density * a3inv, -(1.0 / 3.0));
		  SphP[i].CR_C0[CRpop] = SphP[i].CR_n0[CRpop] * (All.CR_Alpha[CRpop] - 1.0)
		    * pow(SphP[i].CR_q0[CRpop], All.CR_Alpha[CRpop] - 1.0);
		}
	      else
		{
		  SphP[i].CR_E0[CRpop] = 0.0;
		  SphP[i].CR_n0[CRpop] = 0.0;

		  SphP[i].CR_q0[CRpop] = 1.0e10;
		  SphP[i].CR_C0[CRpop] = 0.0;
		}

	      sumnew += CR_E0[CRpop][i];
	      sumold += CR_E0_Old[CRpop][i];
	      sumtransfer += fabs(CR_E0[CRpop][i] - CR_E0_Old[CRpop][i]);
	    }
	}

      MPI_Allreduce(&sumnew, &sumnew_tot, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(&sumold, &sumold_tot, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(&sumtransfer, &sumtransfer_tot, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

      if(ThisTask == 0)
	{
	  printf
	    ("\ncosmic ray diffusion finished. energy_before=%g energy_after=%g  rel-change=%g rel-transfer=%g\n\n",
	     sumold_tot, sumnew_tot, sumold_tot ? ((sumnew_tot - sumold_tot) / sumold_tot) : 0.0,
	     sumold_tot ? (sumtransfer_tot / sumold_tot) : 0.0);
	  fflush(stdout);
	}
    }

  myfree(CR_E0_QVec);
  myfree(CR_E0_DVec);
  myfree(CR_E0_Residual);

  myfree(CR_n0_QVec);
  myfree(CR_n0_DVec);
  myfree(CR_n0_Residual);

  for(CRpop = 0; CRpop < NUMCRPOP; CRpop++)
    {
      myfree(CR_n0_Kappa[CRpop]);
      myfree(CR_n0_Old[CRpop]);
      myfree(CR_n0[CRpop]);

      myfree(CR_E0_Kappa[CRpop]);
      myfree(CR_E0_Old[CRpop]);
      myfree(CR_E0[CRpop]);
    }
}


double cosmic_ray_diffusion_vector_multiply(double *a, double *b)
{
  int i;
  double sum, sumall;

  for(i = 0, sum = 0; i < N_gas; i++)
    if(P[i].Type == 0)
      sum += a[i] * b[i];

  MPI_Allreduce(&sum, &sumall, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  return sumall;
}


void cosmic_ray_diffusion_matrix_multiply(double *cr_E0_in, double *cr_E0_out, double *cr_n0_in,
					  double *cr_n0_out, int CRpop)
{
  int i, j, k, ngrp, ndone, ndone_flag, dummy;
  int sendTask, recvTask, nexport, nimport, place;
  double *cr_E0_sum, *cr_n0_sum;

  /* allocate buffers to arrange communication */

  Ngblist = (int *) mymalloc("Ngblist", NumPart * sizeof(int));

  All.BunchSize =
    (int) ((All.BufferSize * 1024 * 1024) / (sizeof(struct data_index) + sizeof(struct data_nodelist) +
					     sizeof(struct crdiffusiondata_in) +
					     sizeof(struct crdiffusiondata_out) +
					     sizemax(sizeof(struct crdiffusiondata_in),
						     sizeof(struct crdiffusiondata_out))));
  DataIndexTable =
    (struct data_index *) mymalloc("DataIndexTable", All.BunchSize * sizeof(struct data_index));
  DataNodeList =
    (struct data_nodelist *) mymalloc("DataNodeList", All.BunchSize * sizeof(struct data_nodelist));


  cr_E0_sum = (double *) mymalloc("cr_E0_sum", N_gas * sizeof(double));
  cr_n0_sum = (double *) mymalloc("cr_n0_sum", N_gas * sizeof(double));


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
	    if(cosmic_ray_diffusion_evaluate(i, 0,
					     cr_E0_in, cr_E0_out, cr_E0_sum,
					     cr_n0_in, cr_n0_out, cr_n0_sum, &nexport, Send_count, CRpop) < 0)
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

      CR_DiffusionDataGet =
	(struct crdiffusiondata_in *) mymalloc(nimport * sizeof(struct crdiffusiondata_in));
      CR_DiffusionDataIn =
	(struct crdiffusiondata_in *) mymalloc(nexport * sizeof(struct crdiffusiondata_in));

      /* prepare particle data for export */

      for(j = 0; j < nexport; j++)
	{
	  place = DataIndexTable[j].Index;

	  for(k = 0; k < 3; k++)
	    CR_DiffusionDataIn[j].Pos[k] = P[place].Pos[k];

	  CR_DiffusionDataIn[j].Hsml = PPP[place].Hsml;
	  CR_DiffusionDataIn[j].Density = SphP[place].d.Density;
	  CR_DiffusionDataIn[j].CR_E0_Kappa[CRpop] = CR_E0_Kappa[CRpop][place];
	  CR_DiffusionDataIn[j].CR_n0_Kappa[CRpop] = CR_n0_Kappa[CRpop][place];

	  memcpy(CR_DiffusionDataIn[j].NodeList,
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
		  MPI_Sendrecv(&CR_DiffusionDataIn[Send_offset[recvTask]],
			       Send_count[recvTask] * sizeof(struct crdiffusiondata_in), MPI_BYTE,
			       recvTask, TAG_HYDRO_A,
			       &CR_DiffusionDataGet[Recv_offset[recvTask]],
			       Recv_count[recvTask] * sizeof(struct crdiffusiondata_in), MPI_BYTE,
			       recvTask, TAG_HYDRO_A, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}
	    }
	}

      myfree(CR_DiffusionDataIn);
      CR_DiffusionDataResult =
	(struct crdiffusiondata_out *) mymalloc(nimport * sizeof(struct crdiffusiondata_out));
      CR_DiffusionDataOut =
	(struct crdiffusiondata_out *) mymalloc(nexport * sizeof(struct crdiffusiondata_out));


      /* now do the particles that were sent to us */
      for(j = 0; j < nimport; j++)
	cosmic_ray_diffusion_evaluate(j, 1,
				      cr_E0_in, cr_E0_out, cr_E0_sum,
				      cr_n0_in, cr_n0_out, cr_n0_sum, &dummy, &dummy, CRpop);

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
		  MPI_Sendrecv(&CR_DiffusionDataResult[Recv_offset[recvTask]],
			       Recv_count[recvTask] * sizeof(struct crdiffusiondata_out),
			       MPI_BYTE, recvTask, TAG_HYDRO_B,
			       &CR_DiffusionDataOut[Send_offset[recvTask]],
			       Send_count[recvTask] * sizeof(struct crdiffusiondata_out),
			       MPI_BYTE, recvTask, TAG_HYDRO_B, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}
	    }
	}

      /* add the result to the local particles */
      for(j = 0; j < nexport; j++)
	{
	  place = DataIndexTable[j].Index;

	  cr_E0_out[place] += CR_DiffusionDataOut[j].CR_E0_Out;
	  cr_E0_sum[place] += CR_DiffusionDataOut[j].CR_E0_Sum;
	  cr_n0_out[place] += CR_DiffusionDataOut[j].CR_n0_Out;
	  cr_n0_sum[place] += CR_DiffusionDataOut[j].CR_n0_Sum;
	}

      myfree(CR_DiffusionDataOut);
      myfree(CR_DiffusionDataResult);
      myfree(CR_DiffusionDataGet);
    }
  while(ndone < NTask);


  /* do final operations  */

  for(i = 0; i < N_gas; i++)
    if(P[i].Type == 0)
      {
	cr_E0_out[i] += cr_E0_in[i] * (1 + cr_E0_sum[i]);
	cr_n0_out[i] += cr_n0_in[i] * (1 + cr_n0_sum[i]);
      }

  myfree(cr_n0_sum);
  myfree(cr_E0_sum);

  myfree(DataNodeList);
  myfree(DataIndexTable);
  myfree(Ngblist);
}





int cosmic_ray_diffusion_evaluate(int target, int mode,
				  double *cr_E0_in, double *cr_E0_out, double *cr_E0_sum,
				  double *cr_n0_in, double *cr_n0_out, double *cr_n0_sum,
				  int *nexport, int *nsend_local, int CRpop)
{
  int startnode, numngb, listindex = 0;
  int j, n;
  MyDouble *pos;
  MyFloat h_i, rho;
  double dx, dy, dz;
  double h_i2, hinv_i, hinv4_i, hinv_j, hinv4_j;
  double dwk_i, h_j, dwk_j, dwk;
  double r, r2, u, CR_E0_Kappa_i, kappa_mean, w, wfac, cr_E0_out_sum, cr_E0_w_sum;
  double CR_n0_Kappa_i, cr_n0_out_sum, cr_n0_w_sum;

  if(mode == 0)
    {
      pos = P[target].Pos;
      h_i = PPP[target].Hsml;
      rho = SphP[target].d.Density;
      CR_E0_Kappa_i = CR_E0_Kappa[CRpop][target];
      CR_n0_Kappa_i = CR_n0_Kappa[CRpop][target];
    }
  else
    {
      pos = CR_DiffusionDataGet[target].Pos;
      h_i = CR_DiffusionDataGet[target].Hsml;
      rho = CR_DiffusionDataGet[target].Density;
      CR_E0_Kappa_i = CR_DiffusionDataGet[target].CR_E0_Kappa[CRpop];
      CR_n0_Kappa_i = CR_DiffusionDataGet[target].CR_n0_Kappa[CRpop];
    }

  h_i2 = h_i * h_i;
  hinv_i = 1.0 / h_i;
#ifndef  TWODIMS
  hinv4_i = hinv_i * hinv_i * hinv_i * hinv_i;
#else
  hinv4_i = hinv_i * hinv_i * hinv_i / boxSize_Z;
#endif

  /* initialize variables before SPH loop is started */
  cr_E0_out_sum = 0;
  cr_E0_w_sum = 0;
  cr_n0_out_sum = 0;
  cr_n0_w_sum = 0;

  /* Now start the actual SPH computation for this particle */

  if(mode == 0)
    {
      startnode = All.MaxPart;	/* root node */
    }
  else
    {
      startnode = CR_DiffusionDataGet[target].NodeList[0];
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

	      h_j = PPP[j].Hsml;
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


		      dwk = 0.5 * (dwk_i + dwk_j);
		      wfac = 2.0 * P[j].Mass / (rho * SphP[j].d.Density) * (-dwk) / r;

		      /* cosmic ray diffusion equation kernel */
		      if((CR_E0_Kappa_i + CR_E0_Kappa[CRpop][j]) > 0)
			kappa_mean =
			  2 * (CR_E0_Kappa_i * CR_E0_Kappa[CRpop][j]) / (CR_E0_Kappa_i +
									 CR_E0_Kappa[CRpop][j]);
		      else
			kappa_mean = 0;

		      w = wfac * kappa_mean;
		      cr_E0_out_sum += (-w * cr_E0_in[j]);
		      cr_E0_w_sum += w;


		      /* cosmic ray diffusion equation kernel */
		      if((CR_n0_Kappa_i + CR_n0_Kappa[CRpop][j]) > 0)
			kappa_mean =
			  2 * (CR_n0_Kappa_i * CR_n0_Kappa[CRpop][j]) / (CR_n0_Kappa_i +
									 CR_n0_Kappa[CRpop][j]);
		      else
			kappa_mean = 0;

		      w = wfac * kappa_mean;
		      cr_n0_out_sum += (-w * cr_n0_in[j]);
		      cr_n0_w_sum += w;
		    }
		}
	    }
	}

      if(mode == 1)
	{
	  listindex++;
	  if(listindex < NODELISTLENGTH)
	    {
	      startnode = CR_DiffusionDataGet[target].NodeList[listindex];
	      if(startnode >= 0)
		startnode = Nodes[startnode].u.d.nextnode;	/* open it */
	    }
	}
    }


  /* Now collect the result at the right place */
  if(mode == 0)
    {
      cr_E0_out[target] = cr_E0_out_sum;
      cr_E0_sum[target] = cr_E0_w_sum;
      cr_n0_out[target] = cr_n0_out_sum;
      cr_n0_sum[target] = cr_n0_w_sum;
    }
  else
    {
      CR_DiffusionDataResult[target].CR_E0_Out = cr_E0_out_sum;
      CR_DiffusionDataResult[target].CR_E0_Sum = cr_E0_w_sum;
      CR_DiffusionDataResult[target].CR_n0_Out = cr_n0_out_sum;
      CR_DiffusionDataResult[target].CR_n0_Sum = cr_n0_w_sum;
    }

  return 0;
}


#endif
