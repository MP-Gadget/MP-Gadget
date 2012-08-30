#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <gsl/gsl_math.h>

#include "allvars.h"
#include "proto.h"

#ifdef CR_DIFFUSION_GREEN
#include "cosmic_rays.h"

void compute_diff_weights(int target, int mode);
void scatter_diffusion(int target, int mode);

#define m_p (PROTONMASS * All.HubbleParam / All.UnitMass_in_g)

void greenf_diffusion(void)
{
  int *noffset, *nbuffer, *nsend, *nsend_local, *numlist, *ndonelist;
  int i, j, n;
  int ndone;
  long long ntot, ntotleft;
  int maxfill, source;
  int level, ngrp, sendTask, recvTask;
  int place, nexport;
  double cr_efac_i, kappa_egy;
  double kappa, a3inv, egysum, egytot, egytot_before;
  double meanKineticEnergy, qmeanKin;
  double CR_q_i;
  MPI_Status status;


  if(ThisTask == 0)
    {
      printf("Doing diffusion step with Green function method\n");
      fflush(stdout);
    }

  for(CRpop = 0; CRpop < NUMCRPOP; CRpop++)
    {

      noffset = mymalloc("noffset", sizeof(int) * NTask);	/* offsets of bunches in common list */
      nbuffer = mymalloc("nbuffer", sizeof(int) * NTask);
      nsend_local = mymalloc("nsend_local", sizeof(int) * NTask);
      nsend = mymalloc("nsend", sizeof(int) * NTask * NTask);
      ndonelist = mymalloc("ndonelist", sizeof(int) * NTask);


      if(All.ComovingIntegrationOn)
	a3inv = 1 / (All.Time * All.Time * All.Time);
      else
	a3inv = 1;


      egysum = 0;

      for(n = 0, NumSphUpdate = 0; n < N_gas; n++)
	{
	  if(P[n].Type == 0)
	    {
	      NumSphUpdate++;

	      egysum += SphP[n].CR_E0[CRpop];

	      SphP[n].CR_DeltaE[CRpop] = 0;
	      SphP[n].CR_DeltaN[CRpop] = 0;

	      kappa = All.CR_DiffusionCoeff;

	      if(All.CR_DiffusionDensScaling != 0.0)
		kappa *=
		  pow(SphP[n].d.Density * a3inv / All.CR_DiffusionDensZero, All.CR_DiffusionDensScaling);

	      if(All.CR_DiffusionEntropyScaling != 0.0)
		kappa *= pow(SphP[n].Entropy / All.CR_DiffusionEntropyZero, All.CR_DiffusionEntropyScaling);

	      if(SphP[n].CR_E0[CRpop] > 0)
		{
		  CR_q_i = SphP[n].CR_q0[CRpop] * pow(SphP[n].d.Density * a3inv, 0.33333);

		  cr_efac_i =
		    CR_Tab_MeanEnergy(CR_q_i, All.CR_Alpha[CRpop] - 0.3333, CRpop)
		    / CR_Tab_MeanEnergy(CR_q_i, All.CR_Alpha[CRpop], CRpop);

		  kappa *= (All.CR_Alpha[CRpop] - 1) / (All.CR_Alpha[CRpop] - 1.33333) * pow(CR_q_i, 0.3333);

		  kappa_egy = kappa * cr_efac_i;
		}
	      else
		kappa_egy = kappa;


	      SphP[n].CR_Kappa[CRpop] = kappa;
	      SphP[n].CR_Kappa_egy[CRpop] = kappa_egy;
	    }
	}


      MPI_Allreduce(&egysum, &egytot_before, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

      numlist = mymalloc("numlist", NTask * sizeof(int) * NTask);
      MPI_Allgather(&NumSphUpdate, 1, MPI_INT, numlist, 1, MPI_INT, MPI_COMM_WORLD);
      for(i = 0, ntot = 0; i < NTask; i++)
	ntot += numlist[i];
      myfree(numlist);


      /* first, sum the weights */


      i = 0;			/* beginn with this index */
      ntotleft = ntot;		/* particles left for all tasks together */

      while(ntotleft > 0)
	{
	  for(j = 0; j < NTask; j++)
	    nsend_local[j] = 0;

	  /* do local particles and prepare export list */

	  for(nexport = 0, ndone = 0; i < NumPart && nexport < All.BunchSizeDensity - NTask; i++)
	    if(P[i].Type == 0)
	      {
		ndone++;

		for(j = 0; j < NTask; j++)
		  Exportflag[j] = 0;

		compute_diff_weights(i, 0);

		for(j = 0; j < NTask; j++)
		  {
		    if(Exportflag[j])
		      {
			DensDataIn[nexport].Pos[0] = P[i].Pos[0];
			DensDataIn[nexport].Pos[1] = P[i].Pos[1];
			DensDataIn[nexport].Pos[2] = P[i].Pos[2];
			DensDataIn[nexport].Hsml = PPP[i].Hsml;
			DensDataIn[nexport].CR_Kappa[CRpop] = SphP[i].CR_Kappa[CRpop];
			DensDataIn[nexport].CR_Kappa_egy[CRpop] = SphP[i].CR_Kappa_egy[CRpop];

			DensDataIn[nexport].Index = i;
			DensDataIn[nexport].Task = j;
			nexport++;
			nsend_local[j]++;
		      }
		  }
	      }


	  qsort(DensDataIn, nexport, sizeof(struct densdata_in), dens_compare_key);

	  for(j = 1, noffset[0] = 0; j < NTask; j++)
	    noffset[j] = noffset[j - 1] + nsend_local[j - 1];

	  MPI_Allgather(nsend_local, NTask, MPI_INT, nsend, NTask, MPI_INT, MPI_COMM_WORLD);


	  /* now do the particles that need to be exported */

	  for(level = 1; level < (1 << PTask); level++)
	    {
	      for(j = 0; j < NTask; j++)
		nbuffer[j] = 0;
	      for(ngrp = level; ngrp < (1 << PTask); ngrp++)
		{
		  maxfill = 0;
		  for(j = 0; j < NTask; j++)
		    {
		      if((j ^ ngrp) < NTask)
			if(maxfill < nbuffer[j] + nsend[(j ^ ngrp) * NTask + j])
			  maxfill = nbuffer[j] + nsend[(j ^ ngrp) * NTask + j];
		    }
		  if(maxfill >= All.BunchSizeDensity)
		    break;

		  sendTask = ThisTask;
		  recvTask = ThisTask ^ ngrp;

		  if(recvTask < NTask)
		    {
		      if(nsend[ThisTask * NTask + recvTask] > 0 || nsend[recvTask * NTask + ThisTask] > 0)
			{
			  /* get the particles */
			  MPI_Sendrecv(&DensDataIn[noffset[recvTask]],
				       nsend_local[recvTask] * sizeof(struct densdata_in), MPI_BYTE,
				       recvTask, TAG_CONDUCT_A,
				       &DensDataGet[nbuffer[ThisTask]],
				       nsend[recvTask * NTask + ThisTask] * sizeof(struct densdata_in),
				       MPI_BYTE, recvTask, TAG_CONDUCT_A, MPI_COMM_WORLD, &status);
			}
		    }

		  for(j = 0; j < NTask; j++)
		    if((j ^ ngrp) < NTask)
		      nbuffer[j] += nsend[(j ^ ngrp) * NTask + j];
		}


	      for(j = 0; j < nbuffer[ThisTask]; j++)
		compute_diff_weights(j, 1);


	      for(j = 0; j < NTask; j++)
		nbuffer[j] = 0;
	      for(ngrp = level; ngrp < (1 << PTask); ngrp++)
		{
		  maxfill = 0;
		  for(j = 0; j < NTask; j++)
		    {
		      if((j ^ ngrp) < NTask)
			if(maxfill < nbuffer[j] + nsend[(j ^ ngrp) * NTask + j])
			  maxfill = nbuffer[j] + nsend[(j ^ ngrp) * NTask + j];
		    }
		  if(maxfill >= All.BunchSizeDensity)
		    break;

		  sendTask = ThisTask;
		  recvTask = ThisTask ^ ngrp;

		  if(recvTask < NTask)
		    {
		      if(nsend[ThisTask * NTask + recvTask] > 0 || nsend[recvTask * NTask + ThisTask] > 0)
			{
			  /* send the results */
			  MPI_Sendrecv(&DensDataResult[nbuffer[ThisTask]],
				       nsend[recvTask * NTask + ThisTask] * sizeof(struct densdata_out),
				       MPI_BYTE, recvTask, TAG_CONDUCT_B,
				       &DensDataPartialResult[noffset[recvTask]],
				       nsend_local[recvTask] * sizeof(struct densdata_out),
				       MPI_BYTE, recvTask, TAG_CONDUCT_B, MPI_COMM_WORLD, &status);

			  /* add the result to the particles */
			  for(j = 0; j < nsend_local[recvTask]; j++)
			    {
			      source = j + noffset[recvTask];
			      place = DensDataIn[source].Index;

			      SphP[place].CR_WeightSum += DensDataPartialResult[source].CR_WeightSum;
			      SphP[place].CR_WeightSum_egy += DensDataPartialResult[source].CR_WeightSum_egy;
			    }
			}
		    }

		  for(j = 0; j < NTask; j++)
		    if((j ^ ngrp) < NTask)
		      nbuffer[j] += nsend[(j ^ ngrp) * NTask + j];
		}
	      level = ngrp - 1;
	    }

	  MPI_Allgather(&ndone, 1, MPI_INT, ndonelist, 1, MPI_INT, MPI_COMM_WORLD);
	  for(j = 0; j < NTask; j++)
	    ntotleft -= ndonelist[j];
	}



  /*************** now do the diffusion step itself */





      numlist = mymalloc("numlist", NTask * sizeof(int) * NTask);
      MPI_Allgather(&NumSphUpdate, 1, MPI_INT, numlist, 1, MPI_INT, MPI_COMM_WORLD);
      for(i = 0, ntot = 0; i < NTask; i++)
	ntot += numlist[i];
      myfree(numlist);

      /* first, sum the weights */

      i = 0;			/* beginn with this index */
      ntotleft = ntot;		/* particles left for all tasks together */

      while(ntotleft > 0)
	{
	  for(j = 0; j < NTask; j++)
	    nsend_local[j] = 0;

	  /* do local particles and prepare export list */

	  for(nexport = 0, ndone = 0; i < NumPart && nexport < All.BunchSizeDensity - NTask; i++)
	    if(P[i].Type == 0)
	      {
		ndone++;

		for(j = 0; j < NTask; j++)
		  Exportflag[j] = 0;

		scatter_diffusion(i, 0);

		for(j = 0; j < NTask; j++)
		  {
		    if(Exportflag[j])
		      {
			DensDataIn[nexport].Pos[0] = P[i].Pos[0];
			DensDataIn[nexport].Pos[1] = P[i].Pos[1];
			DensDataIn[nexport].Pos[2] = P[i].Pos[2];

			DensDataIn[nexport].CR_Kappa[CRpop] = SphP[i].CR_Kappa[CRpop];
			DensDataIn[nexport].CR_Kappa_egy[CRpop] = SphP[i].CR_Kappa_egy[CRpop];
			DensDataIn[nexport].CR_WeightSum = SphP[i].CR_WeightSum;
			DensDataIn[nexport].CR_WeightSum_egy = SphP[i].CR_WeightSum_egy;
			DensDataIn[nexport].Hsml = PPP[i].Hsml;
			DensDataIn[nexport].CR_E0[CRpop] = SphP[i].CR_E0[CRpop];
			DensDataIn[nexport].CR_n0[CRpop] = SphP[i].CR_n0[CRpop];

			DensDataIn[nexport].Index = i;
			DensDataIn[nexport].Task = j;
			nexport++;
			nsend_local[j]++;
		      }
		  }
	      }


	  qsort(DensDataIn, nexport, sizeof(struct densdata_in), dens_compare_key);

	  for(j = 1, noffset[0] = 0; j < NTask; j++)
	    noffset[j] = noffset[j - 1] + nsend_local[j - 1];

	  MPI_Allgather(nsend_local, NTask, MPI_INT, nsend, NTask, MPI_INT, MPI_COMM_WORLD);


	  /* now do the particles that need to be exported */

	  for(level = 1; level < (1 << PTask); level++)
	    {
	      for(j = 0; j < NTask; j++)
		nbuffer[j] = 0;
	      for(ngrp = level; ngrp < (1 << PTask); ngrp++)
		{
		  maxfill = 0;
		  for(j = 0; j < NTask; j++)
		    {
		      if((j ^ ngrp) < NTask)
			if(maxfill < nbuffer[j] + nsend[(j ^ ngrp) * NTask + j])
			  maxfill = nbuffer[j] + nsend[(j ^ ngrp) * NTask + j];
		    }
		  if(maxfill >= All.BunchSizeDensity)
		    break;

		  sendTask = ThisTask;
		  recvTask = ThisTask ^ ngrp;

		  if(recvTask < NTask)
		    {
		      if(nsend[ThisTask * NTask + recvTask] > 0 || nsend[recvTask * NTask + ThisTask] > 0)
			{
			  /* get the particles */
			  MPI_Sendrecv(&DensDataIn[noffset[recvTask]],
				       nsend_local[recvTask] * sizeof(struct densdata_in), MPI_BYTE,
				       recvTask, TAG_CONDUCT_A,
				       &DensDataGet[nbuffer[ThisTask]],
				       nsend[recvTask * NTask + ThisTask] * sizeof(struct densdata_in),
				       MPI_BYTE, recvTask, TAG_CONDUCT_A, MPI_COMM_WORLD, &status);
			}
		    }

		  for(j = 0; j < NTask; j++)
		    if((j ^ ngrp) < NTask)
		      nbuffer[j] += nsend[(j ^ ngrp) * NTask + j];
		}


	      for(j = 0; j < nbuffer[ThisTask]; j++)
		scatter_diffusion(j, 1);


	      level = ngrp - 1;
	    }

	  MPI_Allgather(&ndone, 1, MPI_INT, ndonelist, 1, MPI_INT, MPI_COMM_WORLD);
	  for(j = 0; j < NTask; j++)
	    ntotleft -= ndonelist[j];
	}



      /* now set the new cosmic ray prorperties */


      egysum = 0;

      for(n = 0; n < N_gas; n++)
	{
	  if(P[n].Type == 0)
	    {
	      SphP[n].CR_E0[CRpop] = SphP[n].CR_DeltaE[CRpop];
	      SphP[n].CR_n0[CRpop] = SphP[n].CR_DeltaN[CRpop];

	      egysum += SphP[n].CR_E0[CRpop];

	      SphP[n].CR_DeltaE[CRpop] = 0;
	      SphP[n].CR_DeltaN[CRpop] = 0;

	      if(SphP[n].CR_n0[CRpop] > 1.0e-12 && SphP[n].CR_E0[CRpop] > 0)
		{
		  meanKineticEnergy = SphP[n].CR_E0[CRpop] * m_p / SphP[n].CR_n0[CRpop];

		  qmeanKin = CR_q_from_mean_kinetic_energy(meanKineticEnergy[CRpop], CRpop);

		  SphP[n].CR_q0[CRpop] = qmeanKin * pow(SphP[n].d.Density * a3inv, -(1.0 / 3.0));
		  SphP[n].CR_C0[CRpop] = SphP[n].CR_n0[CRpop] * (All.CR_Alpha[CRpop] - 1.0) *
		    pow(SphP[n].CR_q0[CRpop], All.CR_Alpha[CRpop] - 1.0);
		}
	      else
		{
		  SphP[n].CR_E0[CRpop] = 0.0;
		  SphP[n].CR_n0[CRpop] = 0.0;

		  SphP[n].CR_q0[CRpop] = 1.0e10;
		  SphP[n].CR_C0[CRpop] = 0.0;
		}
	    }
	}

      MPI_Allreduce(&egysum, &egytot, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      if(ThisTask == 0)
	{
	  printf("Energy before/after= %g | %g\n", egytot_before, egytot);
	  fflush(stdout);
	}


      myfree(ndonelist);
      myfree(nsend);
      myfree(nsend_local);
      myfree(nbuffer);
      myfree(noffset);


      All.TimeOfLastDiffusion = All.Time;
    }
}




void compute_diff_weights(int target, int mode)
{
  int j, n;
  int startnode, numngb_inbox;
  double h;
  double weightsum, weightsum_egy, kappa, kappa_egy, kappaeff, kappaeff_egy;;
  double dx, dy, dz, r, r2;
  MyFloat *pos;

  if(mode == 0)
    {
      pos = P[target].Pos;
      h = PPP[target].Hsml;
      kappa = SphP[target].CR_Kappa;
      kappa_egy = SphP[target].CR_Kappa_egy;
    }
  else
    {
      pos = DensDataGet[target].Pos;
      h = DensDataGet[target].Hsml;
      kappa = DensDataGet[target].CR_Kappa;
      kappa_egy = DensDataGet[target].CR_Kappa_egy;
    }


  weightsum = weightsum_egy = 0;

  startnode = All.MaxPart;

  kappaeff = 4 * kappa * (All.Time - All.TimeOfLastDiffusion);
  kappaeff_egy = 4 * kappa_egy * (All.Time - All.TimeOfLastDiffusion);

  if(kappaeff > 0)
    {
      do
	{
	  numngb_inbox = ngb_treefind_variable(&pos[0], h, &startnode);

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

	      if(r2 < h * h)
		{
		  r = sqrt(r2);

		  weightsum += P[j].Mass * exp(-r2 / kappaeff);
		  weightsum_egy += P[j].Mass * exp(-r2 / kappaeff_egy);
		}
	    }
	}
      while(startnode >= 0);
    }

  if(mode == 0)
    {
      SphP[target].CR_WeightSum = weightsum;
      SphP[target].CR_WeightSum_egy = weightsum_egy;
    }
  else
    {
      DensDataResult[target].CR_WeightSum = weightsum;
      DensDataResult[target].CR_WeightSum_egy = weightsum_egy;
    }

}



void scatter_diffusion(int target, int mode)
{
  int j, n;
  int startnode, numngb_inbox;
  double h, weight;
  double weightsum, kappa, kappaeff, CR_E0, CR_n0;
  double weightsum_egy, kappa_egy, kappaeff_egy;
  double dx, dy, dz, r, r2;
  MyFloat *pos;

  if(mode == 0)
    {
      pos = P[target].Pos;
      h = PPP[target].Hsml;
      kappa = SphP[target].CR_Kappa;
      kappa_egy = SphP[target].CR_Kappa_egy;
      weightsum = SphP[target].CR_WeightSum;
      weightsum_egy = SphP[target].CR_WeightSum_egy;
      CR_E0 = SphP[target].CR_E0;
      CR_n0 = SphP[target].CR_n0;
    }
  else
    {
      pos = DensDataGet[target].Pos;
      h = DensDataGet[target].Hsml;
      kappa = DensDataGet[target].CR_Kappa;
      kappa_egy = DensDataGet[target].CR_Kappa_egy;
      weightsum = DensDataGet[target].CR_WeightSum;
      weightsum_egy = DensDataGet[target].CR_WeightSum_egy;
      CR_E0 = DensDataGet[target].CR_E0;
      CR_n0 = DensDataGet[target].CR_n0;
    }


  startnode = All.MaxPart;

  kappaeff = 4 * kappa * (All.Time - All.TimeOfLastDiffusion);
  kappaeff_egy = 4 * kappa_egy * (All.Time - All.TimeOfLastDiffusion);

  if(kappaeff > 0)
    {
      do
	{
	  numngb_inbox = ngb_treefind_variable(&pos[0], h, &startnode);

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

	      if(r2 < h * h)
		{
		  r = sqrt(r2);

		  weight = P[j].Mass * exp(-r2 / kappaeff) / weightsum;
		  SphP[j].CR_DeltaN += CR_n0 * weight;

		  weight = P[j].Mass * exp(-r2 / kappaeff_egy) / weightsum_egy;
		  SphP[j].CR_DeltaE += CR_E0 * weight;

		}
	    }
	}
      while(startnode >= 0);
    }
}






#endif
