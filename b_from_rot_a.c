#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <gsl/gsl_math.h>

#include "allvars.h"
#include "proto.h"

/*! \file b_from_rot_a.c 
 *  \brief calculates ror(b) to aloow an initial vector potential
 *
 *  This file is only called once after reading the initial condition.
 *  It is used to calculate b_ini from a initial vector potential, if needed.
 */

#ifdef BFROMROTA

void rot_a(void)
{
  long long ntot, ntotleft;
  int ndone;
  int *noffset, *nbuffer, *nsend, *nsend_local, *numlist, *ndonelist;
  int i, j, n;
  int npleft;
  int maxfill, source;
  int level, ngrp, sendTask, recvTask;
  int place, nexport;
  double dmax1, dmax2, fac;
  double sumt, sumcomm;

  MPI_Status status;

  noffset = (int *) mymalloc("noffset", sizeof(int) * NTask);	/* offsets of bunches in common list */
  nbuffer = (int *) mymalloc("nbuffer", sizeof(int) * NTask);
  nsend_local = (int *) mymalloc("nsend_local", sizeof(int) * NTask);
  nsend = (int *) mymalloc("nsend", sizeof(int) * NTask * NTask);
  ndonelist = (int *) mymalloc("ndonelist", sizeof(int) * NTask);

  NumSphUpdate = N_gas;

  numlist = (int *) mymalloc("numlist", NTask * sizeof(int) * NTask);
  MPI_Allgather(&NumSphUpdate, 1, MPI_INT, numlist, 1, MPI_INT, MPI_COMM_WORLD);
  for(i = 0, ntot = 0; i < NTask; i++)
    ntot += numlist[i];
  myfree(numlist);


  i = 0;			/* beginn with this index */
  ntotleft = ntot;		/* particles left for all tasks together */

  while(ntotleft > 0)
    {
      for(j = 0; j < NTask; j++)
	nsend_local[j] = 0;

      /* do local particles and prepare export list */
      for(nexport = 0, ndone = 0; i < N_gas && nexport < All.BunchSizeDensity - NTask; i++)
	{
	  ndone++;

	  for(j = 0; j < NTask; j++)
	    Exportflag[j] = 0;

	  rot_a_evaluate(i, 0);

	  for(j = 0; j < NTask; j++)
	    {
	      if(Exportflag[j])
		{
		  DensDataIn[nexport].Pos[0] = P[i].Pos[0];
		  DensDataIn[nexport].Pos[1] = P[i].Pos[1];
		  DensDataIn[nexport].Pos[2] = P[i].Pos[2];
		  /* using velocity structure for BPred ... */
		  DensDataIn[nexport].Vel[0] = SphP[i].BPred[0];
		  DensDataIn[nexport].Vel[1] = SphP[i].BPred[1];
		  DensDataIn[nexport].Vel[2] = SphP[i].BPred[2];

		  DensDataIn[nexport].Hsml = PPP[i].Hsml;
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
				   recvTask, TAG_DENS_A,
				   &DensDataGet[nbuffer[ThisTask]],
				   nsend[recvTask * NTask + ThisTask] * sizeof(struct densdata_in),
				   MPI_BYTE, recvTask, TAG_DENS_A, MPI_COMM_WORLD, &status);
		    }
		}

	      for(j = 0; j < NTask; j++)
		if((j ^ ngrp) < NTask)
		  nbuffer[j] += nsend[(j ^ ngrp) * NTask + j];
	    }

	  for(j = 0; j < nbuffer[ThisTask]; j++)
	    {
	      rot_a_evaluate(j, 1);
	    }

	  MPI_Barrier(MPI_COMM_WORLD);

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
				   MPI_BYTE, recvTask, TAG_DENS_B,
				   &DensDataPartialResult[noffset[recvTask]],
				   nsend_local[recvTask] * sizeof(struct densdata_out),
				   MPI_BYTE, recvTask, TAG_DENS_B, MPI_COMM_WORLD, &status);

		      /* add the result to the particles */
		      for(j = 0; j < nsend_local[recvTask]; j++)
			{
			  source = j + noffset[recvTask];
			  place = DensDataIn[source].Index;

			  SphP[place].BSmooth[0] += DensDataPartialResult[source].BSmooth[0];
			  SphP[place].BSmooth[1] += DensDataPartialResult[source].BSmooth[1];
			  SphP[place].BSmooth[2] += DensDataPartialResult[source].BSmooth[2];
			  SphP[place].DensityNorm += DensDataPartialResult[source].DensityNorm;
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

  for(i = 0; i < N_gas; i++)
    for(j = 0; j < 3; j++)
#ifndef IGNORE_PERIODIC_IN_ROTA
      SphP[i].B[j] = SphP[i].BPred[j] = SphP[i].BSmooth[j] / SphP[i].d.Density;
#else
      SphP[i].B[j] = SphP[i].BPred[j] = SphP[i].BSmooth[j] / SphP[i].DensityNorm;
#endif
  myfree(ndonelist);
  myfree(nsend);
  myfree(nsend_local);
  myfree(nbuffer);
  myfree(noffset);

}



/*! This function represents the core of the calculus of rot of vector potential
 *  target particle may either be local, or reside in the communication
 *  buffer.
 */
void rot_a_evaluate(int target, int mode)
{
  int j, n;
  int startnode, numngb, numngb_inbox;
  double h, h2, fac, hinv, hinv3, hinv4;

  double wk, dwk;
  double dx, dy, dz, r, r2, u, mass_j, rho;
  double dbx, dby, dbz;

  double rotb[3];

  MyFloat *pos, *b;

  rho = rotb[0] = rotb[1] = rotb[2] = 0;


  if(mode == 0)
    {
      pos = P[target].Pos;
      b = SphP[target].BPred;
      h = PPP[target].Hsml;
    }
  else
    {
      pos = DensDataGet[target].Pos;
      b = DensDataGet[target].Vel;	/* Using Vel structure for BPred !!!! */
      h = DensDataGet[target].Hsml;
    }


  h2 = h * h;
  hinv = 1.0 / h;
#ifndef  TWODIMS
  hinv3 = hinv * hinv * hinv;
#else
  hinv3 = hinv * hinv / boxSize_Z;
#endif
  hinv4 = hinv3 * hinv;

  startnode = All.MaxPart;
  do
    {
      numngb_inbox = ngb_treefind_variable(&pos[0], 2 * h, &startnode);

      for(n = 0; n < numngb_inbox; n++)
	{
	  j = Ngblist[n];

	  dx = pos[0] - P[j].Pos[0];
	  dy = pos[1] - P[j].Pos[1];
	  dz = pos[2] - P[j].Pos[2];

	  dbx = b[0] - SphP[j].BPred[0];
	  dby = b[1] - SphP[j].BPred[1];
	  dbz = b[2] - SphP[j].BPred[2];

#ifndef IGNORE_PERIODIC_IN_ROTA
#ifdef PERIODIC			/*  now find the closest image in the given box size  */
	  if(dx > boxHalf_X)
	    dx -= boxSize_X;
	  if(dx < -boxHalf_X)
	    dx += boxSize_X;
	  if(dy > boxHalf_Y)
	    {
#ifdef BRIOWU
	      dbz /= dy;
#endif
	      dy -= boxSize_Y;
#ifdef BRIOWU
	      dbz *= dy;
#endif
	    }
	  if(dy < -boxHalf_Y)
	    {
#ifdef BRIOWU
	      dbz /= dy;
#endif
	      dy += boxSize_Y;
#ifdef BRIOWU
	      dbz *= dy;
#endif
	    }
	  if(dz > boxHalf_Z)
	    {
#ifdef BRIOWU
	      dbx /= dz;
#endif
	      dz -= boxSize_Z;
#ifdef BRIOWU
	      dbx *= dz;
#endif
	    }
	  if(dz < -boxHalf_Z)
	    {
#ifdef BRIOWU
	      dbx /= dz;
#endif
	      dz += boxSize_Z;
#ifdef BRIOWU
	      dbx *= dz;
#endif
	    }
#endif
#endif
	  r2 = dx * dx + dy * dy + dz * dz;

	  if(r2 < h2)
	    {
	      r = sqrt(r2);
	      u = r * hinv;
	      if(u < 0.5)
		{
		  wk = hinv3 * (KERNEL_COEFF_1 + KERNEL_COEFF_2 * (u - 1) * u * u);
		  dwk = hinv4 * u * (KERNEL_COEFF_3 * u - KERNEL_COEFF_4);
		}
	      else
		{
		  wk = hinv3 * KERNEL_COEFF_5 * (1.0 - u) * (1.0 - u) * (1.0 - u);
		  dwk = hinv4 * KERNEL_COEFF_6 * (1.0 - u) * (1.0 - u);
		}

	      mass_j = P[j].Mass;

	      rho += mass_j * wk;

	      if(r > 0)
		{
		  fac = mass_j * dwk / r;
#ifndef BRIOWU
		  rotb[0] += FLT(fac * (dz * dby - dy * dbz));
		  rotb[1] += FLT(fac * (dx * dbz - dz * dbx));
		  rotb[2] += FLT(fac * (dy * dbx - dx * dby));
#else
		  rotb[0] += FLT(fac * (-dy * dbz));
		  rotb[1] += FLT(fac * (-dz * dbx));
		  rotb[2] += FLT(fac * (0.0));
#endif
		}
	    }
	}

    }
  while(startnode >= 0);

  if(mode == 0)
    {
      SphP[target].BSmooth[0] = rotb[0];	/* giving back rot(B) in the Smooth */
      SphP[target].BSmooth[1] = rotb[1];	/* Data structure !!! */
      SphP[target].BSmooth[2] = rotb[2];
      SphP[target].DensityNorm = rho;
    }
  else
    {
      DensDataResult[target].BSmooth[0] = rotb[0];
      DensDataResult[target].BSmooth[1] = rotb[1];
      DensDataResult[target].BSmooth[2] = rotb[2];
      DensDataResult[target].DensityNorm = rho;
    }
}

#endif
