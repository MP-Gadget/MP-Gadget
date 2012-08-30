#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <gsl/gsl_math.h>

#include "allvars.h"
#include "proto.h"


#ifdef MHM


extern struct kindata_in
{
  MyFloat Pos[3];
  MyFloat Hsml;
  MyFloat Density;
  MyFloat Energy;
  int Index;
  int Task;
}
 *KinDataIn, *KinDataGet;


void kinetic_feedback_mhm(void)
{
  long long ntot, ntotleft;
  int ndone;
  int *noffset, *nbuffer, *nsend, *nsend_local, *numlist, *ndonelist;
  int i, j, n;
  int maxfill;
  int level, ngrp, sendTask, recvTask;
  int nexport;
  MPI_Status status;

  noffset = mymalloc("noffset", sizeof(int) * NTask);	/* offsets of bunches in common list */
  nbuffer = mymalloc("nbuffer", sizeof(int) * NTask);
  nsend_local = mymalloc("nsend_local", sizeof(int) * NTask);
  nsend = mymalloc("nsend", sizeof(int) * NTask * NTask);
  ndonelist = mymalloc("ndonelist", sizeof(int) * NTask);

  for(n = 0, NumSphUpdate = 0; n < N_gas; n++)
    {
      if(P[n].Type == 0)
	{
	  if(P[n].Ti_endstep == All.Ti_Current)
	    NumSphUpdate++;
	}
    }

  numlist = mymalloc("numlist", NTask * sizeof(int) * NTask);
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

      for(nexport = 0, ndone = 0; i < N_gas && nexport < All.BunchSizeKinetic - NTask; i++)
	if(P[i].Type == 0)
	  if(P[i].Ti_endstep == All.Ti_Current)
	    {
	      ndone++;

	      for(j = 0; j < NTask; j++)
		Exportflag[j] = 0;

	      kinetic_evaluate(i, 0);

	      for(j = 0; j < NTask; j++)
		{
		  if(Exportflag[j])
		    {
		      KinDataIn[nexport].Pos[0] = P[i].Pos[0];
		      KinDataIn[nexport].Pos[1] = P[i].Pos[1];
		      KinDataIn[nexport].Pos[2] = P[i].Pos[2];

		      KinDataIn[nexport].Density = SphP[i].Density;
		      KinDataIn[nexport].Energy = SphP[i].FeedbackEnergy;

		      KinDataIn[nexport].Hsml = PPP[i].Hsml;
		      KinDataIn[nexport].Index = i;
		      KinDataIn[nexport].Task = j;
		      nexport++;
		      nsend_local[j]++;
		    }
		}
	    }


      qsort(KinDataIn, nexport, sizeof(struct kindata_in), kin_compare_key);

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
		      MPI_Sendrecv(&KinDataIn[noffset[recvTask]],
				   nsend_local[recvTask] * sizeof(struct kindata_in), MPI_BYTE,
				   recvTask, TAG_DENS_A,
				   &KinDataGet[nbuffer[ThisTask]],
				   nsend[recvTask * NTask + ThisTask] * sizeof(struct kindata_in),
				   MPI_BYTE, recvTask, TAG_DENS_A, MPI_COMM_WORLD, &status);
		    }
		}

	      for(j = 0; j < NTask; j++)
		if((j ^ ngrp) < NTask)
		  nbuffer[j] += nsend[(j ^ ngrp) * NTask + j];
	    }

	  for(j = 0; j < nbuffer[ThisTask]; j++)
	    kinetic_evaluate(j, 1);

	  level = ngrp - 1;
	}

      MPI_Allgather(&ndone, 1, MPI_INT, ndonelist, 1, MPI_INT, MPI_COMM_WORLD);

      for(j = 0; j < NTask; j++)
	ntotleft -= ndonelist[j];
    }


  myfree(ndonelist);
  myfree(nsend);
  myfree(nsend_local);
  myfree(nbuffer);
  myfree(noffset);


}



/*! This function represents the core of the SPH density computation. The
 *  target particle may either be local, or reside in the communication
 *  buffer.
 */
void kinetic_evaluate(int target, int mode)
{
  int j, n;
  int startnode, numngb, numngb_inbox;
  double h, h2, hinv, hinv3;
  double rho, weight, wk, energy, delta_v;
  double dx, dy, dz, r, r2, u;
  double dvx, dvy, dvz;
  MyFloat *pos;

  if(mode == 0)
    {
      pos = P[target].Pos;
      h = PPP[target].Hsml;
      rho = SphP[target].Density;
      energy = SphP[target].FeedbackEnergy;
    }
  else
    {
      pos = KinDataGet[target].Pos;
      h = KinDataGet[target].Hsml;
      rho = KinDataGet[target].Density;
      energy = KinDataGet[target].Energy;
    }


  h2 = h * h;
  hinv = 1.0 / h;
#ifndef  TWODIMS
  hinv3 = hinv * hinv * hinv;
#else
  hinv3 = hinv * hinv / boxSize_Z;
#endif


  startnode = All.MaxPart;
  numngb = 0;
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

	  if(r2 < h2)
	    {
	      r = sqrt(r2);

	      if(r > 0)
		{

		  u = r * hinv;

		  if(u < 0.5)
		    wk = hinv3 * (KERNEL_COEFF_1 + KERNEL_COEFF_2 * (u - 1) * u * u);
		  else
		    wk = hinv3 * KERNEL_COEFF_5 * (1.0 - u) * (1.0 - u) * (1.0 - u);

		  weight = wk * P[j].Mass / rho;

		  delta_v = sqrt(2 * energy * weight / P[j].Mass);

		  dvx = delta_v * (-dx) / r;
		  dvy = delta_v * (-dy) / r;
		  dvz = delta_v * (-dz) / r;

		  P[j].Vel[0] += dvx;
		  P[j].Vel[1] += dvy;
		  P[j].Vel[2] += dvz;

		  SphP[j].VelPred[0] += dvx;
		  SphP[j].VelPred[1] += dvy;
		  SphP[j].VelPred[2] += dvz;
		}
	    }
	}
    }
  while(startnode >= 0);
}





/*! This routine is a comparison kernel used in a sort routine to group
 *  particles that are exported to the same processor.
 */
int kin_compare_key(const void *a, const void *b)
{
  if(((struct kindata_in *) a)->Task < (((struct kindata_in *) b)->Task))
    return -1;

  if(((struct kindata_in *) a)->Task > (((struct kindata_in *) b)->Task))
    return +1;

  return 0;
}


#endif
