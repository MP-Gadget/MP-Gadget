#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <gsl/gsl_math.h>

#ifdef RADTRANSFER
#ifdef SFR

#include "allvars.h"
#include "proto.h"


/*! Structure for communication during the density computation. Holds data that is sent to other processors.
 */
static struct densdata_in
{
  MyDouble Pos[3];
  MyFloat HsmlSfr;
  int NodeList[NODELISTLENGTH];
}
 *DensDataIn, *DensDataGet;


static struct densdata_out
{
  MyDouble DensitySfr;
  MyDouble DhsmlDensityFactorSfr;
  MyDouble NgbSfr;
}
 *DensDataResult, *DensDataOut;

void density_sfr(void)
{
  MyFloat *Left, *Right;
  int i, j, ndone, ndone_flag, npleft, dummy, iter = 0;
  int ngrp, sendTask, recvTask, place, nexport, nimport;
  long long ntot;
  double dmax1, dmax2, fac;
  double desnumngb;

  CPU_Step[CPU_DENSMISC] += measure_time();

  Left = (MyFloat *) mymalloc("Left", N_gas * sizeof(MyFloat));
  Right = (MyFloat *) mymalloc("Right", N_gas * sizeof(MyFloat));

  for(i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i])
    {
      if(P[i].Type == 0)
	if(SphP[i].Sfr > 0)
	  {
	    Left[i] = Right[i] = 0;
	    SphP[i].HsmlSfr = PPP[i].Hsml;
	  }
    }

  /* allocate buffers to arrange communication */


  Ngblist = (int *) mymalloc("Ngblist", N_gas * sizeof(int));

  All.BunchSize =
    (int) ((All.BufferSize * 1024 * 1024) / (sizeof(struct data_index) + sizeof(struct data_nodelist) +
					     sizeof(struct densdata_in) + sizeof(struct densdata_out) +
					     sizemax(sizeof(struct densdata_in),
						     sizeof(struct densdata_out))));
  DataIndexTable =
    (struct data_index *) mymalloc("DataIndexTable", All.BunchSize * sizeof(struct data_index));
  DataNodeList =
    (struct data_nodelist *) mymalloc("DataNodeList", All.BunchSize * sizeof(struct data_nodelist));

  desnumngb = All.DesNumNgb;

  /* we will repeat the whole thing for those particles where we didn't find enough neighbours */
  do
    {
      i = FirstActiveParticle;	/* begin with this index */

      do
	{
	  for(j = 0; j < NTask; j++)
	    {
	      Send_count[j] = 0;
	      Exportflag[j] = -1;
	    }

	  /* do local particles and prepare export list */
	  for(nexport = 0; i >= 0; i = NextActiveParticle[i])
	    {
	      if(P[i].Type == 0)
		if(SphP[i].Sfr > 0)
		  {
		    if(density_sfr_evaluate(i, 0, &nexport, Send_count) < 0)
		      break;
		  }
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

	  DensDataGet =
	    (struct densdata_in *) mymalloc("	  DensDataGet", nimport * sizeof(struct densdata_in));
	  DensDataIn =
	    (struct densdata_in *) mymalloc("	  DensDataIn", nexport * sizeof(struct densdata_in));

	  /* prepare particle data for export */
	  for(j = 0; j < nexport; j++)
	    {
	      place = DataIndexTable[j].Index;

	      DensDataIn[j].Pos[0] = P[place].Pos[0];
	      DensDataIn[j].Pos[1] = P[place].Pos[1];
	      DensDataIn[j].Pos[2] = P[place].Pos[2];
	      DensDataIn[j].HsmlSfr = SphP[place].HsmlSfr;

	      memcpy(DensDataIn[j].NodeList,
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
		      MPI_Sendrecv(&DensDataIn[Send_offset[recvTask]],
				   Send_count[recvTask] * sizeof(struct densdata_in), MPI_BYTE,
				   recvTask, TAG_DENS_A,
				   &DensDataGet[Recv_offset[recvTask]],
				   Recv_count[recvTask] * sizeof(struct densdata_in), MPI_BYTE,
				   recvTask, TAG_DENS_A, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		    }
		}
	    }

	  myfree(DensDataIn);
	  DensDataResult =
	    (struct densdata_out *) mymalloc("	  DensDataResult", nimport * sizeof(struct densdata_out));
	  DensDataOut =
	    (struct densdata_out *) mymalloc("	  DensDataOut", nexport * sizeof(struct densdata_out));


	  /* now do the particles that were sent to us */

	  for(j = 0; j < nimport; j++)
	    density_sfr_evaluate(j, 1, &dummy, &dummy);

	  if(i < 0)
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
		      MPI_Sendrecv(&DensDataResult[Recv_offset[recvTask]],
				   Recv_count[recvTask] * sizeof(struct densdata_out),
				   MPI_BYTE, recvTask, TAG_DENS_B,
				   &DensDataOut[Send_offset[recvTask]],
				   Send_count[recvTask] * sizeof(struct densdata_out),
				   MPI_BYTE, recvTask, TAG_DENS_B, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		    }
		}

	    }

	  /* add the result to the local particles */
	  for(j = 0; j < nexport; j++)
	    {
	      place = DataIndexTable[j].Index;

	      SphP[place].NgbSfr += DensDataOut[j].NgbSfr;
	      SphP[place].DhsmlDensityFactorSfr += DensDataOut[j].DhsmlDensityFactorSfr;
	      SphP[place].DensitySfr += DensDataOut[j].DensitySfr;
	    }

	  myfree(DensDataOut);
	  myfree(DensDataResult);
	  myfree(DensDataGet);

	}
      while(ndone < NTask);

      /* do final operations on results */
      for(i = FirstActiveParticle, npleft = 0; i >= 0; i = NextActiveParticle[i])
	{
	  if(P[i].Type == 0)
	    if(SphP[i].Sfr > 0)
	      {
		SphP[i].DhsmlDensityFactorSfr *= SphP[i].HsmlSfr / (3.0 * SphP[i].DensitySfr);

		if(SphP[i].DhsmlDensityFactorSfr > -0.9)
		  SphP[i].DhsmlDensityFactorSfr = 1 / (1 + SphP[i].DhsmlDensityFactorSfr);
		else
		  SphP[i].DhsmlDensityFactorSfr = 1;

		/* now check whether we had enough neighbours */
		desnumngb = 32;

		if(SphP[i].NgbSfr < (desnumngb - All.MaxNumNgbDeviation) ||
		   (SphP[i].NgbSfr > (desnumngb + All.MaxNumNgbDeviation)
		    && SphP[i].HsmlSfr > (1.01 * All.MinGasHsml)))
		  {
		    /* need to redo this particle */
		    npleft++;

		    if(Left[i] > 0 && Right[i] > 0)
		      if((Right[i] - Left[i]) < 1.0e-3 * Left[i])
			{
			  /* this one should be ok */
			  npleft--;
			  continue;
			}

		    if(SphP[i].NgbSfr < (desnumngb - All.MaxNumNgbDeviation))
		      Left[i] = DMAX(SphP[i].HsmlSfr, Left[i]);
		    else
		      {
			if(Right[i] != 0)
			  {
			    if(SphP[i].HsmlSfr < Right[i])
			      Right[i] = SphP[i].HsmlSfr;
			  }
			else
			  Right[i] = SphP[i].HsmlSfr;
		      }

		    if(iter >= MAXITER - 10)
		      {
			printf
			  ("i=%d task=%d ID=%d Hsml=%g Left=%g Right=%g Ngbs=%g Right-Left=%g\n   pos=(%g|%g|%g)\n",
			   i, ThisTask, (int) P[i].ID, SphP[i].HsmlSfr, Left[i], Right[i],
			   (float) SphP[i].NgbSfr, Right[i] - Left[i], P[i].Pos[0], P[i].Pos[1], P[i].Pos[2]);
			fflush(stdout);
		      }

		    if(Right[i] > 0 && Left[i] > 0)
		      SphP[i].HsmlSfr = pow(0.5 * (pow(Left[i], 3) + pow(Right[i], 3)), 1.0 / 3);
		    else
		      {
			if(Right[i] == 0 && Left[i] == 0)
			  endrun(8188);	/* can't occur */

			if(Right[i] == 0 && Left[i] > 0)
			  {
			    if(P[i].Type == 0 && fabs(SphP[i].NgbSfr - desnumngb) < 0.5 * desnumngb)
			      {
				fac = 1 - (SphP[i].NgbSfr -
					   desnumngb) / (3.0 * SphP[i].NgbSfr) *
				  SphP[i].DhsmlDensityFactorSfr;

				if(fac < 1.26)
				  SphP[i].HsmlSfr *= fac;
				else
				  SphP[i].HsmlSfr *= 1.26;
			      }
			    else
			      SphP[i].HsmlSfr *= 1.26;
			  }

			if(Right[i] > 0 && Left[i] == 0)
			  {
			    if(P[i].Type == 0 && fabs(SphP[i].NgbSfr - desnumngb) < 0.5 * desnumngb)
			      {
				fac = 1 - (SphP[i].NgbSfr -
					   desnumngb) / (3.0 * SphP[i].NgbSfr) *
				  SphP[i].DhsmlDensityFactorSfr;

				if(fac > 1 / 1.26)
				  SphP[i].HsmlSfr *= fac;
				else
				  SphP[i].HsmlSfr /= 1.26;
			      }
			    else
			      SphP[i].HsmlSfr /= 1.26;
			  }
		      }

		    if(SphP[i].HsmlSfr < All.MinGasHsml)
		      SphP[i].HsmlSfr = All.MinGasHsml;


		  }
	      }
	}

      sumup_large_ints(1, &npleft, &ntot);

      if(ntot > 0)
	{
	  iter++;

	  if(iter > 0 && ThisTask == 0)
	    {
	      printf("ngb iteration %d: need to repeat for %d%09d particles.\n", iter,
		     (int) (ntot / 1000000000), (int) (ntot % 1000000000));
	      fflush(stdout);
	    }

	  if(iter > MAXITER)
	    {
	      printf("failed to converge in neighbour iteration in density()\n");
	      fflush(stdout);
	      endrun(1155);
	    }
	}
    }
  while(ntot > 0);


  myfree(DataNodeList);
  myfree(DataIndexTable);
  myfree(Ngblist);
  myfree(Right);
  myfree(Left);
}


/*! This function represents the core of the SPH density computation. The
 *  target particle may either be local, or reside in the communication
 *  buffer.
 */
int density_sfr_evaluate(int target, int mode, int *nexport, int *nsend_local)
{
  int j, n;
  int startnode, numngb, numngb_inbox, listindex = 0;
  double h, h2, hinv, hinv3, hinv4;
  MyLongDouble rho;
  double wk, dwk;
  double dx, dy, dz, r, r2, u, mass_j;
  MyLongDouble weighted_numngb;
  MyLongDouble dhsmlrho;
  MyDouble *pos;

  rho = weighted_numngb = dhsmlrho = 0;

  if(mode == 0)
    {
      pos = P[target].Pos;
      h = SphP[target].HsmlSfr;
    }
  else
    {
      pos = DensDataGet[target].Pos;
      h = DensDataGet[target].HsmlSfr;
    }


  h2 = h * h;
  hinv = 1.0 / h;
  hinv3 = hinv * hinv * hinv;
  hinv4 = hinv3 * hinv;

  if(mode == 0)
    {
      startnode = All.MaxPart;	/* root node */
    }
  else
    {
      startnode = DensDataGet[target].NodeList[0];
      startnode = Nodes[startnode].u.d.nextnode;	/* open it */
    }

  numngb = 0;

  while(startnode >= 0)
    {
      while(startnode >= 0)
	{
	  numngb_inbox = ngb_treefind_stars(pos, h, target, &startnode, mode, nexport, nsend_local);

	  if(numngb_inbox < 0)
	    return -1;

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
		  numngb++;

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

		  rho += FLT(mass_j * wk);

		  weighted_numngb += FLT(NORM_COEFF * wk / hinv3);

		  dhsmlrho += FLT(-mass_j * (NUMDIMS * hinv * wk + u * dwk));

		}
	    }
	}

      if(mode == 1)
	{
	  listindex++;
	  if(listindex < NODELISTLENGTH)
	    {
	      startnode = DensDataGet[target].NodeList[listindex];
	      if(startnode >= 0)
		startnode = Nodes[startnode].u.d.nextnode;	/* open it */
	    }
	}
    }

  if(mode == 0)
    {
      SphP[target].NgbSfr = weighted_numngb;
      SphP[target].DensitySfr = rho;
      SphP[target].DhsmlDensityFactorSfr = dhsmlrho;
    }
  else
    {
      DensDataResult[target].DensitySfr = rho;
      DensDataResult[target].NgbSfr = weighted_numngb;
      DensDataResult[target].DhsmlDensityFactorSfr = dhsmlrho;

    }

  return 0;
}

static struct stardata_in
{
  MyDouble Pos[3], Mass, Sfr;
  MyFloat HsmlSfr;
  int NodeList[NODELISTLENGTH];
}
 *StarDataIn, *StarDataGet;

void sfr_lum(void)
{
  int j;
  int i, dummy;
  int ngrp, sendTask, recvTask, place, nexport, nimport, ndone, ndone_flag;

  /* clear Je in all gas particles */
  for(j = 0; j < N_gas; j++)
    if(P[j].Type == 0)
      for(i = 0; i < N_BINS; i++)
	SphP[j].Je[i] = 0;

  /* allocate buffers to arrange communication */

  Ngblist = (int *) mymalloc("Ngblist", NumPart * sizeof(int));

  All.BunchSize =
    (int) ((All.BufferSize * 1024 * 1024) / (sizeof(struct data_index) + sizeof(struct data_nodelist) +
					     2 * sizeof(struct stardata_in)));
  DataIndexTable =
    (struct data_index *) mymalloc("DataIndexTable", All.BunchSize * sizeof(struct data_index));
  DataNodeList =
    (struct data_nodelist *) mymalloc("DataNodeList", All.BunchSize * sizeof(struct data_nodelist));

  i = FirstActiveParticle;	/* beginn with this index */

  do
    {
      for(j = 0; j < NTask; j++)
	{
	  Send_count[j] = 0;
	  Exportflag[j] = -1;
	}

      /* do local particles and prepare export list */
      for(nexport = 0; i >= 0; i = NextActiveParticle[i])
	{
	  if(P[i].Type == 0)
	    if(SphP[i].Sfr > 0)
	      {
		if(sfr_lum_evaluate(i, 0, &nexport, Send_count) < 0)
		  break;
	      }
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

      StarDataGet = (struct stardata_in *) mymalloc("StarDataGet", nimport * sizeof(struct stardata_in));
      StarDataIn = (struct stardata_in *) mymalloc("StarDataIn", nexport * sizeof(struct stardata_in));

      /* prepare particle data for export */
      for(j = 0; j < nexport; j++)
	{
	  place = DataIndexTable[j].Index;

	  StarDataIn[j].Pos[0] = P[place].Pos[0];
	  StarDataIn[j].Pos[1] = P[place].Pos[1];
	  StarDataIn[j].Pos[2] = P[place].Pos[2];
	  StarDataIn[j].HsmlSfr = SphP[place].HsmlSfr;
	  StarDataIn[j].Mass = P[place].Mass;
	  StarDataIn[j].Sfr = SphP[place].Sfr;

	  memcpy(StarDataIn[j].NodeList,
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
		  MPI_Sendrecv(&StarDataIn[Send_offset[recvTask]],
			       Send_count[recvTask] * sizeof(struct stardata_in), MPI_BYTE,
			       recvTask, TAG_DENS_A,
			       &StarDataGet[Recv_offset[recvTask]],
			       Recv_count[recvTask] * sizeof(struct stardata_in), MPI_BYTE,
			       recvTask, TAG_DENS_A, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}
	    }
	}

      myfree(StarDataIn);


      /* now do the particles that were sent to us */

      for(j = 0; j < nimport; j++)
	sfr_lum_evaluate(j, 1, &dummy, &dummy);

      /* check whether this is the last iteration */
      if(i < 0)
	ndone_flag = 1;
      else
	ndone_flag = 0;

      MPI_Allreduce(&ndone_flag, &ndone, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

      myfree(StarDataGet);
    }
  while(ndone < NTask);

  myfree(DataNodeList);
  myfree(DataIndexTable);
  myfree(Ngblist);
}

int sfr_lum_evaluate(int target, int mode, int *nexport, int *nsend_local)
{
  int i, j, n, numngb;
  int startnode, listindex = 0;
  double h, hinv, h2, mass_j, hinv3;
  double wk, mass, density, sfr, lum;
  double dx, dy, dz, r, r2, u, a3inv;
  MyDouble *pos;

#ifdef PERIODIC
  double boxsize, boxhalf;

  boxsize = All.BoxSize;
  boxhalf = 0.5 * All.BoxSize;
#endif

  if(All.ComovingIntegrationOn)
    a3inv = 1.0 / (All.Time * All.Time * All.Time);
  else
    a3inv = 1.0;

  if(mode == 0)
    {
      pos = P[target].Pos;
      h = SphP[target].HsmlSfr;
      mass = P[target].Mass;
      sfr = SphP[target].Sfr;
    }
  else
    {
      pos = StarDataGet[target].Pos;
      h = StarDataGet[target].HsmlSfr;
      mass = StarDataGet[target].Mass;
      sfr = StarDataGet[target].Sfr;
    }

  lum = sfr * All.IonizingLumPerSFR * All.UnitTime_in_s / All.HubbleParam;

  h2 = h * h;
  hinv = 1.0 / h;
  hinv3 = hinv * hinv * hinv;

  if(mode == 0)
    {
      startnode = All.MaxPart;	/* root node */
    }
  else
    {
      startnode = StarDataGet[target].NodeList[0];
      startnode = Nodes[startnode].u.d.nextnode;	/* open it */
    }

  while(startnode >= 0)
    {
      while(startnode >= 0)
	{
	  numngb = ngb_treefind_stars(pos, h, target, &startnode, mode, nexport, nsend_local);

	  if(numngb < 0)
	    return -1;

	  for(n = 0; n < numngb; n++)
	    {
	      j = Ngblist[n];

	      dx = pos[0] - P[j].Pos[0];
	      dy = pos[1] - P[j].Pos[1];
	      dz = pos[2] - P[j].Pos[2];
#ifdef PERIODIC
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

	      if(r2 < h2)
		{
		  u = r * hinv;

		  if(u < 0.5)
		    wk = hinv3 * (KERNEL_COEFF_1 + KERNEL_COEFF_2 * (u - 1) * u * u);
		  else
		    wk = hinv3 * KERNEL_COEFF_5 * (1.0 - u) * (1.0 - u) * (1.0 - u);

		}
	      else
		wk = 0;

	      mass_j = P[j].Mass;
	      density = SphP[j].d.Density;

	      for(i = 0; i < N_BINS; i++)
		SphP[j].Je[i] += lum * wk;
	    }
	}
      
      if(mode == 1)
	{
	  listindex++;
	  if(listindex < NODELISTLENGTH)
	    {
	      startnode = StarDataGet[target].NodeList[listindex];
	      if(startnode >= 0)
		startnode = Nodes[startnode].u.d.nextnode;	/* open it */
	    }
	}
    }

  return 0;
}

#endif
#endif
