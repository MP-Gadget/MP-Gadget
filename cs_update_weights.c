#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <gsl/gsl_math.h>

#include "allvars.h"
#include "proto.h"

#ifdef CS_MODEL
#include "cs_metals.h"


/*! Structure for communication during the density computation. Holds data that is sent to other processors.
 */
static struct updateweight_in
{
  MyDouble Pos[3];
  MyFloat Hsml;
  int NodeList[NODELISTLENGTH];
}
 *UpdateweightIn, *UpdateweightGet;


static struct updateweight_out
{
  MyLongDouble Ngb;
}
 *UpdateweightResult, *UpdateweightOut;


/*! \file density.c
 *  \brief SPH density computation and smoothing length determination
 *
 *  This file contains the "first SPH loop", where the SPH densities and some
 *  auxiliary quantities are computed.  There is also functionality that
 *  corrects the smoothing length if needed.
 */


/* This function updates the weights for SN before exploding. This is
 * necessary due to the fact that gas particles neighbours of a given star
 * could have been transformed into stars and they need to be taken off the
 * neighbour list for the exploding star.
 */
void cs_update_weights(void)
{
  MyFloat *Left, *Right;
  int i, j, ndone, ndone_flag, npleft, dummy, iter = 0;
  int ngrp, sendTask, recvTask, place, nexport, nimport;
  long long ntot;
  double dmax1, dmax2;
  double desnumngb;

  if(ThisTask == 0)
    {
      printf("... start update weights phase = %d ...\n", Flag_phase);
      fflush(stdout);
    }

  Left = (MyFloat *) mymalloc("Left", NumPart * sizeof(MyFloat));
  Right = (MyFloat *) mymalloc("Right", NumPart * sizeof(MyFloat));

  for(i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i])
    {
      if(P[i].Type == 6 || P[i].Type == 7)
	{
	  Left[i] = Right[i] = 0;
	}
    }

  /* allocate buffers to arrange communication */
  Ngblist = (int *) mymalloc("Ngblist", NumPart * sizeof(int));
  R2ngblist = (double *) mymalloc("R2ngblist", NumPart * sizeof(double));


  All.BunchSize =
    (int) ((All.BufferSize * 1024 * 1024) / (sizeof(struct data_index) + sizeof(struct data_nodelist) +
					     sizeof(struct updateweight_in) +
					     sizeof(struct updateweight_out) +
					     sizemax(sizeof(struct updateweight_in),
						     sizeof(struct updateweight_out))));
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
	      if((P[i].Type == 6 || P[i].Type == 7) && P[i].TimeBin >= 0)
		{
		  if(cs_update_weight_evaluate(i, 0, &nexport, Send_count) < 0)
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

	  UpdateweightGet =
	    (struct updateweight_in *) mymalloc("	  UpdateweightGet",
						nimport * sizeof(struct updateweight_in));
	  UpdateweightIn =
	    (struct updateweight_in *) mymalloc("	  UpdateweightIn",
						nexport * sizeof(struct updateweight_in));

	  /* prepare particle data for export */
	  for(j = 0; j < nexport; j++)
	    {
	      place = DataIndexTable[j].Index;

	      UpdateweightIn[j].Pos[0] = P[place].Pos[0];
	      UpdateweightIn[j].Pos[1] = P[place].Pos[1];
	      UpdateweightIn[j].Pos[2] = P[place].Pos[2];
	      UpdateweightIn[j].Hsml = PPP[place].Hsml;

	      memcpy(UpdateweightIn[j].NodeList,
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
		      MPI_Sendrecv(&UpdateweightIn[Send_offset[recvTask]],
				   Send_count[recvTask] * sizeof(struct updateweight_in), MPI_BYTE,
				   recvTask, TAG_DENS_A,
				   &UpdateweightGet[Recv_offset[recvTask]],
				   Recv_count[recvTask] * sizeof(struct updateweight_in), MPI_BYTE,
				   recvTask, TAG_DENS_A, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		    }
		}
	    }


	  myfree(UpdateweightIn);
	  UpdateweightResult =
	    (struct updateweight_out *) mymalloc("	  UpdateweightResult",
						 nimport * sizeof(struct updateweight_out));
	  UpdateweightOut =
	    (struct updateweight_out *) mymalloc("	  UpdateweightOut",
						 nexport * sizeof(struct updateweight_out));


	  /* now do the particles that were sent to us */

	  for(j = 0; j < nimport; j++)
	    cs_update_weight_evaluate(j, 1, &dummy, &dummy);

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
		      MPI_Sendrecv(&UpdateweightResult[Recv_offset[recvTask]],
				   Recv_count[recvTask] * sizeof(struct updateweight_out),
				   MPI_BYTE, recvTask, TAG_DENS_B,
				   &UpdateweightOut[Send_offset[recvTask]],
				   Send_count[recvTask] * sizeof(struct updateweight_out),
				   MPI_BYTE, recvTask, TAG_DENS_B, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		    }
		}

	    }


	  /* add the result to the local particles */
	  for(j = 0; j < nexport; j++)
	    {
	      place = DataIndexTable[j].Index;

	      PPP[place].n.dNumNgb += UpdateweightOut[j].Ngb;
	    }

	  myfree(UpdateweightOut);
	  myfree(UpdateweightResult);
	  myfree(UpdateweightGet);
	}
      while(ndone < NTask);


      /* do final operations on results */
      for(i = FirstActiveParticle, npleft = 0; i >= 0; i = NextActiveParticle[i])
	{
	  if(P[i].Type == 6 || P[i].Type == 7)
	    {
#ifdef FLTROUNDOFFREDUCTION
	      PPP[i].n.NumNgb = FLT(PPP[i].n.dNumNgb);
#endif

	      /* now check whether we had enough neighbours */

	      if(PPP[i].n.NumNgb < (desnumngb - All.MaxNumNgbDeviation) ||
		 (PPP[i].n.NumNgb > (desnumngb + All.MaxNumNgbDeviation)
		  && PPP[i].Hsml > (1.01 * All.MinGasHsml)))
		{
		  /* need to redo this particle */
		  npleft++;

		  if(Left[i] > 0 && Right[i] > 0)
		    if((Right[i] - Left[i]) < 1.0e-3 * Left[i])
		      {
			/* this one should be ok */
			npleft--;
			P[i].TimeBin = -P[i].TimeBin - 1;	/* Mark as inactive */
			continue;
		      }

		  if(PPP[i].n.NumNgb < (desnumngb - All.MaxNumNgbDeviation))
		    Left[i] = DMAX(PPP[i].Hsml, Left[i]);
		  else
		    {
		      if(Right[i] != 0)
			{
			  if(PPP[i].Hsml < Right[i])
			    Right[i] = PPP[i].Hsml;
			}
		      else
			Right[i] = PPP[i].Hsml;
		    }

		  if(iter >= MAXITER - 10)
		    {
		      printf
			("i=%d task=%d ID=%d Hsml=%g Left=%g Right=%g Ngbs=%g Right-Left=%g\n   pos=(%g|%g|%g)\n",
			 i, ThisTask, (int) P[i].ID, PPP[i].Hsml, Left[i], Right[i],
			 (float) PPP[i].n.NumNgb, Right[i] - Left[i], P[i].Pos[0], P[i].Pos[1], P[i].Pos[2]);
		      fflush(stdout);
		    }

		  if(Right[i] > 0 && Left[i] > 0)
		    PPP[i].Hsml = pow(0.5 * (pow(Left[i], 3) + pow(Right[i], 3)), 1.0 / 3);
		  else
		    {
		      if(Right[i] == 0 && Left[i] == 0)
			endrun(8188);	/* can't occur */

		      if(Right[i] == 0 && Left[i] > 0)
			{
			  PPP[i].Hsml *= 1.26;
			}

		      if(Right[i] > 0 && Left[i] == 0)
			{
			  PPP[i].Hsml /= 1.26;
			}
		    }

		  if(PPP[i].Hsml < All.MinGasHsml)
		    PPP[i].Hsml = All.MinGasHsml;
		}
	      else
		P[i].TimeBin = -P[i].TimeBin - 1;	/* Mark as inactive */


	      /* CECILIA */
	      if(iter == MAXITER)
		{
		  int old_type;

		  old_type = P[i].Type;
		  P[i].Type = 4;	/* no SN mark any more */
		  PPP[i].n.NumNgb = 0;
		  printf("part=%d of type=%d was assigned NumNgb=%g and type=%d\n", i, old_type,
			 PPP[i].n.NumNgb, P[i].Type);
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
#ifndef CS_FEEDBACK
	      printf("failed to converge in neighbour iteration in update_weights \n");
	      fflush(stdout);
	      endrun(1155);
#else
	      /* CECILIA */
	      if(Flag_phase == 2)	/* HOT */
		{
		  printf("Not enough hot neighbours for energy/metal distribution part=%d Type=%d\n", i,
			 P[i].Type);
		  fflush(stdout);
		  break;
		  /* endrun(1156); */
		}
	      else
		{
		  printf("Not enough cold neighbours for energy/metal distribution part=%d Type=%d\n", i,
			 P[i].Type);
		  fflush(stdout);
		  break;
		}
#endif
	    }
	}
    }
  while(ntot > 0);


  myfree(DataNodeList);
  myfree(DataIndexTable);
  myfree(R2ngblist);
  myfree(Ngblist);
  myfree(Right);
  myfree(Left);

  /* mark as active again */
  for(i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i])
    if(P[i].TimeBin < 0)
      P[i].TimeBin = -P[i].TimeBin - 1;

  /* collect some timing information */

  if(ThisTask == 0)
    {
      printf("... update weights phase = %d done...\n", Flag_phase);
      fflush(stdout);
    }

}


/*! This function represents the core of the SPH density computation. The
 *  target particle may either be local, or reside in the communication
 *  buffer.
 */
int cs_update_weight_evaluate(int target, int mode, int *nexport, int *nsend_local)
{
  int j, n;
  int startnode, numngb, listindex = 0;
  double h, h2, hinv, hinv3;
  double wk;
  double r, r2, u;
  MyLongDouble weighted_numngb;
  MyDouble *pos;

  weighted_numngb = 0;

  if(mode == 0)
    {
      pos = P[target].Pos;
      h = PPP[target].Hsml;
    }
  else
    {
      pos = UpdateweightGet[target].Pos;
      h = UpdateweightGet[target].Hsml;
    }

  h2 = h * h;
  hinv = 1.0 / h;
#ifndef  TWODIMS
  hinv3 = hinv * hinv * hinv;
#else
  hinv3 = hinv * hinv / boxSize_Z;
#endif

  if(mode == 0)
    {
      startnode = All.MaxPart;	/* root node */
    }
  else
    {
      startnode = UpdateweightGet[target].NodeList[0];
      startnode = Nodes[startnode].u.d.nextnode;	/* open it */
    }

  numngb = 0;

  while(startnode >= 0)
    {
      while(startnode >= 0)
	{
	  numngb = cs_ngb_treefind_variable_phases(pos, h, target, &startnode, mode, nexport, nsend_local);

	  if(numngb < 0)
	    return -1;

	  for(n = 0; n < numngb; n++)
	    {
	      j = Ngblist[n];
	      r2 = R2ngblist[n];

	      r = sqrt(r2);

	      u = r * hinv;

	      if(u < 0.5)
		{
		  wk = hinv3 * (KERNEL_COEFF_1 + KERNEL_COEFF_2 * (u - 1) * u * u);
		}
	      else
		{
		  wk = hinv3 * KERNEL_COEFF_5 * (1.0 - u) * (1.0 - u) * (1.0 - u);
		}

	      weighted_numngb += FLT(NORM_COEFF * wk / hinv3);	/* 4.0/3 * PI = 4.188790204786 */
	    }
	}

      if(mode == 1)
	{
	  listindex++;
	  if(listindex < NODELISTLENGTH)
	    {
	      startnode = UpdateweightGet[target].NodeList[listindex];
	      if(startnode >= 0)
		startnode = Nodes[startnode].u.d.nextnode;	/* open it */
	    }
	}
    }

  if(mode == 0)
    {
      PPP[target].n.dNumNgb = weighted_numngb;
    }
  else
    {
      UpdateweightResult[target].Ngb = weighted_numngb;
    }

  return 0;
}

#endif
