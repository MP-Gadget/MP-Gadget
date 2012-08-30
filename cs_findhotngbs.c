#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>



#include "allvars.h"
#include "proto.h"

#if defined(CS_MODEL) && defined(CS_FEEDBACK)

#include "cs_metals.h"

static struct hotngbs_in
{
  MyDouble Pos[3];
  MyFloat HotHsml;
  MyFloat Entropy;
  int NodeList[NODELISTLENGTH];
}
 *HotNgbsIn, *HotNgbsGet;

static struct hotngbs_out
{
  MyLongDouble DensitySum;
  MyLongDouble EntropySum;
  int HotNgbNum;
}
 *HotNgbsResult, *HotNgbsOut;


void cs_find_hot_neighbours(void)
{
  MyFloat *Left, *Right;
  int nimport;
  int i, j, n, ndone_flag, dummy;
  int ndone, ntot, npleft;
  int iter = 0;
  int ngrp, sendTask, recvTask;
  int place, nexport;
  double dmax1, dmax2;
  double xhyd, yhel, ne, mu, energy, temp;
  double a3inv;


  if(All.ComovingIntegrationOn)
    a3inv = 1 / (All.Time * All.Time * All.Time);
  else
    a3inv = 1;

  /* allocate buffers to arrange communication */

  Left = (MyFloat *) mymalloc("Left", NumPart * sizeof(MyFloat));
  Right = (MyFloat *) mymalloc("Right", NumPart * sizeof(MyFloat));

  Ngblist = (int *) mymalloc("Ngblist", NumPart * sizeof(int));

  All.BunchSize =
    (int) ((All.BufferSize * 1024 * 1024) / (sizeof(struct data_index) + sizeof(struct data_nodelist) +
					     sizeof(struct hotngbs_in) + sizeof(struct hotngbs_out) +
					     sizemax(sizeof(struct hotngbs_in), sizeof(struct hotngbs_out))));
  DataIndexTable =
    (struct data_index *) mymalloc("DataIndexTable", All.BunchSize * sizeof(struct data_index));
  DataNodeList =
    (struct data_nodelist *) mymalloc("DataNodeList", All.BunchSize * sizeof(struct data_nodelist));


  CPU_Step[CPU_MISC] += measure_time();




  for(n = FirstActiveParticle; n >= 0; n = NextActiveParticle[n])
    {
      if(P[n].Type == 0)
	{
	  /* select reservoir and cold phase particles */
	  if(P[n].EnergySN > 0 && SphP[n].d.Density * a3inv > All.PhysDensThresh * All.DensFrac_Phase)
	    {
	      xhyd = P[n].Zm[6] / P[n].Mass;
	      yhel = (1 - xhyd) / (4. * xhyd);

	      ne = SphP[n].Ne;
	      mu = (1 + 4 * yhel) / (1 + yhel + ne);
	      energy = SphP[n].Entropy * P[n].Mass / GAMMA_MINUS1 * pow(SphP[n].d.Density * a3inv, GAMMA_MINUS1);	/* Total Energys */
	      temp = GAMMA_MINUS1 / BOLTZMANN * energy / P[n].Mass * PROTONMASS * mu;
	      temp *= All.UnitEnergy_in_cgs / All.UnitMass_in_g;	/* Temperature in Kelvin */

	      if(temp < All.Tcrit_Phase)
		{
		  Left[n] = Right[n] = 0;

		  if(!(SphP[n].HotHsml > 0.))
		    SphP[n].HotHsml = All.InitialHotHsmlFactor * PPP[n].Hsml;	/* Estimation of HotHsml : ONLY first step */

		  P[n].Type = 10;	/* temporarily mark particles of interest with this number */
		}
	    }
	}
    }



  /* we will repeat the whole thing for those particles where we didn't find enough neighbours */
  do
    {
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
	    if(P[i].Type == 10 && P[i].TimeBin >= 0)
	      {
		if(cs_hotngbs_evaluate(i, 0, &nexport, Send_count) < 0)
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

	  HotNgbsGet = (struct hotngbs_in *) mymalloc("	  HotNgbsGet", nimport * sizeof(struct hotngbs_in));
	  HotNgbsIn = (struct hotngbs_in *) mymalloc("	  HotNgbsIn", nexport * sizeof(struct hotngbs_in));

	  /* prepare particle data for export */
	  for(j = 0; j < nexport; j++)
	    {
	      place = DataIndexTable[j].Index;

	      HotNgbsIn[j].Pos[0] = P[place].Pos[0];
	      HotNgbsIn[j].Pos[1] = P[place].Pos[1];
	      HotNgbsIn[j].Pos[2] = P[place].Pos[2];
	      HotNgbsIn[j].HotHsml = SphP[place].HotHsml;
	      HotNgbsIn[j].Entropy = SphP[place].Entropy;
	      memcpy(HotNgbsIn[j].NodeList,
		     DataNodeList[DataIndexTable[j].IndexGet].NodeList, NODELISTLENGTH * sizeof(int));
	    }


	  for(ngrp = 1; ngrp < (1 << PTask); ngrp++)
	    {
	      sendTask = ThisTask;
	      recvTask = ThisTask ^ ngrp;

	      if(recvTask < NTask)
		{
		  if(Send_count[recvTask] > 0 || Recv_count[recvTask] > 0)
		    {
		      /* get the particles */
		      MPI_Sendrecv(&HotNgbsIn[Send_offset[recvTask]],
				   Send_count[recvTask] * sizeof(struct hotngbs_in), MPI_BYTE,
				   recvTask, TAG_DENS_A,
				   &HotNgbsGet[Recv_offset[recvTask]],
				   Recv_count[recvTask] * sizeof(struct hotngbs_in), MPI_BYTE,
				   recvTask, TAG_DENS_A, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		    }
		}
	    }

	  myfree(HotNgbsIn);
	  HotNgbsResult =
	    (struct hotngbs_out *) mymalloc("	  HotNgbsResult", nimport * sizeof(struct hotngbs_out));
	  HotNgbsOut =
	    (struct hotngbs_out *) mymalloc("	  HotNgbsOut", nexport * sizeof(struct hotngbs_out));

	  /* now do the particles that need to be exported */
	  for(j = 0; j < nimport; j++)
	    cs_hotngbs_evaluate(j, 1, &dummy, &dummy);


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
		      MPI_Sendrecv(&HotNgbsResult[Recv_offset[recvTask]],
				   Recv_count[recvTask] * sizeof(struct hotngbs_out),
				   MPI_BYTE, recvTask, TAG_DENS_B,
				   &HotNgbsOut[Send_offset[recvTask]],
				   Send_count[recvTask] * sizeof(struct hotngbs_out),
				   MPI_BYTE, recvTask, TAG_DENS_B, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		    }
		}

	    }


	  /* add the result to the local particles */

	  for(j = 0; j < nexport; j++)
	    {
	      place = DataIndexTable[j].Index;

	      SphP[place].da.dDensityAvg += HotNgbsOut[j].DensitySum;
	      SphP[place].ea.dEntropyAvg += HotNgbsOut[j].EntropySum;
	      SphP[place].HotNgbNum += HotNgbsOut[j].HotNgbNum;
	    }

	  myfree(HotNgbsOut);
	  myfree(HotNgbsResult);
	  myfree(HotNgbsGet);
	}
      while(ndone < NTask);

      /* do final operations on results */
      for(i = FirstActiveParticle, npleft = 0; i >= 0; i = NextActiveParticle[i])
	{
	  if(P[i].Type == 10 && P[i].TimeBin >= 0)
	    {
#ifdef FLTROUNDOFFREDUCTION
	      SphP[i].da.DensityAvg = FLT(SphP[i].da.dDensityAvg);
	      SphP[i].ea.EntropyAvg = FLT(SphP[i].ea.dEntropyAvg);
#endif
	      if(SphP[i].HotNgbNum > 0)
		{
		  SphP[i].da.DensityAvg /= SphP[i].HotNgbNum;
		  SphP[i].ea.EntropyAvg /= SphP[i].HotNgbNum;
		}
	      else
		{
		  SphP[i].da.DensityAvg = 0;
		  SphP[i].ea.EntropyAvg = 0;
		}

	      /* now check whether we had enough neighbours */

	      if(SphP[i].HotNgbNum < (All.DesNumNgb - All.MaxNumHotNgbDeviation) ||
		 (SphP[i].HotNgbNum > (All.DesNumNgb + All.MaxNumHotNgbDeviation)))
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

		  if(SphP[i].HotNgbNum < (All.DesNumNgb - All.MaxNumHotNgbDeviation))
		    Left[i] = DMAX(SphP[i].HotHsml, Left[i]);
		  else
		    {
		      if(Right[i] != 0)
			{
			  if(SphP[i].HotHsml < Right[i])
			    Right[i] = SphP[i].HotHsml;
			}
		      else
			Right[i] = SphP[i].HotHsml;
		    }

		  if(Left[i] > All.MaxHotHsmlParam * PPP[i].Hsml)	/* prevent us from searching too far */
		    {
		      npleft--;
		      P[i].TimeBin = -P[i].TimeBin - 1;	/* Mark as inactive */


		      /* Ad-hoc definition of SAvg and RhoAvg when there are no hot neighbours  */
		      /* Note that a minimum nunmber of hot neighbours are required for promotion, see c_enrichment.c  */
		      if(SphP[i].HotNgbNum == 0)
			{
			  SphP[i].da.DensityAvg = SphP[i].d.Density / 100;
			  SphP[i].ea.EntropyAvg = SphP[i].Entropy * 1000;

			  printf("WARNING: Used ad-hoc values for SAvg and RhoAvg, No hot neighbours\n");
			}

		      continue;
		    }

		  if(iter >= MAXITER_HOT - 10)
		    {
		      printf
			("i=%d task=%d ID=%d Hsml=%g Left=%g Right=%g Ngbs=%g Right-Left=%g\n   pos=(%g|%g|%g)\n",
			 i, ThisTask, P[i].ID, SphP[i].HotHsml, Left[i], Right[i],
			 (float) SphP[i].HotNgbNum, Right[i] - Left[i], P[i].Pos[0], P[i].Pos[1],
			 P[i].Pos[2]);
		      fflush(stdout);
		    }

		  if(Right[i] > 0 && Left[i] > 0)
		    SphP[i].HotHsml = pow(0.5 * (pow(Left[i], 3) + pow(Right[i], 3)), 1.0 / 3);
		  else
		    {
		      if(Right[i] == 0 && Left[i] == 0)
			endrun(8188);	/* can't occur */

		      if(Right[i] == 0 && Left[i] > 0)
			SphP[i].HotHsml *= 1.26;

		      if(Right[i] > 0 && Left[i] == 0)
			SphP[i].HotHsml /= 1.26;
		    }
		}
	      else
		P[i].TimeBin = -P[i].TimeBin - 1;	/* Mark as inactive */
	    }
	}


      MPI_Allreduce(&npleft, &ntot, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

      if(ntot > 0)
	{
	  iter++;

	  if(iter > 0 && ThisTask == 0)
	    {
	      printf("hotngb iteration %d: need to repeat for %d particles.\n", iter, ntot);
	      fflush(stdout);
	    }

	  if(iter > MAXITER_HOT)
	    {
	      printf("failed to converge in hot-neighbour iteration\n");
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


  for(i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i])
    if(P[i].Type == 10)
      {
	P[i].Type = 0;
	/* mark as active again */
	if(P[i].TimeBin < 0)
	  P[i].TimeBin = -P[i].TimeBin - 1;
      }


  CPU_Step[CPU_HOTNGBS] += measure_time();

}



int cs_hotngbs_evaluate(int target, int mode, int *nexport, int *nsend_local)
{
  int j, n;
  int startnode, numngb_insphere, listindex = 0;
  double h, h2;
  MyDouble *pos;
  MyFloat entropy;
  MyLongDouble density_sum;
  MyLongDouble entropy_sum;
  int hotngb_count;


#ifdef PERIODIC
  double boxSize, boxHalf;

  boxSize = All.BoxSize;
  boxHalf = 0.5 * All.BoxSize;
#endif


  density_sum = 0;
  entropy_sum = 0;
  hotngb_count = 0;

  if(mode == 0)
    {
      pos = P[target].Pos;
      entropy = SphP[target].Entropy;
      h = SphP[target].HotHsml;
    }
  else
    {
      pos = HotNgbsGet[target].Pos;
      entropy = HotNgbsGet[target].Entropy;
      h = HotNgbsGet[target].HotHsml;
    }


  h2 = h * h;

  if(mode == 0)
    {
      startnode = All.MaxPart;	/* root node */
    }
  else
    {
      startnode = HotNgbsGet[target].NodeList[0];
      startnode = Nodes[startnode].u.d.nextnode;	/* open it */
    }


  while(startnode >= 0)
    {
      while(startnode >= 0)
	{
	  numngb_insphere = cs_ngb_treefind_hotngbs(&pos[0], h, target, &startnode, entropy,
						    mode, nexport, nsend_local);

	  if(numngb_insphere < 0)
	    return -1;

	  for(n = 0; n < numngb_insphere; n++)
	    {
	      j = Ngblist[n];

	      density_sum += SphP[j].d.Density;
	      entropy_sum += SphP[j].Entropy;
	      hotngb_count++;
	    }
	}

      if(mode == 1)
	{
	  listindex++;
	  if(listindex < NODELISTLENGTH)
	    {
	      startnode = HotNgbsGet[target].NodeList[listindex];
	      if(startnode >= 0)
		startnode = Nodes[startnode].u.d.nextnode;	/* open it */
	    }
	}
    }
  while(startnode >= 0);


  if(mode == 0)
    {
      SphP[target].da.dDensityAvg = density_sum;
      SphP[target].ea.dEntropyAvg = entropy_sum;
      SphP[target].HotNgbNum = hotngb_count;
    }
  else
    {
      HotNgbsResult[target].DensitySum = density_sum;
      HotNgbsResult[target].EntropySum = entropy_sum;
      HotNgbsResult[target].HotNgbNum = hotngb_count;
    }

  return 0;
}


#endif
