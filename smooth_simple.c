#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <gsl/gsl_math.h>

#include "allvars.h"
#include "proto.h"

#if (defined(DIVBCLEANING_DEDNER) || defined(SMOOTH_ROTB) || defined(BSMOOTH) || defined(SCAL_PRO_CLEAN)) || defined(VECT_POTENTIAL) || (defined(LT_STELLAREVOLUTION) && !defined(LT_DONTUSE_DENSITY_in_WEIGHT)) || defined(LT_SMOOTH_Z) || defined(LT_SMOOTH_XCLD) || defined(LT_TRACK_WINDS) || defined(LT_BH) || defined(LT_BH_LOG)

#if defined(LT_STELLAREVOLUTION)
int smooth_isactive(int);
#endif

#ifdef LT_BH_LOG
double a3inv, dmax1, dmax2, dmin1, dmin2;

typedef struct
{
  double W, R, Rf, N;
  double Z, T, Rho, Dist;
} BHLOG;

BHLOG  BHAvg, BHMin, BHMax;
BHLOG  allBHAvg, allBHMin, allBHMax;
double BHCumM, allBHCumM;
int    BHN, allBHN;
#endif

/*! Structure for communication during the density computation. Holds data that is sent to other processors.
 */
static struct smoothdata_in
{
  MyDouble Pos[3];
  MyFloat Hsml;
  int NodeList[NODELISTLENGTH];
#if (defined(LT_STELLAREVOLUTION) && !defined(LT_DONTUSE_DENSITY_in_WEIGHT)) || defined(LT_BH_LOG)
/* #if defined(LT_SEvDbg) */
/*   MyIDType ID; */
/* #endif */
  int Type;
#endif
#ifdef BLACK_HOLES
#ifdef LT_BH_CUT_KERNEL
  float CutHsml;
#endif
#ifdef LT_DF_BH_ONLY_RADIOMODE_KERNEL
  double BH_MdotEddington;
#endif
#endif
}
 *SmoothDataIn, *SmoothDataGet;


static struct smoothdata_out
{
#ifdef SMOOTH_PHI
  MyFloat SmoothPhi;
#endif
#ifdef VECT_POTENTIAL
  MyFloat SmoothA[3];
#endif
#if defined(DIVBCLEANING_DEDNER) || defined(SCAL_PRO_CLEAN)
  MyFloat SmoothDivB;
#endif
#ifdef SMOOTH_ROTB
  MyFloat SmoothRotB[3];
#endif
#if defined(BSMOOTH)
  MyFloat BSmooth[3];
#endif
#if defined(BLACK_HOLES)
  MyLongDouble SmoothedEntr;
#endif
#if defined(LT_STELLAREVOLUTION)
#if defined(LT_SMOOTH_Z)
  FLOAT Zsmooth;
  FLOAT Zsmooth_a;
  FLOAT Zsmooth_b;
#if defined(LT_SMOOTH_SIZE) || defined(LT_SMOOTH_NGB)
  FLOAT SmoothDens;
  FLOAT SmoothDens_b;
  int SmoothNgb;
#endif
#endif
#if defined(LT_SMOOTH_XCLD)
  FLOAT XCLDsmooth;
#endif
#if defined(LT_TRACK_WINDS)
  FLOAT AvgHsml;
#endif
#endif
  
  MyFloat DensityNorm;

#if defined(LT_BH) || defined(LT_BH_LOG)
  int    InnerNgb;
  MyDouble dBH_AltDensity;
#endif
#ifdef LT_BH_LOG
  MyFloat  MinW, MaxW, AvgW;
  MyFloat  CumM, CumCM, AvgZ, AvgTemp, AvgRho, MinDist, AvgDist;
#endif  
}
 *SmoothDataResult, *SmoothDataOut;

void smoothed_values(void)
{
  int ngrp, sendTask, recvTask, place, nexport, nimport;
  int i, j, ndone, ndone_flag, dummy;
  double timeall = 0, timecomp1 = 0, timecomp2 = 0, timecommsumm1 = 0, timecommsumm2 = 0, timewait1 =
    0, timewait2 = 0;
  double timecomp, timecomm, timewait;
  double tstart, tend, t0, t1;

#if defined(BSMOOTH)
  int Smooth_Flag = 0;
  double dB[3];
#endif

#ifdef LT_SMOOTH_SIZE
  double SmoothSize;
#endif
#ifdef LT_BH
  unsigned int           NumBHUpdate;
  unsigned long long int ntotBH;
#endif

  /* Display information message that this step is executed on Task 0 ... */
  if(ThisTask == 0)
    {
      printf("Updating SPH interpolants for:"
#if defined(SMOOTH_PHI)
	     " (Phi - Dedner)"
#endif /* SMOOTH_PHI */
#if defined(VECT_POTENTIAL)
	     " (Vect A)"
#endif /* VECT_POTENTIAL */
#if defined(DIVBCLEANING_DEDNER) || defined(SCAL_PRO_CLEAN)
	     " (DivB) "
#endif
#ifdef SMOOTH_ROTB
	     " (rotB)"
#endif /* SMOOTH_ROTB */
#if defined(BSMOOTH)
	     " (B)"
#endif /* BSMOOTH */
#if defined(LT_STELLAREVOLUTION) && !defined(LT_DONTUSE_DENSITY_in_WEIGHT)
	     " (rho around stars)"
#endif
#ifdef LT_SMOOTH_Z
	     " (metallicity)"
#endif
#ifdef LT_SMOOTH_XCLD
	     " (cloud fraction)"
#endif
#ifdef LT_TRACK_WINDS
	     " (Hsml)"
#endif
#ifdef LT_BH
             " (BH kernel)"
#endif             
	     "\n");
#ifdef BSMOOTH
      printf("Flag_FullStep = %d, Main TimestepCounts = %d\n", Flag_FullStep, All.MainTimestepCounts);
#endif
    }
#if defined(BSMOOTH)
  if(Flag_FullStep == 1)
    {
      if((All.MainTimestepCounts % All.BSmoothInt == 0) && (All.BSmoothInt >= 0))
	{
	  Smooth_Flag = 1;
	  if(ThisTask == 0)
	    printf("Smoothing B %d, %f\n", All.BSmoothInt, All.BSmoothFrac);
	}
      All.MainTimestepCounts++;
    }
#endif

#ifdef LT_BH_LOG
  if(All.ComovingIntegrationOn)
    a3inv = 1 / (All.Time * All.Time * All.Time);
  else
    a3inv = 1;
#endif

  
  Ngblist = (int *) mymalloc("Ngblist", NumPart * sizeof(int));

  All.BunchSize =
    (int) ((All.BufferSize * 1024 * 1024) / (sizeof(struct data_index) + sizeof(struct data_nodelist) +
					     sizeof(struct smoothdata_in) + sizeof(struct smoothdata_out) +
					     sizemax(sizeof(struct smoothdata_in),
						     sizeof(struct smoothdata_out))));

  DataIndexTable =
    (struct data_index *) mymalloc("DataIndexTable", All.BunchSize * sizeof(struct data_index));
  DataNodeList =
    (struct data_nodelist *) mymalloc("DataNodeList", All.BunchSize * sizeof(struct data_nodelist));

#if defined(LT_STELLAREVOLUTION) && !defined(LT_DONTUSE_DENSITY_in_WEIGHT)
  unsigned int appended, tot_appended, alreadyactive, tot_alreadyactive;

  appended = append_chemicallyactive_particles(&alreadyactive);
  MPI_Reduce(&appended, &tot_appended, 1, MPI_UNSIGNED, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&alreadyactive, &tot_alreadyactive, 1, MPI_UNSIGNED, MPI_SUM, 0, MPI_COMM_WORLD);
  if(ThisTask == 0 && (tot_appended > 0 || tot_alreadyactive > 0))
    printf("%u chemically active particles queued for smoothing calculation (%u already active)..\n",
	   tot_appended, tot_alreadyactive);
  fflush(stdout);
#endif

  CPU_Step[CPU_SMTHMISC] += measure_time();
  t0 = second();

  i = FirstActiveParticle;	/* begin with this index */

  do
    {
      for(j = 0; j < NTask; j++)
	{
	  Send_count[j] = 0;
	  Exportflag[j] = -1;
	}

      /* do local particles and prepare export list */
      tstart = second();
      for(nexport = 0; i >= 0; i = NextActiveParticle[i])
	{
#if !defined(LT_STELLAREVOLUTION)
	  if(density_isactive(i))
#else
	  if(smooth_isactive(i))
#endif
	    {
	      if(smoothed_evaluate(i, 0, &nexport, Send_count) < 0)
		break;
	    }
	}
      tend = second();
      timecomp1 += timediff(tstart, tend);

#ifdef MYSORT
      mysort_dataindex(DataIndexTable, nexport, sizeof(struct data_index), data_index_compare);
#else
      qsort(DataIndexTable, nexport, sizeof(struct data_index), data_index_compare);
#endif

      tstart = second();

      MPI_Alltoall(Send_count, 1, MPI_INT, Recv_count, 1, MPI_INT, MPI_COMM_WORLD);

      tend = second();
      timewait1 += timediff(tstart, tend);

      for(j = 0, nimport = 0, Recv_offset[0] = 0, Send_offset[0] = 0; j < NTask; j++)
	{
	  nimport += Recv_count[j];

	  if(j > 0)
	    {
	      Send_offset[j] = Send_offset[j - 1] + Send_count[j - 1];
	      Recv_offset[j] = Recv_offset[j - 1] + Recv_count[j - 1];
	    }
	}

      SmoothDataGet =
	(struct smoothdata_in *) mymalloc("SmoothDataGet", nimport * sizeof(struct smoothdata_in));
      SmoothDataIn =
	(struct smoothdata_in *) mymalloc("SmoothDataIn", nexport * sizeof(struct smoothdata_in));

      /* prepare particle data for export */
      for(j = 0; j < nexport; j++)
	{
	  place = DataIndexTable[j].Index;

	  SmoothDataIn[j].Pos[0] = P[place].Pos[0];
	  SmoothDataIn[j].Pos[1] = P[place].Pos[1];
	  SmoothDataIn[j].Pos[2] = P[place].Pos[2];
	  SmoothDataIn[j].Hsml = PPP[place].Hsml;
#if defined(LT_STELLAREVOLUTION)
/* #if defined(LT_SEvDbg) */
/* 	  SmoothDataIn[j].ID = P[place].ID; */
/* #endif */
#if !defined(LT_DONTUSE_DENSITY_in_WEIGHT)
	  SmoothDataIn[j].Type = P[place].Type;
#endif
#endif
#if defined(LT_SMOOTH_SIZE) || defined(LT_SMOOTH_NGB)
#ifdef LT_SMOOTH_NGB
          SmoothDataIn[j].SmoothHsml = SphP[place].SmoothHsml;
#else
          if((SmoothDataIn[j].SmoothHsml =
              SmoothDataIn[j].Hsml * All.SmoothRegionSize) > All.SmoothRegionSizeMax)
            SmoothDataIn[j].SmoothHsml = All.SmoothRegionSizeMax;
#endif
#endif
#ifdef LT_BH_CUT_KERNEL
          SmoothDataIn[j].CutHsml = P[place].CutHsml;
#endif
	  memcpy(SmoothDataIn[j].NodeList,
		 DataNodeList[DataIndexTable[j].IndexGet].NodeList, NODELISTLENGTH * sizeof(int));
	}

      /* exchange particle data */
      tstart = second();
      for(ngrp = 1; ngrp < (1 << PTask); ngrp++)
	{
	  sendTask = ThisTask;
	  recvTask = ThisTask ^ ngrp;

	  if(recvTask < NTask)
	    {
	      if(Send_count[recvTask] > 0 || Recv_count[recvTask] > 0)
		{
		  /* get the particles */
		  MPI_Sendrecv(&SmoothDataIn[Send_offset[recvTask]],
			       Send_count[recvTask] * sizeof(struct smoothdata_in), MPI_BYTE,
			       recvTask, TAG_SMOOTH_A,
			       &SmoothDataGet[Recv_offset[recvTask]],
			       Recv_count[recvTask] * sizeof(struct smoothdata_in), MPI_BYTE,
			       recvTask, TAG_SMOOTH_A, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}
	    }
	}
      tend = second();
      timecommsumm1 += timediff(tstart, tend);

      myfree(SmoothDataIn);
      SmoothDataResult =
	(struct smoothdata_out *) mymalloc("SmoothDataResult", nimport * sizeof(struct smoothdata_out));
      SmoothDataOut =
	(struct smoothdata_out *) mymalloc("SmoothDataOut", nexport * sizeof(struct smoothdata_out));


      /* now do the particles that were sent to us */

      tstart = second();
      for(j = 0; j < nimport; j++)
	smoothed_evaluate(j, 1, &dummy, &dummy);
      tend = second();
      timecomp2 += timediff(tstart, tend);

      if(i < 0)
	ndone_flag = 1;
      else
	ndone_flag = 0;

      tstart = second();
      MPI_Allreduce(&ndone_flag, &ndone, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
      tend = second();
      timewait2 += timediff(tstart, tend);


      /* get the result */
      tstart = second();
      for(ngrp = 1; ngrp < (1 << PTask); ngrp++)
	{
	  sendTask = ThisTask;
	  recvTask = ThisTask ^ ngrp;
	  if(recvTask < NTask)
	    {
	      if(Send_count[recvTask] > 0 || Recv_count[recvTask] > 0)
		{
		  /* send the results */
		  MPI_Sendrecv(&SmoothDataResult[Recv_offset[recvTask]],
			       Recv_count[recvTask] * sizeof(struct smoothdata_out),
			       MPI_BYTE, recvTask, TAG_SMOOTH_B,
			       &SmoothDataOut[Send_offset[recvTask]],
			       Send_count[recvTask] * sizeof(struct smoothdata_out),
			       MPI_BYTE, recvTask, TAG_SMOOTH_B, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}
	    }
	}
      tend = second();
      timecommsumm2 += timediff(tstart, tend);


      /* add the result to the local particles */
      tstart = second();
      for(j = 0; j < nexport; j++)
	{
	  place = DataIndexTable[j].Index;

	  if(P[place].Type == 0)
	    {
#if defined(SMOOTH_PHI)
	      SphP[place].SmoothPhi += SmoothDataOut[j].SmoothPhi;
#endif /* SMOOTH_PHI */
#if defined(VECT_POTENTIAL)
	      SphP[place].SmoothA[0] += SmoothDataOut[j].SmoothA[0];
	      SphP[place].SmoothA[1] += SmoothDataOut[j].SmoothA[1];
	      SphP[place].SmoothA[2] += SmoothDataOut[j].SmoothA[2];
#endif /* VECT_POTENTIAL */
#if defined(DIVBCLEANING_DEDNER) || defined(SCAL_PRO_CLEAN)
	      SphP[place].SmoothDivB += SmoothDataOut[j].SmoothDivB;
#endif
#ifdef SMOOTH_ROTB
	      SphP[place].SmoothedRotB[0] += SmoothDataOut[j].SmoothRotB[0];
	      SphP[place].SmoothedRotB[1] += SmoothDataOut[j].SmoothRotB[1];
	      SphP[place].SmoothedRotB[2] += SmoothDataOut[j].SmoothRotB[2];
#endif /* SMOOTH_ROTB */

#if defined(BSMOOTH)
	      SphP[place].BSmooth[0] += SmoothDataOut[j].BSmooth[0];
	      SphP[place].BSmooth[1] += SmoothDataOut[j].BSmooth[1];
	      SphP[place].BSmooth[2] += SmoothDataOut[j].BSmooth[2];
#endif /* BSMOOTH */
#ifdef LT_SMOOTH_Z
	      SphP[place].Zsmooth += SmoothDataOut[j].Zsmooth;
#ifdef LT_SMOOTH_Z_DETAILS
	      SphP[place].Zsmooth_a += (float) SmoothDataOut[j].Zrho_a;
	      SphP[place].Zsmooth_b += (float) SmoothDataOut[j].Zrho_b;
	      SphP[place].SmoothDens_b += (float) SmoothDataOut[j].SmoothDens_b;
#endif
#if defined(LT_SMOOTH_SIZE) || defined(LT_SMOOTH_NGB)
	      SphP[place].SmoothDens += SmoothDataOut[j].SmoothDens;
	      SphP[place].SmoothNgb += SmoothDataOut[j].SmoothNgb;
#endif
#endif /* LT_SMOOTH_Z */

#ifdef LT_SMOOTH_CLDX
	      SphP[place].XCLDsmooth += SmoothDataOut[j].XCLDsmooth;
#endif /* LT_SMOOTH_CLDX */
#ifdef LT_TRACK_WINDS
	      SphP[place].AvgHsml += SmoothDataOut[j].AvgHsml;
#endif /* LT_TRACK_WINDS */
	      SphP[place].DensityNorm += SmoothDataOut[j].DensityNorm;
	    }

#ifdef LT_BH
	  if(P[place].Type == 5)   /* protect to write into SphP with traget not to be a gas particle ! */
	    {
	      P[place].b9.dBH_AltDensity += SmoothDataOut[j].dBH_AltDensity;
#ifdef LT_BH_LOG
	      if(P[place].MinW > SmoothDataOut[j].MinW)
		P[place].MinW             = SmoothDataOut[j].MinW;
	      if(P[place].MinDist > SmoothDataOut[j].MinDist)
		P[place].MinDist          = SmoothDataOut[j].MinDist;
	      if(P[place].MaxW < SmoothDataOut[j].MaxW)
		P[place].MaxW             = SmoothDataOut[j].MaxW;
	      P[place].AvgW              += SmoothDataOut[j].AvgW;
	      P[place].CumM              += SmoothDataOut[j].CumM;
	      P[place].CumCM             += SmoothDataOut[j].CumCM;
	      P[place].AvgZ              += SmoothDataOut[j].AvgZ;
	      P[place].AvgTemp           += SmoothDataOut[j].AvgTemp;
	      P[place].AvgRho            += SmoothDataOut[j].AvgRho;
	      P[place].AvgDist           += SmoothDataOut[j].AvgDist;
#endif  /* LT_BH_LOG */              
	    }
#endif  /* LT_BH */
#if defined(LT_STELLAREVOLUTION) && !defined(LT_DONTUSE_DENSITY_in_WEIGHT)
	  if(P[place].Type == 4)
	    MetP[P[place].MetID].weight += SmoothDataOut[j].DensityNorm;
#endif
	}
      tend = second();
      timecomp1 += timediff(tstart, tend);


      myfree(SmoothDataOut);
      myfree(SmoothDataResult);
      myfree(SmoothDataGet);
    }
  while(ndone < NTask);

  myfree(DataNodeList);
  myfree(DataIndexTable);
  myfree(Ngblist);



  /* do final operations on results */
  tstart = second();
  for(i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i])
    {
#if !defined(LT_STELLAREVOLUTION)
      if(density_isactive(i))
#else
      if(smooth_isactive(i))
#endif
        {
          if(P[i].Type == 0)
            {
#if defined(SMOOTH_PHI)
              SphP[i].SmoothPhi /= SphP[i].DensityNorm;
#endif /* SMOOTH_PHI */
#if defined(VECT_POTENTIAL)
              SphP[i].SmoothA[0] /= SphP[i].DensityNorm;
              SphP[i].SmoothA[1] /= SphP[i].DensityNorm;
              SphP[i].SmoothA[2] /= SphP[i].DensityNorm;
              SphP[i].APred[0] = SphP[i].SmoothA[0];
              SphP[i].APred[1] = SphP[i].SmoothA[1];
              SphP[i].APred[2] = SphP[i].SmoothA[2];
#endif
#if defined(DIVBCLEANING_DEDNER) || defined(SCAL_PRO_CLEAN)
              SphP[i].SmoothDivB /= SphP[i].DensityNorm;
#endif
#ifdef SMOOTH_ROTB
              SphP[i].SmoothedRotB[0] /= SphP[i].DensityNorm;
              SphP[i].SmoothedRotB[1] /= SphP[i].DensityNorm;
              SphP[i].SmoothedRotB[2] /= SphP[i].DensityNorm;
#endif /* SMOOTH_ROTB */

#if defined(BSMOOTH)
              SphP[i].BSmooth[0] /= SphP[i].DensityNorm;
              SphP[i].BSmooth[1] /= SphP[i].DensityNorm;
              SphP[i].BSmooth[2] /= SphP[i].DensityNorm;

              if(Smooth_Flag == 1)
                {
                  dB[0] = All.BSmoothFrac * (SphP[i].BSmooth[0] - SphP[i].BPred[0]);
                  dB[1] = All.BSmoothFrac * (SphP[i].BSmooth[1] - SphP[i].BPred[1]);
                  dB[2] = All.BSmoothFrac * (SphP[i].BSmooth[2] - SphP[i].BPred[2]);
                  SphP[i].BPred[0] += dB[0];
                  SphP[i].BPred[1] += dB[1];
                  SphP[i].BPred[2] += dB[2];
#ifndef EULERPOTENTIALS
                  SphP[i].B[0] += dB[0];
                  SphP[i].B[1] += dB[1];
                  SphP[i].B[2] += dB[2];
#endif
                }
#endif /* BSMOOTH */

#ifdef LT_SMOOTH_Z		/* > ============================================== < */
              /* >  SMOOTH Z                                      < */

#if !defined(LT_SMOOTH_SIZE) && !defined(LT_SMOOTH_NGB)
              SphP[i].Zsmooth /= SphP[i].DensityNorm;
#ifdef LT_SMOOTHZ_DETAILS
              SphP[i].Zsmooth_a /=
                SphP[i].DensityNorm * (P[i].Mass - get_metalmass(SphP[i].Metals) - SphP[i].Metals[Hel]);
              SphP[i].Zsmooth_b /= SphP[i].SmoothDens_b;
#endif
#else
              SphP[i].Zsmooth /= SphP[i].SmoothDens;
#ifdef LT_SMOOTHZ_DETAILS
              SphP[i].Zsmooth_a /=
                SphP[i].SmoothDens * (P[i].Mass - get_metalmass(SphP[i].Metals) - SphP[i].Metals[Hel]);
              SphP[i].Zsmooth_b /= SphP[i].SmoothDens_b;
#endif
#endif
#endif /* LT_SMOOTH:Z */
#ifdef LT_SMOOTH_XCLD
              SphP[i].XCLDsmooth /= SphP[i].DensityNorm;
#endif
#ifdef LT_TRACK_WINDS
              SphP[i].AvgHsml /= SphP[i].DensityNorm;
#endif
#if defined(LT_SMOOTH_SIZE)
              AvgSmoothN++;

              if((SmoothSize = PPP[i].Hsml * All.SmoothRegionSize) > All.SmoothRegionSizeMax)
                SmoothSize = All.SmoothRegionSizeMax;
              AvgSmoothSize += SmoothSize;
              if(SmoothSize < MinSmoothSize)
                MinSmoothSize = SmoothSize;
              if(SmoothSize > MaxSmoothSize)
                MaxSmoothSize = SmoothSize;

              AvgSmoothNgb += SphP[i].SmoothNgb;
              if(SphP[i].SmoothNgb < MinSmoothNgb)
                MinSmoothNgb = SphP[i].SmoothNgb;
              if(SphP[i].SmoothNgb > MaxSmoothNgb)
                MaxSmoothNgb = SphP[i].SmoothNgb;
#endif
            }

#if defined(LT_BH) || defined(LT_BH_LOG)
          if(P[i].Type == 5)
            {
#ifdef LT_BH            
              P[i].b9.BH_AltDensity = FLT(P[i].b9.dBH_AltDensity);
#endif
#ifdef LT_BH_LOG                
              if(P[i].CumM > 0)
                {
#ifdef LT_BH_CUT_KERNEL
#ifdef LT_DF_BH_ONLY_RADIOMODE_KERNEL
                  if(P[i].BH_MdotEddington < All.BH_radio_treshold)
                    {
#endif
		      if(P[i].b9.BH_AltDensity > 0)
			{
			  P[i].AvgW        /= (P[i].b9.BH_AltDensity * P[i].InnerNgb);
			  P[i].MinW        /= P[i].b9.BH_AltDensity;
			  P[i].MaxW        /= P[i].b9.BH_AltDensity;
			}
#ifdef LT_DF_BH_ONLY_RADIOMODE_KERNEL
                    }
                  else
                    {
		      if(P[i].b1.BH_Density > 0)
			{
			  P[i].AvgW        /= (P[i].b1.BH_Density * P[i].InnerNgb);
			  P[i].MinW        /= P[i].b1.BH_Density;
			  P[i].MaxW        /= P[i].b1.BH_Density;                    
			}
                    }
#endif
#endif		
                  P[i].AvgZ        /= P[i].CumM;
                  P[i].AvgTemp     /= P[i].CumM;
                  P[i].AvgRho      /= P[i].CumM;
                  P[i].AvgDist     /= P[i].CumM;
                }
              else
                {
                  P[i].AvgW = 0;
                  P[i].AvgZ = P[i].AvgTemp = P[i].AvgRho = 0;
                }

#ifdef LT_BH            
              if(P[i].InnerNgb < 10)
                fprintf(FdBlackHolesWarn,"%8.6e !! BH %10u @%4d has got very few neighbours: %d %8.6e\t %8.6e\t %8.6e\t %8.6e\n",
                        All.Time, P[i].ID, ThisTask, P[i].InnerNgb, P[i].Hsml, P[i].MinDist, P[i].CumM, P[i].b9.BH_AltDensity);
#else
              if(P[i].InnerNgb < 10)
                fprintf(FdBlackHolesWarn,"%8.6e !! BH %10u @%4d has got very few neighbours: %d %8.6e\t %8.6e\t %8.6e\t %8.6e\n",
                        All.Time, P[i].ID, ThisTask, P[i].InnerNgb, P[i].Hsml, P[i].MinDist, P[i].CumM, P[i].b1.BH_Density);
#endif
              
#endif
            }
#endif      
        }
    }
  tend = second();
  timecomp1 += timediff(tstart, tend);

#if defined(LT_STELLAREVOLUTION) && !defined(LT_DONTUSE_DENSITY_in_WEIGHT)
  drop_chemicallyactive_particles();
#endif

  /* collect some timing information */

  t1 = WallclockTime = second();
  timeall += timediff(t0, t1);

  timecomp = timecomp1 + timecomp2;
  timewait = timewait1 + timewait2;
  timecomm = timecommsumm1 + timecommsumm2;

  CPU_Step[CPU_SMTHCOMPUTE] += timecomp;
  CPU_Step[CPU_SMTHWAIT] += timewait;
  CPU_Step[CPU_SMTHCOMM] += timecomm;
  CPU_Step[CPU_SMTHMISC] += timeall - (timecomp + timewait + timecomm);

#ifdef LT_BH_LOG
  BHCumM = 0;
  BHN    = 0;
  
  memset(&BHAvg, 0, sizeof(BHLOG));
  memset(&BHMax , 0, sizeof(BHLOG));
#ifdef LT_BH_CUT_KERNEL
  BHMin.Rf = 1e10;
  BHMax.Rf = 0;
#endif
  BHMin.R = 1e10;
  BHMin.W = 1.0;
  BHMin.N = 1e10;
  BHMin.T = BHMin.Rho = BHMin.Z = BHMin.Dist = 1e19;
  for(i = 0; i < NumPart; i++)
    if(P[i].Type == 5)
      {
        BHN++;
        
        BHAvg.W   += P[i].AvgW;
        BHMin.W    = DMIN(BHMin.W, P[i].MinW);
        BHMax.W    = DMAX(BHMax.W, P[i].MaxW); 
        
        BHAvg.R   += PPP[i].Hsml;
        BHMin.R    = DMIN(BHMin.R, P[i].Hsml);
        BHMax.R    = DMAX(BHMax.R, P[i].Hsml);
        
#ifdef LT_BH_CUT_KERNEL          
        BHAvg.Rf  += P[i].CutHsml;
        BHMin.Rf   = DMIN(BHMin.Rf, P[i].CutHsml);
        BHMax.Rf   = DMAX(BHMax.Rf, P[i].CutHsml);
#endif
        
        BHAvg.N   += (double)P[i].InnerNgb;
        BHMin.N    = DMIN(BHMin.N, P[i].InnerNgb);
        BHMax.N    = DMAX(BHMax.N, P[i].InnerNgb);
        
        BHAvg.Z   += P[i].AvgZ;
        BHMin.Z    = DMIN(BHMin.Z, P[i].AvgZ);
        BHMax.Z    = DMAX(BHMax.Z, P[i].AvgZ); 
        
        BHAvg.T   += P[i].AvgTemp;
        BHMin.T    = DMIN(BHMin.T, P[i].AvgTemp);
        BHMax.T    = DMAX(BHMax.T, P[i].AvgTemp);
        
        BHAvg.Rho += P[i].AvgRho;
        BHMin.Rho  = DMIN(BHMin.Rho, P[i].AvgRho);
        BHMax.Rho  = DMAX(BHMax.Rho, P[i].AvgRho);

        BHAvg.Dist += P[i].AvgDist;
        BHMin.Dist  = DMIN(BHMin.Dist, P[i].MinDist);
      }
  MPI_Allreduce(&BHN, &allBHN, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  if(allBHN > 0)
    {
      
      MPI_Reduce(&BHAvg, &allBHAvg, sizeof(BHLOG) / sizeof(double), MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
      MPI_Reduce(&BHMin, &allBHMin, sizeof(BHLOG) / sizeof(double), MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
      MPI_Reduce(&BHMax, &allBHMax, sizeof(BHLOG) / sizeof(double), MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

      allBHAvg.W   /= allBHN;
      allBHAvg.R   /= allBHN;
      allBHAvg.Rf  /= allBHN;
      allBHAvg.N   /= allBHN;
      allBHAvg.Z   /= allBHN;
      allBHAvg.T   /= allBHN;
      allBHAvg.Rho /= allBHN;
      allBHAvg.Dist/= allBHN;
      
      if(ThisTask == 0)
        {
#ifdef LT_BH_CUT_KERNEL
          fprintf(FdBlackHolesProfile, "%8.6e\t "                                   /* time       */
                  "%8.6e\t %8.6e\t %8.6e\t %8.6e\t %8.6e\t %8.6e\t %8.6e\t %8.6e\t" /* average    */
                  "%8.6e\t %8.6e\t %8.6e\t %8.6e\t %8.6e\t %8.6e\t %8.6e\t %8.6e\t" /* min        */
                  "%8.6e\t %8.6e\t %8.6e\t %8.6e\t %8.6e\t %8.6e\t %8.6e\t %8s\n",  /* max        */
                  All.Time,
                  allBHAvg.Rho, allBHAvg.T, allBHAvg.Z, allBHAvg.R, allBHAvg.Rf, allBHAvg.W, allBHAvg.N, allBHAvg.Dist,
                  allBHMin.Rho, allBHMin.T, allBHMin.Z, allBHMin.R, allBHMin.Rf, allBHMin.W, allBHMin.N, allBHMin.Dist,
                  allBHMax.Rho, allBHMax.T, allBHMax.Z, allBHMax.R, allBHMax.Rf, allBHMax.W, allBHMax.N, "   -   ");
#else
          fprintf(FdBlackHolesProfile, "%8.6e\t "                            /* time       */
                  "%8.6e\t %8.6e\t %8.6e\t %8.6e\t %8.6e\t %8.6e\t %8.6e\t"  /* average    */
                  "%8.6e\t %8.6e\t %8.6e\t %8.6e\t %8.6e\t %8.6e\t %8.6e\t"  /* min        */
                  "%8.6e\t %8.6e\t %8.6e\t %8.6e\t %8.6e\t %8.6e\n %8.6e\t", /* max        */
                  All.Time,
                  allBHAvg.Rho, allBHAvg.T, allBHAvg.Z, allBHAvg.R, allBHAvg.W, allBHAvg.N, allBHAvg.Dist,
                  allBHMin.Rho, allBHMin.T, allBHMin.Z, allBHMin.R, allBHMin.W, allBHMin.N, allBHMin.Dist,
                  allBHMax.Rho, allBHMax.T, allBHMax.Z, allBHMax.R, allBHMax.W, allBHMax.N, "   -   ");
#endif
	  fflush(FdBlackHolesProfile);
        }
    }

#endif

}



/*! This function represents the core of the SPH density computation. The
*  target particle may either be local, or reside in the communication
*  buffer.
*/
int smoothed_evaluate(int target, int mode, int *nexport, int *nsend_local)
{
  int j, n, listindex = 0;
  int startnode, numngb_inbox;
  double h, h2, hinv, hinv3, hinv4;
  double wk, dwk;
  double dx, dy, dz, r, r2, u, mass_j;
  MyFloat *pos;
  double DensityNorm = 0;

#ifdef SMOOTH_PHI
  double SmoothPhi = 0.0;
#endif /* SMOOTH_PHI */
#if defined(VECT_POTENTIAL)
  double smoothA[3];

  smoothA[0] = smoothA[1] = smoothA[2] = 0.0;
#endif
#if defined(DIVBCLEANING_DEDNER) || defined(SCAL_PRO_CLEAN)
  double SmoothDivB = 0.0;
#endif
#ifdef SMOOTH_ROTB
  double smoothrotb[3];
#endif /* SMOOTH_ROTB */

#if defined(BSMOOTH)
  double BSmooth[3];

  BSmooth[0] = BSmooth[1] = BSmooth[2] = 0;
#endif /* BSMOOTH */

/*  MyIDType myID; */

#if !defined(LT_DONTUSE_DENSITY_in_WEIGHT)
  int Type;
#endif

#if defined(LT_SMOOTH_Z)	/* ======= LT_SMOOTH_Z */
  double SmoothSize;
  double getmetallicity;
  DOUBLE Zrho = 0;

#ifdef LT_SMOOTH_Z_DETAILS
  DOUBLE Zrho_a = 0;
  DOUBLE Zrho_b = 0;
#endif

#if defined(LT_SMOOTH_SIZE) || defined(LT_SMOOTH_NGB)	/* >>>>> */
  int SmoothNgb;
  double su;
  DOUBLE SmoothDens = 0;
  DOUBLE SmoothHsml, shinv, shinv3;

#ifdef LT_SMOOTHZ_DETAILS
  DOUBLE SmoothDens_b = 0;
#endif

#ifdef LT_SMOOTH_NGB
  int smoothcc;
#endif
#endif /* <<<<< closes #if defined(LT_SMOOTH_SIZE) || defined(LT_SMOOTH_NGB) */

#endif /* closes LT_SMOOTH_Z */

#ifdef LT_SMOOTH_XCLD
  DOUBLE XCLDsmooth;
#endif

#ifdef LT_TRACK_WINDS
  DOUBLE AvgHsml;
#endif

#ifdef LT_BH
  double AltRho;
  double BHCutHsml;
  double MinDist, AvgDist;
#endif
#ifdef LT_BH_LOG
  double MinW, MaxW;
  double AvgZ, AvgTemp, AvgRho, AvgW;
  double temperature, ne_guess, CumM, CumCM;;
#endif

#ifdef LT_DF_BH_ONLY_RADIOMODE_KERNEL
  double BHmdotedd;
#endif
  
#if defined(LT_SMOOTH_Z) || defined(LT_SMOOTH_SIZE) || defined(LT_SMOOTH_NGB) || defined(LT_BH) || defined(LT_BH_LOG)
  double swk;
#endif

#ifdef SMOOTH_ROTB
  smoothrotb[0] = smoothrotb[1] = smoothrotb[2] = 0;
#endif /* SMOOTH_ROTB */

#if defined(BSMOOTH)
  BSmooth[0] = BSmooth[1] = BSmooth[2] = 0;
#endif /* BSMOOTH */

  if(mode == 0)
    {
      pos = P[target].Pos;
      h = PPP[target].Hsml;
#if defined(LT_STELLAREVOLUTION)
/* #if defined(LT_SEvDbg) */
/*       myID = P[target].ID; */
/* #endif */
#if !defined(LT_DONTUSE_DENSITY_in_WEIGHT)
      Type = P[target].Type;
#endif
#endif
#ifdef LT_BH
#ifdef LT_BH_CUT_KERNEL
      BHCutHsml = P[target].CutHsml;
#else
      BHCutHsml = h;
#endif
#endif
#ifdef LT_DF_BH_ONLY_RADIOMODE_KERNEL
      BHmdotedd = P[target].BH_MdotEddington;
#endif      
    }
  else
    {
      pos = SmoothDataGet[target].Pos;
      h = SmoothDataGet[target].Hsml;
#if defined(LT_STELLAREVOLUTION)
/* #if defined(LT_SEvDbg) */
/*       myID = SmoothDataGet[target].ID; */
/* #endif */
#if !defined(LT_DONTUSE_DENSITY_in_WEIGHT)
      Type = SmoothDataGet[target].Type;
#endif
#endif
#ifdef LT_BH
#ifdef LT_BH_CUT_KERNEL
      BHCutHsml = SmoothDataGet[target].CutHsml; 
#else
      BHCutHsml = h;
#endif
#endif
#ifdef LT_DF_BH_ONLY_RADIOMODE_KERNEL
      BHmdotedd = SmoothDataGet[target].BH_MdotEddington;
#endif      
    }


  h2 = h * h;
  hinv = 1.0 / h;
#ifndef  TWODIMS
  hinv3 = hinv * hinv * hinv;
#else
  hinv3 = hinv * hinv / boxSize_Z;
#endif
  hinv4 = hinv3 * hinv;

#ifdef LT_BH
  AltRho = 0;
#endif
#ifdef LT_BH_LOG
  MinW   = 1;
  MaxW = AvgW = AvgRho = AvgTemp = AvgZ = AvgDist = CumM = CumCM = 0;
  MinDist = h;
#endif
  
#if defined(LT_SMOOTH_SIZE) || defined(LT_SMOOTH_NGB)
#ifdef LT_SMOOTH_SIZE
  if((SmoothHsml = h * All.SmoothRegionSize) > All.SmoothRegionSizeMax)
    SmoothHsml = All.SmoothRegionSizeMax;
#endif
#ifdef LT_SMOOTH_NGB
  SmoothHsml = h;
  memset(NearestSmooth, 0, All.DesNumNbSmooth * sizeof(float));
#endif
  shinv = 1.0 / SmoothHsml;
  shinv3 = shinv * shinv * shinv;
  SmoothNgb = 0;
#endif


  if(mode == 0)
    {
      startnode = All.MaxPart;	/* root node */
    }
  else
    {
      startnode = SmoothDataGet[target].NodeList[0];
      startnode = Nodes[startnode].u.d.nextnode;	/* open it */
    }

  while(startnode >= 0)
    {
      while(startnode >= 0)
	{
	  numngb_inbox = ngb_treefind_variable(&pos[0], h, target, &startnode, mode, nexport, nsend_local);

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
		  r = sqrt(r2);

#ifdef LT_SMOOTH_NGB
		  for(smoothcc = 0; smoothcc < All.DesNumNgbSmooth; smoothcc++)
		    if(Nearest[smoothcc] > (float) r)
		      Nearest[smoothcc] = (float) r;
#endif

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
		  wk /= SphP[j].d.Density;

#if defined(SMOOTH_PHI)
		  SmoothPhi += mass_j * wk * SphP[j].PhiPred;
#endif /* SMOOTH_PHI */
#if defined(VECT_POTENTIAL)
		  smoothA[0] += mass_j * wk * SphP[j].APred[0];
		  smoothA[1] += mass_j * wk * SphP[j].APred[1];
		  smoothA[2] += mass_j * wk * SphP[j].APred[2];
#endif
#if defined(DIVBCLEANING_DEDNER) || defined(SCAL_PRO_CLEAN)
		  SmoothDivB += mass_j * wk * SphP[j].divB;
#endif
#ifdef SMOOTH_ROTB
		  smoothrotb[0] += mass_j * wk * SphP[j].RotB[0];
		  smoothrotb[1] += mass_j * wk * SphP[j].RotB[1];
		  smoothrotb[2] += mass_j * wk * SphP[j].RotB[2];
#endif /* SMOOTH_ROTB */

#if defined(BSMOOTH)
		  BSmooth[0] += mass_j * wk * SphP[j].BPred[0];
		  BSmooth[1] += mass_j * wk * SphP[j].BPred[1];
		  BSmooth[2] += mass_j * wk * SphP[j].BPred[2];
#endif /* BSMOOTH */

#if defined(LT_SMOOTH_Z)	/* ========== LT_SMOOTH_Z */
#if !defined(LT_SMOOTH_SIZE) && !defined(LT_SMOOTH_NGB)	/* SMOOTH_SIZE and SMOOT_NGB are NOT defined */
		  getmetallicity = get_metallicity(j, -1);
		  Zrho += (DOUBLE) getmetallicity * mass_j * wk;
                  swk = wk;
                  SmoothDens_b += mass_j * wk * SphP[j.d.Density;
#else /* SMOOTH_SIZE or SMOOTH_NGB are defined */
		  if(r <= SmoothHsml)
		    {
		      SmoothNgb++;
		      su = r * shinv;
		      if(su < 0.5)
			swk = shinv3 * (KERNEL_COEFF_1 + KERNEL_COEFF_2 * (su - 1) * su * su);
		      else
			swk = shinv3 * KERNEL_COEFF_5 * (1.0 - su) * (1.0 - su) * (1.0 - su);
                      swk /= SphP[j].d.Density;
                      
		      getmetallicity = get_metallicity(j, -1);
		      Zrho += (DOUBLE) getmetallicity * mass_j * swk;

		      SmoothDens += mass_j * swk;
#ifdef LT_SMOOTH_Z_DETAILS
		      SmoothDens_b += SmoothDens * SphP[j].d.Density;
#endif
                    }
#endif /* closes SMOOTH_SIZE || SMOOTH_NGB */
#ifdef LT_SMOOTH_Z_DETAILS
                  Zrho_a += (DOUBLE) get_metalmass(SphP[j].Metals) * mass_j * swk;         /* smooth the metal mass */
                  Zrho_b += (DOUBLE) getmetallicity * mass_j * swk * SphP[j].d.Density;    /* smooth Z without density */
#endif                      
#endif /* ========== closes LT_SMOOTH_Z */

#ifdef LT_SMOOTH_XCLD
		  XCLDsmooth += (DOUBLE) SphP[j].x * mass_j * wk;
#endif /* LT_SMOOTH_XCLD */

#ifdef LT_TRACK_WINDS
		  AvgHsml += (DOUBLE) SphP[j].Hsml * mass_j * wk;
#endif /* LT_TRACK_WINDS */

#ifdef LT_BH                                                             /* ===== LT_BH */
                  swk = 0;
#ifdef LT_BH_CUT_KERNEL
#ifdef LT_DF_BH_ONLY_RADIOMODE_KERNEL
                  if ( ((BHmdotedd < All.BH_radio_treshold) && (r <= BHCutHsml)) ||
                       (BHmdotedd >= All.BH_radio_treshold)	)			/* :: radio mode on */
#else
                  if(r <= BHCutHsml)
#endif
                    {
#endif
                      if(r > 0 && r < MinDist)
                        {
                          MinDist = r;
                          AvgDist += r * mass_j;
                        }
#ifdef LT_BH_DONOT_USEDENSITY_IN_KERNEL
                      swk = 1.0;
#else
                      swk = 1.0 / SphP[j].d.Density;
#endif
#ifndef LT_BH_USETOPHAT_KERNEL
#ifdef LT_DF_BH_ONLY_RADIOMODE_KERNEL
                      if( BHmdotedd >= All.BH_radio_treshold ) 		/* :: radio mode off */
                        {
                          swk *= wk;
                        }
#else
                      swk *= wk;
#endif
#endif
#ifdef LT_BH_CUT_KERNEL
                    }
#endif
                  AltRho += swk * mass_j;
#endif /* LT_BH */

#ifdef LT_BH_LOG                                                         /* ===== LT_BH_LOG */
#ifndef LT_BH
                  swk = wk;
#endif
#ifdef LT_BH_CUT_KERNEL
#ifdef LT_DF_BH_ONLY_RADIOMODE_KERNEL
                  if ( BHmdotedd < All.BH_radio_treshold )			/* :: radio mode on */
                    {
#endif
                      if(r <= BHCutHsml)
                        {
#endif
                          if(swk < MinW)
                            MinW = swk * mass_j;
                          if(swk > MaxW)
                            MaxW = swk * mass_j;
                          AvgW  += swk * mass_j;
#ifdef LT_BH_CUT_KERNEL
                        }
#ifdef LT_DF_BH_ONLY_RADIOMODE_KERNEL
                    }
#endif
#endif
                  CumM  += mass_j;
                  CumCM += SphP[j].x * mass_j;

#ifdef LT_STELLAREVOLUTION                  
                  AvgZ += get_metallicity(j, -1) * mass_j;
#endif
                  
                  ne_guess = 1.0;
#ifdef COOLING
                  temperature = convert_u_to_temp(DMAX(All.MinEgySpec,
                                                       SphP[j].Entropy / GAMMA_MINUS1 * pow(SphP[j].d.Density * a3inv, GAMMA_MINUS1)) *
                                                  All.UnitPressure_in_cgs / All.UnitDensity_in_cgs,
                                                  SphP[j].d.Density * All.UnitDensity_in_cgs * All.HubbleParam *
                                                  All.HubbleParam * a3inv, &ne_guess);
#else
                  Temperature = SphP[j].Entropy / GAMMA_MINUS1 * pow(SphP[j].d.Density * a3inv, GAMMA_MINUS1) *
                    All.UnitPressure_in_cgs / All.UnitDensity_in_cgs,
                    SphP[j].d.Density * All.UnitDensity_in_cgs * All.HubbleParam *
                    All.HubbleParam * a3inv;   /* internal energy */
                  double mu, yhelium = (1 - XH) / (4 * XH);
                  mu = (1 + 4 * yhelium) / (1 + yhelium + *ne_guess);
                  Temperature *= GAMMA_MINUS1 / BOLTZMANN * PROTONMASS * mu;
#endif
                  AvgTemp += temperature * mass_j;
                  
                  AvgRho  += SphP[j].d.Density * mass_j;
#endif /* LT_BH_LOG */
                      
		  DensityNorm += mass_j * wk;
                }
	    }
	}
      if(mode == 1)
	{
	  listindex++;
	  if(listindex < NODELISTLENGTH)
	    {
	      startnode = SmoothDataGet[target].NodeList[listindex];
	      if(startnode >= 0)
		startnode = Nodes[startnode].u.d.nextnode;	/* open it */
	    }
	}
    }


  if(mode == 0)
    {
#if defined(LT_STELLAREVOLUTION) && !defined(LT_DONTUSE_DENSITY_in_WEIGHT)
      if(Type == 4)
	MetP[P[target].MetID].weight = DensityNorm;

      if(Type == 0)   /* protect to write into SphP with traget not to be a gas particle ! */
	{
#endif

#if defined(SMOOTH_PHI)
          SphP[target].SmoothPhi = SmoothPhi;
#endif /* SMOOTH_PHI */
#if defined(VECT_POTENTIAL)
          SphP[target].SmoothA[0] = smoothA[0];
          SphP[target].SmoothA[1] = smoothA[1];
          SphP[target].SmoothA[2] = smoothA[2];
#endif
#if defined(DIVBCLEANING_DEDNER) || defined(SCAL_PRO_CLEAN)
          SphP[target].SmoothDivB = SmoothDivB;
#endif
#ifdef SMOOTH_ROTB
          SphP[target].SmoothedRotB[0] = smoothrotb[0];
          SphP[target].SmoothedRotB[1] = smoothrotb[1];
          SphP[target].SmoothedRotB[2] = smoothrotb[2];
#endif /* SMOOTH_ROTB */

#if defined(BSMOOTH)
          SphP[target].BSmooth[0] = BSmooth[0];
          SphP[target].BSmooth[1] = BSmooth[1];
          SphP[target].BSmooth[2] = BSmooth[2];
#endif /* BSMOOTH */

#if defined(LT_SMOOTH_Z)
          SphP[target].Zsmooth = (FLOAT) Zrho;
#if defined(LT_SMOOTH_SIZE) || defined(LT_SMOOTH_NGB)
          SphP[target].SmoothDens = (float) SmoothDens;
          SphP[target].SmoothNgb = SmoothNgb;
#endif
#if defined(LT_SMOOTH_Z_DETAILS)
          SphP[target].Zsmooth_a = (FLOAT) Zrho_a;
          SphP[target].Zsmooth_b = (FLOAT) Zrho_b;
          SphP[target].SmoothDens_b = (float) SmoothDens_b;
#endif          
#endif /* LT_SMOOTH_Z */

#if defined(LT_SMOOTH_XCLD)
          SphP[target].XCLDsmooth = (float) XCLDsmooth;
#endif /* LT_SMOOTH_XCLD */

#if defined(LT_TRACK_WINDS)
          SphP[target].AvgHsml = (float) AvgHsml;
#endif /* LT_TRACK_WINDS */

          SphP[target].DensityNorm = DensityNorm;

#if defined(LT_STELLAREVOLUTION) && !defined(LT_DONTUSE_DENSITY_in_WEIGHT)
        }
#endif

#ifdef LT_BH
      if(Type == 5)   /* protect to write into SphP with traget not to be a gas particle ! */
        {
          P[target].b9.dBH_AltDensity = AltRho;
#ifdef LT_BH_LOG
          P[target].MinW              = MinW;
          P[target].MaxW              = MaxW;
          P[target].AvgW              = AvgW;
          P[target].CumM              = CumM;
          P[target].CumCM             = CumCM;
          P[target].AvgZ              = AvgZ;
          P[target].AvgTemp           = AvgTemp;
          P[target].AvgRho            = AvgRho;
          P[target].AvgDist           = AvgDist;
          P[target].MinDist           = MinDist;
#endif
	}
#endif
    }
  else
    {
#if defined(SMOOTH_PHI)
      SmoothDataResult[target].SmoothPhi = SmoothPhi;
#endif /* SMOOTH_PHI */
#if defined(VECT_POTENTIAL)
      SmoothDataResult[target].SmoothA[0] = smoothA[0];
      SmoothDataResult[target].SmoothA[1] = smoothA[1];
      SmoothDataResult[target].SmoothA[2] = smoothA[2];
#endif
#if defined(DIVBCLEANING_DEDNER) || defined(SCAL_PRO_CLEAN)
      SmoothDataResult[target].SmoothDivB = SmoothDivB;
#endif
#ifdef SMOOTH_ROTB
      SmoothDataResult[target].SmoothRotB[0] = smoothrotb[0];
      SmoothDataResult[target].SmoothRotB[1] = smoothrotb[1];
      SmoothDataResult[target].SmoothRotB[2] = smoothrotb[2];
#endif /* SMOOTH_ROTB */

#if defined(BSMOOTH)
      SmoothDataResult[target].BSmooth[0] = BSmooth[0];
      SmoothDataResult[target].BSmooth[1] = BSmooth[1];
      SmoothDataResult[target].BSmooth[2] = BSmooth[2];
#endif /* BSMOOTH */

#if defined(LT_SMOOTH_Z)
      SmoothDataResult[target].Zsmooth = (FLOAT) Zrho;
#if defined(LT_SMOOTH_SIZE) || defined(LT_SMOOTH_NGB)
      SmoothDataResult[target].SmoothDens = (float) SmoothDens;
      SmoothDataResult[target].SmoothNgb = SmoothNgb;
#endif
#if defined(LT_SMOOTH_Z_DETAILS)
      SmoothDataResult[target].Zsmooth_a = (FLOAT) Zrho_a;
      SmoothDataResult[target].Zsmooth_b = (FLOAT) Zrho_b;
      SmoothDataResult[target].SmoothDens_b = (float) SmoothDens_b;
#endif
#endif /* LT_SMOOTH_Z */

#if defined(LT_SMOOTH_XCLD)
      SmoothDataResult[target].XCLDsmooth = (float) XCLDsmooth;
#endif /* LT_SMOOTH_XCLD */

#if defined(LT_TRACK_WINDS)
      SmoothDataResult[target].AvgHsml = (float) AvgHsml;
#endif /* LT_TRACK_WINDS */
#ifdef LT_BH
      SmoothDataResult[target].dBH_AltDensity = AltRho;
#endif
#ifdef LT_BH_LOG
      SmoothDataResult[target].MinW           = MinW;
      SmoothDataResult[target].MaxW           = MaxW;
      SmoothDataResult[target].AvgW           = AvgW;
      SmoothDataResult[target].CumM           = CumM;
      SmoothDataResult[target].CumCM          = CumCM;
      SmoothDataResult[target].AvgZ           = AvgZ;
      SmoothDataResult[target].AvgTemp        = AvgTemp;
      SmoothDataResult[target].AvgRho         = AvgRho;
      SmoothDataResult[target].AvgDist        = AvgDist;
      SmoothDataResult[target].MinDist        = MinDist;
#endif
      SmoothDataResult[target].DensityNorm = DensityNorm;
      
    }

  return 0;
}


#if defined(LT_STELLAREVOLUTION)

int smooth_isactive(int i)
{
  if(P[i].TimeBin < 0)
    return 0;

#if (defined(DIVBCLEANING_DEDNER) || defined(SMOOTH_ROTB) || defined(BSMOOTH) || defined(SCAL_PRO_CLEAN)) || defined(VECT_POTENTIAL) || defined(LT_SMOOTH_Z) || defined(LT_SMOOTH_XCLD) || defined(LT_TRACK_WINDS)
  if(P[i].Type == 0)
    return 1;
#endif

#if defined(LT_STELLAREVOLUTION) && !defined(LT_DONTUSE_DENSITY_in_WEIGHT)
  if((P[i].Type & 15) == 4)
    if(TimeBinActive[MetP[P[i].MetID].ChemTimeBin])
      return 1;
#endif

#ifdef BLACK_HOLES
  if(P[i].Type == 5)
    return 1;
#endif

  return 0;

}
#endif


#endif
