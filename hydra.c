#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <gsl/gsl_math.h>


#include "allvars.h"
#include "proto.h"
#include "densitykernel.h"
#ifdef COSMIC_RAYS
#include "cosmic_rays.h"
#endif
#ifdef MACHNUM
#include "machfinder.h"
#endif

#ifdef JD_DPP
#include "cr_electrons.h"
#endif	

#ifndef DEBUG
#define NDEBUG
#endif
#include <assert.h>

#ifdef NUM_THREADS
#include <pthread.h>
#endif

#ifdef NUM_THREADS
extern pthread_mutex_t mutex_nexport;
extern pthread_mutex_t mutex_partnodedrift;

#define LOCK_NEXPORT     pthread_mutex_lock(&mutex_nexport);
#define UNLOCK_NEXPORT   pthread_mutex_unlock(&mutex_nexport);
#else
#define LOCK_NEXPORT
#define UNLOCK_NEXPORT
#endif

#if defined(MAGNETIC) && defined(SFR)
#define POW_CC 1./3.
#endif

extern int NextParticle;
extern int Nexport, Nimport;
extern int BufferFullFlag;
extern int NextJ;
extern int TimerFlag;

/*! \file hydra.c
*  \brief Computation of SPH forces and rate of entropy generation
*
*  This file contains the "second SPH loop", where the SPH forces are
*  computed, and where the rate of change of entropy due to the shock heating
*  (via artificial viscosity) is computed.
*/


struct hydrodata_in
{
#ifdef DENSITY_INDEPENDENT_SPH
      MyFloat EgyRho;
      MyFloat EntVarPred;
#endif

  MyDouble Pos[3];
  MyFloat Vel[3];
  MyFloat Hsml;
  MyFloat Mass;
  MyFloat Density;
  MyFloat Pressure;
  MyFloat F1;
  MyFloat DhsmlDensityFactor;
  int Timestep;

#ifdef PARTICLE_DEBUG
  MyIDType ID;			/*!< particle identifier */
#endif

#ifdef JD_VTURB
  MyFloat Vbulk[3];
#endif

#ifdef MAGNETIC
  MyFloat BPred[3];
#ifdef VECT_POTENTIAL
  MyFloat Apred[3];
#endif
#ifdef ALFA_OMEGA_DYN
  MyFloat alfaomega;
#endif
#ifdef EULER_DISSIPATION
  MyFloat EulerA, EulerB;
#endif
#ifdef TIME_DEP_MAGN_DISP
  MyFloat Balpha;
#endif
#ifdef DIVBCLEANING_DEDNER
  MyFloat PhiPred;
#endif
#if defined(MAGNETIC_DIFFUSION) || defined(ROT_IN_MAG_DIS)
  MyFloat RotB[3];
#endif
#endif
#ifdef TIME_DEP_ART_VISC
  MyFloat alpha;
#endif

#if defined(NAVIERSTOKES)
  MyFloat Entropy;
#endif



#ifdef NAVIERSTOKES
  MyFloat stressoffdiag[3];
  MyFloat stressdiag[3];
  MyFloat shear_viscosity;
#endif

#ifdef NAVIERSTOKES_BULK
  MyFloat divvel;
#endif

#ifdef EOS_DEGENERATE
  MyFloat dpdr;
#endif

#ifndef DONOTUSENODELIST
  int NodeList[NODELISTLENGTH];
#endif
}
 *HydroDataIn, *HydroDataGet;


struct hydrodata_out
{
  MyLongDouble Acc[3];
  MyLongDouble DtEntropy;
#ifdef ALTERNATIVE_VISCOUS_TIMESTEP
  MyFloat MinViscousDt;
#else
  MyFloat MaxSignalVel;
#endif
#ifdef JD_VTURB
  MyFloat Vrms;
#endif
#if defined(MAGNETIC) && (!defined(EULERPOTENTIALS) || !defined(VECT_POTENTIAL))
  MyFloat DtB[3];
#ifdef DIVBFORCE3
  MyFloat magacc[3];
  MyFloat magcorr[3];
#endif
#ifdef DIVBCLEANING_DEDNER
  MyFloat GradPhi[3];
#endif
#endif
#if defined(EULERPOTENTIALS) && defined(EULER_DISSIPATION)
  MyFloat DtEulerA, DtEulerB;
#endif
#ifdef VECT_POTENTIAL
  MyFloat dta[3];
#endif
#if  defined(CR_SHOCK)
  MyFloat CR_EnergyChange[NUMCRPOP];
  MyFloat CR_BaryonFractionChange[NUMCRPOP];
#endif

#ifdef HYDRO_COST_FACTOR
  int Ninteractions;
#endif
}
 *HydroDataResult, *HydroDataOut;



#ifdef MACHNUM
double hubble_a, atime, hubble_a2, fac_mu, fac_vsic_fix, a3inv, fac_egy;
#else
static double hubble_a, atime, hubble_a2, fac_mu, fac_vsic_fix, a3inv, fac_egy;
#endif

/*! This function is the driver routine for the calculation of hydrodynamical
*  force and rate of change of entropy due to shock heating for all active
*  particles .
*/
void hydro_force(void)
{
  int i, j, k, ngrp, ndone, ndone_flag;
  int sendTask, recvTask, place;
  double soundspeed_i;
  double timeall = 0, timecomp1 = 0, timecomp2 = 0, timecommsumm1 = 0, timecommsumm2 = 0, timewait1 =
    0, timewait2 = 0, timenetwork = 0;
  double timecomp, timecomm, timewait, tstart, tend, t0, t1;

  int save_NextParticle;

  long long n_exported = 0;

#ifdef NAVIERSTOKES
  double fac;
#endif

#if (!defined(COOLING) && !defined(CR_SHOCK) && (defined(CR_DISSIPATION) || defined(CR_THERMALIZATION)))
  double utherm;
  double dt;
  int CRpop;
#endif

#if defined(CR_SHOCK)
  double rShockEnergy;
  double rNonRethermalizedEnergy;

#ifndef COOLING
  double utherm, CRpop;
#endif
#endif

#ifdef WINDS
  double windspeed, hsml_c;

#endif


#ifdef TIME_DEP_ART_VISC
  double f, cs_h;
#endif
#if defined(MAGNETIC) && defined(MAGFORCE)
#if defined(TIME_DEP_MAGN_DISP) || defined(DIVBCLEANING_DEDNER)
  double mu0 = 1;
#endif
#endif

#if defined(DIVBCLEANING_DEDNER) || defined(DIVBFORCE3)
  double phiphi, tmpb;
#endif
#if defined(HEALPIX)
  double r_new, t[3];
  long ipix;
  int count = 0;
  int count2 = 0;
  int total_count = 0;
  double ded_heal_fac = 0;
#endif

#ifdef NUCLEAR_NETWORK
  double dedt_nuc;
  int nuc_particles = 0;
  int nuc_particles_sum;
#endif

#ifdef WAKEUP
  for(i = 0; i < NumPart; i++)
    {
      if(P[i].Type == 0)
	SphP[i].wakeup = 0;
    }
#endif

  if(All.ComovingIntegrationOn)
    {
      /* Factors for comoving integration of hydro */
      hubble_a = hubble_function(All.Time);
      hubble_a2 = All.Time * All.Time * hubble_a;

      fac_mu = pow(All.Time, 3 * (GAMMA - 1) / 2) / All.Time;

      fac_egy = pow(All.Time, 3 * (GAMMA - 1));

      fac_vsic_fix = hubble_a * pow(All.Time, 3 * GAMMA_MINUS1);

      a3inv = 1 / (All.Time * All.Time * All.Time);
      atime = All.Time;
    }
  else
    hubble_a = hubble_a2 = atime = fac_mu = fac_vsic_fix = a3inv = fac_egy = 1.0;

#if defined(MAGFORCE) && defined(TIME_DEP_MAGN_DISP) || defined(DIVBCLEANING_DEDNER)
#ifndef MU0_UNITY
  mu0 *= (4 * M_PI);
  mu0 /= All.UnitTime_in_s * All.UnitTime_in_s * All.UnitLength_in_cm / All.UnitMass_in_g;
  if(All.ComovingIntegrationOn)
    mu0 /= (All.HubbleParam * All.HubbleParam);
#endif
#endif


  /* allocate buffers to arrange communication */

  int NTaskTimesNumPart;

  NTaskTimesNumPart = NumPart;
#ifdef NUM_THREADS
  NTaskTimesNumPart = NUM_THREADS * NumPart;
#endif

  Ngblist = (int *) mymalloc("Ngblist", NTaskTimesNumPart * sizeof(int));

  All.BunchSize =
    (int) ((All.BufferSize * 1024 * 1024) / (sizeof(struct data_index) + sizeof(struct data_nodelist) +
					     sizeof(struct hydrodata_in) +
					     sizeof(struct hydrodata_out) +
					     sizemax(sizeof(struct hydrodata_in),
						     sizeof(struct hydrodata_out))));
  DataIndexTable =
    (struct data_index *) mymalloc("DataIndexTable", All.BunchSize * sizeof(struct data_index));
  DataNodeList =
    (struct data_nodelist *) mymalloc("DataNodeList", All.BunchSize * sizeof(struct data_nodelist));


  CPU_Step[CPU_HYDMISC] += measure_time();
  t0 = second();

  NextParticle = FirstActiveParticle;	/* beginn with this index */

  do
    {

      BufferFullFlag = 0;
      Nexport = 0;
      save_NextParticle = NextParticle;

      for(j = 0; j < NTask; j++)
	{
	  Send_count[j] = 0;
	  Exportflag[j] = -1;
	}

      /* do local particles and prepare export list */
      tstart = second();

#ifdef NUM_THREADS
      pthread_t mythreads[NUM_THREADS - 1];
      int threadid[NUM_THREADS - 1];
      pthread_attr_t attr;

      pthread_attr_init(&attr);
      pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
      pthread_mutex_init(&mutex_nexport, NULL);
      pthread_mutex_init(&mutex_partnodedrift, NULL);

      TimerFlag = 0;

      for(j = 0; j < NUM_THREADS - 1; j++)
	{
	  threadid[j] = j + 1;
	  pthread_create(&mythreads[j], &attr, hydro_evaluate_primary, &threadid[j]);
	}
#endif
      int mainthreadid = 0;

      hydro_evaluate_primary(&mainthreadid);	/* do local particles and prepare export list */

#ifdef NUM_THREADS
      for(j = 0; j < NUM_THREADS - 1; j++)
	pthread_join(mythreads[j], NULL);
#endif


      tend = second();
      timecomp1 += timediff(tstart, tend);

      if(BufferFullFlag)
	{
	  int last_nextparticle = NextParticle;

	  NextParticle = save_NextParticle;

	  while(NextParticle >= 0)
	    {
	      if(NextParticle == last_nextparticle)
		break;

	      if(ProcessedFlag[NextParticle] != 1)
		break;

	      ProcessedFlag[NextParticle] = 2;

	      NextParticle = NextActiveParticle[NextParticle];
	    }

	  if(NextParticle == save_NextParticle)
	    {
	      /* in this case, the buffer is too small to process even a single particle */
	      endrun(12998);
	    }


	  int new_export = 0;

	  for(j = 0, k = 0; j < Nexport; j++)
	    if(ProcessedFlag[DataIndexTable[j].Index] != 2)
	      {
		if(k < j + 1)
		  k = j + 1;

		for(; k < Nexport; k++)
		  if(ProcessedFlag[DataIndexTable[k].Index] == 2)
		    {
		      int old_index = DataIndexTable[j].Index;

		      DataIndexTable[j] = DataIndexTable[k];
		      DataNodeList[j] = DataNodeList[k];
		      DataIndexTable[j].IndexGet = j;
		      new_export++;

		      DataIndexTable[k].Index = old_index;
		      k++;
		      break;
		    }
	      }
	    else
	      new_export++;

	  Nexport = new_export;

	}


      n_exported += Nexport;

      for(j = 0; j < NTask; j++)
	Send_count[j] = 0;
      for(j = 0; j < Nexport; j++)
	Send_count[DataIndexTable[j].Task]++;

#ifdef MYSORT
      mysort_dataindex(DataIndexTable, Nexport, sizeof(struct data_index), data_index_compare);
#else
      qsort(DataIndexTable, Nexport, sizeof(struct data_index), data_index_compare);
#endif

      tstart = second();

      MPI_Alltoall(Send_count, 1, MPI_INT, Recv_count, 1, MPI_INT, MPI_COMM_WORLD);

      tend = second();
      timewait1 += timediff(tstart, tend);

      for(j = 0, Nimport = 0, Recv_offset[0] = 0, Send_offset[0] = 0; j < NTask; j++)
	{
	  Nimport += Recv_count[j];

	  if(j > 0)
	    {
	      Send_offset[j] = Send_offset[j - 1] + Send_count[j - 1];
	      Recv_offset[j] = Recv_offset[j - 1] + Recv_count[j - 1];
	    }
	}

      HydroDataGet = (struct hydrodata_in *) mymalloc("HydroDataGet", Nimport * sizeof(struct hydrodata_in));
      HydroDataIn = (struct hydrodata_in *) mymalloc("HydroDataIn", Nexport * sizeof(struct hydrodata_in));

      /* prepare particle data for export */

      for(j = 0; j < Nexport; j++)
	{
	  place = DataIndexTable[j].Index;

	  for(k = 0; k < 3; k++)
	    {
	      HydroDataIn[j].Pos[k] = P[place].Pos[k];
	      HydroDataIn[j].Vel[k] = SphP[place].VelPred[k];
	    }
	  HydroDataIn[j].Hsml = PPP[place].Hsml;
	  HydroDataIn[j].Mass = P[place].Mass;
	  HydroDataIn[j].Density = SphP[place].d.Density;
#ifdef DENSITY_INDEPENDENT_SPH
      HydroDataIn[j].EgyRho = SphP[place].EgyWtDensity;
      HydroDataIn[j].EntVarPred = SphP[place].EntVarPred;
      HydroDataIn[j].DhsmlDensityFactor = SphP[place].DhsmlEgyDensityFactor;
#else
	  HydroDataIn[j].DhsmlDensityFactor = SphP[place].h.DhsmlDensityFactor;
#endif

	  HydroDataIn[j].Pressure = SphP[place].Pressure;
	  HydroDataIn[j].Timestep = (P[place].TimeBin ? (1 << P[place].TimeBin) : 0);
#ifdef EOS_DEGENERATE
	  HydroDataIn[j].dpdr = SphP[place].dpdr;
#endif

	  /* calculation of F1 */
#ifndef ALTVISCOSITY
#ifndef EOS_DEGENERATE
	  soundspeed_i = sqrt(GAMMA * SphP[place].Pressure / SphP[place].EOMDensity);
#else
	  soundspeed_i = sqrt(SphP[place].dpdr);
#endif
#ifndef NAVIERSTOKES
	  HydroDataIn[j].F1 = fabs(SphP[place].v.DivVel) /
	    (fabs(SphP[place].v.DivVel) + SphP[place].r.CurlVel +
	     0.0001 * soundspeed_i / PPP[place].Hsml / fac_mu);
#else
	  HydroDataIn[j].F1 = fabs(SphP[place].v.DivVel) /
	    (fabs(SphP[place].v.DivVel) + SphP[place].u.s.CurlVel +
	     0.0001 * soundspeed_i / PPP[place].Hsml / fac_mu);
#endif

#else
	  HydroDataIn[j].F1 = SphP[place].v.DivVel;
#endif

#ifndef DONOTUSENODELIST
	  memcpy(HydroDataIn[j].NodeList,
		 DataNodeList[DataIndexTable[j].IndexGet].NodeList, NODELISTLENGTH * sizeof(int));
#endif

#ifdef JD_VTURB
		HydroDataIn[j].Vbulk[0] = SphP[place].Vbulk[0];
		HydroDataIn[j].Vbulk[1] = SphP[place].Vbulk[1];
		HydroDataIn[j].Vbulk[2] = SphP[place].Vbulk[2];
#endif

#ifdef MAGNETIC
	  for(k = 0; k < 3; k++)
	    {
#ifndef SFR
	      HydroDataIn[j].BPred[k] = SphP[place].BPred[k];
#else
	      HydroDataIn[j].BPred[k] = SphP[place].BPred[k] * pow(1.-SphP[place].XColdCloud,2.*POW_CC);
#endif
#if defined(MAGNETIC_DIFFUSION) || defined(ROT_IN_MAG_DIS)
#ifdef SMOOTH_ROTB
	      HydroDataIn[j].RotB[k] = SphP[place].SmoothedRotB[k];
#else
	      HydroDataIn[j].RotB[k] = SphP[place].RotB[k];
#endif
#ifdef SFR
	      HydroDataIn[j].RotB[k] *= pow(1.-SphP[place].XColdCloud,3.*POW_CC);
#endif
#endif
	    }
#ifdef ALFA_OMEGA_DYN
	  HydroDataIn[j].alfaomega =
	    Sph[place].r.Rot[0] * Sph[place].VelPred[0] + Sph[place].r.Rot[1] * Sph[place].VelPred[1] +
	    Sph[place].r.Rot[2] * Sph[place].VelPred[2];
#endif
#if defined(EULERPOTENTIALS) && defined(EULER_DISSIPATION)
	  HydroDataIn[j].EulerA = SphP[place].EulerA;
	  HydroDataIn[j].EulerB = SphP[place].EulerB;
#endif
#ifdef VECT_POTENTIAL
	  HydroDataIn[j].Apred[0] = SphP[place].APred[0];
	  HydroDataIn[j].Apred[1] = SphP[place].APred[1];
	  HydroDataIn[j].Apred[2] = SphP[place].APred[2];
#endif
#ifdef DIVBCLEANING_DEDNER
#ifdef SMOOTH_PHI
	  HydroDataIn[j].PhiPred = SphP[place].SmoothPhi;
#else
	  HydroDataIn[j].PhiPred = SphP[place].PhiPred;
#endif
#endif
#endif


#if defined(NAVIERSTOKES)
	  HydroDataIn[j].Entropy = SphP[place].Entropy;
#endif


#ifdef TIME_DEP_ART_VISC
	  HydroDataIn[j].alpha = SphP[place].alpha;
#endif


#ifdef PARTICLE_DEBUG
	  HydroDataIn[j].ID = P[place].ID;
#endif

#ifdef NAVIERSTOKES
	  for(k = 0; k < 3; k++)
	    {
	      HydroDataIn[j].stressdiag[k] = SphP[i].u.s.StressDiag[k];
	      HydroDataIn[j].stressoffdiag[k] = SphP[i].u.s.StressOffDiag[k];
	    }
	  HydroDataIn[j].shear_viscosity = get_shear_viscosity(i);

#ifdef NAVIERSTOKES_BULK
	  HydroDataIn[j].divvel = SphP[i].u.s.DivVel;
#endif
#endif

#ifdef TIME_DEP_MAGN_DISP
	  HydroDataIn[j].Balpha = SphP[place].Balpha;
#endif
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
		  MPI_Sendrecv(&HydroDataIn[Send_offset[recvTask]],
			       Send_count[recvTask] * sizeof(struct hydrodata_in), MPI_BYTE,
			       recvTask, TAG_HYDRO_A,
			       &HydroDataGet[Recv_offset[recvTask]],
			       Recv_count[recvTask] * sizeof(struct hydrodata_in), MPI_BYTE,
			       recvTask, TAG_HYDRO_A, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}
	    }
	}
      tend = second();
      timecommsumm1 += timediff(tstart, tend);


      myfree(HydroDataIn);
      HydroDataResult =
	(struct hydrodata_out *) mymalloc("HydroDataResult", Nimport * sizeof(struct hydrodata_out));
      HydroDataOut =
	(struct hydrodata_out *) mymalloc("HydroDataOut", Nexport * sizeof(struct hydrodata_out));


      report_memory_usage(&HighMark_sphhydro, "SPH_HYDRO");

      /* now do the particles that were sent to us */

      tstart = second();

      NextJ = 0;

#ifdef NUM_THREADS
      for(j = 0; j < NUM_THREADS - 1; j++)
	pthread_create(&mythreads[j], &attr, hydro_evaluate_secondary, &threadid[j]);
#endif
      hydro_evaluate_secondary(&mainthreadid);

#ifdef NUM_THREADS
      for(j = 0; j < NUM_THREADS - 1; j++)
	pthread_join(mythreads[j], NULL);

      pthread_mutex_destroy(&mutex_partnodedrift);
      pthread_mutex_destroy(&mutex_nexport);
      pthread_attr_destroy(&attr);
#endif

      tend = second();
      timecomp2 += timediff(tstart, tend);

      if(NextParticle < 0)
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
		  MPI_Sendrecv(&HydroDataResult[Recv_offset[recvTask]],
			       Recv_count[recvTask] * sizeof(struct hydrodata_out),
			       MPI_BYTE, recvTask, TAG_HYDRO_B,
			       &HydroDataOut[Send_offset[recvTask]],
			       Send_count[recvTask] * sizeof(struct hydrodata_out),
			       MPI_BYTE, recvTask, TAG_HYDRO_B, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}
	    }
	}
      tend = second();
      timecommsumm2 += timediff(tstart, tend);



      /* add the result to the local particles */
      tstart = second();
      for(j = 0; j < Nexport; j++)
	{
	  place = DataIndexTable[j].Index;

	  for(k = 0; k < 3; k++)
	    {
	      SphP[place].a.dHydroAccel[k] += HydroDataOut[j].Acc[k];
	    }

	  SphP[place].e.dDtEntropy += HydroDataOut[j].DtEntropy;

#ifdef HYDRO_COST_FACTOR
	  if(All.ComovingIntegrationOn)
	    P[place].GravCost += HYDRO_COST_FACTOR * All.Time * HydroDataOut[j].Ninteractions;
	  else
	    P[place].GravCost += HYDRO_COST_FACTOR * HydroDataOut[j].Ninteractions;
#endif

#ifdef ALTERNATIVE_VISCOUS_TIMESTEP
	  if(SphP[place].MinViscousDt > HydroDataOut[j].MinViscousDt)
	    SphP[place].MinViscousDt = HydroDataOut[j].MinViscousDt;
#else
	  if(SphP[place].MaxSignalVel < HydroDataOut[j].MaxSignalVel)
	    SphP[place].MaxSignalVel = HydroDataOut[j].MaxSignalVel;
#endif

#ifdef JD_VTURB
	  SphP[place].Vrms += HydroDataOut[j].Vrms; 
#endif

#if defined(MAGNETIC) && ( !defined(EULERPOTENTIALS) || !defined(VECT_POTENTIAL) )
	  for(k = 0; k < 3; k++)
	    SphP[place].DtB[k] += HydroDataOut[j].DtB[k];
#endif
#ifdef DIVBFORCE3
	  for(k = 0; k < 3; k++)
	    SphP[place].magacc[k] += HydroDataOut[j].magacc[k];
	  for(k = 0; k < 3; k++)
	    SphP[place].magcorr[k] += HydroDataOut[j].magcorr[k];
#endif
#ifdef DIVBCLEANING_DEDNER
	  for(k = 0; k < 3; k++)
	    SphP[place].GradPhi[k] += HydroDataOut[j].GradPhi[k];
#endif
#if VECT_POTENTIAL
	  for(k = 0; k < 3; k++)
	    SphP[place].DtA[k] += HydroDataOut[j].dta[k];
#endif
#if defined(EULERPOTENTIALS) && defined(EULER_DISSIPATION)
	  SphP[place].DtEulerA += HydroDataOut[j].DtEulerA;
	  SphP[place].DtEulerB += HydroDataOut[j].DtEulerB;
#endif
	}
      tend = second();
      timecomp1 += timediff(tstart, tend);

      myfree(HydroDataOut);
      myfree(HydroDataResult);
      myfree(HydroDataGet);
    }
  while(ndone < NTask);


  myfree(DataNodeList);
  myfree(DataIndexTable);

  myfree(Ngblist);


  /* do final operations on results */


#ifdef FLTROUNDOFFREDUCTION
  for(i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i])
    if(P[i].Type == 0)
      {
	SphP[i].e.DtEntropy = FLT(SphP[i].e.dDtEntropy);

	for(j = 0; j < 3; j++)
	  SphP[i].a.HydroAccel[j] = FLT(SphP[i].a.dHydroAccel[j]);
      }
#endif



  for(i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i])
    if(P[i].Type == 0)
      {
#ifdef CR_SHOCK
	/* state right here:
	 *
	 * _c denotes comoving quantities
	 * _p denotes physical quantities
	 *
	 *
	 * Delta u_p = rho_p^(gamma-1)/(gamma-1) Delta A
	 *
	 * Delta A = dA/dloga * Delta loga
	 *
	 * dA/dloga = DtE * (gamma-1) / ( H(a) a^2 rho_c^(gamma-1)
	 *
	 * => Delta u_p = DtE * dloga / ( H(a) a^2 a^(3(gamma-1)) )
	 */

	if(SphP[i].e.DtEntropy > 0.0)
	  {
	    rShockEnergy = SphP[i].e.DtEntropy *
	      (P[i].TimeBin ? (1 << P[i].TimeBin) : 0) * All.Timebase_interval / hubble_a2 / fac_egy;
	  }
	else
	  {
	    rShockEnergy = 0.0;
	  }

#endif /* CR_SHOCK */

#ifndef EOS_DEGENERATE

#ifndef TRADITIONAL_SPH_FORMULATION
	/* Translate energy change rate into entropy change rate */
    SphP[i].e.DtEntropy *= GAMMA_MINUS1 / (hubble_a2 * pow(SphP[i].EOMDensity, GAMMA_MINUS1));
#endif

#else
	/* DtEntropy stores the energy change rate in internal units */
	SphP[i].e.DtEntropy *= All.UnitEnergy_in_cgs / All.UnitTime_in_s;
#endif

#ifdef MACHNUM

	/* Estimates the Mach number of particle i for non-radiative runs,
	 * or the Mach number, density jump and specific energy jump
	 * in case of cosmic rays!
	 */
#if (CR_SHOCK == 2)
	GetMachNumberCR(SphP + i);
#else

	GetMachNumber(SphP + i);
#endif /* COSMIC_RAYS */
#endif /* MACHNUM */
#ifdef MACHSTATISTIC
	GetShock_DtEnergy(SphP + i);
#endif

#ifdef CR_SHOCK
	if(rShockEnergy > 0.0)
	  {
	    /* Feed fraction "All.CR_ShockEfficiency" into CR and see what
	     * amount of energy instantly gets rethermalized
	     *
	     * for this, we need the physical time step, which is
	     * Delta t_p = Delta t_c / hubble_a
	     */

	    /* The  CR_find_alpha_InjectTo induces an error in the density jump since it can set
	     *  Particle->Shock_DensityJump = 1.0 + 1.0e-6 which is used in ShockInject as the input DensityJump
	     *  if (NUMCRPOP > 1)
	     *  {
	     *  #if ( CR_SHOCK == 1 )
	     *  InjPopulation = CR_Find_Alpha_to_InjectTo(All.CR_ShockAlpha);
	     *  #else
	     *  InjPopulation = CR_find_alpha_InjectTo(SphP + i);
	     *  #endif
	     *  }
	     *  else
	     *  InjPopulation = 0;
	     */

	    rNonRethermalizedEnergy =
	      CR_Particle_ShockInject(SphP + i,
				      rShockEnergy,
				      (P[i].TimeBin ? (1 << P[i].TimeBin) : 0) * All.Timebase_interval /
				      hubble_a);

	    /* Fraction of total energy that went and remained in CR is
	     * rNonRethermalizedEnergy / rShockEnergy,
	     * hence, we conserve energy if we do:
	     */
#ifndef CR_NO_CHANGE
	    SphP[i].e.DtEntropy *= (1.0 - rNonRethermalizedEnergy / rShockEnergy);
#endif /* CR_NO_CHANGE */

	    assert(rNonRethermalizedEnergy >= 0.0);

	    assert(rNonRethermalizedEnergy <= (rShockEnergy * All.CR_ShockEfficiency));


#if (!defined(COOLING) && (defined(CR_DISSIPATION) || defined(CR_THERMALIZATION)))
	    utherm = 0.0;
	    for(CRpop = 0; CRpop < NUMCRPOP; CRpop++)
	      utherm +=
		CR_Particle_ThermalizeAndDissipate(SphP + i,
						   (P[i].TimeBin ? (1 << P[i].TimeBin) : 0) *
						   All.Timebase_interval / hubble_a, CRpop);

	    /* we need to add this thermalized energy to the internal energy */

	    SphP[i].e.DtEntropy += GAMMA_MINUS1 * utherm * fac_egy / pow(SphP[i].d.Density, GAMMA_MINUS1) /
	      ((P[i].TimeBin ? (1 << P[i].TimeBin) : 0) * All.Timebase_interval);
#endif

	  }
#endif /* CR_SHOCK */


#if (!defined(COOLING) && !defined(CR_SHOCK) && (defined(CR_DISSIPATION) || defined(CR_THERMALIZATION)))
	double utherm;
	double dt;
	int CRpop;

	dt = (P[i].TimeBin ? (1 << P[i].TimeBin) : 0) * All.Timebase_interval / hubble_a;

	if(P[i].TimeBin)	/* upon start-up, we need to protect against dt==0 */
	  {
	    if(dt > 0)
	      {
		for(CRpop = 0; CRpop < NUMCRPOP; CRpop++)
		  {
		    utherm = CR_Particle_ThermalizeAndDissipate(SphP + i, dt, CRpop);

		    SphP[i].e.DtEntropy +=
		      GAMMA_MINUS1 * utherm * fac_egy / pow(SphP[i].d.Density,
							    GAMMA_MINUS1) / (dt * hubble_a);
		  }
	      }
	  }
#endif

#ifdef NAVIERSTOKES
	/* sigma_ab * sigma_ab */
	for(k = 0, fac = 0; k < 3; k++)
	  {
	    fac += SphP[i].u.s.StressDiag[k] * SphP[i].u.s.StressDiag[k] +
	      2 * SphP[i].u.s.StressOffDiag[k] * SphP[i].u.s.StressOffDiag[k];
	  }

#ifndef NAVIERSTOKES_CONSTANT	/*entropy increase due to the shear viscosity */
#ifdef NS_TIMESTEP
	SphP[i].ViscEntropyChange = 0.5 * GAMMA_MINUS1 /
	  (hubble_a2 * pow(SphP[i].d.Density, GAMMA_MINUS1)) *
	  get_shear_viscosity(i) / SphP[i].d.Density * fac *
	  pow((SphP[i].Entropy * pow(SphP[i].d.Density * a3inv, GAMMA_MINUS1) / GAMMA_MINUS1), 2.5);

	SphP[i].e.DtEntropy += SphP[i].ViscEntropyChange;
#else
	SphP[i].e.DtEntropy += 0.5 * GAMMA_MINUS1 /
	  (hubble_a2 * pow(SphP[i].d.Density, GAMMA_MINUS1)) *
	  get_shear_viscosity(i) / SphP[i].d.Density * fac *
	  pow((SphP[i].Entropy * pow(SphP[i].d.Density * a3inv, GAMMA_MINUS1) / GAMMA_MINUS1), 2.5);
#endif

#else
	SphP[i].e.DtEntropy += 0.5 * GAMMA_MINUS1 /
	  (hubble_a2 * pow(SphP[i].d.Density, GAMMA_MINUS1)) *
	  get_shear_viscosity(i) / SphP[i].d.Density * fac;

#ifdef NS_TIMESTEP
	SphP[i].ViscEntropyChange = 0.5 * GAMMA_MINUS1 /
	  (hubble_a2 * pow(SphP[i].d.Density, GAMMA_MINUS1)) *
	  get_shear_viscosity(i) / SphP[i].d.Density * fac;
#endif

#endif

#ifdef NAVIERSTOKES_BULK	/*entropy increase due to the bulk viscosity */
	SphP[i].e.DtEntropy += GAMMA_MINUS1 /
	  (hubble_a2 * pow(SphP[i].d.Density, GAMMA_MINUS1)) *
	  All.NavierStokes_BulkViscosity / SphP[i].d.Density * pow(SphP[i].u.s.a4.DivVel, 2);

#ifdef NS_TIMESTEP
	SphP[i].ViscEntropyChange = GAMMA_MINUS1 /
	  (hubble_a2 * pow(SphP[i].d.Density, GAMMA_MINUS1)) *
	  All.NavierStokes_BulkViscosity / SphP[i].d.Density * pow(SphP[i].u.s.a4.DivVel, 2);
#endif

#endif

#endif /* these entropy increases directly follow from the general heat transfer equation */


#ifdef JD_VTURB
	SphP[i].Vrms += (SphP[i].VelPred[0]-SphP[i].Vbulk[0])*(SphP[i].VelPred[0]-SphP[i].Vbulk[0]) 
					  + (SphP[i].VelPred[1]-SphP[i].Vbulk[1])*(SphP[i].VelPred[1]-SphP[i].Vbulk[1]) 
					  + (SphP[i].VelPred[2]-SphP[i].Vbulk[2])*(SphP[i].VelPred[2]-SphP[i].Vbulk[2]);
	SphP[i].Vrms = sqrt(SphP[i].Vrms/SphP[i].TrueNGB);
#endif

#if defined(JD_DPP) && !defined(JD_DPPONSNAPSHOTONLY)
	compute_Dpp(i);
#endif

#if defined(MAGNETIC) && !defined(EULERPOTENTIALS) && !defined(VECT_POTENTIAL)
	/* take care of cosmological dilution */
	if(All.ComovingIntegrationOn)
	  for(k = 0; k < 3; k++)
#ifndef SFR
	    SphP[i].DtB[k] -= 2.0 * SphP[i].BPred[k];
#else
	    SphP[i].DtB[k] -= 2.0 * SphP[i].BPred[k] * pow(1.-SphP[i].XColdCloud,2.*POW_CC);
#endif
#endif

#ifdef WINDS
	/* if we have winds, we decouple particles briefly if delaytime>0 */

	if(SphP[i].DelayTime > 0)
	  {
	    for(k = 0; k < 3; k++)
	      SphP[i].a.HydroAccel[k] = 0;

	    SphP[i].e.DtEntropy = 0;

#ifdef NOWINDTIMESTEPPING
	    SphP[i].MaxSignalVel = 2 * sqrt(GAMMA * SphP[i].Pressure / SphP[i].d.Density);
#else
	    windspeed = sqrt(2 * All.WindEnergyFraction * All.FactorSN *
			     All.EgySpecSN / (1 - All.FactorSN) / All.WindEfficiency) * All.Time;
	    windspeed *= fac_mu;
	    hsml_c = pow(All.WindFreeTravelDensFac * All.PhysDensThresh /
			 (SphP[i].d.Density * a3inv), (1. / 3.));
	    SphP[i].MaxSignalVel = hsml_c * DMAX((2 * windspeed), SphP[i].MaxSignalVel);
#endif
	  }
#endif

#if VECT_POTENTIAL
/*check if SFR cahnge is needed */
	SphP[i].DtA[0] +=
	  (SphP[i].VelPred[1] * SphP[i].BPred[2] -
	   SphP[i].VelPred[2] * SphP[i].BPred[1]) / (atime * atime * hubble_a);
	SphP[i].DtA[1] +=
	  (SphP[i].VelPred[2] * SphP[i].BPred[0] -
	   SphP[i].VelPred[0] * SphP[i].BPred[2]) / (atime * atime * hubble_a);
	SphP[i].DtA[2] +=
	  (SphP[i].VelPred[0] * SphP[i].BPred[1] -
	   SphP[i].VelPred[1] * SphP[i].BPred[0]) / (atime * atime * hubble_a);
	if(All.ComovingIntegrationOn)
	  for(k = 0; k < 3; k++)
	    SphP[i].DtA[k] -= SphP[i].APred[k];

#endif


#if defined(HEALPIX)
	r_new = 0;
  	ded_heal_fac = 1.;
	for(k = 0; k < 3; k++)
	  {
	    t[k] = P[i].Pos[k] - SysState.CenterOfMassComp[0][k];
	    r_new = r_new + t[k] * t[k];
	  }
	r_new = sqrt(r_new);
	vec2pix_nest((long) All.Nside, t, &ipix);
	if(r_new > All.healpixmap[ipix] * HEALPIX)
	  {
	    SphP[i].e.DtEntropy = 0;
	    for(k = 0; k < 3; k++)
	      {
		SphP[i].a.HydroAccel[k] = 0;
		SphP[i].VelPred[k] = 0.0;
		P[i].Vel[k] = 0.0;
	      }
  	    ded_heal_fac = 2.;
	    SphP[i].v.DivVel = 0.0;
	    count++;
	    if(r_new > All.healpixmap[ipix] * HEALPIX * 1.5)
	      {
		count2++;
	      }
	  }
#endif

#ifdef TIME_DEP_ART_VISC
#if !defined(EOS_DEGENERATE)
	cs_h = sqrt(GAMMA * SphP[i].Pressure / SphP[i].d.Density) / PPP[i].Hsml;
#else
	cs_h = sqrt(SphP[i].dpdr) / PPP[i].Hsml;
#endif
	f = fabs(SphP[i].v.DivVel) / (fabs(SphP[i].v.DivVel) + SphP[i].r.CurlVel + 0.0001 * cs_h / fac_mu);
	SphP[i].Dtalpha = -(SphP[i].alpha - All.AlphaMin) * All.DecayTime *
	  0.5 * SphP[i].MaxSignalVel / (PPP[i].Hsml * fac_mu)
	  + f * All.ViscSource * DMAX(0.0, -SphP[i].v.DivVel);
	if(All.ComovingIntegrationOn)
	  SphP[i].Dtalpha /= (hubble_a * All.Time * All.Time);
#endif
#ifdef MAGNETIC
#ifdef TIME_DEP_MAGN_DISP
	SphP[i].DtBalpha = -(SphP[i].Balpha - All.ArtMagDispMin) * All.ArtMagDispTime *
	  0.5 * SphP[i].MaxSignalVel / (PPP[i].Hsml * fac_mu)
#ifndef ROT_IN_MAG_DIS
	  + All.ArtMagDispSource * fabs(SphP[i].divB) / sqrt(mu0 * SphP[i].d.Density);
#else
#ifdef SMOOTH_ROTB
	  + All.ArtMagDispSource / sqrt(mu0 * SphP[i].d.Density) *
	  DMAX(fabs(SphP[i].divB), fabs(sqrt(SphP[i].SmoothedRotB[0] * SphP[i].SmoothedRotB[0] +
					     SphP[i].SmoothedRotB[1] * SphP[i].SmoothedRotB[1] +
					     SphP[i].SmoothedRotB[2] * SphP[i].SmoothedRotB[2])));
#else
	  + All.ArtMagDispSource / sqrt(mu0 * SphP[i].d.Density) *
	  DMAX(fabs(SphP[i].divB), fabs(sqrt(SphP[i].RotB[0] * SphP[i].RotB[0] +
					     SphP[i].RotB[1] * SphP[i].RotB[1] +
					     SphP[i].RotB[2] * SphP[i].RotB[2])));
#endif /* End SMOOTH_ROTB        */
#endif /* End ROT_IN_MAG_DIS     */
#endif /* End TIME_DEP_MAGN_DISP */

#ifdef DIVBFORCE3
	phiphi = sqrt(pow( SphP[i].magcorr[0] 	   , 2.)+ pow( SphP[i].magcorr[1]      ,2.) +pow( SphP[i].magcorr[2] 	  ,2.));
	tmpb =   sqrt(pow( SphP[i].magacc[0] 	   , 2.)+ pow( SphP[i].magacc[1]       ,2.) +pow( SphP[i].magacc[2] 	  ,2.));

	if(phiphi > DIVBFORCE3 * tmpb)
	   for(k = 0; k < 3; k++)
		SphP[i].magcorr[k]*= DIVBFORCE3 * tmpb / phiphi;

	for(k = 0; k < 3; k++)
		SphP[i].a.HydroAccel[k]+=(SphP[i].magacc[k]-SphP[i].magcorr[k]);

#endif

#ifdef DIVBCLEANING_DEDNER
	tmpb = 0.5 * SphP[i].MaxSignalVel;
	phiphi = tmpb * All.DivBcleanHyperbolicSigma * atime
#ifdef HEALPIX
	  / ded_heal_fac 
#endif
#ifdef SFR 
	  * pow(1.-SphP[i].XColdCloud,3.*POW_CC)
#endif
	  * SphP[i].SmoothDivB;
#ifdef SMOOTH_PHI
	phiphi += SphP[i].SmoothPhi *
#else
	phiphi += SphP[i].PhiPred *
#endif
#ifdef HEALPIX
	  ded_heal_fac * 
#endif
#ifdef SFR 
	  pow(1.-SphP[i].XColdCloud,POW_CC) *
#endif
	  All.DivBcleanParabolicSigma / PPP[i].Hsml;
	
	if(All.ComovingIntegrationOn)
	  SphP[i].DtPhi =
#ifdef SMOOTH_PHI
	    - SphP[i].SmoothPhi
#else
	    - SphP[i].PhiPred
#endif
#ifdef SFR 
	  * pow(1.-SphP[i].XColdCloud,POW_CC)
#endif
	  - ( phiphi * tmpb) / (hubble_a * atime);	///carefull with the + or not +
	else
	  SphP[i].DtPhi = (-phiphi * tmpb);
	
	if(All.ComovingIntegrationOn){
		SphP[i].GradPhi[0]*=1/(hubble_a * atime);
		SphP[i].GradPhi[1]*=1/(hubble_a * atime);
		SphP[i].GradPhi[2]*=1/(hubble_a * atime);
	}
	phiphi = sqrt(pow( SphP[i].GradPhi[0] , 2.)+pow( SphP[i].GradPhi[1]  ,2.)+pow( SphP[i].GradPhi[2] ,2.));
	tmpb   = sqrt(pow( SphP[i].DtB[0]      ,2.)+pow( SphP[i].DtB[1]      ,2.)+pow( SphP[i].DtB[2]     ,2.));
	
	if(phiphi > All.DivBcleanQ * tmpb){
		SphP[i].GradPhi[0]*= All.DivBcleanQ * tmpb / phiphi;
		SphP[i].GradPhi[1]*= All.DivBcleanQ * tmpb / phiphi;
		SphP[i].GradPhi[2]*= All.DivBcleanQ * tmpb / phiphi;
	}	
	
	SphP[i].e.DtEntropy += mu0 * (SphP[i].BPred[0] * SphP[i].GradPhi[0] + SphP[i].BPred[1] * SphP[i].GradPhi[1] + SphP[i].BPred[2] * SphP[i].GradPhi[2]) 
#ifdef SFR
				* pow(1.-SphP[i].XColdCloud,3.*POW_CC)
#endif
				* GAMMA_MINUS1 / (hubble_a2 * pow(SphP[i].d.Density, GAMMA_MINUS1));

	SphP[i].DtB[0]+=SphP[i].GradPhi[0];
	SphP[i].DtB[1]+=SphP[i].GradPhi[1];
	SphP[i].DtB[2]+=SphP[i].GradPhi[2];


#endif /* End DEDNER */
#endif /* End Magnetic */

#ifdef SPH_BND_PARTICLES
	if(P[i].ID == 0)
	  {
	    SphP[i].e.DtEntropy = 0;
#ifdef NS_TIMESTEP
	    SphP[i].ViscEntropyChange = 0;
#endif

#ifdef DIVBCLEANING_DEDNER
	    SphP[i].DtPhi = 0;
#endif
#if defined(MAGNETIC) && !defined(EULERPOTENTIALS) && !defined(VECT_POTENTIAL)
	    for(k = 0; k < 3; k++)
	      SphP[i].DtB[k] = 0;
#endif

	    for(k = 0; k < 3; k++)
	      SphP[i].a.HydroAccel[k] = 0;
	  }
#endif
      }

#if defined(HEALPIX)
  MPI_Allreduce(&count, &total_count, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  count = 0;
  MPI_Allreduce(&count2, &count, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  if(total_count > 0)
    {
      if(ThisTask == 0)
	printf(" hey %i (%i) particles where freeezed and limit is %f \n", total_count, count,
	       (float) All.TotN_gas / 1000.0);
      if(total_count * 1000.0 > All.TotN_gas)	/*//for normal resolution ~100 */
	{
	  if(ThisTask == 0)
	    printf(" Next calculation of Healpix\n");
	  healpix_halo_cond(All.healpixmap);

	}
      total_count = 0;
      count2 = 0;
      fflush(stdout);
    }
#endif

#ifdef NUCLEAR_NETWORK
  if(ThisTask == 0)
    {
      printf("Doing nuclear network.\n");
    }
  MPI_Barrier(MPI_COMM_WORLD);

  tstart = second();

  for(i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i])
    if(P[i].Type == 0)
      {
	/* evaluate network here, but do it only for temperatures > 10^7 K */
	if(SphP[i].temp > 1e7)
	  {
	    nuc_particles++;
	    network_integrate(SphP[i].temp, SphP[i].d.Density * All.UnitDensity_in_cgs, SphP[i].xnuc,
			      SphP[i].dxnuc,
			      (P[i].TimeBin ? (1 << P[i].TimeBin) : 0) * All.Timebase_interval *
			      All.UnitTime_in_s, &dedt_nuc);
	    SphP[i].e.DtEntropy += dedt_nuc * All.UnitEnergy_in_cgs / All.UnitTime_in_s;
	  }
	else
	  {
	    for(k = 0; k < EOS_NSPECIES; k++)
	      {
		SphP[i].dxnuc[k] = 0;
	      }
	  }
      }

  tend = second();
  timenetwork += timediff(tstart, tend);

  MPI_Allreduce(&nuc_particles, &nuc_particles_sum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  if(ThisTask == 0)
    {
      printf("Nuclear network done for %d particles.\n", nuc_particles_sum);
    }

  timewait1 += timediff(tend, second());
#endif

#ifdef RT_RAD_PRESSURE
  for(i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i])
    if(P[i].Type == 0)
      {
	if(All.Time != All.TimeBegin)
	  for(j = 0; j < N_BINS; j++)
	    {
	      for(k = 0; k < 3; k++)
		SphP[i].a.HydroAccel[k] += SphP[i].dn_gamma[j] *
		  (HYDROGEN_MASSFRAC * SphP[i].d.Density) / (PROTONMASS / All.UnitMass_in_g *
							     All.HubbleParam) * SphP[i].n[k][j] * 13.6 *
		  1.60184e-12 / All.UnitEnergy_in_cgs * All.HubbleParam / (C / All.UnitVelocity_in_cm_per_s) /
		  ((P[i].TimeBin ? (1 << P[i].TimeBin) : 0) * All.Timebase_interval) / SphP[i].d.Density;
	    }
      }
#endif

  /* collect some timing information */

  t1 = WallclockTime = second();
  timeall += timediff(t0, t1);

  timecomp = timecomp1 + timecomp2;
  timewait = timewait1 + timewait2;
  timecomm = timecommsumm1 + timecommsumm2;

  CPU_Step[CPU_HYDCOMPUTE] += timecomp;
  CPU_Step[CPU_HYDWAIT] += timewait;
  CPU_Step[CPU_HYDCOMM] += timecomm;
  CPU_Step[CPU_HYDNETWORK] += timenetwork;
  CPU_Step[CPU_HYDMISC] += timeall - (timecomp + timewait + timecomm + timenetwork);
}




/*! This function is the 'core' of the SPH force computation. A target
*  particle is specified which may either be local, or reside in the
*  communication buffer.
*/
int hydro_evaluate(int target, int mode, int *exportflag, int *exportnodecount, int *exportindex,
		   int *ngblist)
{
  int startnode, numngb, listindex = 0;
  int j, k, n, timestep;
  MyDouble *pos;
  MyFloat *vel;
  MyFloat mass, dhsmlDensityFactor, rho, pressure, f1, f2;
  MyLongDouble acc[3], dtEntropy;

#ifdef DENSITY_INDEPENDENT_SPH
  double egyrho = 0, entvarpred = 0;
#endif

#ifdef HYDRO_COST_FACTOR
  int ninteractions = 0;
#endif


#ifdef ALTERNATIVE_VISCOUS_TIMESTEP
  MyFloat minViscousDt;
#else
  MyFloat maxSignalVel;
#endif
  double dx, dy, dz, dvx, dvy, dvz;
  double p_over_rho2_i, p_over_rho2_j, soundspeed_i, soundspeed_j;
  double hfc, vdotr, vdotr2, visc, mu_ij, rho_ij, vsig;
  double h_i;
  density_kernel_t kernel_i;
  double h_j;
  density_kernel_t kernel_j;
  double dwk_i, dwk_j;
  double r, r2, u;
  double hfc_visc;

#ifdef TRADITIONAL_SPH_FORMULATION
  double hfc_egy;
#endif

  double BulkVisc_ij;

#ifdef NAVIERSTOKES
  double faci, facj;
  MyFloat *stressdiag;
  MyFloat *stressoffdiag;
  MyFloat shear_viscosity;

#ifdef VISCOSITY_SATURATION
  double VelLengthScale_i, VelLengthScale_j;
  double IonMeanFreePath_i, IonMeanFreePath_j;
#endif
#ifdef NAVIERSTOKES_BULK
  double facbi, facbj;
  MyFloat divvel;
#endif
#endif

#if defined(NAVIERSTOKES)
  double Entropy;
#endif

#ifdef TIME_DEP_ART_VISC
  MyFloat alpha;
#endif

#ifdef ALTVISCOSITY
  double mu_i, mu_j;
#endif

#ifndef NOVISCOSITYLIMITER
  double dt;
#endif

#ifdef JD_VTURB
	MyFloat vRms=0;
	MyFloat vBulk[3]={0};
#endif

#ifdef MAGNETIC
  MyFloat bpred[3];

#ifdef ALFA_OMEGA_DYN
  double alfaomega;
#endif
#if ( !defined(EULERPOTENTIALS) || !defined(VECT_POTENTIAL) )
  double dtB[3];
#endif

  double dBx, dBy, dBz;
  double magfac, magfac_i, magfac_j, magfac_i_base;
  double mu0_1;

#if defined(MAGNETIC_DIFFUSION) || defined(VECT_POTENTIAL)
  double magfac_diff;
#endif

#ifdef MAGFORCE
  double mm_i[3][3], mm_j[3][3];
  double b2_i, b2_j;
  int l;
#endif

#if defined(MAGNETIC_DISSIPATION) || defined(DIVBCLEANING_DEDNER) || defined(EULER_DISSIPATION) || defined(MAGNETIC_DIFFUSION)
  double magfac_sym;
#endif

#ifdef MAGNETIC_DISSIPATION
  double dTu_diss_b, Balpha_ij;

#ifdef MAGDISSIPATION_PERPEN
  double mft, mvt[3];
#endif
#ifdef TIME_DEP_MAGN_DISP
  double Balpha;
#endif
#endif

#ifdef EULER_DISSIPATION
  double eulA, eulB, dTu_diss_eul, alpha_ij_eul;
  double dteulA, dteulB;
#endif
#ifdef DIVBFORCE3
  double magacc[3];
  double magcorr[3];
#endif

#ifdef DIVBCLEANING_DEDNER
  double PhiPred, phifac;
  double gradphi[3];
#endif
#ifdef MAGNETIC_SIGNALVEL
  double magneticspeed_i, magneticspeed_j, vcsa2_i, vcsa2_j, Bpro2_i, Bpro2_j;
#endif
#ifdef VECT_POTENTIAL
  double dta[3];
  double Apred[3];

  dta[0] = dta[1] = dta[2] = 0.0;
#endif
#if defined(MAGNETIC_DIFFUSION) || defined(ROT_IN_MAG_DIS)
  MyFloat rotb[3];
#endif

#endif /* end magnetic */

#ifdef PARTICLE_DEBUG
  MyIDType ID;			/*!< particle identifier */
#endif

#ifdef CONVENTIONAL_VISCOSITY
  double c_ij, h_ij;
#endif

  if(mode == 0)
    {
      pos = P[target].Pos;
      vel = SphP[target].VelPred;
      h_i = PPP[target].Hsml;
      mass = P[target].Mass;
      rho = SphP[target].d.Density;
      pressure = SphP[target].Pressure;
      timestep = (P[target].TimeBin ? (1 << P[target].TimeBin) : 0);

#ifdef DENSITY_INDEPENDENT_SPH
      egyrho = SphP[target].EgyWtDensity;
      entvarpred = SphP[target].EntVarPred;
      dhsmlDensityFactor = SphP[target].DhsmlEgyDensityFactor;
#else
      dhsmlDensityFactor = SphP[target].h.DhsmlDensityFactor;
#endif

#ifndef EOS_DEGENERATE
#ifdef DENSITY_INDEPENDENT_SPH
      soundspeed_i = sqrt(GAMMA * pressure / egyrho);
#else
      soundspeed_i = sqrt(GAMMA * pressure / rho);
#endif
#else
      soundspeed_i = sqrt(SphP[target].dpdr);
#endif

#ifndef ALTVISCOSITY
#ifndef NAVIERSTOKES
      f1 = fabs(SphP[target].v.DivVel) /
	(fabs(SphP[target].v.DivVel) + SphP[target].r.CurlVel +
	 0.0001 * soundspeed_i / PPP[target].Hsml / fac_mu);
#else
      f1 = fabs(SphP[target].v.DivVel) /
	(fabs(SphP[target].v.DivVel) + SphP[target].u.s.CurlVel +
	 0.0001 * soundspeed_i / PPP[target].Hsml / fac_mu);
#endif
#else
      f1 = SphP[target].v.DivVel;
#endif

#ifdef JD_VTURB
		vBulk[0] = SphP[target].Vbulk[0];
		vBulk[1] = SphP[target].Vbulk[1];
		vBulk[2] = SphP[target].Vbulk[2];
#endif

#ifdef MAGNETIC
#ifndef SFR
      bpred[0] = SphP[target].BPred[0];
      bpred[1] = SphP[target].BPred[1];
      bpred[2] = SphP[target].BPred[2];
#else
      bpred[0] = SphP[target].BPred[0] * pow(1.-SphP[target].XColdCloud,2.*POW_CC);
      bpred[1] = SphP[target].BPred[1] * pow(1.-SphP[target].XColdCloud,2.*POW_CC);
      bpred[2] = SphP[target].BPred[2] * pow(1.-SphP[target].XColdCloud,2.*POW_CC);
#endif
#ifdef ALFA_OMEGA_DYN
      alfaomega =
	Sph[target].r.Rot[0] * Sph[target].VelPred[0] + Sph[target].r.Rot[1] * Sph[target].VelPred[1] +
	Sph[target].r.Rot[2] * Sph[target].VelPred[2];
#endif
#ifdef VECT_POTENTIAL
      Apred[0] = SphP[target].APred[0];
      Apred[1] = SphP[target].APred[1];
      Apred[2] = SphP[target].APred[2];
#endif
#ifdef DIVBCLEANING_DEDNER
#ifdef SMOOTH_PHI
      PhiPred = SphP[target].SmoothPhi;
#else
      PhiPred = SphP[target].PhiPred;
#endif
#ifdef SFR
      PhiPred *= pow(1.-SphP[target].XColdCloud,POW_CC);
#endif
#endif
#if defined(MAGNETIC_DIFFUSION) || defined(ROT_IN_MAG_DIS)
#ifdef SMOOTH_ROTB
      rotb[0] = SphP[target].SmoothedRotB[0];
      rotb[1] = SphP[target].SmoothedRotB[1];
      rotb[2] = SphP[target].SmoothedRotB[2];
#else
      rotb[0] = SphP[target].RotB[0];
      rotb[1] = SphP[target].RotB[1];
      rotb[2] = SphP[target].RotB[2];
#endif
#ifdef SFR
      rotb[0] *= pow(1.-SphP[target].XColdCloud,3.*POW_CC);
      rotb[1] *= pow(1.-SphP[target].XColdCloud,3.*POW_CC);
      rotb[2] *= pow(1.-SphP[target].XColdCloud,3.*POW_CC);
#endif
#endif
#ifdef TIME_DEP_MAGN_DISP
      Balpha = SphP[target].Balpha;
#endif
#ifdef EULER_DISSIPATION
      eulA = SphP[target].EulerA;
      eulB = SphP[target].EulerB;
#endif
#endif /*  MAGNETIC  */

#ifdef TIME_DEP_ART_VISC
      alpha = SphP[target].alpha;
#endif

#if defined(NAVIERSTOKES)
      Entropy = SphP[target].Entropy;
#endif



#ifdef PARTICLE_DEBUG
      ID = P[target].ID;
#endif

#ifdef NAVIERSTOKES
      stressdiag = SphP[target].u.s.StressDiag;
      stressoffdiag = SphP[target].u.s.StressOffDiag;
      shear_viscosity = get_shear_viscosity(target);
#ifdef NAVIERSTOKES_BULK
      divvel = SphP[target].u.s.a4.DivVel;
#endif
#endif

    }
  else
    {
      pos = HydroDataGet[target].Pos;
      vel = HydroDataGet[target].Vel;
      h_i = HydroDataGet[target].Hsml;
      mass = HydroDataGet[target].Mass;
      dhsmlDensityFactor = HydroDataGet[target].DhsmlDensityFactor;
#ifdef DENSITY_INDEPENDENT_SPH
      egyrho = HydroDataGet[target].EgyRho;
      entvarpred = HydroDataGet[target].EntVarPred;
#endif
      rho = HydroDataGet[target].Density;
      pressure = HydroDataGet[target].Pressure;
      timestep = HydroDataGet[target].Timestep;
#ifndef EOS_DEGENERATE
#ifdef DENSITY_INDEPENDENT_SPH
      soundspeed_i = sqrt(GAMMA * pressure / egyrho);
#else
      soundspeed_i = sqrt(GAMMA * pressure / rho);
#endif
#else
      soundspeed_i = sqrt(HydroDataGet[target].dpdr);
#endif
      f1 = HydroDataGet[target].F1;

#ifdef JD_VTURB
		vBulk[0] = HydroDataGet[target].Vbulk[0];
		vBulk[1] = HydroDataGet[target].Vbulk[1];
		vBulk[2] = HydroDataGet[target].Vbulk[2];
#endif
#ifdef MAGNETIC
      bpred[0] = HydroDataGet[target].BPred[0];
      bpred[1] = HydroDataGet[target].BPred[1];
      bpred[2] = HydroDataGet[target].BPred[2];
#ifdef ALFA_OMEGA_DYN
      alfaomega = HydroDataGet[target].alfaomega;
#endif
#ifdef VECT_POTENTIAL
      Apred[0] = HydroDataGet[target].Apred[0];
      Apred[1] = HydroDataGet[target].Apred[1];
      Apred[2] = HydroDataGet[target].Apred[2];
#endif
#ifdef DIVBCLEANING_DEDNER
      PhiPred = HydroDataGet[target].PhiPred;
#endif
#if defined(MAGNETIC_DIFFUSION) || defined(ROT_IN_MAG_DIS)
      rotb[0] = HydroDataGet[target].RotB[0];
      rotb[1] = HydroDataGet[target].RotB[1];
      rotb[2] = HydroDataGet[target].RotB[2];
#endif
#ifdef TIME_DEP_MAGN_DISP
      Balpha = HydroDataGet[target].Balpha;
#endif
#ifdef EULER_DISSIPATION
      eulA = HydroDataGet[target].EulerA;
      eulB = HydroDataGet[target].EulerB;
#endif
#endif /* MAGNETIC */

#ifdef TIME_DEP_ART_VISC
      alpha = HydroDataGet[target].alpha;
#endif

#if defined(NAVIERSTOKES)
      Entropy = HydroDataGet[target].Entropy;
#endif


#ifdef PARTICLE_DEBUG
      ID = HydroDataGet[target].ID;
#endif


#ifdef NAVIERSTOKES
      stressdiag = HydroDataGet[target].stressdiag;
      stressoffdiag = HydroDataGet[target].stressoffdiag;
      shear_viscosity = HydroDataGet[target].shear_viscosity;
#endif
#ifdef NAVIERSTOKES
      stressdiag = HydroDataGet[target].stressdiag;
      stressoffdiag = HydroDataGet[target].stressoffdiag;
      shear_viscosity = HydroDataGet[target].shear_viscosity;
#ifdef NAVIERSTOKES_BULK
      divvel = HydroDataGet[target].divvel;
#endif
#endif
    }


  /* initialize variables before SPH loop is started */

  acc[0] = acc[1] = acc[2] = dtEntropy = 0;
  density_kernel_init(&kernel_i, h_i);

#ifdef DIVBFORCE3
  magacc[0]=magacc[1]=magacc[2]=0.0;
  magcorr[0]=magcorr[1]=magcorr[2]=0.0;
#endif


#ifdef MAGNETIC
#if ( !defined(EULERPOTENTIALS) || !defined(VECT_POTENTIAL) )
  for(k = 0; k < 3; k++)
    dtB[k] = 0;
#endif
#ifdef EULER_DISSIPATION
  dteulA = 0;
  dteulB = 0;
#endif
  mu0_1 = 1;
#ifndef MU0_UNITY
  mu0_1 /= (4 * M_PI);
  mu0_1 *= All.UnitTime_in_s * All.UnitTime_in_s * All.UnitLength_in_cm / (All.UnitMass_in_g);
  if(All.ComovingIntegrationOn)
    mu0_1 /= (All.HubbleParam * All.HubbleParam);

#endif
#ifdef DIVBCLEANING_DEDNER
  gradphi[2]= gradphi[1]=  gradphi[0]=0.0;
#endif
#ifdef MAGFORCE
  magfac_i_base = 1 / (rho * rho);
#ifndef MU0_UNITY
  magfac_i_base /= (4 * M_PI);
#endif
#ifdef CORRECTBFRC
  magfac_i_base *= dhsmlDensityFactor;
#endif
  for(k = 0, b2_i = 0; k < 3; k++)
    {
      b2_i += bpred[k] * bpred[k];
      for(l = 0; l < 3; l++)
	mm_i[k][l] = bpred[k] * bpred[l];
    }
  for(k = 0; k < 3; k++)
    mm_i[k][k] -= 0.5 * b2_i;
#ifdef MAGNETIC_SIGNALVEL
#ifdef ALFVEN_VEL_LIMITER
  vcsa2_i = soundspeed_i * soundspeed_i +
    DMIN(mu0_1 * b2_i / rho, ALFVEN_VEL_LIMITER * soundspeed_i * soundspeed_i);
#else
  vcsa2_i = soundspeed_i * soundspeed_i + mu0_1 * b2_i / rho;
#endif
#endif
#endif /* end of MAGFORCE */
#endif /* end of MAGNETIC */

#ifndef TRADITIONAL_SPH_FORMULATION
#ifdef DENSITY_INDEPENDENT_SPH
  p_over_rho2_i = pressure / (egyrho * egyrho);
#else
  p_over_rho2_i = pressure / (rho * rho);
#ifndef NO_DHSML
  p_over_rho2_i *= dhsmlDensityFactor;
#endif
#endif
#else
  p_over_rho2_i = pressure / (rho * rho);
#endif

#ifdef ALTERNATIVE_VISCOUS_TIMESTEP
  minViscousDt = 1.0e32;
#else
  maxSignalVel = soundspeed_i;
#endif


  /* Now start the actual SPH computation for this particle */

  if(mode == 0)
    {
      startnode = All.MaxPart;	/* root node */
    }
  else
    {
#ifndef DONOTUSENODELIST
      startnode = HydroDataGet[target].NodeList[0];
      startnode = Nodes[startnode].u.d.nextnode;	/* open it */
#else
      startnode = All.MaxPart;	/* root node */
#endif
    }

  while(startnode >= 0)
    {
      while(startnode >= 0)
	{
	  numngb =
	    ngb_treefind_pairs_threads(pos, h_i, target, &startnode, mode, exportflag, exportnodecount,
				       exportindex, ngblist);

	  if(numngb < 0)
	    return -1;

	  for(n = 0; n < numngb; n++)
	    {
	      j = ngblist[n];

#ifdef HYDRO_COST_FACTOR
	      ninteractions++;
#endif

#ifdef BLACK_HOLES
	      if(P[j].Mass == 0)
		continue;
#endif

#ifdef NOWINDTIMESTEPPING
#ifdef WINDS
	      if(P[j].Type == 0)
		if(SphP[j].DelayTime > 0)	/* ignore the wind particles */
		  continue;
#endif
#endif
	      dx = pos[0] - P[j].Pos[0];
	      dy = pos[1] - P[j].Pos[1];
	      dz = pos[2] - P[j].Pos[2];
#ifdef PERIODIC			/*  now find the closest image in the given box size  */
	      dx = NEAREST_X(dx);
	      dy = NEAREST_Y(dy);
	      dz = NEAREST_Z(dz);
#endif
	      r2 = dx * dx + dy * dy + dz * dz;
	      h_j = PPP[j].Hsml;
          density_kernel_init(&kernel_j, h_j);
	      if(r2 < kernel_i.HH || r2 < kernel_j.HH)
		{
		  r = sqrt(r2);
		  if(r > 0)
		    {
              p_over_rho2_j = SphP[j].Pressure / (SphP[j].EOMDensity * SphP[j].EOMDensity);

#ifndef EOS_DEGENERATE
#ifdef DENSITY_INDEPENDENT_SPH
              soundspeed_j = sqrt(GAMMA * SphP[j].Pressure / SphP[j].EOMDensity);
#else
              soundspeed_j = sqrt(GAMMA * p_over_rho2_j * SphP[j].d.Density);
#endif
#else
              soundspeed_j = sqrt(SphP[j].dpdr);
#endif

		      dvx = vel[0] - SphP[j].VelPred[0];
		      dvy = vel[1] - SphP[j].VelPred[1];
		      dvz = vel[2] - SphP[j].VelPred[2];
		      vdotr = dx * dvx + dy * dvy + dz * dvz;
		      rho_ij = 0.5 * (rho + SphP[j].d.Density);

		      if(All.ComovingIntegrationOn)
			vdotr2 = vdotr + hubble_a2 * r2;
		      else
			vdotr2 = vdotr;

              dwk_i = density_kernel_dwk(&kernel_i, r * kernel_i.Hinv);
              dwk_j = density_kernel_dwk(&kernel_j, r * kernel_j.Hinv);

#ifdef JD_VTURB
			if ( h_i >= PPP[j].Hsml)  /* Make sure j is inside targets hsml */
				vRms += (SphP[j].VelPred[0]-vBulk[0])*(SphP[j].VelPred[0]-vBulk[0]) 
					  	+ (SphP[j].VelPred[1]-vBulk[1])*(SphP[j].VelPred[1]-vBulk[1]) 
						   + (SphP[j].VelPred[2]-vBulk[2])*(SphP[j].VelPred[2]-vBulk[2]);
#endif

#ifdef MAGNETIC
#ifndef SFR
		      dBx = bpred[0] - SphP[j].BPred[0];
		      dBy = bpred[1] - SphP[j].BPred[1];
		      dBz = bpred[2] - SphP[j].BPred[2];
#else
		      dBx = bpred[0] - SphP[j].BPred[0] * pow(1.-SphP[j].XColdCloud,2.*POW_CC);
		      dBy = bpred[1] - SphP[j].BPred[1] * pow(1.-SphP[j].XColdCloud,2.*POW_CC);
		      dBz = bpred[2] - SphP[j].BPred[2] * pow(1.-SphP[j].XColdCloud,2.*POW_CC);
#endif

		      magfac = P[j].Mass / r;	/* we moved 'dwk_i / rho' down ! */
		      if(All.ComovingIntegrationOn)
			magfac *= 1. / (hubble_a * All.Time * All.Time);
		      /* last factor takes care of all cosmological prefactor */
#ifdef CORRECTDB
		      magfac *= dhsmlDensityFactor;
#endif
#if defined(MAGNETIC_DISSIPATION) || defined(DIVBCLEANING_DEDNER) || defined(EULER_DISSIPATION) || defined(MAGNETIC_DIFFUSION)
		      magfac_sym = magfac * (dwk_i + dwk_j) * 0.5;
#endif
#ifdef MAGNETIC_DISSIPATION
#ifdef TIME_DEP_MAGN_DISP
		      Balpha_ij = 0.5 * (Balpha + SphP[j].Balpha);
#else
		      Balpha_ij = All.ArtMagDispConst;
#endif
#endif
		      magfac *= dwk_i / rho;
#if VECT_POTENTIAL
		      dta[0] +=
			P[j].Mass * dwk_i / r * (Apred[0] -
						 SphP[j].APred[0]) * dx * vel[0] / (rho * atime * atime *
										    hubble_a);
		      dta[1] +=
			P[j].Mass * dwk_i / r * (Apred[1] -
						 SphP[j].APred[1]) * dy * vel[1] / (rho * atime * atime *
										    hubble_a);
		      dta[2] +=
			P[j].Mass * dwk_i / r * (Apred[2] -
						 SphP[j].APred[2]) * dz * vel[2] / (rho * atime * atime *
										    hubble_a);
		      dta[0] +=
			P[j].Mass * dwk_i / r * ((Apred[0] - SphP[j].APred[0]) * dx * vel[0] +
						 (Apred[0] - SphP[j].APred[0]) * dy * vel[1] + (Apred[0] -
												SphP[j].
												APred[0]) *
						 dz * vel[2]) / (rho * atime * atime * hubble_a);

#endif
#if ( !defined(EULERPOTENTIALS) || !defined(VECT_POTENTIAL) )
		      dtB[0] +=
			magfac * ((bpred[0] * dvy - bpred[1] * dvx) * dy +
				  (bpred[0] * dvz - bpred[2] * dvx) * dz);
		      dtB[1] +=
			magfac * ((bpred[1] * dvz - bpred[2] * dvy) * dz +
				  (bpred[1] * dvx - bpred[0] * dvy) * dx);
		      dtB[2] +=
			magfac * ((bpred[2] * dvx - bpred[0] * dvz) * dx +
				  (bpred[2] * dvy - bpred[1] * dvz) * dy);
#endif
#ifdef MAGNETIC_DIFFUSION  
		      magfac_diff = (All.MagneticEta + All.MagneticEta) * magfac_sym / (rho_ij * rho_ij);
		      dtB[0] += magfac_diff * rho * dBx;
		      dtB[1] += magfac_diff * rho * dBy;
		      dtB[2] += magfac_diff * rho * dBz;
#ifdef MAGNETIC_DIFFUSION_HEAT
		      if(All.ComovingIntegrationOn)
			magfac_diff *= (hubble_a * All.Time * All.Time * All.Time * All.Time * All.Time);
		      dtEntropy -= 0.5 * magfac_diff * mu0_1 * (dBx * dBx + dBy * dBy + dBz * dBz);
#endif
#endif
#ifdef MAGFORCE
		      magfac_j = 1 / (SphP[j].d.Density * SphP[j].d.Density);
#ifndef MU0_UNITY
		      magfac_j /= (4 * M_PI);
#endif
#ifdef CORRECTBFRC
		      magfac_j *= dwk_j * SphP[j].h.DhsmlDensityFactor;
		      magfac_i = dwk_i * magfac_i_base;
#else
		      magfac_i = magfac_i_base;
#endif
		      for(k = 0, b2_j = 0; k < 3; k++)
			{
#ifndef SFR
			  b2_j += SphP[j].BPred[k] * SphP[j].BPred[k];
			  for(l = 0; l < 3; l++)
			    mm_j[k][l] = SphP[j].BPred[k] * SphP[j].BPred[l];
#else
			  b2_j += SphP[j].BPred[k] * SphP[j].BPred[k] * pow(1.-SphP[j].XColdCloud,4.*POW_CC);
			  for(l = 0; l < 3; l++)
			    mm_j[k][l] = SphP[j].BPred[k] * SphP[j].BPred[l] * pow(1.-SphP[j].XColdCloud,4.*POW_CC);
#endif
			}
		      for(k = 0; k < 3; k++)
			mm_j[k][k] -= 0.5 * b2_j;

#ifdef DIVBCLEANING_DEDNER
		      phifac = magfac_sym * rho / rho_ij;
#ifndef SFR
#ifdef SMOOTH_PHI
		      phifac *= (PhiPred - SphP[j].SmoothPhi) / (rho_ij);
#else
		      phifac *= (PhiPred - SphP[j].PhiPred) / (rho_ij);
#endif
#else /* SFR */ 
#ifdef SMOOTH_PHI
		      phifac *= (PhiPred - SphP[j].SmoothPhi * pow(1.-SphP[j].XColdCloud,POW_CC)) / (rho_ij);
#else
		      phifac *= (PhiPred - SphP[j].PhiPred   * pow(1.-SphP[j].XColdCloud,POW_CC)) / (rho_ij);
#endif 
#endif /* SFR */
		      
		      gradphi[0]+=phifac *dx;
		      gradphi[1]+=phifac *dy;
		      gradphi[2]+=phifac *dz;
#endif
#ifdef MAGNETIC_SIGNALVEL
#ifdef ALFVEN_VEL_LIMITER
		      vcsa2_j = soundspeed_j * soundspeed_j +
			DMIN(mu0_1 * b2_j / SphP[j].d.Density,
			     ALFVEN_VEL_LIMITER * soundspeed_j * soundspeed_j);
#else
		      vcsa2_j = soundspeed_j * soundspeed_j + mu0_1 * b2_j / SphP[j].d.Density;
#endif
#ifndef SFR
		      Bpro2_j = (SphP[j].BPred[0] * dx + SphP[j].BPred[1] * dy + SphP[j].BPred[2] * dz) / r;
#else
		      Bpro2_j = (SphP[j].BPred[0] * dx + SphP[j].BPred[1] * dy + SphP[j].BPred[2] * dz) * pow(1.-SphP[j].XColdCloud,2.*POW_CC) / r;
#endif
		      Bpro2_j *= Bpro2_j;
		      magneticspeed_j = sqrt(vcsa2_j +
					     sqrt(DMAX((vcsa2_j * vcsa2_j -
							4 * soundspeed_j * soundspeed_j * Bpro2_j
							* mu0_1 / SphP[j].d.Density), 0))) / 1.4142136;
		      Bpro2_i = (bpred[0] * dx + bpred[1] * dy + bpred[2] * dz) / r;
		      Bpro2_i *= Bpro2_i;
		      magneticspeed_i = sqrt(vcsa2_i +
					     sqrt(DMAX((vcsa2_i * vcsa2_i -
							4 * soundspeed_i * soundspeed_i * Bpro2_i
							* mu0_1 / rho), 0))) / 1.4142136;
#endif
#ifdef MAGNETIC_DISSIPATION
		      dTu_diss_b = -magfac_sym * Balpha_ij * (dBx * dBx + dBy * dBy + dBz * dBz);
#endif
#ifdef CORRECTBFRC
		      magfac = P[j].Mass / r;
#else
		      magfac = P[j].Mass * 0.5 * (dwk_i + dwk_j) / r;
#endif
		      if(All.ComovingIntegrationOn)
			magfac *= pow(All.Time, 3 * GAMMA);
		      /* last factor takes care of all cosmological prefactor */
#ifndef MU0_UNITY
		      magfac *= All.UnitTime_in_s * All.UnitTime_in_s *
			All.UnitLength_in_cm / All.UnitMass_in_g;
		      if(All.ComovingIntegrationOn)
			magfac /= (All.HubbleParam * All.HubbleParam);
		      /* take care of B unit conversion into GADGET units ! */
#endif
		      for(k = 0; k < 3; k++)
#ifndef DIVBFORCE3
			acc[k] +=
#else
		        magacc[k]+=
#endif
			  magfac * ((mm_i[k][0] * magfac_i + mm_j[k][0] * magfac_j) * dx +
				    (mm_i[k][1] * magfac_i + mm_j[k][1] * magfac_j) * dy +
				    (mm_i[k][2] * magfac_i + mm_j[k][2] * magfac_j) * dz);
#if defined(DIVBFORCE) && !defined(DIVBFORCE3)
		      for(k = 0; k < 3; k++)
			acc[k] -=
#ifndef SFR
			  magfac * bpred[k] *(((bpred[0]) * magfac_i + (SphP[j].BPred[0]) * magfac_j) * dx
				            + ((bpred[1]) * magfac_i + (SphP[j].BPred[1]) * magfac_j) * dy
					    + ((bpred[2]) * magfac_i + (SphP[j].BPred[2]) * magfac_j) * dz);
#else
			  magfac * (	((bpred[k] * bpred[0]) * magfac_i + (bpred[k] * SphP[j].BPred[0] * pow(1.-SphP[j].XColdCloud,2.*POW_CC)) * magfac_j) * dx
				    +   ((bpred[k] * bpred[1]) * magfac_i + (bpred[k] * SphP[j].BPred[1] * pow(1.-SphP[j].XColdCloud,2.*POW_CC)) * magfac_j) * dy
				    +   ((bpred[k] * bpred[2]) * magfac_i + (bpred[k] * SphP[j].BPred[2] * pow(1.-SphP[j].XColdCloud,2.*POW_CC)) * magfac_j) * dz);
#endif
#endif
#if defined(DIVBFORCE3) && !defined(DIVBFORCE)
		      for(k = 0; k < 3; k++)
			magcorr[k] +=
#ifndef SFR
			  magfac * bpred[k] *(((bpred[0]) * magfac_i + (SphP[j].BPred[0]) * magfac_j) * dx
				            + ((bpred[1]) * magfac_i + (SphP[j].BPred[1]) * magfac_j) * dy
					    + ((bpred[2]) * magfac_i + (SphP[j].BPred[2]) * magfac_j) * dz);
#else
			  magfac * bpred[k] *(((bpred[0]) * magfac_i + (SphP[j].BPred[0] * pow(1.-SphP[j].XColdCloud,2.*POW_CC) ) * magfac_j) * dx
				            + ((bpred[1]) * magfac_i + (SphP[j].BPred[1] * pow(1.-SphP[j].XColdCloud,2.*POW_CC) ) * magfac_j) * dy
					    + ((bpred[2]) * magfac_i + (SphP[j].BPred[2] * pow(1.-SphP[j].XColdCloud,2.*POW_CC) ) * magfac_j) * dz);
#endif
#endif
#endif /* end MAG FORCE   */
#ifdef ALFA_OMEGA_DYN // Known Bug
		      dtB[0] += magfac * alfaomega * All.Tau_A0 / 3.0 * (dBy * dz - dBy * dy);
		      dtB[1] += magfac * alfaomega * All.Tau_A0 / 3.0 * (dBz * dx - dBx * dz);
		      dtB[2] += magfac * alfaomega * All.Tau_A0 / 3.0 * (dBx * dy - dBy * dx);
#endif
#endif /* end of MAGNETIC */


#ifndef MAGNETIC_SIGNALVEL
		      vsig = soundspeed_i + soundspeed_j;
#else
		      vsig = magneticspeed_i + magneticspeed_j;
#endif


#ifndef ALTERNATIVE_VISCOUS_TIMESTEP
		      if(vsig > maxSignalVel)
			maxSignalVel = vsig;
#endif
		      if(vdotr2 < 0)	/* ... artificial viscosity */
			{
#ifndef ALTVISCOSITY
#ifndef CONVENTIONAL_VISCOSITY
			  mu_ij = fac_mu * vdotr2 / r;	/* note: this is negative! */
#else
			  c_ij = 0.5 * (soundspeed_i + soundspeed_j);
			  h_ij = 0.5 * (h_i + h_j);
			  mu_ij = fac_mu * h_ij * vdotr2 / (r2 + 0.0001 * h_ij * h_ij);
#endif
#ifdef MAGNETIC
			  vsig -= 1.5 * mu_ij;
#else
			  vsig -= 3 * mu_ij;
#endif


#ifndef ALTERNATIVE_VISCOUS_TIMESTEP
			  if(vsig > maxSignalVel)
			    maxSignalVel = vsig;
#endif

#ifndef NAVIERSTOKES
			  f2 =
			    fabs(SphP[j].v.DivVel) / (fabs(SphP[j].v.DivVel) + SphP[j].r.CurlVel +
						      0.0001 * soundspeed_j / fac_mu / PPP[j].Hsml);
#else
			  f2 =
			    fabs(SphP[j].v.DivVel) / (fabs(SphP[j].v.DivVel) + SphP[j].u.s.CurlVel +
						      0.0001 * soundspeed_j / fac_mu / PPP[j].Hsml);
#endif

#ifdef NO_SHEAR_VISCOSITY_LIMITER
			  f1 = f2 = 1;
#endif
#ifdef TIME_DEP_ART_VISC
			  BulkVisc_ij = 0.5 * (alpha + SphP[j].alpha);
#else
			  BulkVisc_ij = All.ArtBulkViscConst;
#endif
#ifndef CONVENTIONAL_VISCOSITY
			  visc = 0.25 * BulkVisc_ij * vsig * (-mu_ij) / rho_ij * (f1 + f2);
#else
			  visc =
			    (-BulkVisc_ij * mu_ij * c_ij + 2 * BulkVisc_ij * mu_ij * mu_ij) /
			    rho_ij * (f1 + f2) * 0.5;
#endif

#else /* start of ALTVISCOSITY block */
			  if(f1 < 0)
			    mu_i = h_i * fabs(f1);	/* f1 hold here the velocity divergence of particle i */
			  else
			    mu_i = 0;
			  if(SphP[j].v.DivVel < 0)
			    mu_j = h_j * fabs(SphP[j].v.DivVel);
			  else
			    mu_j = 0;
			  visc = All.ArtBulkViscConst * ((soundspeed_i + mu_i) * mu_i / rho +
							 (soundspeed_j + mu_j) * mu_j / SphP[j].d.Density);
#endif /* end of ALTVISCOSITY block */


			  /* .... end artificial viscosity evaluation */
			  /* now make sure that viscous acceleration is not too large */
#ifdef ALTERNATIVE_VISCOUS_TIMESTEP
			  if(visc > 0)
			    {
			      dt = fac_vsic_fix * vdotr2 /
				(0.5 * (mass + P[j].Mass) * (dwk_i + dwk_j) * r * visc);

			      dt /= hubble_a;

			      if(dt < minViscousDt)
				minViscousDt = dt;
			    }
#endif

#ifndef NOVISCOSITYLIMITER
			  dt =
			    2 * IMAX(timestep,
				     (P[j].TimeBin ? (1 << P[j].TimeBin) : 0)) * All.Timebase_interval;
			  if(dt > 0 && (dwk_i + dwk_j) < 0)
			    {
#ifdef BLACK_HOLES
			      if((mass + P[j].Mass) > 0)
#endif
				visc = DMIN(visc, 0.5 * fac_vsic_fix * vdotr2 /
					    (0.5 * (mass + P[j].Mass) * (dwk_i + dwk_j) * r * dt));
			    }
#endif
			}
		      else
			{
			  visc = 0;
			}
#ifndef TRADITIONAL_SPH_FORMULATION
		      hfc_visc = 0.5 * P[j].Mass * visc * (dwk_i + dwk_j) / r;

#ifdef DENSITY_INDEPENDENT_SPH
              hfc = hfc_visc;

          /* leading-order term */
              hfc += P[j].Mass *
                      (dwk_i*p_over_rho2_i*SphP[j].EntVarPred/entvarpred +
                       dwk_j*p_over_rho2_j*entvarpred/SphP[j].EntVarPred) / r;

          /* grad-h corrections */
              hfc += P[j].Mass *
                      (dwk_i*p_over_rho2_i*egyrho/rho*dhsmlDensityFactor +
                       dwk_j*p_over_rho2_j*SphP[j].EgyWtDensity/SphP[j].d.Density*SphP[j].DhsmlEgyDensityFactor) / r;
#else
		      p_over_rho2_j *= SphP[j].h.DhsmlDensityFactor;
		      /* Formulation derived from the Lagrangian */
		      hfc = hfc_visc + P[j].Mass * (p_over_rho2_i * dwk_i + p_over_rho2_j * dwk_j) / r;
#endif
#else
		      hfc = hfc_visc +
			0.5 * P[j].Mass * (dwk_i + dwk_j) / r * (p_over_rho2_i + p_over_rho2_j);

		      /* hfc_egy = 0.5 * P[j].Mass * (dwk_i + dwk_j) / r * (p_over_rho2_i + p_over_rho2_j); */
		      hfc_egy = P[j].Mass * (dwk_i + dwk_j) / r * (p_over_rho2_i);
#endif

#ifdef WINDS
		      if(P[j].Type == 0)
			if(SphP[j].DelayTime > 0)	/* No force by wind particles */
			  {
			    hfc = hfc_visc = 0;
			  }
#endif

#ifndef NOACCEL
		      acc[0] += FLT(-hfc * dx);
		      acc[1] += FLT(-hfc * dy);
		      acc[2] += FLT(-hfc * dz);
#endif

#if !defined(EOS_DEGENERATE) && !defined(TRADITIONAL_SPH_FORMULATION)
		      dtEntropy += FLT(0.5 * hfc_visc * vdotr2);
#else

#ifdef TRADITIONAL_SPH_FORMULATION
		      dtEntropy += FLT(0.5 * (hfc_visc + hfc_egy) * vdotr2);
#else
		      dtEntropy += FLT(0.5 * hfc * vdotr2);
#endif
#endif


#ifdef NAVIERSTOKES
		      faci = mass * shear_viscosity / (rho * rho) * dwk_i / r;

#ifndef NAVIERSTOKES_CONSTANT
		      faci *= pow((Entropy * pow(rho * a3inv, GAMMA_MINUS1) / GAMMA_MINUS1), 2.5);	/*multiplied by E^5/2 */
#endif
		      facj = P[j].Mass * get_shear_viscosity(j) /
			(SphP[j].d.Density * SphP[j].d.Density) * dwk_j / r;

#ifndef NAVIERSTOKES_CONSTANT
		      facj *= pow((SphP[j].Entropy * pow(SphP[j].d.Density * a3inv, GAMMA_MINUS1) / GAMMA_MINUS1), 2.5);	/*multiplied by E^5/2 */
#endif

#ifdef NAVIERSTOKES_BULK
		      facbi = mass * All.NavierStokes_BulkViscosity / (rho * rho) * dwk_i / r;
		      facbj = P[j].Mass * All.NavierStokes_BulkViscosity /
			(SphP[j].d.Density * SphP[j].d.Density) * dwk_j / r;
#endif

#ifdef WINDS
		      if(P[j].Type == 0)
			if(SphP[j].DelayTime > 0)	/* No visc for wind particles */
			  {
			    faci = facj = 0;
#ifdef NAVIERSTOKES_BULK
			    facbi = facbj = 0;
#endif
			  }
#endif

#ifdef VISCOSITY_SATURATION
		      IonMeanFreePath_i = All.IonMeanFreePath * pow((Entropy * pow(rho * a3inv, GAMMA_MINUS1) / GAMMA_MINUS1), 2.0) / rho;	/* u^2/rho */

		      IonMeanFreePath_j = All.IonMeanFreePath * pow((SphP[j].Entropy * pow(SphP[j].d.Density * a3inv, GAMMA_MINUS1) / GAMMA_MINUS1), 2.0) / SphP[j].d.Density;	/* u^2/rho */

		      for(k = 0, VelLengthScale_i = 0, VelLengthScale_j = 0; k < 3; k++)
			{
			  if(fabs(stressdiag[k]) > 0)
			    {
			      VelLengthScale_i = 2 * soundspeed_i / fabs(stressdiag[k]);

			      if(VelLengthScale_i < IonMeanFreePath_i && VelLengthScale_i > 0)
				{
				  stressdiag[k] = stressdiag[k] * (VelLengthScale_i / IonMeanFreePath_i);

				}
			    }
			  if(fabs(SphP[j].u.s.StressDiag[k]) > 0)
			    {
			      VelLengthScale_j = 2 * soundspeed_j / fabs(SphP[j].u.s.StressDiag[k]);

			      if(VelLengthScale_j < IonMeanFreePath_j && VelLengthScale_j > 0)
				{
				  SphP[j].u.s.StressDiag[k] = SphP[j].u.s.StressDiag[k] *
				    (VelLengthScale_j / IonMeanFreePath_j);

				}
			    }
			  if(fabs(stressoffdiag[k]) > 0)
			    {
			      VelLengthScale_i = 2 * soundspeed_i / fabs(stressoffdiag[k]);

			      if(VelLengthScale_i < IonMeanFreePath_i && VelLengthScale_i > 0)
				{
				  stressoffdiag[k] =
				    stressoffdiag[k] * (VelLengthScale_i / IonMeanFreePath_i);
				}
			    }
			  if(fabs(SphP[j].u.s.StressOffDiag[k]) > 0)
			    {
			      VelLengthScale_j = 2 * soundspeed_j / fabs(SphP[j].u.s.StressOffDiag[k]);

			      if(VelLengthScale_j < IonMeanFreePath_j && VelLengthScale_j > 0)
				{
				  SphP[j].u.s.StressOffDiag[k] = SphP[j].u.s.StressOffDiag[k] *
				    (VelLengthScale_j / IonMeanFreePath_j);
				}
			    }
			}
#endif

		      /* Acceleration due to the shear viscosity */
		      acc[0] += faci * (stressdiag[0] * dx + stressoffdiag[0] * dy + stressoffdiag[1] * dz)
			+ facj * (SphP[j].u.s.StressDiag[0] * dx + SphP[j].u.s.StressOffDiag[0] * dy +
				  SphP[j].u.s.StressOffDiag[1] * dz);

		      acc[1] += faci * (stressoffdiag[0] * dx + stressdiag[1] * dy + stressoffdiag[2] * dz)
			+ facj * (SphP[j].u.s.StressOffDiag[0] * dx + SphP[j].u.s.StressDiag[1] * dy +
				  SphP[j].u.s.StressOffDiag[2] * dz);

		      acc[2] += faci * (stressoffdiag[1] * dx + stressoffdiag[2] * dy + stressdiag[2] * dz)
			+ facj * (SphP[j].u.s.StressOffDiag[1] * dx + SphP[j].u.s.StressOffDiag[2] * dy +
				  SphP[j].u.s.StressDiag[2] * dz);

		      /*Acceleration due to the bulk viscosity */
#ifdef NAVIERSTOKES_BULK
#ifdef VISCOSITY_SATURATION
		      VelLengthScale_i = 0;
		      VelLengthScale_j = 0;

		      if(fabs(divvel) > 0)
			{
			  VelLengthScale_i = 3 * soundspeed_i / fabs(divvel);

			  if(VelLengthScale_i < IonMeanFreePath_i && VelLengthScale_i > 0)
			    {
			      divvel = divvel * (VelLengthScale_i / IonMeanFreePath_i);
			    }
			}

		      if(fabs(SphP[j].u.s.a4.DivVel) > 0)
			{
			  VelLengthScale_j = 3 * soundspeed_j / fabs(SphP[j].u.s.a4.DivVel);

			  if(VelLengthScale_j < IonMeanFreePath_j && VelLengthScale_j > 0)
			    {
			      SphP[j].u.s.a4.DivVel = SphP[j].u.s.a4.DivVel *
				(VelLengthScale_j / IonMeanFreePath_j);

			    }
			}
#endif


		      acc[0] += facbi * divvel * dx + facbj * SphP[j].u.s.a4.DivVel * dx;
		      acc[1] += facbi * divvel * dy + facbj * SphP[j].u.s.a4.DivVel * dy;
		      acc[2] += facbi * divvel * dz + facbj * SphP[j].u.s.a4.DivVel * dz;
#endif
#endif /* end NAVIERSTOKES */


#ifdef MAGNETIC
#ifdef EULER_DISSIPATION
		      alpha_ij_eul = All.ArtMagDispConst;

		      dteulA +=
			alpha_ij_eul * 0.5 * vsig * (eulA -
						     SphP[j].EulerA) * magfac_sym * r * rho / (rho_ij *
											       rho_ij);
		      dteulB +=
			alpha_ij_eul * 0.5 * vsig * (eulB -
						     SphP[j].EulerB) * magfac_sym * r * rho / (rho_ij *
											       rho_ij);

		      dTu_diss_eul = -magfac_sym * alpha_ij_eul * (dBx * dBx + dBy * dBy + dBz * dBz);
		      dtEntropy += dTu_diss_eul * 0.25 * vsig * mu0_1 * r / (rho_ij * rho_ij);
#endif
#ifdef MAGNETIC_DISSIPATION
		      magfac_sym *= vsig * 0.5 * Balpha_ij * r * rho / (rho_ij * rho_ij);
		      dtEntropy += dTu_diss_b * 0.25 * vsig * mu0_1 * r / (rho_ij * rho_ij);
		      dtB[0] += magfac_sym * dBx;
		      dtB[1] += magfac_sym * dBy;
		      dtB[2] += magfac_sym * dBz;
#endif
#endif

#ifdef WAKEUP
		      if(vsig > WAKEUP * SphP[j].MaxSignalVel)
			{
			  SphP[j].wakeup = 1;
			}
#endif
		    }
		}
	    }
	}

#ifndef DONOTUSENODELIST
      if(mode == 1)
	{
	  listindex++;
	  if(listindex < NODELISTLENGTH)
	    {
	      startnode = HydroDataGet[target].NodeList[listindex];
	      if(startnode >= 0)
		startnode = Nodes[startnode].u.d.nextnode;	/* open it */
	    }
	}
#endif
    }


  /* Now collect the result at the right place */
  if(mode == 0)
    {
      for(k = 0; k < 3; k++)
	SphP[target].a.dHydroAccel[k] = acc[k];
      SphP[target].e.dDtEntropy = dtEntropy;
#ifdef DIVBFORCE3
      for(k = 0; k < 3; k++)
         SphP[target].magacc[k] = magacc[k];
      for(k = 0; k < 3; k++)
         SphP[target].magcorr[k] = magcorr[k];
#endif
#ifdef HYDRO_COST_FACTOR
      if(All.ComovingIntegrationOn)
	P[target].GravCost += HYDRO_COST_FACTOR * All.Time * ninteractions;
      else
        P[target].GravCost += HYDRO_COST_FACTOR * ninteractions;
#endif

#ifdef ALTERNATIVE_VISCOUS_TIMESTEP
      SphP[target].MinViscousDt = minViscousDt;
#else
      SphP[target].MaxSignalVel = maxSignalVel;
#endif

#ifdef JD_VTURB
		SphP[target].Vrms = vRms; 
#endif

#if defined(MAGNETIC) && ( !defined(EULERPOTENTIALS) || !defined(VECT_POTENTIAL) )
      for(k = 0; k < 3; k++)
	SphP[target].DtB[k] = dtB[k];
#endif
#ifdef DIVBCLEANING_DEDNER
      for(k = 0; k < 3; k++)
      	SphP[target].GradPhi[k] = gradphi[k];
#endif
#ifdef VECT_POTENTIAL
      SphP[target].DtA[0] = dta[0];
      SphP[target].DtA[1] = dta[1];
      SphP[target].DtA[2] = dta[2];
#endif
#ifdef EULER_DISSIPATION
      SphP[target].DtEulerA = dteulA;
      SphP[target].DtEulerB = dteulB;
#endif
    }
  else
    {
      for(k = 0; k < 3; k++)
      HydroDataResult[target].Acc[k] = acc[k];
      HydroDataResult[target].DtEntropy = dtEntropy;
#ifdef DIVBFORCE3
      for(k = 0; k < 3; k++)
      HydroDataResult[target].magacc[k] = magacc[k];
      for(k = 0; k < 3; k++)
      HydroDataResult[target].magcorr[k] = magcorr[k];
#endif
#ifdef HYDRO_COST_FACTOR
      HydroDataResult[target].Ninteractions = ninteractions;
#endif

#ifdef ALTERNATIVE_VISCOUS_TIMESTEP
      HydroDataResult[target].MinViscousDt = minViscousDt;
#else
      HydroDataResult[target].MaxSignalVel = maxSignalVel;
#endif
#ifdef JD_VTURB
		HydroDataResult[target].Vrms = vRms; 
#endif
#if defined(MAGNETIC) && ( !defined(EULERPOTENTIALS) || !defined(VECT_POTENTIAL) )
      for(k = 0; k < 3; k++)
	HydroDataResult[target].DtB[k] = dtB[k];
#ifdef DIVBCLEANING_DEDNER
      for(k = 0; k < 3; k++)
        HydroDataResult[target].GradPhi[k] = gradphi[k];
#endif
#endif
#ifdef VECT_POTENTIAL
      HydroDataResult[target].dta[0] = dta[0];
      HydroDataResult[target].dta[1] = dta[1];
      HydroDataResult[target].dta[2] = dta[2];
#endif
#ifdef EULER_DISSIPATION
      HydroDataResult[target].DtEulerA = dteulA;
      HydroDataResult[target].DtEulerB = dteulB;
#endif
    }

  return 0;
}

void *hydro_evaluate_primary(void *p)
{
  int thread_id = *(int *) p;

  int i, j;

  int *exportflag, *exportnodecount, *exportindex, *ngblist;


  ngblist = Ngblist + thread_id * NumPart;
  exportflag = Exportflag + thread_id * NTask;
  exportnodecount = Exportnodecount + thread_id * NTask;
  exportindex = Exportindex + thread_id * NTask;

  /* Note: exportflag is local to each thread */
  for(j = 0; j < NTask; j++)
    exportflag[j] = -1;

  while(1)
    {
      LOCK_NEXPORT;

      if(BufferFullFlag != 0 || NextParticle < 0)
	{
	  UNLOCK_NEXPORT;
	  break;
	}

      i = NextParticle;
      ProcessedFlag[i] = 0;
      NextParticle = NextActiveParticle[NextParticle];
      UNLOCK_NEXPORT;

      if(P[i].Type == 0)
	{
	  if(hydro_evaluate(i, 0, exportflag, exportnodecount, exportindex, ngblist) < 0)
	    break;		/* export buffer has filled up */
	}

      ProcessedFlag[i] = 1;	/* particle successfully finished */

    }

  return NULL;

}



void *hydro_evaluate_secondary(void *p)
{
  int thread_id = *(int *) p;

  int j, dummy, *ngblist;

  ngblist = Ngblist + thread_id * NumPart;


  while(1)
    {
      LOCK_NEXPORT;
      j = NextJ;
      NextJ++;
      UNLOCK_NEXPORT;

      if(j >= Nimport)
	break;

      hydro_evaluate(j, 1, &dummy, &dummy, &dummy, ngblist);
    }

  return NULL;

}
