#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <gsl/gsl_math.h>

#include "allvars.h"
#include "proto.h"

/*! \file blackhole.c
 *  \brief routines for gas accretion onto black holes, and black hole mergers
 */

#ifdef BLACK_HOLES

#if defined (BH_THERMALFEEDBACK)
#if defined (LT_DF_BH) && defined (LT_DF_BH_MASS_SWITCH)
/* switch the feedback at given BH mass */
static double BH_mass_switch;
#endif
#endif

static struct blackholedata_in
{
  MyDouble Pos[3];
  MyFloat Density;
  MyFloat FBDensity;
  MyFloat Mdot;
  MyFloat Dt;
  MyFloat Hsml;
  MyFloat Mass;
  MyFloat BH_Mass;
  MyFloat Vel[3];
  MyFloat Csnd;
  MyIDType ID;
  int Index;
  int NodeList[NODELISTLENGTH];
#ifdef LT_BH
  double AltRho;
#ifdef LT_BH_CUT_KERNEL
  float  CutHsml;
#endif
#endif
#ifdef LT_DF_BH_ONLY_RADIOMODE_KERNEL
  MyFloat BH_MdotEddington;  
#endif
#ifdef BH_KINETICFEEDBACK
  MyFloat ActiveTime;
  MyFloat ActiveEnergy;
#endif  
}
 *BlackholeDataIn, *BlackholeDataGet;

static struct blackholedata_out
{
  MyLongDouble Mass;
  MyLongDouble BH_Mass;
  MyLongDouble AccretedMomentum[3];
#ifdef REPOSITION_ON_POTMIN
  MyFloat BH_MinPotPos[3];
  MyFloat BH_MinPot;
#endif
#ifdef BH_BUBBLES
  MyLongDouble BH_Mass_bubbles;
#ifdef UNIFIED_FEEDBACK
  MyLongDouble BH_Mass_radio;
#endif
#endif
#ifdef BH_COUNTPROGS
  int BH_CountProgs;
#endif
#ifdef LT_BH
  MyLongDouble PartialContrib;
#endif
}
 *BlackholeDataResult, *BlackholeDataOut;

#define BHPOTVALUEINIT 1.0e30

static double hubble_a, ascale, a3inv;

static int N_gas_swallowed, N_BH_swallowed;
#ifdef LT_BH_ACCRETE_SLICES
static int N_gas_slices_swallowed;
#endif

void blackhole_accretion(void)
{
  int i, j, k, n, bin;
  int ndone_flag, ndone;
  int ngrp, sendTask, recvTask, place, nexport, nimport, dummy;
  int Ntot_gas_swallowed, Ntot_BH_swallowed;
  double mdot, rho, bhvel, soundspeed, meddington, dt, mdot_in_msun_per_year;
  double mass_real, total_mass_real, medd, total_mdoteddington;
  double mass_holes, total_mass_holes, total_mdot;
  double injection, total_injection;
#ifdef BONDI
  double norm;
#endif

#ifdef BH_BUBBLES
  MyFloat bh_center[3];
  double *bh_dmass, *tot_bh_dmass;
  float *bh_posx, *bh_posy, *bh_posz;
  float *tot_bh_posx, *tot_bh_posy, *tot_bh_posz;
  int l, num_activebh = 0, total_num_activebh = 0;
  int *common_num_activebh, *disp;
  MyIDType *bh_id, *tot_bh_id;
#endif
  MPI_Status status;

#ifdef LT_BH_ACCRETE_SLICES
  int Ntot_gas_slices_swallowed;
#endif
  
  if(ThisTask == 0)
    {
      printf("Beginning black-hole accretion\n");
      fflush(stdout);
    }

  CPU_Step[CPU_MISC] += measure_time();

  if(All.ComovingIntegrationOn)
    {
      ascale = All.Time;
      hubble_a = hubble_function(All.Time);
      a3inv = 1.0 / (All.Time * All.Time * All.Time);
    }
  else
    hubble_a = ascale = a3inv = 1;

#if defined (BH_THERMALFEEDBACK) || defined (BH_KINETICFEEDBACK)
#if defined (LT_DF_BH) && defined(LT_DF_BH_MASS_SWITCH)
    /* switch feedback mode at BH mass of 10^9 Msun */
    All.BH_mass_switch = 0.1/All.HubbleParam;
#endif 
#endif

  /* Let's first compute the Mdot values */

  for(n = FirstActiveParticle; n >= 0; n = NextActiveParticle[n])
    if(P[n].Type == 5)
      {
	mdot = 0;		/* if no accretion model is enabled, we have mdot=0 */

	rho = P[n].b1.BH_Density;

#ifdef BH_USE_GASVEL_IN_BONDI
	bhvel = sqrt(pow(P[n].Vel[0] - P[n].b3.BH_SurroundingGasVel[0], 2) +
		     pow(P[n].Vel[1] - P[n].b3.BH_SurroundingGasVel[1], 2) +
		     pow(P[n].Vel[2] - P[n].b3.BH_SurroundingGasVel[2], 2));
#else
	bhvel = 0;
#endif

	if(All.ComovingIntegrationOn)
	  {
	    bhvel /= All.Time;
	    rho /= pow(All.Time, 3);
	  }

	soundspeed = sqrt(GAMMA * P[n].b2.BH_Entropy * pow(rho, GAMMA_MINUS1));

#ifndef LT_DF_BH                          
	/* Note: we take here a radiative efficiency of 0.1 for Eddington accretion */
	meddington = (4 * M_PI * GRAVITY * C * PROTONMASS / (0.1 * C * C * THOMPSON)) * P[n].BH_Mass
	  * All.UnitTime_in_s;
#else
        /* Note: we insert as parameter the radiative efficiency */
        meddington = (4 * M_PI * GRAVITY * C * PROTONMASS / (All.BH_Radiative_Efficiency * C * C * THOMPSON)) * P[n].BH_Mass
          * All.UnitTime_in_s;
#endif

#ifdef BONDI
        norm = pow((pow(soundspeed, 2) + pow(bhvel, 2)), 1.5);
	if(norm > 0)
	  mdot = 4. * M_PI * All.BlackHoleAccretionFactor * All.G * All.G *
	    P[n].BH_Mass * P[n].BH_Mass * rho / norm;
	else
	  mdot = 0;
#endif


#ifdef ENFORCE_EDDINGTON_LIMIT
	if(mdot > All.BlackHoleEddingtonFactor * meddington)
	  mdot = All.BlackHoleEddingtonFactor * meddington;
#endif
	P[n].BH_Mdot = mdot;

	if(P[n].BH_Mass > 0)
	  {
	    fprintf(FdBlackHolesDetails, "BH=" IDFMT " %g %g %g %g %g %g   %2.7f %2.7f %2.7f %g %g %g %g\n",
		    P[n].ID, All.Time, P[n].BH_Mass, mdot, rho, soundspeed, bhvel, P[n].Pos[0],P[n].Pos[1],P[n].Pos[2],
		    P[n].b3.BH_SurroundingGasVel[0],
	            P[n].b3.BH_SurroundingGasVel[1], 
	            P[n].b3.BH_SurroundingGasVel[2], PPP[n].Hsml);
	  }

	dt = (P[n].TimeBin ? (1 << P[n].TimeBin) : 0) * All.Timebase_interval / hubble_a;

#ifdef BH_DRAG
	/* add a drag force for the black-holes,
	   accounting for the accretion */
	double fac;

	if(P[n].BH_Mass > 0)
	  {
	    fac = P[n].BH_Mdot * dt / P[n].BH_Mass;
	    /*
	    fac = meddington * dt / P[n].BH_Mass;
	    */
	    if(fac > 1)
	      fac = 1;

	    if(dt > 0)
	      for(k = 0; k < 3; k++)
		P[n].g.GravAccel[k] +=
		  -ascale * ascale * fac / dt * (P[n].Vel[k] - P[n].b3.BH_SurroundingGasVel[k]) / ascale;
	  }
#endif

#ifdef KD_FRICTION
	/* add a friction force for the black-holes,
	   accounting for the environment */
	double fac_friction, relvel=0;
#ifdef KD_FRICTION_DYNAMIC
        double x,a_erf,erf,lambda;
#endif

	if(P[n].BH_Mass > 0)
	  {
	    /* averaged value for colomb logarithm and integral over the distribution function */
            /* fac_friction = log(lambda) * [erf(x) - 2*x*exp(-x^2)/sqrt(pi)]                  */
            /*       lambda = b_max * v^2 / G / (M+m)                                          */
            /*        b_max = Size of system (e.g. Rvir)                                       */
            /*            v = Relative velocity of BH with respect to the environment          */
            /*            M = Mass of BH                                                       */
            /*            m = individual mass elements composing the large system (e.g. m<<M)  */
            /*            x = v/sqrt(2)/sigma                                                  */
            /*        sigma = width of the max. distr. of the host system                      */
            /*                (e.g. sigma = v_disp / 3                                         */  

	    if(dt > 0)
	      {
		for(k = 0; k < 3; k++)
		  relvel += pow(P[n].Vel[k] - P[n].BH_SurroundingVel[k],2);
#ifdef KD_FRICTION_DYNAMIC
		a_erf = 8 * (M_PI - 3)/(3 * M_PI * (4. - M_PI));
		x = sqrt(relvel) / sqrt(2) / P[n].BH_sigma; 
		/* First term is aproximation of the error function */
                fac_friction = x / fabs(x) * sqrt(1 - exp(- x * x * (4 / M_PI + a_erf * x * x) / (1 + a_erf * x * x)))
		               - 2 * x / sqrt(M_PI) * exp(- x * x);
                lambda = P[n].BH_bmax * relvel / All.G / P[n].BH_Mass;
                printf("Task %d: x=%e, log(lambda)=%e, facerf=%e m=%e, sigma=%e\n",
                       ThisTask,x,log(lambda),fac_friction,P[n].BH_Mass,P[n].BH_sigma);
                fac_friction *= log(lambda);
#else
		fac_friction = 10;
#endif

                fac_friction *= 4 * M_PI * All.G * All.G * P[n].BH_SurroundingDensity * P[n].BH_Mass / relvel / sqrt(relvel);
		printf("Task %d: fac = %e, vrel=%e, acc=(%e,%e,%e), adcc=(%e,%e,%e)\n",
		       ThisTask,fac_friction,sqrt(relvel), P[n].g.GravAccel[0],P[n].g.GravAccel[1], P[n].g.GravAccel[2],
		       fac_friction * (P[n].Vel[0] - P[n].BH_SurroundingVel[0]),
		       fac_friction * (P[n].Vel[1] - P[n].BH_SurroundingVel[1]),
		       fac_friction * (P[n].Vel[2] - P[n].BH_SurroundingVel[2]));
		for(k = 0; k < 3; k++)
		  P[n].g.GravAccel[k] -= fac_friction * (P[n].Vel[k] - P[n].BH_SurroundingVel[k]); 
	      }
	  }
#endif

#if defined LT_DF_BH && (defined (BH_THERMALFEEDBACK) || defined (BH_KINETICFEEDBACK))
        P[n].BH_Mass += (1 - All.BH_Radiative_Efficiency) * P[n].BH_Mdot * dt;
#else
	P[n].BH_Mass += P[n].BH_Mdot * dt;
#endif

#if defined(LT_DF_BH) && (defined (BH_THERMALFEEDBACK) || defined (BH_KINETICFEEDBACK))

#ifdef LT_DF_BH_MASS_SWITCH		  
        /* switch btw different modes by changing the feedback factor */
        if(P[n].BH_Mass >= BH_mass_switch)
#endif
#ifdef LT_DF_BH_BHAR_SWITCH
	  if(P[n].BH_Mdot < All.BH_radio_treshold * meddington)
            /* switch the feedback factor using an accretion rate based treshold */
#endif
            All.BlackHoleFeedbackFactor = 0.2;
	  else
            All.BlackHoleFeedbackFactor = 0.05;
#endif	  
        
#ifdef BH_BUBBLES
	P[n].BH_Mass_bubbles += P[n].BH_Mdot * dt;
#ifdef UNIFIED_FEEDBACK
	if(P[n].BH_Mdot < All.RadioThreshold * meddington)
	  P[n].BH_Mass_radio += P[n].BH_Mdot * dt;
#endif
#endif
      }


  /* Now let's invoke the functions that stochasticall swallow gas
   * and deal with black hole mergers.
   */

  if(ThisTask == 0)
    {
      printf("Start swallowing of gas particles and black holes\n");
      fflush(stdout);
    }


  N_gas_swallowed = N_BH_swallowed = 0;
#ifdef LT_BH_ACCRETE_SLICES
  N_gas_slices_swallowed = 0;
#endif


  /* allocate buffers to arrange communication */

  Ngblist = (int *) mymalloc("Ngblist", NumPart * sizeof(int));

  All.BunchSize =
    (int) ((All.BufferSize * 1024 * 1024) / (sizeof(struct data_index) + sizeof(struct data_nodelist) +
					     sizeof(struct blackholedata_in) +
					     sizeof(struct blackholedata_out) +
					     sizemax(sizeof(struct blackholedata_in),
						     sizeof(struct blackholedata_out))));
  DataIndexTable =
    (struct data_index *) mymalloc("DataIndexTable", All.BunchSize * sizeof(struct data_index));
  DataNodeList =
    (struct data_nodelist *) mymalloc("DataNodeList", All.BunchSize * sizeof(struct data_nodelist));


  /** Let's first spread the feedback energy, and determine which particles may be swalled by whom */

  i = FirstActiveParticle;	/* first particle for this task */

  do
    {
      for(j = 0; j < NTask; j++)
	{
	  Send_count[j] = 0;
	  Exportflag[j] = -1;
	}

      /* do local particles and prepare export list */

      for(nexport = 0; i >= 0; i = NextActiveParticle[i])
	if(P[i].Type == 5)
	  if(blackhole_evaluate(i, 0, &nexport, Send_count) < 0)
	    break;

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

      BlackholeDataGet =
	(struct blackholedata_in *) mymalloc("BlackholeDataGet", nimport * sizeof(struct blackholedata_in));
      BlackholeDataIn =
	(struct blackholedata_in *) mymalloc("BlackholeDataIn", nexport * sizeof(struct blackholedata_in));

      for(j = 0; j < nexport; j++)
	{
	  place = DataIndexTable[j].Index;

	  for(k = 0; k < 3; k++)
	    {
	      BlackholeDataIn[j].Pos[k] = P[place].Pos[k];
	      BlackholeDataIn[j].Vel[k] = P[place].Vel[k];
	    }

	  BlackholeDataIn[j].Hsml = PPP[place].Hsml;
	  BlackholeDataIn[j].Mass = P[place].Mass;
	  BlackholeDataIn[j].BH_Mass = P[place].BH_Mass;
	  BlackholeDataIn[j].Density = P[place].b1.BH_Density;
	  BlackholeDataIn[j].FBDensity = P[place].b1_FB.BH_FB_Density;
	  BlackholeDataIn[j].Mdot = P[place].BH_Mdot;
	  BlackholeDataIn[j].Csnd =
	    sqrt(GAMMA * P[place].b2.BH_Entropy *
		 pow(P[place].b1.BH_Density / (ascale * ascale * ascale), GAMMA_MINUS1));
	  BlackholeDataIn[j].Dt =
	    (P[place].TimeBin ? (1 << P[place].TimeBin) : 0) * All.Timebase_interval / hubble_a;
	  BlackholeDataIn[j].ID = P[place].ID;
#ifdef LT_BH
          BlackholeDataIn[j].AltRho  = P[place].b9.BH_AltDensity;
#ifdef LT_BH_CUT_KERNEL          
          BlackholeDataIn[j].CutHsml = P[place].CutHsml;
#endif
#endif
	  memcpy(BlackholeDataIn[j].NodeList,
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
		  MPI_Sendrecv(&BlackholeDataIn[Send_offset[recvTask]],
			       Send_count[recvTask] * sizeof(struct blackholedata_in), MPI_BYTE,
			       recvTask, TAG_DENS_A,
			       &BlackholeDataGet[Recv_offset[recvTask]],
			       Recv_count[recvTask] * sizeof(struct blackholedata_in), MPI_BYTE,
			       recvTask, TAG_DENS_A, MPI_COMM_WORLD, &status);
		}
	    }
	}


      myfree(BlackholeDataIn);
      BlackholeDataResult =
	(struct blackholedata_out *) mymalloc("BlackholeDataResult",
					      nimport * sizeof(struct blackholedata_out));
      BlackholeDataOut =
	(struct blackholedata_out *) mymalloc("BlackholeDataOut", nexport * sizeof(struct blackholedata_out));


      /* now do the particles that were sent to us */

      for(j = 0; j < nimport; j++)
	blackhole_evaluate(j, 1, &dummy, &dummy);

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
		  MPI_Sendrecv(&BlackholeDataResult[Recv_offset[recvTask]],
			       Recv_count[recvTask] * sizeof(struct blackholedata_out),
			       MPI_BYTE, recvTask, TAG_DENS_B,
			       &BlackholeDataOut[Send_offset[recvTask]],
			       Send_count[recvTask] * sizeof(struct blackholedata_out),
			       MPI_BYTE, recvTask, TAG_DENS_B, MPI_COMM_WORLD, &status);
		}
	    }

	}

      /* add the result to the particles */
      for(j = 0; j < nexport; j++)
	{
	  place = DataIndexTable[j].Index;

#ifdef REPOSITION_ON_POTMIN
	  if(P[place].BH_MinPot > BlackholeDataOut[j].BH_MinPot)
	    {
	      P[place].BH_MinPot = BlackholeDataOut[j].BH_MinPot;
	      for(k = 0; k < 3; k++)
		P[place].BH_MinPotPos[k] = BlackholeDataOut[j].BH_MinPotPos[k];
	    }
#endif
#ifdef LT_BH
          P[place].Normalization += BlackholeDataOut[j].PartialContrib;
#endif
#ifdef KD_SMOOTHED_MOMENTUM_ACCRETION 
	  for(k = 0; k < 3; k++)
	    P[place].b6.dBH_accreted_momentum[k] += BlackholeDataOut[j].AccretedMomentum[k];
#endif
	}

      myfree(BlackholeDataOut);
      myfree(BlackholeDataResult);
      myfree(BlackholeDataGet);
    }
  while(ndone < NTask);





  /* Now do the swallowing of particles */

  i = FirstActiveParticle;	/* first particle for this task */

  do
    {
      for(j = 0; j < NTask; j++)
	{
	  Send_count[j] = 0;
	  Exportflag[j] = -1;
	}

      /* do local particles and prepare export list */

      for(nexport = 0; i >= 0; i = NextActiveParticle[i])
	if(P[i].Type == 5)
	  if(P[i].SwallowID == 0)
	    if(blackhole_evaluate_swallow(i, 0, &nexport, Send_count) < 0)
	      break;


      qsort(DataIndexTable, nexport, sizeof(struct data_index), data_index_compare);

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

      BlackholeDataGet =
	(struct blackholedata_in *) mymalloc("BlackholeDataGet", nimport * sizeof(struct blackholedata_in));
      BlackholeDataIn =
	(struct blackholedata_in *) mymalloc("BlackholeDataIn", nexport * sizeof(struct blackholedata_in));

      for(j = 0; j < nexport; j++)
	{
	  place = DataIndexTable[j].Index;

	  for(k = 0; k < 3; k++)
	    BlackholeDataIn[j].Pos[k] = P[place].Pos[k];

	  BlackholeDataIn[j].Hsml = PPP[place].Hsml;
	  BlackholeDataIn[j].BH_Mass = P[place].BH_Mass;
	  BlackholeDataIn[j].ID = P[place].ID;

	  memcpy(BlackholeDataIn[j].NodeList,
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
		  MPI_Sendrecv(&BlackholeDataIn[Send_offset[recvTask]],
			       Send_count[recvTask] * sizeof(struct blackholedata_in), MPI_BYTE,
			       recvTask, TAG_DENS_A,
			       &BlackholeDataGet[Recv_offset[recvTask]],
			       Recv_count[recvTask] * sizeof(struct blackholedata_in), MPI_BYTE,
			       recvTask, TAG_DENS_A, MPI_COMM_WORLD, &status);
		}
	    }
	}


      myfree(BlackholeDataIn);
      BlackholeDataResult =
	(struct blackholedata_out *) mymalloc("BlackholeDataResult",
					      nimport * sizeof(struct blackholedata_out));
      BlackholeDataOut =
	(struct blackholedata_out *) mymalloc("BlackholeDataOut", nexport * sizeof(struct blackholedata_out));


      /* now do the particles that were sent to us */

      for(j = 0; j < nimport; j++)
	blackhole_evaluate_swallow(j, 1, &dummy, &dummy);

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
		  MPI_Sendrecv(&BlackholeDataResult[Recv_offset[recvTask]],
			       Recv_count[recvTask] * sizeof(struct blackholedata_out),
			       MPI_BYTE, recvTask, TAG_DENS_B,
			       &BlackholeDataOut[Send_offset[recvTask]],
			       Send_count[recvTask] * sizeof(struct blackholedata_out),
			       MPI_BYTE, recvTask, TAG_DENS_B, MPI_COMM_WORLD, &status);
		}
	    }

	}

      /* add the result to the particles */
      for(j = 0; j < nexport; j++)
	{
	  place = DataIndexTable[j].Index;

	  P[place].b4.dBH_accreted_Mass += BlackholeDataOut[j].Mass;
	  P[place].b5.dBH_accreted_BHMass += BlackholeDataOut[j].BH_Mass;
#ifdef BH_BUBBLES
	  P[place].b7.dBH_accreted_BHMass_bubbles += BlackholeDataOut[j].BH_Mass_bubbles;
#ifdef UNIFIED_FEEDBACK
	  P[place].b8.dBH_accreted_BHMass_radio += BlackholeDataOut[j].BH_Mass_radio;
#endif
#endif
#ifndef KD_SMOOTHED_MOMENTUM_ACCRETION 
	  for(k = 0; k < 3; k++)
	    P[place].b6.dBH_accreted_momentum[k] += BlackholeDataOut[j].AccretedMomentum[k];
#endif
#ifdef BH_COUNTPROGS
	  P[place].BH_CountProgs += BlackholeDataOut[j].BH_CountProgs;
#endif
	}

      myfree(BlackholeDataOut);
      myfree(BlackholeDataResult);
      myfree(BlackholeDataGet);
    }
  while(ndone < NTask);


  myfree(DataNodeList);
  myfree(DataIndexTable);
  myfree(Ngblist);


  MPI_Reduce(&N_gas_swallowed, &Ntot_gas_swallowed, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&N_BH_swallowed, &Ntot_BH_swallowed, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

#ifndef LT_BH_ACCRETE_SLICES  
  if(ThisTask == 0)
    {
      printf("Accretion done: %d gas particles swallowed, %d BH particles swallowed\n",
	     Ntot_gas_swallowed, Ntot_BH_swallowed);
      fflush(stdout);
    }
#else
  MPI_Reduce(&N_gas_slices_swallowed, &Ntot_gas_slices_swallowed, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

  if(ThisTask == 0)
    {
      printf("Accretion done: %d gas slices swallowed, %d last-slices swallowed (gas particles disappeared), %d BH particles swallowed\n",
	     Ntot_gas_slices_swallowed, Ntot_gas_swallowed, Ntot_BH_swallowed);
      fflush(stdout);
    }
#endif



#ifdef REPOSITION_ON_POTMIN
  for(n = FirstActiveParticle; n >= 0; n = NextActiveParticle[n])
    if(P[n].Type == 5)
      if(P[n].BH_MinPot < 0.5 * BHPOTVALUEINIT)
	for(k = 0; k < 3; k++)
	  P[n].Pos[k] = P[n].BH_MinPotPos[k];
#endif


  for(n = 0; n < TIMEBINS; n++)
    {
      if(TimeBinActive[n])
	{
	  TimeBin_BH_mass[n] = 0;
	  TimeBin_BH_dynamicalmass[n] = 0;
	  TimeBin_BH_Mdot[n] = 0;
	  TimeBin_BH_Medd[n] = 0;
	  TimeBin_GAS_Injection[n] = 0;
	}
    }
  for(n = FirstActiveParticle; n >= 0; n = NextActiveParticle[n])
    if(P[n].Type == 0) {
	  bin = P[n].TimeBin;
      TimeBin_GAS_Injection[bin] += SphP[n].i.dInjected_BH_Energy;
    }

  for(n = FirstActiveParticle; n >= 0; n = NextActiveParticle[n])
    if(P[n].Type == 5)
      {
#ifdef FLTROUNDOFFREDUCTION
	P[n].b4.BH_accreted_Mass = FLT(P[n].b4.dBH_accreted_Mass);
	P[n].b5.BH_accreted_BHMass = FLT(P[n].b5.dBH_accreted_BHMass);
#ifdef BH_BUBBLES
	P[n].b7.BH_accreted_BHMass_bubbles = FLT(P[n].b7.dBH_accreted_BHMass_bubbles);
#ifdef UNIFIED_FEEDBACK
	P[n].b8.BH_accreted_BHMass_radio = FLT(P[n].b8.dBH_accreted_BHMass_radio);
#endif
#endif
	for(k = 0; k < 3; k++)
	  P[n].b6.BH_accreted_momentum[k] = FLT(P[n].b6.dBH_accreted_momentum[k]);
#endif
	if(P[n].b4.BH_accreted_Mass > 0)
	  {
#ifndef KD_IGNORE_ACCRETED_MOMENTUM
	    for(k = 0; k < 3; k++)
	      P[n].Vel[k] =
		(P[n].Vel[k] * P[n].Mass + P[n].b6.BH_accreted_momentum[k]) /
		(P[n].Mass + P[n].b4.BH_accreted_Mass);
#endif

	    P[n].Mass += P[n].b4.BH_accreted_Mass;
	    P[n].BH_Mass += P[n].b5.BH_accreted_BHMass;
#ifdef BH_BUBBLES
	    P[n].BH_Mass_bubbles += P[n].b7.BH_accreted_BHMass_bubbles;
#ifdef UNIFIED_FEEDBACK
	    P[n].BH_Mass_radio += P[n].b8.BH_accreted_BHMass_radio;
#endif
#endif
	    P[n].b4.BH_accreted_Mass = 0;
	  }

#ifdef LT_BH        
        if( fabs(P[n].Normalization - 1) > 1e-4)
          {
            if(P[i].b9.BH_AltDensity > 0)
              
              printf("\t\t +-----------------------------------------------+\n"
                     "\t\t + BH warn : discrepancy in energy distribution   \n"
#if defined(LT_BH_LOG)                         
                     "\t\t + as large as %g (over %d neighbours, with %g Hsml)\n"
#else
                     "\t\t + as large as %g \n"
#endif
                     "\t\t + for BH %u @ proc %d\n"
                     "\t\t +-----------------------------------------------+\n",
                     P[n].Normalization - 1,
#if defined(LT_BH_LOG)                         
                     P[n].InnerNgb, P[i].Hsml,
#endif
                     P[n].ID, ThisTask); fflush(stdout);
          }
#endif
        
	bin = P[n].TimeBin;
	TimeBin_BH_mass[bin] += P[n].BH_Mass;
	TimeBin_BH_dynamicalmass[bin] += P[n].Mass;
	TimeBin_BH_Mdot[bin] += P[n].BH_Mdot;
	if(P[n].BH_Mass > 0)
	  TimeBin_BH_Medd[bin] += P[n].BH_Mdot / P[n].BH_Mass;

#ifdef BH_BUBBLES
	if(P[n].BH_Mass_bubbles > 0
	   && P[n].BH_Mass_bubbles > All.BlackHoleRadioTriggeringFactor * P[n].BH_Mass_ini)
	  num_activebh++;
#endif
      }

#ifdef BH_BUBBLES
  Ngblist = (int *) mymalloc("Ngblist", NumPart * sizeof(int));

  MPI_Allreduce(&num_activebh, &total_num_activebh, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  if(ThisTask == 0)
    {
      printf("The total number of active BHs is: %d\n", total_num_activebh);
      fflush(stdout);
    }

  if(total_num_activebh > 0)
    {
      bh_dmass = mymalloc("bh_dmass", num_activebh * sizeof(double));
      tot_bh_dmass = mymalloc("tot_bh_dmass", total_num_activebh * sizeof(double));
      bh_posx = mymalloc("bh_posx", num_activebh * sizeof(float));
      bh_posy = mymalloc("bh_posy", num_activebh * sizeof(float));
      bh_posz = mymalloc("bh_posz", num_activebh * sizeof(float));
      tot_bh_posx = mymalloc("tot_bh_posx", total_num_activebh * sizeof(float));
      tot_bh_posy = mymalloc("tot_bh_posy", total_num_activebh * sizeof(float));
      tot_bh_posz = mymalloc("tot_bh_posz", total_num_activebh * sizeof(float));
      //      bh_id = mymalloc("bh_id", num_activebh * sizeof(unsigned int));
      //      tot_bh_id = mymalloc("tot_bh_id", total_num_activebh * sizeof(unsigned int));
      bh_id = mymalloc("bh_id", num_activebh * sizeof(MyIDType));
      tot_bh_id = mymalloc("tot_bh_id", total_num_activebh * sizeof(MyIDType));

      for(n = 0; n < num_activebh; n++)
	{
	  bh_dmass[n] = 0.0;
	  bh_posx[n] = 0.0;
	  bh_posy[n] = 0.0;
	  bh_posz[n] = 0.0;
	  bh_id[n] = 0;
	}

      for(n = 0; n < total_num_activebh; n++)
	{
	  tot_bh_dmass[n] = 0.0;
	  tot_bh_posx[n] = 0.0;
	  tot_bh_posy[n] = 0.0;
	  tot_bh_posz[n] = 0.0;
	  tot_bh_id[n] = 0;
	}

      for(n = FirstActiveParticle, l = 0; n >= 0; n = NextActiveParticle[n])
	if(P[n].Type == 5)
	  {
	    if(P[n].BH_Mass_bubbles > 0
	       && P[n].BH_Mass_bubbles > All.BlackHoleRadioTriggeringFactor * P[n].BH_Mass_ini)
	      {
#ifndef UNIFIED_FEEDBACK
		bh_dmass[l] = P[n].BH_Mass_bubbles - P[n].BH_Mass_ini;
#else
		bh_dmass[l] = P[n].BH_Mass_radio - P[n].BH_Mass_ini;
		P[n].BH_Mass_radio = P[n].BH_Mass;
#endif
		P[n].BH_Mass_ini = P[n].BH_Mass;
		P[n].BH_Mass_bubbles = P[n].BH_Mass;

		bh_posx[l] = P[n].Pos[0];
		bh_posy[l] = P[n].Pos[1];
		bh_posz[l] = P[n].Pos[2];
		bh_id[l] = P[n].ID;

		l++;
	      }
	  }
      common_num_activebh = mymalloc("common_num_activebh", NTask * sizeof(int));
      disp = mymalloc("disp", NTask * sizeof(int));

      MPI_Allgather(&num_activebh, 1, MPI_INT, common_num_activebh, 1, MPI_INT, MPI_COMM_WORLD);

      for(k = 1, disp[0] = 0; k < NTask; k++)
	disp[k] = disp[k - 1] + common_num_activebh[k - 1];


      MPI_Allgatherv(bh_dmass, num_activebh, MPI_DOUBLE, tot_bh_dmass, common_num_activebh, disp, MPI_DOUBLE,
		     MPI_COMM_WORLD);
      MPI_Allgatherv(bh_posx, num_activebh, MPI_FLOAT, tot_bh_posx, common_num_activebh, disp, MPI_FLOAT,
		     MPI_COMM_WORLD);
      MPI_Allgatherv(bh_posy, num_activebh, MPI_FLOAT, tot_bh_posy, common_num_activebh, disp, MPI_FLOAT,
		     MPI_COMM_WORLD);
      MPI_Allgatherv(bh_posz, num_activebh, MPI_FLOAT, tot_bh_posz, common_num_activebh, disp, MPI_FLOAT,
		     MPI_COMM_WORLD);

#ifndef LONGIDS
      MPI_Allgatherv(bh_id, num_activebh, MPI_UNSIGNED, tot_bh_id, common_num_activebh, disp, MPI_UNSIGNED,
		     MPI_COMM_WORLD);
#else
      MPI_Allgatherv(bh_id, num_activebh, MPI_UNSIGNED_LONG_LONG, tot_bh_id, common_num_activebh, disp, MPI_UNSIGNED_LONG_LONG,
		     MPI_COMM_WORLD);
#endif      

      for(l = 0; l < total_num_activebh; l++)
	{
	  bh_center[0] = tot_bh_posx[l];
	  bh_center[1] = tot_bh_posy[l];
	  bh_center[2] = tot_bh_posz[l];

	  if(tot_bh_dmass[l] > 0)
	    bh_bubble(tot_bh_dmass[l], bh_center, tot_bh_id[l]);

	}

      myfree(disp);
      myfree(common_num_activebh);
      myfree(tot_bh_id);
      myfree(bh_id);
      myfree(tot_bh_posz);
      myfree(tot_bh_posy);
      myfree(tot_bh_posx);
      myfree(bh_posz);
      myfree(bh_posy);
      myfree(bh_posx);
      myfree(tot_bh_dmass);
      myfree(bh_dmass);
    }
  myfree(Ngblist);
#endif

  mdot = 0;
  mass_holes = 0;
  mass_real = 0;
  medd = 0;
  injection = 0;
  for(bin = 0; bin < TIMEBINS; bin++)
    if(TimeBinCount[bin])
      {
	mass_holes += TimeBin_BH_mass[bin];
	mass_real += TimeBin_BH_dynamicalmass[bin];
	mdot += TimeBin_BH_Mdot[bin];
	medd += TimeBin_BH_Medd[bin];
	injection += TimeBin_GAS_Injection[bin];
      }

  MPI_Reduce(&mass_holes, &total_mass_holes, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&mass_real, &total_mass_real, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&mdot, &total_mdot, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&medd, &total_mdoteddington, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&injection, &total_injection, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  if(ThisTask == 0)
    {
      /* convert to solar masses per yr */
      mdot_in_msun_per_year =
	total_mdot * (All.UnitMass_in_g / SOLAR_MASS) / (All.UnitTime_in_s / SEC_PER_YEAR);

      total_mdoteddington *= 1.0 / ((4 * M_PI * GRAVITY * C * PROTONMASS /
				     (0.1 * C * C * THOMPSON)) * All.UnitTime_in_s);

      fprintf(FdBlackHoles, "%g %d %g %g %g %g %g %g\n",
	      All.Time, All.TotBHs, total_mass_holes, total_mdot, mdot_in_msun_per_year,
	      total_mass_real, total_mdoteddington, total_injection);
      fflush(FdBlackHoles);
    }


  fflush(FdBlackHolesDetails);

  CPU_Step[CPU_BLACKHOLES] += measure_time();
}






int blackhole_evaluate(int target, int mode, int *nexport, int *nSend_local)
{

#ifdef KD_SMOOTHED_MOMENTUM_ACCRETION 
  MyFloat accreted_momentum[3] = {0,0,0}, dmin1, dmin2;
#endif

  int startnode, numngb, j, k, n, index, listindex = 0;
  MyIDType id;
  MyFloat *pos, *velocity, h_i, dt, mdot, rho, mass, bh_mass, csnd;
  double dx, dy, dz, r2, r, vrel;
  double hsearch;
  double fb_rho, fb_rho2=0.0, rho2=0.0;
  double hcache[4];
  double hsearchcache[4];

#ifdef UNIFIED_FEEDBACK
  double meddington;
#endif

#ifdef BH_KINETICFEEDBACK
  /*  double deltavel; */
  double activetime, activeenergy;
#endif
#ifdef BH_THERMALFEEDBACK
  double energy;
#endif
#ifdef REPOSITION_ON_POTMIN
  MyFloat minpotpos[3] = { 0, 0, 0 }, minpot = BHPOTVALUEINIT;
#endif

#ifdef LT_BH
  double contrib;
  double swk;
  double AltRho;
  double CutHsml;
  double adding_energy;
#endif
#ifdef LT_DF_BH_ONLY_RADIOMODE_KERNEL
  double mdotedd;		// [nota] inserisco variabile da esportare
#endif
  
  if(mode == 0)
    {
      pos = P[target].Pos;
      rho = P[target].b1.BH_Density;
      fb_rho = P[target].b1_FB.BH_FB_Density;
      mdot = P[target].BH_Mdot;
      dt = (P[target].TimeBin ? (1 << P[target].TimeBin) : 0) * All.Timebase_interval / hubble_a;
      h_i = PPP[target].Hsml;
      mass = P[target].Mass;
      bh_mass = P[target].BH_Mass;
      velocity = P[target].Vel;
      csnd =
	sqrt(GAMMA * P[target].b2.BH_Entropy *
	     pow(P[target].b1.BH_Density / (ascale * ascale * ascale), GAMMA_MINUS1));
      index = target;
      id = P[target].ID;
#ifdef BH_KINETICFEEDBACK
      activetime = P[target].ActiveTime;
      activeenergy = P[target].ActiveEnergy;
#endif
#ifdef LT_BH
      AltRho  = P[target].b9.BH_AltDensity;
#ifdef LT_BH_CUT_KERNEL      
      CutHsml = P[target].CutHsml;
#else
      CutHsml = PPP[target].Hsml;
#endif

#ifdef LT_DF_BH_ONLY_RADIOMODE_KERNEL
      mdotedd = P[target].BH_MdotEddington;
#ifdef LT_BH_CUT_KERNEL      
      if( mdotedd >= All.BH_radio_treshold )
        CutHsml = PPP[target].Hsml;
#endif
#endif
#endif      
    }
  else
    {
      pos = BlackholeDataGet[target].Pos;
      rho = BlackholeDataGet[target].Density;
      fb_rho = BlackholeDataGet[target].FBDensity;
      mdot = BlackholeDataGet[target].Mdot;
      dt = BlackholeDataGet[target].Dt;
      h_i = BlackholeDataGet[target].Hsml;
      mass = BlackholeDataGet[target].Mass;
      bh_mass = BlackholeDataGet[target].BH_Mass;
      velocity = BlackholeDataGet[target].Vel;
      csnd = BlackholeDataGet[target].Csnd;
      index = BlackholeDataGet[target].Index;
      id = BlackholeDataGet[target].ID;
#ifdef BH_KINETICFEEDBACK
      activetime = BlackholeDataGet[target].ActiveTime;
      activeenergy = BlackholeDataGet[target].ActiveEnergy;
#endif
#ifdef LT_BH
      AltRho  = BlackholeDataGet[target].AltRho;
#ifdef LT_BH_CUT_KERNEL      
      CutHsml = BlackholeDataGet[target].CutHsml;
#else
      CutHsml = h_i;
#endif
#endif
      
#ifdef LT_DF_BH_ONLY_RADIOMODE_KERNEL
      mdotedd = BlackholeDataGet[target].BH_MdotEddington;
#ifdef LT_BH_CUT_KERNEL      
      if( mdotedd >= All.BH_radio_treshold )
        CutHsml = h_i;
#endif
#endif      
    }

#ifdef LT_BH
  contrib = 0;
#endif
  
  hsearch = density_decide_hsearch(5, h_i);
  density_kernel_cache_h(h_i, hcache);
  density_kernel_cache_h(hsearch, hsearchcache);
  /* initialize variables before SPH loop is started */

  /* Now start the actual SPH computation for this particle */
  if(mode == 0)
    {
      startnode = All.MaxPart;	/* root node */
    }
  else
    {
      startnode = BlackholeDataGet[target].NodeList[0];
      startnode = Nodes[startnode].u.d.nextnode;	/* open it */
    }

  while(startnode >= 0)
    {
      while(startnode >= 0)
	{
	  numngb = ngb_treefind_blackhole(pos, hsearch, target, &startnode, mode, nexport, nSend_local);

	  if(numngb < 0)
	    return -1;

	  for(n = 0; n < numngb; n++)
	    {
	      j = Ngblist[n];

	      if(P[j].Mass > 0)
		{
		  if(mass > 0)
		    {
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

#ifdef REPOSITION_ON_POTMIN
			  /* if this option is switched on, we may also encounter dark matter particles or stars */
		      if(r2 < hcache[H2])
			{
			  if(P[j].p.Potential < minpot)
			    {
			      if(P[j].Type == 0 || P[j].Type == 1 || P[j].Type == 4 || P[j].Type == 5)
				{
				  /* compute relative velocities */

				  for(k = 0, vrel = 0; k < 3; k++)
				    vrel += (P[j].Vel[k] - velocity[k]) * (P[j].Vel[k] - velocity[k]);

				  vrel = sqrt(vrel) / ascale;

				  if(vrel <= 0.25 * csnd)
				    {
				      minpot = P[j].p.Potential;
				      for(k = 0; k < 3; k++)
					minpotpos[k] = P[j].Pos[k];
				    }
				}
			    }
                        }
#endif
		       if(P[j].Type == 5 && r2 < hcache[H2])	/* we have a black hole merger */
			    {
			      if(id != P[j].ID)
				{
				  /* compute relative velocity of BHs */

				  for(k = 0, vrel = 0; k < 3; k++)
				    vrel += (P[j].Vel[k] - velocity[k]) * (P[j].Vel[k] - velocity[k]);

				  vrel = sqrt(vrel) / ascale;

				  if(vrel > 0.5 * csnd)
				    {
				      fprintf(FdBlackHolesDetails,
					      "ThisTask=%d, time=%g: id="IDFMT" would like to swallow "IDFMT", but vrel=%g csnd=%g\n",
					      ThisTask, All.Time, id, P[j].ID, vrel, csnd);
				    }
				  else
				    {
				      if(P[j].SwallowID < id && P[j].ID < id)
					P[j].SwallowID = id;
				    }
				}
			    }
		      if(P[j].Type == 0 && r2 < hcache[H2])
		            {
			      /* here we have a gas particle */

			      r = sqrt(r2);
                              double wk;
                              density_kernel(r, hcache, &wk, NULL);
                              rho2 += P[j].Mass * wk;
#ifdef SWALLOWGAS
			      /* compute accretion probability */
			      double p, w;

#ifndef LT_BH_ACCRETE_SLICES                              
			      if((bh_mass - mass) > 0 && rho > 0)
				p = (bh_mass - mass) * wk / rho;
#else
			      if((bh_mass - mass) > 0 && rho > 0)
				p = (bh_mass - mass) * wk / rho * (All.NBHslices - SphP[j].NSlicesSwallowed);
#endif
			      else
				p = 0;

			      /* compute random number, uniform in [0,1] */
			      w = get_random_number(P[j].ID);
			      if(w < p)
				{
				  if(P[j].SwallowID < id)
				    P[j].SwallowID = id;
				}
#ifdef KD_SMOOTHED_MOMENTUM_ACCRETION 
			      for(k = 0; k < 3; k++)
				accreted_momentum[k] += DMIN(p,1.0) * P[j].Mass * P[j].Vel[k];
#endif
#endif

                          }
                      if(P[j].Type == 0 && r2 < hsearchcache[H2]) {
			    r = sqrt(r2);
                            double wk;
                            density_kernel(r, hsearchcache, &wk, NULL);
			    if(P[j].Mass > 0)
				{
#ifndef LT_BH                                  
#ifdef BH_THERMALFEEDBACK
#ifndef UNIFIED_FEEDBACK
				  energy = All.BlackHoleFeedbackFactor * 0.1 * mdot * dt *
				    pow(C / All.UnitVelocity_in_cm_per_s, 2);
                                  
				  if(fb_rho > 0)
                                    {
                                      SphP[j].i.dInjected_BH_Energy += FLT(energy * P[j].Mass * wk / fb_rho);
                                        fb_rho2 += P[j].Mass * wk;
#ifdef LT_BH_ACCRETE_SLICES                 
                                      if(P[j].SwallowID == id)
                                        SphP[j].i.dInjected_BH_Energy -= FLT(energy * P[j].Mass / (All.NBHslices - SphP[j].NSlicesSwallowed) * wk / fb_rho);
#endif
                                    }

#else
				  meddington = (4 * M_PI * GRAVITY * C *
						PROTONMASS / (0.1 * C * C * THOMPSON)) * bh_mass *
				    All.UnitTime_in_s;

				  if(mdot > All.RadioThreshold * meddington)
				    {
				      energy =
					All.BlackHoleFeedbackFactor * 0.1 * mdot * dt * pow(C /
											    All.
											    UnitVelocity_in_cm_per_s,
											    2);
				      if(fb_rho > 0)
					SphP[j].i.dInjected_BH_Energy += FLT(energy * P[j].Mass * wk / fb_rho);
				    }
#endif
#endif
#else                                                                    /* LT_BH */

#ifdef LT_DF_BH_ONLY_RADIOMODE_KERNEL
                                  if(mdotedd < All.BH_radio_treshold)
                                    {
#endif
                                  
                                  if(AltRho == 0)
                                    continue;
#ifdef LT_BH_CUT_KERNEL
                                  if(r >  CutHsml)
                                    continue;                            /* note: in case LT_DF_BH_ONLY_RADIOMODE_KERNEL is on  */
                                                                         /* this condition is never fullfilled when radio mode  */
                                                                         /* is off, since in that case CutHsml is equal to Hsml */
#endif

                                                                         /* :: LT_BH calculate weight */
#ifdef LT_BH_DONOT_USEDENSITY_IN_KERNEL
                                  swk = 1.0;
#else
                                  swk = 1.0 / SphP[j].d.Density;
#endif

#ifndef LT_BH_USETOPHAT_KERNEL
                                  swk *= wk;
#endif
                                                                         /* :: end of LT_BH weight    */

#ifdef LT_DF_BH_ONLY_RADIOMODE_KERNEL
                                    }
                                  else
                                    {
                                      AltRho = fb_rho;
                                      swk = wk;
                                    }
#endif

#ifdef BH_THERMALFEEDBACK                                                /* ==: THERMAL :======================= */
                                  energy = All.BlackHoleFeedbackFactor * 0.1 * mdot * dt *
                                    pow(C / All.UnitVelocity_in_cm_per_s, 2);
				  if(AltRho > 0)
				    adding_energy = FLT(energy * P[j].Mass * swk / AltRho);
                                  else
				    adding_energy = 0;
#ifdef LT_BH_ACCRETE_SLICES
                                  if(P[j].SwallowID == id && AltRho > 0)
                                    adding_energy -= energy * P[j].Mass / (All.NBHslices - SphP[j].NSlicesSwallowed) * swk / AltRho;
#endif
				  if(AltRho > 0)   
				    contrib += P[j].Mass * swk / AltRho;

#endif                                                                   /* ----------------: end of thermal :-- */

#ifdef BH_KINETICFEEDBACK                                                /* ==: KINETIC :======================= */
                                  if(activetime > All.BlackHoleActiveTime)
                                    {
				      if(AltRho > 0) 
					adding_energy = FLT(activeenergy * P[j].Mass * swk / AltRho);
				      else
					adding_energy = 0;
#ifdef LT_BH_ACCRETE_SLICES
                                      if(P[j].SwallowID == id && AltRho > 0)
                                        adding_energy -= activeenergy * P[j].Mass / (All.NBHslices - SphP[j].NSlicesSwallowed) * swk / AltRho;
#endif
                                    }
#endif                                                                   /* ----------------: end of kinetic :-- */

                                                                         /* :: here it should be put the control on the temperature */
                                  
                                  SphP[j].i.dInjected_BH_Energy += FLT(adding_energy);
                              
#endif                                                                   /* closes LT_BH */
				}

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
	      startnode = BlackholeDataGet[target].NodeList[listindex];
	      if(startnode >= 0)
		startnode = Nodes[startnode].u.d.nextnode;	/* open it */
	    }
	}
    }

  /* Now collect the result at the right place */
  if(mode == 0)
    {
#ifdef REPOSITION_ON_POTMIN
      for(k = 0; k < 3; k++)
	P[target].BH_MinPotPos[k] = minpotpos[k];
      P[target].BH_MinPot = minpot;
#endif
#ifdef LT_BH      
      P[target].Normalization = contrib;
#endif
#ifdef KD_SMOOTHED_MOMENTUM_ACCRETION 
      for(k = 0; k < 3; k++)
	P[target].b6.dBH_accreted_momentum[k] = accreted_momentum[k];
#endif
    }
  else
    {
#ifdef REPOSITION_ON_POTMIN
      for(k = 0; k < 3; k++)
	BlackholeDataResult[target].BH_MinPotPos[k] = minpotpos[k];
      BlackholeDataResult[target].BH_MinPot = minpot;
#endif
#ifdef LT_BH      
      BlackholeDataResult[target].PartialContrib = contrib;
#endif
#ifdef KD_SMOOTHED_MOMENTUM_ACCRETION 
      for(k = 0; k < 3; k++)
	BlackholeDataResult[target].AccretedMomentum[k] = accreted_momentum[k];
#endif

    }

#if 0
    if(fb_rho2 >0 || rho2 > 0)
      printf("id"IDFMT" Task %d fb_rho: %g fb_rho2 %g rho %g rho2 %g\n", id, ThisTask, fb_rho, fb_rho2, rho, rho2);

#endif
  return 0;
}


int blackhole_evaluate_swallow(int target, int mode, int *nexport, int *nSend_local)
{
  int startnode, numngb, j, k, n, bin, listindex = 0;
  MyIDType id;
  MyLongDouble accreted_mass, accreted_BH_mass, accreted_momentum[3];
  MyFloat *pos, h_i, bh_mass;

#ifdef BH_BUBBLES
  MyLongDouble accreted_BH_mass_bubbles = 0;
  MyLongDouble accreted_BH_mass_radio = 0;
#endif

#ifdef LT_BH_ACCRETE_SLICES
  MyLongDouble myaccreted_mass;
#ifdef LT_STELLAREVOLUTION
  double dec_factor;
#endif
#endif
  
  if(mode == 0)
    {
      pos = P[target].Pos;
      h_i = PPP[target].Hsml;
      id = P[target].ID;
      bh_mass = P[target].BH_Mass;
    }
  else
    {
      pos = BlackholeDataGet[target].Pos;
      h_i = BlackholeDataGet[target].Hsml;
      id = BlackholeDataGet[target].ID;
      bh_mass = BlackholeDataGet[target].BH_Mass;
    }

  accreted_mass = 0;
  accreted_BH_mass = 0;
  accreted_momentum[0] = accreted_momentum[1] = accreted_momentum[2] = 0;
#ifdef BH_COUNTPROGS
  int accreted_BH_progs = 0;
#endif


  if(mode == 0)
    {
      startnode = All.MaxPart;	/* root node */
    }
  else
    {
      startnode = BlackholeDataGet[target].NodeList[0];
      startnode = Nodes[startnode].u.d.nextnode;	/* open it */
    }

  while(startnode >= 0)
    {
      while(startnode >= 0)
	{
	  numngb = ngb_treefind_blackhole(pos, h_i, target, &startnode, mode, nexport, nSend_local);

	  if(numngb < 0)
	    return -1;

	  for(n = 0; n < numngb; n++)
	    {
	      j = Ngblist[n];

	      if(P[j].SwallowID == id)
		{
		  if(P[j].Type == 5)	/* we have a black hole merger */
		    {
		      fprintf(FdBlackHolesDetails,
			      "ThisTask=%d, time=%g: id="IDFMT" swallows "IDFMT" (%g %g)\n",
			      ThisTask, All.Time, id, P[j].ID, bh_mass, P[j].BH_Mass);

		      accreted_mass += FLT(P[j].Mass);
		      accreted_BH_mass += FLT(P[j].BH_Mass);
#ifdef BH_BUBBLES
		      accreted_BH_mass_bubbles += FLT(P[j].BH_Mass_bubbles - P[j].BH_Mass_ini);
#ifdef UNIFIED_FEEDBACK
		      accreted_BH_mass_radio += FLT(P[j].BH_Mass_radio - P[j].BH_Mass_ini);
#endif
#endif
		      for(k = 0; k < 3; k++)
			accreted_momentum[k] += FLT(P[j].Mass * P[j].Vel[k]);

#ifdef BH_COUNTPROGS
		      accreted_BH_progs += P[j].BH_CountProgs;
#endif

		      bin = P[j].TimeBin;

		      TimeBin_BH_mass[bin] -= P[j].BH_Mass;
		      TimeBin_BH_dynamicalmass[bin] -= P[j].Mass;
		      TimeBin_BH_Mdot[bin] -= P[j].BH_Mdot;
		      if(P[j].BH_Mass > 0)
			TimeBin_BH_Medd[bin] -= P[j].BH_Mdot / P[j].BH_Mass;

		      P[j].Mass = 0;
		      P[j].BH_Mass = 0;
		      P[j].BH_Mdot = 0;

#ifdef BH_BUBBLES
		      P[j].BH_Mass_bubbles = 0;
		      P[j].BH_Mass_ini = 0;
#ifdef UNIFIED_FEEDBACK
		      P[j].BH_Mass_radio = 0;
#endif
#endif
		      N_BH_swallowed++;
		    }
		}

	      if(P[j].Type == 0)
		{
		  if(P[j].SwallowID == id)
		    {
#ifndef LT_BH_ACCRETE_SLICES
		      accreted_mass += FLT(P[j].Mass);

		      for(k = 0; k < 3; k++)
			accreted_momentum[k] += FLT(P[j].Mass * P[j].Vel[k]);

		      P[j].Mass = 0;
		      bin = P[j].TimeBin;
              TimeBin_GAS_Injection[bin] += SphP[j].i.dInjected_BH_Energy;
		      N_gas_swallowed++;
#else
                      if(SphP[j].NSlicesSwallowed == All.NBHslices)
                        {
                          myaccreted_mass = P[j].Mass;
                          P[j].Mass = 0;
#ifdef LT_STELLAREVOLUTION                          
                          for(k = 0; k < LT_NMetP; k++)
                            SphP[j].Metals[k] = 0;
#endif
                          N_gas_swallowed++;
                        }
                      else
                        {
                          myaccreted_mass = FLT(P[j].Mass / (All.NBHslices - SphP[j].NSlicesSwallowed));
#ifdef LT_STELLAREVOLUTION
                          dec_factor = myaccreted_mass / P[j].Mass;
                          for(k = 0; k < LT_NMetP; k++)
                            SphP[j].Metals[k] *= dec_factor;
#endif
                          P[j].Mass -= myaccreted_mass;
                          SphP[j].NSlicesSwallowed++;                      
                        }
                      
                      accreted_mass  += myaccreted_mass;
                      
                      for(k = 0; k < 3; k++)
                        accreted_momentum[k] += FLT(myaccreted_mass * P[j].Vel[k]);		  
                      
                      N_gas_slices_swallowed++;                      
#endif
		    }
		}
	    }
	}
      if(mode == 1)
	{
	  listindex++;
	  if(listindex < NODELISTLENGTH)
	    {
	      startnode = BlackholeDataGet[target].NodeList[listindex];
	      if(startnode >= 0)
		startnode = Nodes[startnode].u.d.nextnode;	/* open it */
	    }
	}
    }

  /* Now collect the result at the right place */
  if(mode == 0)
    {
      P[target].b4.dBH_accreted_Mass = accreted_mass;
      P[target].b5.dBH_accreted_BHMass = accreted_BH_mass;
#ifndef KD_SMOOTHED_MOMENTUM_ACCRETION 
      for(k = 0; k < 3; k++)
	P[target].b6.dBH_accreted_momentum[k] = accreted_momentum[k];
#endif
#ifdef BH_BUBBLES
      P[target].b7.dBH_accreted_BHMass_bubbles = accreted_BH_mass_bubbles;
#ifdef UNIFIED_FEEDBACK
      P[target].b8.dBH_accreted_BHMass_radio = accreted_BH_mass_radio;
#endif
#endif
#ifdef BH_COUNTPROGS
      P[target].BH_CountProgs += accreted_BH_progs;
#endif
    }
  else
    {
      BlackholeDataResult[target].Mass = accreted_mass;
      BlackholeDataResult[target].BH_Mass = accreted_BH_mass;
#ifndef KD_SMOOTHED_MOMENTUM_ACCRETION 
      for(k = 0; k < 3; k++)
	BlackholeDataResult[target].AccretedMomentum[k] = accreted_momentum[k];
#endif
#ifdef BH_BUBBLES
      BlackholeDataResult[target].BH_Mass_bubbles = accreted_BH_mass_bubbles;
#ifdef UNIFIED_FEEDBACK
      BlackholeDataResult[target].BH_Mass_radio = accreted_BH_mass_radio;
#endif
#endif
#ifdef BH_COUNTPROGS
      BlackholeDataResult[target].BH_CountProgs = accreted_BH_progs;
#endif
    }

  return 0;
}




int ngb_treefind_blackhole(MyDouble searchcenter[3], MyFloat hsml, int target, int *startnode, int mode,
			   int *nexport, int *nsend_local)
{
  int numngb, no, p, task, nexport_save;
  struct NODE *current;
  MyDouble dx, dy, dz, dist;

#ifdef PERIODIC
  MyDouble xtmp;
#endif
  nexport_save = *nexport;

  numngb = 0;
  no = *startnode;

  while(no >= 0)
    {
      if(no < All.MaxPart)	/* single particle */
	{
	  p = no;
	  no = Nextnode[no];

#ifndef REPOSITION_ON_POTMIN
	  if(P[p].Type != 0 && P[p].Type != 5)
	    continue;
#endif
	  dist = hsml;
	  dx = NGB_PERIODIC_LONG_X(P[p].Pos[0] - searchcenter[0]);
	  if(dx > dist)
	    continue;
	  dy = NGB_PERIODIC_LONG_Y(P[p].Pos[1] - searchcenter[1]);
	  if(dy > dist)
	    continue;
	  dz = NGB_PERIODIC_LONG_Z(P[p].Pos[2] - searchcenter[2]);
	  if(dz > dist)
	    continue;
	  if(dx * dx + dy * dy + dz * dz > dist * dist)
	    continue;

	  Ngblist[numngb++] = p;
	}
      else
	{
	  if(no >= All.MaxPart + MaxNodes)	/* pseudo particle */
	    {
	      if(mode == 1)
		endrun(12312);

	      if(Exportflag[task = DomainTask[no - (All.MaxPart + MaxNodes)]] != target)
		{
		  Exportflag[task] = target;
		  Exportnodecount[task] = NODELISTLENGTH;
		}

	      if(Exportnodecount[task] == NODELISTLENGTH)
		{
		  if(*nexport >= All.BunchSize)
		    {
		      *nexport = nexport_save;
		      for(task = 0; task < NTask; task++)
			nsend_local[task] = 0;
		      for(no = 0; no < nexport_save; no++)
			nsend_local[DataIndexTable[no].Task]++;
		      return -1;
		    }
		  Exportnodecount[task] = 0;
		  Exportindex[task] = *nexport;
		  DataIndexTable[*nexport].Task = task;
		  DataIndexTable[*nexport].Index = target;
		  DataIndexTable[*nexport].IndexGet = *nexport;
		  *nexport = *nexport + 1;
		  nsend_local[task]++;
		}

	      DataNodeList[Exportindex[task]].NodeList[Exportnodecount[task]++] =
		DomainNodeIndex[no - (All.MaxPart + MaxNodes)];

	      if(Exportnodecount[task] < NODELISTLENGTH)
		DataNodeList[Exportindex[task]].NodeList[Exportnodecount[task]] = -1;

	      no = Nextnode[no - MaxNodes];
	      continue;
	    }

	  current = &Nodes[no];

	  if(mode == 1)
	    {
	      if(current->u.d.bitflags & (1 << BITFLAG_TOPLEVEL))	/* we reached a top-level node again, which means that we are done with the branch */
		{
		  *startnode = -1;
		  return numngb;
		}
	    }

	  no = current->u.d.sibling;	/* in case the node can be discarded */

	  dist = hsml + 0.5 * current->len;;
	  dx = NGB_PERIODIC_LONG_X(current->center[0] - searchcenter[0]);
	  if(dx > dist)
	    continue;
	  dy = NGB_PERIODIC_LONG_Y(current->center[1] - searchcenter[1]);
	  if(dy > dist)
	    continue;
	  dz = NGB_PERIODIC_LONG_Z(current->center[2] - searchcenter[2]);
	  if(dz > dist)
	    continue;
	  /* now test against the minimal sphere enclosing everything */
	  dist += FACT1 * current->len;
	  if(dx * dx + dy * dy + dz * dz > dist * dist)
	    continue;

	  no = current->u.d.nextnode;	/* ok, we need to open the node */
	}
    }

  *startnode = -1;
  return numngb;
}

#ifdef BH_BUBBLES
void bh_bubble(double bh_dmass, MyFloat center[3], MyIDType BH_id)
{
  double phi, theta;
  double dx, dy, dz, rr, r2, dE;
  double E_bubble, totE_bubble;
  double BubbleDistance = 0.0, BubbleRadius = 0.0, BubbleEnergy = 0.0;
  double ICMDensity;
  double Mass_bubble, totMass_bubble;
  double u_to_temp_fac;
  MyDouble pos[3];
  int numngb, tot_numngb, startnode, numngb_inbox;
  int n, i, j, dummy;

#ifdef CR_BUBBLES
  double tinj = 0.0, instant_reheat = 0.0;
  double sum_instant_reheat = 0.0, tot_instant_reheat = 0.0;
#endif

  u_to_temp_fac = (4 / (8 - 5 * (1 - HYDROGEN_MASSFRAC))) * PROTONMASS /
    BOLTZMANN * GAMMA_MINUS1 * All.UnitEnergy_in_cgs / All.UnitMass_in_g;

  if(All.ComovingIntegrationOn)
    {

      BubbleDistance = All.BubbleDistance;
      BubbleRadius = All.BubbleRadius;

      /*switch to comoving if it is assumed that Rbub should be constant with redshift */

      /* BubbleDistance = All.BubbleDistance / All.Time;
         BubbleRadius = All.BubbleRadius / All.Time; */
    }
  else
    {
      BubbleDistance = All.BubbleDistance;
      BubbleRadius = All.BubbleRadius;
    }

  BubbleEnergy = All.RadioFeedbackFactor * 0.1 * bh_dmass * All.UnitMass_in_g / All.HubbleParam * pow(C, 2);	/*in cgs units */

  phi = 2 * M_PI * get_random_number(BH_id);
  theta = acos(2 * get_random_number(BH_id + 1) - 1);
  rr = pow(get_random_number(BH_id + 2), 1. / 3.) * BubbleDistance;

  pos[0] = sin(theta) * cos(phi);
  pos[1] = sin(theta) * sin(phi);
  pos[2] = cos(theta);

  for(i = 0; i < 3; i++)
    pos[i] *= rr;

  for(i = 0; i < 3; i++)
    pos[i] += center[i];


  /* First, let's see how many particles are in the bubble of the default radius */

  numngb = 0;
  E_bubble = 0.;
  Mass_bubble = 0.;

  startnode = All.MaxPart;
  do
    {
      numngb_inbox = ngb_treefind_variable(pos, BubbleRadius, -1, &startnode, 0, &dummy, &dummy);

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

	  if(r2 < BubbleRadius * BubbleRadius)
	    {
	      if(P[j].Type == 0)
		{
		  numngb++;

		  if(All.ComovingIntegrationOn)
		    E_bubble +=
		      SphP[j].Entropy * P[j].Mass * pow(SphP[j].d.Density / pow(All.Time, 3),
							GAMMA_MINUS1) / GAMMA_MINUS1;
		  else
		    E_bubble +=
		      SphP[j].Entropy * P[j].Mass * pow(SphP[j].d.Density, GAMMA_MINUS1) / GAMMA_MINUS1;

		  Mass_bubble += P[j].Mass;
		}
	    }
	}
    }
  while(startnode >= 0);


  MPI_Allreduce(&numngb, &tot_numngb, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&E_bubble, &totE_bubble, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&Mass_bubble, &totMass_bubble, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  totE_bubble *= All.UnitEnergy_in_cgs;

  if(totMass_bubble > 0)
    {
      if(ThisTask == 0)
	{
	  printf("found %d particles in bubble with energy %g and total mass %g \n",
		 tot_numngb, totE_bubble, totMass_bubble);
	  fflush(stdout);
	}


      /*calculate comoving density of ICM inside the bubble */

      ICMDensity = totMass_bubble / (4.0 * M_PI / 3.0 * pow(BubbleRadius, 3));

      if(All.ComovingIntegrationOn)
	ICMDensity = ICMDensity / (pow(All.Time, 3));	/*now physical */

      /*Rbub=R0*[(Ejet/Ejet,0)/(rho_ICM/rho_ICM,0)]^(1./5.) - physical */

      rr = rr / BubbleDistance;

      BubbleRadius =
	All.BubbleRadius * pow((BubbleEnergy * All.DefaultICMDensity / (All.BubbleEnergy * ICMDensity)),
			       1. / 5.);

      BubbleDistance =
	All.BubbleDistance * pow((BubbleEnergy * All.DefaultICMDensity / (All.BubbleEnergy * ICMDensity)),
				 1. / 5.);

      if(All.ComovingIntegrationOn)
	{
	  /*switch to comoving if it is assumed that Rbub should be constant with redshift */
	  /* BubbleRadius = BubbleRadius / All.Time;
	     BubbleDistance = BubbleDistance / All.Time; */
	}

      /*recalculate pos */
      rr = rr * BubbleDistance;

      pos[0] = sin(theta) * cos(phi);
      pos[1] = sin(theta) * sin(phi);
      pos[2] = cos(theta);

      for(i = 0; i < 3; i++)
	pos[i] *= rr;

      for(i = 0; i < 3; i++)
	pos[i] += center[i];

      /* now find particles in Bubble again,
         and recalculate number, mass and energy */

      numngb = 0;
      E_bubble = 0.;
      Mass_bubble = 0.;
      tot_numngb = 0;
      totE_bubble = 0.;
      totMass_bubble = 0.;

      startnode = All.MaxPart;

      do
	{
	  numngb_inbox = ngb_treefind_variable(pos, BubbleRadius, -1, &startnode, 0, &dummy, &dummy);

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

	      if(r2 < BubbleRadius * BubbleRadius)
		{
		  if(P[j].Type == 0 && P[j].Mass > 0)
		    {
		      numngb++;

		      if(All.ComovingIntegrationOn)
			E_bubble +=
			  SphP[j].Entropy * P[j].Mass * pow(SphP[j].d.Density / pow(All.Time, 3),
							    GAMMA_MINUS1) / GAMMA_MINUS1;
		      else
			E_bubble +=
			  SphP[j].Entropy * P[j].Mass * pow(SphP[j].d.Density, GAMMA_MINUS1) / GAMMA_MINUS1;

		      Mass_bubble += P[j].Mass;
		    }
		}
	    }
	}
      while(startnode >= 0);


      MPI_Allreduce(&numngb, &tot_numngb, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(&E_bubble, &totE_bubble, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(&Mass_bubble, &totMass_bubble, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

      totE_bubble *= All.UnitEnergy_in_cgs;

      if(totMass_bubble > 0)
	{
	  if(ThisTask == 0)
	    {
	      printf("found %d particles in bubble of rescaled radius with energy %g and total mass %g \n",
		     tot_numngb, totE_bubble, totMass_bubble);
	      printf("energy shall be increased by: (Eini+Einj)/Eini = %g \n",
		     (BubbleEnergy + totE_bubble) / totE_bubble);
	      fflush(stdout);
	    }
	}

      /* now find particles in Bubble again, and inject energy */

#ifdef CR_BUBBLES
      sum_instant_reheat = 0.0;
      tot_instant_reheat = 0.0;
#endif

      startnode = All.MaxPart;

      do
	{
	  numngb_inbox = ngb_treefind_variable(pos, BubbleRadius, -1, &startnode, 0, &dummy, &dummy);

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

	      if(r2 < BubbleRadius * BubbleRadius)
		{
		  if(P[j].Type == 0 && P[j].Mass > 0)
		    {
		      /* energy we want to inject in this particle */

		      if(All.StarformationOn)
			dE = ((BubbleEnergy / All.UnitEnergy_in_cgs) / totMass_bubble) * P[j].Mass;
		      else
			dE = (BubbleEnergy / All.UnitEnergy_in_cgs) / tot_numngb;

		      if(u_to_temp_fac * dE / P[j].Mass > 5.0e9)
			dE = 5.0e9 * P[j].Mass / u_to_temp_fac;

#ifndef CR_BUBBLES
		      if(All.ComovingIntegrationOn)
			SphP[j].Entropy +=
			  GAMMA_MINUS1 * dE / P[j].Mass / pow(SphP[j].d.Density / pow(All.Time, 3),
							      GAMMA_MINUS1);
		      else
			SphP[j].Entropy +=
			  GAMMA_MINUS1 * dE / P[j].Mass / pow(SphP[j].d.Density, GAMMA_MINUS1);
#else

		      if(All.ComovingIntegrationOn)
			tinj = 10.0 * All.HubbleParam * hubble_a / All.UnitTime_in_Megayears;
		      else
			tinj = 10.0 * All.HubbleParam / All.UnitTime_in_Megayears;

		      instant_reheat =
			CR_Particle_SupernovaFeedback(&SphP[j], dE / P[j].Mass * All.CR_AGNEff, tinj);

		      if(instant_reheat > 0)
			{
			  if(All.ComovingIntegrationOn)
			    SphP[j].Entropy +=
			      instant_reheat * GAMMA_MINUS1 / pow(SphP[j].d.Density / pow(All.Time, 3),
								  GAMMA_MINUS1);
			  else
			    SphP[j].Entropy +=
			      instant_reheat * GAMMA_MINUS1 / pow(SphP[j].d.Density, GAMMA_MINUS1);
			}

		      if(All.CR_AGNEff < 1)
			{
			  if(All.ComovingIntegrationOn)
			    SphP[j].Entropy +=
			      (1 -
			       All.CR_AGNEff) * dE * GAMMA_MINUS1 / P[j].Mass / pow(SphP[j].d.Density /
										    pow(All.Time, 3),
										    GAMMA_MINUS1);
			  else
			    SphP[j].Entropy +=
			      (1 - All.CR_AGNEff) * dE * GAMMA_MINUS1 / P[j].Mass / pow(SphP[j].d.Density,
											GAMMA_MINUS1);
			}


		      sum_instant_reheat += instant_reheat * P[j].Mass;
#endif

		    }
		}
	    }
	}
      while(startnode >= 0);

#ifdef CR_BUBBLES
      MPI_Allreduce(&sum_instant_reheat, &tot_instant_reheat, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

      if(ThisTask == 0)
	{
	  printf("Total BubbleEnergy %g Thermalized Energy %g \n", BubbleEnergy,
		 tot_instant_reheat * All.UnitEnergy_in_cgs);
	  fflush(stdout);

	}
#endif
    }
  else
    {
      if(ThisTask == 0)
	{
	  printf("No particles in bubble found! \n");
	  fflush(stdout);
	}

    }

}
#endif /* end of BH_BUBBLE */

#endif
