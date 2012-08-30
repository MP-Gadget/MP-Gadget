#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <gsl/gsl_math.h>

#include "allvars.h"
#include "proto.h"


/*! \file cs_enrichment.c
 *
 *  This file contains the routines for the enrichment by SNIa and SNII.
 *
 */

#ifdef CS_MODEL
#ifdef CS_ENRICH

#include "cs_metals.h"


static double distributed_metals, total_metals_distributed;

#ifdef CS_FEEDBACK
static double distributed_energy, total_energy_distributed;
#endif


static struct metaldata_in
{
  MyDouble Pos[3];
  MyFloat ZmReservoir[12];
  MyFloat Hsml;
#ifdef CS_FEEDBACK
  MyFloat EnergySN;
  MyFloat EnergySNCold;
#endif
  MyFloat NumNgb;
  int NodeList[NODELISTLENGTH];
} *MetalDataIn, *MetalDataGet;


static struct metaldata_tmp
{
  MyLongDouble Zm[12];
  MyLongDouble Energy;
  MyLongDouble EnergySN;
} *MetalDataTmp;



/* This function contains the main enrichment loop for exploding SNIa or SNII.*/
void cs_enrichment(void)
{
  int i, j, ndone_flag, dummy;
  int ndone, place, nimport;
  int ngrp, sendTask, recvTask;
  int nexport;
  int ik;
  double delta_metalsI, sum_SNI = 0, rate_SNI = 0, total_SNI;
  double delta_metalsII, sum_SNII = 0, rate_SNII = 0, total_SNII;
  double hubble_a, time_hubble_a, a3inv;
  double check = 0.;
  double total_metals_produced;

#ifdef CS_FEEDBACK
  double energy_produced, total_energy_produced;
#endif

  if(ThisTask == 0)
    {
      printf("... start enrichment phase = %d ...\n", Flag_phase);
      fflush(stdout);
    }



#ifdef CS_FEEDBACK
  if(Flag_phase == 1)
#endif
    {
      total_metals_produced = 0;
      distributed_metals = 0;
#ifdef CS_FEEDBACK
      distributed_energy = 0;
      energy_produced = 0;
      total_energy_produced = 0;
#endif
    }


  if(All.ComovingIntegrationOn)	/* Factors for comoving integration of hydro */
    {
      a3inv = 1 / (All.Time * All.Time * All.Time);
      hubble_a = All.Hubble * sqrt(All.Omega0 / (All.Time * All.Time * All.Time)
				   + (1 - All.Omega0 - All.OmegaLambda) / (All.Time * All.Time) +
				   All.OmegaLambda);
      time_hubble_a = All.Time * hubble_a;
    }
  else
    a3inv = time_hubble_a = 1;


  Ngblist = (int *) mymalloc("Ngblist", NumPart * sizeof(int));
  R2ngblist = (double *) mymalloc("R2ngblist", NumPart * sizeof(double));

  All.BunchSize =
    (int) ((All.BufferSize * 1024 * 1024) / (sizeof(struct data_index) + sizeof(struct data_nodelist) +
					     2 * sizeof(struct metaldata_in)));
  DataIndexTable =
    (struct data_index *) mymalloc("DataIndexTable", All.BunchSize * sizeof(struct data_index));
  DataNodeList =
    (struct data_nodelist *) mymalloc("DataNodeList", All.BunchSize * sizeof(struct data_nodelist));

  MetalDataTmp = (struct metaldata_tmp *) mymalloc("MetalDataTmp", N_gas * sizeof(struct metaldata_tmp));

  for(i = 0; i < N_gas; i++)
    {
      for(j = 0; j < 12; j++)
	MetalDataTmp[i].Zm[j] = 0;

      MetalDataTmp[i].Energy = 0;
      MetalDataTmp[i].EnergySN = 0;
    }

  for(i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i])
    {
      if((P[i].Type == 6 || P[i].Type == 7) && PPP[i].n.NumNgb != 0.)	/* CECILIA */
	{
#ifdef CS_FEEDBACK
	  if(Flag_phase == 1)	/* Only once we should produce the metals */
#endif
	    {
#ifdef CS_SNII
	      if(P[i].Type == 7)
		{
		  delta_metalsII = cs_SNII_yields(i);
		  sum_SNII += delta_metalsII;
		}
#endif
#ifdef CS_SNI
	      if(P[i].Type == 6)
		{
		  delta_metalsI = cs_SNI_yields(i);
		  sum_SNI += delta_metalsI;
		}
#endif
#if defined(CS_FEEDBACK)
	      energy_produced += P[i].EnergySN;
	      energy_produced += P[i].EnergySNCold;
#endif


	    }
	}
    }


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
	if(P[i].Type == 6 || P[i].Type == 7)
	  {
	    if(cs_enrichment_evaluate(i, 0, &nexport, Send_count) < 0)
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



      MetalDataGet = (struct metaldata_in *) mymalloc("MetalDataGet", nimport * sizeof(struct metaldata_in));
      MetalDataIn = (struct metaldata_in *) mymalloc("MetalDataIn", nexport * sizeof(struct metaldata_in));


      /* prepare particle data for export */
      for(j = 0; j < nexport; j++)
	{
	  place = DataIndexTable[j].Index;

	  MetalDataIn[j].Pos[0] = P[place].Pos[0];
	  MetalDataIn[j].Pos[1] = P[place].Pos[1];
	  MetalDataIn[j].Pos[2] = P[place].Pos[2];

	  MetalDataIn[j].Hsml = PPP[place].Hsml;
	  MetalDataIn[j].NumNgb = PPP[place].n.NumNgb;
	  for(ik = 0; ik < 12; ik++)
	    MetalDataIn[j].ZmReservoir[ik] = P[place].ZmReservoir[ik];
#ifdef CS_FEEDBACK
	  MetalDataIn[j].EnergySN = P[place].EnergySN;
	  MetalDataIn[j].EnergySNCold = P[place].EnergySNCold;
#endif
	  memcpy(MetalDataIn[j].NodeList,
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
		  MPI_Sendrecv(&MetalDataIn[Send_offset[recvTask]],
			       Send_count[recvTask] * sizeof(struct metaldata_in), MPI_BYTE,
			       recvTask, TAG_HYDRO_A,
			       &MetalDataGet[Recv_offset[recvTask]],
			       Recv_count[recvTask] * sizeof(struct metaldata_in), MPI_BYTE,
			       recvTask, TAG_HYDRO_A, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}
	    }
	}

      myfree(MetalDataIn);

      /* let's do the imported particles */
      for(j = 0; j < nimport; j++)
	cs_enrichment_evaluate(j, 1, &dummy, &dummy);


      if(i < 0)
	ndone_flag = 1;
      else
	ndone_flag = 0;

      MPI_Allreduce(&ndone_flag, &ndone, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

      myfree(MetalDataGet);
    }
  while(ndone < NTask);		/* this is the end of the do-loop over all particle exchanges */


  /* CECILIA */
  for(i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i])
    {
      if((P[i].Type == 6 || P[i].Type == 7) && PPP[i].n.NumNgb != 0.)	/* CECILIA */
	{
	  if(Flag_phase == 1)	/* COLD */
	    {
	      for(j = 0; j < 12; j++)
		P[i].ZmReservoir[j] *= (1 - All.SN_Energy_frac_cold);

	      P[i].EnergySN *= (1 - All.SN_Energy_frac_cold);
	      P[i].EnergySNCold = 0.;
	    }
	}
    }

  for(i = 0; i < N_gas; i++)
    {
      MyLongDouble sum;

      for(j = 0, sum = 0; j < 12; j++)
	{
	  sum += MetalDataTmp[i].Zm[j];
	  P[i].Zm[j] += FLT(MetalDataTmp[i].Zm[j]);
	}

      P[i].Mass += FLT(sum);

      SphP[i].Entropy +=
	FLT(MetalDataTmp[i].Energy) / P[i].Mass * GAMMA_MINUS1 / pow(SphP[i].d.Density * a3inv, GAMMA_MINUS1);
      P[i].EnergySN += FLT(MetalDataTmp[i].EnergySN);
    }


#ifdef CS_FEEDBACK
  if(Flag_phase == 1)
    {
      MPI_Reduce(&energy_produced, &total_energy_produced, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
#ifdef CS_TESTS
      Energy_feedback = total_energy_produced;
#endif
    }
#endif

#ifdef CS_FEEDBACK
  if(Flag_phase == 1)
#endif
    {
      MPI_Reduce(&sum_SNI, &total_SNI, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
      MPI_Reduce(&sum_SNII, &total_SNII, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);


      if(ThisTask == 0)
	{
	  if(All.TimeStep > 0)
	    {
	      rate_SNII = total_SNII / (All.TimeStep / time_hubble_a);
	      rate_SNI = total_SNI / (All.TimeStep / time_hubble_a);
	    }
	  else
	    {
	      rate_SNII = 0;
	      rate_SNI = 0;
	    }

	  total_metals_produced = total_SNI + total_SNII;
	}
    }

  if(ThisTask == 0)
    {
#ifndef CS_FEEDBACK
      if(rate_SNI > 0 || rate_SNII > 0)
	fprintf(FdSN, "%g %g %g %g %g \n", All.Time, total_SNI, rate_SNI, total_SNII, rate_SNII);
      fflush(FdSN);
#else
      if(Flag_phase == 1)
	{
	  if(rate_SNI > 0 || rate_SNII > 0 || total_energy_produced > 0)
	    fprintf(FdSN, "%g %g %g %g %g %g\n", All.Time, total_SNI, rate_SNI, total_SNII, rate_SNII,
		    total_energy_produced);
	  fflush(FdSN);
	}
      else if(rate_SNI > 0 || rate_SNII > 0)	/* total_energy_produced still has info from Flag_phase=1 */
	endrun(3852387);	/* metals can only be produced when Flag_phase = 1 */
#endif
    }


  if(Flag_phase == 2)
    {
      MPI_Reduce(&distributed_metals, &total_metals_distributed, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
#ifdef CS_FEEDBACK
      MPI_Reduce(&distributed_energy, &total_energy_distributed, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
#endif
    }


  if(ThisTask == 0 && Flag_phase == 2)
    if(total_metals_produced > 0)
      {
	/* Some checks in case something is not working properly */
#ifdef CS_FEEDBACK
	if(total_energy_produced - total_energy_distributed > 1.e-8)	/* compare the rest because of rounding */
	  endrun(297546);
#endif
	if((total_metals_produced - total_metals_distributed) > 1.e-8)	/* same for metals */
	  endrun(297567);

#ifdef CS_TESTS
	fprintf(FdSNTest,
		"Test2=%g total_energy_produced=%g total_energy_distributed=%g total_metals_produced=%g total_metals_distributed=%g\n",
		All.Time, total_energy_produced, total_energy_distributed, total_metals_produced,
		total_metals_distributed);
	fflush(FdSNTest);
#endif
      }



  /* do final operations */

  if(Flag_phase == 0 || Flag_phase == 2)
    for(i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i])
      if(P[i].Type == 6 || P[i].Type == 7)
	{
	  if(PPP[i].n.NumNgb != 0.)	/* CECILIA */
	    {

	      check = 0;
	      for(ik = 0; ik < 12; ik++)
		check += P[i].ZmReservoir[ik];
	      if(check == 0)
		{
		  printf("ERROR Part=%d enters cleaning in enrichment with Reservoir=%g\n", P[i].ID, check);
		  endrun(334);
		}

	      for(ik = 0; ik < 12; ik++)
		P[i].ZmReservoir[ik] = 0;

	      if(P[i].Type == 6)	/* SNIa */
		{
		  PPP[i].Hsml = 0;	/* to avoid exploding as SNIa again */
		  PPP[i].n.NumNgb = 0;	/* to avoid exploding as SNII again */
		}

	      if(P[i].Type == 7)	/* SNII */
		PPP[i].n.NumNgb = 0;	/* to avoid exploding as SNII again */


	      P[i].Type &= 4;

#ifdef CS_FEEDBACK
	      P[i].EnergySN = 0;	/* only stars should enter here */
	      P[i].EnergySNCold = 0;	/* only stars should enter here */
#endif
	    }
	  else
	    {
	      /* if the particle comes with NumNgb = 0 it did not distribute anything --> NumNgb should be set
	         again to a non-zero value or these metals/energy will never be distributed */
	      PPP[i].n.NumNgb = 1.;
	    }
	}

  myfree(MetalDataTmp);
  myfree(DataNodeList);
  myfree(DataIndexTable);
  myfree(R2ngblist);
  myfree(Ngblist);

  if(ThisTask == 0)
    {
      printf("...  enrichment phase = %d done...\n", Flag_phase);
      fflush(stdout);
    }

}



/*! This function represents the core of the enrichment calculation. The
 *  target particle may either be local, or reside in the communication
 *  buffer.
 */
int cs_enrichment_evaluate(int target, int mode, int *nexport, int *nsend_local)
{
  int j, n, ik;
  int startnode, numngb_inbox, listindex = 0;
  double h, h2, hinv, hinv3;
  double wk;
  double dx, dy, dz, r, r2, u, mass_j;
  double inv_wk_i;
  MyDouble *pos;
  MyFloat *reservoir;
  MyFloat numngb;
  double frac_phase = 0;

#ifdef CS_FEEDBACK
  double a3inv;
  double xhyd, yhel, ne, mu;
  double energy, temp;
  double energySN, energySNCold;
#endif

#ifdef PERIODIC
  double boxSize, boxHalf;

  boxSize = All.BoxSize;
  boxHalf = 0.5 * All.BoxSize;
#endif

#ifdef CS_FEEDBACK
  if(All.ComovingIntegrationOn)
    a3inv = 1 / (All.Time * All.Time * All.Time);
  else
    a3inv = 1;

  if(Flag_phase == 1)
    frac_phase = All.SN_Energy_frac_cold;
  if(Flag_phase == 2)		/* CECILIA */
    frac_phase = 1.;		/* now we substract from the reservoirs what is distributed in Flag_phase=1 */
#else
  frac_phase = 1;
#endif

  if(mode == 0)
    {
      pos = P[target].Pos;
      h = PPP[target].Hsml;
      numngb = PPP[target].n.NumNgb;
      reservoir = P[target].ZmReservoir;
#ifdef CS_FEEDBACK
      energySN = P[target].EnergySN;
      energySNCold = P[target].EnergySNCold;
#endif
    }
  else
    {
      pos = MetalDataGet[target].Pos;
      h = MetalDataGet[target].Hsml;
      numngb = MetalDataGet[target].NumNgb;
      reservoir = MetalDataGet[target].ZmReservoir;
#ifdef CS_FEEDBACK
      energySN = MetalDataGet[target].EnergySN;
      energySNCold = MetalDataGet[target].EnergySNCold;
#endif
    }

  h2 = h * h;
  hinv = 1.0 / h;
  hinv3 = hinv * hinv * hinv;

  inv_wk_i = 4 * M_PI / 3.0 / numngb;	/* renormalization of  the kernel */

  if(mode == 0)
    {
      startnode = All.MaxPart;	/* root node */
    }
  else
    {
      startnode = MetalDataGet[target].NodeList[0];
      startnode = Nodes[startnode].u.d.nextnode;	/* open it */
    }

  while(startnode >= 0)
    {
      while(startnode >= 0)
	{
	  numngb_inbox =
	    cs_ngb_treefind_variable_phases(pos, h, target, &startnode, mode, nexport, nsend_local);

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

		  u = r * hinv;

		  if(u < 0.5)
		    wk = (KERNEL_COEFF_1 + KERNEL_COEFF_2 * (u - 1) * u * u);
		  else
		    wk = KERNEL_COEFF_5 * (1.0 - u) * (1.0 - u) * (1.0 - u);

		  mass_j = P[j].Mass;

#ifdef CS_FEEDBACK
		  xhyd = P[j].Zm[6] / P[j].Mass;
		  yhel = (1 - xhyd) / (4. * xhyd);

		  ne = SphP[j].Ne;
		  mu = (1 + 4 * yhel) / (1 + yhel + ne);
		  energy = SphP[j].Entropy * P[j].Mass / GAMMA_MINUS1 * pow(SphP[j].d.Density * a3inv, GAMMA_MINUS1);	/* Total Energys */
		  temp = GAMMA_MINUS1 / BOLTZMANN * energy / P[j].Mass * PROTONMASS * mu;
		  temp *= All.UnitEnergy_in_cgs / All.UnitMass_in_g;	/* Temperature in Kelvin */

		  if(Flag_phase == 2)
		    if(temp < All.Tcrit_Phase
		       && SphP[j].d.Density * a3inv > All.PhysDensThresh * All.DensFrac_Phase)
		      {
			printf
			  (" ERROR  Phase=%d Part=%d  temp=%g K tempcrit=%g rho=%g internal rhocrit=%g ne=%g\n",
			   Flag_phase, P[j].ID, temp, All.Tcrit_Phase, SphP[j].d.Density * a3inv,
			   All.PhysDensThresh * All.DensFrac_Phase, ne);
			fflush(stdout);
			endrun(88912);	/* can't occur */
		      }

		  if(Flag_phase == 1)
		    if(!(temp < All.Tcrit_Phase
			 && SphP[j].d.Density * a3inv > All.PhysDensThresh * All.DensFrac_Phase))
		      {
			printf(" ERROR  Phase=%d  temp=%g K tempcrit=%g rho=%g internal rhocrit=%g \n",
			       Flag_phase, temp, All.Tcrit_Phase,
			       SphP[j].d.Density * a3inv, All.PhysDensThresh * All.DensFrac_Phase);
			fflush(stdout);
			endrun(888);	/* can't occur */
		      }
#endif


		  for(ik = 0; ik < 12; ik++)
		    {
		      MetalDataTmp[j].Zm[ik] += (frac_phase * reservoir[ik] * wk * inv_wk_i);
		      distributed_metals += frac_phase * reservoir[ik] * wk * inv_wk_i;
		    }

#ifdef CS_FEEDBACK
		  if(Flag_phase == 2)
		    {
		      energy = frac_phase * energySN * wk * inv_wk_i;	/* total energy */

		      distributed_energy += energy;

		      MetalDataTmp[j].Energy += energy;
		    }

		  if(Flag_phase == 1)
		    {
		      MetalDataTmp[j].EnergySN += frac_phase * energySN * wk * inv_wk_i;	/* accumulate energy */

		      MetalDataTmp[j].EnergySN += energySNCold * wk * inv_wk_i;	/* adds up reservoir from converted stars */

		      distributed_energy += frac_phase * energySN * wk * inv_wk_i;
		      distributed_energy += energySNCold * wk * inv_wk_i;
		    }
#endif
		}
	    }
	}

      if(mode == 1)
	{
	  listindex++;
	  if(listindex < NODELISTLENGTH)
	    {
	      startnode = MetalDataGet[target].NodeList[listindex];
	      if(startnode >= 0)
		startnode = Nodes[startnode].u.d.nextnode;	/* open it */
	    }
	}
    }

  return 0;
}


/* This function marks stars that have to explode as SNII or SNIa. */

void cs_flag_SN_starparticles(void)
{
  int n, numSNI, ntotI;
  int numSNII, ntotII;
  double time_hubble_a = 0, hubble_a;
  float age;
  float deltaSNI;
  double tlife_SNII = 0, metal;
  int ik;


  if(All.ComovingIntegrationOn)
    {
      hubble_a = All.Hubble * sqrt(All.Omega0 / (All.Time * All.Time * All.Time)
				   + (1 - All.Omega0 - All.OmegaLambda) / (All.Time * All.Time) +
				   All.OmegaLambda);
      time_hubble_a = All.Time * hubble_a;
    }
  else
    time_hubble_a = 1;

  for(n = FirstActiveParticle, numSNI = 0, numSNII = 0; n >= 0; n = NextActiveParticle[n])
    if(P[n].Type == 4)
      {
	if(All.ComovingIntegrationOn)
	  age = cs_integrated_time(n, time_hubble_a);
	else
	  age = All.Time - P[n].StellarAge;

#ifdef CS_SNII
	tlife_SNII = All.TlifeSNII;

	if(PPP[n].n.NumNgb != 0)
	   /*SNII*/
	  {
	    metal = 0;
	    for(ik = 1; ik < 12; ik++)	/* all chemical elements but H & He */
	      if(ik != 6)
		metal += P[n].Zm[ik];

	    metal /= P[n].Mass;

	    /* Raiteri estimations of mean tlife for SNII according to metallicity */
	    if(All.Raiteri_TlifeSNII == 1)
	      {
		if(metal <= 1.e-5)
		  tlife_SNII = Raiteri_COEFF_1;
		else if(metal <= 1.e-4)
		  tlife_SNII = Raiteri_COEFF_2;
		else if(metal <= 1.e-3)
		  tlife_SNII = Raiteri_COEFF_3;
		else if(metal <= 1.e-2)
		  tlife_SNII = Raiteri_COEFF_4;

		if(metal > 1.e-2)
		  tlife_SNII = Raiteri_COEFF_5;
	      }


	    if(age >= tlife_SNII)
	       /*SNII*/
	      {
		numSNII++;
		P[n].Type |= 3;	/* we mark SNII */

		if(age >= All.MinTlifeSNI)
		  {
		    printf("Tlife SNII > Tlife SNIa min\n");
		    endrun(32890);
		  }
	      }
	  }
#endif
#ifdef CS_SNI
	deltaSNI = All.MinTlifeSNI + (All.MaxTlifeSNI - All.MinTlifeSNI) * get_random_number(P[n].ID);

	if(age >= deltaSNI && PPP[n].Hsml != 0)
	   /*SNI*/
	  {
	    numSNI++;
	    P[n].Type |= 2;	/* we mark SNI */
	  }
#endif

      }

#ifdef CS_SNI
  MPI_Allreduce(&numSNI, &ntotI, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  if(ThisTask == 0 && ntotI > 0)
    printf("flagging ntot=%d stars as exploding type SNIa\n", ntotI);
#endif
#ifdef CS_SNII
  MPI_Allreduce(&numSNII, &ntotII, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  if(ThisTask == 0 && ntotII > 0)
    printf("flagging ntot=%d stars as exploding type SNII\n", ntotII);
#endif

}


/* Routine of integration for calculating time interval from expansion factor interval.
This is relevant only if ComovingIntegration is ON. */
double cs_integrated_time(int indice, double time_hubble_a)
{
  double t1, t2, t3;
  double f1, f2, f3;
  double deltat;

  t1 = P[indice].StellarAge;
  t3 = All.Time;
  t2 = t1 + (t3 - t1) / 2;

  f1 = 1 / (t1 * All.Hubble * sqrt(All.Omega0 / (t1 * t1 * t1)
				   + (1 - All.Omega0 - All.OmegaLambda) / (t1 * t1) + All.OmegaLambda));
  f2 = 1 / (t2 * All.Hubble * sqrt(All.Omega0 / (t2 * t2 * t2)
				   + (1 - All.Omega0 - All.OmegaLambda) / (t2 * t2) + All.OmegaLambda));
  f3 = 1 / time_hubble_a;

  deltat = (t3 - t1) / 2. * (f1 / 3. + 4. / 3. * f2 + f3 / 3.);

  return deltat;
}



#ifdef CS_FEEDBACK
void cs_promotion(void)
{
  int i;
  double a3inv, ne, mu, temp;
  double sum_reservoir_add = 0, total_reservoir_add;
  int n_promoted = 0, total_promoted;
  double sum_reservoir_promotion = 0, total_reservoir_promotion;
  double sum_reservoir_non_promotion = 0, total_reservoir_non_promotion;
  double sum_reservoir_hot = 0, total_reservoir_hot;
  double sum_reservoir_rest = 0, total_reservoir_rest;
  double new_egy, promotion_energy, hot_egy, cold_egy;
  double xhyd, yhel, energy, u_i;
  double entropyold, entropynew, critical_entropy;
  double dt, dtime, time_hubble_a, Tau_prom, prob, hubble_a;
  int n_selected = 0, total_selected;


  if(All.ComovingIntegrationOn)
    {
      a3inv = 1 / (All.Time * All.Time * All.Time);
      hubble_a = All.Hubble * sqrt(All.Omega0 / (All.Time * All.Time * All.Time)
				   + (1 - All.Omega0 - All.OmegaLambda) / (All.Time * All.Time) +
				   All.OmegaLambda);
      time_hubble_a = All.Time * hubble_a;
    }
  else
    {
      a3inv = 1;
      time_hubble_a = 1;
    }

#ifdef CS_TESTS
  InternalEnergy = 0;
  Energy_cooling = 0;
  Energy_promotion = 0;
  Energy_feedback = 0;


/*Note: not only for active particles! */
  for(i = 0; i < NumPart; i++)
    if(P[i].Type == 0)
      {
	u_i = SphP[i].Entropy / GAMMA_MINUS1 * pow(SphP[i].d.Density * a3inv, GAMMA_MINUS1);
	InternalEnergy += u_i * P[i].Mass;
      }
#endif

  for(i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i])
    if(P[i].Type == 0)
      if(P[i].EnergySN > 0)
	{
	  xhyd = P[i].Zm[6] / P[i].Mass;
	  yhel = (1 - xhyd) / (4. * xhyd);

	  ne = SphP[i].Ne;
	  mu = (1 + 4 * yhel) / (1 + yhel + ne);
	  energy = SphP[i].Entropy * P[i].Mass / GAMMA_MINUS1 * pow(SphP[i].d.Density * a3inv, GAMMA_MINUS1);	/* Total Energys */
	  temp = GAMMA_MINUS1 / BOLTZMANN * energy / P[i].Mass * PROTONMASS * mu;
	  temp *= All.UnitEnergy_in_cgs / All.UnitMass_in_g;	/* Temperature in Kelvin */


	  sum_reservoir_add += P[i].EnergySN;

	  /* consider only cold phase particles for promotion */

	  if(temp < All.Tcrit_Phase && SphP[i].d.Density * a3inv > All.PhysDensThresh * All.DensFrac_Phase)
	    {
	      cold_egy = SphP[i].Entropy / GAMMA_MINUS1 * pow(SphP[i].d.Density * a3inv, GAMMA_MINUS1);
	      hot_egy =
		SphP[i].ea.EntropyAvg / GAMMA_MINUS1 * pow(SphP[i].da.DensityAvg * a3inv, GAMMA_MINUS1);

	      promotion_energy = GAMMA * P[i].Mass * (hot_egy - cold_egy);

	      critical_entropy = All.Tcrit_Phase / pow(All.PhysDensThresh * All.DensFrac_Phase, GAMMA_MINUS1);
	      critical_entropy *= All.UnitMass_in_g / All.UnitEnergy_in_cgs * BOLTZMANN / PROTONMASS / mu;


	      new_egy = cold_egy + P[i].EnergySN / P[i].Mass;
	      entropynew = new_egy * GAMMA_MINUS1 / pow(SphP[i].da.DensityAvg * a3inv, GAMMA_MINUS1);


	      if(P[i].EnergySN > promotion_energy)
		{
		  if(!
		     (P[i].EnergySN > promotion_energy && SphP[i].HotNgbNum > 20.
		      && SphP[i].ea.EntropyAvg > critical_entropy && entropynew > SphP[i].ea.EntropyAvg))
		    printf("Non-promoted %d %g %g %d %g %g %g \n", P[i].ID, P[i].EnergySN, promotion_energy,
			   SphP[i].HotNgbNum, SphP[i].ea.EntropyAvg, critical_entropy, entropynew);
		}


	      if(promotion_energy > 0 && P[i].EnergySN > promotion_energy
		 && SphP[i].HotNgbNum > (All.DesNumNgb - All.MaxNumHotNgbDeviation)
		 && entropynew > SphP[i].ea.EntropyAvg)
		{

		  n_selected++;

		  dt = (P[i].TimeBin ? (1 << P[i].TimeBin) : 0) * All.Timebase_interval;

		  if(All.ComovingIntegrationOn)
		    dtime = All.Time * dt / time_hubble_a;
		  else
		    dtime = dt;

		  Tau_prom = promotion_energy / (P[i].EnergySN / dtime);

		  prob = (P[i].EnergySN - promotion_energy) / P[i].EnergySN;

		  printf("id=%d T=%g  dt=%g p=%g Ep=%g %g %g %g %g \n", P[i].ID, Tau_prom, dtime, prob,
			 promotion_energy, hot_egy, cold_egy, SphP[i].da.DensityAvg, SphP[i].ea.EntropyAvg);
		  fflush(0);


		  /*if(P[i].EnergySN > promotion_energy && SphP[i].HotNgbNum > 5 && entropynew > SphP[i].EntropyAvg) *//* ok, we can promote the particle */
		  if(get_random_number(P[i].ID) < prob)	/* ok, promote particle */
		    {

		      sum_reservoir_promotion += P[i].EnergySN;

		      n_promoted++;

		      entropyold = SphP[i].Entropy;

#ifdef CS_TESTS
		      Energy_promotion += hot_egy - new_egy;	/* should be < 0 */
#endif

		      SphP[i].Entropy = SphP[i].ea.EntropyAvg;

		      PPP[i].Hsml *= pow(SphP[i].d.Density / SphP[i].da.DensityAvg, 1.0 / 3);
		      P[i].EnergySN = -1;


		      SphP[i].DensPromotion = SphP[i].d.Density;
		      SphP[i].TempPromotion = temp;

		      printf
			("Promoted Part=%d DensityOld=%g DensityAvg=%g EntropyOld=%g EntropyNew=%g NHotNgbs=%d\n",
			 P[i].ID, SphP[i].DensityOld, SphP[i].da.DensityAvg, entropyold, SphP[i].Entropy,
			 SphP[i].HotNgbNum);
		      fflush(stdout);
		    }
		  else
		    sum_reservoir_non_promotion += P[i].EnergySN;
		}
	      else
		sum_reservoir_non_promotion += P[i].EnergySN;
	    }
	  else			/* hot particle */
	    {
	      sum_reservoir_hot += P[i].EnergySN;
	      /* thermalize energy in the reservoir */
	      SphP[i].Entropy +=
		P[i].EnergySN / P[i].Mass * GAMMA_MINUS1 / pow(SphP[i].d.Density * a3inv, GAMMA_MINUS1);
	      P[i].EnergySN = 0;
	    }
	}

  MPI_Reduce(&n_selected, &total_selected, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&n_promoted, &total_promoted, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

  if(ThisTask == 0 && total_promoted > 0)
    {
      //      printf("---> Promoted %d particles \n", total_promoted);
      printf("---> Promoted %d particles from %d selected\n", total_promoted, total_selected);
      fflush(stdout);
    }


  MPI_Reduce(&sum_reservoir_add, &total_reservoir_add, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  for(i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i])
    if(P[i].Type == 0)
      sum_reservoir_rest += P[i].EnergySN;

  MPI_Reduce(&sum_reservoir_rest, &total_reservoir_rest, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&sum_reservoir_promotion, &total_reservoir_promotion, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&sum_reservoir_non_promotion, &total_reservoir_non_promotion, 1, MPI_DOUBLE, MPI_SUM, 0,
	     MPI_COMM_WORLD);
  MPI_Reduce(&sum_reservoir_hot, &total_reservoir_hot, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

#ifdef CS_TESTS
  if(ThisTask == 0 && total_reservoir_add > 0)
    {
      fprintf(FdPromTest, "%g %d %g %g %g %g %g\n", All.Time,
	      total_promoted, total_reservoir_add, total_reservoir_rest, total_reservoir_promotion,
	      total_reservoir_non_promotion, total_reservoir_hot);
      fflush(FdPromTest);
    }
#endif

  if(ThisTask == 0)
    {
      if((total_reservoir_rest - total_reservoir_non_promotion - total_promoted) > 1.e-8)	/* compare the rest because of rounding */
	{
	  printf("ERROR-A %g %g %d\n", total_reservoir_rest, total_reservoir_non_promotion, total_promoted);
	  endrun(3756);
	}
      if((total_reservoir_rest - total_reservoir_promotion - total_reservoir_non_promotion -
	  total_reservoir_hot) > 1.e-8)
	{
	  printf("ERROR-B %g %g %g %g\n", total_reservoir_rest, total_reservoir_promotion,
		 total_reservoir_non_promotion, total_reservoir_hot);
	  endrun(3757787);
	}
    }
}
#endif



/* This function computes the SNIa yields from Iwamoto et al. 1999 in units of solar mass */
double cs_SNI_yields(int indice)
{
  double nSNI = 0;
  double delta_metals = 0;
  int ik = 0;
  double check = 0.;

  for(ik = 0; ik < 12; ik++)
    check += P[indice].ZmReservoir[ik];
  if(check > 0)
#ifdef CS_FEEDBACK
    if(Flag_phase == 1)
#endif
      {
	printf("ERROR Part=%d enters SNI_yields  with Reservoir=%g\n", P[indice].ID, check);
	endrun(3534);
      }

  nSNI = P[indice].Mass * All.RateSNI;

  P[indice].ZmReservoir[0] = 0;	/*  He   */
  P[indice].ZmReservoir[1] = 4.83e-2 * nSNI;	/*  14C  */
  P[indice].ZmReservoir[2] = 8.50e-3 * nSNI;	/*  24Mg */
  P[indice].ZmReservoir[3] = 1.43e-1 * nSNI;	/*  16O  */
  P[indice].ZmReservoir[4] = 6.25e-1 * nSNI;	/*  56Fe */
  P[indice].ZmReservoir[5] = 1.54e-1 * nSNI;	/*  28Si */
  P[indice].ZmReservoir[6] = 0;	/*  H    */
  P[indice].ZmReservoir[7] = 1.16e-6 * nSNI;	/*  14N  */
  P[indice].ZmReservoir[8] = 2.02e-3 * nSNI;	/*  20Ne */
  P[indice].ZmReservoir[9] = 8.46e-2 * nSNI;	/*  32S  */
  P[indice].ZmReservoir[10] = 1.19e-2 * nSNI;	/*  40Ca */
  P[indice].ZmReservoir[11] = 0;	/*  62Zn */

  for(ik = 1; ik < 11; ik++)
    if(ik != 6)
      delta_metals += P[indice].ZmReservoir[ik];


  for(ik = 0; ik < 12; ik++)
    P[indice].Zm[ik] *= (1 - delta_metals / P[indice].Mass);

  P[indice].Mass -= delta_metals;


#ifdef CS_FEEDBACK
  P[indice].EnergySN += nSNI * SN_Energy / SOLAR_MASS * All.UnitMass_in_g / All.HubbleParam;	/* conversion to internal units */
#endif

  return delta_metals;
}
#endif /* end of CS_ENRICH */


#ifdef CS_TESTS
void cs_energy_test(void)
{
  double TotInternalEnergy = 0;
  double TotEnergy_cooling = 0;
  double TotEnergy_promotion = 0;
  double TotEnergy_feedback = 0;
  double Energy_reservoir = 0, TotEnergy_reservoir = 0;
  double a3inv;
  int i;


  if(All.ComovingIntegrationOn)	/* Factors for comoving integration of hydro */
    a3inv = 1 / (All.Time * All.Time * All.Time);
  else
    a3inv = 1;


#ifdef CS_TESTS
  MPI_Reduce(&InternalEnergy, &TotInternalEnergy, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&Energy_cooling, &TotEnergy_cooling, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);


#ifdef CS_FEEDBACK
  /*Energy Test - Not only for active particles! */
  for(i = 0; i < NumPart; i++)
    if(P[i].Type == 0)
      Energy_reservoir += P[i].EnergySN;

  MPI_Reduce(&Energy_promotion, &TotEnergy_promotion, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&Energy_feedback, &TotEnergy_feedback, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&Energy_reservoir, &TotEnergy_reservoir, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

#endif

  if(ThisTask == 0)
    fprintf(FdEgyTest, "%g %g %g %g %g %g \n", All.Time, TotInternalEnergy, TotEnergy_cooling,
	    TotEnergy_promotion, TotEnergy_feedback, TotEnergy_reservoir);
  fflush(FdEgyTest);

}
#endif
#endif




void cs_copy_densities(void)
{
  int i;

  for(i = 0; i < N_gas; i++)
    if(P[i].Type == 0)
      SphP[i].DensityOld = SphP[i].d.Density;

  /*  if(ThisTask == 0)
     {printf("...end copy densities...");
     fflush(0);
     } */
}



void cs_find_low_density_tail(void)
{
  float *rho, *rho_common;
  int i, count, contrib, listlen, *contrib_list, *contrib_offset;

#define NUM_DENSITY_TAIL (2*All.DesNumNgb)

  rho = (float *) mymalloc("rho", N_gas * sizeof(float));
  contrib_list = (int *) mymalloc("contrib_list", NTask * sizeof(int));
  contrib_offset = (int *) mymalloc("contrib_offset", NTask * sizeof(int));

  for(i = 0, count = 0; i < N_gas; i++)
    if(P[i].Type == 0)
      rho[count++] = SphP[i].DensityOld;	/*note: DensityOld = Density here */

  qsort(rho, count, sizeof(float), cs_compare_density_values);

  contrib = NUM_DENSITY_TAIL;

  if(count < contrib)
    contrib = count;

  MPI_Allreduce(&contrib, &listlen, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allgather(&contrib, 1, MPI_INT, contrib_list, 1, MPI_INT, MPI_COMM_WORLD);
  for(i = 1, contrib_offset[0] = 0; i < NTask; i++)
    contrib_offset[i] = contrib_offset[i - 1] + contrib_list[i - 1];

  if(listlen < NUM_DENSITY_TAIL)
    {
      printf("total number of gas particles is less than %d!\n", NUM_DENSITY_TAIL);
      endrun(13123127);
    }

  rho_common = (float *) mymalloc("rho_common", listlen * sizeof(float));

  MPI_Allgatherv(rho, contrib, MPI_FLOAT, rho_common, contrib_list, contrib_offset, MPI_FLOAT,
		 MPI_COMM_WORLD);

  qsort(rho_common, listlen, sizeof(float), cs_compare_density_values);

  All.DensityTailThreshold = rho_common[NUM_DENSITY_TAIL - 1];

  myfree(rho_common);
  myfree(contrib_offset);
  myfree(contrib_list);
  myfree(rho);
}


int cs_compare_density_values(const void *a, const void *b)
{
  if(*((float *) a) < *((float *) b))
    return -1;

  if(*((float *) a) > *((float *) b))
    return +1;

  return 0;
}




#endif
