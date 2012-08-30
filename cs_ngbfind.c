#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "allvars.h"
#include "proto.h"

#ifdef NUM_THREADS
#include <pthread.h>
#endif


#ifdef CS_MODEL



#ifdef PERIODIC
#define SIGNED_NGB_PERIODIC_LONG_X(x) (xtmp=x,(xtmp>boxHalf_X)?(xtmp-boxSize_X):((xtmp<-boxHalf_X)?(xtmp+boxSize_X):xtmp))
#define SIGNED_NGB_PERIODIC_LONG_Y(x) (xtmp=x,(xtmp>boxHalf_Y)?(xtmp-boxSize_Y):((xtmp<-boxHalf_Y)?(xtmp+boxSize_Y):xtmp))
#define SIGNED_NGB_PERIODIC_LONG_Z(x) (xtmp=x,(xtmp>boxHalf_Z)?(xtmp-boxSize_Z):((xtmp<-boxHalf_Z)?(xtmp+boxSize_Z):xtmp))
#else
#define SIGNED_NGB_PERIODIC_LONG_X(x) (x)
#define SIGNED_NGB_PERIODIC_LONG_Y(x) (x)
#define SIGNED_NGB_PERIODIC_LONG_Z(x) (x)
#endif


#include "cs_metals.h"

extern int Nexport;
extern int BufferFullFlag;

#ifdef NUM_THREADS
extern pthread_mutex_t mutex_nexport, mutex_partnodedrift;

#define LOCK_NEXPORT         pthread_mutex_lock(&mutex_nexport);
#define UNLOCK_NEXPORT       pthread_mutex_unlock(&mutex_nexport);
#define LOCK_PARTNODEDRIFT   pthread_mutex_lock(&mutex_partnodedrift);
#define UNLOCK_PARTNODEDRIFT pthread_mutex_unlock(&mutex_partnodedrift);
#else
#define LOCK_NEXPORT
#define UNLOCK_NEXPORT
#define LOCK_PARTNODEDRIFT
#define UNLOCK_PARTNODEDRIFT
#endif


int cs_ngb_treefind_variable_phases(MyDouble searchcenter[3], MyFloat hsml, int target, int *startnode,
				    int mode, int *nexport, int *nsend_local)
{
  int numngb, no, p, task, nexport_save;
  struct NODE *current;
  double r2;
  MyDouble dx, dy, dz, dist;

#ifdef PERIODIC
  MyDouble xtmp;
#endif
  nexport_save = *nexport;

  numngb = 0;
  no = *startnode;

#ifdef CS_FEEDBACK
  double a3inv;

  if(All.ComovingIntegrationOn)
    a3inv = 1 / (All.Time * All.Time * All.Time);
  else
    a3inv = 1.;

  /*
     double ne, mu, u, temp = 0;
     double xhyd, yhel;
     int nhot, ncold;

     if(All.ComovingIntegrationOn)
     {
     fac_mu = pow(All.Time, 3 * (GAMMA - 1) / 2) / All.Time;
     hubble_a = All.Omega0 / (All.Time * All.Time * All.Time)
     + (1 - All.Omega0 - All.OmegaLambda) / (All.Time * All.Time)
     #ifdef DARKENERGY
     + DarkEnergy_a(All.Time);
     #else
     + All.OmegaLambda;
     #endif
     hubble_a = All.Hubble * sqrt(hubble_a);
     hubble_a2 = All.Time * All.Time * hubble_a;
     }
     else
     {
     fac_mu = hubble_a = hubble_a2 = 1;
     }
   */
#endif



  while(no >= 0)
    {
      if(no < All.MaxPart)	/* single particle */
	{
	  p = no;
	  no = Nextnode[no];

	  if(P[p].Type > 0)
	    continue;

	  if(P[p].Ti_current != All.Ti_Current)
	    drift_particle(p, All.Ti_Current);

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
	  if((r2 = dx * dx + dy * dy + dz * dz) > dist * dist)
	    continue;

#ifdef CS_FEEDBACK
	  double xhyd, yhel, ne, mu, u, temp;

	  if(SphP[p].d.Density > 0)
	    {
	      xhyd = P[p].Zm[6] / P[p].Mass;
	      yhel = (1 - xhyd) / (4. * xhyd);

	      ne = SphP[p].Ne;
	      mu = (1 + 4 * yhel) / (1 + yhel + ne);
	      u = SphP[p].Entropy / GAMMA_MINUS1 * pow(SphP[p].d.Density * a3inv, GAMMA_MINUS1);	/* energy per mass unit */
	      temp = GAMMA_MINUS1 / BOLTZMANN * u * PROTONMASS * mu;
	      temp *= All.UnitEnergy_in_cgs / All.UnitMass_in_g;	/* Temperature in Kelvin */
	    }

	  if(Flag_phase == 2)	/* Hot Phase */
	    {
	      if((temp < All.Tcrit_Phase
		  && SphP[p].d.Density * a3inv > All.PhysDensThresh * All.DensFrac_Phase)
		 && SphP[p].DensityOld > All.DensityTailThreshold)
		{
		  continue;
		}
	    }

	  if(Flag_phase == 1)	/* Cold Phase */
	    {
	      if(!
		 (temp < All.Tcrit_Phase
		  && SphP[p].d.Density * a3inv > All.PhysDensThresh * All.DensFrac_Phase))
		{
		  continue;
		}
	    }
#endif
	  R2ngblist[numngb] = r2;
	  Ngblist[numngb++] = p;
	}
      else
	{
	  if(no >= All.MaxPart + MaxNodes)	/* pseudo particle */
	    {
	      if(mode == 1)
		endrun(12312);

	      if(target >= 0)	/* if no target is given, export will not occur */
		{
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
			  if(nexport_save == 0)
			    endrun(13004);	/* in this case, the buffer is too small to process even a single particle */
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
		}

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

	  if(current->Ti_current != All.Ti_Current)
	    force_drift_node(no, All.Ti_Current);

	  if(!(current->u.d.bitflags & (1 << BITFLAG_MULTIPLEPARTICLES)))
	    {
	      if(current->u.d.mass)	/* open cell */
		{
		  no = current->u.d.nextnode;
		  continue;
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



int cs_ngb_treefind_variable_decoupling(MyDouble searchcenter[3], MyFloat hsml, int target,
					int *startnode, MyFloat densityold,
					MyFloat entropy, MyFloat * vel,
					int mode, int *nexport, int *nsend_local)
{
  int numngb, no, p, task, nexport_save;
  struct NODE *current;
  MyDouble dx, dy, dz, dist;

#ifdef PERIODIC
  MyDouble xtmp;
#endif

  nexport_save = *nexport;

  double fac_mu, hubble_a, hubble_a2;
  double r, r2, dvx, dvy, dvz, vdotr, vdotr2, soundspeed_i, soundspeed_j, c_ij, mu_ij;

#ifdef CS_FEEDBACK
  double a3inv = 1;
  double ne, mu, u, temp = 0;
  double xhyd, yhel;
  int ncold, nhot;


  if(All.ComovingIntegrationOn)
    a3inv = 1 / (All.Time * All.Time * All.Time);
  else
    a3inv = 1.;
#endif


  if(All.ComovingIntegrationOn)
    {
      fac_mu = pow(All.Time, 3 * (GAMMA - 1) / 2) / All.Time;
      hubble_a = All.Omega0 / (All.Time * All.Time * All.Time)
	+ (1 - All.Omega0 - All.OmegaLambda) / (All.Time * All.Time)
#ifdef DARKENERGY
	+ DarkEnergy_a(All.Time);
#else
	+ All.OmegaLambda;
#endif
      hubble_a = All.Hubble * sqrt(hubble_a);
      hubble_a2 = All.Time * All.Time * hubble_a;
    }
  else
    {
      fac_mu = hubble_a = hubble_a2 = 1;
    }



  numngb = 0;
  no = *startnode;

  while(no >= 0)
    {
      if(no < All.MaxPart)	/* single particle */
	{
	  p = no;
	  no = Nextnode[no];

	  if(P[p].Type > 0)
	    continue;

	  if(P[p].Ti_current != All.Ti_Current)
	    drift_particle(p, All.Ti_Current);

	  dist = hsml;
	  dx = SIGNED_NGB_PERIODIC_LONG_X(searchcenter[0] - P[p].Pos[0]);
	  if(fabs(dx) > dist)
	    continue;
	  dy = SIGNED_NGB_PERIODIC_LONG_Y(searchcenter[1] - P[p].Pos[1]);
	  if(fabs(dy) > dist)
	    continue;
	  dz = SIGNED_NGB_PERIODIC_LONG_Z(searchcenter[2] - P[p].Pos[2]);
	  if(fabs(dz) > dist)
	    continue;
	  if((r2 = dx * dx + dy * dy + dz * dz) > dist * dist)
	    continue;



#ifdef CS_FEEDBACK
	  if(SphP[p].d.Density > 0)
	    {
	      xhyd = P[p].Zm[6] / P[p].Mass;
	      yhel = (1 - xhyd) / (4. * xhyd);

	      ne = SphP[p].Ne;
	      mu = (1 + 4 * yhel) / (1 + yhel + ne);
	      u = SphP[p].Entropy / GAMMA_MINUS1 * pow(SphP[p].d.Density * a3inv, GAMMA_MINUS1);	/* energy per mass unit */
	      temp = GAMMA_MINUS1 / BOLTZMANN * u * PROTONMASS * mu;
	      temp *= All.UnitEnergy_in_cgs / All.UnitMass_in_g;	/* Temperature in Kelvin */
	    }

	  if(Flag_phase == 2)	/* Hot Phase */
	    {
	      if((temp < All.Tcrit_Phase
		  && SphP[p].d.Density * a3inv > All.PhysDensThresh * All.DensFrac_Phase)
		 && SphP[p].DensityOld > All.DensityTailThreshold)
		{
		  ncold++;
		  continue;
		}
	      nhot++;
	    }

	  if(Flag_phase == 1)	/* Cold Phase */
	    {
	      if(!
		 (temp < All.Tcrit_Phase
		  && SphP[p].d.Density * a3inv > All.PhysDensThresh * All.DensFrac_Phase))
		{
		  nhot++;
		  continue;
		}
	      ncold++;
	    }
#endif

	  if(densityold > 0)	/* note: if zero is passed, we have a star (or the first iteration upon start-up) */
	    {
	      dvx = vel[0] - SphP[p].VelPred[0];
	      dvy = vel[1] - SphP[p].VelPred[1];
	      dvz = vel[2] - SphP[p].VelPred[2];
	      vdotr = dx * dvx + dy * dvy + dz * dvz;
	      r = sqrt(r2);

	      if(All.ComovingIntegrationOn)
		vdotr2 = vdotr + hubble_a2 * r2;
	      else
		vdotr2 = vdotr;

	      soundspeed_i = sqrt(GAMMA * entropy * pow(densityold, GAMMA_MINUS1));
	      soundspeed_j = sqrt(GAMMA * SphP[p].Entropy * pow(SphP[p].DensityOld, GAMMA_MINUS1));
	      c_ij = 0.5 * (soundspeed_i + soundspeed_j);

	      if(vdotr2 > 0)
		mu_ij = 0;
	      else
		{
#ifndef CONVENTIONAL_VISCOSITY
		  if(r > 0)
		    mu_ij = fac_mu * (-vdotr2) / r;	/* note: this is positive! */
		  else
		    mu_ij = 0;
#else
		  h_ij = 0.5 * (h_i + h_j);
		  mu_ij = fac_mu * h_ij * (-vdotr2) / (r2 + 0.0001 * h_ij * h_ij);
#endif
		}

#ifndef CS_FEEDBACK
	      if(((entropy > All.DecouplingParam * SphP[p].Entropy && mu_ij < c_ij)
		  && SphP[p].DensityOld > All.DensityTailThreshold))
		continue;
#else
	      if(((entropy > All.DecouplingParam * SphP[p].Entropy && mu_ij < c_ij)
		  && SphP[p].DensityOld > All.DensityTailThreshold) || SphP[p].DensPromotion > 0)
		continue;
#endif
	    }

	  Ngblist[numngb++] = p;
	}
      else
	{
	  if(no >= All.MaxPart + MaxNodes)	/* pseudo particle */
	    {
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
		      if(nexport_save == 0)
			endrun(13004);	/* in this case, the buffer is too small to process even a single particle */
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

	  if(current->Ti_current != All.Ti_Current)
	    force_drift_node(no, All.Ti_Current);

	  if(!(current->u.d.bitflags & (1 << BITFLAG_MULTIPLEPARTICLES)))
	    {
	      if(current->u.d.mass)	/* open cell */
		{
		  no = current->u.d.nextnode;
		  continue;
		}
	    }

	  no = current->u.d.sibling;	/* in case the node can be discarded */

	  dist = hsml + 0.5 * current->len;
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



int cs_ngb_treefind_variable_decoupling_threads(MyDouble searchcenter[3], MyFloat hsml, int target,
						int *startnode, MyFloat densityold,
						MyFloat entropy, MyFloat * vel,
						int mode, int *exportflag, int *exportnodecount,
						int *exportindex, int *ngblist)
{
  int numngb, no, nexp, p, task;
  struct NODE *current;
  MyDouble dx, dy, dz, dist;

#ifdef PERIODIC
  MyDouble xtmp;
#endif

  double fac_mu, hubble_a, hubble_a2;
  double r, r2, dvx, dvy, dvz, vdotr, vdotr2, soundspeed_i, soundspeed_j, c_ij, mu_ij;

#ifdef CS_FEEDBACK
  double a3inv = 1;
  double ne, mu, u, temp = 0;
  double xhyd, yhel;
  int ncold, nhot;

  LOCK_PARTNODEDRIFT;
  if(All.ComovingIntegrationOn)
    a3inv = 1 / (All.Time * All.Time * All.Time);
  else
    a3inv = 1.;
  UNLOCK_PARTNODEDRIFT;
#endif

  LOCK_PARTNODEDRIFT;
  if(All.ComovingIntegrationOn)
    {
      fac_mu = pow(All.Time, 3 * (GAMMA - 1) / 2) / All.Time;
      hubble_a = All.Omega0 / (All.Time * All.Time * All.Time)
	+ (1 - All.Omega0 - All.OmegaLambda) / (All.Time * All.Time)
#ifdef DARKENERGY
	+ DarkEnergy_a(All.Time);
#else
	+ All.OmegaLambda;
#endif
      hubble_a = All.Hubble * sqrt(hubble_a);
      hubble_a2 = All.Time * All.Time * hubble_a;
    }
  else
    {
      fac_mu = hubble_a = hubble_a2 = 1;
    }
  UNLOCK_PARTNODEDRIFT;


  numngb = 0;
  no = *startnode;

  while(no >= 0)
    {
      if(no < All.MaxPart)	/* single particle */
	{
	  p = no;
	  no = Nextnode[no];

	  if(P[p].Type > 0)
	    continue;

	  if(P[p].Ti_current != All.Ti_Current)
	    {
	      LOCK_PARTNODEDRIFT;
	      drift_particle(p, All.Ti_Current);
	      UNLOCK_PARTNODEDRIFT;
	    }

	  dist = hsml;
	  dx = SIGNED_NGB_PERIODIC_LONG_X(searchcenter[0] - P[p].Pos[0]);
	  if(fabs(dx) > dist)
	    continue;
	  dy = SIGNED_NGB_PERIODIC_LONG_Y(searchcenter[1] - P[p].Pos[1]);
	  if(fabs(dy) > dist)
	    continue;
	  dz = SIGNED_NGB_PERIODIC_LONG_Z(searchcenter[2] - P[p].Pos[2]);
	  if(fabs(dz) > dist)
	    continue;
	  if((r2 = dx * dx + dy * dy + dz * dz) > dist * dist)
	    continue;



#ifdef CS_FEEDBACK
	  if(SphP[p].d.Density > 0)
	    {
	      xhyd = P[p].Zm[6] / P[p].Mass;
	      yhel = (1 - xhyd) / (4. * xhyd);

	      ne = SphP[p].Ne;
	      mu = (1 + 4 * yhel) / (1 + yhel + ne);
	      u = SphP[p].Entropy / GAMMA_MINUS1 * pow(SphP[p].d.Density * a3inv, GAMMA_MINUS1);	/* energy per mass unit */
	      temp = GAMMA_MINUS1 / BOLTZMANN * u * PROTONMASS * mu;
	      temp *= All.UnitEnergy_in_cgs / All.UnitMass_in_g;	/* Temperature in Kelvin */
	    }

	  if(Flag_phase == 2)	/* Hot Phase */
	    {
	      if((temp < All.Tcrit_Phase
		  && SphP[p].d.Density * a3inv > All.PhysDensThresh * All.DensFrac_Phase)
		 && SphP[p].DensityOld > All.DensityTailThreshold)
		{
		  ncold++;
		  continue;
		}
	      nhot++;
	    }

	  if(Flag_phase == 1)	/* Cold Phase */
	    {
	      if(!
		 (temp < All.Tcrit_Phase
		  && SphP[p].d.Density * a3inv > All.PhysDensThresh * All.DensFrac_Phase))
		{
		  nhot++;
		  continue;
		}
	      ncold++;
	    }
#endif

	  if(densityold > 0)	/* note: if zero is passed, we have a star (or the first iteration upon start-up) */
	    {
	      dvx = vel[0] - SphP[p].VelPred[0];
	      dvy = vel[1] - SphP[p].VelPred[1];
	      dvz = vel[2] - SphP[p].VelPred[2];
	      vdotr = dx * dvx + dy * dvy + dz * dvz;
	      r = sqrt(r2);

	      if(All.ComovingIntegrationOn)
		vdotr2 = vdotr + hubble_a2 * r2;
	      else
		vdotr2 = vdotr;

	      soundspeed_i = sqrt(GAMMA * entropy * pow(densityold, GAMMA_MINUS1));
	      soundspeed_j = sqrt(GAMMA * SphP[p].Entropy * pow(SphP[p].DensityOld, GAMMA_MINUS1));
	      c_ij = 0.5 * (soundspeed_i + soundspeed_j);

	      if(vdotr2 > 0)
		mu_ij = 0;
	      else
		{
#ifndef CONVENTIONAL_VISCOSITY
		  if(r > 0)
		    mu_ij = fac_mu * (-vdotr2) / r;	/* note: this is positive! */
		  else
		    mu_ij = 0;
#else
		  h_ij = 0.5 * (h_i + h_j);
		  mu_ij = fac_mu * h_ij * (-vdotr2) / (r2 + 0.0001 * h_ij * h_ij);
#endif
		}

#ifndef CS_FEEDBACK
	      if(((entropy > All.DecouplingParam * SphP[p].Entropy && mu_ij < c_ij)
		  && SphP[p].DensityOld > All.DensityTailThreshold))
		continue;
#else
	      if(((entropy > All.DecouplingParam * SphP[p].Entropy && mu_ij < c_ij)
		  && SphP[p].DensityOld > All.DensityTailThreshold) || SphP[p].DensPromotion > 0)
		continue;
#endif
	    }

	  ngblist[numngb++] = p;
	}
      else
	{
	  if(no >= All.MaxPart + MaxNodes)	/* pseudo particle */
	    {
	      if(mode == 1)
		endrun(12312);

	      if(target >= 0)	/* if no target is given, export will not occur */
		{
		  if(exportflag[task = DomainTask[no - (All.MaxPart + MaxNodes)]] != target)
		    {
		      exportflag[task] = target;
		      exportnodecount[task] = NODELISTLENGTH;
		    }

		  if(exportnodecount[task] == NODELISTLENGTH)
		    {
		      LOCK_NEXPORT;
		      if(Nexport >= All.BunchSize)
			{
			  /* out if buffer space. Need to discard work for this particle and interrupt */
			  BufferFullFlag = 1;
			  UNLOCK_NEXPORT;
			  return -1;
			}
		      nexp = Nexport;
		      Nexport++;
		      UNLOCK_NEXPORT;
		      exportnodecount[task] = 0;
		      exportindex[task] = nexp;
		      DataIndexTable[nexp].Task = task;
		      DataIndexTable[nexp].Index = target;
		      DataIndexTable[nexp].IndexGet = nexp;
		    }

		  DataNodeList[exportindex[task]].NodeList[exportnodecount[task]++] =
		    DomainNodeIndex[no - (All.MaxPart + MaxNodes)];

		  if(exportnodecount[task] < NODELISTLENGTH)
		    DataNodeList[exportindex[task]].NodeList[exportnodecount[task]] = -1;

		}

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

	  if(current->Ti_current != All.Ti_Current)
	    {
	      LOCK_PARTNODEDRIFT;
	      force_drift_node(no, All.Ti_Current);
	      UNLOCK_PARTNODEDRIFT;
	    }

	  if(!(current->u.d.bitflags & (1 << BITFLAG_MULTIPLEPARTICLES)))
	    {
	      if(current->u.d.mass)	/* open cell */
		{
		  no = current->u.d.nextnode;
		  continue;
		}
	    }

	  no = current->u.d.sibling;	/* in case the node can be discarded */

	  dist = hsml + 0.5 * current->len;
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


int cs_ngb_treefind_pairs(MyDouble searchcenter[3], MyFloat hsml, int target, int *startnode,
			  MyFloat density, MyFloat entropy, MyFloat * vel,
			  int mode, int *nexport, int *nsend_local)
{
  int no, p, numngb, task, nexport_save;
  MyDouble dist, dx, dy, dz, dvx, dvy, dvz, dmax1, dmax2;
  double r2, vdotr, vdotr2, r, soundspeed_i, soundspeed_j, c_ij, mu_ij, fac_mu;
  double hubble_a, hubble_a2;
  struct NODE *current;

#ifdef PERIODIC
  MyDouble xtmp;
#endif
  nexport_save = *nexport;


#if defined(ALTVISCOSITY) && defined(CONVENTIONAL_VISCOSITY)
  double h_ij;
#endif

  if(All.ComovingIntegrationOn)
    {
      fac_mu = pow(All.Time, 3 * (GAMMA - 1) / 2) / All.Time;
      hubble_a = All.Omega0 / (All.Time * All.Time * All.Time)
	+ (1 - All.Omega0 - All.OmegaLambda) / (All.Time * All.Time)
#ifdef DARKENERGY
	+ DarkEnergy_a(All.Time);
#else
	+ All.OmegaLambda;
#endif
      hubble_a = All.Hubble * sqrt(hubble_a);
      hubble_a2 = All.Time * All.Time * hubble_a;
    }
  else
    {
      fac_mu = hubble_a = hubble_a2 = 1;
    }








  numngb = 0;
  no = *startnode;

  while(no >= 0)
    {
      if(no < All.MaxPart)	/* single particle */
	{
	  p = no;
	  no = Nextnode[no];

	  if(P[p].Type > 0)
	    continue;

	  if(P[p].Ti_current != All.Ti_Current)
	    drift_particle(p, All.Ti_Current);

	  dist = DMAX(PPP[p].Hsml, hsml);

	  dx = SIGNED_NGB_PERIODIC_LONG_X(searchcenter[0] - P[p].Pos[0]);
	  if(fabs(dx) > dist)
	    continue;
	  dy = SIGNED_NGB_PERIODIC_LONG_Y(searchcenter[1] - P[p].Pos[1]);
	  if(fabs(dy) > dist)
	    continue;
	  dz = SIGNED_NGB_PERIODIC_LONG_Z(searchcenter[2] - P[p].Pos[2]);
	  if(fabs(dz) > dist)
	    continue;
	  if((r2 = dx * dx + dy * dy + dz * dz) > dist * dist)
	    continue;


	  if(density > 0)	/* note: stars should not get in here... */
	    {
	      dvx = vel[0] - SphP[p].VelPred[0];
	      dvy = vel[1] - SphP[p].VelPred[1];
	      dvz = vel[2] - SphP[p].VelPred[2];
	      vdotr = dx * dvx + dy * dvy + dz * dvz;
	      r = sqrt(r2);

	      if(All.ComovingIntegrationOn)
		vdotr2 = vdotr + hubble_a2 * r2;
	      else
		vdotr2 = vdotr;

	      soundspeed_i = sqrt(GAMMA * entropy * pow(density, GAMMA_MINUS1));
	      soundspeed_j = sqrt(GAMMA * SphP[p].Entropy * pow(SphP[p].d.Density, GAMMA_MINUS1));
	      c_ij = 0.5 * (soundspeed_i + soundspeed_j);

	      if(vdotr2 > 0)
		mu_ij = 0;
	      else
		{
#ifndef CONVENTIONAL_VISCOSITY
		  if(r > 0)
		    mu_ij = fac_mu * (-vdotr2) / r;	/* note this is positive */
		  else
		    mu_ij = 0;
#else
		  h_ij = 0.5 * (h_i + h_j);
		  mu_ij = fac_mu * h_ij * (-vdotr2) / (r2 + 0.0001 * h_ij * h_ij);
#endif
		}

#ifndef CS_FEEDBACK
	      if(((entropy > All.DecouplingParam * SphP[p].Entropy
		   || SphP[p].Entropy > All.DecouplingParam * entropy) && mu_ij < c_ij))
		continue;
#else
	      if(((entropy > All.DecouplingParam * SphP[p].Entropy
		   || SphP[p].Entropy > All.DecouplingParam * entropy) && mu_ij < c_ij)
		 || SphP[p].DensPromotion > 0)
		continue;
#endif
	    }

	  Ngblist[numngb++] = p;	/* Note: unlike in previous versions of the code, the buffer 
					   can hold up to all particles */
	}
      else
	{
	  if(no >= All.MaxPart + MaxNodes)	/* pseudo particle */
	    {
#ifdef DONOTUSENODELIST
	      if(mode == 1)
		{
		  no = Nextnode[no - MaxNodes];
		  continue;
		}
#endif
	      if(mode == 1)
		endrun(23131);

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
		      if(nexport_save == 0)
			endrun(13003);	/* in this case, the buffer is too small to process even a single particle */
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

#ifndef DONOTUSENODELIST
	      DataNodeList[Exportindex[task]].NodeList[Exportnodecount[task]++] =
		DomainNodeIndex[no - (All.MaxPart + MaxNodes)];

	      if(Exportnodecount[task] < NODELISTLENGTH)
		DataNodeList[Exportindex[task]].NodeList[Exportnodecount[task]] = -1;
#endif
	      no = Nextnode[no - MaxNodes];
	      continue;
	    }

	  current = &Nodes[no];

#ifndef DONOTUSENODELIST
	  if(mode == 1)
	    {
	      if(current->u.d.bitflags & (1 << BITFLAG_TOPLEVEL))	/* we reached a top-level node again, which means that we are done with the branch */
		{
		  *startnode = -1;
		  return numngb;
		}
	    }
#endif

	  if(current->Ti_current != All.Ti_Current)
	    force_drift_node(no, All.Ti_Current);

	  if(!(current->u.d.bitflags & (1 << BITFLAG_MULTIPLEPARTICLES)))
	    {
	      if(current->u.d.mass)	/* open cell */
		{
		  no = current->u.d.nextnode;
		  continue;
		}
	    }

	  dist = DMAX(Extnodes[no].hmax, hsml) + 0.5 * current->len;

	  no = current->u.d.sibling;	/* in case the node can be discarded */

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




int cs_ngb_treefind_pairs_threads(MyDouble searchcenter[3], MyFloat hsml, int target, int *startnode,
				  MyFloat density, MyFloat entropy, MyFloat * vel,
				  int mode, int *exportflag, int *exportnodecount, int *exportindex,
				  int *ngblist)
{
  int no, p, numngb, task, nexp;
  MyDouble dist, dx, dy, dz, dvx, dvy, dvz, dmax1, dmax2;
  double r2, vdotr, vdotr2, r, soundspeed_i, soundspeed_j, c_ij, mu_ij, fac_mu;
  double hubble_a, hubble_a2;
  struct NODE *current;

#ifdef PERIODIC
  MyDouble xtmp;
#endif


#if defined(ALTVISCOSITY) && defined(CONVENTIONAL_VISCOSITY)
  double h_ij;
#endif
  LOCK_PARTNODEDRIFT;
  if(All.ComovingIntegrationOn)
    {
      fac_mu = pow(All.Time, 3 * (GAMMA - 1) / 2) / All.Time;
      hubble_a = All.Omega0 / (All.Time * All.Time * All.Time)
	+ (1 - All.Omega0 - All.OmegaLambda) / (All.Time * All.Time)
#ifdef DARKENERGY
	+ DarkEnergy_a(All.Time);
#else
	+ All.OmegaLambda;
#endif
      hubble_a = All.Hubble * sqrt(hubble_a);
      hubble_a2 = All.Time * All.Time * hubble_a;
    }
  else
    {
      fac_mu = hubble_a = hubble_a2 = 1;
    }
  UNLOCK_PARTNODEDRIFT;


  numngb = 0;
  no = *startnode;

  while(no >= 0)
    {
      if(no < All.MaxPart)	/* single particle */
	{
	  p = no;
	  no = Nextnode[no];

	  if(P[p].Type > 0)
	    continue;

	  if(P[p].Ti_current != All.Ti_Current)
	    {
	      LOCK_PARTNODEDRIFT;
	      drift_particle(p, All.Ti_Current);
	      UNLOCK_PARTNODEDRIFT;
	    }

	  dist = DMAX(PPP[p].Hsml, hsml);

	  dx = SIGNED_NGB_PERIODIC_LONG_X(searchcenter[0] - P[p].Pos[0]);
	  if(fabs(dx) > dist)
	    continue;
	  dy = SIGNED_NGB_PERIODIC_LONG_Y(searchcenter[1] - P[p].Pos[1]);
	  if(fabs(dy) > dist)
	    continue;
	  dz = SIGNED_NGB_PERIODIC_LONG_Z(searchcenter[2] - P[p].Pos[2]);
	  if(fabs(dz) > dist)
	    continue;
	  if((r2 = dx * dx + dy * dy + dz * dz) > dist * dist)
	    continue;


	  if(density > 0)	/* note: stars should not get in here... */
	    {
	      dvx = vel[0] - SphP[p].VelPred[0];
	      dvy = vel[1] - SphP[p].VelPred[1];
	      dvz = vel[2] - SphP[p].VelPred[2];
	      vdotr = dx * dvx + dy * dvy + dz * dvz;
	      r = sqrt(r2);

	      if(All.ComovingIntegrationOn)
		vdotr2 = vdotr + hubble_a2 * r2;
	      else
		vdotr2 = vdotr;

	      soundspeed_i = sqrt(GAMMA * entropy * pow(density, GAMMA_MINUS1));
	      soundspeed_j = sqrt(GAMMA * SphP[p].Entropy * pow(SphP[p].d.Density, GAMMA_MINUS1));
	      c_ij = 0.5 * (soundspeed_i + soundspeed_j);

	      if(vdotr2 > 0)
		mu_ij = 0;
	      else
		{
#ifndef CONVENTIONAL_VISCOSITY
		  if(r > 0)
		    mu_ij = fac_mu * (-vdotr2) / r;	/* note this is positive */
		  else
		    mu_ij = 0;
#else
		  h_ij = 0.5 * (h_i + h_j);
		  mu_ij = fac_mu * h_ij * (-vdotr2) / (r2 + 0.0001 * h_ij * h_ij);
#endif
		}

#ifndef CS_FEEDBACK
	      if(((entropy > All.DecouplingParam * SphP[p].Entropy
		   || SphP[p].Entropy > All.DecouplingParam * entropy) && mu_ij < c_ij))
		continue;
#else
	      if(((entropy > All.DecouplingParam * SphP[p].Entropy
		   || SphP[p].Entropy > All.DecouplingParam * entropy) && mu_ij < c_ij)
		 || SphP[p].DensPromotion > 0)
		continue;
#endif
	    }

	  ngblist[numngb++] = p;	/* Note: unlike in previous versions of the code, the buffer 
					   can hold up to all particles */
	}
      else
	{
	  if(no >= All.MaxPart + MaxNodes)	/* pseudo particle */
	    {
	      if(mode == 1)
		endrun(23131);

	      if(target >= 0)	/* if no target is given, export will not occur */
		{
		  if(exportflag[task = DomainTask[no - (All.MaxPart + MaxNodes)]] != target)
		    {
		      exportflag[task] = target;
		      exportnodecount[task] = NODELISTLENGTH;
		    }

		  if(exportnodecount[task] == NODELISTLENGTH)
		    {
		      LOCK_NEXPORT;
		      if(Nexport >= All.BunchSize)
			{
			  /* out if buffer space. Need to discard work for this particle and interrupt */
			  BufferFullFlag = 1;
			  UNLOCK_NEXPORT;
			  return -1;
			}
		      nexp = Nexport;
		      Nexport++;
		      UNLOCK_NEXPORT;
		      exportnodecount[task] = 0;
		      exportindex[task] = nexp;
		      DataIndexTable[nexp].Task = task;
		      DataIndexTable[nexp].Index = target;
		      DataIndexTable[nexp].IndexGet = nexp;
		    }

		  DataNodeList[exportindex[task]].NodeList[exportnodecount[task]++] =
		    DomainNodeIndex[no - (All.MaxPart + MaxNodes)];

		  if(exportnodecount[task] < NODELISTLENGTH)
		    DataNodeList[exportindex[task]].NodeList[exportnodecount[task]] = -1;

		}

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

	  if(current->Ti_current != All.Ti_Current)
	    {
	      LOCK_PARTNODEDRIFT;
	      force_drift_node(no, All.Ti_Current);
	      UNLOCK_PARTNODEDRIFT;
	    }

	  if(!(current->u.d.bitflags & (1 << BITFLAG_MULTIPLEPARTICLES)))
	    {
	      if(current->u.d.mass)	/* open cell */
		{
		  no = current->u.d.nextnode;
		  continue;
		}
	    }

	  dist = DMAX(Extnodes[no].hmax, hsml) + 0.5 * current->len;

	  no = current->u.d.sibling;	/* in case the node can be discarded */

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






int cs_ngb_treefind_hotngbs(MyDouble searchcenter[3], MyFloat hsml, int target, int *startnode,
			    MyFloat entropy, int mode, int *nexport, int *nsend_local)
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

	  if(P[p].Type > 0)
	    continue;

	  if(P[p].Ti_current != All.Ti_Current)
	    drift_particle(p, All.Ti_Current);

	  if(All.DecouplingParam * entropy > SphP[p].Entropy)	/* if neighbour is not ignoring us, we ignore it */
	    continue;

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

	      if(target >= 0)	/* if no target is given, export will not occur */
		{
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
			  if(nexport_save == 0)
			    endrun(13004);	/* in this case, the buffer is too small to process even a single particle */
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
		}

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

	  if(current->Ti_current != All.Ti_Current)
	    force_drift_node(no, All.Ti_Current);

	  if(!(current->u.d.bitflags & (1 << BITFLAG_MULTIPLEPARTICLES)))
	    {
	      if(current->u.d.mass)	/* open cell */
		{
		  no = current->u.d.nextnode;
		  continue;
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






#endif
