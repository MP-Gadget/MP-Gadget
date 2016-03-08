#ifndef FOF_H
#define FOF_H

#ifdef FOF

#include "allvars.h"

extern uint64_t Ngroups, TotNgroups;
extern int64_t TotNids;

extern struct group_properties
{
  int OriginalTask;
  int Length;
  MyIDType MinID;
  MyIDType MinIDTask;
  int GrNr;
  int LenType[6];
  double MassType[6];
  double Mass;
  double CM[3];
  double Vel[3];
  MyDouble FirstPos[3];
#ifdef SFR
  double Sfr;
#endif
#ifdef BLACK_HOLES
  double BH_Mass;
  double BH_Mdot;
  double MaxDens;
#endif
  int seed_index; 
  int seed_task;
} *Group;

#endif

#endif
