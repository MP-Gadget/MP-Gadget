#ifndef FOF_H
#define FOF_H

#include "allvars.h"



extern int Ngroups, TotNgroups;
extern int64_t TotNids;
struct group_particle_host {
    int index;
    int task;
    double Value;
};
extern struct group_properties
{
  int Len;
  ptrdiff_t Offset;
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
  int index_maxdens, task_maxdens;
#endif
#ifdef GAL_PART
  double Gal_Mass;
  double Gal_SFR;
  double AngularMomentum[3];
  struct group_particle_host DenseGas;
  struct group_particle_host CentralGalaxy;
#endif

} *Group;

#endif
