#ifndef FOF_H
#define FOF_H

#include "allvars.h"

int fof_compare_FOF_PList_MinID(const void *a, const void *b);
int fof_compare_FOF_GList_MinID(const void *a, const void *b);
int fof_compare_FOF_GList_MinIDTask(const void *a, const void *b);
int fof_compare_FOF_GList_LocCountTaskDiffMinID(const void *a, const void *b);
int fof_compare_FOF_GList_ExtCountMinID(const void *a, const void *b);
int fof_compare_Group_GrNr(const void *a, const void *b);
int fof_compare_Group_MinIDTask(const void *a, const void *b);
int fof_compare_Group_MinID(const void *a, const void *b);
int fof_compare_ID_list_GrNrID(const void *a, const void *b);
int fof_compare_Group_MinIDTask_MinID(const void *a, const void *b);
int fof_compare_Group_Len(const void *a, const void *b);

void fof_compute_group_properties(int gr, int start, int len);
void fof_exchange_group_data(void);
void fof_finish_group_properties(void);


extern int Ngroups, TotNgroups;
extern long long TotNids;

extern struct group_properties
{
  int Len;
  unsigned int Offset;
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

#ifdef SUBFIND
  int Nsubs;
  int FirstSub;
  MyDouble Pos[3];
  double M_Mean200, R_Mean200;
  double M_Crit200, R_Crit200;
  double M_TopHat200, R_TopHat200;
  int ContaminationLen;
  double ContaminationMass;
#ifdef SO_VEL_DISPERSIONS  
  double VelDisp_Mean200, VelDisp_Crit200, VelDisp_TopHat200;
#endif
#endif

} *Group;

#endif
