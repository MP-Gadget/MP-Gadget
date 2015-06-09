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
  struct group_particle_host DenseGas;
  struct group_particle_host CentralGalaxy;
#endif

} *Group;

#endif
