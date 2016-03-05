#ifndef FOF_H
#define FOF_H

#ifdef FOF

#include "allvars.h"

int fof_compare_FOF_PList_MinID(const void *a, const void *b);
int fof_compare_FOF_GList_MinID(const void *a, const void *b);
int fof_compare_FOF_GList_MinIDTask(const void *a, const void *b);
int fof_compare_Group_MinIDTask(const void *a, const void *b);
int fof_compare_Group_MinID(const void *a, const void *b);

void fof_compute_group_properties(int gr, int start, int len);
void fof_exchange_group_data(void);
void fof_finish_group_properties(void);

int fof_find_dmparticles_evaluate(int target, int mode, int *nexport, int *nsend_local);

void fof_compute_group_properties(int gr, int start, int len);

void fof_fof(int num);
void fof_find_groups(void);
void fof_exchange_id_lists(void);
void fof_compile_catalogue(void);
void fof_save_groups(int num);
double fof_periodic(double x);
double fof_periodic_wrap(double x);
void fof_find_nearest_dmparticle(void);

void fof_make_black_holes(void);


extern uint64_t Ngroups, TotNgroups;
extern int64_t TotNids;

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

} *Group;

#endif

#endif
