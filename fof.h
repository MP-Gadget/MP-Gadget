#ifndef FOF_H
#define FOF_H

#ifdef FOF

#include "allvars.h"

void fof_init();
void fof_fof(int num);

extern uint64_t Ngroups, TotNgroups;
extern int64_t TotNids;

struct BaseGroup {
    int OriginalTask;
    int OriginalIndex;
    int Length;
    int GrNr;
    MyIDType MinID;
    int MinIDTask;
    float FirstPos[3];
};

extern struct Group 
{
    struct BaseGroup base; 
    int Length;
    int LenType[6];
    double MassType[6];
    double Mass;
    double CM[3];
    double Vel[3];

    double Imom[3][3];

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
} * Group;

#endif

#endif
