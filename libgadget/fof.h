#ifndef FOF_H
#define FOF_H

#include "allvars.h"

void fof_init(void);

/*Computes the Group structure, saved as a global array below*/
void fof_fof(void);

/*Frees the Group structure*/
void
fof_finish(void);

/*Uses the Group structure to seed blackholes*/
void
fof_seed(void);

/*Saves the Group structure to disc.*/
void
fof_save_groups(int num);

extern uint64_t Ngroups, TotNgroups;

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

    double Imom[3][3]; /* sum M r_j r_k */
    double Jmom[3]; /* sum M R_i x V_i  */

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
