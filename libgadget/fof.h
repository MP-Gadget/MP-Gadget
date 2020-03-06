#ifndef FOF_H
#define FOF_H

#include "forcetree.h"
#include "utils/paramset.h"
#include "timestep.h"

void set_fof_params(ParameterSet * ps);

void fof_init(double DMMeanSeparation);

struct BaseGroup {
    int OriginalTask;
    int OriginalIndex;
    int Length;
    int GrNr;
    MyIDType MinID;
    int MinIDTask;
    /* Note: this is in the translated frame,
     * subtract CurrentParticleOffset to get the physical frame.*/
    float FirstPos[3];
};

struct Group
{
    struct BaseGroup base;
    int Length;
    int LenType[6];
    double MassType[6];
    double Mass;
    /* Note: this is in the translated frame,
     * subtract CurrentParticleOffset to get the physical frame.*/
    double CM[3];
    double Vel[3];

    double Imom[3][3]; /* sum M r_j r_k */
    double Jmom[3]; /* sum M R_i x V_i  */

    double Sfr;
    /*These are used for storing black hole properties*/
    double BH_Mass;
    double BH_Mdot;
    double MaxDens;

    int seed_index;
    int seed_task;
};

/* Structure to hold all allocated FOF groups*/
typedef struct FOFGroups
{
    struct Group * Group;
    /* Ngroups is maximally NumPart,
     * so can be 32-bit*/
    int Ngroups;
    int64_t TotNgroups;
} FOFGroups;

/*Computes the Group structure, saved as a global array below*/
FOFGroups fof_fof(ForceTree * tree, MPI_Comm Comm);

/*Frees the Group structure*/
void fof_finish(FOFGroups * fof);

/*Uses the Group structure to seed blackholes.
 * The tree and active particle structs are used only because we may need to reallocate them. */
void fof_seed(FOFGroups * fof, ForceTree * tree, ActiveParticles * act, MPI_Comm Comm);

/*Saves the Group structure to disc.*/
void fof_save_groups(FOFGroups * fof, int num, MPI_Comm Comm);


#endif
