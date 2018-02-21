#ifndef __GARBAGE_H
#define __GARBAGE_H
#include <mpi.h>
#include "utils.h"

#include "types.h"
#include "partmanager.h"

extern struct slots_manager_type {
    char * Base; /* memory ptr that holds of all slots */
    struct {
        char * ptr; /* aliasing ptr for this slot */
        int maxsize; /* max number of supported slots */
        int size; /* currently used slots*/
        size_t elsize; /* itemsize */
        int enabled;
    } info[6];
} SlotsManager[1];

/* Slot particle data structures: first the base extension slot, then black holes,
 * then stars, then SPH. SPH still has some compile-time optional elements.
 * Each particle also has the base data, stored in particle_data.*/
struct particle_data_ext {
    struct {
       /* used at GC for reverse link to P */
        int ReverseLink;
    } gc;
    unsigned int IsGarbage : 1; /* marked if the slot is garbage. use slots_mark_garbage to mark this with the base particle index*/
    MyIDType ID; /* for data consistency check, same as particle ID */
};

/* Data stored for each black hole in addition to collisionless data*/
struct bh_particle_data {
    struct particle_data_ext base;

    int CountProgs;

    MyFloat Mass;
    MyFloat Mdot;
    MyFloat FeedbackWeightSum;
    MyFloat Density;
    MyFloat Entropy;
    MyFloat Pressure;
    MyFloat SurroundingGasVel[3];
    MyFloat FormationTime;		/*!< formation time of black hole. */

    MyFloat accreted_Mass;
    MyFloat accreted_BHMass;
    MyFloat accreted_momentum[3];

    int JumpToMinPot;
    double  MinPotPos[3];
    MyFloat MinPotVel[3];
    MyFloat MinPot;

    MyIDType SwallowID; /* Allows marking of a merging particle. Used only in blackhole.c.
                           Set to -1 in init.c and only reinitialised if a merger takes place.*/

    short int TimeBinLimit;
};

/*Data for each star particle*/
struct star_particle_data
{
    struct particle_data_ext base;
    MyFloat FormationTime;		/*!< formation time of star particle */
    MyFloat BirthDensity;		/*!< Density of gas particle at star formation. */
    MyFloat Metallicity;		/*!< metallicity of star particle */
};

/* the following structure holds data that is stored for each SPH particle in addition to the collisionless
 * variables.
 */
struct sph_particle_data
{
    struct particle_data_ext base;

#ifdef DENSITY_INDEPENDENT_SPH
    MyFloat EgyWtDensity;           /*!< 'effective' rho to use in hydro equations */
    MyFloat DhsmlEgyDensityFactor;  /*!< correction factor for density-independent entropy formulation */
#define EOMDensity EgyWtDensity
#define DhsmlEOMDensityFactor DhsmlEgyDensityFactor
#else
#define EOMDensity Density
#define DhsmlEOMDensityFactor DhsmlDensityFactor
#endif
    MyFloat EntVarPred;         /*!< Predicted entropy at current particle drift time for SPH computation*/
    MyFloat Metallicity;		/*!< metallicity of gas particle */
    MyFloat Entropy;		/*!< Entropy (actually entropic function) at kick time of particle */
    MyFloat MaxSignalVel;           /*!< maximum signal velocity */
    MyFloat       Density;		/*!< current baryonic mass density of particle */
    MyFloat       DtEntropy;		/*!< rate of change of entropy */
    MyFloat       HydroAccel[3];	/*!< acceleration due to hydrodynamical force */
    MyFloat       DhsmlDensityFactor;	/*!< correction factor needed in entropy formulation of SPH */
    MyFloat       DivVel;		/*!< local velocity divergence */
    MyFloat       CurlVel;     	        /*!< local velocity curl */
    MyFloat       Rot[3];		/*!< local velocity curl */
    MyFloat Ne;  /*!< electron fraction, expressed as local electron number
                   density normalized to the hydrogen number density. Gives
                   indirectly ionization state and mean molecular weight. */

#ifdef BLACK_HOLES
    MyIDType SwallowID; /* Allows marking of a particle being eaten by a black hole. Used only in blackhole.c.
                           Set to -1 in init.c and only reinitialised if a merger takes place.*/
    MyFloat       Injected_BH_Energy;
#endif

#ifdef SFR
    MyFloat DelayTime;		/*!< SH03: remaining maximum decoupling time of wind particle */
                            /*!< VS08: remaining waiting for wind particle to be eligible to form winds again */
#endif

#ifdef SPH_GRAD_RHO
    MyFloat GradRho[3];
#endif
};

/* shortcuts for accessing different slots directly by the index */
#define SphP ((struct sph_particle_data*) SlotsManager->info[0].ptr)
#define StarP ((struct star_particle_data*) SlotsManager->info[4].ptr)
#define BhP ((struct bh_particle_data*) SlotsManager->info[5].ptr)

/* shortcuts for accessing slots from base particle index */
#define SPHP(i) SphP[P[i].PI]
#define BHP(i) BhP[P[i].PI]
#define STARP(i) StarP[P[i].PI]

extern MPI_Datatype MPI_TYPE_PARTICLE;
extern MPI_Datatype MPI_TYPE_SLOT[6];

/* shortcuts to access base slot attributes */
#define BASESLOT_PI(PI, ptype) ((struct particle_data_ext *)(SlotsManager->info[ptype].ptr + SlotsManager->info[ptype].elsize * (PI)))
#define BASESLOT(i) BASESLOT_PI(P[i].PI, P[i].Type)

void slots_init(void);
/*Enable a slot on type ptype. All slots are disabled after slots_init().*/
void slots_set_enabled(int ptype, size_t elsize);
void slots_free();
void slots_mark_garbage(int i);
void slots_setup_topology();
void slots_setup_id();
int slots_fork(int parent, int ptype);
int slots_gc(int * compact_slots);
void slots_gc_sorted(void);
void slots_reserve(int where, int atleast[6]);
void slots_check_id_consistency();

typedef struct {
    EIBase base;
    int parent;
    int child;
} EISlotsFork;

#endif
