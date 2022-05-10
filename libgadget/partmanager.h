#ifndef _PART_DATA_H
#define _PART_DATA_H

#include "types.h"
#include "utils/peano.h"

/*! This structure holds all the information that is
 * stored for each particle of the simulation.
 */
struct particle_data
{
    double Pos[3];   /*!< particle position at its current time */
    float Mass;     /*!< particle mass */

    struct {
        /* particle type.  0=gas, 1=halo, 2=disk, 3=bulge, 4=stars, 5=bndry */
        unsigned int Type                 :4;

        unsigned int IsGarbage            :1; /* True for a garbage particle. readonly: Use slots_mark_garbage to mark this.*/
        unsigned int Swallowed            :1; /* True if the particle is being swallowed; used in BH to determine swallower and swallowee;*/
        unsigned int spare_1              :1; /*Unused, ensures alignment to a char*/
        unsigned int BHHeated              :1; /* Flags that particle was heated by a BH this timestep*/
        unsigned char Generation; /* How many particles it has spawned; used to generate unique particle ID.
                                     may wrap around with too many SFR/BH if a feedback model goes rogue */

        unsigned char TimeBinHydro; /* Time step bin for hydro; 0 for unassigned. Must be smaller than the gravity timebin.
                                     * Star formation, cooling, and BH accretion takes place on the hydro timestep.
                                     * Dynamic friction is also the hydro timestep because it relies on the gas density. */
        unsigned char TimeBinGravity; /* Time step bin for gravity; 0 for unassigned.*/
        unsigned char HeIIIionized; /* True if the particle has undergone helium reionization.
                                     * This could be a bitfield: it isn't because we need to change it in an atomic.
                                     * Changing a bitfield in an atomic seems to work in OpenMP 5.0 on gcc 9 and icc 18 and 19,
                                     * so we should be able to make it a bitfield at some point. */
        /* To ensure alignment to a 32-bit boundary.*/
        unsigned char spare[3];
    };

    int PI; /* particle property index; used by BH, SPH and STAR.
                        points to the corresponding structure in (SPH|BH|STAR)P array.*/
    inttime_t Ti_drift;       /*!< current time of the particle position. The same for all particles. */

#ifdef DEBUG
    /* Kick times for both hydro and grav*/
    inttime_t Ti_kick_hydro;
    inttime_t Ti_kick_grav;
#endif
    MyIDType ID;

    MyFloat Vel[3];   /* particle velocity at its current time */
    MyFloat GravAccel[3];  /* Particle acceleration due to short-range gravity.
                            * For non-hierarchical gravity this is the acceleration from all particles.
                            * For hierarchical gravity it is the acceleration from all active particles.*/

    MyFloat GravPM[3];		/* particle acceleration due to long-range PM gravity force */

    MyFloat OldAcc; /* Magnitude of full acceleration on the last PM step, for tree opening criterion. */
    MyFloat Potential;		/* Gravitational potential. This is the total potential only on a PM timestep,
                             * after gravtree+gravpm is called. We do not save the potential on short timesteps
                             * for hierarchical gravity as it would only be from active particles.*/

    /* DtHsml is 1/3 DivVel * Hsml evaluated at the last active timestep for this particle.
     * This predicts Hsml during the current timestep in the way used in Gadget-4, more accurate
     * than the Gadget-2 prediction which could run away in deep timesteps. Used also
     * to limit timesteps by density change. */
    union {
        MyFloat DtHsml;
        /* This is the destination task during the fof particle exchange.
         * It is never used outside of that code, and the
         * particles are copied into a new PartManager before setting it,
         * so it is safe to union with DtHsml.*/
        int TargetTask;
    };
    MyFloat Hsml;

    /* These two are transient but hard to move
     * to private arrays because they need to travel
     * with the particle during exchange*/
    /* The peano key is a hash of the position used in the domain decomposition.
     * It is slow to generate and used to rebuild the tree, so we store it here.*/
    peano_t Key; /* only by domain.c and force_tree_rebuild */
    /* FOF Group number: only has meaning during FOF.*/
    int64_t GrNr;
};

extern struct part_manager_type {
    struct particle_data *Base; /* Pointer to particle data on local processor. */
    /*!< number of particles on the LOCAL processor: number of valid entries in P array. */
    int64_t NumPart;
    /*!< Amount of memory we have available for particles locally: maximum size of P array. */
    int64_t MaxPart;
    /* Random shift applied to the box. This is changed
     * every domain decomposition to prevent correlated
     * errors building up in the tree force. */
    double CurrentParticleOffset[3];
    /* Current box size so we can work out periodic boundaries*/
    double BoxSize;
} PartManager[1];

/*Compatibility define*/
#define P PartManager->Base

/*Allocate memory for the particles*/
void particle_alloc_memory(struct part_manager_type * PartManager, double BoxSize, int64_t MaxPart);

/* Updates the global storing the current random offset of the particles,
 * and stores the relative offset from the last random offset in rel_random_shift.
 * RandomParticleOffset is the max adjustment as a fraction of the box. */
void update_random_offset(struct part_manager_type * PartManager, double * rel_random_shift, double RandomParticleOffset);

/* Finds the correct relative position accounting for periodicity*/
#define NEAREST(x, BoxSize) (((x)>0.5*BoxSize)?((x)-BoxSize):(((x)<-0.5*BoxSize)?((x)+BoxSize):(x)))

static inline double DMAX(double a, double b) {
    if(a > b) return a;
    return b;
}
static inline double DMIN(double a, double b) {
    if(a < b) return a;
    return b;
}
#endif
