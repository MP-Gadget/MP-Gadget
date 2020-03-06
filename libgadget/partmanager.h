#ifndef _PART_DATA_H
#define _PART_DATA_H

#include "types.h"
#include "utils/peano.h"

/*! This structure holds all the information that is
 * stored for each particle of the simulation.
 */
struct particle_data
{
    inttime_t Ti_drift;       /*!< current time of the particle position */
    inttime_t Ti_kick;        /*!< current time of the particle momentum */

    double Pos[3];   /*!< particle position at its current time */
    float Mass;     /*!< particle mass */

    struct {
        /* particle type.  0=gas, 1=halo, 2=disk, 3=bulge, 4=stars, 5=bndry */
        unsigned int Type                 :4;

        unsigned int IsGarbage            :1; /* True for a garbage particle. readonly: Use slots_mark_garbage to mark this.*/
        unsigned int Swallowed            :1; /* True if the particle is being swallowed; used in BH to determine swallower and swallowee;*/
        unsigned int spare_1              :1; /*Unused, ensures alignment to a char*/
        unsigned int spare_2              :1;
        unsigned char Generation; /* How many particles it has spawned; used to generate unique particle ID.
                                     may wrap around with too many SFR/BH if a feedback model goes rogue */

        signed char TimeBin; /* Time step bin; -1 for unassigned.*/
        /* To ensure alignment to a 32-bit boundary.*/
        unsigned char HeIIIionized; /* True if the particle has undergone helium reionization.
                                     * This could be a bitfield: it isn't because we need to change it in an atomic.
                                     * Changing a bitfield in an atomic seems to work in OpenMP 5.0 on gcc 9 and icc 18 and 19,
                                     * so we should be able to make it a bitfield at some point. */
    };

    int PI; /* particle property index; used by BH, SPH and STAR.
                        points to the corresponding structure in (SPH|BH|STAR)P array.*/
    MyIDType ID;

    MyFloat Vel[3];   /* particle velocity at its current time */
    MyFloat GravAccel[3];  /* particle acceleration due to short-range gravity */

    MyFloat GravPM[3];		/* particle acceleration due to long-range PM gravity force */

    MyFloat Potential;		/* gravitational potential. This is the total potential after gravtree+gravpm is called. */

    MyFloat Hsml;

    /* Union these two because they are transients: they are hard to move
     * to private arrays because they need to travel with the particle during exchange*/
    union {
        /* The peano key is a hash of the position used in the domain decomposition.
         * It is slow to generate so we store it here.*/
        peano_t Key; /* only by domain.c and force_tree_rebuild */
        /* FOF Group number: only has meaning during FOF.*/
        int64_t GrNr;
    };

};

extern struct part_manager_type {
    struct particle_data *Base; /* Pointer to particle data on local processor. */
    /*!< number of particles on the LOCAL processor: number of valid entries in P array. */
    int NumPart;
    /*!< Amount of memory we have available for particles locally: maximum size of P array. */
    int MaxPart;
    /* Random shift applied to the box. This is changed
     * every domain decomposition to prevent correlated
     * errors building up in the tree force. */
    double CurrentParticleOffset[3];
} PartManager[1];

/*Compatibility define*/
#define P PartManager->Base

/*Allocate memory for the particles*/
void particle_alloc_memory(int MaxPart);

/* gravitational and hydrodynamical softening lengths (given in terms of an `equivalent' Plummer softening
    * length)
    *
    * five groups of particles are supported 0=gas,1=halo,2=disk,3=bulge,4=stars
    */
extern double GravitySofteningTable[6];

static inline double FORCE_SOFTENING(int i)
{
    if (GravitySofteningTable[0] == 0 && P[i].Type == 0) {
        return P[i].Hsml;
    } else {
        /* Force is newtonian beyond this.*/
        return 2.8 * GravitySofteningTable[P[i].Type];
    }
}

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
static inline int IMAX(int a, int b) {
    if(a > b) return a;
    return b;
}
static inline int IMIN(int a, int b) {
    if(a < b) return a;
    return b;
}

#endif
