#ifndef _PART_DATA_H
#define _PART_DATA_H

#include "types.h"
#include "utils/peano.h"

/*! This structure holds all the information that is
 * stored for each particle of the simulation.
 */
struct particle_data
{
    float GravCost;     /*!< weight factor used for balancing the work-load */

    inttime_t Ti_drift;       /*!< current time of the particle position */
    inttime_t Ti_kick;        /*!< current time of the particle momentum */

    double Pos[3];   /*!< particle position at its current time */
    float Mass;     /*!< particle mass */

    struct {
        /* particle type.  0=gas, 1=halo, 2=disk, 3=bulge, 4=stars, 5=bndry */
        unsigned int Type                 :4;

        unsigned int IsGarbage            :1; /* True for a garbage particle. readonly: Use slots_mark_garbage to mark this.*/
        unsigned int DensityIterationDone :1; /* True if the density-like iterations already finished; */
        unsigned int Swallowed            :1; /* True if the particle is being swallowed; used in BH to determine swallower and swallowee;*/
        unsigned int spare_0            :1;

        unsigned char Generation; /* How many particles it has spawned; used to generate unique particle ID.
                                     may wrap around with too many SFR/BH if a feedback model goes rogue */

        signed char TimeBin; /* Time step bin; -1 for unassigned.*/
    };

    int PI; /* particle property index; used by BH, SPH and STAR.
                        points to the corresponding structure in (SPH|BH|STAR)P array.*/
    MyIDType ID;

    MyFloat Vel[3];   /* particle velocity at its current time */
    MyFloat GravAccel[3];  /* particle acceleration due to short-range gravity */

    MyFloat GravPM[3];		/* particle acceleration due to long-range PM gravity force */

    MyFloat Potential;		/* gravitational potential. This is the total potential after gravtree+gravpm is called. */

    MyFloat Hsml;

    /* The peano key is a hash of the position used in the domain decomposition.
     * It is slow to generate so we store it here.*/
    peano_t Key; /* only by domain.c and forcetre.c */

    union {
        /* the following variables are transients.
         * FIXME: move them into the corresponding modules! Is it possible? */

        MyFloat NumNgb; /* Number of neighbours; only used in density.c */

        int RegionInd; /* which region the particle belongs to; only by petapm.c */

        struct {
            /* used by fof.c which calls domain_exchange that doesn't uses peano_t */
            int64_t GrNr;
            int origintask;
        };
    };

};

extern struct part_manager_type {
    struct particle_data *Base; /* Pointer to particle data on local processor. */
    /*!< number of particles on the LOCAL processor: number of valid entries in P array. */
    int NumPart;
    /*!< Amount of memory we have available for particles locally: maximum size of P array. */
    int MaxPart;
} PartManager[1];

/*Compatibility define*/
#define P PartManager->Base

/*Allocate memory for the particles*/
void particle_alloc_memory(int MaxPart);

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
