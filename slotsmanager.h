#ifndef __GARBAGE_H
#define __GARBAGE_H

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

int domain_fork_particle(int parent, int ptype);
int domain_garbage_collection(void);
void domain_slots_init();
void domain_slots_grow(int newSlots[6]);

#endif
