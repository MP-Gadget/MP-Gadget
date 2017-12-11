#ifndef __GARBAGE_H
#define __GARBAGE_H
#include "event.h"

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
void slots_reserve(int atleast[6], int collective);
void slots_check_id_consistency();

typedef struct {
    EIBase base;
    int parent;
    int child;
} EISlotsFork;

#endif
