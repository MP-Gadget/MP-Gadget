#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "allvars.h"
#include "proto.h"
#include "treewalk.h"
#include "mymalloc.h"
#include "endrun.h"
#include "timestep.h"

/* This routine allocates memory for
 * particle storage, both the collisionless and the SPH particles.
 * The memory for the ordered binary tree of the timeline
 * is also allocated.
 */
void allocate_memory(void)
{
    size_t bytes;
    TreeWalk_allocate_memory();
    timestep_allocate_memory(All.MaxPart);

    P = (struct particle_data *) mymalloc("P", bytes = All.MaxPart * sizeof(struct particle_data));

    /* clear the memory to avoid valgrind errors;
     *
     * note that I tried to set each component in P to zero but
     * valgrind still complains in PFFT
     * seems to be to do with how the struct is padded and
     * the missing holes being accessed by __kmp_atomic fucntions. 
     * (memory lock etc?)
     * */
    memset(P, 0, sizeof(struct particle_data) * All.MaxPart);
#ifdef OPENMP_USE_SPINLOCK
    {
        int i;
        for(i = 0; i < All.MaxPart; i ++) {
            pthread_spin_init(&P[i].SpinLock, 0);
        }
    }
#endif
    message(0, "Allocated %g MByte for particle storage.\n", bytes / (1024.0 * 1024.0));

    if(NTotal[0] > 0) {
        SphP = (struct sph_particle_data *) mymalloc("SphP", bytes =
                     All.MaxPart * sizeof(struct sph_particle_data));
        message(0, "Allocated %g MByte for storage of SPH data.\n", bytes / (1024.0 * 1024.0));
    }
    if(All.BlackHoleOn) {
        BhP = (struct bh_particle_data *) mymalloc("BhP", bytes =
                     All.MaxPartBh * sizeof(struct bh_particle_data));
        message(0, "Allocated %g MByte for storage of BH data.\n", bytes / (1024.0 * 1024.0));
    }
}
