#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "allvars.h"
#include "proto.h"







/* This routine allocates memory for
 * particle storage, both the collisionless and the SPH particles.
 * The memory for the ordered binary tree of the timeline
 * is also allocated.
 */
void allocate_memory(void)
{
    size_t bytes;

    int NTaskTimesThreads;

    NTaskTimesThreads = All.NumThreads * NTask;

    Exportflag = (int *) mymalloc("Exportflag", NTaskTimesThreads * sizeof(int));
    Exportindex = (int *) mymalloc("Exportindex", NTaskTimesThreads * sizeof(int));
    Exportnodecount = (int *) mymalloc("Exportnodecount", NTaskTimesThreads * sizeof(int));

    Send_count = (int *) mymalloc("Send_count", sizeof(int) * NTask);
    Send_offset = (int *) mymalloc("Send_offset", sizeof(int) * NTask);
    Recv_count = (int *) mymalloc("Recv_count", sizeof(int) * NTask);
    Recv_offset = (int *) mymalloc("Recv_offset", sizeof(int) * NTask);

    NextActiveParticle = (int *) mymalloc("NextActiveParticle", bytes = All.MaxPart * sizeof(int));

    NextInTimeBin = (int *) mymalloc("NextInTimeBin", bytes = All.MaxPart * sizeof(int));

    PrevInTimeBin = (int *) mymalloc("PrevInTimeBin", bytes = All.MaxPart * sizeof(int));

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
    if(ThisTask == 0)
            printf("\nAllocated %g MByte for particle storage.\n\n", bytes / (1024.0 * 1024.0));

    SphP = (struct sph_particle_data *) mymalloc("SphP", bytes =
                     All.MaxPartSph * sizeof(struct sph_particle_data));
    if(ThisTask == 0)
        printf("Allocated %g MByte for storage of SPH data.\n\n", bytes / (1024.0 * 1024.0));
    BhP = (struct bh_particle_data *) mymalloc("BhP", bytes =
                     All.MaxPartBh * sizeof(struct bh_particle_data));
    if(ThisTask == 0)
        printf("Allocated %g MByte for storage of BH data.\n\n", bytes / (1024.0 * 1024.0));
}
