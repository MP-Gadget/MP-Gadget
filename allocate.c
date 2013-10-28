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

    double bytes_tot = 0;

    int NTaskTimesThreads;

    NTaskTimesThreads = NTask;
#ifdef NUM_THREADS
    NTaskTimesThreads = NUM_THREADS * NTask;
#endif

    Exportflag = (int *) mymalloc("Exportflag", NTaskTimesThreads * sizeof(int));
    Exportindex = (int *) mymalloc("Exportindex", NTaskTimesThreads * sizeof(int));
    Exportnodecount = (int *) mymalloc("Exportnodecount", NTaskTimesThreads * sizeof(int));

    Send_count = (int *) mymalloc("Send_count", sizeof(int) * NTask);
    Send_offset = (int *) mymalloc("Send_offset", sizeof(int) * NTask);
    Recv_count = (int *) mymalloc("Recv_count", sizeof(int) * NTask);
    Recv_offset = (int *) mymalloc("Recv_offset", sizeof(int) * NTask);

    ProcessedFlag = (unsigned char *) mymalloc("ProcessedFlag", bytes = All.MaxPart * sizeof(unsigned char));
    bytes_tot += bytes;

    NextActiveParticle = (int *) mymalloc("NextActiveParticle", bytes = All.MaxPart * sizeof(int));
    bytes_tot += bytes;

    NextInTimeBin = (int *) mymalloc("NextInTimeBin", bytes = All.MaxPart * sizeof(int));
    bytes_tot += bytes;

    PrevInTimeBin = (int *) mymalloc("PrevInTimeBin", bytes = All.MaxPart * sizeof(int));
    bytes_tot += bytes;


    if(All.MaxPart > 0)
    {
        if(!(P = (struct particle_data *) mymalloc("P", bytes = All.MaxPart * sizeof(struct particle_data))))
        {
            printf("failed to allocate memory for `P' (%g MB).\n", bytes / (1024.0 * 1024.0));
            endrun(1);
        }
        bytes_tot += bytes;

        if(ThisTask == 0)
            printf("\nAllocated %g MByte for particle storage.\n\n", bytes_tot / (1024.0 * 1024.0));
    }

    if(All.MaxPartSph > 0)
    {
        bytes_tot = 0;

        if(!
                (SphP =
                 (struct sph_particle_data *) mymalloc("SphP", bytes =
                     All.MaxPartSph * sizeof(struct sph_particle_data))))
        {
            printf("failed to allocate memory for `SphP' (%g MB).\n", bytes / (1024.0 * 1024.0));
            endrun(1);
        }
        bytes_tot += bytes;

        if(ThisTask == 0)
            printf("Allocated %g MByte for storage of SPH data.\n\n", bytes_tot / (1024.0 * 1024.0));

    }

    if(All.MaxPartBh > 0)
    {
        bytes_tot = 0;

        if(!
                (BhP =
                 (struct bh_particle_data *) mymalloc("BhP", bytes =
                     All.MaxPartBh * sizeof(struct bh_particle_data))))
        {
            printf("failed to allocate memory for `BhP' (%g MB).\n", bytes / (1024.0 * 1024.0));
            endrun(1);
        }
        bytes_tot += bytes;

        if(ThisTask == 0)
            printf("Allocated %g MByte for storage of BH data.\n\n", bytes_tot / (1024.0 * 1024.0));

    }
}
