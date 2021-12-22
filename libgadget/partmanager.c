#include <string.h>

#include "utils.h"

#include "partmanager.h"

/*! This structure holds all the information that is
 * stored for each particle of the simulation on the local processor.
 */
struct part_manager_type PartManager[1] = {{0}};

void
particle_alloc_memory(double BoxSize, int64_t MaxPart)
{
    size_t bytes;
    PartManager->Base = (struct particle_data *) mymalloc("P", bytes = MaxPart * sizeof(struct particle_data));
    PartManager->MaxPart = MaxPart;
    PartManager->NumPart = 0;
    if(MaxPart >= 1L<<31 || MaxPart < 0)
        endrun(5, "Trying to store %ld particles on a single node, more than fit in an int32, not supported\n", MaxPart);
    memset(PartManager->CurrentParticleOffset, 0, 3*sizeof(double));

    PartManager->BoxSize = BoxSize;
    /* clear the memory to avoid valgrind errors;
     *
     * note that I tried to set each component in P to zero but
     * valgrind still complains in PFFT
     * seems to be to do with how the struct is padded and
     * the missing holes being accessed by __kmp_atomic functions.
     * (memory lock etc?)
     * */
    memset(P, 0, sizeof(struct particle_data) * MaxPart);
    message(0, "Allocated %g MByte for storing %ld particles.\n", bytes / (1024.0 * 1024.0), MaxPart);
}
