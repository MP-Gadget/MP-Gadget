#include <string.h>
#include "partmanager.h"
#include "mymalloc.h"
#include "endrun.h"

/* Local number of particles */
int NumPart;
int part_MaxPart;

/*! This structure holds all the information that is
 * stored for each particle of the simulation on the local processor.
 */
struct particle_data *P;
/*Softening table*/
double GravitySofteningTable[6];


void
particle_alloc_memory(int MaxPart)
{
    size_t bytes;
    part_MaxPart = MaxPart;

    P = (struct particle_data *) mymalloc("P", bytes = MaxPart * sizeof(struct particle_data));

    /* clear the memory to avoid valgrind errors;
     *
     * note that I tried to set each component in P to zero but
     * valgrind still complains in PFFT
     * seems to be to do with how the struct is padded and
     * the missing holes being accessed by __kmp_atomic functions.
     * (memory lock etc?)
     * */
    memset(P, 0, sizeof(struct particle_data) * MaxPart);
#ifdef OPENMP_USE_SPINLOCK
    {
        int i;
        for(i = 0; i < MaxPart; i ++) {
            pthread_spin_init(&P[i].SpinLock, 0);
        }
    }
#endif
    message(0, "Allocated %g MByte for particle storage.\n", bytes / (1024.0 * 1024.0));
}
