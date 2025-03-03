#include <string.h>
#include <math.h>

#include "partmanager.h"
#include "utils/mymalloc.h"
#include "utils/endrun.h"
#include "utils/system.h"

/*! This structure holds all the information that is
 * stored for each particle of the simulation on the local processor.
 */
struct part_manager_type PartManager[1] = {{0}};

void
particle_alloc_memory(struct part_manager_type * PartManager, double BoxSize, int64_t MaxPart)
{
    size_t bytes;
    PartManager->Base = (struct particle_data *) mymalloc("P", bytes = MaxPart * sizeof(struct particle_data));
    PartManager->MaxPart = MaxPart;
    PartManager->NumPart = 0;
    if(MaxPart >= 1L<<31 || MaxPart < 0)
        endrun(5, "Trying to store %ld particles on a single node, more than fit in an int32, not supported\n", MaxPart);
    memset(PartManager->CurrentParticleOffset, 0, 3*sizeof(double));
    memset(PartManager->Xmin, 0, 3*sizeof(double));
    memset(PartManager->Xmax, 0, 3*sizeof(double));

    PartManager->BoxSize = BoxSize;
    /* clear the memory to avoid valgrind errors;
     *
     * note that I tried to set each component in P to zero but
     * valgrind still complains in PFFT
     * seems to be to do with how the struct is padded and
     * the missing holes being accessed by __kmp_atomic functions.
     * (memory lock etc?)
     * */
    memset(PartManager->Base, 0, sizeof(struct particle_data) * MaxPart);
    message(0, "Allocated %g MByte for storing %ld particles.\n", bytes / (1024.0 * 1024.0), MaxPart);
}

/* We operate in a situation where the particles are in a coordinate frame
 * offset slightly from the ICs (to avoid correlated tree errors).
 * This function updates the global variable containing that offset, and
 * stores the relative shift from the last offset in the rel_random_shift output
 * array. */
void
update_random_offset(struct part_manager_type * PartManager, double * rel_random_shift, double RandomParticleOffset, const uint64_t seed)
{
    /* Note random number table is the same on all processors*/
    RandTable rnd = set_random_numbers(seed, 3);
    int i;
    for (i = 0; i < 3; i++) {
        /* Note random number table is the same on all processors*/
        double rr = get_random_number(i, &rnd);
        /* Upstream Gadget uses a random fraction of the box, but since all we need
         * is to adjust the tree openings, and the tree force is zero anyway on the
         * scale of a few PM grid cells, this seems enough.*/
        rr *= RandomParticleOffset * PartManager->BoxSize;
        /* Subtract the old random shift first.*/
        rel_random_shift[i] = rr - PartManager->CurrentParticleOffset[i];
        PartManager->CurrentParticleOffset[i] = rr;
    }
    free_random_numbers(&rnd);
    message(0, "Internal particle offset is now %g %g %g\n", PartManager->CurrentParticleOffset[0], PartManager->CurrentParticleOffset[1], PartManager->CurrentParticleOffset[2]);
#ifdef DEBUG
    /* Check explicitly that the vector is the same on all processors*/
    double test_random_shift[3] = {0};
    for (i = 0; i < 3; i++)
        test_random_shift[i] = PartManager->CurrentParticleOffset[i];
    MPI_Bcast(test_random_shift, 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    for (i = 0; i < 3; i++)
        if(test_random_shift[i] != PartManager->CurrentParticleOffset[i])
            endrun(44, "Random shift %d is %g != %g on task 0!\n", i, test_random_shift[i], PartManager->CurrentParticleOffset[i]);
#endif
}

void
update_offset(struct part_manager_type * PartManager, double * rel_random_shift)
{
    int i;
    message(0, " *** particle offset before update *** %g %g %g\n", PartManager->CurrentParticleOffset[0], PartManager->CurrentParticleOffset[1], PartManager->CurrentParticleOffset[2]);
    for (i = 0; i < 3; i++) {
        /* Note: -Xmin should be the rellative random_shift */
        rel_random_shift[i] = - PartManager->Xmin[i] + 200.; /* TODO: update this ad hoc 50kpc padding*/    
        PartManager->CurrentParticleOffset[i] = rel_random_shift[i] + PartManager->CurrentParticleOffset[i];
    }
    message(0, "*** relative particle offset *** %g %g %g\n", rel_random_shift[0], rel_random_shift[1], rel_random_shift[2]);
    message(0, "Internal particle offset is now %g %g %g\n", PartManager->CurrentParticleOffset[0], PartManager->CurrentParticleOffset[1], PartManager->CurrentParticleOffset[2]);
}

/* Calculate the box size based on particle positions*/
void  
set_lbox_nonperiodic(struct part_manager_type * PartManager) {

    int i;
    double xmin = 1.0e30; 
    double ymin = 1.0e30; 
    double zmin = 1.0e30; 

    double xmax = -1.0e30; 
    double ymax = -1.0e30; 
    double zmax = -1.0e30; 


    #pragma omp parallel for reduction(min: xmin,ymin,zmin) reduction(max: xmax,ymax,zmax)
    for(i = 0; i < PartManager->NumPart; i ++) {
        if(P[i].IsGarbage || P[i].Swallowed)
            continue;
        // const struct particle_data * pp = &PartManager->Base[i];
        xmin = (xmin < P[i].Pos[0]) ? xmin:P[i].Pos[0]; 
        ymin = (ymin < P[i].Pos[1]) ? ymin:P[i].Pos[1]; 
        zmin = (zmin < P[i].Pos[2]) ? zmin:P[i].Pos[2]; 
        xmax = (xmax > P[i].Pos[0]) ? xmax:P[i].Pos[0]; 
        ymax = (ymax > P[i].Pos[1]) ? ymax:P[i].Pos[1]; 
        zmax = (zmax > P[i].Pos[2]) ? zmax:P[i].Pos[2]; 
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &xmin, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &ymin, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &zmin, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);

    MPI_Allreduce(MPI_IN_PLACE, &xmax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &ymax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &zmax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    PartManager->Xmin[0] = xmin;
    PartManager->Xmin[1] = ymin;
    PartManager->Xmin[2] = zmin;

    PartManager->Xmax[0] = xmax;
    PartManager->Xmax[1] = ymax;
    PartManager->Xmax[2] = zmax;

    MPI_Barrier(MPI_COMM_WORLD);

    for(i = 0; i < 3; i ++) {
        if ((PartManager->Xmax[i] - PartManager->Xmin[i]) + 400 > PartManager->BoxSize)
            PartManager->BoxSize = (PartManager->Xmax[i] - PartManager->Xmin[i])  + 400;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    message(0, "***** Inside PartManager set_lbox_nonperiodic ****\n");
    message(0, "***** Xmin=(%g, %g, %g)  **** \n", PartManager->Xmin[0], PartManager->Xmin[1], PartManager->Xmin[2]);
    message(0, "***** Xmax=(%g, %g, %g)  **** \n", PartManager->Xmax[0], PartManager->Xmax[1], PartManager->Xmax[2]);
    message(0, "***** Box=%g  **** \n", PartManager->BoxSize);
    MPI_Barrier(MPI_COMM_WORLD);



}
