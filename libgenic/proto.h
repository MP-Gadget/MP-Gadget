#ifndef GENIC_PROTO_H
#define GENIC_PROTO_H
#include <bigfile.h>
#include <stdint.h>
#include "power.h"
#include "allvars.h"
#include "libgadget/petapm.h"

typedef struct IDGenerator {
    int size[3];
    int offset[3];
    int Ngrid;
    double BoxSize;
    int NumPart;
} IDGenerator;

void
idgen_init(IDGenerator * idgen, PetaPM * pm, int Ngrid, double BoxSize);

uint64_t
idgen_create_id_from_index(IDGenerator * idgen, int index);

void
idgen_create_pos_from_index(IDGenerator * idgen, int index, double pos[3]);

/* Compute the displacement and velocity from the initial homogeneous particle distribution,
 * using the cosmological transfer functions. */
void displacement_fields(PetaPM * pm, enum TransferType Type, struct ic_part_data * dispICP, const int NumPart, Cosmology * CP, const struct genic_config GenicConfig);

/* Fill ICP with NumPart particles spaced on a regular 3D grid, whose structure is stored in the IDGenerator. */
int setup_grid(IDGenerator * idgen, double shift, double mass, struct ic_part_data * ICP);

/* Fill ICP with NumPart particles spaced out as a Lagrangian glass, calling glass_evolve
 * to move the particles with reversed gravity. */
int setup_glass(IDGenerator * idgen, PetaPM * pm, double shift, int seed, double mass, struct ic_part_data * ICP, const double UnitLength_in_cm, const char * OutputDir);

/* Evolve a distribution of particles with a reversed gravitational force. */
void glass_evolve(PetaPM * pm, int nsteps, const char * pkoutname, struct ic_part_data * ICP, const int NumPart, const double UnitLength_in_cm, const char * OutputDir);

/* Save the header of the ICs. */
void saveheader(BigFile * bf, int64_t TotNumPartCDM, int64_t TotNumPartGas, int64_t TotNuPart, double nufrac, const double BoxSize, Cosmology * CP, const struct genic_config GenicConfig);

/*Compute the mass array from the cosmology*/
void compute_mass(double * mass, int64_t TotNumPartCDM, int64_t TotNumPartGas, int64_t TotNuPart, double nufrac, const double BoxSize, Cosmology * CP, const struct genic_config GenicConfig);

/* Save positions, velocities and IDs of a particle type to the ICs. */
void
write_particle_data(IDGenerator * idgen,
                    const int Type,
                    BigFile * bf,
                    const uint64_t FirstID,
                    const int SavePrePos,
                    int NumFiles, int NumWriters,
                    struct ic_part_data * curICP);

/*Read a parameter file*/
void read_parameterfile(char *fname, struct genic_config * GenicConfig, int * ShowBacktrace, double * MaxMemSizePerNode, Cosmology * CP);

/*Defined in save.c used in power.c*/
void _bigfile_utils_create_block_from_c_array(BigFile * bf, void * baseptr, const char * name, const char * dtype, size_t dims[], ptrdiff_t elsize, int NumFiles, int NumWriters, MPI_Comm comm);

#endif
