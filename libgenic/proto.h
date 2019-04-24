#ifndef GENIC_PROTO_H
#define GENIC_PROTO_H
#include <bigfile.h>
#include <stdint.h>
#include "power.h"

/* Compute the displacement and velocity from the initial homogeneous particle distribution,
 * using the cosmological transfer functions. */
void displacement_fields(enum TransferType Type, struct ic_part_data * dispICP, const int NumPart);

/* Fill ICP with NumPart particles spaced on a regular 3D grid. */
int setup_grid(double shift, int Ngrid, double mass, int NumPart, struct ic_part_data * ICP);

/* Fill ICP with NumPart particles spaced out as a Lagrangian glass, calling glass_evolve
 * to move the particles with reversed gravity. */
int setup_glass(double shift, int Ngrid, int seed, double mass, int NumPart, struct ic_part_data * ICP);

/* Evolve a distribution of particles with a reversed gravitational force. */
void glass_evolve(int nsteps, char * pkoutname, struct ic_part_data * ICP, const int NumPart);

/* Returns number of local particles and computes the local offsets in each dimension
 * for Ngrid^3 particles*/
int get_size_offset(int * size, int * offset, int Ngrid);

/* Works out the id of a particle from its index and the processor number*/
uint64_t id_offset_from_index(const int i, const int Ngrid);

/* Save the header of the ICs. */
void saveheader(BigFile * bf, int64_t TotNumPartCDM, int64_t TotNumPartGas, int64_t TotNuPart, double nufrac);

/*Compute the mass array from the cosmology*/
void compute_mass(double * mass, int64_t TotNumPartCDM, int64_t TotNumPartGas, int64_t TotNuPart, double nufrac);

/* Save positions, velocities and IDs of a particle type to the ICs. */
void write_particle_data(const int Type, BigFile * bf, const uint64_t FirstID, const int Ngrid, struct ic_part_data * curICP, const int NumPart);

/*Read a parameter file*/
void  read_parameterfile(char *fname);
#endif
