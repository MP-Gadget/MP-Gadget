#ifndef PLANE_H
#define PLANE_H

#include "cosmology.h"
#include "utils/paramset.h"

/* The set_plane_params function initializes the parameters needed for computing lensing potential planes. It sets the normal directions and cut points for the planes, the resolution of the planes, and their thickness. The function reads these parameters from a ParameterSet object, broadcasts the initialized parameters to all processes in an MPI environment, and ensures consistency across different processes. */
void set_plane_params(ParameterSet * ps);

/* The write_plane function computes and writes the lensing potential planes. It reads the simulation parameters and variables, the plane parameters, and the cosmological parameters. It computes the comoving distance, loops over the cut points and normal directions to generate lensing potential planes, and saves the potential plane data. */
void write_plane(int snapnum, const double atime, Cosmology * CP, const char * OutputDir, const double UnitVelocity_in_cm_per_s, const double UnitLength_in_cm);

#endif
