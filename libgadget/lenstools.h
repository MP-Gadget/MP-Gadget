#ifndef LENSTOOLS_H
#define LENSTOOLS_H

#include <stddef.h>
#include "cosmology.h"

// Macro to access the 3D array elements using the 1D array
#define ACCESS_3D(array, i, j, k, Ny, Nz) ((array)[((i) * (Ny) + (j)) * (Nz) + (k)])

// Macro to access the 2D array elements using the 1D array
#define ACCESS_2D(array, i, j, Ny) ((array)[(i) * (Ny) + (j)])

// Function prototypes

// Simulates cutting a plane with a Gaussian grid
int64_t cutPlaneGaussianGrid(int num_particles_tot, double comoving_distance, double Lbox, const Cosmology * CP, const double atime, const int normal, const double center, const double thickness, const double *left_corner, const int plane_resolution, double *lensing_potential);

// Saves the potential plane data
void savePotentialPlane(double *data, int rows, int cols, const char filename[128], double Lbox, const Cosmology * CP, double redshift, double comoving_distance, int64_t num_particles, const double UnitLength_in_cm);

// Function to allocate a 2D array as a 1D array
double *allocate_2d_array_as_1d(int Nx, int Ny);

// Function to allocate a 3D array as a 1D array
double *allocate_3d_array_as_1d(int Nx, int Ny, int Nz);

#endif // LENSTOOLS_H
