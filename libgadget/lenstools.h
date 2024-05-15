#ifndef LENSTOOLS_H
#define LENSTOOLS_H

#include <stddef.h>
#include "cosmology.h"

// Function prototypes

// Simulates cutting a plane with a Gaussian grid
int64_t cutPlaneGaussianGrid(int num_particles_tot, double comoving_distance, double Lbox, const Cosmology * CP, const double atime, const int normal, const double center, const double thickness, const double *left_corner, const int plane_resolution, double **lensing_potential);

// Saves the potential plane data
void savePotentialPlane(double **data, int rows, int cols, const char filename[128], double Lbox, const Cosmology * CP, double redshift, double comoving_distance, int64_t num_particles);

// Function to allocate a 2D array
double **allocate_2d_array(int Nx, int Ny);

// Function to free a 2D array
void free_2d_array(double **array, int Nx);

#endif // LENSTOOLS_H
