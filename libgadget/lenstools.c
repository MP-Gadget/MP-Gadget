#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <fftw3.h> 
#include <string.h>

#ifdef USE_CFITSIO
#include "fitsio.h"
#endif

#include "lenstools.h"
#include "partmanager.h"
#include "cosmology.h"
#include "physconst.h"
#include "utils.h"

void linspace(double start, double stop, int num, double *result) {
    double step = (stop - start) / (num - 1);
    for (int i = 0; i < num; i++) {
        result[i] = start + i * step;
    }
}

// Function to allocate a 1D array to be used as a 3D array, and initialize elements to zero
double *allocate_3d_array_as_1d(int Nx, int Ny, int Nz) {
    double *array = mymalloc("3d", Nx * Ny * Nz * sizeof(double));
    // Initialize elements with memset
    memset(array, 0, Nx * Ny * Nz * sizeof(double));
    return array;
}

// Function to allocate a 1D array to be used as a 2D array, and initialize elements to zero
double *allocate_2d_array_as_1d(int Nx, int Ny) {
    double *array = mymalloc("2d", Nx * Ny * sizeof(double));
    // Initialize elements with memset
    memset(array, 0, Nx * Ny * sizeof(double));

    return array;
}

typedef struct {
    int nx, ny, nz;
} GridDimensions;

// Function to determine the bin index for a given value
int find_bin(double value, double *bins, int resolution, const double L) { // L is the box size
    // float index
    float iflt = (value - bins[0]) / (bins[resolution] - bins[0]) * resolution;

    // round down to the nearest integer
    int index = (int)floorf(iflt);

    // check if the value is within the range
    if (index >= 0 && index < resolution) {
        return index;
    }
    else {
        return -1;
    }
    endrun(0, "Error in find_bin\n");
}

void grid3d_ngb(const struct particle_data * Parts, int num_particles, double **binning, GridDimensions dims, double *density) { // adpated from grid3d_nfw in lenstools
    
    double position[3];
    // Process each particle
    for (int p = 0; p < num_particles; p++) {

        // remove offset
        for(int d = 0; d < 3; d ++) {
            position[d] = Parts[p].Pos[d] - PartManager->CurrentParticleOffset[d];
            while(position[d] > PartManager->BoxSize) position[d] -= PartManager->BoxSize;
            while(position[d] <= 0) position[d] += PartManager->BoxSize;
        }
        int ix = find_bin(position[0], binning[0], dims.nx, PartManager->BoxSize);
        int iy = find_bin(position[1], binning[1], dims.ny, PartManager->BoxSize);
        int iz = find_bin(position[2], binning[2], dims.nz, PartManager->BoxSize);

        // continue if the particle is outside the grid
        if (ix == -1 || iy == -1 || iz == -1) {
            continue;
        }
        // Increment the density in the appropriate bin
        ACCESS_3D(density, ix, iy, iz, dims.ny, dims.nz)++;
    }

}

void projectDensity(double *density, double *density_projected, GridDimensions dims, int normal) {
    // z; x, y
    // y; z, x
    // x; y, z
    int DimNorm = (normal == 0) ? dims.nx : (normal == 1) ? dims.ny : dims.nz;
    int Dim0 = (normal == 2) ? dims.nx : (normal == 0) ? dims.ny : dims.nx;
    int Dim1 = (normal == 2) ? dims.ny : (normal == 1) ? dims.nz : dims.nz;

    for (int i = 0; i < Dim0; i++) {
        for (int j = 0; j < Dim1; j++) {
            ACCESS_2D(density_projected, i, j, Dim1) = 0;
            for (int k = 0; k < DimNorm; k++) {
                if (normal == 0) {
                    ACCESS_2D(density_projected, i, j, Dim1) += ACCESS_3D(density, k, i, j, dims.ny, dims.nz);
                } else if (normal == 1) {
                    ACCESS_2D(density_projected, i, j, Dim1) += ACCESS_3D(density, i, k, j, dims.ny, dims.nz);
                } else {
                    ACCESS_2D(density_projected, i, j, Dim1) += ACCESS_3D(density, i, j, k, dims.ny, dims.nz);
                }
            }
        }
    }
}

void calculate_lensing_potential(double *density_projected, int plane_resolution, double bin_resolution_0, double bin_resolution_1, double chi,double smooth, double *lensing_potential) {
    // Allocate the complex FFT output array
    fftw_complex *density_ft = fftw_malloc(sizeof(fftw_complex) * plane_resolution * (plane_resolution / 2 + 1));
    // Initialize density_ft to zero
    for (int i = 0; i < plane_resolution * (plane_resolution / 2 + 1); i++) {
        density_ft[i][0] = 0.0;  // Real part
        density_ft[i][1] = 0.0;  // Imaginary part
    }

    double *l_squared = allocate_2d_array_as_1d(plane_resolution, plane_resolution / 2 + 1);

    // Create FFTW plans
    fftw_plan forward_plan = fftw_plan_dft_r2c_2d(plane_resolution, plane_resolution, density_projected, density_ft, FFTW_ESTIMATE);
    fftw_plan backward_plan = fftw_plan_dft_c2r_2d(plane_resolution, plane_resolution, density_ft, lensing_potential, FFTW_ESTIMATE);

    // Compute l_squared (multipoles)
    for (int i = 0; i < plane_resolution; i++) {
        for (int j = 0; j < plane_resolution / 2 + 1; j++) {
            double lx = i < plane_resolution / 2 ? i : -(plane_resolution - i);
            lx /= plane_resolution;
            double ly = j;  // Since rfftn outputs only the non-negative frequencies
            ly /= plane_resolution;
            // l_squared[i][j] = lx * lx + ly * ly;
            ACCESS_2D(l_squared, i, j, plane_resolution / 2 + 1) = lx * lx + ly * ly;
        }
    }
    l_squared[0 + 0 * plane_resolution] = 1.0;  // Avoid division by zero at the DC component

    // Perform the forward FFT
    fftw_execute(forward_plan);

    // Solve the Poisson equation and apply Gaussian smoothing in the frequency domain
    for (int i = 0; i < plane_resolution; i++) {
        for (int j = 0; j < plane_resolution / 2 + 1; j++) {
            int idx = i * (plane_resolution / 2 + 1) + j;
            double factor = -2.0 * (bin_resolution_0 * bin_resolution_1 / (chi * chi)) / (ACCESS_2D(l_squared, i, j, plane_resolution / 2 + 1) * 4 * M_PI * M_PI);
            density_ft[idx][0] *= factor * exp(-0.5 * ((2.0 * M_PI * smooth) * (2.0 * M_PI * smooth)) * ACCESS_2D(l_squared, i, j, plane_resolution / 2 + 1));
            density_ft[idx][1] *= factor * exp(-0.5 * ((2.0 * M_PI * smooth) * (2.0 * M_PI * smooth)) * ACCESS_2D(l_squared, i, j, plane_resolution / 2 + 1));
        }
    }

    // Perform the inverse FFT
    fftw_execute(backward_plan);

    // Normalize the output of the inverse FFT
    for (int i = 0; i < plane_resolution; i++) {
        for (int j = 0; j < plane_resolution; j++) {
            ACCESS_2D(lensing_potential, i, j, plane_resolution) /= (plane_resolution * plane_resolution);
        }
    }
    
    // Cleanup
    fftw_destroy_plan(forward_plan);
    fftw_destroy_plan(backward_plan);
    fftw_free(density_ft);
    // for (int i = 0; i < plane_resolution; i++) {
    //     free(l_squared[i]);
    // }
    myfree(l_squared);
}

int64_t cutPlaneGaussianGrid(int num_particles_tot, double comoving_distance, double Lbox, const Cosmology * CP, const double atime, const int normal, const double center, const double thickness, const double *left_corner, const int plane_resolution, double *lensing_potential) {
    // Get the rank of the current process
    int ThisTask;
    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
    // smooth
    float smooth = 1.0; // fixed in our case

    int num_particles_rank = PartManager->NumPart;  // dark matter-only simulation: NumPart = number of dark matter particles

    double *density_projected = allocate_2d_array_as_1d(plane_resolution, plane_resolution);

    int thickness_resolution = 1;  // Number of bins along the thickness direction, fixed to 1 for now

    // cosmological normalization factor
    double H0 = 100 * CP->HubbleParam * 3.2407793e-20;  // Hubble constant in cgs units
    double cosmo_normalization = 1.5 * pow(H0, 2) * CP->Omega0 / pow(LIGHTCGS, 2);  

    // Binning for directions perpendicular to 'normal'
    double *binning[3];
    int plane_directions[2] = { (normal + 1) % 3, (normal + 2) % 3 };

    for (int i = 0; i < 3; i++) {
        int resolution = (i == normal) ? thickness_resolution : plane_resolution;
        binning[i] = malloc((resolution + 1) * sizeof(double));
        if (binning[i] == NULL) {
            fprintf(stderr, "Memory allocation failed for binning %d on rank %d\n", i, ThisTask);
            // Free other allocated resources
            for (int j = 0; j < i; j++) {
                free(binning[j]);
            }
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        double start = (i == normal) ? (center - thickness / 2) : left_corner[i];
        double stop = (i == normal) ? (center + thickness / 2) : (left_corner[i] + Lbox);
        linspace(start, stop, resolution + 1, binning[i]);
    }


    // bin resolution (cell size in kpc/h)
    double bin_resolution[3];
    bin_resolution[plane_directions[0]] = Lbox / plane_resolution;
    bin_resolution[plane_directions[1]] = Lbox / plane_resolution;
    bin_resolution[normal] = thickness / thickness_resolution;

    // density normalization
    double density_normalization = bin_resolution[normal] * comoving_distance * pow(CM_PER_KPC/CP->HubbleParam, 2) / atime;


    // density 3D array
    GridDimensions dims;
    // Set dimensions based on the orientation specified by 'normal'
    dims.nx = (normal == 0) ? thickness_resolution : plane_resolution;
    dims.ny = (normal == 1) ? thickness_resolution : plane_resolution;
    dims.nz = (normal == 2) ? thickness_resolution : plane_resolution;
    // printf("nx: %d, ny: %d, nz: %d\n", dims.nx, dims.ny, dims.nz);

    double *density = allocate_3d_array_as_1d(dims.nx, dims.ny, dims.nz);

    grid3d_nfw(P, num_particles_rank, binning, dims, density);

    projectDensity(density, density_projected, dims, normal);

    myfree(density);

    //number of particles on the plane
    int64_t num_particles_plane = 0;
    for (int i = 0; i < plane_resolution; i++) {
        for (int j = 0; j < plane_resolution; j++) {
            num_particles_plane += ACCESS_2D(density_projected, i, j, plane_resolution);
        }
    }

    if (num_particles_plane == 0) {
        return 0;
    }
    // normalize the density to the density fluctuation
    double density_norm_factor = 1. / num_particles_tot * (pow(Lbox,3) / (bin_resolution[0] * bin_resolution[1] * bin_resolution[2]));

    for (int i = 0; i < plane_resolution; i++) {
        for (int j = 0; j < plane_resolution; j++) {
            ACCESS_2D(density_projected, i, j, plane_resolution) *= density_norm_factor;
        }
    }

    // Calculate the lensing potential by solving the Poisson equation
    calculate_lensing_potential(density_projected, plane_resolution, bin_resolution[plane_directions[0]], bin_resolution[plane_directions[1]], comoving_distance, smooth, lensing_potential);

    // normalize the lensing potential
    for (int i = 0; i < plane_resolution; i++) {
        for (int j = 0; j < plane_resolution; j++) {
            ACCESS_2D(lensing_potential, i, j, plane_resolution) *= cosmo_normalization * density_normalization;
        }
    }
    myfree(density_projected);
    // Free the binning arrays
    for (int i = 0; i < 3; i++) {
        free(binning[i]);
    }
    return num_particles_plane;
}

#ifdef USE_CFITSIO
void savePotentialPlane(double *data, int rows, int cols, const char filename[128], double Lbox, const Cosmology * CP, double redshift, double comoving_distance, int64_t num_particles, const double UnitLength_in_cm) {
    fitsfile *fptr;       // Pointer to the FITS file; defined in fitsio.h
    int status = 0;       // Status must be initialized to zero.
    long naxes[2] = {cols, rows};  // image dimensions
    char newFilename[128];
    // Format the filename to include '!' to overwrite existing files
    sprintf(newFilename, "!%s", filename);

    // Create the file
    if (fits_create_file(&fptr, newFilename, &status)) {
        printf("Error creating FITS file: %s\n", filename);
        fits_report_error(stderr, status);
        return;
    }
    
    // Create the primary image (double precision)
    if (fits_create_img(fptr, DOUBLE_IMG, 2, naxes, &status)) {
        fits_report_error(stderr, status);
        return;
    }
    double H0 = CP->HubbleParam * 100;
    double Lbox_Mpc = Lbox * UnitLength_in_cm / CM_PER_MPC;  // Box size in Mpc/h
    double comoving_distance_Mpc = comoving_distance * UnitLength_in_cm / CM_PER_MPC;
    double Ode0 = CP->OmegaLambda > 0 ? CP->OmegaLambda : CP->Omega_fld;
    // Insert a blank line as a separator
    fits_write_record(fptr, "        ", &status);
    // Add headers to the FITS file
    fits_update_key(fptr, TDOUBLE, "H0", &H0, "Hubble constant in km/s*Mpc", &status);
    // fits_update_key(fptr, TSTRING, " ", &cosmo.h, "Dimensionless Hubble constant", &status);
    fits_update_key(fptr, TDOUBLE, "h", &CP->HubbleParam, "Dimensionless Hubble constant", &status);
    fits_update_key(fptr, TDOUBLE, "OMEGA_M", &CP->Omega0, "Dark Matter density", &status);
    fits_update_key(fptr, TDOUBLE, "OMEGA_L", &Ode0, "Dark Energy density", &status);
    fits_update_key(fptr, TDOUBLE, "W0", &CP->w0_fld, "Dark Energy equation of state", &status);
    fits_update_key(fptr, TDOUBLE, "WA", &CP->wa_fld, "Dark Energy running equation of state", &status);

    fits_update_key(fptr, TDOUBLE, "Z", &redshift, "Redshift of the lens plane", &status);
    fits_update_key(fptr, TDOUBLE, "CHI", (&comoving_distance_Mpc), "Comoving distance in Mpc/h", &status);
    fits_update_key(fptr, TDOUBLE, "SIDE", &(Lbox_Mpc), "Side length in Mpc/h", &status);
    fits_update_key(fptr, TLONGLONG, "NPART", &num_particles, "Number of particles on the plane", &status);
    fits_update_key(fptr, TSTRING, "UNIT", "rad2    ", "Pixel value unit", &status);

    // Write the 2D array of doubles to the image
    long fpixel[2] = {1, 1};  // first pixel to write (1-based indexing)
    if (fits_write_pix(fptr, TDOUBLE, fpixel, rows * cols, data, &status)) {
        fits_report_error(stderr, status);
        // free(tempData);
        return;
    }

    // free(tempData);


    // Close the FITS file
    if (fits_close_file(fptr, &status)) {
        printf("Error closing FITS file.\n");
        fits_report_error(stderr, status);
    }
}
#endif

