#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
/* do NOT use complex.h it breaks the code */
#include <pfft.h>

#include "petapm.h"
#include "genic/allvars.h"
#include "genic/proto.h"
#include "walltime.h"
#include "endrun.h"
#include "cosmology.h"

#define MESH2K(i) petapm_mesh_to_k(i)
static void density_transfer(int64_t k2, int kpos[3], pfft_complex * value);
static void disp_x_transfer(int64_t k2, int kpos[3], pfft_complex * value);
static void disp_y_transfer(int64_t k2, int kpos[3], pfft_complex * value);
static void disp_z_transfer(int64_t k2, int kpos[3], pfft_complex * value);
static void readout_density(int i, double * mesh, double weight);
static void readout_force_x(int i, double * mesh, double weight);
static void readout_force_y(int i, double * mesh, double weight);
static void readout_force_z(int i, double * mesh, double weight);
static void gaussian_fill(PetaPMRegion * region, pfft_complex * rho_k);
static void setup_grid();

static double Dplus;

void initialize_ffts(void) {
    petapm_init(Box, Nmesh, 1);
    setup_grid();

    Dplus = GrowthFactor(InitTime, 1.0);
}

uint64_t ijk_to_id(int i, int j, int k) {
    uint64_t id = ((uint64_t) i) * Ngrid * Ngrid + ((uint64_t)j) * Ngrid + k + 1;
    return id;
}

void free_ffts(void)
{
}

static void setup_grid() {
    int * ThisTask2d = petapm_get_thistask2d();
    int * NTask2d = petapm_get_ntask2d();
    int size[3];
    int offset[3];
    int k;
    NumPart = 1;
    for(k = 0; k < 2; k ++) {
        offset[k] = (ThisTask2d[k]) * Ngrid / NTask2d[k];
        size[k] = (ThisTask2d[k] + 1) * Ngrid / NTask2d[k];
        size[k] -= offset[k];
        NumPart *= size[k];
    }
    offset[2] = 0;
    size[2] = Ngrid;
    NumPart *= size[2];
    P = (struct part_data *) calloc(NumPart, sizeof(struct part_data));

    int i;
    for(i = 0; i < NumPart; i ++) {
        int x, y, z;
        x = i / (size[2] * size[1]) + offset[0];
        y = (i % (size[1] * size[2])) / size[2] + offset[1];
        z = (i % size[2]) + offset[2];
        P[i].Pos[0] = x * Box / Ngrid;
        P[i].Pos[1] = y * Box / Ngrid;
        P[i].Pos[2] = z * Box / Ngrid;
        P[i].Mass = 1.0;
        P[i].ID = ijk_to_id(x, y, z);
    }
}

static PetaPMRegion * makeregion(void * userdata, int * Nregions) {
    PetaPMRegion * regions = malloc(sizeof(PetaPMRegion));
    int k;
    int r = 0;
    int i;
    double min[3] = {Box, Box, Box};
    double max[3] = {0, 0, 0.};

    for(i = 0; i < NumPart; i ++) {
        for(k = 0; k < 3; k ++) {
            if(min[k] > P[i].Pos[k]) 
                min[k] = P[i].Pos[k];
            if(max[k] < P[i].Pos[k]) 
                max[k] = P[i].Pos[k];
        }
        P[i].RegionInd = 0;
    }

    for(k = 0; k < 3; k ++) {
        regions[r].offset[k] = floor(min[k] / Box * Nmesh - 1);
        regions[r].size[k] = ceil(max[k] / Box * Nmesh + 2);
        regions[r].size[k] -= regions[r].offset[k];
    }

    /* setup the internal data structure of the region */
    petapm_region_init_strides(&regions[r]);
    *Nregions = 1.0;
    return regions;
}

void displacement_fields() {
    PetaPMFunctions functions[] = {
        {"Density", density_transfer, readout_density},
        {"DispX", disp_x_transfer, readout_force_x},
        {"DispY", disp_y_transfer, readout_force_y},
        {"DispZ", disp_z_transfer, readout_force_z},
        {NULL, NULL, NULL },
    };
    PetaPMParticleStruct pstruct = {
        P,   
        sizeof(P[0]),
        ((char*) &P[0].Pos[0]) - (char*) P,
        ((char*) &P[0].Mass) - (char*) P,
        ((char*) &P[0].RegionInd) - (char*) P,
        NumPart,
    };
    petapm_force_init(
           makeregion, 
           &pstruct, NULL);

    gaussian_fill(petapm_get_fourier_region(),
            petapm_get_rho_k());

    petapm_force_c2r(functions);
    petapm_force_finish();
    double maxdisp = 0;
    int i;
    for(i = 0; i < NumPart; i ++) {
        int k;
        for (k = 0; k < 3; k ++) {
            double dis = P[i].Vel[k];
            if(dis > maxdisp) {
                maxdisp = dis;
            }
        }
    }
    double maxdispall;
    MPI_Reduce(&maxdisp, &maxdispall, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    message(0, "max disp = %g in units of cell sep %g \n", maxdispall, maxdispall / (Box / Nmesh) );

    double hubble_a = hubble_function(InitTime);

    double vel_prefac = InitTime * hubble_a * F_Omega(InitTime);

    if(UsePeculiarVelocity) {
        /* already for peculiar velocity */
        message(0, "Producing Peculliar Velocity in the output.\n");
    } else {
        vel_prefac /= sqrt(InitTime);	/* converts to Gadget velocity */
    }
    message(0, "vel_prefac= %g  hubble_a=%g fom=%g \n", vel_prefac, hubble_a, F_Omega(InitTime));

    for(i = 0; i < NumPart; i++)
    {
        int k;
        for(k = 0; k < 3; k++)
        {
            P[i].Pos[k] += P[i].Vel[k];
            P[i].Vel[k] *= vel_prefac;
            P[i].Pos[k] = periodic_wrap(P[i].Pos[k]);
        }
    }
    walltime_measure("/Disp/Finalize");
}

/********************
 * transfer functions for 
 *
 * potential from mass in cell
 *
 * and 
 *
 * force from potential
 *
 *********************/
static void gaussian_fill(PetaPMRegion * region, pfft_complex * rho_k) {
    gsl_rng * random_generator_seed = gsl_rng_alloc(gsl_rng_ranlxd1);
    gsl_rng_set(random_generator_seed, Seed);

    unsigned int * seedtable = malloc(sizeof(unsigned int) * Nmesh * Nmesh);

#define SETSEED(i, j) { \
    unsigned int seed = 0x7fffffff * gsl_rng_uniform(random_generator_seed); \
    seedtable[(i) * Nmesh + (j)] = seed; \
}

    int i;
    for(i = 0; i < Nmesh / 2; i++)
    {
        int j;
        for(j = 0; j < i; j++) SETSEED(i, j)
        for(j = 0; j < i + 1; j++) SETSEED(j, i)
        for(j = 0; j < i; j++) SETSEED(Nmesh - 1 - i, j)
        for(j = 0; j < i + 1; j++) SETSEED(Nmesh - 1 - j, i)
        for(j = 0; j < i; j++) SETSEED(i, Nmesh - 1 - j)
        for(j = 0; j < i + 1; j++) SETSEED(j, Nmesh - 1 - i)
        for(j = 0; j < i; j++) SETSEED(Nmesh - 1 - i, Nmesh - 1 - j)
        for(j = 0; j < i + 1; j++) SETSEED(Nmesh - 1 - j, Nmesh - 1 - i)
    }
    gsl_rng_free(random_generator_seed);

    double fac = pow(1.0 / Box, 1.5);
    for(i = region->offset[2]; i < region->offset[2] + region->size[2]; i ++) {
        gsl_rng * random_generator0 = gsl_rng_alloc(gsl_rng_ranlxd1);
        gsl_rng * random_generator1 = gsl_rng_alloc(gsl_rng_ranlxd1);
        int j;
        for(j = region->offset[0]; j < region->offset[0] + region->size[0]; j ++) {
            /* always pull the gaussian from the lower quadrant plane for k = 0
             * plane*/
            int hermitian = 0;
            if(i == 0) {
                int jj = MESH2K(j);
                if(jj != j) {
                    jj = - jj;
                    hermitian = 1; 
                }
                gsl_rng_set(random_generator0, seedtable[i * Nmesh + jj]);
            } else {
                int ii = MESH2K(i);
                if(i != ii) {
                    ii = - ii;
                    int jj = j!= 0?(Nmesh - j):0;
                    hermitian = 1;
                    gsl_rng_set(random_generator0, seedtable[ii * Nmesh + jj]);
                }  else {
                    gsl_rng_set(random_generator0, seedtable[i * Nmesh + j]);
                }
            } 

            gsl_rng_set(random_generator1, seedtable[i * Nmesh + j]);

            /* two skips to maintains consistency of generators in lower quandrant */
            double skip = gsl_rng_uniform(random_generator1);
            skip = gsl_rng_uniform(random_generator1);
            int k;
            for(k = 0; k <= Nmesh / 2; k ++) {
                /* on k = 0 plane, we use the lower quadrant generator, 
                 * then hermit transform the result if it is nessessary */
                gsl_rng * random_generator = k?random_generator1:random_generator0;
                hermitian *= k==0;

                ptrdiff_t ip = region->strides[0] * (j - region->offset[0]) 
                             + region->strides[1] * (k - region->offset[1]) 
                             + region->strides[2] * (i - region->offset[2]);
                double phase = gsl_rng_uniform(random_generator) * 2 * M_PI;
                double ampl = 0;
                do ampl = gsl_rng_uniform(random_generator); while(ampl == 0);
                if(k < region->offset[1]) continue;
                if(k >= region->offset[1] + region->size[1]) continue;
                int64_t kmag2 = (int64_t)MESH2K(i) * MESH2K(i) + (int64_t)MESH2K(j) * MESH2K(j) + (int64_t)MESH2K(k) * MESH2K(k);
                double kmag = sqrt(kmag2) * 2 * M_PI / Box;
                double p_of_k = - log(ampl);
			    if(SphereMode == 1) {
			      if(kmag2 >= (Nsample/ 2) * (Nsample / 2))	/* select a sphere in k-space */ {
                    p_of_k = 0;
                  }
			    }
                if(i == Nmesh / 2) {
                    p_of_k = 0;
                }
                if(j == Nmesh / 2) {
                    p_of_k = 0;
                }
                if(k >= Nmesh / 2) {
                    /* this is to cut off at the Nyquist*/
                    p_of_k = 0;
                }
                p_of_k *= PowerSpec(kmag);
                double delta = fac * sqrt(p_of_k) / Dplus;
                rho_k[ip][0] = delta * cos(phase);
                rho_k[ip][1] = delta * sin(phase);
                if(hermitian) {
                    rho_k[ip][1] *= -1;
                }
            }
        }
        gsl_rng_free(random_generator0);
        gsl_rng_free(random_generator1);
    }
    free(seedtable);
#if 0
    /* dump the gaussian field for debugging 
     * 
     * the z directioin is in axis 1.
     *
     * */
    FILE * rhokf = fopen("rhok", "w");
    printf("strides %td %td %td\n", 
            region->strides[0],
            region->strides[1],
            region->strides[2]);
    printf("size %td %td %td\n", 
            region->size[0],
            region->size[1],
            region->size[2]);
    printf("offset %td %td %td\n", 
            region->offset[0],
            region->offset[1],
            region->offset[2]);
    fwrite(rho_k, sizeof(double) * fftsize, 1, rhokf);
    fclose(rhokf);
#endif
}

/* unnormalized sinc function sin(x) / x */
static double sinc_unnormed(double x) {
    if(x < 1e-5 && x > -1e-5) {
        double x2 = x * x;
        return 1.0 - x2 / 6. + x2  * x2 / 120.;
    } else {
        return sin(x) / x;
    }
}
static void cic_transfer(int64_t k2, int kpos[3], pfft_complex *value) {
    if(k2 == 0) {
        /* remote zero mode corresponding to the mean */
        value[0][0] = 0.0;
        value[0][1] = 0.0;
        return;
    } 
    double f = 1.0;
    /* the CIC deconvolution kernel is
     *
     * sinc_unnormed(k_x L / 2 Nmesh) ** 2
     *
     * k_x = kpos * 2pi / L
     *
     * */
    int k;
    for(k = 0; k < 3; k ++) {
        double tmp = (kpos[k] * M_PI) / Nmesh;
        tmp = sinc_unnormed(tmp);
        f *= 1. / (tmp * tmp);
    }
    /* 
     * first decovolution is CIC in par->mesh
     * second decovolution is correcting readout 
     * I don't understand the second yet!
     * */
    double fac = f;
    value[0][0] *= fac;
    value[0][1] *= fac;
}

/* the transfer functions for force in fourier space applied to potential */
/* super lanzcos in CH6 P 122 Digital Filters by Richard W. Hamming */
static double super_lanzcos_diff_kernel_3(double w) {
/* order N = 3*/
    return 1. / 594 * 
       (126 * sin(w) + 193 * sin(2 * w) + 142 * sin (3 * w) - 86 * sin(4 * w));
}
static double super_lanzcos_diff_kernel_2(double w) {
/* order N = 2*/
    return 1 / 126. * (58 * sin(w) + 67 * sin (2 * w) - 22 * sin(3 * w));
}
static double super_lanzcos_diff_kernel_1(double w) {
/* order N = 1 */
/* 
 * This is the same as GADGET-2 but in fourier space: 
 * see gadget-2 paper and Hamming's book.
 * c1 = 2 / 3, c2 = 1 / 12
 * */
    return 1 / 6.0 * (8 * sin (w) - sin (2 * w));
}
static void density_transfer(int64_t k2, int kpos[3], pfft_complex * value) {
    if(k2) {
        /* density is smoothed in k space by a gaussian kernel of 1 mesh grid */
        double r2 = 1.0 / Nmesh;
        r2 *= r2;
        double fac = exp(- k2 * r2);
        value[0][0] *= fac;
        value[0][1] *= fac;
    }
}
static void disp_x_transfer(int64_t k2, int kpos[3], pfft_complex * value) {
    if(k2) {
        double fac = (Box / (2 * M_PI)) * kpos[0] / k2;
        /*
         We avoid high precision kernels to maintain compatibility with N-GenIC.
         The following formular shall cross check with fac in the limit of 
         native diff_kernel (disp_y, disp_z shall match too!)
         
        double fac1 = (2 * M_PI) / Box;
        double fac = diff_kernel(kpos[0] * (2 * M_PI / Nmesh)) * (Nmesh / Box) / (
                    k2 * fac1 * fac1);
                    */
        double tmp = value[0][0];
        value[0][0] = - value[0][1] * fac;
        value[0][1] = tmp * fac;
    }
}
static void disp_y_transfer(int64_t k2, int kpos[3], pfft_complex * value) {
    if(k2) {
        double fac = (Box / (2 * M_PI)) * kpos[1] / k2;
        double tmp = value[0][0];
        value[0][0] = - value[0][1] * fac;
        value[0][1] = tmp * fac;
    }
}
static void disp_z_transfer(int64_t k2, int kpos[3], pfft_complex * value) {
    if(k2) {
        double fac = (Box / (2 * M_PI)) * kpos[2] / k2;
        double tmp = value[0][0];
        value[0][0] = - value[0][1] * fac;
        value[0][1] = tmp * fac;
    }
}

/**************
 * functions iterating over particle / mesh pairs
 ***************/
static void readout_density(int i, double * mesh, double weight) {
    P[i].Density += weight * mesh[0];
}
static void readout_force_x(int i, double * mesh, double weight) {
    P[i].Vel[0] += weight * mesh[0];
}
static void readout_force_y(int i, double * mesh, double weight) {
    P[i].Vel[1] += weight * mesh[0];
}
static void readout_force_z(int i, double * mesh, double weight) {
    P[i].Vel[2] += weight * mesh[0];
}
