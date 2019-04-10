#include <mpi.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include <gsl/gsl_rng.h>

#include "allvars.h"
#include "proto.h"

#include <libgadget/petapm.h>
#include <libgadget/walltime.h>
#include <libgadget/utils.h>
#include <libgadget/powerspectrum.h>


struct _powerspectrum PowerSpectrum;
static void measure_power_spectrum(int64_t k2, int kpos[3], pfft_complex * value);
static void potential_transfer(int64_t k2, int kpos[3], pfft_complex * value);
static void force_x_transfer(int64_t k2, int kpos[3], pfft_complex * value);
static void force_y_transfer(int64_t k2, int kpos[3], pfft_complex * value);
static void force_z_transfer(int64_t k2, int kpos[3], pfft_complex * value);
static void readout_force_x(int i, double * mesh, double weight);
static void readout_force_y(int i, double * mesh, double weight);
static void readout_force_z(int i, double * mesh, double weight);
static PetaPMFunctions functions [] =
{
    {"ForceX", force_x_transfer, readout_force_x},
    {"ForceY", force_y_transfer, readout_force_y},
    {"ForceZ", force_z_transfer, readout_force_z},
    {NULL, NULL, NULL},
};

static PetaPMGlobalFunctions global_functions = {measure_power_spectrum, NULL, potential_transfer};

static PetaPMRegion * _prepare(void * userdata, int * Nregions);

static void glass_force(double t_f, struct ic_part_data * ICP, const int NumPart);
static void glass_stats(struct ic_part_data * ICP, int NumPart);

int
setup_glass(double shift, int Ngrid, int seed, int NumPart, struct ic_part_data * ICP)
{
    int size[3];
    int offset[3];
    get_size_offset(size, offset, Ngrid);

    gsl_rng * rng = gsl_rng_alloc(gsl_rng_ranlxd1);
    gsl_rng_set(rng, seed + ThisTask);
    memset(ICP, 0, NumPart*sizeof(struct ic_part_data));

    int i;
    #pragma omp parallel for
    for(i = 0; i < NumPart; i ++) {
        double x, y, z;
        x = i / (size[2] * size[1]) + offset[0];
        y = (i % (size[1] * size[2])) / size[2] + offset[1];
        z = (i % size[2]) + offset[2];
        /* a spread of 3 will kill most of the grid anisotropy structure;
         * and still being local */
        x += 3 * (gsl_rng_uniform(rng) - 0.5);
        y += 3 * (gsl_rng_uniform(rng) - 0.5);
        z += 3 * (gsl_rng_uniform(rng) - 0.5);
        ICP[i].Pos[0] = x * All.BoxSize / Ngrid + shift;
        ICP[i].Pos[1] = y * All.BoxSize / Ngrid + shift;
        ICP[i].Pos[2] = z * All.BoxSize / Ngrid + shift;
        ICP[i].Mass = 1.0;
    }

    gsl_rng_free(rng);

    glass_evolve(seed, ICP, NumPart);
    return NumPart;
}

void glass_evolve(int seed, struct ic_part_data * ICP, const int NumPart)
{
    int i;
    int step = 0;
    double t_x = 0;
    double t_v = 0;
    double t_f = 0;

    /*Allocate memory for a power spectrum*/
    powerspectrum_alloc(&PowerSpectrum, All.Nmesh, All.NumThreads, 0);

    glass_force(t_x, ICP, NumPart);

    /* Our pick of the units ensures there is an oscillation period of 2 * M_PI.
     *
     * (don't ask me how this worked -- I think I failed this problem in physics undergrad. )
     * We use 4 steps per oscillation, and end at
     *
     * 12 + 1 = 13, the first time phase is M_PI / 2, a close encounter to the minimum.
     *
     * */
    for(step = 0; step < 14; step++) {
        /* leap-frog, K D D F K */
        double dt = M_PI / 2; /* step size */
        double hdt = 0.5 * dt; /* half a step */
        int d;
        /*
         * Use inverted gravity with a damping term proportional to the velocity.
         *
         * The magic setup was studied in
         *      https://github.com/rainwoodman/fastpm-python/blob/1be020b/Example/Glass.ipynb
         * */

        /* Kick */
        for(i = 0; i < NumPart; i ++) {
            for(d = 0; d < 3; d ++) {
                /* mind the damping term */
                ICP[i].Vel[d] += (ICP[i].Disp[d] - ICP[i].Vel[d]) * hdt;
            }
        }
        t_x += hdt;

        /* Drift */
        for(i = 0; i < NumPart; i ++) {
            for(d = 0; d < 3; d ++) {
                ICP[i].Pos[d] += ICP[i].Vel[d] * dt;
           }
        }
        t_v += dt;

        glass_force(t_x, ICP, NumPart);
        t_f = t_x;

        /* Kick */
        for(i = 0; i < NumPart; i ++) {
            for(d = 0; d < 3; d ++) {
                /* mind the damping term */
                ICP[i].Vel[d] += (ICP[i].Disp[d] - ICP[i].Vel[d]) * hdt;
            }
        }

        t_x += hdt;
        message(0, "Generating glass, step = %d, t_f= %g, t_v = %g, t_x = %g\n", step, t_f / (2 * M_PI), t_v / (2 *M_PI), t_x / (2 * M_PI));
        glass_stats(ICP, NumPart);

        /*Now save the power spectrum*/
        if(ThisTask == 0) {
            char * fn = fastpm_strdup_printf("powerspectrum-glass-%08X", seed);
            powerspectrum_save(&PowerSpectrum, All.OutputDir, fn, t_f, 1.0);
            free(fn);
        }
    }

    /*We are done with the power spectrum, free it*/
    powerspectrum_free(&PowerSpectrum, 0);
}


static void
glass_stats(struct ic_part_data * ICP, int NumPart) {
    int i;
    double disp2 = 0;
    double vel2 = 0;
    double n = NumPart;
    for(i = 0; i < NumPart; i++)
    {
        int k;
        for(k = 0; k < 3; k++)
        {
            double dis = ICP[i].Disp[k];
            double vel = ICP[i].Vel[k];
            disp2 += dis * dis;
            vel2 += vel * vel;
        }
    }
    MPI_Allreduce(MPI_IN_PLACE, &disp2, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &vel2, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &n, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    message(0, "Force std = %g, vel std = %g\n", sqrt(disp2 / n), sqrt(vel2 / n));
}

struct ic_prep_data
{
    struct ic_part_data * curICP;
    int NumPart;
};
/*Global to pass the particle data to the readout functions*/
static struct ic_part_data * curICP;

/* Computes the gravitational force on the PM grid
 * and saves the total matter power spectrum.*/
static void glass_force(double t_f, struct ic_part_data * ICP, const int NumPart) {

    PetaPMParticleStruct pstruct = {
        ICP,
        sizeof(ICP[0]),
        (char*) &ICP[0].Pos[0]  - (char*) ICP,
        (char*) &ICP[0].Mass  - (char*) ICP,
        (char*) &ICP[0].RegionInd - (char*) ICP,
        NULL,
        NumPart,
    };
    curICP = ICP;

    int i;
    #pragma omp parallel for
    for(i = 0; i < NumPart; i++)
    {
        ICP[i].Disp[0] = ICP[i].Disp[1] = ICP[i].Disp[2] = 0;
    }

    powerspectrum_zero(&PowerSpectrum);

    struct ic_prep_data icprep = {ICP, NumPart};
    /*
     * we apply potential transfer immediately after the R2C transform,
     * Therefore the force transfer functions are based on the potential,
     * not the density.
     * */
    petapm_force(_prepare, &global_functions, functions, &pstruct, &icprep);

    powerspectrum_sum(&PowerSpectrum, All.BoxSize*All.UnitLength_in_cm);
    walltime_measure("/LongRange");
}

static double pot_factor;

static PetaPMRegion * _prepare(void * userdata, int * Nregions)
{
    struct ic_prep_data * icprep = (struct ic_prep_data *) userdata;
    int NumPart = icprep->NumPart;
    struct ic_part_data * ICP = icprep->curICP;
    int64_t ntot = NumPart;

    MPI_Allreduce(MPI_IN_PLACE, &ntot, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);

    double nbar = ntot; /* 1 / pow(All.Nmesh, 3) is included by the FFT, screw it. */

    /* dimensionless invert gravity;
     *
     * pot = nabla ^ -2 delta
     *
     * (2pi / L) ** -2 is for kint -> k.
     * nbar is mean number per cell to get delta */
    pot_factor = -1 * (-1) * pow(2 * M_PI / All.BoxSize, -2) / nbar;

    PetaPMRegion * regions = mymalloc2("Regions", sizeof(PetaPMRegion));
    int k;
    int r = 0;
    int i;
    double min[3] = {All.BoxSize, All.BoxSize, All.BoxSize};
    double max[3] = {0, 0, 0.};

    for(i = 0; i < NumPart; i ++) {
        for(k = 0; k < 3; k ++) {
            if(min[k] > ICP[i].Pos[k])
                min[k] = ICP[i].Pos[k];
            if(max[k] < ICP[i].Pos[k])
                max[k] = ICP[i].Pos[k];
        }
        ICP[i].RegionInd = 0;
    }

    for(k = 0; k < 3; k ++) {
        regions[r].offset[k] = floor(min[k] / All.BoxSize * All.Nmesh - 1);
        regions[r].size[k] = ceil(max[k] / All.BoxSize * All.Nmesh + 2);
        regions[r].size[k] -= regions[r].offset[k];
    }

    /* setup the internal data structure of the region */
    petapm_region_init_strides(&regions[r]);
    *Nregions = 1;

    walltime_measure("/PMgrav/Regions");
    return regions;
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

/* unnormalized sinc function sin(x) / x */
static double sinc_unnormed(double x) {
    if(x < 1e-5 && x > -1e-5) {
        double x2 = x * x;
        return 1.0 - x2 / 6. + x2  * x2 / 120.;
    } else {
        return sin(x) / x;
    }
}

/* Compute the power spectrum of the fourier transformed grid in value.
 * Store it in the PowerSpectrum structure */
void powerspectrum_add_mode(const int64_t k2, const int kpos[3], pfft_complex * const value, const double invwindow) {

    if(k2 == 0) {
        /* Save zero mode corresponding to the mean as the normalisation factor.*/
        PowerSpectrum.Norm = (value[0][0] * value[0][0] + value[0][1] * value[0][1]);
        return;
    }
    /* Measure power spectrum: we don't want the zero mode.
     * Some modes with k_z = 0 or N/2 have weight 1, the rest have weight 2.
     * This is because of the symmetry of the real fft. */
    if(k2 > 0) {
        /*How many bins per unit (log) interval in k?*/
        const double binsperunit=(PowerSpectrum.size-1)/log(sqrt(3)*All.Nmesh/2.0);
        int kint=floor(binsperunit*log(k2)/2.);
        int w;
        const double keff = sqrt(kpos[0]*kpos[0]+kpos[1]*kpos[1]+kpos[2]*kpos[2]);
        const double m = (value[0][0] * value[0][0] + value[0][1] * value[0][1]);
        /*Make sure we do not overflow (although this should never happen)*/
        if(kint >= PowerSpectrum.size)
            return;
        if(kpos[2] == 0 || kpos[2] == All.Nmesh/2) w = 1;
        else w = 2;
        /*Make sure we use thread-local memory to avoid racing.*/
        const int index = kint + omp_get_thread_num() * PowerSpectrum.size;
        /*Multiply P(k) by inverse window function*/
        PowerSpectrum.Power[index] += w * m * invwindow * invwindow;
        PowerSpectrum.Nmodes[index] += w;
        PowerSpectrum.kk[index] += w * keff;
    }

}

/*Just read the power spectrum, without changing the input value.*/
static void measure_power_spectrum(int64_t k2, int kpos[3], pfft_complex *value) {
    double f = 1.0;
    /* the CIC deconvolution kernel is
     *
     * sinc_unnormed(k_x L / 2 All.Nmesh) ** 2
     *
     * k_x = kpos * 2pi / L
     *
     * */
    int k;
    for(k = 0; k < 3; k ++) {
        double tmp = (kpos[k] * M_PI) / All.Nmesh;
        tmp = sinc_unnormed(tmp);
        f *= 1. / (tmp * tmp);
    }
    powerspectrum_add_mode(k2, kpos, value, f);
}

static void potential_transfer(int64_t k2, int kpos[3], pfft_complex *value) {

    double f = 1.0;
    const double smth = 1.0 / k2;
    /* the CIC deconvolution kernel is
     *
     * sinc_unnormed(k_x L / 2 All.Nmesh) ** 2
     *
     * k_x = kpos * 2pi / L
     *
     * */
    /*
     * first decovolution is CIC in par->mesh
     * second decovolution is correcting readout
     * I don't understand the second yet!
     * */
    const double fac = pot_factor * smth * f * f;

    /*Compute the power spectrum*/
    powerspectrum_add_mode(k2, kpos, value, f);

    if(k2 == 0) {
        /* Remove zero mode corresponding to the mean.*/
        value[0][0] = 0.0;
        value[0][1] = 0.0;
        return;
    }

    value[0][0] *= fac;
    value[0][1] *= fac;
}

/* the transfer functions for force in fourier space applied to potential */
/* super lanzcos in CH6 P 122 Digital Filters by Richard W. Hamming */
static double diff_kernel(double w) {
/* order N = 1 */
/*
 * This is the same as GADGET-2 but in fourier space:
 * see gadget-2 paper and Hamming's book.
 * c1 = 2 / 3, c2 = 1 / 12
 * */
    return 1 / 6.0 * (8 * sin (w) - sin (2 * w));
}

static void force_transfer(int k, pfft_complex * value) {
    double tmp0;
    double tmp1;
    /*
     * negative sign is from force_x = - Del_x pot
     *
     * filter is   i K(w)
     * */
    double fac = -1 * diff_kernel (k * (2 * M_PI / All.Nmesh)) * (All.Nmesh / All.BoxSize);
    tmp0 = - value[0][1] * fac;
    tmp1 = value[0][0] * fac;
    value[0][0] = tmp0;
    value[0][1] = tmp1;
}
static void force_x_transfer(int64_t k2, int kpos[3], pfft_complex * value) {
    force_transfer(kpos[0], value);
}
static void force_y_transfer(int64_t k2, int kpos[3], pfft_complex * value) {
    force_transfer(kpos[1], value);
}
static void force_z_transfer(int64_t k2, int kpos[3], pfft_complex * value) {
    force_transfer(kpos[2], value);
}
static void readout_force_x(int i, double * mesh, double weight) {
    curICP[i].Disp[0] += weight * mesh[0];
}
static void readout_force_y(int i, double * mesh, double weight) {
    curICP[i].Disp[1] += weight * mesh[0];
}
static void readout_force_z(int i, double * mesh, double weight) {
    curICP[i].Disp[2] += weight * mesh[0];
}
