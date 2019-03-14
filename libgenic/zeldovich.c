#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
/* do NOT use complex.h it breaks the code */
#include <pfft.h>
#include <gsl/gsl_rng.h>
#include "allvars.h"
#include "proto.h"
#include "power.h"

#include <libgadget/petapm.h>
#include <libgadget/walltime.h>
#include <libgadget/utils.h>

#define MESH2K(i) petapm_mesh_to_k(i)
static void density_transfer(int64_t k2, int kpos[3], pfft_complex * value);
static void vel_x_transfer(int64_t k2, int kpos[3], pfft_complex * value);
static void vel_y_transfer(int64_t k2, int kpos[3], pfft_complex * value);
static void vel_z_transfer(int64_t k2, int kpos[3], pfft_complex * value);
static void disp_x_transfer(int64_t k2, int kpos[3], pfft_complex * value);
static void disp_y_transfer(int64_t k2, int kpos[3], pfft_complex * value);
static void disp_z_transfer(int64_t k2, int kpos[3], pfft_complex * value);
static void readout_density(int i, double * mesh, double weight);
static void readout_vel_x(int i, double * mesh, double weight);
static void readout_vel_y(int i, double * mesh, double weight);
static void readout_vel_z(int i, double * mesh, double weight);
static void readout_disp_x(int i, double * mesh, double weight);
static void readout_disp_y(int i, double * mesh, double weight);
static void readout_disp_z(int i, double * mesh, double weight);
static void gaussian_fill(PetaPMRegion * region, pfft_complex * rho_k, int UnitaryAmplitude, int InvertPhase);

static inline double periodic_wrap(double x)
{
  while(x >= All.BoxSize)
    x -= All.BoxSize;

  while(x < 0)
    x += All.BoxSize;

  return x;
}

uint64_t
ijk_to_id(int i, int j, int k, int Ngrid) {
    uint64_t id = ((uint64_t) i) * Ngrid * Ngrid + ((uint64_t)j) * Ngrid + k + 1;
    return id;
}

void free_ffts(void)
{
    myfree(ICP);
}

/*Helper function to get size and offset of particles to the global grid.*/
static int
get_size_offset(int * size, int * offset, int Ngrid)
{
    int * ThisTask2d = petapm_get_thistask2d();
    int * NTask2d = petapm_get_ntask2d();
    int k;
    int npart = 1;
    for(k = 0; k < 2; k ++) {
        offset[k] = (ThisTask2d[k]) * Ngrid / NTask2d[k];
        size[k] = (ThisTask2d[k] + 1) * Ngrid / NTask2d[k];
        size[k] -= offset[k];
        npart *= size[k];
    }
    offset[2] = 0;
    size[2] = Ngrid;
    npart *= size[2];
    return npart;
}

uint64_t
id_offset_from_index(const int i, const int Ngrid)
{
    int size[3];
    int offset[3];
    get_size_offset(size, offset, Ngrid);
    int x = i / (size[2] * size[1]) + offset[0];
    int y = (i % (size[1] * size[2])) / size[2] + offset[1];
    int z = (i % size[2]) + offset[2];
    return ijk_to_id(x, y, z, Ngrid);
}

void
setup_grid(double shift, int Ngrid)
{
    int size[3];
    int offset[3];
    NumPart = get_size_offset(size, offset, Ngrid);
    ICP = (struct ic_part_data *) mymalloc("PartTable", NumPart*sizeof(struct ic_part_data));
    memset(ICP, 0, NumPart*sizeof(struct ic_part_data));

    int i;
    #pragma omp parallel for
    for(i = 0; i < NumPart; i ++) {
        int x, y, z;
        x = i / (size[2] * size[1]) + offset[0];
        y = (i % (size[1] * size[2])) / size[2] + offset[1];
        z = (i % size[2]) + offset[2];
        ICP[i].Pos[0] = x * All.BoxSize / Ngrid + shift;
        ICP[i].Pos[1] = y * All.BoxSize / Ngrid + shift;
        ICP[i].Pos[2] = z * All.BoxSize / Ngrid + shift;
        ICP[i].Mass = 1.0;
    }
}

static PetaPMRegion * makeregion(void * userdata, int * Nregions) {
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
    return regions;
}

/*Global to pass type to *_transfer functions*/
static enum TransferType ptype;

void displacement_fields(enum TransferType Type) {
    /*MUST set this before doing force.*/
    ptype = Type;
    PetaPMParticleStruct pstruct = {
        ICP,
        sizeof(ICP[0]),
        ((char*) &ICP[0].Pos[0]) - (char*) ICP,
        ((char*) &ICP[0].Mass) - (char*) ICP,
        ((char*) &ICP[0].RegionInd) - (char*) ICP,
        NULL,
        NumPart,
    };

    int i;
    #pragma omp parallel for
    for(i = 0; i < NumPart; i++)
    {
        memset(&ICP[i].Disp[0], 0, sizeof(ICP[i].Disp));
        memset(&ICP[i].Vel[0], 0, sizeof(ICP[i].Vel));
        ICP[i].Density = 0;
    }


    /* This reads out the displacements into P.Disp and the velocities into P.Vel.
     * Disp is used to avoid changing the particle positions mid-way through.
     * Note that for the velocities we do NOT just use the velocity transfer functions.
     * The reason is because of the gauge: velocity transfer is in synchronous gauge for CLASS,
     * newtonian gauge for CAMB. But we want N-body gauge, which we get by taking the time derivative
     * of the synchronous gauge density perturbations. See arxiv:1505.04756*/
    PetaPMFunctions functions[] = {
        {"Density", density_transfer, readout_density},
        {"DispX", disp_x_transfer, readout_disp_x},
        {"DispY", disp_y_transfer, readout_disp_y},
        {"DispZ", disp_z_transfer, readout_disp_z},
        {"VelX", vel_x_transfer, readout_vel_x},
        {"VelY", vel_y_transfer, readout_vel_y},
        {"VelZ", vel_z_transfer, readout_vel_z},
        {NULL, NULL, NULL },
    };

    /*Set up the velocity pre-factors*/
    const double hubble_a = hubble_function(All.TimeIC);

    double vel_prefac = All.TimeIC * hubble_a;

    if(All.IO.UsePeculiarVelocity) {
        /* already for peculiar velocity */
        message(0, "Producing Peculiar Velocity in the output.\n");
    } else {
        vel_prefac /= sqrt(All.TimeIC);	/* converts to Gadget velocity */
    }

    if(!All2.PowerP.ScaleDepVelocity) {
        vel_prefac *= F_Omega(All.TimeIC);
        /* If different transfer functions are disabled, we can copy displacements to velocities
         * and we don't need the extra transfers.*/
        functions[4].name = NULL;
    }

    PetaPMRegion * regions = petapm_force_init(
           makeregion,
           &pstruct, NULL);

    /*This allocates the memory*/
    pfft_complex * rho_k = petapm_alloc_rhok();

    gaussian_fill(petapm_get_fourier_region(),
		  rho_k, All2.UnitaryAmplitude, All2.InvertPhase);

    petapm_force_c2r(rho_k, regions, functions);

    myfree(rho_k);
    myfree(regions);
    petapm_force_finish();

    double maxdisp = 0, maxvel = 0;

    #pragma omp parallel for reduction(max:maxdisp, maxvel)
    for(i = 0; i < NumPart; i++)
    {
        int k;
        double absv = 0;
        for(k = 0; k < 3; k++)
        {
            double dis = ICP[i].Disp[k];
            if(dis > maxdisp)
                maxdisp = dis;
            /*Copy displacements to positions.*/
            ICP[i].Pos[k] += ICP[i].Disp[k];
            /*Copy displacements to velocities if not done already*/
            if(!All2.PowerP.ScaleDepVelocity)
                ICP[i].Vel[k] = ICP[i].Disp[k];
            ICP[i].Vel[k] *= vel_prefac;
            absv += ICP[i].Vel[k] * ICP[i].Vel[k];
            ICP[i].Pos[k] = periodic_wrap(ICP[i].Pos[k]);
        }
        if(absv > maxvel)
            maxvel = absv;
    }
    MPI_Allreduce(MPI_IN_PLACE, &maxdisp, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    message(0, "Type = %d max disp = %g in units of cell sep %g \n", ptype, maxdisp, maxdisp / (All.BoxSize / All.Nmesh) );

    MPI_Allreduce(MPI_IN_PLACE, &maxvel, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    message(0, "Max vel=%g km/s, vel_prefac= %g  hubble_a=%g fom=%g \n", sqrt(maxvel), vel_prefac, hubble_a, F_Omega(All.TimeIC));

    walltime_measure("/Disp/Finalize");
    MPI_Barrier(MPI_COMM_WORLD);
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

static void density_transfer(int64_t k2, int kpos[3], pfft_complex * value) {
    if(k2) {
        /* density is smoothed in k space by a gaussian kernel of 1 mesh grid */
        double r2 = 1.0 / All.Nmesh;
        r2 *= r2;
        double fac = exp(- k2 * r2);

        double kmag = sqrt(k2) * 2 * M_PI / All.BoxSize;
        fac *= DeltaSpec(kmag, ptype) / sqrt(All.BoxSize * All.BoxSize * All.BoxSize);

        value[0][0] *= fac;
        value[0][1] *= fac;
    }
}

static void disp_transfer(int64_t k2, int kaxis, pfft_complex * value, int include_growth) {
    if(k2) {
        double fac = 1./ (2 * M_PI) / sqrt(All.BoxSize) * kaxis / k2;
        /*
         We avoid high precision kernels to maintain compatibility with N-GenIC.
         The following formular shall cross check with fac in the limit of
         native diff_kernel (disp_y, disp_z shall match too!)

        double fac1 = (2 * M_PI) / All.BoxSize;
        double fac = diff_kernel(kaxis * (2 * M_PI / All.Nmesh)) * (All.Nmesh / All.BoxSize) / (
                    k2 * fac1 * fac1);
                    */
        double kmag = sqrt(k2) * 2 * M_PI / All.BoxSize;
        /*Multiply by derivative of scale-dependent growth function*/
        if(include_growth)
            fac *= dlogGrowth(kmag, ptype);
        else
            fac *= DeltaSpec(kmag, ptype);
        double tmp = value[0][0];
        value[0][0] = - value[0][1] * fac;
        value[0][1] = tmp * fac;
    }
}

static void vel_x_transfer(int64_t k2, int kpos[3], pfft_complex * value) {
    disp_transfer(k2, kpos[0], value, 1);
}
static void vel_y_transfer(int64_t k2, int kpos[3], pfft_complex * value) {
    disp_transfer(k2, kpos[1], value, 1);
}
static void vel_z_transfer(int64_t k2, int kpos[3], pfft_complex * value) {
    disp_transfer(k2, kpos[2], value, 1);
}

static void disp_x_transfer(int64_t k2, int kpos[3], pfft_complex * value) {
    disp_transfer(k2, kpos[0], value, 0);
}
static void disp_y_transfer(int64_t k2, int kpos[3], pfft_complex * value) {
    disp_transfer(k2, kpos[1], value, 0);
}
static void disp_z_transfer(int64_t k2, int kpos[3], pfft_complex * value) {
    disp_transfer(k2, kpos[2], value, 0);
}

/**************
 * functions iterating over particle / mesh pairs
 ***************/
static void readout_density(int i, double * mesh, double weight) {
    ICP[i].Density += weight * mesh[0];
}
static void readout_vel_x(int i, double * mesh, double weight) {
    ICP[i].Vel[0] += weight * mesh[0];
}
static void readout_vel_y(int i, double * mesh, double weight) {
    ICP[i].Vel[1] += weight * mesh[0];
}
static void readout_vel_z(int i, double * mesh, double weight) {
    ICP[i].Vel[2] += weight * mesh[0];
}

static void readout_disp_x(int i, double * mesh, double weight) {
    ICP[i].Disp[0] += weight * mesh[0];
}
static void readout_disp_y(int i, double * mesh, double weight) {
    ICP[i].Disp[1] += weight * mesh[0];
}
static void readout_disp_z(int i, double * mesh, double weight) {
    ICP[i].Disp[2] += weight * mesh[0];
}


/*
 * The following functions are from fastpm/libfastpm/initialcondition.c.
 * Agrees with nbodykit's pmesh/whitenoise.c, which agrees with n-genic.
 * */
typedef struct {
    struct {
        ptrdiff_t start[3];
        ptrdiff_t size[3];
        ptrdiff_t strides[3];
        ptrdiff_t total;
    } ORegion;
    int Nmesh[3];
} PM;

static inline void
SETSEED(PM * pm, unsigned int * table[2][2], int i, int j, gsl_rng * rng)
{
    unsigned int seed = 0x7fffffff * gsl_rng_uniform(rng);

    int ii[2] = {i, (pm->Nmesh[0] - i) % pm->Nmesh[0]};
    int jj[2] = {j, (pm->Nmesh[1] - j) % pm->Nmesh[1]};
    int d1, d2;
    for(d1 = 0; d1 < 2; d1++) {
        ii[d1] -= pm->ORegion.start[0];
        jj[d1] -= pm->ORegion.start[1];
    }
    for(d1 = 0; d1 < 2; d1++)
    for(d2 = 0; d2 < 2; d2++) {
        if( ii[d1] >= 0 &&
            ii[d1] < pm->ORegion.size[0] &&
            jj[d2] >= 0 &&
            jj[d2] < pm->ORegion.size[1]
        ) {
            table[d1][d2][ii[d1] * pm->ORegion.size[1] + jj[d2]] = seed;
        }
    }
}
static inline unsigned int
GETSEED(PM * pm, unsigned int * table[2][2], int i, int j, int d1, int d2)
{
    i -= pm->ORegion.start[0];
    j -= pm->ORegion.start[1];
    if(i < 0) abort();
    if(j < 0) abort();
    if(i >= pm->ORegion.size[0]) abort();
    if(j >= pm->ORegion.size[1]) abort();
    return table[d1][d2][i * pm->ORegion.size[1] + j];
}

static void
SAMPLE(gsl_rng * rng, double * ampl, double * phase)
{
    *phase = gsl_rng_uniform(rng) * 2 * M_PI;
    *ampl = 0;
    do *ampl = gsl_rng_uniform(rng); while(*ampl == 0);
}

static void
pmic_fill_gaussian_gadget(PM * pm, double * delta_k, int seed, int setUnitaryAmplitude, int setInvertPhase)
{
    /* Fill delta_k with gadget scheme */
    int d;
    int i, j, k;

    gsl_rng * rng = gsl_rng_alloc(gsl_rng_ranlxd1);
    gsl_rng_set(rng, seed);

    unsigned int * seedtable[2][2];
    for(i = 0; i < 2; i ++)
    for(j = 0; j < 2; j ++) {
            seedtable[i][j] = calloc(pm->ORegion.size[0] * pm->ORegion.size[1], sizeof(int));
    }

    for(i = 0; i < pm->Nmesh[0] / 2; i++) {
        for(j = 0; j < i; j++) SETSEED(pm, seedtable, i, j, rng);
        for(j = 0; j < i + 1; j++) SETSEED(pm, seedtable, j, i, rng);
        for(j = 0; j < i; j++) SETSEED(pm, seedtable, pm->Nmesh[0] - 1 - i, j, rng);
        for(j = 0; j < i + 1; j++) SETSEED(pm, seedtable, pm->Nmesh[1] - 1 - j, i, rng);
        for(j = 0; j < i; j++) SETSEED(pm, seedtable, i, pm->Nmesh[1] - 1 - j, rng);
        for(j = 0; j < i + 1; j++) SETSEED(pm, seedtable, j, pm->Nmesh[0] - 1 - i, rng);
        for(j = 0; j < i; j++) SETSEED(pm, seedtable, pm->Nmesh[0] - 1 - i, pm->Nmesh[1] - 1 - j, rng);
        for(j = 0; j < i + 1; j++) SETSEED(pm, seedtable, pm->Nmesh[1] - 1 - j, pm->Nmesh[0] - 1 - i, rng);
    }
    gsl_rng_free(rng);

    ptrdiff_t irel[3];
    for(i = pm->ORegion.start[0];
        i < pm->ORegion.start[0] + pm->ORegion.size[0];
        i ++) {

        gsl_rng * lower_rng = gsl_rng_alloc(gsl_rng_ranlxd1);
        gsl_rng * this_rng = gsl_rng_alloc(gsl_rng_ranlxd1);

        int ci = pm->Nmesh[0] - i;
        if(ci >= pm->Nmesh[0]) ci -= pm->Nmesh[0];

        for(j = pm->ORegion.start[1];
            j < pm->ORegion.start[1] + pm->ORegion.size[1];
            j ++) {
            /* always pull the gaussian from the lower quadrant plane for k = 0
             * plane*/
            /* always pull the whitenoise from the lower quadrant plane for k = 0
             * plane and k == All.Nmesh / 2 plane*/
            int d1 = 0, d2 = 0;
            int cj = pm->Nmesh[1] - j;
            if(cj >= pm->Nmesh[1]) cj -= pm->Nmesh[1];

            /* d1, d2 points to the conjugate quandrant */
            if( (ci == i && cj < j)
             || (ci < i && cj != j)
             || (ci < i && cj == j)) {
                d1 = 1;
                d2 = 1;
            }

            unsigned int seed_conj, seed_this;
            /* the lower quadrant generator */
            seed_conj = GETSEED(pm, seedtable, i, j, d1, d2);
            gsl_rng_set(lower_rng, seed_conj);

            seed_this = GETSEED(pm, seedtable, i, j, 0, 0);
            gsl_rng_set(this_rng, seed_this);

            for(k = 0; k <= pm->Nmesh[2] / 2; k ++) {
                int use_conj = (d1 != 0 || d2 != 0) && (k == 0 || k == pm->Nmesh[2] / 2);

                double ampl, phase;
                if(use_conj) {
                    /* on k = 0 and All.Nmesh/2 plane, we use the lower quadrant generator,
                     * then hermit transform the result if it is nessessary */
                    SAMPLE(this_rng, &ampl, &phase);
                    SAMPLE(lower_rng, &ampl, &phase);
                } else {
                    SAMPLE(lower_rng, &ampl, &phase);
                    SAMPLE(this_rng, &ampl, &phase);
                }

                ptrdiff_t iabs[3] = {i, j, k};
                ptrdiff_t ip = 0;
                for(d = 0; d < 3; d ++) {
                    irel[d] = iabs[d] - pm->ORegion.start[d];
                    ip += pm->ORegion.strides[d] * irel[d];
                }

                if(irel[2] < 0) continue;
                if(irel[2] >= pm->ORegion.size[2]) continue;

                /* we want two numbers that are of std ~ 1/sqrt(2) */
                ampl = sqrt(- log(ampl));

                if (setUnitaryAmplitude) ampl = 1.0; /* cos and sin gives 1/sqrt(2)*/


                if (setInvertPhase){
                  phase += M_PI; /*invert phase*/
                }

                (delta_k + 2 * ip)[0] = ampl * cos(phase);
                (delta_k + 2 * ip)[1] = ampl * sin(phase);

                if(use_conj) {
                    (delta_k + 2 * ip)[1] *= -1;
                }

                if((pm->Nmesh[0] - iabs[0]) % pm->Nmesh[0] == iabs[0] &&
                   (pm->Nmesh[1] - iabs[1]) % pm->Nmesh[1] == iabs[1] &&
                   (pm->Nmesh[2] - iabs[2]) % pm->Nmesh[2] == iabs[2]) {
                    /* The mode is self conjuguate, thus imaginary mode must be zero */
                    (delta_k + 2 * ip)[1] = 0;
                    (delta_k + 2 * ip)[0] = ampl * cos(phase);
                }

                if(iabs[0] == 0 && iabs[1] == 0 && iabs[2] == 0) {
                    /* the mean is zero */
                    (delta_k + 2 * ip)[0] = 0;
                    (delta_k + 2 * ip)[1] = 0;
                }
            }
        }
        gsl_rng_free(lower_rng);
        gsl_rng_free(this_rng);
    }
    for(i = 0; i < 2; i ++)
    for(j = 0; j < 2; j ++) {
        free(seedtable[i][j]);
    }
/*
    char * fn[1000];
    sprintf(fn, "canvas.dump.f4.%d", pm->ThisTask);
    fwrite(pm->canvas, sizeof(pm->canvas[0]), pm->ORegion.total * 2, fopen(fn, "w"));
*/
}

/* Using fastpm's gaussian_fill for ngenic agreement. */
static void
gaussian_fill(PetaPMRegion * region, pfft_complex * rho_k, int setUnitaryAmplitude, int setInvertPhase)
{
    /* fastpm deals with strides properly; petapm not. So we translate it here. */
    PM pm[1];
    int d;
    for (d = 0; d < 3; d ++) {
        pm->Nmesh[d] = All.Nmesh;
    }

    pm->ORegion.start[0] = region->offset[2];
    pm->ORegion.size[0] = region->size[2];
    pm->ORegion.strides[0] = region->strides[2];
    pm->ORegion.start[1] = region->offset[0];
    pm->ORegion.size[1] = region->size[0];
    pm->ORegion.strides[1] = region->strides[0];
    pm->ORegion.start[2] = region->offset[1];
    pm->ORegion.size[2] = region->size[1];
    pm->ORegion.strides[2] = region->strides[1];

    pm->ORegion.total = region->totalsize;
    pmic_fill_gaussian_gadget(pm, (double*) rho_k, All2.Seed, setUnitaryAmplitude, setInvertPhase);

#if 0
    /* dump the gaussian field for debugging
     *
     * the z directioin is in axis 1.
     *
     * */
    FILE * rhokf = fopen("rhok", "w");
    printf("strides %td %td %td\n",
            pm->ORegion.strides[0],
            pm->ORegion.strides[1],
            pm->ORegion.strides[2]);
    printf("size %td %td %td\n",
            pm->ORegion.size[0],
            pm->ORegion.size[1],
            pm->ORegion.size[2]);
    printf("offset %td %td %td\n",
            pm->ORegion.start[0],
            pm->ORegion.start[1],
            pm->ORegion.start[2]);
    fwrite(rho_k, sizeof(double) * region->totalsize, 1, rhokf);
    fclose(rhokf);
#endif
}
