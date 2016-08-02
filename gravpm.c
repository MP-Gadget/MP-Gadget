#ifndef PETAPM_ORDER
#define PETAPM_ORDER 1
#warning Using low resolution force differentiation kernel. Consider using -DPETAPM_ORDER=3
#endif
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "allvars.h"
#include "proto.h"
#include "forcetree.h"
#include "petapm.h"
#include "domain.h"
#include "endrun.h"

static struct {
    double * k;
    double * P;
    double * Nmodes;
    size_t size;
    double Norm;
} PowerSpectrum;

static int pm_mark_region_for_node(int startno, int rid);
static void convert_node_to_region(PetaPMRegion * r);

static void potential_transfer(int64_t k2, int kpos[3], pfft_complex * value);
static void force_x_transfer(int64_t k2, int kpos[3], pfft_complex * value);
static void force_y_transfer(int64_t k2, int kpos[3], pfft_complex * value);
static void force_z_transfer(int64_t k2, int kpos[3], pfft_complex * value);
static void readout_potential(int i, double * mesh, double weight);
static void readout_force_x(int i, double * mesh, double weight);
static void readout_force_y(int i, double * mesh, double weight);
static void readout_force_z(int i, double * mesh, double weight);
static PetaPMFunctions functions [] =
{
    {"Potential", NULL, readout_potential},
    {"ForceX", force_x_transfer, readout_force_x},
    {"ForceY", force_y_transfer, readout_force_y},
    {"ForceZ", force_z_transfer, readout_force_z},
    {NULL, NULL, NULL},
};

static PetaPMRegion * _prepare(void * userdata, int * Nregions);

void gravpm_init_periodic() {
    petapm_init(All.BoxSize, All.Nmesh, All.NumThreads, All.MeasureFFT);
    PowerSpectrum.size = All.Nmesh / 2;
    PowerSpectrum.k = malloc(sizeof(double) * All.Nmesh / 2);
    PowerSpectrum.P = malloc(sizeof(double) * All.Nmesh / 2);
    PowerSpectrum.Nmodes = malloc(sizeof(double) * All.Nmesh / 2);
}
void gravpm_force() {
    PetaPMParticleStruct pstruct = {
        P,
        sizeof(P[0]),
        (char*) &P[0].Pos[0]  - (char*) P,
        (char*) &P[0].Mass  - (char*) P,
        (char*) &P[0].RegionInd - (char*) P,
        NumPart,
    };

    memset(PowerSpectrum.k, 0, sizeof(double) * PowerSpectrum.size);
    memset(PowerSpectrum.P, 0, sizeof(double) * PowerSpectrum.size);
    memset(PowerSpectrum.Nmodes, 0, sizeof(double) * PowerSpectrum.size);
    PowerSpectrum.Norm = 0;
    /* 
     * we apply potential transfer immediately after the R2C transform,
     * Therefore the force transfer functions are based on the potential,
     * not the density.
     * */
    petapm_force(_prepare, potential_transfer, functions, &pstruct, NULL);

    MPI_Allreduce(MPI_IN_PLACE, &PowerSpectrum.Norm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, PowerSpectrum.k, PowerSpectrum.size, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, PowerSpectrum.P, PowerSpectrum.size, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, PowerSpectrum.Nmodes, PowerSpectrum.size, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    int i;
    for(i = 0; i < PowerSpectrum.size; i ++) {
        if(PowerSpectrum.Nmodes[i] == 0) continue;
        PowerSpectrum.P[i] /= PowerSpectrum.Nmodes[i];
        PowerSpectrum.P[i] /= PowerSpectrum.Norm;
        PowerSpectrum.k[i] /= PowerSpectrum.Nmodes[i];
        /* Mpc/h units */
        PowerSpectrum.k[i] *= 2 * M_PI / (All.BoxSize / 1000);
        PowerSpectrum.P[i] *= pow(All.BoxSize / 1000., 3.0);
 
    }
    if(ThisTask == 0) {
        char fname[1024];
        sprintf(fname, "%s/powerspectrum-%0.4f.txt", All.OutputDir, All.Time);
        message(1, "Writing Power Spectrum to %s\n", fname);
        FILE * fp = fopen(fname, "w");
        fprintf(fp, "# in Mpc/h Units \n");
        fprintf(fp, "# D1 = %g \n", All.cf.D1);
        fprintf(fp, "# k P N P(z=0)\n");
        for(i = 0; i < PowerSpectrum.size; i ++) {
            fprintf(fp, "%g %g %g %g\n", PowerSpectrum.k[i], PowerSpectrum.P[i], PowerSpectrum.Nmodes[i],
                        PowerSpectrum.P[i] / (All.cf.D1 * All.cf.D1));
        }
        fclose(fp);
    }
}

static double pot_factor;

static PetaPMRegion * _prepare(void * userdata, int * Nregions) {
    All.Asmth[0] = ASMTH * All.BoxSize / All.Nmesh;
    All.Rcut[0] = RCUT * All.Asmth[0];
    /* fac is - 4pi G     (L / 2pi) **2 / L ** 3 
     *        Gravity       k2            DFT (dk **3, but )
     * */

    pot_factor = - All.G / (M_PI * All.BoxSize);	/* to get potential */


    /* 
     *
     * walks down the tree, identify nodes that contains local mass and
     * are sufficiently large in volume.
     *
     * for each nodes, a mesh region is created.
     * the particles in a node are linked to their hosting region 
     * (each particle belongs
     * to exactly one region even though it may be covered by two) 
     *
     * */
    int no;
    /* In worst case, each topleave becomes a region: thus
     * NTopleaves is sufficient */
    PetaPMRegion * regions = malloc(sizeof(PetaPMRegion) * NTopleaves);

    int r = 0;

    no = All.MaxPart; /* start with the root */
    while(no >= 0) {
        force_drift_node(no, All.Ti_Current);

        if(!(Nodes[no].u.d.bitflags & (1 << BITFLAG_DEPENDS_ON_LOCAL_MASS))) {
            /* node doesn't contain particles on this process, do not open */
            no = Nodes[no].u.d.sibling;
            continue;
        }
        if(
            /* node is large */
           (Nodes[no].len <= All.BoxSize / All.Nmesh * 24)
           ||  
            /* node is a top leaf */
            (
            !(Nodes[no].u.d.bitflags & (1 << BITFLAG_INTERNAL_TOPLEVEL))
            && (Nodes[no].u.d.bitflags & (1 << BITFLAG_TOPLEVEL)))
                ) {
            regions[r].no = no;
            r ++;
            /* do not open */
            no = Nodes[no].u.d.sibling;
            continue;
        } 
        /* open */
        no = Nodes[no].u.d.nextnode;
    }

    *Nregions = r;
    int maxNregions;
    MPI_Reduce(&r, &maxNregions, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    message(0, "max number of regions is %d\n", maxNregions);

    /* now lets mark particles to their hosting region */
    int numpart = 0;
#pragma omp parallel for reduction(+: numpart)
    for(r = 0; r < *Nregions; r++) {
        regions[r].numpart = pm_mark_region_for_node(regions[r].no, r);
        numpart += regions[r].numpart;
    }
    /* All particles shall have been processed just once. Otherwise we die */
    if(numpart != NumPart) {
        endrun(1, "numpart = %d\n", numpart);
    }
    for(r =0; r < *Nregions; r++) {
        convert_node_to_region(&regions[r]);
    }
    force_treefree();
    walltime_measure("/PMgrav/Regions");
    return regions;
}

static int pm_mark_region_for_node(int startno, int rid) {
    int numpart = 0;
    int p;
    int endno = Nodes[startno].u.d.sibling;
    int no = Nodes[startno].u.d.nextnode;
    while(no >= 0)
    {
        if(no < All.MaxPart)	/* single particle */
        {
            p = no;
            no = Nextnode[no];
            drift_particle(p, All.Ti_Current);
            P[p].RegionInd = rid;
            /* 
             *
             * Enlarge the startno so that it encloses all particles 
             * this happens if a BH particle is relocated to a PotMin
             * out-side the (enlarged )drifted node.
             * because the POTMIN relocation is unphysical, this can
             * happen immediately after a BH is seeded at the dense-most
             * gas particle. rare rare event!
             *
             * */
            int k;
            for(k = 0; k < 3; k ++) {
                double l = P[p].Pos[k] - Nodes[startno].center[k];
                if (l < - 0.5 * All.BoxSize) {
                    l += All.BoxSize;
                }
                if (l > 0.5 * All.BoxSize) {
                    l -= All.BoxSize;
                }
                if (l < 0) {
                    l = - l;
                }
                l = l * 2;
                if (l > Nodes[startno].len) {
                    message(1, "enlarging node size from %g to %g, dueto particle of type %d at %g %g %g id=%ld\n",
                        Nodes[startno].len, l, P[p].Type, P[p].Pos[0], P[p].Pos[1], P[p].Pos[2], P[p].ID);
                    Nodes[startno].len = l;
                }
            }
            numpart ++;
        }
        else
        {
            if(no >= All.MaxPart + MaxNodes)	/* pseudo particle */
            {
                /* skip pseudo particles */
                no = Nextnode[no - MaxNodes];
                continue;
            }

            if(no == endno)
                /* we arrived to the sibling which means that we are done with the node */
            {
                break;
            }
            force_drift_node(no, All.Ti_Current);

            no = Nodes[no].u.d.nextnode;	/* ok, we need to open the node */
        }
    }
    return numpart;
}


static void convert_node_to_region(PetaPMRegion * r) {
    int k;
    double cellsize = All.BoxSize / All.Nmesh;
    int no = r->no;
#if 0
    printf("task = %d no = %d len = %g hmax = %g center = %g %g %g\n",
            ThisTask, no, Nodes[no].len, Extnodes[no].hmax, 
            Nodes[no].center[0],
            Nodes[no].center[1],
            Nodes[no].center[2]);
#endif
    for(k = 0; k < 3; k ++) {
        r->offset[k] = floor((Nodes[no].center[k] - Nodes[no].len * 0.5) / cellsize);
        int end = (int) ceil((Nodes[no].center[k] + Nodes[no].len * 0.5) / cellsize) + 1;
        r->size[k] = end - r->offset[k] + 1;
        r->center[k] = Nodes[no].center[k];
    }

    /* setup the internal data structure of the region */
    petapm_region_init_strides(r);

    r->len  = Nodes[no].len;
    r->hmax = Extnodes[no].hmax;
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

static void potential_transfer(int64_t k2, int kpos[3], pfft_complex *value) {
    double asmth2 = (2 * M_PI) * All.Asmth[0] / All.BoxSize;
    asmth2 *= asmth2;
    double f = 1.0;
    double smth = exp(-k2 * asmth2) / k2;
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
    /* 
     * first decovolution is CIC in par->mesh
     * second decovolution is correcting readout 
     * I don't understand the second yet!
     * */
    double fac = pot_factor * smth * f * f;

    /* Measure power spectrum */
    int kint = floor(sqrt(k2));
    if(kint >= 0 && kint < PowerSpectrum.size) {
        int w;
        double m = (value[0][0] * value[0][0] + value[0][1] * value[0][1]);
        if(kpos[2] == 0) w = 1;
        else w = 2;
        if(k2 == 0) PowerSpectrum.Norm += m;
        PowerSpectrum.P[kint] += w * m;
        PowerSpectrum.Nmodes[kint] += w;
        PowerSpectrum.k[kint] += w * kint;
    }

    value[0][0] *= fac;
    value[0][1] *= fac;

    if(k2 == 0) {
        /* remove zero mode corresponding to the mean */
        value[0][0] = 0.0;
        value[0][1] = 0.0;
        return;
    }
}

/* the transfer functions for force in fourier space applied to potential */
/* super lanzcos in CH6 P 122 Digital Filters by Richard W. Hamming */
#if PETAPM_ORDER == 3
static double super_lanzcos_diff_kernel_3(double w) {
/* order N = 3*/
    return 1. / 594 *
       (126 * sin(w) + 193 * sin(2 * w) + 142 * sin (3 * w) - 86 * sin(4 * w));
}
#endif
#if PETAPM_ORDER == 2
static double super_lanzcos_diff_kernel_2(double w) {
/* order N = 2*/
    return 1 / 126. * (58 * sin(w) + 67 * sin (2 * w) - 22 * sin(3 * w));
}
#endif
#if PETAPM_ORDER == 1
static double super_lanzcos_diff_kernel_1(double w) {
/* order N = 1 */
/* 
 * This is the same as GADGET-2 but in fourier space: 
 * see gadget-2 paper and Hamming's book.
 * c1 = 2 / 3, c2 = 1 / 12
 * */
    return 1 / 6.0 * (8 * sin (w) - sin (2 * w));
}
#endif
static double diff_kernel(double w) {
#if PETAPM_ORDER == 1
        return super_lanzcos_diff_kernel_1(w);
#endif
#if PETAPM_ORDER == 2
        return super_lanzcos_diff_kernel_2(w);
#endif
#if PETAPM_ORDER == 3
        return super_lanzcos_diff_kernel_3(w);
#endif
#if PETAPM_ORDER > 3 
#error PETAPM_ORDER too high.
#endif
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
static void readout_potential(int i, double * mesh, double weight) {
    P[i].PM_Potential += weight * mesh[0];
}
static void readout_force_x(int i, double * mesh, double weight) {
    P[i].GravPM[0] += weight * mesh[0];
}
static void readout_force_y(int i, double * mesh, double weight) {
    P[i].GravPM[1] += weight * mesh[0];
}
static void readout_force_z(int i, double * mesh, double weight) {
    P[i].GravPM[2] += weight * mesh[0];
}

