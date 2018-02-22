#include <mpi.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "utils.h"

#include "allvars.h"
#include "partmanager.h"
#include "forcetree.h"
#include "petapm.h"
#include "powerspectrum.h"
#include "domain.h"

#include "cosmology.h"
#include "kspace-neutrinos/delta_pow.h"
#include "kspace-neutrinos/delta_tot_table.h"

/*Global variable to store power spectrum*/
struct _powerspectrum PowerSpectrum;

/* Structure which holds pointers to the stored
 * neutrino power spectrum*/
_delta_pow nu_pow;
/*Structure which holds the neutrino state*/
_delta_tot_table delta_tot_table;

void powerspectrum_nu_save(struct _delta_pow *nu_pow, const char * OutputDir, const double Time);

static int pm_mark_region_for_node(int startno, int rid);
static void convert_node_to_region(PetaPMRegion * r);

static int hybrid_nu_gravpm_is_active(int i);
static void potential_transfer(int64_t k2, int kpos[3], pfft_complex * value);
static void measure_power_spectrum(int64_t k2, int kpos[3], pfft_complex * value);
static void compute_neutrino_power(void);
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

static PetaPMGlobalFunctions global_functions = {NULL, NULL, potential_transfer};

static PetaPMRegion * _prepare(void * userdata, int * Nregions);

void gravpm_init_periodic() {
    petapm_init(All.BoxSize, All.Nmesh, All.NumThreads);
    powerspectrum_alloc(&PowerSpectrum, All.Nmesh, All.NumThreads);
    /*Initialise the kspace neutrino code if it is enabled.
     * Mpc units are used to match power spectrum code.*/
    if(All.MassiveNuLinRespOn) {
        /*Set the private copy of the task in delta_tot_table*/
        delta_tot_table.ThisTask = ThisTask;
        allocate_delta_tot_table(&delta_tot_table, All.Nmesh, All.TimeIC, All.TimeMax, All.CP.Omega0, &All.CP.ONu, All.UnitTime_in_s, 3.085678e24, 0);
        global_functions.global_readout = measure_power_spectrum;
        global_functions.global_analysis = compute_neutrino_power;
    }
}

/* Computes the gravitational force on the PM grid
 * and saves the total matter power spectrum.*/
void gravpm_force(void) {
    PetaPMParticleStruct pstruct = {
        P,
        sizeof(P[0]),
        (char*) &P[0].Pos[0]  - (char*) P,
        (char*) &P[0].Mass  - (char*) P,
        (char*) &P[0].RegionInd - (char*) P,
        (All.HybridNeutrinosOn ? &hybrid_nu_gravpm_is_active : NULL),
        PartManager->NumPart,
    };

    powerspectrum_zero(&PowerSpectrum);
    int i;
    #pragma omp parallel for
    for(i = 0; i < PartManager->NumPart; i++)
    {
        P[i].GravPM[0] = P[i].GravPM[1] = P[i].GravPM[2] = 0;
    }
    /*
     * we apply potential transfer immediately after the R2C transform,
     * Therefore the force transfer functions are based on the potential,
     * not the density.
     * */
    petapm_force(_prepare, &global_functions, functions, &pstruct, NULL);
    powerspectrum_sum(&PowerSpectrum, All.BoxSize*All.UnitLength_in_cm);
    /*Now save the power spectrum*/
    if(ThisTask == 0)
        powerspectrum_save(&PowerSpectrum, All.OutputDir, All.Time, GrowthFactor(All.Time, 1.0));
    if(ThisTask == 0 && All.MassiveNuLinRespOn)
        powerspectrum_nu_save(&nu_pow, All.OutputDir, All.Time);
    walltime_measure("/LongRange");
    /*Rebuild the force tree we freed in _prepare to save memory*/
    force_tree_rebuild();
}

static double pot_factor;

static PetaPMRegion * _prepare(void * userdata, int * Nregions) {
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
    /* In worst case, each topleave becomes a region: thus
     * NTopLeaves is sufficient */
    PetaPMRegion * regions = mymalloc2("Regions", sizeof(PetaPMRegion) * NTopLeaves);

    int r = 0;

    int no = RootNode; /* start with the root */
    while(no >= 0) {

        if(!(Nodes[no].f.DependsOnLocalMass)) {
            /* node doesn't contain particles on this process, do not open */
            no = Nodes[no].u.d.sibling;
            continue;
        }
        if(
            /* node is large */
           (Nodes[no].len <= All.BoxSize / All.Nmesh * 24)
           ||
            /* node is a top leaf */
            ( !Nodes[no].f.InternalTopLevel && (Nodes[no].f.TopLevel) )
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

    int i;
    for(i =0; i < PartManager->NumPart; i ++) {
        P[i].RegionInd = -1;
    }

    /* now lets mark particles to their hosting region */
    int numpart = 0;
#pragma omp parallel for reduction(+: numpart)
    for(r = 0; r < *Nregions; r++) {
        regions[r].numpart = pm_mark_region_for_node(regions[r].no, r);
        numpart += regions[r].numpart;
    }
    for(i =0; i < PartManager->NumPart; i ++) {
        if(P[i].RegionInd == -1) {
            message(1, "i = %d not assigned to a region\n", i);
        }
    }
    /* All particles shall have been processed just once. Otherwise we die */
    if(numpart != PartManager->NumPart) {
        endrun(1, "Processed only %d particles out of %d\n", numpart, PartManager->NumPart);
    }
    for(r =0; r < *Nregions; r++) {
        convert_node_to_region(&regions[r]);
    }
    /*This is done to conserve memory during the PM step*/
    if(force_tree_allocated()) force_tree_free();
    walltime_measure("/PMgrav/Regions");
    return regions;
}

static int pm_mark_region_for_node(int startno, int rid) {
    int numpart = 0;
    int p;
    int no = startno;
    int endno = Nodes[startno].u.d.sibling;
    while(no >= 0 && no != endno)
    {
        if(node_is_particle(no))	/* single particle */
        {
            p = no;
            no = Nextnode[no];
            /* when we are in PM, all particles must have been synced. */
            if (P[p].Ti_drift != All.Ti_Current) {
                abort();
            }

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
                l = fabs(l * 2);
                if (l > Nodes[startno].len) {
                    if(l > Nodes[startno].len * (1+ 1e-7))
                    message(1, "enlarging node size from %g to %g, due to particle of type %d at %g %g %g id=%ld\n",
                        Nodes[startno].len, l, P[p].Type, P[p].Pos[0], P[p].Pos[1], P[p].Pos[2], P[p].ID);
                    Nodes[startno].len = l;
                }
            }
            numpart ++;
        }
        else
        {
            if(node_is_pseudo_particle(no))	/* pseudo particle */
            {
                /* skip pseudo particles */
                no = Nextnode[no - MaxNodes];
                continue;
            }

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
            ThisTask, no, Nodes[no].len, Nodes[no].hmax,
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

/* Compute neutrino power spectrum.
 * This should happen after the CFT is computed,
 * and after powerspectrum_add_mode() has been called,
 * but before potential_transfer is called.*/
static void compute_neutrino_power() {
    if(!All.MassiveNuLinRespOn)
        return;
    /*Note the power spectrum is now in Mpc units*/
    powerspectrum_sum(&PowerSpectrum, All.BoxSize*All.UnitLength_in_cm);
    int i;
    /*Get delta_cdm_curr , which is P(k)^1/2, and skip bins with zero modes:*/
    int nk_nonzero = 0;
    for(i=0;i<PowerSpectrum.size;i++){
        if (PowerSpectrum.Nmodes[i] == 0)
            continue;
        PowerSpectrum.Pnuratio[nk_nonzero] = sqrt(PowerSpectrum.Power[i]);
        PowerSpectrum.kk[nk_nonzero] = PowerSpectrum.kk[i];
        nk_nonzero++;
    }
    double Pnu[nk_nonzero];
    memset(Pnu,0, nk_nonzero*sizeof(double));
    /*This sets up P_nu_curr.*/
    /*This is done on the first timestep: we need nk_nonzero for it to work.*/
    if(!delta_tot_table.delta_tot_init_done) {
        _transfer_init_table transfer_init;
        if(ThisTask == 0) {
            allocate_transfer_init_table(&transfer_init, All.BoxSize, 3.085678e24, All.CAMBInputSpectrum_UnitLength_in_cm, All.CAMBTransferFunction);
        }
        /*Broadcast the transfer size*/
        MPI_Bcast(&(transfer_init.NPowerTable), 1,MPI_INT,0,MPI_COMM_WORLD);
        /*Allocate the memory unless we are on task 0, in which case it is already allocated*/
        if(ThisTask != 0)
          transfer_init.logk = (double *) mymalloc("Transfer_functions", 2*transfer_init.NPowerTable* sizeof(double));
        transfer_init.T_nu=transfer_init.logk+transfer_init.NPowerTable;
        /*Broadcast the transfer table*/
        MPI_Bcast(transfer_init.logk,2*transfer_init.NPowerTable,MPI_DOUBLE,0,MPI_COMM_WORLD);
        /*Initialise delta_tot*/
        delta_tot_init(&delta_tot_table, nk_nonzero, PowerSpectrum.kk, PowerSpectrum.Power, &transfer_init, All.Time);
        free_transfer_init_table(&transfer_init);
    }
    const double partnu = particle_nu_fraction(&All.CP.ONu.hybnu, All.Time, 0);
    double kspace_prefac = 0;
    if(1 - partnu > 1e-3) {
        get_delta_nu_update(&delta_tot_table, All.Time, nk_nonzero, PowerSpectrum.kk, PowerSpectrum.Pnuratio, Pnu, NULL);
        message(0,"Done getting neutrino power: nk= %d, k = %g, delta_nu = %g, delta_cdm = %g,\n",nk_nonzero, PowerSpectrum.kk[1],Pnu[1],PowerSpectrum.Pnuratio[1]);
        /*kspace_prefac = M_nu (analytic) / M_particles */
        const double OmegaNu_nop = get_omega_nu_nopart(&All.CP.ONu, All.Time);
        const double omega_hybrid = get_omega_nu(&All.CP.ONu, 1) * partnu / pow(All.Time, 3);
        /* Omega0 - Omega in neutrinos + Omega in particle neutrinos = Omega in particles*/
        kspace_prefac = OmegaNu_nop/(delta_tot_table.Omeganonu/pow(All.Time,3) + omega_hybrid);
    }
    /*We want to interpolate in log space*/
    for(i=0;i<nk_nonzero;i++){
        PowerSpectrum.logknu[i] = log(PowerSpectrum.kk[i]);
        PowerSpectrum.Pnuratio[i] = Pnu[i]/PowerSpectrum.Pnuratio[i];
    }
    init_delta_pow(&nu_pow, PowerSpectrum.logknu, PowerSpectrum.Pnuratio, nk_nonzero, kspace_prefac);
    /*Zero power spectrum, which is stored with the neutrinos*/
    powerspectrum_zero(&PowerSpectrum);
}

void powerspectrum_nu_save(struct _delta_pow *nu_pow, const char * OutputDir, const double Time)
{
    int i;
    char fname[1024];
    /* Now save the neutrino power spectrum*/
    snprintf(fname, 1024,"%s/powerspectrum-nu-%0.4f.txt", OutputDir, Time);
    FILE * fp = fopen(fname, "w");
    fprintf(fp, "# in Mpc/h Units \n");
    fprintf(fp, "# (k P_nu(k))\n");
    fprintf(fp, "# a= %g\n", Time);
    fprintf(fp, "# nk = %d\n", delta_tot_table.nk);
    for(i = 0; i < delta_tot_table.nk; i++){
        fprintf(fp, "%g %g\n", exp(nu_pow->logkk[i]), pow(delta_tot_table.delta_nu_last[i],2));
    }
    fclose(fp);
    /*Clean up the neutrino memory now we saved the power spectrum.*/
    free_d_pow(nu_pow);
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

    const double asmth2 = pow((2 * M_PI) * All.Asmth / All.Nmesh,2);
    double f = 1.0;
    const double smth = exp(-k2 * asmth2) / k2;
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
    const double fac = pot_factor * smth * f * f;

    /*Add neutrino power if desired*/
    if(All.MassiveNuLinRespOn && k2 > 0) {
        /*Change the units of k to match those of logkk*/
        double logk2 = log(sqrt(k2) * 2 * M_PI / (All.BoxSize * All.UnitLength_in_cm/ 3.085678e24 ));
        /* Note get_neutrino_powerspec returns Omega_nu / (Omega0 -OmegaNu) * delta_nu / P_cdm^1/2, which is dimensionless.
         * So below is: M_cdm * delta_cdm (1 + Omega_nu/(Omega0-OmegaNu) (delta_nu / delta_cdm))
         *            = M_cdm * (delta_cdm (Omega0 - OmegaNu)/Omega0 + Omega_nu/Omega0 delta_nu) * Omega0 / (Omega0-OmegaNu)
         *            = M_cdm * Omega0 / (Omega0-OmegaNu) * (delta_cdm (1 - f_nu)  + f_nu delta_nu) )
         *            = M_cdm * Omega0 / (Omega0-OmegaNu) * delta_t
         *            = (M_cdm + M_nu) * delta_t
         * This is correct for the forces, and gives the right power spectrum,
         * once we multiply PowerSpectrum.Norm by (Omega0 / (Omega0 - OmegaNu))**2 */
        const double nufac = 1 + get_dnudcdm_powerspec(&nu_pow, logk2);
        value[0][0] *= nufac;
        value[0][1] *= nufac;
    }

    /*Compute the power spectrum*/
    powerspectrum_add_mode(k2, kpos, value, f);
    if(k2 == 0) {
        if(All.MassiveNuLinRespOn) {
            const double MtotbyMcdm = All.CP.Omega0/(All.CP.Omega0 - pow(All.Time,3)*get_omega_nu_nopart(&All.CP.ONu, All.Time));
            PowerSpectrum.Norm *= MtotbyMcdm*MtotbyMcdm;
        }
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

/*This function decides if a particle is actively gravitating; tracers are not.*/
static int hybrid_nu_gravpm_is_active(int i) {
    if (particle_nu_fraction(&All.CP.ONu.hybnu, All.Time, 0) == 0. && (P[i].Type == All.FastParticleType))
        return 0;
    else
        return 1;
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
    P[i].Potential += weight * mesh[0];
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

