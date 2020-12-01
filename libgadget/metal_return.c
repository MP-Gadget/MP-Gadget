#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_interp2d.h>

#include "physconst.h"
#include "walltime.h"
#include "slotsmanager.h"
#include "treewalk.h"
#include "metal_return.h"
#include "densitykernel.h"
#include "density.h"
#include "cosmology.h"
#include "winds.h"
#include "utils/spinlocks.h"
#include "metal_tables.h"

/*! \file metal_return.c
 *  \brief Compute the mass return rate of metals from stellar evolution.
 *
 *  This file returns metals from stars with some delay.
 *  Delayed sources followed are AGB stars, SNII and Sn1a.
 *  Mass from each type of star is stored as a separate value.
 *  Since the species-specific yields do not affect anything
 *  (the cooling function depends only on total metallicity),
 *  actual species-specific yields are *not* specified and can
 *  be given in post-processing.
 */

static struct metal_return_params
{
    double Sn1aN0;
    double tau8msun;
} MetalParams;

/*Set the parameters of the hydro module*/
void
set_metal_return_params(ParameterSet * ps)
{
    int ThisTask;
    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
    if(ThisTask == 0) {
        MetalParams.Sn1aN0 = param_get_double(ps, "MetalsSn1aN0");
        MetalParams.tau8msun = param_get_double(ps, "MetalsTau8msun");
    }
    MPI_Bcast(&MetalParams, sizeof(struct metal_return_params), MPI_BYTE, 0, MPI_COMM_WORLD);
}

struct MetalReturnPriv {
    double atime;
    inttime_t Ti_Current;
    Cosmology *CP;
    MyFloat * StarVolumeSPH;
    gsl_interp2d * lifetime_interp;
    struct SpinLocks * spin;
//    struct Yields
};

#define METALS_GET_PRIV(tw) ((struct MetalReturnPriv*) ((tw)->priv))

typedef struct {
    TreeWalkQueryBase base;
    MyFloat Metallicity[NMETALS];
    MyFloat Mass;
    MyFloat Hsml;
    MyFloat StarVolumeSPH;
    /* This is the metal generated this timestep.
     * Unused metals are added to the star.*/
    MyFloat TotalMetalGenerated[NMETALS];
} TreeWalkQueryMetals;

typedef struct {
    TreeWalkResultBase base;
    /* This is the total mass returned to
     * the surrounding gas particles, for mass conservation.*/
    MyFloat MassReturn[NMETALS];
    /* This is the metal generated this timestep.
     * Unused metals are added to the star.*/
    MyFloat TotalMetalGenerated[NMETALS];
    int Ninteractions;
} TreeWalkResultMetals;

typedef struct {
    TreeWalkNgbIterBase base;
} TreeWalkNgbIterMetals;

static int
metal_return_haswork(int n, TreeWalk * tw);

static void
metal_return_ngbiter(
    TreeWalkQueryMetals * I,
    TreeWalkResultMetals * O,
    TreeWalkNgbIterMetals * iter,
    LocalTreeWalk * lv
   );

static void
metal_return_copy(int place, TreeWalkQueryMetals * input, TreeWalk * tw);

static void
metal_return_reduce(int place, TreeWalkResultMetals * result, enum TreeWalkReduceMode mode, TreeWalk * tw);

void setup_metal_table_interp(gsl_interp2d * lifetime_interp)
{
    lifetime_interp = gsl_interp2d_alloc(gsl_interp2d_bilinear, LIFE_NMET, LIFE_NMASS);
    gsl_interp2d_init(lifetime_interp, lifetime_metallicity, lifetime_masses, lifetime, LIFE_NMET, LIFE_NMASS);
}


/* The Chabrier IMF used for computing SnII and AGB yields.
 * See 1305.2913 eq 3*/
static double chabrier_imf(double mass)
{
    if(mass <= 1) {
        return 0.852464 / mass * exp(- pow(log(mass / 0.079)/ 0.69, 2)/2);
    }
    else {
        return 0.237912 * pow(mass, -2.3);
    }
}

/* Compute the difference in internal time units between two scale factors.
 * These two scale factors should be close together so the Hubble function is constant.*/
static double atime_to_myr(Cosmology *CP, double atime1, double atime2)
{
    /* t = dt/da da = 1/(Ha) da*/
    /* Approximate hubble function as constant here: we only care
     * about metal return over a single timestep*/
    return (atime1 - atime2) / (hubble_function(CP, atime1) * atime1) / SEC_PER_MEGAYEAR;
}

/* Find the mass bins which die in this timestep using the lifetime table.
 * dtstart, dtend - time at start and end of timestep in Myr.
 * stellarmetal - metallicity of the star.
 * lifetime_tables - 2D interpolation table of the lifetime.
 * masshigh, masslow - pointers in which to store the high and low lifetime limits
 */
static void find_mass_bin_limits(double * masslow, double * masshigh, const double dtstart, const double dtend, double stellarmetal, gsl_interp2d * lifetime_tables)
{
    gsl_interp_accel *metalacc = gsl_interp_accel_alloc();
    gsl_interp_accel *massacc = gsl_interp_accel_alloc();
    if(stellarmetal < lifetime_metallicity[0])
        stellarmetal = lifetime_metallicity[0];
    if(stellarmetal > lifetime_metallicity[LIFE_NMET-1])
        stellarmetal = lifetime_metallicity[LIFE_NMET-1];

    double mass = lifetime_masses[LIFE_NMASS-1];

    /* Simple linear search to find the root. We can do better than this. See:
         gsl_root_fsolver_falsepos. */
    while(mass >= lifetime_masses[0]) {
        double tlife = gsl_interp2d_eval(lifetime_tables, lifetime_metallicity, lifetime_masses, lifetime, stellarmetal, mass, metalacc, massacc);
        if(tlife/1e6 > dtstart) {
            break;
        }
        mass /= 1.1;
    }
    /* Largest mass which dies in this timestep*/
    *masshigh = mass;
    while(mass >= lifetime_masses[0]) {
        double tlife = gsl_interp2d_eval(lifetime_tables, lifetime_metallicity, lifetime_masses, lifetime, stellarmetal, mass, metalacc, massacc);
        if(tlife/1e6 > dtend) {
            break;
        }
        mass /= 1.1;
    }
    /* Smallest mass which dies in this timestep*/
    *masslow = mass;
}

#define GSL_WORKSPACE 1000

double chabrier_mass (double mass, void * params)
{
    return mass * chabrier_imf(mass);
}

/* Compute the total metal yield for this star in this timestep*/
static void metal_yield(double dtmyrstart, double dtmyrend, double stellarmetal, gsl_interp2d * lifetime_interp, MyFloat * MetalGenerated)
{
    /* Number of Sn1a events follows a delay time distribution (1305.2913, eq. 10) */
    const double sn1aindex = 1.12;
    const double tau8msun = 40;
    /* Number of Sn1a events from this star*/
    double Sn1aDTD = MetalParams.Sn1aN0 * pow(dtmyrend / tau8msun, -sn1aindex) * (sn1aindex-1)/tau8msun;
    MetalGenerated[SN1a] = Sn1aDTD * (dtmyrend - dtmyrstart);

    double masshigh, masslow;
    find_mass_bin_limits(&masslow, &masshigh, dtmyrstart, dtmyrend, stellarmetal, lifetime_interp);
    /* Number of AGB stars/SnII by integrating the IMF*/
    //gsl_integration chabrier_imf()
    gsl_integration_romberg_workspace * gsl_work = gsl_integration_romberg_alloc(GSL_WORKSPACE);
    gsl_function ff = {chabrier_mass, NULL};

    double metalyield;
    size_t neval;
    gsl_integration_romberg(&ff, masslow, masshigh, 1e-4, 1e-3, &metalyield, &neval, gsl_work);
}

/*! This function is the driver routine for the calculation of metal return. */
void
metal_return(const ActiveParticles * act, const ForceTree * const tree, const double atime, double * StarVolumeSPH)
{
    TreeWalk tw[1] = {{0}};

    struct MetalReturnPriv priv[1];

    tw->ev_label = "METALS";
    tw->visit = (TreeWalkVisitFunction) treewalk_visit_ngbiter;
    tw->ngbiter = (TreeWalkNgbIterFunction) metal_return_ngbiter;
    tw->ngbiter_type_elsize = sizeof(TreeWalkNgbIterMetals);
    tw->haswork = metal_return_haswork;
    tw->fill = (TreeWalkFillQueryFunction) metal_return_copy;
    tw->reduce = (TreeWalkReduceResultFunction) metal_return_reduce;
    tw->postprocess = NULL;
    tw->query_type_elsize = sizeof(TreeWalkQueryMetals);
    tw->result_type_elsize = sizeof(TreeWalkResultMetals);
    tw->tree = tree;
    tw->priv = priv;

    if(!tree->hmax_computed_flag)
        endrun(5, "Metal called before hmax computed\n");
    /* Initialize some time factors*/
    METALS_GET_PRIV(tw)->atime = atime;
    METALS_GET_PRIV(tw)->StarVolumeSPH = StarVolumeSPH;
    setup_metal_table_interp(METALS_GET_PRIV(tw)->lifetime_interp);

    priv->spin = init_spinlocks(SlotsManager->info[0].size);
    treewalk_run(tw, act->ActiveParticle, act->NumActiveParticle);
    free_spinlocks(priv->spin);

    /* collect some timing information */
    walltime_measure("/SPH/Metals");
}

static void
metal_return_copy(int place, TreeWalkQueryMetals * input, TreeWalk * tw)
{
    int j;
    for(j = 0; j< NMETALS; j++)
        input->Metallicity[j] = STARP(place).Metallicity[j];
    input->Mass = P[place].Mass;
    input->Hsml = P[place].Hsml;
    input->StarVolumeSPH = METALS_GET_PRIV(tw)->StarVolumeSPH[place];
    double endtime = METALS_GET_PRIV(tw)->atime + get_dloga_for_bin(P[place].TimeBin, METALS_GET_PRIV(tw)->Ti_Current);
    double dtmyrend = atime_to_myr(METALS_GET_PRIV(tw)->CP, STARP(place).FormationTime, endtime);
    double dtmyrstart = atime_to_myr(METALS_GET_PRIV(tw)->CP, STARP(place).FormationTime, METALS_GET_PRIV(tw)->atime);
    /* Do TotalMetalGenerated by computing the yield at this time.*/
    metal_yield(dtmyrstart, dtmyrend, input->Metallicity[Total], METALS_GET_PRIV(tw)->lifetime_interp, input->TotalMetalGenerated);
}

static void
metal_return_reduce(int place, TreeWalkResultMetals * result, enum TreeWalkReduceMode mode, TreeWalk * tw)
{
    int j;
    /* Conserve mass returned*/
    P[place].Mass -= result->MassReturn[Total];
    /* TODO: What to do about the enrichment of the star particle?*/
    for(j = 0; j < NMETALS; j++)
        STARP(place).Metallicity[j] += result->MassReturn[j] / P[place].Mass;
}

/*! This function is the 'core' of the SPH force computation. A target
 *  particle is specified which may either be local, or reside in the
 *  communication buffer.
 */
static void
metal_return_ngbiter(
    TreeWalkQueryMetals * I,
    TreeWalkResultMetals * O,
    TreeWalkNgbIterMetals * iter,
    LocalTreeWalk * lv
   )
{
    if(iter->base.other == -1) {
        /* Only return metals to gas*/
        iter->base.mask = 1;
        iter->base.Hsml = I->Hsml;
        iter->base.symmetric = NGB_TREEFIND_ASYMMETRIC;
        /* Initialise the mass lost by this star in this timestep*/
        int j;
        for(j = 0; j < NMETALS; j++) {
            O->MassReturn[j] = 0;
        }
        return;
    }

    int other = iter->base.other;
    double r2 = iter->base.r2;

    if(P[other].Mass == 0) {
        endrun(12, "Encountered zero mass particle during hydro;"
                  " We haven't implemented tracer particles and this shall not happen\n");
    }

    /* Wind particles do not interact hydrodynamically: don't receive metal mass.*/
    if(winds_is_particle_decoupled(other))
        return;

    DensityKernel kernel_j;

    density_kernel_init(&kernel_j, P[other].Hsml, GetDensityKernelType());

    if(r2 > 0 && r2 < kernel_j.HH)
    {
        double wk = density_kernel_wk(&kernel_j, iter->base.r);
        int i;
        /* Volume of particle weighted by the SPH kernel*/
        double volume = P[other].Mass / SPHP(other).Density;
        double ThisMetals[NMETALS];
        for(i = 0; i < NMETALS; i++)
            ThisMetals[i] = wk * volume * I->TotalMetalGenerated[i] / I->StarVolumeSPH;
        for(i = 0; i < NMETALS; i++)
            O->MassReturn[i] += ThisMetals[i];
        lock_spinlock(other, METALS_GET_PRIV(lv->tw)->spin);
        /* Add the metals to the particle.*/
        for(i = 0; i < NMETALS; i++)
            SPHP(other).Metallicity[i] += ThisMetals[i];
        /* Add metals weighted by SPH kernel*/
        unlock_spinlock(other, METALS_GET_PRIV(lv->tw)->spin);
    }
    O->Ninteractions++;
}

/* Only stars return metals to the gas*/
static int
metal_return_haswork(int i, TreeWalk * tw)
{
    return P[i].Type == 4;
}
