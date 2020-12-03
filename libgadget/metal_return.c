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

#if NMETALS != NSPECIES
    #pragma error " Inconsistency in metal number between slots and metals"
#endif

/* Largest mass in the SnII tables*/
#define MAXMASS 40
/* Only used for IMF normalisation*/
#define MINMASS 0.1

static struct metal_return_params
{
    double Sn1aN0;
} MetalParams;

/*Set the parameters of the hydro module*/
void
set_metal_return_params(ParameterSet * ps)
{
    int ThisTask;
    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
    if(ThisTask == 0) {
        MetalParams.Sn1aN0 = param_get_double(ps, "MetalsSn1aN0");
    }
    MPI_Bcast(&MetalParams, sizeof(struct metal_return_params), MPI_BYTE, 0, MPI_COMM_WORLD);
}

struct MetalReturnPriv {
    double atime;
    inttime_t Ti_Current;
    MyFloat * StellarAges;
    MyFloat * MassReturn;
    double imf_norm;
    Cosmology *CP;
    MyFloat * StarVolumeSPH;
    gsl_interp2d * lifetime_interp;
    struct SpinLocks * spin;
//    struct Yields
};

#define METALS_GET_PRIV(tw) ((struct MetalReturnPriv*) ((tw)->priv))

typedef struct {
    TreeWalkQueryBase base;
    MyFloat Metallicity;
    MyFloat Mass;
    MyFloat Hsml;
    MyFloat StarVolumeSPH;
    /* This is the metal/mass generated this timestep.*/
    MyFloat TotalMetalGenerated[NMETALS];
    MyFloat MassGenerated;
    MyFloat MetalGenerated;
} TreeWalkQueryMetals;

typedef struct {
    TreeWalkResultBase base;
    /* This is the total mass returned to
     * the surrounding gas particles, for mass conservation.*/
    MyFloat MassReturn;
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
metal_return_postprocess(int place, TreeWalk * tw);

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

/* Compute factor to normalise the total mass in the IMF to unity.*/
double compute_imf_norm(void)
{
    double norm;
    size_t neval;
    gsl_integration_romberg_workspace * gsl_work = gsl_integration_romberg_alloc(GSL_WORKSPACE);
    gsl_function ff = {chabrier_mass, NULL};
    gsl_integration_romberg(&ff, MINMASS, MAXMASS, 1e-4, 1e-3, &norm, &neval, gsl_work);
    return norm;
}

/* Compute number of Sn1a */
static double sn1a_number(double dtmyrstart, double dtmyrend)
{
    /* Number of Sn1a events follows a delay time distribution (1305.2913, eq. 10) */
    const double sn1aindex = 1.12;
    const double tau8msun = 40;
    /* Total number of Sn1a events from this star: integral evaluated from t=0 to t=hubble time. TODO: Fix factor of h*/
    const double totalSN1a = 1- pow(1/(0.7*HUBBLE*SEC_PER_MEGAYEAR)/tau8msun, 1-sn1aindex);
    /* This is the integral of the DTD, normalised to the N0 rate.*/
    double Nsn1a = MetalParams.Sn1aN0 /totalSN1a * (pow(dtmyrstart / tau8msun, 1-sn1aindex) - pow(dtmyrend / tau8msun, 1-sn1aindex));
    return Nsn1a;
}

/* Compute the total mass yield for this star in this timestep*/
static double mass_yield(double dtmyrstart, double dtmyrend, double stellarmetal, gsl_interp2d * lifetime_interp, double imf_norm)
{
    double masshigh, masslow;
    find_mass_bin_limits(&masslow, &masshigh, dtmyrstart, dtmyrend, stellarmetal, lifetime_interp);
    /* Number of AGB stars/SnII by integrating the IMF*/
    //gsl_integration chabrier_imf()
    gsl_integration_romberg_workspace * gsl_work = gsl_integration_romberg_alloc(GSL_WORKSPACE);
    gsl_function ff = {chabrier_mass, NULL};
    double massyield;
    size_t neval;
    gsl_integration_romberg(&ff, masslow, masshigh, 1e-2, 1e-2, &massyield, &neval, gsl_work);
    /* Fraction of the IMF which goes off this timestep*/
    massyield /= imf_norm;
    /* Mass yield from Sn1a*/
    double Nsn1a = sn1a_number(dtmyrstart, dtmyrend);
    massyield += Nsn1a * sn1a_total_metals;
    return massyield;
}

/* Compute the total metal yield for this star in this timestep*/
static double metal_yield(double dtmyrstart, double dtmyrend, double stellarmetal, gsl_interp2d * lifetime_interp,
                          MyFloat * MetalYields, MyFloat * MetalGenerated)
{
    double MassReturn = 0;
    double Nsn1a = sn1a_number(dtmyrstart, dtmyrend);
    int i;
    for(i = 0; i < NMETALS; i++)
        MetalYields[i] += Nsn1a * sn1a_yields[i];
    *MetalGenerated += Nsn1a * sn1a_total_metals;
    MassReturn += Nsn1a * sn1a_total_metals;

    double masshigh, masslow;
    find_mass_bin_limits(&masslow, &masshigh, dtmyrstart, dtmyrend, stellarmetal, lifetime_interp);
    /* Number of AGB stars/SnII by integrating the IMF*/
    //gsl_integration chabrier_imf()
    gsl_integration_romberg_workspace * gsl_work = gsl_integration_romberg_alloc(GSL_WORKSPACE);
    gsl_function ff = {chabrier_mass, NULL};

    double metalyield;
    size_t neval;
    gsl_integration_romberg(&ff, masslow, masshigh, 1e-4, 1e-3, &metalyield, &neval, gsl_work);

    return MassReturn;
}

/*! This function is the driver routine for the calculation of metal return. */
void
metal_return(const ActiveParticles * act, const ForceTree * const tree, Cosmology * CP, const double atime, double * StarVolumeSPH)
{
    TreeWalk tw[1] = {{0}};

    struct MetalReturnPriv priv[1];

    tw->ev_label = "METALS";
    tw->visit = (TreeWalkVisitFunction) treewalk_visit_ngbiter;
    tw->ngbiter = (TreeWalkNgbIterFunction) metal_return_ngbiter;
    tw->ngbiter_type_elsize = sizeof(TreeWalkNgbIterMetals);
    tw->haswork = metal_return_haswork;
    tw->fill = (TreeWalkFillQueryFunction) metal_return_copy;
    tw->reduce = NULL;
    tw->postprocess = (TreeWalkProcessFunction) metal_return_postprocess;
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
    priv->StellarAges = mymalloc("StellarAges", SlotsManager->info[4].size);
    priv->MassReturn = mymalloc("MassReturn", SlotsManager->info[4].size);
    priv->imf_norm = compute_imf_norm();
    int i;
    #pragma omp parallel for
    for(i=0; i < act->NumActiveParticle;i++)
    {
        int p_i = act->ActiveParticle ? act->ActiveParticle[i] : i;
        if(P[p_i].Type != 4)
            continue;
        priv->StellarAges[P[p_i].PI] = atime_to_myr(CP, STARP(p_i).FormationTime, atime);
        priv->MassReturn[P[p_i].PI] = mass_yield(STARP(p_i).LastEnrichmentMyr, priv->StellarAges[P[p_i].PI], STARP(p_i).Metallicity, priv->lifetime_interp, priv->imf_norm);
    }

    priv->spin = init_spinlocks(SlotsManager->info[0].size);
    treewalk_run(tw, act->ActiveParticle, act->NumActiveParticle);
    free_spinlocks(priv->spin);

    myfree(priv->MassReturn);
    myfree(priv->StellarAges);
    /* collect some timing information */
    walltime_measure("/SPH/Metals");
}

static void
metal_return_copy(int place, TreeWalkQueryMetals * input, TreeWalk * tw)
{
    input->Metallicity = STARP(place).Metallicity;
    input->Mass = P[place].Mass;
    input->Hsml = P[place].Hsml;
    input->StarVolumeSPH = METALS_GET_PRIV(tw)->StarVolumeSPH[place];
    int pi = P[place].PI;
    double dtmyrend = METALS_GET_PRIV(tw)->StellarAges[pi];
    double dtmyrstart = STARP(place).LastEnrichmentMyr;
    /* Do TotalMetalGenerated by computing the yield at this time.*/
    input->MassGenerated = METALS_GET_PRIV(tw)->MassReturn[pi];
    metal_yield(dtmyrstart, dtmyrend, input->Metallicity, METALS_GET_PRIV(tw)->lifetime_interp, input->TotalMetalGenerated, &input->MetalGenerated);
}

static void
metal_return_postprocess(int place, TreeWalk * tw)
{
    /* Conserve mass returned*/
    P[place].Mass -= METALS_GET_PRIV(tw)->MassReturn[P[place].PI];
    /* Update the last enrichment time*/
    STARP(place).LastEnrichmentMyr = METALS_GET_PRIV(tw)->StellarAges[P[place].PI];
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
        O->MassReturn = 0;
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
        /* Keep track of how much was returned for conservation purposes*/
        double thismass = wk * I->MassGenerated / I->StarVolumeSPH;
        O->MassReturn += thismass;
        double thismetal = wk * I->MetalGenerated / I->StarVolumeSPH;
        lock_spinlock(other, METALS_GET_PRIV(lv->tw)->spin);
        /* Add the metals to the particle.*/
        for(i = 0; i < NMETALS; i++)
            SPHP(other).Metals[i] += ThisMetals[i];
        /* Update total metallicity*/
        SPHP(other).Metallicity = (SPHP(other).Metallicity * P[other].Mass + thismetal)/(P[other].Mass + thismass);
        /* Update mass*/
        P[other].Mass += thismass;
        /* Add metals weighted by SPH kernel*/
        unlock_spinlock(other, METALS_GET_PRIV(lv->tw)->spin);
    }
    O->Ninteractions++;
}

/* Only stars return metals to the gas*/
static int
metal_return_haswork(int i, TreeWalk * tw)
{
    int pi = P[i].PI;
    /* Don't do enrichment from all stars, just young stars or those with significant enrichment*/
    int young = METALS_GET_PRIV(tw)->StellarAges[pi] < 100;
    int massreturned = METALS_GET_PRIV(tw)->MassReturn[pi] > 1e-4 * P[i].Mass;
    return P[i].Type == 4 && (young || massreturned);
}
