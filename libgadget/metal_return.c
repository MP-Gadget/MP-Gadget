#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_interp2d.h>
#include <omp.h>

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

#define GSL_WORKSPACE 1000

MyFloat * stellar_density(const ActiveParticles * act, const ForceTree * const tree);

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

struct interps
{
    gsl_interp2d * lifetime_interp;
    gsl_interp2d * agb_mass_interp;
    gsl_interp2d * agb_metallicity_interp;
    gsl_interp2d * agb_metals_interp[NMETALS];
    gsl_interp2d * snii_mass_interp;
    gsl_interp2d * snii_metallicity_interp;
    gsl_interp2d * snii_metals_interp[NMETALS];
};

void setup_metal_table_interp(struct interps * interp)
{
    interp->lifetime_interp = gsl_interp2d_alloc(gsl_interp2d_bilinear, LIFE_NMET, LIFE_NMASS);
    gsl_interp2d_init(interp->lifetime_interp, lifetime_metallicity, lifetime_masses, lifetime, LIFE_NMET, LIFE_NMASS);
    interp->agb_mass_interp = gsl_interp2d_alloc(gsl_interp2d_bilinear, AGB_NMET, AGB_NMASS);
    gsl_interp2d_init(interp->agb_mass_interp, agb_metallicities, agb_masses, agb_total_mass, AGB_NMET, AGB_NMASS);
    interp->agb_metallicity_interp = gsl_interp2d_alloc(gsl_interp2d_bilinear, AGB_NMET, AGB_NMASS);
    gsl_interp2d_init(interp->agb_metallicity_interp, agb_metallicities, agb_masses, agb_total_metals, AGB_NMET, AGB_NMASS);
    int i;
    for(i=0; i<NMETALS; i++) {
        interp->agb_metals_interp[i] = gsl_interp2d_alloc(gsl_interp2d_bilinear, AGB_NMET, AGB_NMASS);
        gsl_interp2d_init(interp->agb_metals_interp[i], agb_metallicities, agb_masses, agb_yield[i], AGB_NMET, AGB_NMASS);
    }
    interp->snii_mass_interp = gsl_interp2d_alloc(gsl_interp2d_bilinear, SNII_NMET, SNII_NMASS);
    gsl_interp2d_init(interp->snii_mass_interp, snii_metallicities, snii_masses, snii_total_mass, SNII_NMET, SNII_NMASS);
    interp->snii_metallicity_interp = gsl_interp2d_alloc(gsl_interp2d_bilinear, SNII_NMET, SNII_NMASS);
    gsl_interp2d_init(interp->snii_metallicity_interp, snii_metallicities, snii_masses, snii_total_metals, SNII_NMET, SNII_NMASS);
    for(i=0; i<NMETALS; i++) {
        interp->snii_metals_interp[i] = gsl_interp2d_alloc(gsl_interp2d_bilinear, SNII_NMET, SNII_NMASS);
        gsl_interp2d_init(interp->snii_metals_interp[i], snii_metallicities, snii_masses, snii_yield[i], SNII_NMET, SNII_NMASS);
    }
}

struct MetalReturnPriv {
    double atime;
    inttime_t Ti_Current;
    MyFloat * StellarAges;
    MyFloat * MassReturn;
    double imf_norm;
    double hub;
    double Unit_Mass_in_g;
    Cosmology *CP;
    MyFloat * StarVolumeSPH;
    struct interps interp;
    struct SpinLocks * spin;
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
    MyFloat StarVolumeSPH;
} TreeWalkResultMetalDensity;

typedef struct {
    TreeWalkResultBase base;
    /* This is the total mass returned to
     * the surrounding gas particles, for mass conservation.*/
    MyFloat MassReturn;
    int Ninteractions;
} TreeWalkResultMetals;

typedef struct {
    TreeWalkNgbIterBase base;
    DensityKernel kernel;
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

double atime_integ(double atime, void * params)
{
    Cosmology * CP = (Cosmology *) params;
    return 1/(hubble_function(CP, atime) * atime);
}

/* Compute the difference in internal time units between two scale factors.*/
static double atime_to_myr(Cosmology *CP, double atime1, double atime2)
{
    /* t = dt/da da = 1/(Ha) da*/
    /* Approximate hubble function as constant here: we only care
     * about metal return over a single timestep*/
    gsl_function ff = {atime_integ, CP};
    double tmyr, abserr;
    gsl_integration_workspace * gsl_work = gsl_integration_workspace_alloc(GSL_WORKSPACE);
    gsl_integration_qag(&ff, atime1, atime2, 1e-4, 1e-3, GSL_WORKSPACE, GSL_INTEG_GAUSS61, gsl_work, &tmyr, &abserr);
    return tmyr * CP->UnitTime_in_s / SEC_PER_MEGAYEAR;
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

/* Parameters of the interpolator
 * to hand to the imf integral.
 * Use different interpolation structures
 * for mass return, metal return and yield.*/
struct imf_integ_params
{
    gsl_interp2d * interp;
    const double * masses;
    const double * metallicities;
    const double * weights;
    double metallicity;
};

double chabrier_imf_integ (double mass, void * params)
{
    struct imf_integ_params * para = (struct imf_integ_params * ) params;
    /* This is needed so that the yield for SNII with masses between 8 and 13 Msun
     * are the same as the smallest mass in the table, 13 Msun,
     * but they still contribute their number density to the IMF.*/
    double intpmass = mass;
    if(mass < para->masses[0])
        intpmass = para->masses[0];
    if(mass > para->masses[para->interp->ysize-1])
        intpmass = para->masses[para->interp->ysize-1];
    double weight = gsl_interp2d_eval(para->interp, para->metallicities, para->masses, para->weights, para->metallicity, intpmass, NULL, NULL);
    return weight * chabrier_imf(mass);
}

double chabrier_mass(double mass, void * params)
{
    return mass * chabrier_imf(mass);
}

/* Compute factor to normalise the total mass in the IMF to unity.*/
double compute_imf_norm(void)
{
    double norm, abserr;
    gsl_function ff = {chabrier_mass, NULL};
    gsl_integration_workspace * gsl_work = gsl_integration_workspace_alloc(GSL_WORKSPACE);
    gsl_integration_qag(&ff, MINMASS, MAXMASS, 1e-4, 1e-3, GSL_WORKSPACE, GSL_INTEG_GAUSS61, gsl_work, &norm, &abserr);
    return norm;
}

/* Compute number of Sn1a */
static double sn1a_number(double dtmyrstart, double dtmyrend, double hub)
{
    /* Number of Sn1a events follows a delay time distribution (1305.2913, eq. 10) */
    const double sn1aindex = 1.12;
    const double tau8msun = 40;
    if(dtmyrend < tau8msun)
        return 0;
    /* Lower integration limit modelling formation time of WDs*/
    if(dtmyrstart < tau8msun)
        dtmyrstart  = tau8msun;
    /* Total number of Sn1a events from this star: integral evaluated from t=tau8msun to t=hubble time.*/
    const double totalSN1a = 1- pow(1/(hub*HUBBLE * SEC_PER_MEGAYEAR)/tau8msun, 1-sn1aindex);
    /* This is the integral of the DTD, normalised to the N0 rate.*/
    double Nsn1a = MetalParams.Sn1aN0 /totalSN1a * (pow(dtmyrstart / tau8msun, 1-sn1aindex) - pow(dtmyrend / tau8msun, 1-sn1aindex));
    return Nsn1a;
}

static double compute_agb_yield(gsl_interp2d * agb_interp, const double * agb_weights, double stellarmetal, double masslow, double masshigh, gsl_integration_workspace * gsl_work )
{
    struct imf_integ_params para;
    gsl_function ff = {chabrier_imf_integ, &para};
    double agbyield = 0, abserr;
    /* Only return AGB metals for the range of AGB stars*/
    if (masshigh > SNAGBSWITCH)
        masshigh = SNAGBSWITCH;
    if (masslow < agb_masses[0])
        masslow = agb_masses[0];
    if (stellarmetal > agb_metallicities[SNII_NMET-1])
        stellarmetal = agb_metallicities[SNII_NMET-1];
    if (stellarmetal < agb_metallicities[0])
        stellarmetal = agb_metallicities[0];
    /* This happens if no bins in range had dying stars this timestep*/
    if(masslow >= masshigh)
        return 0;
    para.interp = agb_interp;
    para.masses = agb_masses;
    para.metallicities = agb_metallicities;
    para.metallicity = stellarmetal;
    para.weights = agb_weights;
    gsl_integration_qag(&ff, masslow, masshigh, 1e-2, 1e-2, GSL_WORKSPACE, GSL_INTEG_GAUSS61, gsl_work, &agbyield, &abserr);
    return agbyield;
}

static double compute_snii_yield(gsl_interp2d * snii_interp, const double * snii_weights, double stellarmetal, double masslow, double masshigh, gsl_integration_workspace * gsl_work )
{
    struct imf_integ_params para;
    gsl_function ff = {chabrier_imf_integ, &para};
    double yield = 0, abserr;
    /* Only return metals for the range of SNII stars.*/
    if (masshigh > snii_masses[SNII_NMASS-1])
        masshigh = snii_masses[SNII_NMASS-1];
    if (masslow < SNAGBSWITCH)
        masslow = SNAGBSWITCH;
    if (stellarmetal > snii_metallicities[SNII_NMET-1])
        stellarmetal = snii_metallicities[SNII_NMET-1];
    if (stellarmetal < snii_metallicities[0])
        stellarmetal = snii_metallicities[0];
    para.interp = snii_interp;
    para.masses = snii_masses;
    para.metallicities = snii_metallicities;
    para.metallicity = stellarmetal;
    para.weights = snii_weights;
    /* This happens if no bins in range had dying stars this timestep*/
    if(masslow >= masshigh)
        return 0;
    gsl_integration_qag(&ff, masslow, masshigh, 1e-2, 1e-2, GSL_WORKSPACE, GSL_INTEG_GAUSS61, gsl_work, &yield, &abserr);
    return yield;
}

/* Compute the total mass yield for this star in this timestep*/
static double mass_yield(double dtmyrstart, double dtmyrend, double hub, double stellarmetal, struct interps * interp, double imf_norm)
{
    double masshigh, masslow;
    find_mass_bin_limits(&masslow, &masshigh, dtmyrstart, dtmyrend, stellarmetal, interp->lifetime_interp);
    /* Number of AGB stars/SnII by integrating the IMF*/
    gsl_integration_workspace * gsl_work = gsl_integration_workspace_alloc(GSL_WORKSPACE);
    /* Set up for SNII*/
    double massyield = 0;
    massyield += compute_agb_yield(interp->agb_mass_interp, agb_total_mass, stellarmetal, masslow, masshigh, gsl_work);
    massyield += compute_snii_yield(interp->snii_mass_interp, snii_total_mass, stellarmetal, masslow, masshigh, gsl_work);
    /* Fraction of the IMF which goes off this timestep*/
    massyield /= imf_norm;
    /* Mass yield from Sn1a*/
    double Nsn1a = sn1a_number(dtmyrstart, dtmyrend, hub);
    massyield += Nsn1a * sn1a_total_metals;
    return massyield;
}

/* Compute the total metal yield for this star in this timestep*/
static double metal_yield(double dtmyrstart, double dtmyrend, double hub, double stellarmetal, struct interps * interp, MyFloat * MetalYields, double imf_norm)
{
    double MetalGenerated = 0;

    double masshigh, masslow;
    find_mass_bin_limits(&masslow, &masshigh, dtmyrstart, dtmyrend, stellarmetal, interp->lifetime_interp);
    /* Number of AGB stars/SnII by integrating the IMF*/
    gsl_integration_workspace * gsl_work = gsl_integration_workspace_alloc(GSL_WORKSPACE);
    /* Set up for SNII*/
    MetalGenerated += compute_agb_yield(interp->agb_metallicity_interp, agb_total_metals, stellarmetal, masslow, masshigh, gsl_work);
    MetalGenerated += compute_snii_yield(interp->snii_metallicity_interp, snii_total_metals, stellarmetal, masslow, masshigh, gsl_work);
    MetalGenerated /= imf_norm;

    int i;
    for(i = 0; i < NMETALS; i++)
    {
        MetalYields[i] = 0;
        MetalYields[i] += compute_agb_yield(interp->agb_metals_interp[i], agb_yield[i], stellarmetal, masslow, masshigh, gsl_work);
        MetalYields[i] += compute_snii_yield(interp->snii_metals_interp[i], snii_yield[i], stellarmetal, masslow, masshigh, gsl_work);
        MetalYields[i] /= imf_norm;
    }
    double Nsn1a = sn1a_number(dtmyrstart, dtmyrend, hub);
    for(i = 0; i < NMETALS; i++)
        MetalYields[i] += Nsn1a * sn1a_yields[i];
    MetalGenerated += Nsn1a * sn1a_total_metals;

    return MetalGenerated;
}

/*! This function is the driver routine for the calculation of metal return. */
void
metal_return(const ActiveParticles * act, const ForceTree * const tree, Cosmology * CP, const double atime, const double UnitMass_in_g)
{
    TreeWalk tw[1] = {{0}};

    struct MetalReturnPriv priv[1];

    /* Do nothing if no stars yet*/
    int64_t totstar;
    sumup_large_ints(1, &SlotsManager->info[4].size, &totstar);
    if(totstar == 0)
        return;

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
    priv->hub = CP->HubbleParam;
    priv->Unit_Mass_in_g = UnitMass_in_g;
    message(0, "Starting metal return\n");

    /* Initialize some time factors*/
    METALS_GET_PRIV(tw)->atime = atime;
    setup_metal_table_interp(&METALS_GET_PRIV(tw)->interp);
    priv->StellarAges = mymalloc("StellarAges", SlotsManager->info[4].size * sizeof(MyFloat));
    priv->MassReturn = mymalloc("MassReturn", SlotsManager->info[4].size * sizeof(MyFloat));
    priv->imf_norm = compute_imf_norm();
    double unitfactor = SOLAR_MASS / (METALS_GET_PRIV(tw)->Unit_Mass_in_g / METALS_GET_PRIV(tw)->hub);

    int i;
    #pragma omp parallel for
    for(i=0; i < act->NumActiveParticle;i++)
    {
        int p_i = act->ActiveParticle ? act->ActiveParticle[i] : i;
        if(P[p_i].Type != 4)
            continue;
        priv->StellarAges[P[p_i].PI] = atime_to_myr(CP, STARP(p_i).FormationTime, atime);
        priv->MassReturn[P[p_i].PI] = unitfactor * mass_yield(STARP(p_i).LastEnrichmentMyr, priv->StellarAges[P[p_i].PI], STARP(p_i).Metallicity, CP->HubbleParam, &priv->interp, priv->imf_norm);
        /* Guard against making a zero mass particle*/
        if(priv->MassReturn[P[p_i].PI] > 0.9 * P[p_i].Mass)
            priv->MassReturn[P[p_i].PI] = 0.9 * P[p_i].Mass;
    }

    /* Compute total number of weights around each star for actively returning stars*/
    METALS_GET_PRIV(tw)->StarVolumeSPH = stellar_density(act, tree);

    /* Do the metal return*/
    priv->spin = init_spinlocks(SlotsManager->info[0].size);
    treewalk_run(tw, act->ActiveParticle, act->NumActiveParticle);
    free_spinlocks(priv->spin);
    message(0, "Done metal return\n");

    myfree(priv->StarVolumeSPH);
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
    int pi = P[place].PI;
    input->StarVolumeSPH = METALS_GET_PRIV(tw)->StarVolumeSPH[pi];
    if(input->StarVolumeSPH ==0)
        endrun(3, "StarVolumeSPH %g hsml %g\n", input->StarVolumeSPH, input->Hsml);
    double dtmyrend = METALS_GET_PRIV(tw)->StellarAges[pi];
    double dtmyrstart = STARP(place).LastEnrichmentMyr;
    /* Do TotalMetalGenerated by computing the yield at this time.*/
    input->MassGenerated = METALS_GET_PRIV(tw)->MassReturn[pi];
    input->MetalGenerated = metal_yield(dtmyrstart, dtmyrend, input->Metallicity, METALS_GET_PRIV(tw)->hub, &METALS_GET_PRIV(tw)->interp, input->TotalMetalGenerated, METALS_GET_PRIV(tw)->imf_norm);
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
        density_kernel_init(&iter->kernel, I->Hsml, GetDensityKernelType());
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

    if(r2 > 0 && r2 < iter->kernel.HH)
    {
        double wk = density_kernel_wk(&iter->kernel, iter->base.r);
        int i;
        /* Volume of particle weighted by the SPH kernel*/
        double volume = P[other].Mass / SPHP(other).Density;
        double ThisMetals[NMETALS];
        /* Unit conversion factor to internal units: */
        double unitfactor = SOLAR_MASS / (METALS_GET_PRIV(lv->tw)->Unit_Mass_in_g / METALS_GET_PRIV(lv->tw)->hub);
        if(I->StarVolumeSPH ==0)
            endrun(3, "StarVolumeSPH %g hsml %g\n", I->StarVolumeSPH, I->Hsml);
        for(i = 0; i < NMETALS; i++)
            ThisMetals[i] = wk * volume * I->TotalMetalGenerated[i] / I->StarVolumeSPH * unitfactor;
        /* Keep track of how much was returned for conservation purposes*/
        double thismass = wk * I->MassGenerated / I->StarVolumeSPH;
        O->MassReturn += thismass;
        double thismetal = wk * I->MetalGenerated / I->StarVolumeSPH * unitfactor;
        double newmass;
        lock_spinlock(other, METALS_GET_PRIV(lv->tw)->spin);
        /* Add the metals to the particle.*/
        for(i = 0; i < NMETALS; i++)
            SPHP(other).Metals[i] += ThisMetals[i];
        /* Update total metallicity*/
        SPHP(other).Metallicity = (SPHP(other).Metallicity * P[other].Mass + thismetal)/(P[other].Mass + thismass);
        /* Update mass*/
        P[other].Mass += thismass;
        newmass = P[other].Mass;
        /* Add metals weighted by SPH kernel*/
        unlock_spinlock(other, METALS_GET_PRIV(lv->tw)->spin);
        if(newmass <= 0)
            endrun(3, "New mass %g new metal %g in particle %d id %ld from star mass %g metallicity %g\n",
                   newmass, SPHP(other).Metallicity, other, P[other].ID, I->Mass, I->Metallicity);
    }
    O->Ninteractions++;
}

/* Only stars return metals to the gas*/
static int
metal_return_haswork(int i, TreeWalk * tw)
{
    if(P[i].Type != 4)
        return 0;
    int pi = P[i].PI;
    /* New stars or stars with zero mass return will not do anything: nothing has yet died*/
    if(METALS_GET_PRIV(tw)->StellarAges[pi] == 0 || METALS_GET_PRIV(tw)->MassReturn[pi] == 0)
        return 0;
    /* Don't do enrichment from all stars, just young stars or those with significant enrichment*/
    int young = METALS_GET_PRIV(tw)->StellarAges[pi] < 100;
    int massreturned = METALS_GET_PRIV(tw)->MassReturn[pi] > 1e-4 * P[i].Mass;
    return young || massreturned;
}

/* Here comes code to compute the star particle density*/
#define MAXITER 200

typedef struct {
    TreeWalkNgbIterBase base;
    DensityKernel kernel;
    double kernel_volume;
} TreeWalkNgbIterStellarDensity;

typedef struct
{
    TreeWalkQueryBase base;
    MyFloat Hsml;
} TreeWalkQueryStellarDensity;

typedef struct {
    TreeWalkResultBase base;
    MyFloat VolumeSPH;
    MyFloat Ngb;
} TreeWalkResultStellarDensity;

struct StellarDensityPriv {
    /* Current number of neighbours*/
    MyFloat *NumNgb;
    /* Lower and upper bounds on smoothing length*/
    MyFloat *Left, *Right;
    MyFloat * VolumeSPH;
    int NIteration;
    size_t *NPLeft;
    int **NPRedo;
    /*!< Desired number of SPH neighbours */
    double DesNumNgb;
};

#define STELLAR_DENSITY_GET_PRIV(tw) ((struct StellarDensityPriv*) ((tw)->priv))

static void
stellar_density_copy(int place, TreeWalkQueryStellarDensity * I, TreeWalk * tw)
{
    I->Hsml = P[place].Hsml;
}

static void
stellar_density_reduce(int place, TreeWalkResultStellarDensity * remote, enum TreeWalkReduceMode mode, TreeWalk * tw)
{
    TREEWALK_REDUCE(STELLAR_DENSITY_GET_PRIV(tw)->NumNgb[place], remote->Ngb);
    int pi = P[place].PI;
    TREEWALK_REDUCE(STELLAR_DENSITY_GET_PRIV(tw)->VolumeSPH[pi], remote->VolumeSPH);
}

void stellar_density_check_neighbours (int pi, TreeWalk * tw)
{
    /* now check whether we had enough neighbours */

    double desnumngb = STELLAR_DENSITY_GET_PRIV(tw)->DesNumNgb;

    MyFloat * Left = STELLAR_DENSITY_GET_PRIV(tw)->Left;
    MyFloat * Right = STELLAR_DENSITY_GET_PRIV(tw)->Right;
    MyFloat * NumNgb = STELLAR_DENSITY_GET_PRIV(tw)->NumNgb;

    int i = P[pi].PI;

    if(NumNgb[i] < (desnumngb - 2) ||
            (NumNgb[i] > (desnumngb + 2)))
    {
        /* This condition is here to prevent the density code looping forever if it encounters
         * multiple particles at the same position. If this happens you likely have worse
         * problems anyway, so warn also. */
        if((Right[i] - Left[i]) < 1.0e-3 * Left[i])
        {
            /* If this happens probably the exchange is screwed up and all your particles have moved to (0,0,0)*/
            message(1, "Very tight Hsml bounds for i=%d ID=%lu Hsml=%g Left=%g Right=%g Ngbs=%g Right-Left=%g pos=(%g|%g|%g)\n",
             i, P[i].ID, P[i].Hsml, Left[i], Right[i], NumNgb[i], Right[i] - Left[i], P[i].Pos[0], P[i].Pos[1], P[i].Pos[2]);
            P[i].Hsml = Right[i];
            return;
        }

        /* If we need more neighbours, move the lower bound up. If we need fewer, move the upper bound down.*/
        if(NumNgb[i] < desnumngb) {
                Left[i] = P[i].Hsml;
        } else {
                Right[i] = P[i].Hsml;
        }

        /* Next step is geometric mean of previous. */
        if(Right[i] < tw->tree->BoxSize && Left[i] > 0)
            P[i].Hsml = pow(0.5 * (pow(Left[i], 3) + pow(Right[i], 3)), 1.0 / 3);
        else
        {
            if(!(Right[i] < tw->tree->BoxSize) && Left[i] == 0)
                endrun(8188, "Cannot occur. Check for memory corruption: i=%d L = %g R = %g N=%g. Type %d, Pos %g %g %g", i, Left[i], Right[i], NumNgb[i], P[i].Type, P[i].Pos[0], P[i].Pos[1], P[i].Pos[2]);

            /* If this is the first step we can be faster by increasing or decreasing current Hsml by a constant factor*/
            if(Right[i] > 0.99 * tw->tree->BoxSize && Left[i] > 0)
                P[i].Hsml *= 1.26;

            if(Right[i] < 0.99*tw->tree->BoxSize && Left[i] == 0)
                P[i].Hsml /= 1.26;
        }
        /* More work needed: add this particle to the redo queue*/
        int tid = omp_get_thread_num();
        STELLAR_DENSITY_GET_PRIV(tw)->NPRedo[tid][STELLAR_DENSITY_GET_PRIV(tw)->NPLeft[tid]] = i;
        STELLAR_DENSITY_GET_PRIV(tw)->NPLeft[tid] ++;
    }

    if(STELLAR_DENSITY_GET_PRIV(tw)->NIteration >= MAXITER - 10)
    {
         message(1, "i=%d ID=%lu Hsml=%g Left=%g Right=%g Ngbs=%g Right-Left=%g\n   pos=(%g|%g|%g)\n",
             i, P[i].ID, P[i].Hsml, Left[i], Right[i],
             NumNgb[i], Right[i] - Left[i], P[i].Pos[0], P[i].Pos[1], P[i].Pos[2]);
    }
}

static void
stellar_density_ngbiter(
        TreeWalkQueryStellarDensity * I,
        TreeWalkResultStellarDensity * O,
        TreeWalkNgbIterStellarDensity * iter,
        LocalTreeWalk * lv)
{
    if(iter->base.other == -1) {
        const double h = I->Hsml;
        density_kernel_init(&iter->kernel, h, GetDensityKernelType());
        iter->kernel_volume = density_kernel_volume(&iter->kernel);

        iter->base.Hsml = h;
        iter->base.mask = 1; /* gas only */
        iter->base.symmetric = NGB_TREEFIND_ASYMMETRIC;
        return;
    }
    const int other = iter->base.other;
    const double r = iter->base.r;
    const double r2 = iter->base.r2;

    if(r2 < iter->kernel.HH)
    {
        const double u = r * iter->kernel.Hinv;
        const double wk = density_kernel_wk(&iter->kernel, u);
        O->Ngb += wk * iter->kernel_volume;
        /* For stars we need the total weighting, sum(w_k m_k / rho_k).*/
        O->VolumeSPH += P[other].Mass * wk / SPHP(other).Density;
    }
}

MyFloat *
stellar_density(const ActiveParticles * act, const ForceTree * const tree)
{
    TreeWalk tw[1] = {{0}};
    struct StellarDensityPriv priv[1];

    tw->ev_label = "STELLAR_DENSITY";
    tw->visit = (TreeWalkVisitFunction) treewalk_visit_ngbiter;
    tw->ngbiter_type_elsize = sizeof(TreeWalkNgbIterStellarDensity);
    tw->ngbiter = (TreeWalkNgbIterFunction) stellar_density_ngbiter;
    tw->haswork = metal_return_haswork;
    tw->fill = (TreeWalkFillQueryFunction) stellar_density_copy;
    tw->reduce = (TreeWalkReduceResultFunction) stellar_density_reduce;
    tw->postprocess = (TreeWalkProcessFunction) stellar_density_check_neighbours;
    tw->query_type_elsize = sizeof(TreeWalkQueryStellarDensity);
    tw->result_type_elsize = sizeof(TreeWalkResultStellarDensity);
    tw->priv = priv;
    tw->tree = tree;

    int i;
    int64_t ntot = 0;

    walltime_measure("/Misc");
    priv->VolumeSPH = mymalloc("StarVolumeSPH", SlotsManager->info[4].size * sizeof(MyFloat));

    priv->Left = (MyFloat *) mymalloc("DENS_PRIV->Left", SlotsManager->info[4].size * sizeof(MyFloat));
    priv->Right = (MyFloat *) mymalloc("DENS_PRIV->Right", SlotsManager->info[4].size * sizeof(MyFloat));
    priv->NumNgb = (MyFloat *) mymalloc("DENS_PRIV->NumNgb", SlotsManager->info[4].size * sizeof(MyFloat));

    priv->NIteration = 0;
    priv->DesNumNgb = GetNumNgb(GetDensityKernelType());

    /* Init Left and Right: this has to be done before treewalk */
    memset(priv->NumNgb, 0, SlotsManager->info[4].size * sizeof(MyFloat));
    memset(priv->Left, 0, SlotsManager->info[4].size * sizeof(MyFloat));
    #pragma omp parallel for
    for(i = 0; i < SlotsManager->info[4].size; i++)
        priv->Right[i] = tree->BoxSize;

    /* allocate buffers to arrange communication */
    int NumThreads = omp_get_max_threads();
    priv->NPLeft = ta_malloc("NPLeft", size_t, NumThreads);
    priv->NPRedo = ta_malloc("NPRedo", int *, NumThreads);
    int alloc_high = 0;
    int * ReDoQueue = act->ActiveParticle;
    int size = act->NumActiveParticle;
    if(size > SlotsManager->info[4].size)
        size = SlotsManager->info[4].size;

    /* we will repeat the whole thing for those particles where we didn't find enough neighbours */
    do {
        /* The RedoQueue needs enough memory to store every particle on every thread, because
         * we cannot guarantee that the sph particles are evenly spread across threads!*/
        int * CurQueue = ReDoQueue;
        /* The ReDoQueue swaps between high and low allocations so we can have two allocated alternately*/
        if(!alloc_high) {
            ReDoQueue = (int *) mymalloc2("ReDoQueue", size * sizeof(int) * NumThreads);
            alloc_high = 1;
        }
        else {
            ReDoQueue = (int *) mymalloc("ReDoQueue", size * sizeof(int) * NumThreads);
            alloc_high = 0;
        }
        gadget_setup_thread_arrays(ReDoQueue, priv->NPRedo, priv->NPLeft, size, NumThreads);
        treewalk_run(tw, CurQueue, size);

        message(0, "Found density for %d stars\n", tw->WorkSetSize);
        tw->haswork = NULL;
        /* Now done with the current queue*/
        if(priv->NIteration > 0)
            myfree(CurQueue);

        /* Set up the next queue*/
        size = gadget_compact_thread_arrays(ReDoQueue, priv->NPRedo, priv->NPLeft, NumThreads);

        sumup_large_ints(1, &size, &ntot);
        if(ntot == 0){
            myfree(ReDoQueue);
            break;
        }

        /*Shrink memory*/
        ReDoQueue = myrealloc(ReDoQueue, sizeof(int) * size);

        priv->NIteration ++;

        if(priv->NIteration > 0) {
            message(0, "star density iteration %d: need to repeat for %ld particles.\n", priv->NIteration, ntot);
#ifdef DEBUG
            if(ntot == 1 && size > 0 && priv->NIteration > 20 ) {
                int pp = ReDoQueue[0];
                message(1, "Remaining i=%d, t %d, pos %g %g %g, hsml: %g ngb: %g\n", pp, P[pp].Type, P[pp].Pos[0], P[pp].Pos[1], P[pp].Pos[2], P[pp].Hsml, priv->NumNgb[pp]);
            }
#endif
        }
        if(priv->NIteration > MAXITER) {
            endrun(1155, "failed to converge in neighbour iteration in density()\n");
        }
    } while(1);

#ifdef DEBUG
    for(i = 0; i < act->NumActiveParticle; i++) {
        int a = act->ActiveParticle ? act->ActiveParticle[i] : i;
        if(metal_return_haswork(a, tw) && priv->VolumeSPH[P[a].PI] == 0)
            endrun(3, "i = %d pi = %d StarVolumeSPH %g hsml %g\n", a, P[a].PI, priv->VolumeSPH[P[a].PI], P[a].Hsml);
    }
#endif
    ta_free(priv->NPRedo);
    ta_free(priv->NPLeft);
    myfree(priv->NumNgb);
    myfree(priv->Right);
    myfree(priv->Left);
    /* collect some timing information */
    walltime_measure("/SPH/Metals/Density");
    return priv->VolumeSPH;
}
