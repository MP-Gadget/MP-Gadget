#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_interp2d.h>
#include <gsl/gsl_roots.h>
#include <gsl/gsl_errno.h>
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

#define MAXITER 200
#define GSL_WORKSPACE 1000

MyFloat * stellar_density(const ActiveParticles * act, MyFloat * StellarAges, MyFloat * MassReturn, const ForceTree * const tree);

/*! \file metal_return.c
 *  \brief Compute the mass return rate of metals from stellar evolution.
 *
 *  This file returns metals from stars with some delay.
 *  Delayed sources followed are AGB stars, SNII and Sn1a.
 *  9 Species specific yields are stored in the stars and the gas particles.
 *  Gas enrichment is not run every timestep, but only for stars that have
 *  significant enrichment, or are young.
 *  The model closely follows Illustris-TNG, https://arxiv.org/abs/1703.02970
 *  However the tables used are slightly different: we consider SNII between 8 and 40 Msun
 *  following Kobayashi 2006, where they use a hybrid of Kobayashi and Portinari.
 *  AGB yields are from Karakas 2010, like TNG, but stars with mass > 6.5 are
 *  from Doherty 2014, not Fishlock 2014. More details of the model can be found in
 *  the Illustris model Vogelsberger 2013: https://arxiv.org/abs/1305.2913
 *  As the Kobayashi table only goes to 13 Msun, stars with masses 8-13 Msun
 *  are assumed to yield like a 13 Msun star, but scaled by a factor of (M/13).
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
    int SPHWeighting;
} MetalParams;

/*Set the parameters of the hydro module*/
void
set_metal_return_params(ParameterSet * ps)
{
    int ThisTask;
    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
    if(ThisTask == 0) {
        MetalParams.Sn1aN0 = param_get_double(ps, "MetalsSn1aN0");
        MetalParams.SPHWeighting = param_get_int(ps, "MetalsSPHWeighting");
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

/* Build the interpolators for each yield table. We use bilinear interpolation
 * so there is no extra memory allocation and we never free the tables*/
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
    gsl_integration_workspace ** gsl_work;
    MyFloat * StellarAges;
    MyFloat * MassReturn;
    MyFloat * LowDyingMass;
    MyFloat * HighDyingMass;
    double imf_norm;
    double hub;
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
    MyFloat MetalSpeciesGenerated[NMETALS];
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
static double atime_to_myr(Cosmology *CP, double atime1, double atime2, gsl_integration_workspace * gsl_work)
{
    /* t = dt/da da = 1/(Ha) da*/
    /* Approximate hubble function as constant here: we only care
     * about metal return over a single timestep*/
    gsl_function ff = {atime_integ, CP};
    double tmyr, abserr;
    gsl_integration_qag(&ff, atime1, atime2, 1e-4, 1e-3, GSL_WORKSPACE, GSL_INTEG_GAUSS61, gsl_work, &tmyr, &abserr);
    return tmyr * CP->UnitTime_in_s / SEC_PER_MEGAYEAR;
}

/* Functions for the root finder*/
struct massbin_find_params
{
    double dtfind;
    double stellarmetal;
    gsl_interp2d * lifetime_tables;
    gsl_interp_accel * metalacc;
    gsl_interp_accel * massacc;
};

/* This is the inverse of the lifetime function from the tables.
 * Need to find the stars with a given lifetime*/
double
massendlife (double mass, void *params)
{
  struct massbin_find_params *p = (struct massbin_find_params *) params;
  double tlife = gsl_interp2d_eval(p->lifetime_tables, lifetime_metallicity, lifetime_masses, lifetime, p->stellarmetal, mass, p->metalacc, p->massacc);
  double tlifemyr = tlife/1e6;
  return tlifemyr - p->dtfind;
}

/* Solve the lifetime function to find the lowest and highest mass bin that dies this timestep*/
double do_rootfinding(struct massbin_find_params *p, double mass_low, double mass_high)
{
    int iter = 0;
    gsl_function F;

    F.function = &massendlife;
    F.params = p;

    const gsl_root_fsolver_type *T = gsl_root_fsolver_falsepos;
    gsl_root_fsolver * s = gsl_root_fsolver_alloc (T);
    gsl_root_fsolver_set (s, &F, mass_low, mass_high);

    /* Iterate until we have an idea of the mass bins dying this timestep.
     * No check is done for success, but it should always be close enough.*/
    for(iter = 0; iter < MAXITER; iter++)
    {
      gsl_root_fsolver_iterate (s);
      mass_low = gsl_root_fsolver_x_lower (s);
      mass_high = gsl_root_fsolver_x_upper (s);
      /* We don't need to be that accurate because the tables aren't! */
      int status = gsl_root_test_interval (mass_low, mass_high,
                                       0, 0.005);
      //message(4, "lo %g hi %g root %g val %g\n", mass_low, mass_high, gsl_root_fsolver_root(s), massendlife(gsl_root_fsolver_root(s), p));
      if (status == GSL_SUCCESS)
        break;
  }
  double root = gsl_root_fsolver_root(s);
  gsl_root_fsolver_free (s);
  return root;
}

/* Find the mass bins which die in this timestep using the lifetime table.
 * dtstart, dtend - time at start and end of timestep in Myr.
 * stellarmetal - metallicity of the star.
 * lifetime_tables - 2D interpolation table of the lifetime.
 * masshigh, masslow - pointers in which to store the high and low lifetime limits
 */
static void find_mass_bin_limits(double * masslow, double * masshigh, const double dtstart, const double dtend, double stellarmetal, gsl_interp2d * lifetime_tables)
{
    /* Clamp metallicities to the table values.*/
    if(stellarmetal < lifetime_metallicity[0])
        stellarmetal = lifetime_metallicity[0];
    if(stellarmetal > lifetime_metallicity[LIFE_NMET-1])
        stellarmetal = lifetime_metallicity[LIFE_NMET-1];

    /* Find the root with GSL routines. */
    struct massbin_find_params p = {0};
    p.metalacc = gsl_interp_accel_alloc();
    p.massacc = gsl_interp_accel_alloc();
    p.lifetime_tables = lifetime_tables;
    p.stellarmetal = stellarmetal;
    /* First find stars that died before the end of this timebin*/
    p.dtfind = dtend;
    /* If no stars have died yet*/
    if(massendlife (MAXMASS, &p) >= 0)
    {
        *masslow = MAXMASS;
        *masshigh = MAXMASS;
        return;
    }
    /* All stars die before the end of this timestep*/
    if(massendlife (agb_masses[0], &p) <= 0)
        *masslow = lifetime_masses[0];
    else
        *masslow = do_rootfinding(&p, agb_masses[0], MAXMASS);

    /* Now find stars that died before the start of this timebin*/
    p.dtfind = dtstart;
    /* Now we know that life(masslow) = dtend, so life(masslow) > dtstart, so life(masslow) - dtstart > 0
     * This is when no stars have died at the beginning of this timestep.*/
    if(massendlife (MAXMASS, &p) >= 0)
        *masshigh = MAXMASS;
    /* This can sometimes happen due to root finding inaccuracy.
     * Just do this star next timestep.*/
    else if(massendlife (*masslow, &p) <= 0)
        *masshigh = *masslow;
    else
        *masshigh = do_rootfinding(&p, *masslow, MAXMASS);
    gsl_interp_accel_free(p.metalacc);
    gsl_interp_accel_free(p.massacc);
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

/* Integrand for a function which computes a Chabrier IMF weighted quantity.*/
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
    /* This rescales the return by the original mass of the star, if it was outside the table.
     * It means that, for example, an 8 Msun star does not return more than 8 Msun. */
    weight *= (mass/intpmass);
    return weight * chabrier_imf(mass);
}

/* Helper for the IMF normalisation*/
double chabrier_mass(double mass, void * params)
{
    return mass * chabrier_imf(mass);
}

/* Compute factor to normalise the total mass in the IMF to unity.*/
double compute_imf_norm(gsl_integration_workspace * gsl_work)
{
    double norm, abserr;
    gsl_function ff = {chabrier_mass, NULL};
    gsl_integration_qag(&ff, MINMASS, MAXMASS, 1e-4, 1e-3, GSL_WORKSPACE, GSL_INTEG_GAUSS61, gsl_work, &norm, &abserr);
    return norm;
}

/* Compute number of Sn1a: has units of N0 = 1.3e-3, which is SN1A/(unit initial mass in M_sun).
 * Zero for age < 40 Myr. */
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
    /* This is the integral of the DTD, normalised to the N0 rate which is in SN/M_sun.*/
    double Nsn1a = MetalParams.Sn1aN0 /totalSN1a * (pow(dtmyrstart / tau8msun, 1-sn1aindex) - pow(dtmyrend / tau8msun, 1-sn1aindex));
    return Nsn1a;
}

/* Compute yield of AGB stars: this is normalised to the yield which has units of Msun / (unit Msun in the initial SSP and so is really dimensionless.)*/
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
    if (stellarmetal > agb_metallicities[AGB_NMET-1])
        stellarmetal = agb_metallicities[AGB_NMET-1];
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
static double mass_yield(double dtmyrstart, double dtmyrend, double stellarmetal, double hub, struct interps * interp, double imf_norm, gsl_integration_workspace * gsl_work, double *masslow, double * masshigh)
{
    find_mass_bin_limits(masslow, masshigh, dtmyrstart, dtmyrend, stellarmetal, interp->lifetime_interp);
    /* Number of AGB stars/SnII by integrating the IMF*/
    double agbyield = compute_agb_yield(interp->agb_mass_interp, agb_total_mass, stellarmetal, *masslow, *masshigh, gsl_work);
    double sniiyield = compute_snii_yield(interp->snii_mass_interp, snii_total_mass, stellarmetal, *masslow, *masshigh, gsl_work);
    /* Fraction of the IMF which goes off this timestep. Normalised by the total IMF so we get a fraction of the SSP.*/
    double massyield = (agbyield + sniiyield)/imf_norm;
    /* Mass yield from Sn1a*/
    double Nsn1a = sn1a_number(dtmyrstart, dtmyrend, hub);
    massyield += Nsn1a * sn1a_total_metals;
    //message(3, "masslow %g masshigh %g stellarmetal %g dystart %g dtend %g agb %g snii %g sn1a %g imf_norm %g\n",
    //        masslow, masshigh, stellarmetal, dtmyrstart, dtmyrend, agbyield, sniiyield, Nsn1a * sn1a_total_metals, imf_norm);
    return massyield;
}

/* Compute the total metal yield for this star in this timestep*/
static double metal_yield(double dtmyrstart, double dtmyrend, double stellarmetal, double hub, struct interps * interp, MyFloat * MetalYields, double imf_norm, gsl_integration_workspace * gsl_work, double masslow, double masshigh)
{
    double MetalGenerated = 0;
    /* Number of AGB stars/SnII by integrating the IMF*/
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
metal_return(const ActiveParticles * act, const ForceTree * const tree, Cosmology * CP, const double atime)
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

    int nthread = omp_get_max_threads();
    gsl_integration_workspace ** gsl_work = ta_malloc("gsl_work", gsl_integration_workspace *, nthread);
    int i;
    /* Allocate a workspace for each thread*/
    for(i=0; i < nthread; i++)
        gsl_work[i] = gsl_integration_workspace_alloc(GSL_WORKSPACE);

    /* Initialize*/
    setup_metal_table_interp(&METALS_GET_PRIV(tw)->interp);
    priv->StellarAges = mymalloc("StellarAges", SlotsManager->info[4].size * sizeof(MyFloat));
    priv->MassReturn = mymalloc("MassReturn", SlotsManager->info[4].size * sizeof(MyFloat));
    priv->LowDyingMass = mymalloc("LowDyingMass", SlotsManager->info[4].size * sizeof(MyFloat));
    priv->HighDyingMass = mymalloc("HighDyingMass", SlotsManager->info[4].size * sizeof(MyFloat));
    priv->imf_norm = compute_imf_norm(gsl_work[0]);

    /* First find the mass return as a fraction of the total mass and the age of the star.
     * This is done first so we can skip density computation for not active stars*/
    #pragma omp parallel for
    for(i=0; i < act->NumActiveParticle;i++)
    {
        int p_i = act->ActiveParticle ? act->ActiveParticle[i] : i;
        if(P[p_i].Type != 4)
            continue;
        int tid = omp_get_thread_num();
        const int slot = P[p_i].PI;
        priv->StellarAges[slot] = atime_to_myr(CP, STARP(p_i).FormationTime, atime, gsl_work[tid]);
        /* Note this takes care of units*/
        double initialmass = P[p_i].Mass + STARP(p_i).TotalMassReturned;
        priv->MassReturn[slot] = initialmass * mass_yield(STARP(p_i).LastEnrichmentMyr, priv->StellarAges[P[p_i].PI], STARP(p_i).Metallicity, CP->HubbleParam, &priv->interp, priv->imf_norm, gsl_work[tid],&priv->LowDyingMass[slot], &priv->HighDyingMass[slot]);
        //message(3, "Particle %d PI %d massgen %g mass %g initmass %g\n", p_i, P[p_i].PI, priv->MassReturn[P[p_i].PI], P[p_i].Mass, initialmass);
        /* Guard against making a zero mass particle and warn since this should not happen.*/
        if(priv->MassReturn[slot] > 0.9 * P[p_i].Mass) {
            message(1, "Large mass return %g from %d mass %g initial %g\n", priv->MassReturn[slot], p_i, P[p_i].Mass, initialmass);
            priv->MassReturn[slot] = 0.9 * P[p_i].Mass;
        }
    }

    /* Compute total number of weights around each star for actively returning stars*/
    METALS_GET_PRIV(tw)->StarVolumeSPH = stellar_density(act, priv->StellarAges, priv->MassReturn, tree);
    priv->gsl_work = gsl_work;
    message(0, "Starting metal return treewalk\n");
    /* Do the metal return*/
    priv->spin = init_spinlocks(SlotsManager->info[0].size);
    treewalk_run(tw, act->ActiveParticle, act->NumActiveParticle);
    free_spinlocks(priv->spin);

    myfree(priv->StarVolumeSPH);
    myfree(priv->HighDyingMass);
    myfree(priv->LowDyingMass);
    myfree(priv->MassReturn);
    myfree(priv->StellarAges);

    for(i=0; i < nthread; i++)
        gsl_integration_workspace_free(gsl_work[i]);

    ta_free(gsl_work);

    /* collect some timing information */
    walltime_measure("/SPH/Metals/Yield");
}

/* This function is unusually important:
 * it computes the total amount of metals to be returned in this timestep.*/
static void
metal_return_copy(int place, TreeWalkQueryMetals * input, TreeWalk * tw)
{
    input->Metallicity = STARP(place).Metallicity;
    input->Mass = P[place].Mass;
    input->Hsml = P[place].Hsml;
    int pi = P[place].PI;
    input->StarVolumeSPH = METALS_GET_PRIV(tw)->StarVolumeSPH[pi];
    double InitialMass = P[place].Mass + STARP(place).TotalMassReturned;
    double dtmyrend = METALS_GET_PRIV(tw)->StellarAges[pi];
    double dtmyrstart = STARP(place).LastEnrichmentMyr;
    int tid = omp_get_thread_num();
    /* This is the total mass returned from this stellar population this timestep. Note this is already in the desired units.*/
    input->MassGenerated = METALS_GET_PRIV(tw)->MassReturn[pi];
    /* This returns the total amount of metal produced this timestep, and also fills out MetalSpeciesGenerated, which is an
     * element by element table of the metal produced by dying stars this timestep.*/
    double total_z_yield = metal_yield(dtmyrstart, dtmyrend, input->Metallicity, METALS_GET_PRIV(tw)->hub, &METALS_GET_PRIV(tw)->interp, input->MetalSpeciesGenerated, METALS_GET_PRIV(tw)->imf_norm, METALS_GET_PRIV(tw)->gsl_work[tid], METALS_GET_PRIV(tw)->LowDyingMass[pi], METALS_GET_PRIV(tw)->HighDyingMass[pi]);
    /* The total metal returned is the metal created this timestep, plus the metal which was already in the mass returned by the dying stars.*/
    input->MetalGenerated = InitialMass * total_z_yield + STARP(place).Metallicity * input->MassGenerated;
    //message(3, "Particle %d PI %d z %g massgen %g metallicity %g\n", pi, P[pi].PI, total_z_yield, METALS_GET_PRIV(tw)->MassReturn[pi], STARP(place).Metallicity);
    /* It should be positive! If it is not, this is some integration error
     * in the yield table as we cannot destroy metal which is not present.*/
    if(input->MetalGenerated < 0)
        input->MetalGenerated = 0;
    /* Similarly for all the other metal species*/
    int i;
    for(i = 0; i < NMETALS; i++) {
        input->MetalSpeciesGenerated[i] = InitialMass * input->MetalSpeciesGenerated[i] + STARP(place).Metals[i] * input->MassGenerated;
        if(input->MetalSpeciesGenerated[i] < 0)
            input->MetalSpeciesGenerated[i] = 0;
    }
}

/* Update the mass and enrichment variables for the star.
 * Note that the stellar metallicity is not updated, as the
 * metal-forming stars are now dead and their metals in the gas.*/
static void
metal_return_postprocess(int place, TreeWalk * tw)
{
    /* Conserve mass returned*/
    P[place].Mass -= METALS_GET_PRIV(tw)->MassReturn[P[place].PI];
    STARP(place).TotalMassReturned += METALS_GET_PRIV(tw)->MassReturn[P[place].PI];
    /* Update the last enrichment time*/
    STARP(place).LastEnrichmentMyr = METALS_GET_PRIV(tw)->StellarAges[P[place].PI];
}

/*! For all gas particles within the density radius of this star,
 * add a fraction of the total mass and metals generated,
 * weighted by the SPH kernel distance from the star.
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

    const int other = iter->base.other;
    const double r2 = iter->base.r2;
    const double r = iter->base.r;

    if(P[other].Mass == 0) {
        endrun(12, "Encountered zero mass particle during hydro;"
                  " We haven't implemented tracer particles and this shall not happen\n");
    }

    /* Wind particles do not interact hydrodynamically: don't receive metal mass.*/
    if(winds_is_particle_decoupled(other))
        return;

    if(r2 > 0 && r2 < iter->kernel.HH)
    {
        double wk = 1;
        const double u = r * iter->kernel.Hinv;

        if(MetalParams.SPHWeighting)
            wk = density_kernel_wk(&iter->kernel, u);
        double ThisMetals[NMETALS];
        if(I->StarVolumeSPH ==0)
            endrun(3, "StarVolumeSPH %g hsml %g\n", I->StarVolumeSPH, I->Hsml);
        double newmass;
        int pi = P[other].PI;
        lock_spinlock(pi, METALS_GET_PRIV(lv->tw)->spin);
        /* Volume of particle weighted by the SPH kernel*/
        double volume = P[other].Mass / SPHP(other).Density;
        double returnfraction = wk * volume / I->StarVolumeSPH;
        int i;
        for(i = 0; i < NMETALS; i++)
            ThisMetals[i] = returnfraction * I->MetalSpeciesGenerated[i];
        /* Keep track of how much was returned for conservation purposes*/
        double thismass = returnfraction * I->MassGenerated;
        O->MassReturn += thismass;
        /* Add metals weighted by SPH kernel*/
        double thismetal = returnfraction * I->MetalGenerated;
        /* Add the metals to the particle.*/
        for(i = 0; i < NMETALS; i++)
            SPHP(other).Metals[i] = (SPHP(other).Metals[i] * P[other].Mass + ThisMetals[i])/(P[other].Mass + thismass);
        /* Update total metallicity*/
        SPHP(other).Metallicity = (SPHP(other).Metallicity * P[other].Mass + thismetal)/(P[other].Mass + thismass);
        /* Update mass*/
        P[other].Mass += thismass;
        newmass = P[other].Mass;
        unlock_spinlock(pi, METALS_GET_PRIV(lv->tw)->spin);
        if(newmass <= 0)
            endrun(3, "New mass %g new metal %g in particle %d id %ld from star mass %g metallicity %g\n",
                   newmass, SPHP(other).Metallicity, other, P[other].ID, I->Mass, I->Metallicity);
    }
    O->Ninteractions++;
}

/* Find stars returning enough metals to the gas.
 * This is a wrapper function to allow for
 * different private structs in different treewalks*/
static int
metals_haswork(int i, MyFloat * StellarAges, MyFloat * MassReturn)
{
    if(P[i].Type != 4)
        return 0;
    int pi = P[i].PI;
    /* New stars or stars with zero mass return will not do anything: nothing has yet died*/
    if(StellarAges[pi] == 0 || MassReturn[pi] == 0)
        return 0;
    /* Don't do enrichment from all stars, just young stars or those with significant enrichment*/
    int young = StellarAges[pi] < 100;
    int massreturned = MassReturn[pi] > 1e-3 * P[i].Mass;
    return young || massreturned;
}

static int
metal_return_haswork(int i, TreeWalk * tw)
{
    return metals_haswork(i, METALS_GET_PRIV(tw)->StellarAges, METALS_GET_PRIV(tw)->MassReturn);
}

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
    /* For haswork*/
    MyFloat * StellarAges;
    MyFloat * MassReturn;
    /*!< Desired number of SPH neighbours */
    double DesNumNgb;
};

#define STELLAR_DENSITY_GET_PRIV(tw) ((struct StellarDensityPriv*) ((tw)->priv))

static int
stellar_density_haswork(int i, TreeWalk * tw)
{
    return metals_haswork(i, STELLAR_DENSITY_GET_PRIV(tw)->StellarAges, STELLAR_DENSITY_GET_PRIV(tw)->MassReturn);
}

static void
stellar_density_copy(int place, TreeWalkQueryStellarDensity * I, TreeWalk * tw)
{
    I->Hsml = P[place].Hsml;
}

static void
stellar_density_reduce(int place, TreeWalkResultStellarDensity * remote, enum TreeWalkReduceMode mode, TreeWalk * tw)
{
    int pi = P[place].PI;
    TREEWALK_REDUCE(STELLAR_DENSITY_GET_PRIV(tw)->NumNgb[pi], remote->Ngb);
    TREEWALK_REDUCE(STELLAR_DENSITY_GET_PRIV(tw)->VolumeSPH[pi], remote->VolumeSPH);
}

void stellar_density_check_neighbours (int i, TreeWalk * tw)
{
    /* now check whether we had enough neighbours */

    double desnumngb = STELLAR_DENSITY_GET_PRIV(tw)->DesNumNgb;

    MyFloat * Left = STELLAR_DENSITY_GET_PRIV(tw)->Left;
    MyFloat * Right = STELLAR_DENSITY_GET_PRIV(tw)->Right;
    MyFloat * NumNgb = STELLAR_DENSITY_GET_PRIV(tw)->NumNgb;

    int pi = P[i].PI;

    if(NumNgb[pi] < (desnumngb - 2) ||
            (NumNgb[pi] > (desnumngb + 2)))
    {
        /* This condition is here to prevent the density code looping forever if it encounters
         * multiple particles at the same position. If this happens you likely have worse
         * problems anyway, so warn also. */
        if((Right[pi] - Left[pi]) < 1.0e-4 * Left[pi])
        {
            /* If this happens probably the exchange is screwed up and all your particles have moved to (0,0,0)*/
            message(1, "Very tight Hsml bounds for i=%d ID=%lu type %d Hsml=%g Left=%g Right=%g Ngbs=%g des = %g Right-Left=%g pos=(%g|%g|%g)\n",
             i, P[i].ID, P[i].Type, P[i].Hsml, Left[pi], Right[pi], NumNgb[pi], desnumngb, Right[pi] - Left[pi], P[i].Pos[0], P[i].Pos[1], P[i].Pos[2]);
            P[i].Hsml = Right[pi];
            return;
        }

        /* If we need more neighbours, move the lower bound up. If we need fewer, move the upper bound down.*/
        if(NumNgb[pi] < desnumngb) {
                Left[pi] = P[i].Hsml;
        } else {
                Right[pi] = P[i].Hsml;
        }

        /* Next step is geometric mean of previous. */
        if((Right[pi] < tw->tree->BoxSize && Left[pi] > 0) || (P[i].Hsml * 1.26 > 0.99 * tw->tree->BoxSize))
            P[i].Hsml = pow(0.5 * (pow(Left[pi], 3) + pow(Right[pi], 3)), 1.0 / 3);
        else
        {
            if(!(Right[pi] < tw->tree->BoxSize) && Left[pi] == 0)
                endrun(8188, "Cannot occur. Check for memory corruption: i=%d pi %d L = %g R = %g N=%g. Type %d, Pos %g %g %g",
                       i, pi, Left[pi], Right[pi], NumNgb[pi], P[i].Type, P[i].Pos[0], P[i].Pos[1], P[i].Pos[2]);

            /* If this is the first step we can be faster by increasing or decreasing current Hsml by a constant factor*/
            if(Right[pi] > 0.99 * tw->tree->BoxSize && Left[pi] > 0)
                P[i].Hsml *= 1.26;

            if(Right[pi] < 0.99*tw->tree->BoxSize && Left[pi] == 0)
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
             i, P[i].ID, P[i].Hsml, Left[pi], Right[pi],
             NumNgb[pi], Right[pi] - Left[pi], P[i].Pos[0], P[i].Pos[1], P[i].Pos[2]);
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

    /* Wind particles do not interact hydrodynamically: don't receive metal mass.*/
    if(winds_is_particle_decoupled(other))
        return;

    if(r2 < iter->kernel.HH)
    {
        const double u = r * iter->kernel.Hinv;
        double wk = density_kernel_wk(&iter->kernel, u);
        O->Ngb += wk * iter->kernel_volume;
        /* For stars we need the total weighting, sum(w_k m_k / rho_k).*/
        double thisvol = P[other].Mass / SPHP(other).Density;
        if(MetalParams.SPHWeighting)
            thisvol *= wk;
        O->VolumeSPH += thisvol;
    }
}

MyFloat *
stellar_density(const ActiveParticles * act, MyFloat * StellarAges, MyFloat * MassReturn, const ForceTree * const tree)
{
    TreeWalk tw[1] = {{0}};
    struct StellarDensityPriv priv[1];

    tw->ev_label = "STELLAR_DENSITY";
    tw->visit = (TreeWalkVisitFunction) treewalk_visit_ngbiter;
    tw->ngbiter_type_elsize = sizeof(TreeWalkNgbIterStellarDensity);
    tw->ngbiter = (TreeWalkNgbIterFunction) stellar_density_ngbiter;
    tw->haswork = stellar_density_haswork;
    tw->fill = (TreeWalkFillQueryFunction) stellar_density_copy;
    tw->reduce = (TreeWalkReduceResultFunction) stellar_density_reduce;
    tw->postprocess = (TreeWalkProcessFunction) stellar_density_check_neighbours;
    tw->query_type_elsize = sizeof(TreeWalkQueryStellarDensity);
    tw->result_type_elsize = sizeof(TreeWalkResultStellarDensity);
    tw->priv = priv;
    tw->tree = tree;

    int i;
    int64_t ntot = 0;

    priv->StellarAges = StellarAges;
    priv->MassReturn = MassReturn;
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

    walltime_measure("/SPH/Metals/Init");
    /* allocate buffers to arrange communication */
    int NumThreads = omp_get_max_threads();
    priv->NPLeft = ta_malloc("NPLeft", size_t, NumThreads);
    priv->NPRedo = ta_malloc("NPRedo", int *, NumThreads);
    int alloc_high = 0;
    int * ReDoQueue = act->ActiveParticle;
    int size = act->NumActiveParticle;

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
        /* Skip the garbage particles */
        if(P[a].IsGarbage)
            continue;
        if(stellar_density_haswork(a, tw) && priv->VolumeSPH[P[a].PI] == 0)
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
