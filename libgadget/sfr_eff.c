/***
 * Multi-Phase star formation
 *
 * The algorithm here is based on Springel Hernequist 2003, and Okamoto 2010.
 *
 * The source code originally came from sfr_eff.c in Gadget-3. This version has
 * been heavily rewritten to add support for new wind models, new star formation
 * criterions, and more importantly, use the new tree walker routines.
 *
 * I (Yu Feng) feel it is appropriate to release this file with a free license,
 * because the implementation here has diverged from the original code by too far.
 *
 * Functions for self-gravity starformation condition and H2 are derived from Gadget-P
 * and used with permission of Phil Hopkins. Please cite the requisite papers if you use them.
 *
 * */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include "libgadget/timebinmgr.h"
#include "physconst.h"
#include "sfr_eff.h"
#include "cooling.h"
#include "slotsmanager.h"
#include "walltime.h"
#include "winds.h"
/*Only for the star slot reservation*/
#include "forcetree.h"
#include "domain.h"
#include "utils/endrun.h"
#include "utils/mymalloc.h"

/*Parameters of the star formation model*/
static struct SFRParams
{
    /* Master switch enabling star formation*/
    int StarformationOn;
    enum StarformationCriterion StarformationCriterion;  /*!< Type of star formation model. */
    int WindOn; /* if Wind is enabled */
    /*Star formation parameters*/
    double CritOverDensity;
    double CritPhysDensity;
    double OverDensThresh;
    double PhysDensThresh;
    double EgySpecSN;
    double FactorSN;
    double EgySpecCold;
    double FactorEVP;
    double TempSupernova;
    double TempClouds;
    double MaxSfrTimescale;
    int BHFeedbackUseTcool;
    /*!< may be used to set a floor for the gas temperature */
    double MinGasTemp;

    /* Unit conversion factor for the sfr_due_to_h2 function*/
    double tau_fmol_unit;
    /*Lyman alpha forest specific star formation.*/
    double QuickLymanAlphaProbability;
    double QuickLymanAlphaTempThresh;
    /* Number of stars to create from each gas particle*/
    int Generations;
    /* Average starting mass for a gas particle.*/
    double avg_baryon_mass;
    /* U = temp_to_u / meanweight  * T
     * temp_to_u = (1.0 / GAMMA_MINUS1) * (BOLTZMANN / PROTONMASS) / (UnitEnergy_in_cgs / UnitMass_in_g)*/
    double temp_to_u;
    /* COnversion factor from internal SFR units to solar masses per year*/
    double UnitSfr_in_solar_per_year;
    /* The temperature boost from reionisation. Following 1807.09282,
     * we use a fixed, density independent value of 20000 K. I also tried
     * their eq. 3-4 but found that for this low resolution (of the UV grid) the
     * density gradients were too small and Treion was only 10 K.
     */
    double HIReionTemp;
    /* Input files for the various cooling modules*/
    char TreeCoolFile[100];
    char J21CoeffFile[100];
    char MetalCoolFile[100];
    char UVFluctuationFile[100];
    /* Boost SF for dense gas*/
    int BoostSFDenseGas;
    double BoostSFOverDenseFactor;
    /* File with the helium reionization table*/
    char ReionHistFile[100];
} sfr_params;

int get_generations(void)
{
    return sfr_params.Generations;
}

/* Structure storing the results of an evaluation of the star formation model*/
struct sfr_eeqos_data
{
    /* Relaxation time*/
    double trelax;
    /* Star formation timescale*/
    double tsfr;
    /* Internal energy of the gas in the hot phase. */
    double egyhot;
    /* Internal energy of the gas in the cold phase.*/
    double egycold;
    /* Fraction of the gas in the cold cloud phase. */
    double cloudfrac;
    /* Electron fraction after cooling. */
    double ne;
};

/* Computes properties of the gas on star forming equation of state*/
static struct sfr_eeqos_data get_sfr_eeqos(struct particle_data * part, struct sph_particle_data * sph, double dtime, struct UVBG *local_uvbg, const double redshift, const double a3inv);

/*Cooling only: no star formation*/
static void cooling_direct(int i, const double redshift, const double a3inv, const double hubble, const struct UVBG * const GlobalUVBG);

static void cooling_relaxed(int i, double dtime, struct UVBG * local_uvbg, const double redshift, const double a3inv, struct sfr_eeqos_data sfr_data, const struct UVBG * const GlobalUVBG);

/* Update the active particle list when a new star is formed.*/
static int add_new_particle_to_active(const int parent, const int child, ActiveParticles * act);
static int copy_gravaccel_new_particle(const int parent, const int child, MyFloat (* GravAccel)[3], int64_t nstoredgravaccel);

static int make_particle_star(int child, int parent, int placement, double Time);
static int starformation(int i, double *localsfr, MyFloat * sm_out, MyFloat * sum_sm, MyFloat * sum_dtime, MyFloat * GradRho, const double redshift, const double a3inv, const double hubble, const double GravInternal, const struct UVBG * const GlobalUVBG, const RandTable * const rnd);
static int quicklyastarformation(int i, const double a3inv, const RandTable * const rnd);
static double get_sfr_factor_due_to_selfgravity(int i, const double atime, const double a3inv, const double hubble, const double GravInternal);
static double get_sfr_factor_due_to_h2(int i, MyFloat * GradRho_mag, const double atime);
static double get_starformation_rate_full(int i, MyFloat * GradRho, struct sfr_eeqos_data sfr_data, const double atime, const double a3inv, const double hubble, const double GravInternal);
static double get_egyeff(double redshift, double dens, struct UVBG * uvbg);
static double find_star_mass(int i, const double avg_baryon_mass);
/*Get enough memory for new star slots. This may be excessively slow! Don't do it too often.*/
static int * sfr_reserve_slots(ActiveParticles * act, int * NewStars, int NumNewStar, ForceTree * tt);

/* Convert entropy to internal energy*/
static double entropy_to_u(const double density, const double a3inv)
{
    return exp(GAMMA_MINUS1 * log(density * a3inv))/GAMMA_MINUS1;
//     return pow(density * a3inv, GAMMA_MINUS1) / GAMMA_MINUS1;
}

/*Set the parameters of the SFR module*/
void set_sfr_params(ParameterSet * ps)
{
    int ThisTask;
    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
    if(ThisTask == 0) {
        /*Star formation parameters*/
        sfr_params.StarformationCriterion = (enum StarformationCriterion) param_get_enum(ps, "StarformationCriterion");
        sfr_params.CritOverDensity = param_get_double(ps, "CritOverDensity");
        sfr_params.CritPhysDensity = param_get_double(ps, "CritPhysDensity");
        sfr_params.WindOn = param_get_int(ps, "WindOn");
        sfr_params.BoostSFDenseGas = param_get_int(ps, "BoostSFDenseGas");
        sfr_params.BoostSFOverDenseFactor = param_get_double(ps, "BoostSFOverDenseFactor");

        sfr_params.FactorSN = param_get_double(ps, "FactorSN");
        sfr_params.FactorEVP = param_get_double(ps, "FactorEVP");
        sfr_params.TempSupernova = param_get_double(ps, "TempSupernova");
        sfr_params.TempClouds = param_get_double(ps, "TempClouds");
        sfr_params.MaxSfrTimescale = param_get_double(ps, "MaxSfrTimescale");
        sfr_params.Generations = param_get_int(ps, "Generations");
        if(sfr_params.Generations > 14)
            endrun(0, "Generations is %d, only space in the bitfield for 14.\n", sfr_params.Generations);
        sfr_params.MinGasTemp = param_get_double(ps, "MinGasTemp");
        sfr_params.BHFeedbackUseTcool = param_get_int(ps, "BHFeedbackUseTcool");
        if(sfr_params.BHFeedbackUseTcool > 3 || sfr_params.BHFeedbackUseTcool < 0)
            endrun(0, "BHFeedbackUseTcool mode %d not supported\n", sfr_params.BHFeedbackUseTcool);
        /*Lyman-alpha forest parameters*/
        sfr_params.QuickLymanAlphaProbability = param_get_double(ps, "QuickLymanAlphaProbability");
        sfr_params.QuickLymanAlphaTempThresh = param_get_double(ps, "QuickLymanAlphaTempThresh");
        sfr_params.HIReionTemp = param_get_double(ps, "HIReionTemp");

        /* File names*/
        param_get_string2(ps, "TreeCoolFile", sfr_params.TreeCoolFile, sizeof(sfr_params.TreeCoolFile));
        param_get_string2(ps, "J21CoeffFile", sfr_params.J21CoeffFile, sizeof(sfr_params.J21CoeffFile));
        param_get_string2(ps, "UVFluctuationfile", sfr_params.UVFluctuationFile, sizeof(sfr_params.UVFluctuationFile));
        param_get_string2(ps, "MetalCoolFile", sfr_params.MetalCoolFile, sizeof(sfr_params.MetalCoolFile));
        param_get_string2(ps, "ReionHistFile", sfr_params.ReionHistFile, sizeof(sfr_params.ReionHistFile));
    }
    MPI_Bcast(&sfr_params, sizeof(struct SFRParams), MPI_BYTE, 0, MPI_COMM_WORLD);
}

/* cooling and star formation routine.*/
void
cooling_and_starformation(ActiveParticles * act, double Time, double dloga, ForceTree * tree, struct grav_accel_store GravAccel, DomainDecomp * ddecomp, Cosmology *CP, MyFloat * GradRho, RandTable * rnd, FILE * FdSfr)
{
    /*This is a queue for the new stars and their parents, so we can reallocate the slots after the main cooling loop.*/
    gadget_thread_arrays NewStarThread = {0}, NewParentThread = {0}, MaybeWindThread = {0};

    /*Need to capture this so that when NumActiveParticle increases during the loop
     * we don't add extra loop iterations on particles with invalid slots.*/
    const int nactive = act->NumActiveParticle;
    const double a3inv = 1./(Time * Time * Time);
    const double hubble = hubble_function(CP, Time);

    if(sfr_params.StarformationOn) {
        /* Maximally we need the active gas particles*/
        NewStarThread = gadget_setup_thread_arrays("NewStars", 0, act->NumActiveHydro);
        NewParentThread = gadget_setup_thread_arrays("NewParents", 1, act->NumActiveHydro);
    }

    MyFloat * StellarMass = NULL;
    if(sfr_params.WindOn && winds_are_subgrid()) {
        StellarMass = (MyFloat *) mymalloc("StellarMass", SlotsManager->info[0].size * sizeof(MyFloat));
        MaybeWindThread = gadget_setup_thread_arrays("MaybeWind", 0, act->NumActiveHydro);
    }

    /* Get the global UVBG for this redshift. */
    const double redshift = 1./Time - 1;
    struct UVBG GlobalUVBG = get_global_UVBG(redshift);
    double sum_sm = 0, sum_mass_stars = 0, localsfr = 0, sum_dtime = 0;
    int64_t sum_sf_part = 0;

    /* First decide which stars are cooling and which starforming. If star forming we add them to a list.
     * Note the dynamic scheduling: individual particles may have very different loop iteration lengths.
     * Cooling is much slower than sfr. I tried splitting it into a separate loop instead, but this was faster.*/
    #pragma omp parallel reduction(+:localsfr) reduction(+: sum_sm) reduction(+:sum_mass_stars) reduction(+:sum_dtime) reduction(+:sum_sf_part)
    {
        int i;
        const int tid = omp_get_thread_num();
        #pragma omp for schedule(static)
        for(i=0; i < nactive; i++)
        {
            /*Use raw particle number if active_set is null, otherwise use active_set*/
            const int p_i = act->ActiveParticle ? act->ActiveParticle[i] : i;
            /* Skip non-gas or garbage particles */
            if(P[p_i].Type != 0 || P[p_i].IsGarbage || P[p_i].Mass <= 0)
                continue;

            int shall_we_star_form = 0;
            if(sfr_params.StarformationOn) {
                /*Reduce delaytime for wind particles.*/
                winds_evolve(p_i, a3inv, hubble);
                /* check whether we are star forming gas.*/
                if(sfr_params.QuickLymanAlphaProbability > 0)
                    shall_we_star_form = quicklyastarformation(p_i, a3inv, rnd);
                else
                    shall_we_star_form = sfreff_on_eeqos(&SPHP(p_i), a3inv);
            }

            if(shall_we_star_form) {
                int newstar = -1;
                MyFloat sm = 0;
                sum_sf_part++;
                if(sfr_params.QuickLymanAlphaProbability > 0) {
                    /*New star is always the same particle as the parent for quicklya*/
                    newstar = p_i;
                    sum_sm += P[p_i].Mass;
                    sm = P[p_i].Mass;
                } else {
                    newstar = starformation(p_i, &localsfr, &sm, &sum_sm, &sum_dtime, GradRho, redshift, a3inv, hubble, CP->GravInternal, &GlobalUVBG, rnd);
                }
                /*Add this particle to the stellar conversion queue if necessary.*/
                if(newstar >= 0) {
                    NewStarThread.srcs[tid][NewStarThread.sizes[tid]] = newstar;
                    NewStarThread.sizes[tid]++;
                    NewParentThread.srcs[tid][NewParentThread.sizes[tid]] = p_i;
                    NewParentThread.sizes[tid]++;
                }
                /* Add this particle to the queue for consideration to spawn a wind.
                 * Only for subgrid winds. */
                if(MaybeWindThread.sizes && newstar < 0) {
                    MaybeWindThread.srcs[tid][MaybeWindThread.sizes[tid]] = p_i;
                    StellarMass[P[p_i].PI] = sm;
                    MaybeWindThread.sizes[tid]++;
                }
            }
            else
                cooling_direct(p_i, redshift, a3inv, hubble, &GlobalUVBG);
        }
    }

    report_memory_usage("SFR");

    walltime_measure("/Cooling/Cooling");

    /* Do subgrid winds*/
    if(sfr_params.WindOn && winds_are_subgrid()) {
        int * MaybeWind;
        int64_t NumMaybeWind = gadget_compact_thread_arrays(&MaybeWind, &MaybeWindThread);
        winds_subgrid(MaybeWind, NumMaybeWind, Time, StellarMass, rnd);
        myfree(MaybeWind);
        myfree(StellarMass);
    }

    int * NewStars = NewStarThread.dest;
    int * NewParents = NewParentThread.dest;
    int64_t NumNewStar = 0;

    /*Merge step for the queue.*/
    if(NewStars) {
        int64_t NumNewParent = gadget_compact_thread_arrays(&NewParents, &NewParentThread);
        NumNewStar = gadget_compact_thread_arrays(&NewStars, &NewStarThread);
        if(NumNewStar != NumNewParent)
            endrun(3,"%lu new stars, but %lu new parents!\n",NumNewStar, NumNewParent);
        /*Shrink star memory as we keep it for the wind model*/
        NewStars = (int *) myrealloc(NewStars, sizeof(int) * NumNewStar);
    }

    if(!sfr_params.StarformationOn)
        return;

    /*Get some empty slots for the stars*/
    int firststarslot = SlotsManager->info[4].size;
    /* We ran out of slots! We must be forming a lot of stars.
     * There are things in the way of extending the slot list, so we have to move them.
     * The code in sfr_reserve_slots is not elegant, but I cannot think of a better way.*/
    if(sfr_params.StarformationOn && (SlotsManager->info[4].size + NumNewStar >= SlotsManager->info[4].maxsize)) {
        if(NewParents)
            NewParents = (int *) myrealloc(NewParents, sizeof(int) * NumNewStar);
        NewStars = sfr_reserve_slots(act, NewStars, NumNewStar, tree);
    }
    SlotsManager->info[4].size += NumNewStar;

    int64_t stars_converted = 0, stars_spawned = 0, stars_spawned_gravity = 0;
    int i;

    /*Now we turn the particles into stars*/
    #pragma omp parallel for schedule(static) reduction(+:stars_converted) reduction(+:stars_spawned) reduction(+:sum_mass_stars) reduction(+:stars_spawned_gravity)
    for(i=0; i < NumNewStar; i++)
    {
        int child = NewStars[i];
        int parent = NewParents[i];
        make_particle_star(child, parent, firststarslot+i, Time);
        sum_mass_stars += P[child].Mass;
        if(child == parent)
            stars_converted++;
        else {
            /* Update the active particle list when a new star is formed.*/
            stars_spawned_gravity += add_new_particle_to_active(parent, child, act);
            copy_gravaccel_new_particle(parent, child, GravAccel.GravAccel, GravAccel.nstore);
            stars_spawned++;
        }
    }
    act->NumActiveGravity += stars_spawned_gravity;

    /*Done with the parents*/
    myfree(NewParents);

    double total_sum_mass_stars = 0, total_sm = 0, totsfrrate = 0, total_sum_dtime = 0;
    int64_t total_sum_part = 0;

    MPI_Reduce(&localsfr, &totsfrrate, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&sum_sm, &total_sm, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&sum_mass_stars, &total_sum_mass_stars, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&sum_dtime, &total_sum_dtime, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&sum_sf_part, &total_sum_part, 1, MPI_INT64, MPI_SUM, 0, MPI_COMM_WORLD);
    if(FdSfr && total_sm > 0)
    {
        double rate = 0;

        if(total_sum_dtime > 0)
            rate = total_sm * total_sum_part / total_sum_dtime;

        /* convert to solar masses per yr */
        double rate_in_msunperyear = rate * sfr_params.UnitSfr_in_solar_per_year;

        /* Format:
         * Time = current scale factor,
         * total_sm = expected change in stellar mass this timestep.
         * This is: sigma_i dM_* = p_* M_* = M_i (1 - exp(-sm_i / M_i))
         * totsfrrate = current star formation rate in active particles in Msun/year.
         * This is: sigma_i dM_* / dt_i
         * rate_in_msunperyear = expected stellar mass formation rate in Msun/year from total_sm,
         * This is sigma_i dM_* / mean(dt_i), where dt_i is over the starforming particles, and may be
         * moderately different from columns 1 and 2.
         * total_sum_mass_stars = actual mass of stars formed this timestep (discretized total_sm).
         * This should be a noisier version of total_sm. */
        fprintf(FdSfr, "%g %g %g %g %g\n", Time, total_sm, totsfrrate, rate_in_msunperyear,
                total_sum_mass_stars);
        fflush(FdSfr);
    }

    int64_t tot_spawned=0, tot_converted=0;
    MPI_Reduce(&stars_spawned, &tot_spawned, 1, MPI_INT64, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&stars_converted, &tot_converted, 1, MPI_INT64, MPI_SUM, 0, MPI_COMM_WORLD);

    if(tot_spawned || tot_converted)
        message(0, "SFR: spawned %ld stars, converted %ld gas particles into stars\n", tot_spawned, tot_converted);

    walltime_measure("/Cooling/StarFormation");

    /* Now apply the wind model using the list of new stars.*/
    if(sfr_params.WindOn && !winds_are_subgrid())
        winds_and_feedback(NewStars, NumNewStar, Time, rnd, tree, ddecomp);
    myfree(NewStars);
}

/* Get enough memory for new star slots. This may be excessively slow! Don't do it too often.
 * It is also not elegant, but I couldn't think of a better way. May be fragile and need updating
 * if memory allocation patterns change. */
static int *
sfr_reserve_slots(ActiveParticles * act, int * NewStars, int NumNewStar, ForceTree * tree)
{
        /* SlotsManager is below Nodes and ActiveParticleList,
         * so we need to move them out of the way before we extend Nodes.
         * This is quite slow, but need not be collective and is faster than a tree rebuild.
         * Try not to do this too often.*/
        message(1, "Need %ld star slots, more than %ld available. Try increasing SlotsIncreaseFactor on restart.\n", SlotsManager->info[4].size, SlotsManager->info[4].maxsize);
        /*Move the NewStar array to upper memory*/
        int * new_star_tmp = NULL;
        if(NewStars) {
            new_star_tmp = (int *) mymalloc2("newstartmp", NumNewStar*sizeof(int));
            memmove(new_star_tmp, NewStars, NumNewStar * sizeof(int));
            myfree(NewStars);
        }
        /*Move the tree to upper memory*/
        struct NODE * nodes_base_tmp=NULL;
        int *Father_tmp=NULL;
        int *ActiveParticle_tmp=NULL;
        if(force_tree_allocated(tree)) {
            nodes_base_tmp = (struct NODE *) mymalloc2("nodesbasetmp", tree->numnodes * sizeof(struct NODE));
            memmove(nodes_base_tmp, tree->Nodes_base, tree->numnodes * sizeof(struct NODE));
            myfree(tree->Nodes_base);
            Father_tmp = (int *) mymalloc2("Father_tmp", PartManager->MaxPart * sizeof(int));
            memmove(Father_tmp, tree->Father, PartManager->MaxPart * sizeof(int));
            myfree(tree->Father);
        }
        if(act->ActiveParticle) {
            ActiveParticle_tmp = (int *) mymalloc2("ActiveParticle_tmp", act->NumActiveParticle * sizeof(int));
            memmove(ActiveParticle_tmp, act->ActiveParticle, act->NumActiveParticle * sizeof(int));
            myfree(act->ActiveParticle);
        }
        /*Now we can extend the slots! */
        int64_t atleast[6];
        int64_t i;
        for(i = 0; i < 6; i++)
            atleast[i] = SlotsManager->info[i].maxsize;
        atleast[4] += NumNewStar;
        slots_reserve(1, atleast, SlotsManager);

        /*And now we need our memory back in the right place*/
        if(ActiveParticle_tmp) {
            act->ActiveParticle = (int *) mymalloc("ActiveParticle", sizeof(int)*(act->NumActiveParticle + PartManager->MaxPart - PartManager->NumPart));
            memmove(act->ActiveParticle, ActiveParticle_tmp, act->NumActiveParticle * sizeof(int));
            myfree(ActiveParticle_tmp);
        }
        if(force_tree_allocated(tree)) {
            tree->Father = (int *) mymalloc("Father", PartManager->MaxPart * sizeof(int));
            memmove(tree->Father, Father_tmp, PartManager->MaxPart * sizeof(int));
            myfree(Father_tmp);
            tree->Nodes_base = (struct NODE *) mymalloc("Nodes_base", tree->numnodes * sizeof(struct NODE));
            memmove(tree->Nodes_base, nodes_base_tmp, tree->numnodes * sizeof(struct NODE));
            myfree(nodes_base_tmp);
            /*Don't forget to update the Node pointer as well as Node_base!*/
            tree->Nodes = tree->Nodes_base - tree->firstnode;
        }
        if(new_star_tmp) {
            NewStars = (int *) mymalloc("NewStars", NumNewStar*sizeof(int));
            memmove(NewStars, new_star_tmp, NumNewStar * sizeof(int));
            myfree(new_star_tmp);
        }
        return NewStars;
}

static void
cooling_direct(int i, const double redshift, const double a3inv, const double hubble, const struct UVBG * const GlobalUVBG)
{
    /*  the actual time-step */
    double dloga = get_dloga_for_bin(P[i].TimeBinHydro, P[i].Ti_drift);
    double dtime = dloga / hubble;

    double ne = SPHP(i).Ne;	/* electron abundance (gives ionization state and mean molecular weight) */

    const double enttou = entropy_to_u(SPHP(i).Density, a3inv);

    /* Current internal energy including adiabatic change*/
    double uold = SPHP(i).Entropy * enttou;
    double localJ21 = 0;
    double zreion = 0;
#ifdef EXCUR_REION
    localJ21 =  SPHP(i).local_J21;
    zreion = SPHP(i).zreion;
#endif
    struct UVBG uvbg = get_local_UVBG(redshift, GlobalUVBG, P[i].Pos, PartManager->CurrentParticleOffset, localJ21, zreion);
    double lasttime = exp(loga_from_ti(P[i].Ti_drift - dti_from_timebin(P[i].TimeBinHydro)));
    double lastred = 1/lasttime - 1;
    double unew;
    /* The particle reionized this timestep, bump the temperature to the HI reionization temperature.
     * We only do this for non-star-forming gas.*/
    if(sfr_params.HIReionTemp > 0 && uvbg.zreion >= redshift && uvbg.zreion < lastred) {
        /* We assume singly ionised helium at the time of reionisation */
        /* The 100% correct thing to do is to solve for the equilibrium ne based on the local UVBG
         * then calculate the mean weight based on this. The current approach will cause
         * a boost in reionisation temperatures proportional to the residual neutral fraction,
         * which should be relatively small most of the time. The 6 is because helium is singly
         * ionized, not doubly so.*/
        /* TODO: Make sure that not setting SPHP.Ne(i) here doesn't mess up anything between
         * now and the next cooling call when it gets set properly */
        const double meanweight = 4 / (8 - 6 * (1 - HYDROGEN_MASSFRAC));
        unew = sfr_params.temp_to_u / meanweight * sfr_params.HIReionTemp;
        //We don't want gas to cool by ionising
        if(uold > unew) unew = uold;
    }
    else {
        /* mean molecular weight assuming ZERO ionization NEUTRAL GAS*/
        const double meanweight = 4.0 / (1 + 3 * HYDROGEN_MASSFRAC);
        const double MinEgySpec = sfr_params.temp_to_u/meanweight * sfr_params.MinGasTemp;
        unew = DoCooling(redshift, uold, SPHP(i).Density * a3inv, dtime, &uvbg, &ne, SPHP(i).Metallicity, MinEgySpec, P[i].HeIIIionized);
    }

    SPHP(i).Ne = ne;
    /* Update the entropy. This is done after synchronizing kicks and drifts, as per run.c.*/
    SPHP(i).Entropy = unew / enttou;
    /* Cooling gas is not forming stars*/
    SPHP(i).Sfr = 0;
}

/* Returns the density threshold for star formation in comoving units*/
double
sfr_density_threshold(const double atime)
{
    double thresh;
    if(sfr_params.QuickLymanAlphaProbability > 0)
        thresh = sfr_params.OverDensThresh;
    else {
        thresh = sfr_params.PhysDensThresh * (atime * atime * atime);
        if(thresh < sfr_params.OverDensThresh)
            thresh = sfr_params.OverDensThresh;
    }
    return thresh;
}

/* returns 1 if the particle is on the effective equation of state,
 * cooling via the relaxation equation and maybe forming stars.
 * 0 if the particle does not form stars, instead cooling normally.*/
int
sfreff_on_eeqos(const struct sph_particle_data * sph, const double a3inv)
{
    int flag = 0;
    /* no sfr: normal cooling*/
    if(!sfr_params.StarformationOn) {
        return 0;
    }

    if(sph->Density * a3inv >= sfr_params.PhysDensThresh)
        flag = 1;

    if(sph->Density < sfr_params.OverDensThresh)
        flag = 0;

    if(sph->DelayTime > 0)
        flag = 0;   /* only normal cooling for particles in the wind */

    /* The model from 0904.2572 makes gas not star forming if more than 0.5 dex above
     * the effective equation of state (at z=0). This in practice means black hole heated.*/
    if(flag == 1 && sfr_params.BHFeedbackUseTcool == 2) {
        //Redshift is the argument
        double redshift = cbrt(a3inv)-1;
        struct UVBG uvbg = get_global_UVBG(redshift);
        double egyeff = get_egyeff(redshift, sph->Density, &uvbg);
        const double enttou = entropy_to_u(sph->Density, a3inv);
        double unew = sph->Entropy * enttou;
        /* 0.5 dex = 10^0.5 = 3.2 */
        if(unew >= egyeff * 3.2)
            flag = 0;
    }
    return flag;
}

/*Get the neutral fraction of a particle correctly, accounting for being on the star-forming equation of state*/
double get_neutral_fraction_sfreff(double redshift, double hubble, struct particle_data * partdata, struct sph_particle_data * sphdata)
{
    double nh0;
    const double a3inv = (1+redshift)*(1+redshift)*(1+redshift);
    struct UVBG GlobalUVBG = get_global_UVBG(redshift);
    double localJ21 = 0;
    double zreion = 0;
#ifdef EXCUR_REION
    localJ21 =  sphdata->local_J21;
    zreion = sphdata->zreion;
#endif
    struct UVBG uvbg = get_local_UVBG(redshift, &GlobalUVBG, partdata->Pos, PartManager->CurrentParticleOffset, localJ21, zreion);
    double physdens = sphdata->Density * a3inv;

    if(sfr_params.QuickLymanAlphaProbability > 0 || !sfreff_on_eeqos(sphdata, a3inv)) {
        /*This gets the neutral fraction for standard gas*/
        double InternalEnergy = sphdata->Entropy * entropy_to_u(sphdata->Density, a3inv);
        nh0 = GetNeutralFraction(InternalEnergy, physdens, &uvbg, sphdata->Ne);
    }
    else {
        /* This gets the neutral fraction for gas on the star-forming equation of state.
         * This needs special handling because the cold clouds have a different neutral
         * fraction than the hot gas*/
        double dloga = get_dloga_for_bin(partdata->TimeBinHydro, partdata->Ti_drift);
        double dtime = dloga / hubble;
        struct sfr_eeqos_data sfr_data = get_sfr_eeqos(partdata, sphdata, dtime, &uvbg, redshift, a3inv);
        double nh0cold = GetNeutralFraction(sfr_params.EgySpecCold, physdens, &uvbg, sfr_data.ne);
        double nh0hot = GetNeutralFraction(sfr_data.egyhot, physdens, &uvbg, sfr_data.ne);
        nh0 =  nh0cold * sfr_data.cloudfrac + (1-sfr_data.cloudfrac) * nh0hot;
    }
    return nh0;
}

double get_helium_neutral_fraction_sfreff(int ion, double redshift, double hubble, struct particle_data * partdata, struct sph_particle_data * sphdata)
{
    const double a3inv = (1+redshift)*(1+redshift)*(1+redshift);
    double helium;
    struct UVBG GlobalUVBG = get_global_UVBG(redshift);
    double localJ21 = 0;
    double zreion = 0;
#ifdef EXCUR_REION
    localJ21 =  sphdata->local_J21;
    zreion = sphdata->zreion;
#endif
    struct UVBG uvbg = get_local_UVBG(redshift, &GlobalUVBG, partdata->Pos, PartManager->CurrentParticleOffset, localJ21, zreion);
    double physdens = sphdata->Density * a3inv;

    if(sfr_params.QuickLymanAlphaProbability > 0 || !sfreff_on_eeqos(sphdata, a3inv)) {
        /*This gets the neutral fraction for standard gas*/
        double InternalEnergy = sphdata->Entropy * entropy_to_u(sphdata->Density, a3inv);
        helium = GetHeliumIonFraction(ion, InternalEnergy, physdens, &uvbg, sphdata->Ne);
    }
    else {
        /* This gets the neutral fraction for gas on the star-forming equation of state.
         * This needs special handling because the cold clouds have a different neutral
         * fraction than the hot gas*/
        double dloga = get_dloga_for_bin(partdata->TimeBinHydro, partdata->Ti_drift);
        double dtime = dloga / hubble;
        struct sfr_eeqos_data sfr_data = get_sfr_eeqos(partdata, sphdata, dtime, &uvbg, redshift, a3inv);
        double nh0cold = GetHeliumIonFraction(ion, sfr_params.EgySpecCold, physdens, &uvbg, sfr_data.ne);
        double nh0hot = GetHeliumIonFraction(ion, sfr_data.egyhot, physdens, &uvbg, sfr_data.ne);
        helium =  nh0cold * sfr_data.cloudfrac + (1-sfr_data.cloudfrac) * nh0hot;
    }
    return helium;
}
/* This function turns a particle into a star. It returns 1 if a particle was
 * converted and 2 if a new particle was spawned. This is used
 * above to set stars_{spawned|converted}*/
static int make_particle_star(int child, int parent, int placement, double Time)
{
    int retflag = 2;
    if(P[parent].Type != 0)
        endrun(7772, "Only gas forms stars, what's wrong?\n");

    /*Store the SPH particle slot properties, as the PI may be over-written
     *in slots_convert*/
    struct sph_particle_data oldslot = SPHP(parent);

    /*Convert the child slot to the new type.*/
    child = slots_convert(child, 4, placement, PartManager, SlotsManager);

    /*Set properties*/
    STARP(child).FormationTime = Time;
    STARP(child).LastEnrichmentMyr = 0;
    STARP(child).TotalMassReturned = 0;
    STARP(child).BirthDensity = oldslot.Density;
    STARP(child).VDisp = oldslot.VDisp;
    /*Copy metallicity*/
    STARP(child).Metallicity = oldslot.Metallicity;
    int j;
    for(j = 0; j < NMETALS; j++)
        STARP(child).Metals[j] = oldslot.Metals[j];

    return retflag;
}

/* This function cools gas on the effective equation of state*/
static void
cooling_relaxed(int i, double dtime, struct UVBG * local_uvbg, const double redshift, const double a3inv, struct sfr_eeqos_data sfr_data, const struct UVBG * const GlobalUVBG)
{
    const double egyeff = sfr_params.EgySpecCold * sfr_data.cloudfrac + (1 - sfr_data.cloudfrac) * sfr_data.egyhot;
    const double densityfac = entropy_to_u(SPHP(i).Density, a3inv);
    double egycurrent = SPHP(i).Entropy * densityfac;
    double trelax = sfr_data.trelax;
    // const double u_to_temp_fac = (4 / (8 - 5 * (1 - HYDROGEN_MASSFRAC))) * PROTONMASS / BOLTZMANN * GAMMA_MINUS1 * sfr_data.uu_in_cgs;
    /* SB: Added a check to cool on the cooling time when the gas is very hot.
     * Gas which is heated by the BH is capped at T=5e8 or U = 1e7.
     * However, the BH is not active each timestep, so rarely very dense gas can have enough internal energy that it doesn't cool in
     * one timestep and can somehow be pressurized to get even hotter. This can make the shortest timestep in the code even shorter,
     * and is unphysical for star forming gas anyway. For this reason, allow gas with U > 5e6 or T > 1e8 to cool using the cooling time. */
    if(sfr_params.BHFeedbackUseTcool == 3 || (sfr_params.BHFeedbackUseTcool == 1 && (P[i].BHHeated || egycurrent > 5e6)))
    {
        if(egycurrent > egyeff)
        {
            double ne = SPHP(i).Ne;
            /* In practice tcool << trelax*/
            double tcool = GetCoolingTime(redshift, egycurrent, SPHP(i).Density * a3inv, local_uvbg, &ne, SPHP(i).Metallicity);

            /* The point of the star-forming equation of state is to pressurize the gas. However,
             * when the gas has been heated above the equation of state it is pressurized and does not cool successfully.
             * This code uses the cooling time rather than the relaxation time.
             * This reduces the effect of black hole feedback marginally (a 5% reduction in star formation)
             * and dates from the earliest versions of this code available.
             * The main impact is on the high end of the black hole mass function: turning this off
             * removes most massive black holes. */
            if(tcool < trelax && tcool > 0)
                trelax = tcool;
        }
        P[i].BHHeated = 0;
    }

    SPHP(i).Entropy =  (egyeff + (egycurrent - egyeff) * exp(-dtime / trelax)) /densityfac;
}

/*Forms stars according to the quick lyman alpha star formation criterion,
 * which forms stars with a constant probability (usually 1) if they are star forming.
 * Returns 1 if converted a particle to a star, 0 if not.*/
static int
quicklyastarformation(int i, const double a3inv, const RandTable * const rnd)
{
    if(SPHP(i).Density <= sfr_params.OverDensThresh)
        return 0;

    const double enttou = entropy_to_u(SPHP(i).Density, a3inv);
    double unew = SPHP(i).Entropy * enttou;

    const double meanweight = (4 / (8 - 5 * (1 - HYDROGEN_MASSFRAC)));
    double temp = unew * meanweight / sfr_params.temp_to_u;

    if(temp >= sfr_params.QuickLymanAlphaTempThresh)
        return 0;

    if(get_random_number(P[i].ID + 1, rnd) < sfr_params.QuickLymanAlphaProbability)
        return 1;

    return 0;
}

/* Forms stars and winds.
 * Returns -1 if no star formed, otherwise returns the index of the particle which is to be made a star.
 * The star slot is not actually created here, but a particle for it is.
 */
static int
starformation(int i, double *localsfr, MyFloat * sm_out, MyFloat * sum_sm, MyFloat * sum_dtime,MyFloat * GradRho, const double redshift, const double a3inv, const double hubble, const double GravInternal, const struct UVBG * const GlobalUVBG, const RandTable * const rnd)
{
    /*  the proper time-step */
    double dloga = get_dloga_for_bin(P[i].TimeBinHydro, P[i].Ti_drift);
    double dtime = dloga / hubble;
    *sum_dtime += dtime;
    int newstar = -1;
    double localJ21 = 0;
    double zreion = 0;
#ifdef EXCUR_REION
    localJ21 =  SPHP(i).local_J21;
    zreion = SPHP(i).zreion;
#endif
    struct UVBG uvbg = get_local_UVBG(redshift, GlobalUVBG, P[i].Pos, PartManager->CurrentParticleOffset, localJ21, zreion);

    struct sfr_eeqos_data sfr_data = get_sfr_eeqos(&P[i], &SPHP(i), dtime, &uvbg, redshift, a3inv);

    double atime = 1/(1+redshift);
    double smr = get_starformation_rate_full(i, GradRho, sfr_data, atime, a3inv, hubble, GravInternal);

    double sm = smr * dtime;

    *sm_out = sm;
    double p = sm / P[i].Mass;

    double dM = P[i].Mass * (1 - exp(-p));
    *sum_sm += dM;

    /* convert to Solar per Year: this is dM_* / dt = p_* M_* / dt ~ smr when smr << 1 */
    SPHP(i).Sfr = dM /dtime * sfr_params.UnitSfr_in_solar_per_year;
    SPHP(i).Ne = sfr_data.ne;
    *localsfr += SPHP(i).Sfr;

    const double w = get_random_number(P[i].ID, rnd);
    const double frac = (1 - exp(-p));
    SPHP(i).Metallicity += w * METAL_YIELD * frac / sfr_params.Generations;

    /* upon start-up, we need to protect against dloga ==0 */
    if(dloga > 0 && P[i].TimeBinHydro)
        cooling_relaxed(i, dtime, &uvbg, redshift, a3inv, sfr_data, GlobalUVBG);

    double mass_of_star = find_star_mass(i, sfr_params.avg_baryon_mass);
    double prob = dM / mass_of_star;

    int form_star = (get_random_number(P[i].ID + 1, rnd) < prob);
    if(form_star) {
        /* ok, make a star */
        newstar = i;
        /* If we get a fraction of the mass we need to create
         * a new particle for the star and remove mass from i.*/
        if(P[i].Mass >= 1.1 * mass_of_star)
            newstar = slots_split_particle(i, mass_of_star, PartManager);
    }

    /* Add the rest of the metals if we didn't form a star.
     * If we did form a star, add winds to the star-forming particle
     * that formed it if it is still around*/
    if(!form_star || newstar != i) {
        SPHP(i).Metallicity += (1-w) * METAL_YIELD * frac / sfr_params.Generations;
    }
    return newstar;
}

/* Get the parameters of the basic effective
 * equation of state model for a particle.*/
struct sfr_eeqos_data get_sfr_eeqos(struct particle_data * part, struct sph_particle_data * sph, double dtime, struct UVBG *local_uvbg, const double redshift, const double a3inv)
{
    struct sfr_eeqos_data data;
    /* Initialise data to something, just in case.*/
    data.trelax = sfr_params.MaxSfrTimescale;
    data.tsfr = sfr_params.MaxSfrTimescale;
    data.egyhot = sfr_params.EgySpecCold;
    data.cloudfrac = 0;
    data.ne = 0;

    /* This shall never happen, but just in case*/
    if(!sfreff_on_eeqos(sph, a3inv))
        return data;

    data.ne = sph->Ne;
    data.tsfr = sqrt(sfr_params.PhysDensThresh / (sph->Density * a3inv)) * sfr_params.MaxSfrTimescale;
    if(sfr_params.BoostSFDenseGas && ((sph->Density * a3inv) / sfr_params.PhysDensThresh > sfr_params.BoostSFOverDenseFactor))
        data.tsfr = sfr_params.PhysDensThresh / (sph->Density * a3inv) * sfr_params.MaxSfrTimescale;
    /*
     * gadget-p doesn't have this cap.
     * without the cap sm can be bigger than cloudmass.
    */
    if(data.tsfr < dtime && dtime > 0)
        data.tsfr = dtime;

    double factorEVP = pow(sph->Density * a3inv / sfr_params.PhysDensThresh, -0.8) * sfr_params.FactorEVP;

    data.egyhot = sfr_params.EgySpecSN / (1 + factorEVP) + sfr_params.EgySpecCold;
    data.egycold = sfr_params.EgySpecCold;

    double tcool = GetCoolingTime(redshift, data.egyhot, sph->Density * a3inv, local_uvbg, &data.ne, sph->Metallicity);
    double y = data.tsfr / tcool * data.egyhot / (sfr_params.FactorSN * sfr_params.EgySpecSN - (1 - sfr_params.FactorSN) * sfr_params.EgySpecCold);

    data.cloudfrac = 1 + 1 / (2 * y) - sqrt(1 / y + 1 / (4 * y * y));

    data.trelax = data.tsfr * (1 - data.cloudfrac) / data.cloudfrac / (sfr_params.FactorSN * (1 + factorEVP));
    return data;
}

static double get_starformation_rate_full(int i, MyFloat * GradRho, struct sfr_eeqos_data sfr_data, const double atime, const double a3inv, const double hubble, const double GravInternal)
{
    if(!sfreff_on_eeqos(&SPHP(i), a3inv)) {
        return 0;
    }

    double cloudmass = sfr_data.cloudfrac * P[i].Mass;

    double rateOfSF = (1 - sfr_params.FactorSN) * cloudmass / sfr_data.tsfr;

    if (HAS(sfr_params.StarformationCriterion, SFR_CRITERION_MOLECULAR_H2)) {
        if(!GradRho)
            endrun(1, "GradRho not allocated but has SFR_CRITERION_MOLECULAR_H2. Should never happen!\n");
        rateOfSF *= get_sfr_factor_due_to_h2(i, GradRho, atime);
    }
    if (HAS(sfr_params.StarformationCriterion, SFR_CRITERION_SELFGRAVITY)) {
        rateOfSF *= get_sfr_factor_due_to_selfgravity(i, atime, a3inv, hubble, GravInternal);
    }
    return rateOfSF;
}

/*Gets the effective energy*/
static double
get_egyeff(double redshift, double dens, struct UVBG * uvbg)
{
    double tsfr = sqrt(sfr_params.PhysDensThresh / (dens)) * sfr_params.MaxSfrTimescale;
    double factorEVP = pow(dens / sfr_params.PhysDensThresh, -0.8) * sfr_params.FactorEVP;
    double egyhot = sfr_params.EgySpecSN / (1 + factorEVP) + sfr_params.EgySpecCold;

    double ne = 0.5;
    double tcool = GetCoolingTime(redshift, egyhot, dens, uvbg, &ne, 0.0);

    double y = tsfr / tcool * egyhot / (sfr_params.FactorSN * sfr_params.EgySpecSN - (1 - sfr_params.FactorSN) * sfr_params.EgySpecCold);
    double x = 1 + 1 / (2 * y) - sqrt(1 / y + 1 / (4 * y * y));
    return egyhot * (1 - x) + sfr_params.EgySpecCold * x;
}

/* Minimum temperature in internal energy*/
double get_MinEgySpec(void)
{
    /* mean molecular weight assuming ZERO ionization NEUTRAL GAS*/
    const double meanweight = 4.0 / (1 + 3 * HYDROGEN_MASSFRAC);
    /*Enforces a minimum internal energy in cooling. */
    return sfr_params.temp_to_u / meanweight * sfr_params.MinGasTemp;
}

void init_cooling_and_star_formation(int CoolingOn, int StarformationOn, Cosmology * CP, const double avg_baryon_mass, const double BoxSize, const struct UnitSystem units)
{
    struct cooling_units coolunits;
    coolunits.CoolingOn = CoolingOn;
    coolunits.density_in_phys_cgs = units.UnitDensity_in_cgs * CP->HubbleParam * CP->HubbleParam;
    coolunits.uu_in_cgs = units.UnitInternalEnergy_in_cgs;
    coolunits.tt_in_s = units.UnitTime_in_s / CP->HubbleParam;
    /* Get mean cosmic baryon density for photoheating rate from long mean free path photons */
    coolunits.rho_crit_baryon = 3 * pow(CP->HubbleParam * HUBBLE,2) * CP->OmegaBaryon / (8 * M_PI * GRAVITY);

    sfr_params.temp_to_u = (1.0 / GAMMA_MINUS1) * (BOLTZMANN / PROTONMASS) / units.UnitInternalEnergy_in_cgs;

    sfr_params.UnitSfr_in_solar_per_year = (units.UnitMass_in_g / SOLAR_MASS) / (units.UnitTime_in_s / SEC_PER_YEAR);

    init_cooling(sfr_params.TreeCoolFile, sfr_params.J21CoeffFile, sfr_params.MetalCoolFile, sfr_params.ReionHistFile, coolunits, CP);

    if(!CoolingOn)
        return;

    /*Initialize the uv fluctuation table*/
    init_uvf_table(sfr_params.UVFluctuationFile, sizeof(sfr_params.UVFluctuationFile), BoxSize, units.UnitLength_in_cm);

    sfr_params.StarformationOn = StarformationOn;

    if(!StarformationOn)
        return;

    sfr_params.avg_baryon_mass = avg_baryon_mass;

    sfr_params.tau_fmol_unit = units.UnitDensity_in_cgs*CP->HubbleParam*units.UnitLength_in_cm;
    sfr_params.OverDensThresh =
        sfr_params.CritOverDensity * CP->OmegaBaryon * CP->RhoCrit;

    sfr_params.PhysDensThresh = sfr_params.CritPhysDensity * PROTONMASS / HYDROGEN_MASSFRAC / units.UnitDensity_in_cgs;

    /* mean molecular weight assuming ZERO ionization NEUTRAL GAS*/
    double meanweight = 4.0 / (1 + 3 * HYDROGEN_MASSFRAC);
    sfr_params.EgySpecCold = (sfr_params.temp_to_u/meanweight) * sfr_params.TempClouds;

    /* mean molecular weight assuming FULL ionization */
    meanweight = 4 / (8 - 5 * (1 - HYDROGEN_MASSFRAC));
    sfr_params.EgySpecSN = sfr_params.temp_to_u/meanweight * sfr_params.TempSupernova;

    if(sfr_params.PhysDensThresh == 0)
    {
        double egyhot = sfr_params.EgySpecSN / sfr_params.FactorEVP;

        meanweight = 4 / (8 - 5 * (1 - HYDROGEN_MASSFRAC));	/* note: assuming FULL ionization */

        double u4 = sfr_params.temp_to_u/meanweight * 1.0e4;

        double dens = 1.0e6 * CP->RhoCrit;

        double ne = 1.0;

        struct UVBG uvbg = {0};
        /*XXX: We set the threshold without metal cooling
         * and with zero ionization at z=0.
         * It probably make sense to set the parameters with
         * a metalicity dependence.
         * */
        const double tcool = GetCoolingTime(0, egyhot, dens, &uvbg, &ne, 0.0);

        const double coolrate = egyhot / tcool / dens;

        const double x = (egyhot - u4) / (egyhot - sfr_params.EgySpecCold);

        sfr_params.PhysDensThresh = x / pow(1 - x, 2) *
                    (sfr_params.FactorSN * sfr_params.EgySpecSN - (1 - sfr_params.FactorSN) * sfr_params.EgySpecCold)
                    / (sfr_params.MaxSfrTimescale * coolrate);

        message(0, "A0= %g  \n", sfr_params.FactorEVP);
        message(0, "Computed: PhysDensThresh= %g  (int units)         %g h^2 cm^-3\n", sfr_params.PhysDensThresh,
                sfr_params.PhysDensThresh / (PROTONMASS / HYDROGEN_MASSFRAC / units.UnitDensity_in_cgs));
        message(0, "EXPECTED FRACTION OF COLD GAS AT THRESHOLD = %g\n", x);
        message(0, "tcool=%g dens=%g egyhot=%g\n", tcool, dens, egyhot);

        dens = sfr_params.PhysDensThresh * 10;

        double neff;
        do
        {
            double egyeff = get_egyeff(0, dens, &uvbg);

            double peff = GAMMA_MINUS1 * dens * egyeff;

            const double fac = 1 / (log(dens * 1.025) - log(dens));
            neff = -log(peff) * fac;

            dens *= 1.025;
            egyeff = get_egyeff(0, dens, &uvbg);
            peff = GAMMA_MINUS1 * dens * egyeff;

            neff += log(peff) * fac;
        }
        while(neff > 4.0 / 3);

        message(0, "Run-away sets in for dens=%g\n", dens);
        message(0, "Dynamic range for quiescent star formation= %g\n", dens / sfr_params.PhysDensThresh);

        const double sigma = 10.0 / CP->Hubble * 1.0e-10 / pow(1.0e-3, 2);

        message(0, "Isotherm sheet central density: %g   z0=%g\n",
                M_PI * CP->GravInternal * sigma * sigma / (2 * GAMMA_MINUS1) / u4,
                GAMMA_MINUS1 * u4 / (2 * M_PI * CP->GravInternal * sigma));
    }

    if(sfr_params.WindOn) {
        init_winds(sfr_params.FactorSN, sfr_params.EgySpecSN, sfr_params.PhysDensThresh, units.UnitTime_in_s);
    }

}

static double
find_star_mass(int i, const double avg_baryon_mass)
{
    /*Quick Lyman Alpha always turns all of a particle into stars*/
    if(sfr_params.QuickLymanAlphaProbability > 0)
        return P[i].Mass;

    double mass_of_star =  avg_baryon_mass / sfr_params.Generations;
    if(mass_of_star > P[i].Mass) {
        /* if some mass has been stolen by BH, e.g */
        mass_of_star = P[i].Mass;
    }
    /* Conditions to turn the gas into a star. .
     * The mass check makes sure we never get a gas particle which is lighter
     * than the smallest star particle.
     * The Generations check (which can happen because of mass return)
     * ensures we never instantaneously enrich stars above solar. */
    if(P[i].Mass < 2 * mass_of_star  || P[i].Generation > sfr_params.Generations) {
        mass_of_star = P[i].Mass;
    }
    return mass_of_star;
}

/********************
 *
 * The follow functions are from Desika and Gadget-P.
 * We really are mostly concerned about H2 here.
 *
 * You may need a license to run with these modess.

 * */

int sfr_need_to_compute_sph_grad_rho(void)
{
    if (HAS(sfr_params.StarformationCriterion, SFR_CRITERION_MOLECULAR_H2)) {
        return 1;
    }
    return 0;
}
static double ev_NH_from_GradRho(MyFloat gradrho_mag, double hsml, double rho, double include_h)
{
    /* column density from GradRho, copied from gadget-p; what is it
     * calculating? */
    if(rho<=0)
        return 0;
    double ev_NH = 0;
    if(gradrho_mag > 0)
        ev_NH = rho*rho/gradrho_mag;
    if(include_h > 0)
        ev_NH += rho*hsml;
    return ev_NH; // *(Z/Zsolar) add metallicity dependence
}

static double get_sfr_factor_due_to_h2(int i, MyFloat * GradRho_mag, const double atime) {
    /*  Krumholz & Gnedin fitting function for f_H2 as a function of local
     *  properties, from gadget-p; we return the enhancement on SFR in this
     *  function */
    double tau_fmol;
    const double a2 = atime * atime;
    double zoverzsun = SPHP(i).Metallicity/METAL_YIELD;
    double gradrho_mag = GradRho_mag[P[i].PI];
    //message(4, "GradRho %g rho %g hsml %g i %d\n", gradrho_mag, SPHP(i).Density, P[i].Hsml, i);
    tau_fmol = ev_NH_from_GradRho(gradrho_mag,P[i].Hsml,SPHP(i).Density,1) /a2;
    tau_fmol *= (0.1 + zoverzsun);
    if(tau_fmol>0) {
        tau_fmol *= 434.78*sfr_params.tau_fmol_unit;
        double y = 0.756*(1+3.1*pow(zoverzsun,0.365));
        y = log(1+0.6*y+0.01*y*y)/(0.6*tau_fmol);
        y = 1-0.75*y/(1+0.25*y);
        if(y<0) y=0;
        if(y>1) y=1;
        return y;

    } // if(tau_fmol>0)
    return 1.0;
}

static double get_sfr_factor_due_to_selfgravity(int i, const double atime, const double a3inv, const double hubble, const double GravInternal) {
    const double a2 = atime * atime;
    double divv = SPHP(i).DivVel / a2;

    divv += 3.0*hubble * a2; // hubble-flow correction

    if(HAS(sfr_params.StarformationCriterion, SFR_CRITERION_CONVERGENT_FLOW)) {
        if( divv>=0 ) return 0; // restrict to convergent flows (optional) //
    }

    double dv2abs = (divv*divv
            + (SPHP(i).CurlVel/a2)
            * (SPHP(i).CurlVel/a2)
           ); // all in physical units
    double alpha_vir = 0.2387 * dv2abs/(GravInternal * SPHP(i).Density * a3inv);

    double y = 1.0;

    if((alpha_vir < 1.0)
    || (SPHP(i).Density * a3inv > 100. * sfr_params.PhysDensThresh)
    )  {
        y = 66.7;
    } else {
        y = 0.1;
    }
    // PFH: note the latter flag is an arbitrary choice currently set
    // -by hand- to prevent runaway densities from this prescription! //

    if (HAS(sfr_params.StarformationCriterion, SFR_CRITERION_CONTINUOUS_CUTOFF)) {
        // continuous cutoff w alpha_vir instead of sharp (optional) //
        y *= 1.0/(1.0 + alpha_vir);
    }
    return y;
}

/* Update the active particle list when a new star is formed.
 * if the parent is active the child should also be active.
 * Stars must always be (hydro) active on formation. Returns
 * whether particle is gravity active. */
static int
add_new_particle_to_active(const int parent, const int child, ActiveParticles * act)
{

    /* If gravity active, increment the counter*/
    int is_grav_active = is_timebin_active(P[parent].TimeBinGravity, P[parent].Ti_drift);
    /* If either is active, need to be in the active list. */
    if(is_grav_active || is_timebin_active(P[parent].TimeBinHydro, P[parent].Ti_drift)) {
        int64_t childactive = atomic_fetch_and_add_64(&act->NumActiveParticle, 1);
        if(act->ActiveParticle) {
            /* This should never happen because we allocate as much space for active particles as we have space
             * for particles, but just in case*/
            if(childactive >= act->MaxActiveParticle)
                endrun(5, "Tried to add %ld active particles, more than %ld allowed\n", childactive, act->MaxActiveParticle);
            act->ActiveParticle[childactive] = child;
        }
    }
    return is_grav_active;
}

/* Copy the gravitational acceleration if necessary for a new particle.*/
static int
copy_gravaccel_new_particle(const int parent, const int child, MyFloat (* GravAccel)[3], int64_t nstoredgravaccel)
{
    /* If gravity active, copy the grav accel to the new child*/
    int is_grav_active = is_timebin_active(P[parent].TimeBinGravity, P[parent].Ti_drift);
    if(is_grav_active && GravAccel) {
        if(child >= nstoredgravaccel)
            endrun(1, "Not enough space (%ld) in stored GravAccel to copy new star %d from parent %d\n", nstoredgravaccel, child, parent);
        int j;
        for(j=0; j < 3 ; j++)
            GravAccel[child][j] = GravAccel[parent][j];
    }
    return 0;
}
