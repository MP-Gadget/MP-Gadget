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
#include "allvars.h"
#include "sfr_eff.h"
#include "cooling.h"
#include "slotsmanager.h"
#include "winds.h"
#include "hydra.h"
/*Only for the star slot reservation*/
#include "forcetree.h"
#include "timestep.h"
#include "domain.h"

/*Parameters of the star formation model*/
static struct SFRParams
{
    enum StarformationCriterion StarformationCriterion;  /*!< Type of star formation model. */
    /*Star formation parameters*/
    double CritOverDensity;
    double CritPhysDensity;
    double OverDensThresh;
    double PhysDensThresh;
    double EgySpecSN;
    double FactorSN;
    double EgySpecCold;
    double FactorEVP;
    double FeedbackEnergy;
    double TempSupernova;
    double TempClouds;
    double MaxSfrTimescale;
    /*!< may be used to set a floor for the gas temperature */
    double MinGasTemp;

    /*Lyman alpha forest specific star formation.*/
    double QuickLymanAlphaProbability;
    double QuickLymanAlphaTempThresh;
    /* Number of stars to create from each gas particle*/
    int Generations;
} sfr_params;

/* Structure storing the results of an evaluation of the star formation model*/
struct sfr_eeqos_data
{
    /* Relaxation time*/
    double trelax;
    /* Star formation timescale*/
    double tsfr;
    /* Internal energy of the gas in the hot phase. */
    double egyhot;
    /* Fraction of the gas in the cold cloud phase. */
    double cloudfrac;
    /* Electron fraction after cooling. */
    double ne;
};

/*Cooling only: no star formation*/
static void cooling_direct(int i);

static void cooling_relaxed(int i, double egyeff, double dtime, double trelax);

static int sfreff_on_eeqos(int i);
static int make_particle_star(int child, int parent, int placement);
static int starformation(int i, double *localsfr, double * sum_sm);
static int quicklyastarformation(int i);
static double get_sfr_factor_due_to_selfgravity(int i);
static double get_sfr_factor_due_to_h2(int i);
static double get_starformation_rate_full(int i, struct sfr_eeqos_data sfr_data);
static double find_star_mass(int i);
/*Get enough memory for new star slots. This may be excessively slow! Don't do it too often.*/
static int * sfr_reserve_slots(int * NewStars, int NumNewStar, ForceTree * tt);
static struct sfr_eeqos_data get_sfr_eeqos(int i, double dtime);

/*Set the parameters of the SFR module*/
void set_sfr_params(ParameterSet * ps)
{
    int ThisTask;
    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
    if(ThisTask == 0) {
        /*Star formation parameters*/
        sfr_params.StarformationCriterion = param_get_enum(ps, "StarformationCriterion");
        sfr_params.CritOverDensity = param_get_double(ps, "CritOverDensity");
        sfr_params.CritPhysDensity = param_get_double(ps, "CritPhysDensity");

        sfr_params.FactorSN = param_get_double(ps, "FactorSN");
        sfr_params.FactorEVP = param_get_double(ps, "FactorEVP");
        sfr_params.TempSupernova = param_get_double(ps, "TempSupernova");
        sfr_params.TempClouds = param_get_double(ps, "TempClouds");
        sfr_params.MaxSfrTimescale = param_get_double(ps, "MaxSfrTimescale");
        sfr_params.Generations = param_get_int(ps, "Generations");
        sfr_params.MinGasTemp = param_get_double(ps, "MinGasTemp");

        /*Lyman-alpha forest parameters*/
        sfr_params.QuickLymanAlphaProbability = param_get_double(ps, "QuickLymanAlphaProbability");
        sfr_params.QuickLymanAlphaTempThresh = param_get_double(ps, "QuickLymanAlphaTempThresh");
    }
    MPI_Bcast(&sfr_params, sizeof(struct SFRParams), MPI_BYTE, 0, MPI_COMM_WORLD);
}


/* cooling and star formation routine.*/
void cooling_and_starformation(ForceTree * tree)
{
    if(!All.CoolingOn)
        return;

    const int nthreads = omp_get_max_threads();
    /*This is a queue for the new stars and their parents, so we can reallocate the slots after the main cooling loop.*/
    int * NewStars = NULL;
    int * NewParents = NULL;
    int NumNewStar = 0;
    size_t *nqthrsfr = ta_malloc("nqthrsfr", size_t, nthreads);
    int **thrqueuesfr = ta_malloc("thrqueuesfr", int *, nthreads);
    int **thrqueueparent = ta_malloc("thrqueueparent", int *, nthreads);

    /*Need to capture this so that when NumActiveParticle increases during the loop
     * we don't add extra loop iterations on particles with invalid slots.*/
    const int nactive = NumActiveParticle;

    if(All.StarformationOn) {
        /* Need 1 extra for non-integer part and 1 extra
         * for the case where one thread loops an extra time*/
        int narr = nactive/nthreads+2;
        NewStars = mymalloc("NewStars", narr * sizeof(int) * nthreads);
        gadget_setup_thread_arrays(NewStars, thrqueuesfr, nqthrsfr, narr, nthreads);
        NewParents = mymalloc2("NewParents", narr * sizeof(int) * nthreads);
        gadget_setup_thread_arrays(NewParents, thrqueueparent, nqthrsfr, narr, nthreads);
    }

    double sum_sm = 0, sum_mass_stars = 0, localsfr = 0;

    /* First decide which stars are cooling and which starforming. If star forming we add them to a list.
     * Note the dynamic scheduling: individual particles may have very different loop iteration lengths.
     * Cooling is much slower than sfr. I tried splitting it into a separate loop instead, but this was faster.*/
    #pragma omp parallel reduction(+:localsfr) reduction(+: sum_sm) reduction(+:sum_mass_stars)
    {
        int i;
        const int tid = omp_get_thread_num();

        for(i=tid; i < nactive; i+=nthreads)
        {
            /*Use raw particle number if active_set is null, otherwise use active_set*/
            const int p_i = ActiveParticle ? ActiveParticle[i] : i;
            /* Skip non-gas or garbage or swallowed particles */
            if(P[p_i].Type != 0 || P[p_i].IsGarbage || P[p_i].Swallowed || P[p_i].Mass <= 0)
                continue;

            int shall_we_star_form = 0;
            if(All.StarformationOn) {
                /*Reduce delaytime for wind particles.*/
                winds_evolve(p_i, All.cf.a3inv, All.cf.hubble);
                /* check whether we are star forming gas.*/
                if(sfr_params.QuickLymanAlphaProbability > 0)
                    shall_we_star_form = quicklyastarformation(p_i);
                else
                    shall_we_star_form = sfreff_on_eeqos(p_i);
            }

            if(shall_we_star_form) {
                int newstar = -1;
                if(sfr_params.QuickLymanAlphaProbability > 0) {
                    /*New star is always the same particle as the parent for quicklya*/
                    newstar = p_i;
                } else {
                    newstar = starformation(p_i, &localsfr, &sum_sm);
                }
                /*Add this particle to the stellar conversion queue if necessary.*/
                if(newstar >= 0) {
                    int tid = omp_get_thread_num();
                    thrqueuesfr[tid][nqthrsfr[tid]] = newstar;
                    thrqueueparent[tid][nqthrsfr[tid]] = p_i;
                    nqthrsfr[tid]++;
                }
            }
            else
                cooling_direct(p_i);
        }
    }

    report_memory_usage("SFR");
    /*Merge step for the queue.*/
    if(NewStars) {
        NumNewStar = gadget_compact_thread_arrays(NewStars, thrqueuesfr, nqthrsfr, nthreads);
        int NumNewParent = gadget_compact_thread_arrays(NewParents, thrqueueparent, nqthrsfr, nthreads);
        if(NumNewStar != NumNewParent)
            endrun(3,"%d new stars, but %d new parents!\n",NumNewStar, NumNewParent);
        /*Shrink star memory as we keep it for the wind model*/
        NewStars = myrealloc(NewStars, sizeof(int) * NumNewStar);
    }

    ta_free(thrqueueparent);
    ta_free(thrqueuesfr);
    ta_free(nqthrsfr);

    walltime_measure("/Cooling/Cooling");

    /*Get some empty slots for the stars*/
    int firststarslot = SlotsManager->info[4].size;
    /* We ran out of slots! We must be forming a lot of stars.
     * There are things in the way of extending the slot list, so we have to move them.
     * The code in sfr_reserve_slots is not elegant, but I cannot think of a better way.*/
    if(All.StarformationOn && (SlotsManager->info[4].size + NumNewStar >= SlotsManager->info[4].maxsize)) {
        if(NewParents)
            NewParents = myrealloc(NewParents, sizeof(int) * NumNewStar);
        NewStars = sfr_reserve_slots(NewStars, NumNewStar, tree);
    }
    SlotsManager->info[4].size += NumNewStar;

    int stars_converted=0, stars_spawned=0;
    int i;

    /*Now we turn the particles into stars*/
    #pragma omp parallel for reduction(+:stars_converted) reduction(+:stars_spawned) reduction(+:sum_mass_stars)
    for(i=0; i < NumNewStar; i++)
    {
        int child = NewStars[i];
        int parent = NewParents[i];
        make_particle_star(child, parent, firststarslot+i);
        sum_mass_stars += P[child].Mass;
        if(child == parent)
            stars_converted++;
        else
            stars_spawned++;
    }

    if(!All.StarformationOn)
        return;

    /*Done with the parents*/
    myfree(NewParents);

    /* This is allocated high so has to be freed after the parents*/
    if(SphP_scratch->Injected_BH_Energy) {
        myfree(SphP_scratch->Injected_BH_Energy);
        SphP_scratch->Injected_BH_Energy = NULL;
    }

    int64_t tot_spawned=0, tot_converted=0;
    sumup_large_ints(1, &stars_spawned, &tot_spawned);
    sumup_large_ints(1, &stars_converted, &tot_converted);

    double total_sum_mass_stars, total_sm, totsfrrate;

    MPI_Reduce(&localsfr, &totsfrrate, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&sum_sm, &total_sm, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&sum_mass_stars, &total_sum_mass_stars, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if(ThisTask == 0)
    {
        double rate = 0;
        if(All.TimeStep > 0)
            rate = total_sm / (All.TimeStep / (All.Time * All.cf.hubble));

        /* convert to solar masses per yr */

        double rate_in_msunperyear = rate * (All.UnitMass_in_g / SOLAR_MASS) / (All.UnitTime_in_s / SEC_PER_YEAR);

        fprintf(FdSfr, "%g %g %g %g %g\n", All.Time, total_sm, totsfrrate, rate_in_msunperyear,
                total_sum_mass_stars);
        fflush(FdSfr);
    }

    MPIU_Barrier(MPI_COMM_WORLD);
    if(tot_spawned || tot_converted)
        message(0, "SFR: spawned %ld stars, converted %ld gas particles into stars\n", tot_spawned, tot_converted);

    walltime_measure("/Cooling/StarFormation");

    /* Now apply the wind model using the list of new stars.*/
    if(All.WindOn)
        winds_and_feedback(NewStars, NumNewStar, tree);

    myfree(NewStars);
}

/* Get enough memory for new star slots. This may be excessively slow! Don't do it too often.
 * It is also not elegant, but I couldn't think of a better way. May be fragile and need updating
 * if memory allocation patterns change. */
static int *
sfr_reserve_slots(int * NewStars, int NumNewStar, ForceTree * tree)
{
        /* SlotsManager is below Nodes and ActiveParticleList,
         * so we need to move them out of the way before we extend Nodes.
         * This is quite slow, but need not be collective and is faster than a tree rebuild.
         * Try not to do this too often.*/
        message(1, "Need %d star slots, more than %d available. Try increasing SlotsIncreaseFactor on restart.\n", SlotsManager->info[4].size, SlotsManager->info[4].maxsize);
        /*Move the NewStar array to upper memory*/
        int * new_star_tmp = NULL;
        if(NewStars) {
            new_star_tmp = mymalloc2("newstartmp", NumNewStar*sizeof(int));
            memmove(new_star_tmp, NewStars, NumNewStar * sizeof(int));
            myfree(NewStars);
        }
        /*Move the tree to upper memory*/
        struct NODE * nodes_base_tmp=NULL;
        int *Nextnode_tmp=NULL;
        int *Father_tmp=NULL;
        int *ActiveParticle_tmp=NULL;
        if(force_tree_allocated(tree)) {
            nodes_base_tmp = mymalloc2("nodesbasetmp", tree->numnodes * sizeof(struct NODE));
            memmove(nodes_base_tmp, tree->Nodes_base, tree->numnodes * sizeof(struct NODE));
            myfree(tree->Nodes_base);
            Father_tmp = mymalloc2("Father_tmp", PartManager->MaxPart * sizeof(int));
            memmove(Father_tmp, tree->Father, PartManager->MaxPart * sizeof(int));
            myfree(tree->Father);
            Nextnode_tmp = mymalloc2("Nextnode_tmp", tree->Nnextnode * sizeof(int));
            memmove(Nextnode_tmp, tree->Nextnode, tree->Nnextnode * sizeof(int));
            myfree(tree->Nextnode);
        }
        if(ActiveParticle) {
            ActiveParticle_tmp = mymalloc2("ActiveParticle_tmp", NumActiveParticle * sizeof(int));
            memmove(ActiveParticle_tmp, ActiveParticle, NumActiveParticle * sizeof(int));
            myfree(ActiveParticle);
        }
        /*Now we can extend the slots! */
        int atleast[6];
        int i;
        for(i = 0; i < 6; i++)
            atleast[i] = SlotsManager->info[i].maxsize;
        atleast[4] += NumNewStar;
        slots_reserve(1, atleast);

        /*And now we need our memory back in the right place*/
        if(ActiveParticle_tmp) {
            ActiveParticle = mymalloc("ActiveParticle", sizeof(int)*(NumActiveParticle + PartManager->MaxPart - PartManager->NumPart));
            memmove(ActiveParticle, ActiveParticle_tmp, NumActiveParticle * sizeof(int));
            myfree(ActiveParticle_tmp);
        }
        if(force_tree_allocated(tree)) {
            tree->Nextnode = mymalloc("Nextnode", tree->Nnextnode * sizeof(int));
            memmove(tree->Nextnode, Nextnode_tmp, tree->Nnextnode * sizeof(int));
            myfree(Nextnode_tmp);
            tree->Father = mymalloc("Father", PartManager->MaxPart * sizeof(int));
            memmove(tree->Father, Father_tmp, PartManager->MaxPart * sizeof(int));
            myfree(Father_tmp);
            tree->Nodes_base = mymalloc("Nodes_base", tree->numnodes * sizeof(struct NODE));
            memmove(tree->Nodes_base, nodes_base_tmp, tree->numnodes * sizeof(struct NODE));
            myfree(nodes_base_tmp);
            /*Don't forget to update the Node pointer as well as Node_base!*/
            tree->Nodes = tree->Nodes_base - tree->firstnode;
        }
        if(new_star_tmp) {
            NewStars = mymalloc("NewStars", NumNewStar*sizeof(int));
            memmove(NewStars, new_star_tmp, NumNewStar * sizeof(int));
            myfree(new_star_tmp);
        }
        return NewStars;
}

/* Adds the injected black hole energy to an internal energy and caps it at a maximum temperature*/
static double
add_injected_BH_energy(double unew, double injected_BH_energy, double mass)
{
    unew += injected_BH_energy / mass;
    const double u_to_temp_fac = (4 / (8 - 5 * (1 - HYDROGEN_MASSFRAC))) * PROTONMASS / BOLTZMANN * GAMMA_MINUS1
    * All.UnitEnergy_in_cgs / All.UnitMass_in_g;

    double temp = u_to_temp_fac * unew;

    if(temp > 5.0e9)
        unew = 5.0e9 / u_to_temp_fac;

    return unew;
}

static void
cooling_direct(int i) {

    /*  the actual time-step */
    double dloga = get_dloga_for_bin(P[i].TimeBin);
    double dtime = dloga / All.cf.hubble;

    double ne = SPHP(i).Ne;	/* electron abundance (gives ionization state and mean molecular weight) */

    const double enttou = pow(SPH_EOMDensity(i) * All.cf.a3inv, GAMMA_MINUS1) / GAMMA_MINUS1;
    double unew = (SPHP(i).Entropy + SPHP(i).DtEntropy * dloga) * enttou;

    if(SphP_scratch->Injected_BH_Energy && SphP_scratch->Injected_BH_Energy[P[i].PI] > 0)
    {
        unew = add_injected_BH_energy(unew, SphP_scratch->Injected_BH_Energy[P[i].PI], P[i].Mass);
    }

    double redshift = 1./All.Time - 1;
    struct UVBG uvbg = get_local_UVBG(redshift, P[i].Pos);
    unew = DoCooling(redshift, unew, SPHP(i).Density * All.cf.a3inv, dtime, &uvbg, &ne, SPHP(i).Metallicity, All.MinEgySpec);

    SPHP(i).Ne = ne;

    /* upon start-up, we need to protect against dt==0 */
    if(dloga > 0)
    {
        /* note: the adiabatic rate has been already added in ! */
        SPHP(i).DtEntropy = (unew / enttou - SPHP(i).Entropy) / dloga;
    }
}

/* returns 1 if the particle is on the effective equation of state,
 * cooling via the relaxation equation and maybe forming stars.
 * 0 if the particle does not form stars, instead cooling normally.*/
static int
sfreff_on_eeqos(int i)
{
    int flag = 0;
    /* no sfr: normal cooling*/
    if(!All.StarformationOn) {
        return 0;
    }
    /* Check that starformation has not left a massless particle.*/
    if(P[i].Mass == 0) {
        endrun(12, "Particle %d in SFR has mass=%g. Does not happen as no tracer particles.\n", i, P[i].Mass);
    }

    if(SPHP(i).Density * All.cf.a3inv >= sfr_params.PhysDensThresh)
        flag = 1;

    if(SPHP(i).Density < sfr_params.OverDensThresh)
        flag = 0;

    if(SPHP(i).DelayTime > 0)
        flag = 0;		/* only normal cooling for particles in the wind */

    return flag;
}

/*Get the neutral fraction of a particle correctly, accounting for being on the star-forming equation of state*/
double get_neutral_fraction_sfreff(int i, double redshift)
{
    double nh0;
    struct UVBG uvbg = get_local_UVBG(redshift, P[i].Pos);
    double physdens = SPHP(i).Density * All.cf.a3inv;

    if(!All.StarformationOn || sfr_params.QuickLymanAlphaProbability > 0 || !sfreff_on_eeqos(i)) {
        /*This gets the neutral fraction for standard gas*/
        double InternalEnergy = SPHP(i).Entropy / GAMMA_MINUS1 * pow(SPH_EOMDensity(i) * All.cf.a3inv, GAMMA_MINUS1);
        nh0 = GetNeutralFraction(InternalEnergy, physdens, &uvbg, SPHP(i).Ne);
    }
    else {
        /* This gets the neutral fraction for gas on the star-forming equation of state.
         * This needs special handling because the cold clouds have a different neutral
         * fraction than the hot gas*/
        double dloga = get_dloga_for_bin(P[i].TimeBin);
        double dtime = dloga / All.cf.hubble;
        struct sfr_eeqos_data sfr_data = get_sfr_eeqos(i, dtime);
        double nh0cold = GetNeutralFraction(sfr_params.EgySpecCold, physdens, &uvbg, sfr_data.ne);
        double nh0hot = GetNeutralFraction(sfr_data.egyhot, physdens, &uvbg, sfr_data.ne);
        nh0 =  nh0cold * sfr_data.cloudfrac + (1-sfr_data.cloudfrac) * nh0hot;
    }
    return nh0;
}

/* This function turns a particle into a star. It returns 1 if a particle was
 * converted and 2 if a new particle was spawned. This is used
 * above to set stars_{spawned|converted}*/
static int make_particle_star(int child, int parent, int placement)
{
    int retflag = 2;
    if(P[parent].Type != 0)
        endrun(7772, "Only gas forms stars, what's wrong?\n");

    /*Store the SPH particle slot properties, as the PI may be over-written
     *in slots_convert*/
    struct sph_particle_data oldslot = SPHP(parent);

    /*Convert the child slot to the new type.*/
    child = slots_convert(child, 4, placement);

    /*Set properties*/
    STARP(child).FormationTime = All.Time;
    STARP(child).BirthDensity = oldslot.Density;
    /*Copy metallicity*/
    STARP(child).Metallicity = oldslot.Metallicity;
    return retflag;
}

static void cooling_relaxed(int i, double egyeff, double dtime, double trelax) {
    const double densityfac = pow(SPH_EOMDensity(i) * All.cf.a3inv, GAMMA_MINUS1) / GAMMA_MINUS1;
    double egycurrent = SPHP(i).Entropy *  densityfac;

    if(SphP_scratch->Injected_BH_Energy && SphP_scratch->Injected_BH_Energy[P[i].PI] > 0)
    {
        egycurrent = add_injected_BH_energy(egycurrent, SphP_scratch->Injected_BH_Energy[P[i].PI], P[i].Mass);

        if(egycurrent > egyeff)
        {
            double redshift = 1./All.Time - 1;
            struct UVBG uvbg = get_local_UVBG(redshift, P[i].Pos);
            double ne = SPHP(i).Ne;
            /* In practice tcool << trelax*/
            double tcool = GetCoolingTime(redshift, egycurrent, SPHP(i).Density * All.cf.a3inv, &uvbg, &ne, SPHP(i).Metallicity);

            /* If tcool is being used the exponential below is roughly zero. This could more compactly be written:
             * if(Injected_BH_Energy && egycurrent > egyeff)
             *      SPHP(i).Entropy = egyeff/densityfac.
             * In other words, any star-forming gas which is heated by a black hole instantaneously cools
             * to the effective equation of state temperature.
             * This reduces the effect of black hole feedback marginally (a 5% reduction in star formation)
             * and dates from the earliest versions of this code available.
             * The effect is relatively small because star-forming gas cools onto the effective equation
             * of state quickly anyway.
             * It is not clear to me (SPB) what this is modelling but removing it causes hot dense particles
             * with short timesteps to appear around the black holes and slows down the code.
             * Someone might want to check at some point if removing this condition helps or hinders
             * agreement with observations.*/
            if(tcool < trelax && tcool > 0)
                trelax = tcool;
        }
    }

    SPHP(i).Entropy =  (egyeff + (egycurrent - egyeff) * exp(-dtime / trelax)) /densityfac;

    SPHP(i).DtEntropy = 0;

}

/*Forms stars according to the quick lyman alpha star formation criterion,
 * which forms stars with a constant probability (usually 1) if they are star forming.
 * Returns 1 if converted a particle to a star, 0 if not.*/
static int
quicklyastarformation(int i)
{
    if(SPHP(i).Density <= sfr_params.OverDensThresh)
        return 0;

    double dloga = get_dloga_for_bin(P[i].TimeBin);
    const double enttou = pow(SPH_EOMDensity(i) * All.cf.a3inv, GAMMA_MINUS1) / GAMMA_MINUS1;
    double unew = (SPHP(i).Entropy + SPHP(i).DtEntropy * dloga) * enttou;

    const double u_to_temp_fac = (4 / (8 - 5 * (1 - HYDROGEN_MASSFRAC))) * PROTONMASS / BOLTZMANN * GAMMA_MINUS1 * All.UnitEnergy_in_cgs / All.UnitMass_in_g;

    double temp = u_to_temp_fac * unew;

    if(temp >= sfr_params.QuickLymanAlphaTempThresh)
        return 0;

    if(get_random_number(P[i].ID + 1) < sfr_params.QuickLymanAlphaProbability)
        return 1;

    return 0;
}

/* Forms stars and winds.
 * Returns -1 if no star formed, otherwise returns the index of the particle which is to be made a star.
 * The star slot is not actually created here, but a particle for it is.
 */
static int
starformation(int i, double *localsfr, double * sum_sm)
{
    /*  the proper time-step */
    double dloga = get_dloga_for_bin(P[i].TimeBin);
    double dtime = dloga / All.cf.hubble;
    int newstar = -1;

    struct sfr_eeqos_data sfr_data = get_sfr_eeqos(i, dtime);
    SPHP(i).Sfr = get_starformation_rate_full(i, sfr_data);
    SPHP(i).Ne = sfr_data.ne;

    /* amount of stars expect to form */

    double sm = SPHP(i).Sfr * dtime;

    double p = sm / P[i].Mass;

    *sum_sm += P[i].Mass * (1 - exp(-p));
    /* convert to Solar per Year.*/
    *localsfr += SPHP(i).Sfr * (All.UnitMass_in_g / SOLAR_MASS) / (All.UnitTime_in_s / SEC_PER_YEAR);

    double w = get_random_number(P[i].ID);
    SPHP(i).Metallicity += w * METAL_YIELD * (1 - exp(-p));

    if(dloga > 0 && P[i].TimeBin)
    {
        double egyeff = sfr_params.EgySpecCold * sfr_data.cloudfrac + (1 - sfr_data.cloudfrac) * sfr_data.egyhot;
        /* upon start-up, we need to protect against dloga ==0 */
        cooling_relaxed(i, egyeff, dtime, sfr_data.trelax);
    }

    double mass_of_star = find_star_mass(i);
    double prob = P[i].Mass / mass_of_star * (1 - exp(-p));

    int form_star = (get_random_number(P[i].ID + 1) < prob);
    if(form_star) {
        /* ok, make a star */
        newstar = i;
        /* If we get a fraction of the mass we need to create
         * a new particle for the star and remove mass from i.*/
        if(P[i].Mass >= 1.1 * mass_of_star)
            newstar = slots_split_particle(i, mass_of_star);
    }

    /* Add the rest of the metals if we didn't form a star.
     * If we did form a star, add winds to the star-forming particle
     * that formed it if it is still around*/
    if(!form_star || newstar != i)	{
        SPHP(i).Metallicity += (1 - w) * METAL_YIELD * (1 - exp(-p));

        if(All.WindOn) {
            winds_make_after_sf(i, sm, All.cf.a);
        }
    }
    return newstar;
}

/* Get the parameters of the basic effective
 * equation of state model for a particle.*/
static struct sfr_eeqos_data get_sfr_eeqos(int i, double dtime)
{
    struct sfr_eeqos_data data;
    /* Initialise data to something, just in case.*/
    data.trelax = sfr_params.MaxSfrTimescale;
    data.tsfr = sfr_params.MaxSfrTimescale;
    data.egyhot = sfr_params.EgySpecCold;
    data.cloudfrac = 0;
    data.ne = 0;

    /* This shall never happen, but just in case*/
    if(!All.StarformationOn || !sfreff_on_eeqos(i))
        return data;

    data.ne = SPHP(i).Ne;
    data.tsfr = sqrt(sfr_params.PhysDensThresh / (SPHP(i).Density * All.cf.a3inv)) * sfr_params.MaxSfrTimescale;
    /*
     * gadget-p doesn't have this cap.
     * without the cap sm can be bigger than cloudmass.
    */
    if(data.tsfr < dtime)
        data.tsfr = dtime;

    double redshift = 1./All.Time - 1;
    struct UVBG uvbg = get_local_UVBG(redshift, P[i].Pos);

    double factorEVP = pow(SPHP(i).Density * All.cf.a3inv / sfr_params.PhysDensThresh, -0.8) * sfr_params.FactorEVP;

    data.egyhot = sfr_params.EgySpecSN / (1 + factorEVP) + sfr_params.EgySpecCold;

    double tcool = GetCoolingTime(redshift, data.egyhot, SPHP(i).Density * All.cf.a3inv, &uvbg, &data.ne, SPHP(i).Metallicity);
    double y = data.tsfr / tcool * data.egyhot / (sfr_params.FactorSN * sfr_params.EgySpecSN - (1 - sfr_params.FactorSN) * sfr_params.EgySpecCold);

    data.cloudfrac = 1 + 1 / (2 * y) - sqrt(1 / y + 1 / (4 * y * y));

    data.trelax = data.tsfr * (1 - data.cloudfrac) / data.cloudfrac / (sfr_params.FactorSN * (1 + factorEVP));
    return data;
}

static double get_starformation_rate_full(int i, struct sfr_eeqos_data sfr_data)
{
    if(!All.StarformationOn || !sfreff_on_eeqos(i)) {
        return 0;
    }

    double cloudmass = sfr_data.cloudfrac * P[i].Mass;

    double rateOfSF = (1 - sfr_params.FactorSN) * cloudmass / sfr_data.tsfr;

    if (HAS(sfr_params.StarformationCriterion, SFR_CRITERION_MOLECULAR_H2)) {
        rateOfSF *= get_sfr_factor_due_to_h2(i);
    }
    if (HAS(sfr_params.StarformationCriterion, SFR_CRITERION_SELFGRAVITY)) {
        rateOfSF *= get_sfr_factor_due_to_selfgravity(i);
    }
    return rateOfSF;
}

/*Gets the effective energy at z=0*/
static double
get_egyeff(double dens, struct UVBG * uvbg)
{
    double tsfr = sqrt(sfr_params.PhysDensThresh / (dens)) * sfr_params.MaxSfrTimescale;
    double factorEVP = pow(dens / sfr_params.PhysDensThresh, -0.8) * sfr_params.FactorEVP;
    double egyhot = sfr_params.EgySpecSN / (1 + factorEVP) + sfr_params.EgySpecCold;

    double ne = 0.5;
    double tcool = GetCoolingTime(0, egyhot, dens, uvbg, &ne, 0.0);

    double y = tsfr / tcool * egyhot / (sfr_params.FactorSN * sfr_params.EgySpecSN - (1 - sfr_params.FactorSN) * sfr_params.EgySpecCold);
    double x = 1 + 1 / (2 * y) - sqrt(1 / y + 1 / (4 * y * y));
    return egyhot * (1 - x) + sfr_params.EgySpecCold * x;
}

void init_cooling_and_star_formation(void)
{
    struct cooling_units coolunits;
    coolunits.CoolingOn = All.CoolingOn;
    coolunits.density_in_phys_cgs = All.UnitDensity_in_cgs * All.CP.HubbleParam * All.CP.HubbleParam;
    coolunits.uu_in_cgs = All.UnitEnergy_in_cgs / All.UnitMass_in_g;
    coolunits.tt_in_s = All.UnitTime_in_s / All.CP.HubbleParam;

    /* mean molecular weight assuming ZERO ionization NEUTRAL GAS*/
    double meanweight = 4.0 / (1 + 3 * HYDROGEN_MASSFRAC);

    /*Enforces a minimum internal energy in cooling. */
    All.MinEgySpec = 1 / meanweight * (1.0 / GAMMA_MINUS1) * (BOLTZMANN / PROTONMASS) * sfr_params.MinGasTemp / coolunits.uu_in_cgs;

    init_cooling(All.TreeCoolFile, All.MetalCoolFile, All.UVFluctuationFile, coolunits, &All.CP);

    if(!All.StarformationOn)
        return;

    sfr_params.OverDensThresh =
        sfr_params.CritOverDensity * All.CP.OmegaBaryon * 3 * All.CP.Hubble * All.CP.Hubble / (8 * M_PI * All.G);

    sfr_params.PhysDensThresh = sfr_params.CritPhysDensity * PROTONMASS / HYDROGEN_MASSFRAC / All.UnitDensity_in_cgs;

    sfr_params.EgySpecCold = 1 / meanweight * (1.0 / GAMMA_MINUS1) * (BOLTZMANN / PROTONMASS) * sfr_params.TempClouds;
    sfr_params.EgySpecCold *= All.UnitMass_in_g / All.UnitEnergy_in_cgs;

    /* mean molecular weight assuming FULL ionization */
    meanweight = 4 / (8 - 5 * (1 - HYDROGEN_MASSFRAC));

    sfr_params.EgySpecSN = 1 / meanweight * (1.0 / GAMMA_MINUS1) * (BOLTZMANN / PROTONMASS) * sfr_params.TempSupernova;
    sfr_params.EgySpecSN *= All.UnitMass_in_g / All.UnitEnergy_in_cgs;

    if(sfr_params.PhysDensThresh == 0)
    {
        double egyhot = sfr_params.EgySpecSN / sfr_params.FactorEVP;

        meanweight = 4 / (8 - 5 * (1 - HYDROGEN_MASSFRAC));	/* note: assuming FULL ionization */

        double u4 = 1 / meanweight * (1.0 / GAMMA_MINUS1) * (BOLTZMANN / PROTONMASS) * 1.0e4;
        u4 *= All.UnitMass_in_g / All.UnitEnergy_in_cgs;


        double dens = 1.0e6 * 3 * All.CP.Hubble * All.CP.Hubble / (8 * M_PI * All.G);

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

        sfr_params.PhysDensThresh =
            x / pow(1 - x,
                    2) * (sfr_params.FactorSN * sfr_params.EgySpecSN - (1 -
                            sfr_params.FactorSN) * sfr_params.EgySpecCold) /
                        (sfr_params.MaxSfrTimescale * coolrate);

        message(0, "A0= %g  \n", sfr_params.FactorEVP);
        message(0, "Computed: PhysDensThresh= %g  (int units)         %g h^2 cm^-3\n", sfr_params.PhysDensThresh,
                sfr_params.PhysDensThresh / (PROTONMASS / HYDROGEN_MASSFRAC / All.UnitDensity_in_cgs));
        message(0, "EXPECTED FRACTION OF COLD GAS AT THRESHOLD = %g\n", x);
        message(0, "tcool=%g dens=%g egyhot=%g\n", tcool, dens, egyhot);

        dens = sfr_params.PhysDensThresh * 10;

        double neff;
        do
        {
            double egyeff = get_egyeff(dens, &uvbg);

            double peff = GAMMA_MINUS1 * dens * egyeff;

            const double fac = 1 / (log(dens * 1.025) - log(dens));
            neff = -log(peff) * fac;

            dens *= 1.025;
            egyeff = get_egyeff(dens, &uvbg);
            peff = GAMMA_MINUS1 * dens * egyeff;

            neff += log(peff) * fac;
        }
        while(neff > 4.0 / 3);

        message(0, "Run-away sets in for dens=%g\n", dens);
        message(0, "Dynamic range for quiescent star formation= %g\n", dens / sfr_params.PhysDensThresh);

        const double sigma = 10.0 / All.CP.Hubble * 1.0e-10 / pow(1.0e-3, 2);

        message(0, "Isotherm sheet central density: %g   z0=%g\n",
                M_PI * All.G * sigma * sigma / (2 * GAMMA_MINUS1) / u4,
                GAMMA_MINUS1 * u4 / (2 * M_PI * All.G * sigma));

    }

    if(All.WindOn) {
        init_winds(sfr_params.FactorSN, sfr_params.EgySpecSN, sfr_params.PhysDensThresh);
    }

}

static double
find_star_mass(int i)
{
    /*Quick Lyman Alpha always turns all of a particle into stars*/
    if(sfr_params.QuickLymanAlphaProbability > 0)
        return P[i].Mass;

    double mass_of_star =  All.MassTable[0] / sfr_params.Generations;
    if(mass_of_star > P[i].Mass) {
        /* if some mass has been stolen by BH, e.g */
        mass_of_star = P[i].Mass;
    }
    /* if we are the last particle */
    if(fabs(mass_of_star - P[i].Mass) / mass_of_star < 0.5) {
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
static double ev_NH_from_GradRho(MyFloat gradrho[3], double hsml, double rho, double include_h)
{
    /* column density from GradRho, copied from gadget-p; what is it
     * calculating? */
    double gradrho_mag;
    if(rho<=0) {
        gradrho_mag = 0;
    } else {
        gradrho_mag = sqrt(gradrho[0]*gradrho[0]+gradrho[1]*gradrho[1]+gradrho[2]*gradrho[2]);
        if(gradrho_mag > 0) {gradrho_mag = rho*rho/gradrho_mag;} else {gradrho_mag=0;}
        if(include_h > 0) gradrho_mag += include_h*rho*hsml;
    }
    return gradrho_mag; // *(Z/Zsolar) add metallicity dependence
}

static double get_sfr_factor_due_to_h2(int i) {
    /*  Krumholz & Gnedin fitting function for f_H2 as a function of local
     *  properties, from gadget-p; we return the enhancement on SFR in this
     *  function */

    if(!SphP_scratch->GradRho)
        endrun(1, "Needed grad rho but not enabled! Should never happen!\n");
    double tau_fmol;
    double zoverzsun = SPHP(i).Metallicity/METAL_YIELD;
    tau_fmol = ev_NH_from_GradRho(&(SphP_scratch->GradRho[3*P[i].PI]),P[i].Hsml,SPHP(i).Density,1) * All.cf.a2inv;
    tau_fmol *= (0.1 + zoverzsun);
    if(tau_fmol>0) {
        tau_fmol *= 434.78*All.UnitDensity_in_cgs*All.CP.HubbleParam*All.UnitLength_in_cm;
        double y = 0.756*(1+3.1*pow(zoverzsun,0.365));
        y = log(1+0.6*y+0.01*y*y)/(0.6*tau_fmol);
        y = 1-0.75*y/(1+0.25*y);
        if(y<0) y=0;
        if(y>1) y=1;
        return y;

    } // if(tau_fmol>0)
    return 1.0;
}

static double get_sfr_factor_due_to_selfgravity(int i) {
    double divv = SPHP(i).DivVel * All.cf.a2inv;

    divv += 3.0*All.cf.hubble_a2; // hubble-flow correction

    if(HAS(sfr_params.StarformationCriterion, SFR_CRITERION_CONVERGENT_FLOW)) {
        if( divv>=0 ) return 0; // restrict to convergent flows (optional) //
    }

    double dv2abs = (divv*divv
            + (SPHP(i).CurlVel*All.cf.a2inv)
            * (SPHP(i).CurlVel*All.cf.a2inv)
           ); // all in physical units
    double alpha_vir = 0.2387 * dv2abs/(All.G * SPHP(i).Density*All.cf.a3inv);

    double y = 1.0;

    if((alpha_vir < 1.0)
    || (SPHP(i).Density * All.cf.a3inv > 100. * sfr_params.PhysDensThresh)
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
