/** This file implements the quasar heating model of Upton-Sanderbeck et al 2019 (in prep).
 *  A black hole particle with the right mass is chosen at random and an ionizing bubble created around it.
 *  Particles within that bubble are marked as ionized and heated according to emissivity.
 *  New bubbles are created until the total HeIII fraction matches the value in the external table.
 *  There is also a uniform background heating rate for long mean-free-path photons with very high energies.
 *  This heating is added only to not-yet-ionized particles, and is done in cooling_rates.c. Whatever UVB you
 *  use should not include these photons.
 *
 * The text file contains the reionization history and is generated from various physical processes
 * implemented in a python file. The text file fixes the end of helium reionization (which is reasonably well-known).
 * The code has the start time of reionization as a free parameter: if this start of reionization is late then there
 * will be a strong sudden burst of heating.
 *
 * (Loosely) based on lightup_QSOs.py from https://github.com/uptonsanderbeck/helium_reionization
 * HeII_heating.py contains the details of the reionization history and how it is generated.
 *
 * This code should run only during a PM timestep, when all particles are active
 * We lose time resolution but I cannot think of another way to ensure cooling is modelled correctly.
 */

#include <math.h>
#include <mpi.h>
#include <string.h>
#include <gsl/gsl_interp.h>
#include "cooling_qso_lightup.h"
#include "physconst.h"
#include "slotsmanager.h"
#include "partmanager.h"
#include "treewalk.h"
#include "allvars.h"
#include "hydra.h"
#include "drift.h"
#include "walltime.h"
#include "utils/endrun.h"
#include "utils/paramset.h"
#include "utils/mymalloc.h"

#define E0_HeII 54.4 /* HeII ionization potential in eV*/
#define HEMASS 4.002602 /* Helium mass in amu*/

/*Parameters for the quasar driven helium reionization model.*/
struct qso_lightup_params
{
    int QSOLightupOn; /* Master flag enabling the helium reioization heating model.*/

    double qso_candidate_min_mass; /* Minimum mass of a quasar black hole candidate.
                                  To become a quasar a black hole should have a mass between min and max. */
    double qso_candidate_max_mass; /* Minimum mass of a quasar black hole candidate.*/

    double mean_bubble; /* Mean size of the quasar bubble.*/
    double var_bubble; /* Variance of the quasar bubble size.*/

    double heIIIreion_finish_frac; /* When the desired ionization fraction exceeds this value,
                                      the code will flash-ionize all remaining particles*/
    double heIIIreion_start; /* Time at which start_reionization is called and helium III reionization begins*/
};

static struct qso_lightup_params QSOLightupParams;
/* Memory for the helium reionization history. */
/* Instantaneous heating from low-energy (short mean free path)
 * photons to the Quasar to a newly ionized particle.
 * Computed from parameters stored in the text file.
 * In ergs.*/
static double qso_inst_heating;
static int Nreionhist;
static double * He_zz;
static double * XHeIII;
static double * LMFP;
static gsl_interp * HeIII_intp;
static gsl_interp * LMFP_intp;
/* Counter for particles ionized this step*/
static int N_ionized;

/*This is a helper for the tests*/
void set_qso_lightup_par(struct qso_lightup_params qso)
{
    QSOLightupParams = qso;
}

void
set_qso_lightup_params(ParameterSet * ps)
{
    int ThisTask;
    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
    if(ThisTask == 0) {
        QSOLightupParams.QSOLightupOn = param_get_int(ps, "QSOLightupOn");
        QSOLightupParams.qso_candidate_max_mass = param_get_double(ps, "QSOMaxMass");
        QSOLightupParams.qso_candidate_min_mass = param_get_double(ps, "QSOMinMass");
        QSOLightupParams.mean_bubble = param_get_double(ps, "QSOMeanBubble");
        QSOLightupParams.var_bubble = param_get_double(ps, "QSOVarBubble");
        QSOLightupParams.heIIIreion_start = param_get_double(ps, "QSOHeIIIReionStart");
        QSOLightupParams.heIIIreion_finish_frac = param_get_double(ps, "QSOHeIIIReionFinishFrac");
    }
    MPI_Bcast(&QSOLightupParams, sizeof(struct qso_lightup_params), MPI_BYTE, 0, MPI_COMM_WORLD);
}

/* Instantaneous heat injection from HeII reionization
 * (absorption of photons with E<Emax) per helium atom [ergs]
 */
static double
Q_inst(double Emax, double alpha_q)
{
    /* Total ionizing flux for the short mean free path photons*/
    double intflux = (pow(Emax,-alpha_q+1.)-pow(E0_HeII,-alpha_q+1.))/(pow(Emax,-alpha_q)- pow(E0_HeII,-alpha_q));
    /* Heating is input per unit mass, so quasars in denser areas will provide more heating.*/
    double Q_inst = (alpha_q/(alpha_q - 1.0))*intflux -E0_HeII;
    return eVinergs * Q_inst;
}

/* Load in reionization history file and build the interpolators between redshift and XHeII.
 * Format is:
 * Need to load a text file with the shape of reionization, which gives the overall HeIII fraction,
 * and the uniform background heating rate. Includes quasar spectral index, etc and most of the physics.
 * Be careful. Do not double-count uniform long mean free path photons with the homogeneous UVB.
 *
 * quasar spectral index
 * instantaneous absorption threshold energy (in eV)
 * table of 3 columns, redshift, HeIII fraction, uniform background heating.
 * The text file specifies the end redshift of reionization.
 * */
static void
load_heii_reion_hist(const char * reion_hist_file)
{
    int ThisTask;
    FILE * fd = NULL;
    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);

    message(0, "HeII: Loading HeII reionization history from file: %s\n",reion_hist_file);
    if(ThisTask == 0) {
        fd = fopen(reion_hist_file, "r");
        if(!fd)
            endrun(456, "HeII: Could not open reionization history file at: '%s'\n", reion_hist_file);

        /*Find size of file*/
        Nreionhist = 0;
        while(1)
        {
            char buffer[1024];
            char * retval = fgets(buffer, 1024, fd);
            /*Happens on end of file*/
            if(!retval)
                break;
            retval = strtok(buffer, " \t");
            /*Discard comments*/
            if(!retval || retval[0] == '#')
                continue;
            Nreionhist++;
        }
        rewind(fd);
        /* Discard first two lines*/
        Nreionhist -=2;
    }

    MPI_Bcast(&Nreionhist, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if(Nreionhist<= 2)
        endrun(1, "HeII: Reionization history contains: %d entries, not enough.\n", Nreionhist);

    /*Allocate memory for the reionization history table.*/
    He_zz = mymalloc("ReionizationTable", 3 * Nreionhist * sizeof(double));
    XHeIII = He_zz + Nreionhist;
    LMFP = He_zz + 2 * Nreionhist;

    if(ThisTask == 0)
    {
        double qso_spectral_index = 0, photon_threshold_energy = 0;
        int prei = 0;
        int i = 0;
        while(i < Nreionhist)
        {
            char buffer[1024];
            char * line = fgets(buffer, 1024, fd);
            /*Happens on end of file*/
            if(!line)
                break;
            char * retval = strtok(line, " \t");
            if(!retval || retval[0] == '#')
                continue;
            if(prei == 0)
            {
                qso_spectral_index = atof(retval);
                prei++;
                continue;
            }
            else if(prei == 1)
            {
                photon_threshold_energy = atof(retval);
                prei++;
                continue;
            }
            /* First column: redshift. Convert to scale factor so it is increasing.*/
            He_zz[i] = 1./(1+atof(retval));
            /* Second column: HeIII fraction.*/
            retval = strtok(NULL, " \t");
            XHeIII[i] = atof(retval);
            /* Third column: long mean free path photons.*/
            retval = strtok(NULL, " \t");
            LMFP[i] = atof(retval);
            i++;
        }
        fclose(fd);
        qso_inst_heating = Q_inst(photon_threshold_energy, qso_spectral_index);
    }
    /*Broadcast data to other processors*/
    MPI_Bcast(He_zz, 3 * Nreionhist, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&qso_inst_heating, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    /* Initialize the interpolators*/
    HeIII_intp = gsl_interp_alloc(gsl_interp_linear,Nreionhist);
    LMFP_intp = gsl_interp_alloc(gsl_interp_linear,Nreionhist);
    gsl_interp_init(HeIII_intp, He_zz, XHeIII, Nreionhist);
    gsl_interp_init(HeIII_intp, He_zz, LMFP, Nreionhist);

    message(0, "Read %d lines z = %g - %g from file %s\n", Nreionhist, 1/He_zz[0] -1, 1/He_zz[Nreionhist-1]-1, reion_hist_file);
}

void
init_qso_lightup(char * reion_hist_file)
{
    if(QSOLightupParams.QSOLightupOn)
        load_heii_reion_hist(reion_hist_file);
    N_ionized = 0;
}

static double last_zz;
static double last_long_mfp_heating;

/* Get the long mean free path heating. in erg/s/cm^3
 * FIXME: Please check the units are correct!*/
double
get_long_mean_free_path_heating(double redshift)
{
    if(!QSOLightupParams.QSOLightupOn)
        return 0;
    if(redshift == last_zz)
        return last_long_mfp_heating;
    double atime = 1/(1+redshift);
    double long_mfp_heating = gsl_interp_eval(LMFP_intp, He_zz, LMFP, atime, NULL);
    last_zz = redshift;
    last_long_mfp_heating = long_mfp_heating;
    return long_mfp_heating;
}

/* This function gets a random number from a Gaussian distribution using the Box-Muller transform.*/
static double gaussian_rng(double mu, double sigma, const int64_t seed)
{
    double u1 = get_random_number(seed);
    double u2 = get_random_number(seed + 1);
    double z1 = sqrt(-2 * log(u1) ) * cos(2 * M_PI * u2);
    return mu + sigma * z1;
}

/* Build a list of black hole particles which are candidates for becoming a quasar.
 * This excludes black hole particles which have already been quasars and which have the wrong mass.*/
static int
build_qso_candidate_list(int ** qso_cand, int * nqso)
{
    /*Loop over all black holes, building the candidate list.*/
    int i, ncand=0;
    const int nbh = SlotsManager->info[5].size;
    *qso_cand = mymalloc("Quasar_candidates", sizeof(int) * nbh);
    for(i = 0; i < PartManager->NumPart; i++)
    {
        /* Only want black holes*/
        if(P[i].Type != 5 )
            continue;
        /* Count how many particles which are already quasars we have*/
        if(BHP(i).QuasarTime > 0)
            (*nqso)++;
        /* Check that it has the right mass*/
        if(BHP(i).Mass < QSOLightupParams.qso_candidate_min_mass)
            continue;
        if(BHP(i).Mass > QSOLightupParams.qso_candidate_max_mass)
            continue;
        /*Add to the candidate list*/
        (*qso_cand)[ncand] = i;
        ncand++;
    }
    /*Poison value at the end for safety.*/
    (*qso_cand)[ncand] = -1;
    *qso_cand = myrealloc(*qso_cand, (ncand+1) * sizeof(int));
    return ncand;
}

/* Choose a black hole particle at random to host a quasar from the list of black hole particles.
 * This function chooses a single quasar from the total candidate list of quasars on *all* processors.
 * We seed the random number generator off the number of existing quasars.
 * This is done carefully so that we get the same sequence of quasars irrespective of how many processors we are using.
 *
 * FIXME: Ideally we would choose quasars in parallel.
 */
static int
choose_QSO_halo(int ncand, int nqsos, int64_t * ncand_tot, MPI_Comm Comm)
{
    int64_t ncand_total = 0, ncand_before = 0;
    int NTask, i, ThisTask;
    MPI_Comm_size(Comm, &NTask);
    MPI_Comm_rank(Comm, &ThisTask);
    int * candcounts = (int*) ta_malloc("qso_cand_counts", int, NTask);

    /* Get how many candidates are on each processor*/
    MPI_Alltoall(&ncand, 1, MPI_INT, candcounts, 1, MPI_INT, Comm);

    for(i = 0; i < NTask; i++)
    {
        if(i < ThisTask)
            ncand_before += candcounts[i];
        ncand_total += candcounts[i];
    }

    double drand = get_random_number(nqsos);
    int qso = (drand * ncand_total);
    *ncand_tot = ncand_total;
    /* No quasar on this processor*/
    if(qso < ncand_before || qso > ncand_before + ncand)
        return -1;

    /* If the quasar is on this processor, return the
     * index of the quasar in the current candidate array.*/
    return qso - ncand_before;
}

/* Calculates the total ionization fraction of the box.
 */
static double
gas_ionization_fraction(void)
{
    int64_t n_ionized_tot = 0, n_gas_tot = 0;
    int i, n_ionized = 0;
    #pragma omp parallel for reduction(+:n_ionized)
    for (i = 0; i < PartManager->NumPart; i++){
        if (P[i].Type == 0 && P[i].Ionized == 1){
            n_ionized ++;
        }
    }
    /* Get total ionization fraction: note this is only the current gas particles.
     * Particles that become stars are not counted.*/
    sumup_large_ints(1, &n_ionized, &n_ionized_tot);
    sumup_large_ints(1, &SlotsManager->info[0].size, &n_gas_tot);
    return (double) n_ionized_tot / (double) n_gas_tot;
}

/* Do the ionization for a single particle, marking it and adding the heat.
 * No locking is performed so ensure the particle is not being edited in parallel.
 * Returns 1 if ionization was done, 0 otherwise.*/
static int
ionize_single_particle(int other)
{
    /* Only ionize things once*/
    if(P[other].Ionized != 0)
        return 0;

    /* Mark it ionized*/
    P[other].Ionized = 1;

    /* Heat the particle*/

    /* Number of helium atoms per g in the particle*/
    double nheperg = (1 - HYDROGEN_MASSFRAC) / (PROTONMASS * HEMASS);
    /* Total heating per unit mass ergs/g for the particle*/
    double deltau = qso_inst_heating * nheperg;
    double uu_in_cgs = All.UnitEnergy_in_cgs / All.UnitMass_in_g;

    /* Conversion factor between internal energy and entropy.*/
    double utoentropy = GAMMA_MINUS1 * pow(SPH_EOMDensity(other) * All.cf.a3inv, GAMMA_MINUS1);
    /* Convert to entropy in internal units*/
    SPHP(other).Entropy += utoentropy * deltau / uu_in_cgs;
    return 1;
}

struct QSOPriv {
    /* Particle SpinLocks*/
    struct SpinLocks * spin;
};
#define QSO_GET_PRIV(tw) ((struct QSOPriv *) (tw->priv))

/**
 * Ionize and heat the particles
 */
static void
ionize_ngbiter(TreeWalkQueryBase * I,
        TreeWalkResultBase * O,
        TreeWalkNgbIterBase * iter,
        LocalTreeWalk * lv)
{

    if(iter->other == -1) {
        /* Gas only*/
        iter->mask = 1;
        /* Bubble size*/
        double bubble = gaussian_rng(QSOLightupParams.mean_bubble, sqrt(QSOLightupParams.var_bubble), I->ID);
        iter->Hsml = bubble;
        /* Don't care about gas HSML */
        iter->symmetric = NGB_TREEFIND_ASYMMETRIC;
        return;
    }

    int other = iter->other;

    /* Only ionize gas*/
    if(P[other].Type != 0)
        return;

    struct SpinLocks * spin = QSO_GET_PRIV(lv->tw)->spin;

    lock_spinlock(other, spin);

    int ionized = ionize_single_particle(other);

    unlock_spinlock(other, spin);

    if(!ionized)
        return;

    /* Add to the ionization counter*/
    #pragma omp atomic
    N_ionized ++;
}

static void
ionize_copy(int place, TreeWalkQueryBase * I, TreeWalk * tw)
{
    int k;
    for(k = 0; k < 3; k++)
    {
        I->Pos[k] = P[place].Pos[k];
    }
    I->ID = P[place].ID;
}

/* Find all particles within the radius of the HeIII bubble,
 * flag each particle as ionized and add instantaneous heating.
 * Returns the number of particles ionized
 */
static int
ionize_all_part(int qso_ind, int * qso_cand, ForceTree * tree)
{
    /* This treewalk finds not yet ionized particles within the radius of the black hole, ionizes them and
     * adds an instantaneous heating to them. */
    TreeWalk tw[1] = {{0}};

    tw->ev_label = "HELIUM";
    /* This could select the black holes to be made quasars, but we do it below.*/
    tw->haswork = NULL;
    tw->tree = tree;

    /* Reset the particles ionized this step*/
    N_ionized = 0;
    /* We set Hsml to a constant in ngbiter, so this
     * searches a constant distance from the BH.*/
    tw->visit = (TreeWalkVisitFunction) treewalk_visit_ngbiter;
    tw->ngbiter_type_elsize = sizeof(TreeWalkNgbIterBase);
    tw->ngbiter = ionize_ngbiter;

    tw->fill = (TreeWalkFillQueryFunction) ionize_copy;
    tw->reduce = NULL;
    tw->postprocess = NULL;
    tw->query_type_elsize = sizeof(TreeWalkQueryBase);
    tw->result_type_elsize = sizeof(TreeWalkResultBase);

    struct QSOPriv priv[1];
    priv[0].spin = init_spinlocks(PartManager->NumPart);

    tw->priv = priv;

    /* This runs only on one BH*/
    if(qso_ind > 0)
        treewalk_run(tw, &qso_cand[qso_ind], 1);
    else
        treewalk_run(tw, NULL, 0);
    return N_ionized;
}

/* Sequentially turns on quasars.
 * Keeps adding new quasars until need_more_quasars() returns 0.
 */
static void
turn_on_quasars(double redshift, ForceTree * tree)
{
    int nqso=0;
    int * qso_cand;
    int ncand = build_qso_candidate_list(&qso_cand, &nqso);
    walltime_measure("/HeIII/Find");
    int64_t n_gas_tot=0, tot_n_ionized=0, ncand_tot=0;
    sumup_large_ints(1, &SlotsManager->info[0].size, &n_gas_tot);
    double atime = 1./(1 + redshift);
    double desired_ion_frac = gsl_interp_eval(HeIII_intp, He_zz, XHeIII, atime, NULL);
    /* If the desired ionization fraction is above a threshold (by default 0.95)
     * ionize all particles*/
    if(desired_ion_frac > QSOLightupParams.heIIIreion_finish_frac) {
        int i, nionized=0;
        int64_t nion_tot=0;
        #pragma omp parallel for reduction(+: nionized)
        for (i = 0; i < PartManager->NumPart; i++){
            if (P[i].Type == 0)
                nionized += ionize_single_particle(i);
        }
        sumup_large_ints(1, &nionized, &nion_tot);
        message(0, "HeII: Helium ionization finished, flash-ionizing %ld particles (%g of total)\n", nion_tot, (double) nion_tot /(double) n_gas_tot);
    }

    double rhobar = All.CP.OmegaBaryon * (3 * HUBBLE * All.CP.HubbleParam * HUBBLE * All.CP.HubbleParam)/ (8 * M_PI * GRAVITY) / pow(1 + redshift,3);
    double totbubblegasmass = 4 * M_PI / 3. * pow(QSOLightupParams.mean_bubble, 3) * rhobar;
    /* Total expected ionizations if the bubbles do not overlap at all
     * and the bubble is at mean density.*/
    int64_t non_overlapping_bubble_number = n_gas_tot * totbubblegasmass / All.CP.OmegaBaryon;
    double initionfrac = gas_ionization_fraction();
    double curionfrac = initionfrac;
    while (curionfrac < desired_ion_frac){
        /* Get a new quasar*/
        int new_qso = choose_QSO_halo(ncand, nqso, &ncand_tot, MPI_COMM_WORLD);
        /* Make sure someone has a quasar*/
        if(ncand_tot == 0) {
            if(desired_ion_frac - curionfrac > 0.1)
                message(0, "HeII: Ionization fraction %g less than desired ionization fraction of %g because not enough quasars\n", curionfrac, desired_ion_frac);
            break;
        }
        /* Do the ionizations with a tree walk*/
        int n_ionized = ionize_all_part(new_qso, qso_cand, tree);
        int64_t tot_qso_ionized = 0;
        /* Check that the ionization fraction changed*/
        sumup_large_ints(1, &n_ionized, &tot_qso_ionized);
        curionfrac += (double) tot_qso_ionized / (double) n_gas_tot;
        tot_n_ionized += tot_qso_ionized;
        if(new_qso > 0)
            message(1, "HeII: Quasar %d changed the HeIII ionization fraction to %g, ionizing %ld\n", qso_cand[new_qso], curionfrac, tot_qso_ionized);
        /* Break the loop if we do not ionize enough particles this round.
         * Try again next timestep when we will hopefully have new BHs.*/
        if(tot_qso_ionized < 0.01 * non_overlapping_bubble_number)
            break;
        /* Remove this candidate from the list by moving the list down.*/
        if( new_qso >= 0) {
            memmove(qso_cand+new_qso, qso_cand+new_qso+1, ncand - new_qso+1);
            ncand--;
        }
    }
    myfree(qso_cand);
    message(0, "HeII: Reionization changed the HeIII fraction from %g -> %g, ionizing %ld\n", initionfrac, curionfrac, tot_n_ionized);
    walltime_measure("/HeIII/Ionize");
}

/* Starts reionization by selecting the first halo and flagging all particles in the first HeIII bubble*/
void
do_heiii_reionization(double redshift, ForceTree * tree)
{
    if(!QSOLightupParams.QSOLightupOn)
        return;
    if(redshift > QSOLightupParams.heIIIreion_start)
        return;

    walltime_measure("/Misc");
    //message(0, "HeII: Reionization initiated.\n");
    turn_on_quasars(redshift, tree);
}
