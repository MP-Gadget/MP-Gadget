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

/* Need parameters: redshift at which start_reionization is called.
 *                  black hole mass parameter (accretion mass) specifies how clustered the quasars are.
 * start_reionization always switches on one quasar, then waits for the next one until HeIII fraction is high enough.
 *
 * */

#include <math.h>
#include <mpi.h>
#include <string.h>
#include <gsl/gsl_interp.h>
#include "cooling_qso_lightup.h"
#include "physconst.h"
#include "slotsmanager.h"
#include "partmanager.h"
#include "treewalk.h"
#include "drift.h"
#include "utils/endrun.h"
#include "utils/paramset.h"
#include "utils/mymalloc.h"

/*Parameters for the quasar driven helium reionization model.*/
struct qso_lightup_params
{
    int QSOLightupOn; /* Master flag enabling the helium reioization heating model.*/

    double qso_spectral_index; /* Quasar spectral index. Read from the text file. */
    double photon_threshold_energy; /* Photon threshold energy in eV, read from the text file*/

    double qso_candidate_min_mass; /* Minimum mass of a quasar black hole candidate.
                                  To become a quasar a black hole should have a mass between min and max. */
    double qso_candidate_max_mass; /* Minimum mass of a quasar black hole candidate.*/

    double mean_bubble; /* Mean size of the quasar bubble.*/
    double var_bubble; /* Variance of the quasar bubble size.*/

    double heIIIreion_start; /* Time at which start_reionization is called and helium III reionization begins*/
};

static struct qso_lightup_params QSOLightupParams;
/* Memory for the helium reionization history*/
static int Nreionhist;
static double * He_zz;
static double * XHeIII;
static double * LMFP;
static gsl_interp * HeIII_intp;
static gsl_interp * LMFP_intp;
/* Counters*/
static int N_ionized;
static double mass_ionized;


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
    }
    MPI_Bcast(&QSOLightupParams, sizeof(struct qso_lightup_params), MPI_BYTE, 0, MPI_COMM_WORLD);
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
    FILE * fd;
    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);

    if(ThisTask == 0) {
        fd = fopen(reion_hist_file, "r");
        if(!fd)
            endrun(456, "Could not open reionization history file at: '%s'\n", reion_hist_file);

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
        endrun(1, "Reionization history contains: %d entries, not enough.\n", Nreionhist);

    /*Allocate memory for the reionization history table.*/
    He_zz = mymalloc("ReionizationTable", 3 * Nreionhist * sizeof(double));
    XHeIII = He_zz + Nreionhist;
    LMFP = He_zz + 2 * Nreionhist;

    if(ThisTask == 0)
    {
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
                QSOLightupParams.qso_spectral_index = atof(retval);
                prei++;
                continue;
            }
            else if(prei == 1)
            {
                QSOLightupParams.photon_threshold_energy = atof(retval);
                prei++;
                continue;
            }
            /* First column: redshift*/
            He_zz[i] = atof(retval);
            /* Second column: HeIII fraction.*/
            retval = strtok(NULL, " \t");
            XHeIII[i] = atof(retval);
            /* Third column: long mean free path photons.*/
            retval = strtok(NULL, " \t");
            LMFP[i] = atof(retval);
            i++;
        }
        fclose(fd);
    }
    /*Broadcast data to other processors*/
    MPI_Bcast(He_zz, 3 * Nreionhist, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&QSOLightupParams.qso_spectral_index, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&QSOLightupParams.photon_threshold_energy, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    /* Initialize the interpolators*/
    HeIII_intp = gsl_interp_alloc(gsl_interp_linear,Nreionhist);
    LMFP_intp = gsl_interp_alloc(gsl_interp_linear,Nreionhist);
    gsl_interp_init(HeIII_intp, He_zz, XHeIII, Nreionhist);
    gsl_interp_init(HeIII_intp, He_zz, LMFP, Nreionhist);

    message(0, "Read %d lines z = %g - %g from file %s\n", Nreionhist, He_zz[0], He_zz[Nreionhist-1], reion_hist_file);
}

void
init_qso_lightup(char * reion_hist_file)
{
    load_heii_reion_hist(reion_hist_file);
    N_ionized = 0;
    mass_ionized = 0;
}

static double last_zz;
static double last_long_mfp_heating;

/* Get the long mean free path heating. in erg/s/cm^3
 * FIXME: Please check the units are correct! Should be */
double
get_long_mean_free_path_heating(double redshift)
{
    if(!QSOLightupParams.QSOLightupOn)
        return 0;
    if(redshift == last_zz)
        return last_long_mfp_heating;
    double long_mfp_heating = gsl_interp_eval(LMFP_intp, He_zz, LMFP, redshift, NULL);
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
choose_QSO_halo(int ncand, int nqsos, MPI_Comm Comm)
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

    sumup_large_ints(1, &ncand, &ncand_total);
    double drand = get_random_number(nqsos);
    int qso = (drand * ncand_total);
    /* No quasar on this processor*/
    if(qso < ncand_before && qso > ncand_before + ncand)
        return -1;

    /* If the quasar is on this processor, return the
     * index of the quasar in the current candidate array.*/
    return qso - ncand_before;
}

/* Proper emissivity of HeII ionizing photons per quasar per Gyr from Haardt & Madau (2012) (1105.2039.pdf eqn 37)*/
static double
quasar_emissivity_HM12(double redshift, double alpha_q)
{
    double mpctocm = 3.086e24;
    double h_erg_s = 6.626e-27; /* erg s */
    double enhance_fac = 1.0;
    double epsilon_nu = enhance_fac * 3.98e24*pow((1+redshift), 7.68)*exp(-0.28*redshift)/(exp(1.77*redshift)+ 26.3); /*erg s^-1 MPc^-3 Hz^-1*/
    double e = epsilon_nu/(h_erg_s*alpha_q)/(pow(mpctocm,3))*pow(4.,(-alpha_q));
    return e;
    }

/*Proper emissivity of HeII ionizing photons per quasar per Gyr from Khaire + (2015)*/
static double
quasar_emissivity_K15(double redshift, double alpha_q)
{
    double mpctocm = 3.086e24;
    double h_erg_s = 6.626e-27; /* erg s */
    double epsilon_nu = pow(10.,24.6)*pow((1+redshift), 8.9)*exp(-0.36*redshift)/(exp(2.2*redshift)+ 25.1); /*erg s^-1 MPc^-3 Hz^-1)*/
    double e = epsilon_nu/(h_erg_s*alpha_q)/(pow(mpctocm,3))*pow(4.,(-alpha_q));
    return e;
}

/* Calculates the total ionization fraction of the box. If this is less than the desired ionization fraction, returns 1.
 * Otherwise returns 0
 */
static int
need_more_quasars(double redshift)
{
    int64_t n_ionized_tot = 0, n_gas_tot = 0;
    /*int i, n_ionized = 0;
    for (i = 0; i < PartManager->NumPart; i++){
        if (P[i].Type == 0 && P[i].Ionized == 1){
            n_ionized ++;
        }
    }*/
    /* Get total ionization fraction: note this
     * includes all gas particles that have been ionized,
     * thus gas has become stars.*/
    sumup_large_ints(1, &N_ionized, &n_ionized_tot);
    sumup_large_ints(1, &SlotsManager->info[0].size, &n_gas_tot);
    double ionized_frac = (double) n_ionized_tot / (double) n_gas_tot;

    double desired_frac = gsl_interp_eval(HeIII_intp, He_zz, XHeIII, redshift, NULL);

    return ionized_frac < desired_frac;
}

static void
add_instantaneous_helium_heating(int other)
{
    /* Add some energy. FIXME: To implement.*/
    SPHP(other).Entropy += 0;
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

     /* This BH affects only non-ionized gas particles. */
    if(P[other].Type != 0 || P[other].Ionized != 0)
        return;

    struct SpinLocks * spin = QSO_GET_PRIV(lv->tw)->spin;

    lock_spinlock(other, spin);

    /* Mark it ionized*/
    P[other].Ionized = 1;

    /* Heat the particle*/
    add_instantaneous_helium_heating(other);

    unlock_spinlock(other, spin);

    /* Add to the ionization counter*/
    #pragma omp atomic
    N_ionized++;

    #pragma omp atomic
    mass_ionized += P[other].Mass;
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
 */
static void
ionize_all_part(int qso_ind, ForceTree * tree)
{
    /* This treewalk finds not yet ionized particles within the radius of the black hole, ionizes them and
     * adds an instantaneous heating to them. */
    TreeWalk tw[1] = {{0}};

    tw->ev_label = "HELIUM";
    /* This could select the black holes to be made quasars, but we do it below.*/
    tw->haswork = NULL;
    tw->tree = tree;

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
    treewalk_run(tw, &qso_ind, 1);
}

/* Sequentially turns on quasars.
 * Keeps adding new quasars until need_more_quasars() returns 0.
 */
static void
turn_on_quasars(double redshift, ForceTree * tree)
{
    int nqso;
    int * qso_cand;
    int ncand = build_qso_candidate_list(&qso_cand, &nqso);

    while (need_more_quasars(redshift)){
        /* Get a new quasar*/
        int new_qso = choose_QSO_halo(ncand, nqso, MPI_COMM_WORLD);
        ionize_all_part(new_qso, tree);
        /* Remove this candidate from the list by moving the list down.*/
        if( new_qso > 0) {
            memmove(qso_cand+new_qso, qso_cand+new_qso+1, ncand - new_qso);
            ncand--;
        }
    }
    myfree(qso_cand);
}

/* Starts reionization by selecting the first halo and flagging all particles in the first HeIII bubble*/
void
do_heiii_reionization(double redshift, ForceTree * tree)
{
    if(!QSOLightupParams.QSOLightupOn)
        return;
    if(redshift > QSOLightupParams.heIIIreion_start)
        return;

    message(0, "HeII Reionization initiated.");
    turn_on_quasars(redshift, tree);
}
