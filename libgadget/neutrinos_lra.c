/**\file
 * Contains calculations for the Fourier-space semi-linear neutrino method
 * described in Ali-Haimoud and Bird 2012.
 * delta_tot_table stores the state of the integrator, which includes the matter power spectrum over all past time.
 * This file contains routines for manipulating this structure; updating it by computing a new neutrino power spectrum,
 * from the non-linear CDM power.
 */

#include <math.h>
#include <string.h>
#include <bigfile-mpi.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_interp.h>
#include <gsl/gsl_sf_bessel.h>

#include "neutrinos_lra.h"

#include "utils/endrun.h"
#include "utils/mymalloc.h"
#include "utils/string.h"
#include "petaio.h"
#include "cosmology.h"
#include "powerspectrum.h"
#include "physconst.h"
#include "timefac.h"

/** Floating point accuracy*/
#define FLOAT_ACC   1e-6
/** Number of bins in integrations*/
#define GSL_VAL 400

/** Update the last value of delta_tot in the table with a new value computed
 from the given delta_cdm_curr and delta_nu_curr.
 If overwrite is true, overwrite the existing final entry.*/
void update_delta_tot(_delta_tot_table * const d_tot, const double a, const double delta_cdm_curr[], const double delta_nu_curr[], const int overwrite);

/** Main function: given tables of wavenumbers, total delta at Na earlier times (< = a),
 * and initial conditions for neutrinos, computes the current delta_nu.
 * @param d_tot Initialised structure for storing total matter density.
 * @param a Current scale factor.
 * @param delta_nu_curr Pointer to array to store square root of neutrino power spectrum. Main output.
 * @param mnu Neutrino mass in eV.*/
void get_delta_nu(Cosmology * CP, const _delta_tot_table * const d_tot, const double a, double delta_nu_curr[], const double mnu);

/** Function which wraps three get_delta_nu calls to get delta_nu three times,
 * so that the final value is for all neutrino species*/
void get_delta_nu_combined(Cosmology * CP, const _delta_tot_table * const d_tot, const double a, double delta_nu_curr[]);

/** Fit to the special function J(x) that is accurate to better than 3% relative and 0.07% absolute*/
double specialJ(const double x, const double vcmnubylight, const double nufrac_low);

/** Free-streaming length (times Mnu/k_BT_nu, which is dimensionless) for a non-relativistic
particle of momentum q = T0, from scale factor ai to af.
Arguments:
@param logai log of initial scale factor
@param logaf log of final scale factor
@param mnu Neutrino mass in eV
@param light speed of light in internal length units.
@returns free-streaming length in Unit_Length/Unit_Time (same units as light parameter).
*/
double fslength(Cosmology * CP, const double logai, const double logaf, const double light);

/** Combine the CDM and neutrino power spectra together to get the total power.
 * OmegaNua3 = OmegaNu(a) * a^3
 * Omeganonu = Omega0 - OmegaNu(1)
 * Omeganu1 = OmegaNu(1) */
static inline double get_delta_tot(const double delta_nu_curr, const double delta_cdm_curr, const double OmegaNua3, const double Omeganonu, const double Omeganu1, const double particle_nu_fraction)
{
    const double fcdm = 1 - OmegaNua3/(Omeganonu + Omeganu1);
    return fcdm * (delta_cdm_curr + delta_nu_curr * OmegaNua3/(Omeganonu + Omeganu1*particle_nu_fraction));
}


/*Structure which holds the neutrino state*/
_delta_tot_table delta_tot_table;

/** Structure to store the initial transfer functions from CAMB.
 * We store transfer functions because we want to use the
 * CDM + Baryon total matter power spectrum from the
 * first timestep of Gadget, so that possible Rayleigh scattering
 * in the initial conditions is included in the neutrino and radiation components. */
struct _transfer_init_table {
    int NPowerTable;
    double *logk;
    /*This is T_nu / (T_not-nu), where T_not-nu is a weighted average of T_cdm and T_baryon*/
    double *T_nu;
};
typedef struct _transfer_init_table _transfer_init_table;

static _transfer_init_table t_init_data;
static _transfer_init_table * t_init = &t_init_data;

/* Constructor. transfer_init_tabulate must be called before this function.
 * Initialises delta_tot (including from a file) and delta_nu_init from the transfer functions.
 * read_all_nu_state must be called before this if you want reloading from a snapshot to work
 * Note delta_cdm_curr includes baryons, and is only used if not resuming.*/
static void delta_tot_first_init(_delta_tot_table * const d_tot, const int nk_in, const double wavenum[], const double delta_cdm_curr[], const double TimeIC)
{
    int ik;
    d_tot->nk=nk_in;
    const double OmegaNua3=get_omega_nu_nopart(d_tot->omnu, d_tot->TimeTransfer)*pow(d_tot->TimeTransfer,3);
    const double OmegaNu1 = get_omega_nu(d_tot->omnu, 1);
    gsl_interp_accel *acc = gsl_interp_accel_alloc();
    gsl_interp * spline;
    if(t_init->NPowerTable > 2)
        spline = gsl_interp_alloc(gsl_interp_cspline,t_init->NPowerTable);
    else
        spline = gsl_interp_alloc(gsl_interp_linear,t_init->NPowerTable);
    gsl_interp_init(spline,t_init->logk,t_init->T_nu,t_init->NPowerTable);
    /*Check we have a long enough power table: power tables are in log_10*/
    if(log10(wavenum[d_tot->nk-1]) > t_init->logk[t_init->NPowerTable-1])
        endrun(2,"Want k = %g but maximum in CLASS table is %g\n",wavenum[d_tot->nk-1], pow(10, t_init->logk[t_init->NPowerTable-1]));
    for(ik=0;ik<d_tot->nk;ik++) {
            /* T_nu contains T_nu / T_cdm.*/
            double T_nubyT_nonu = gsl_interp_eval(spline,t_init->logk,t_init->T_nu,log10(wavenum[ik]),acc);
            /*Initialise delta_nu_init to use the first timestep's delta_cdm_curr
             * so that it includes potential Rayleigh scattering. */
            d_tot->delta_nu_init[ik] = delta_cdm_curr[ik]*T_nubyT_nonu;
            const double partnu = particle_nu_fraction(&d_tot->omnu->hybnu, TimeIC, 0);
            /*Initialise the first delta_tot*/
            d_tot->delta_tot[ik][0] = get_delta_tot(d_tot->delta_nu_init[ik], delta_cdm_curr[ik], OmegaNua3, d_tot->Omeganonu, OmegaNu1, partnu);
            /*Set up the wavenumber array*/
            d_tot->wavenum[ik] = wavenum[ik];
    }
    gsl_interp_accel_free(acc);
    gsl_interp_free(spline);

    /*If we are not restarting, make sure we set the scale factor*/
    d_tot->scalefact[0]=log(TimeIC);
    d_tot->ia=1;
    return;
}

void delta_nu_from_power(struct _powerspectrum * PowerSpectrum, Cosmology * CP, const double Time, const double TimeIC)
{
    int i;
    /*This is done on the first timestep: we need nk_nonzero for it to work.*/
    if(!delta_tot_table.delta_tot_init_done) {
        if(delta_tot_table.ia == 0) {
            /* Compute delta_nu from the transfer functions if first entry.*/
            delta_tot_first_init(&delta_tot_table, PowerSpectrum->nonzero, PowerSpectrum->kk, PowerSpectrum->Power, TimeIC);
        }

        /*Initialise the first delta_nu*/
        get_delta_nu_combined(CP, &delta_tot_table, exp(delta_tot_table.scalefact[delta_tot_table.ia-1]), delta_tot_table.delta_nu_last);
        delta_tot_table.delta_tot_init_done = 1;
    }
    for(i = 0; i < PowerSpectrum->nonzero; i++)
        PowerSpectrum->logknu[i] = log(PowerSpectrum->kk[i]);

    double * Power_in = PowerSpectrum->Power;
    /* Rebin the input power if necessary*/
    if(delta_tot_table.nk != PowerSpectrum->nonzero) {
        Power_in = (double *) mymalloc("pkint", delta_tot_table.nk * sizeof(double));
        double * logPower = (double *) mymalloc("logpk", PowerSpectrum->nonzero * sizeof(double));
        for(i = 0; i < PowerSpectrum->nonzero; i++)
            logPower[i] = log(PowerSpectrum->Power[i]);
        gsl_interp * pkint = gsl_interp_alloc(gsl_interp_linear, PowerSpectrum->nonzero);
        gsl_interp_init(pkint, PowerSpectrum->logknu, logPower, PowerSpectrum->nonzero);
        gsl_interp_accel * pkacc = gsl_interp_accel_alloc();
        for(i = 0; i < delta_tot_table.nk; i++) {
            double logk = log(delta_tot_table.wavenum[i]);
            if(pkint->xmax < logk || pkint->xmin > logk)
                Power_in[i] = delta_tot_table.delta_tot[i][delta_tot_table.ia-1];
            else
                Power_in[i] = exp(gsl_interp_eval(pkint, PowerSpectrum->logknu, logPower, logk, pkacc));
        }
        myfree(logPower);
        gsl_interp_accel_free(pkacc);
        gsl_interp_free(pkint);
    }

    const double partnu = particle_nu_fraction(&CP->ONu.hybnu, Time, 0);
    /* If we get called twice with the same scale factor, do nothing: delta_nu
     * already stores the neutrino power from the current timestep.*/
    if(1 - partnu > 1e-3 && log(Time)-delta_tot_table.scalefact[delta_tot_table.ia-1] > FLOAT_ACC) {
        /*We need some estimate for delta_tot(current time) to obtain delta_nu(current time).
            Even though delta_tot(current time) is not directly used (the integrand vanishes at a = a(current)),
            it is indeed needed for interpolation */
        /*It was checked that using delta_tot(current time) = delta_cdm(current time) leads to no more than 2%
          error on delta_nu (and moreover for large k). Using the last timestep's delta_nu decreases error even more.
          So we only need one step. */
        /*This increments the number of stored spectra, although the last one is not yet final.*/
        update_delta_tot(&delta_tot_table, Time, Power_in, delta_tot_table.delta_nu_last, 0);
        /*Get the new delta_nu_curr*/
        get_delta_nu_combined(CP, &delta_tot_table, Time, delta_tot_table.delta_nu_last);
        /* Decide whether we save the current time or not */
        if (Time > exp(delta_tot_table.scalefact[delta_tot_table.ia-2]) + 0.009) {
            /* If so update delta_tot(a) correctly, overwriting current power spectrum */
            update_delta_tot(&delta_tot_table, Time, Power_in, delta_tot_table.delta_nu_last, 1);
        }
        /*Otherwise discard the last powerspectrum*/
        else
            delta_tot_table.ia--;

        message(0,"Done getting neutrino power: nk = %d, k = %g, delta_nu = %g, delta_cdm = %g,\n", delta_tot_table.nk, delta_tot_table.wavenum[1], delta_tot_table.delta_nu_last[1], Power_in[1]);
        /*kspace_prefac = M_nu (analytic) / M_particles */
        const double OmegaNu_nop = get_omega_nu_nopart(&CP->ONu, Time);
        const double omega_hybrid = get_omega_nu(&CP->ONu, 1) * partnu / pow(Time, 3);
        /* Omega0 - Omega in neutrinos + Omega in particle neutrinos = Omega in particles*/
        PowerSpectrum->nu_prefac = OmegaNu_nop/(delta_tot_table.Omeganonu/pow(Time,3) + omega_hybrid);
    }
    double * delta_nu_ratio = (double *) mymalloc2("dnu_rat", delta_tot_table.nk * sizeof(double));
    double * logwavenum = (double *) mymalloc2("logwavenum", delta_tot_table.nk * sizeof(double));
    gsl_interp * pkint = gsl_interp_alloc(gsl_interp_linear, delta_tot_table.nk);
    gsl_interp_accel * pkacc = gsl_interp_accel_alloc();
    /*We want to interpolate in log space*/
    for(i=0; i < delta_tot_table.nk; i++) {
        if(isnan(delta_tot_table.delta_nu_last[i]))
            endrun(2004,"delta_nu_curr=%g i=%d delta_cdm_curr=%g kk=%g\n",delta_tot_table.delta_nu_last[i],i,Power_in[i],delta_tot_table.wavenum[i]);
        /*Enforce positivity for sanity reasons*/
        if(delta_tot_table.delta_nu_last[i] < 0)
            delta_tot_table.delta_nu_last[i] = 0;
        delta_nu_ratio[i] = delta_tot_table.delta_nu_last[i]/ Power_in[i];
        logwavenum[i] = log(delta_tot_table.wavenum[i]);
    }
    if(delta_tot_table.nk != PowerSpectrum->nonzero)
        myfree(Power_in);
    gsl_interp_init(pkint, logwavenum, delta_nu_ratio, delta_tot_table.nk);

    /*We want to interpolate in log space*/
    for(i=0; i < PowerSpectrum->nonzero; i++) {
        if(PowerSpectrum->nonzero == delta_tot_table.nk)
            PowerSpectrum->delta_nu_ratio[i] = delta_nu_ratio[i];
        else {
            double logk = PowerSpectrum->logknu[i];
            if(logk > pkint->xmax)
                logk = pkint->xmax;
            PowerSpectrum->delta_nu_ratio[i] = gsl_interp_eval(pkint, logwavenum, delta_nu_ratio, logk, pkacc);
        }
    }

    gsl_interp_accel_free(pkacc);
    gsl_interp_free(pkint);
    myfree(logwavenum);
    myfree(delta_nu_ratio);
}

/*Save the neutrino power spectrum to a file*/
void powerspectrum_nu_save(struct _powerspectrum * PowerSpectrum, const char * OutputDir, const char * filename, const double Time)
{
    int i;
    int ThisTask;
    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
    if(ThisTask != 0)
        return;

    char * fname = fastpm_strdup_printf("%s/%s-%0.4f.txt", OutputDir, filename, Time);
    /* Now save the neutrino power spectrum*/
    FILE * fp = fopen(fname, "w");
    fprintf(fp, "# in Mpc/h Units \n");
    fprintf(fp, "# k P_nu(k) Nmodes\n");
    fprintf(fp, "# a= %g\n", Time);
    fprintf(fp, "# nk = %d\n", PowerSpectrum->nonzero);
    for(i = 0; i < PowerSpectrum->nonzero; i++){
        fprintf(fp, "%g %g %ld\n", PowerSpectrum->kk[i], pow(delta_tot_table.delta_nu_last[i],2), PowerSpectrum->Nmodes[i]);
    }
    fclose(fp);
    myfree(fname);
    /*Clean up the neutrino memory now we saved the power spectrum.*/
    gsl_interp_free(PowerSpectrum->nu_spline);
    gsl_interp_accel_free(PowerSpectrum->nu_acc);
}

void petaio_save_neutrinos(BigFile * bf, int ThisTask)
{
#pragma omp master
    {
    double * scalefact = delta_tot_table.scalefact;
    size_t nk = delta_tot_table.nk, ia = delta_tot_table.ia;
    size_t ik, i;
    double * delta_tot = (double *) mymalloc("tmp_delta",nk * ia * sizeof(double));
    /*Save a flat memory block*/
    for(ik=0;ik< nk;ik++)
        for(i=0;i< ia;i++)
            delta_tot[ik*ia+i] = delta_tot_table.delta_tot[ik][i];

    BigBlock bn;
    if(0 != big_file_mpi_create_block(bf, &bn, "Neutrino", NULL, 0, 0, 0, MPI_COMM_WORLD)) {
        endrun(0, "Failed to create block at %s:%s\n", "Neutrino",
                big_file_get_error_message());
    }
    if ( (0 != big_block_set_attr(&bn, "Nscale", &ia, "u8", 1)) ||
       (0 != big_block_set_attr(&bn, "scalefact", scalefact, "f8", ia)) ||
        (0 != big_block_set_attr(&bn, "Nkval", &nk, "u8", 1)) ) {
        endrun(0, "Failed to write neutrino attributes %s\n",
                    big_file_get_error_message());
    }
    if(0 != big_block_mpi_close(&bn, MPI_COMM_WORLD)) {
        endrun(0, "Failed to close block %s\n",
                    big_file_get_error_message());
    }
    BigArray deltas = {0};
    size_t dims[2] = {0 , ia};
    /*The neutrino state is shared between all processors,
     *so only write on master task*/
    if(ThisTask == 0) {
        dims[0] = nk;
    }
    ptrdiff_t strides[2] = {(ptrdiff_t) (sizeof(double) * ia), (ptrdiff_t) sizeof(double)};
    big_array_init(&deltas, delta_tot, "=f8", 2, dims, strides);
    petaio_save_block(bf, "Neutrino/Deltas", &deltas, 1);
    myfree(delta_tot);
    /*Now write the initial neutrino power*/
    BigArray delta_nu = {0};
    dims[1] = 1;
    strides[0] = sizeof(double);
    big_array_init(&delta_nu, delta_tot_table.delta_nu_init, "=f8", 2, dims, strides);
    petaio_save_block(bf, "Neutrino/DeltaNuInit", &delta_nu, 1);
    /*Now write the k values*/
    BigArray kvalue = {0};
    big_array_init(&kvalue, delta_tot_table.wavenum, "=f8", 2, dims, strides);
    petaio_save_block(bf, "Neutrino/kvalue", &kvalue, 1);
    }
}

void petaio_read_icnutransfer(BigFile * bf, int ThisTask)
{
#pragma omp master
    {
    t_init->NPowerTable = 2;
    BigBlock bn;
    /* Read the size of the ICTransfer block.
     * If we can't read it, just set it to zero*/
    if(0 == big_file_mpi_open_block(bf, &bn, "ICTransfers", MPI_COMM_WORLD)) {
        if(0 != big_block_get_attr(&bn, "Nentry", &t_init->NPowerTable, "u8", 1))
            endrun(0, "Failed to read attr: %s\n", big_file_get_error_message());
        if(0 != big_block_mpi_close(&bn, MPI_COMM_WORLD))
            endrun(0, "Failed to close block %s\n",big_file_get_error_message());
    }
    message(0,"Found transfer function, using %d rows.\n", t_init->NPowerTable);
    t_init->logk = (double *) mymalloc2("Transfer_functions", sizeof(double) * 2*t_init->NPowerTable);
    t_init->T_nu=t_init->logk+t_init->NPowerTable;

    /*Defaults: a very small value*/
    t_init->logk[0] = -100;
    t_init->logk[t_init->NPowerTable-1] = 100;

    t_init->T_nu[0] = 1e-30;
    t_init->T_nu[t_init->NPowerTable-1] = 1e-30;

    /*Now read the arrays*/
    BigArray Tnu = {0};
    BigArray logk = {0};

    size_t dims[2] = {0, 1};
    ptrdiff_t strides[2] = {sizeof(double), sizeof(double)};
    /*The neutrino state is shared between all processors,
     *so only read on master task and broadcast*/
    if(ThisTask == 0) {
        dims[0] = t_init->NPowerTable;
    }
    big_array_init(&Tnu, t_init->T_nu, "=f8", 2, dims, strides);
    big_array_init(&logk, t_init->logk, "=f8", 2, dims, strides);
    /*This is delta_nu / delta_tot: note technically the most massive eigenstate only.
     * But if there are significant differences the mass is so low this is basically zero.*/
    petaio_read_block(bf, "ICTransfers/DELTA_NU", &Tnu, 0);
    petaio_read_block(bf, "ICTransfers/logk", &logk, 0);
    /*Also want d_{cdm+bar} / d_tot so we can get d_nu/(d_cdm+d_b)*/
    double * T_cb = (double *) mymalloc("tmp1", t_init->NPowerTable* sizeof(double));
    T_cb[0] = 1;
    T_cb[t_init->NPowerTable-1] = 1;
    BigArray Tcb = {0};
    big_array_init(&Tcb, T_cb, "=f8", 2, dims, strides);
    petaio_read_block(bf, "ICTransfers/DELTA_CB", &Tcb, 0);
    int i;
    for(i = 0; i < t_init->NPowerTable; i++)
        t_init->T_nu[i] /= T_cb[i];
    myfree(T_cb);
    /*Broadcast the arrays.*/
    MPI_Bcast(t_init->logk,2*t_init->NPowerTable,MPI_DOUBLE,0,MPI_COMM_WORLD);
    }
}

/*Read the neutrino data from the snapshot*/
void petaio_read_neutrinos(BigFile * bf, int ThisTask)
{
#pragma omp master
    {
    size_t nk, ia, ik, i;
    BigBlock bn;
    if(0 != big_file_mpi_open_block(bf, &bn, "Neutrino", MPI_COMM_WORLD)) {
        endrun(0, "Failed to open block at %s:%s\n", "Neutrino",
                    big_file_get_error_message());
    }
    if(
    (0 != big_block_get_attr(&bn, "Nscale", &ia, "u8", 1)) ||
    (0 != big_block_get_attr(&bn, "Nkval", &nk, "u8", 1))) {
        endrun(0, "Failed to read attr: %s\n",
                    big_file_get_error_message());
    }
    double *delta_tot = (double *) mymalloc("tmp_nusave",ia*nk*sizeof(double));
    /*Allocate list of scale factors, and space for delta_tot, in one operation.*/
    if(0 != big_block_get_attr(&bn, "scalefact", delta_tot_table.scalefact, "f8", ia))
        endrun(0, "Failed to read attr: %s\n", big_file_get_error_message());
    if(0 != big_block_mpi_close(&bn, MPI_COMM_WORLD)) {
        endrun(0, "Failed to close block %s\n",
                    big_file_get_error_message());
    }
    BigArray deltas = {0};
    size_t dims[2] = {0, ia};
    ptrdiff_t strides[2] = {(ptrdiff_t) (sizeof(double)*ia), (ptrdiff_t)sizeof(double)};
    /*The neutrino state is shared between all processors,
     *so only read on master task and broadcast*/
    if(ThisTask == 0) {
        dims[0] = nk;
    }
    big_array_init(&deltas, delta_tot, "=f8", 2, dims, strides);
    petaio_read_block(bf, "Neutrino/Deltas", &deltas, 1);
    if(nk > 1Lu*delta_tot_table.nk_allocated || ia > 1Lu*delta_tot_table.namax)
        endrun(5, "Allocated nk %d na %d for neutrino power but need nk %ld na %ld\n", delta_tot_table.nk_allocated, delta_tot_table.namax, nk, ia);
    /*Save a flat memory block*/
    for(ik=0;ik<nk;ik++)
        for(i=0;i<ia;i++)
            delta_tot_table.delta_tot[ik][i] = delta_tot[ik*ia+i];
    delta_tot_table.nk = nk;
    delta_tot_table.ia = ia;
    myfree(delta_tot);
    /* Read the initial delta_nu. This is basically zero anyway,
     * so for backwards compatibility do not require it*/
    BigArray delta_nu = {0};
    dims[1] = 1;
    strides[0] = sizeof(double);
    memset(delta_tot_table.delta_nu_init, 0, delta_tot_table.nk);
    big_array_init(&delta_nu, delta_tot_table.delta_nu_init, "=f8", 2, dims, strides);
    petaio_read_block(bf, "Neutrino/DeltaNuInit", &delta_nu, 0);
    /* Read the k values*/
    BigArray kvalue = {0};
    memset(delta_tot_table.wavenum, 0, delta_tot_table.nk);
    big_array_init(&kvalue, delta_tot_table.wavenum, "=f8", 2, dims, strides);
    petaio_read_block(bf, "Neutrino/kvalue", &kvalue, 0);

    /*Broadcast the arrays.*/
    MPI_Bcast(&(delta_tot_table.ia), 1,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Bcast(&(delta_tot_table.nk), 1,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Bcast(delta_tot_table.delta_nu_init,delta_tot_table.nk,MPI_DOUBLE,0,MPI_COMM_WORLD);
    MPI_Bcast(delta_tot_table.wavenum,delta_tot_table.nk,MPI_DOUBLE,0,MPI_COMM_WORLD);

    if(delta_tot_table.ia > 0) {
        /*Broadcast data for scalefact and delta_tot, Delta_tot is allocated as the same block of memory as scalefact.
          Not all this memory will actually have been used, but it is easiest to bcast all of it.*/
        MPI_Bcast(delta_tot_table.scalefact,delta_tot_table.namax*(delta_tot_table.nk+1),MPI_DOUBLE,0,MPI_COMM_WORLD);
    }
    }
}


/*Allocate memory for delta_tot_table. This is separate from delta_tot_init because we need to allocate memory
 * before we have the information needed to initialise it*/
void init_neutrinos_lra(const int nk_in, const double TimeTransfer, const double TimeMax, const double Omega0, const _omega_nu * const omnu, const double UnitTime_in_s, const double UnitLength_in_cm)
{
   _delta_tot_table *d_tot = &delta_tot_table;
   int count;
   /*Memory allocations need to be done on all processors*/
   d_tot->nk_allocated=nk_in;
   d_tot->nk=nk_in;
   /*Store starting time*/
   d_tot->TimeTransfer = TimeTransfer;
   /* Allocate memory for delta_tot here, so that we can have further memory allocated and freed
    * before delta_tot_init is called. The number nk here should be larger than the actual value needed.*/
   /*Allocate pointers to each k-vector*/
   d_tot->namax=ceil((TimeMax-TimeTransfer)/0.008)+4;
   d_tot->ia=0;
   d_tot->delta_tot =(double **) mymalloc("kspace_delta_tot",nk_in*sizeof(double *));
   /*Allocate list of scale factors, and space for delta_tot, in one operation.*/
   d_tot->scalefact = (double *) mymalloc("kspace_scalefact",d_tot->namax*(nk_in+1)*sizeof(double));
   /*Allocate space for delta_nu and wavenumbers*/
   d_tot->delta_nu_last = (double *) mymalloc("kspace_delta_nu", sizeof(double) * 3 * nk_in);
   d_tot->delta_nu_init = d_tot->delta_nu_last + nk_in;
   d_tot->wavenum = d_tot->delta_nu_init + nk_in;
   /*Allocate actual data. Note that this means data can be accessed either as:
    * delta_tot[k][a] OR as
    * delta_tot[0][a+k*namax] */
   d_tot->delta_tot[0] = d_tot->scalefact+d_tot->namax;
   for(count=1; count< nk_in; count++)
        d_tot->delta_tot[count] = d_tot->delta_tot[0] + count*d_tot->namax;
   /*Allocate space for the initial neutrino power spectrum*/
   /*Setup pointer to the matter density*/
   d_tot->omnu = omnu;
   /*Set the prefactor for delta_nu, and the units system*/
   d_tot->light = LIGHTCGS * UnitTime_in_s/UnitLength_in_cm;
   d_tot->delta_nu_prefac = 1.5 *Omega0 * HUBBLE * HUBBLE * pow(UnitTime_in_s,2)/d_tot->light;
   /*Matter fraction excluding neutrinos*/
   d_tot->Omeganonu = Omega0 - get_omega_nu(omnu, 1);
}

/*Begin functions that do the actual computation of the neutrino power spectra.
 * The algorithms executed are explained in Ali-Haimoud & Bird 2012 and Bird, Ali-Haimoud, Feng & Liu 2018
 * arXiv:1209.0461 and arXiv:1803.09854.
 * This is a Fourier-space linear response method for computing neutrino overdensities from CDM overdensities.*/

/*Function which wraps three get_delta_nu calls to get delta_nu three times,
 * so that the final value is for all neutrino species*/
void get_delta_nu_combined(Cosmology * CP, const _delta_tot_table * const d_tot, const double a, double delta_nu_curr[])
{
    const double Omega_nu_tot=get_omega_nu_nopart(d_tot->omnu, a);
    int mi;
    /*Initialise delta_nu_curr*/
    memset(delta_nu_curr, 0, d_tot->nk*sizeof(double));
    /*Get each neutrinos species and density separately and add them to the total.
     * Neglect perturbations in massless neutrinos.*/
    for(mi=0; mi<NUSPECIES; mi++) {
            if(d_tot->omnu->nu_degeneracies[mi] > 0) {
                 int ik;
                 double * delta_nu_single = (double *) mymalloc("delta_nu_single", sizeof(double) * d_tot->nk);
                 const double omeganu = d_tot->omnu->nu_degeneracies[mi] * omega_nu_single(d_tot->omnu, a, mi);
                 get_delta_nu(CP, d_tot, a, delta_nu_single,d_tot->omnu->RhoNuTab[mi].mnu);
                 for(ik=0; ik<d_tot->nk; ik++)
                    delta_nu_curr[ik]+=delta_nu_single[ik]*omeganu/Omega_nu_tot;
                 myfree(delta_nu_single);
            }
    }
    return;
}

/*Update the last value of delta_tot in the table with a new value computed
 from the given delta_cdm_curr and delta_nu_curr.
 If overwrite is true, overwrite the existing final entry.*/
void update_delta_tot(_delta_tot_table * const d_tot, const double a, const double delta_cdm_curr[], const double delta_nu_curr[], const int overwrite)
{
  const double OmegaNua3 = get_omega_nu_nopart(d_tot->omnu, a)*pow(a,3);
  const double OmegaNu1 = get_omega_nu(d_tot->omnu, 1);
  const double partnu = particle_nu_fraction(&d_tot->omnu->hybnu, a, 0);
  int ik;
  if(!overwrite)
    d_tot->ia++;
  /*Update the scale factor*/
  d_tot->scalefact[d_tot->ia-1] = log(a);
  /* Update delta_tot(a)*/
  for (ik = 0; ik < d_tot->nk; ik++){
    d_tot->delta_tot[ik][d_tot->ia-1] = get_delta_tot(delta_nu_curr[ik], delta_cdm_curr[ik], OmegaNua3, d_tot->Omeganonu, OmegaNu1,partnu);
  }
}

/*Kernel function for the fslength integration*/
double fslength_int(const double loga, void *params)
{
    Cosmology * CP = (Cosmology *) params;
    /*This should be M_nu / k_B T_nu (which is dimensionless)*/
    const double a = exp(loga);
    return 1./a/(a*hubble_function(CP, a));
}

/******************************************************************************************************
Free-streaming length (times Mnu/k_BT_nu, which is dimensionless) for a non-relativistic
particle of momentum q = T0, from scale factor ai to af.
Arguments:
logai - log of initial scale factor
logaf - log of final scale factor
light - speed of light in internal units.
Result is in Unit_Length/Unit_Time.
******************************************************************************************************/
double fslength(Cosmology * CP, const double logai, const double logaf, const double light)
{
    double abserr;
    if (logai >= logaf)
        return 0;

    // Define the integrand as a lambda function wrapping fslength_int
    auto integrand = [CP](double loga) {
        return fslength_int(loga, (void *)CP);
    };

    // Use Tanh-Sinh adaptive integration
    double fslength_val = tanh_sinh_integrate_adaptive(integrand, logai, logaf, &abserr, 1e-6);

    return light * fslength_val;
}

/**************************************************************************************************
Fit to the special function J(x) that is accurate to better than 3% relative and 0.07% absolute
    J(x) = Integrate[(Sin[q*x]/(q*x))*(q^2/(Exp[q] + 1)), {q, 0, Infinity}]
    and J(0) = 1.
    Mathematica gives this in terms of the PolyGamma function:
   (PolyGamma[1, 1/2 - i x/2] - PolyGamma[1, 1 - i x/2] -    PolyGamma[1, 1/2 + i x/2] +
   PolyGamma[1, 1 + i x/2])/(12 x Zeta[3]), which we could evaluate exactly if we wanted to.
***************************************************************************************************/
static inline double specialJ_fit(const double x)
{

  double x2, x4, x8;
  if (x <= 0.)
      return 1.;
  x2 = x*x;
  x4 = x2*x2;
  x8 = x4*x4;

  return (1.+ 0.0168 * x2 + 0.0407* x4)/(1. + 2.1734 * x2 + 1.6787 * exp(4.1811*log(x)) +  0.1467 * x8);
}

/*Asymptotic series expansion from YAH. Not good when qc * x is small, but fine otherwise.*/
static inline double II(const double x, const double qc, const int n)
{
    return (n*n+n*n*n*qc+n*qc*x*x - x*x)* qc*gsl_sf_bessel_j0(qc*x) + (2*n+n*n*qc+qc*x*x)*cos(qc*x);
}

/* Fourier transform of truncated Fermi Dirac distribution, with support on q > qc only.
 * qc is a dimensionless momentum (normalized to TNU),
 * mnu is in eV. x has units of inverse dimensionless momentum
 * This is an approximation to integral f_0(q) q^2 j_0(qX) dq between qc and infinity.
 * It gives the fraction of the integral that is due to neutrinos above a certain threshold.
 * Arguments: vcmnu is vcrit*mnu/LIGHT */
static inline double Jfrac_high(const double x, const double qc, const double nufrac_low)
{
    double integ=0;
    int n;
    for(n=1; n<20; n++)
    {
        integ+= -1*pow((-1),n)*exp(-n*qc)/(n*n+x*x)/(n*n+x*x)*II(x,qc,n);
    }
    /* Normalise with integral_qc^infty(f_0(q)q^2 dq), same as I(X).
     * So that as qc-> infinity, this -> specialJ_fit(x)*/
    integ /= 1.5 * 1.202056903159594 * (1 - nufrac_low);
    return integ;
}

/*Function that picks whether to use the truncated integrator or not*/
double specialJ(const double x, const double qc, const double nufrac_low)
{
  if( qc > 0 ) {
   return Jfrac_high(x, qc, nufrac_low);
  }
  return specialJ_fit(x);
}

/**A structure for the parameters for the below integration kernel*/
struct _delta_nu_int_params
{
    /**Current wavenumber*/
    double k;
    /**Neutrino mass divided by k_B T_nu*/
    double mnubykT;
    gsl_interp_accel *acc;
    gsl_interp *spline;
    Cosmology * CP;
    /**Precomputed free-streaming lengths*/
    gsl_interp_accel *fs_acc;
    gsl_interp *fs_spline;
    double * fslengths;
    double * fsscales;
    /**Make sure this is at the same k as above*/
    double * delta_tot;
    double * scale;
    /** qc is a dimensionless momentum (normalized to TNU): v_c * mnu / (k_B * T_nu).
     * This is the critical momentum for hybrid neutrinos: it is unused if
     * hybrid neutrinos are not defined, but left here to save ifdefs.*/
    double qc;
    /*Fraction of neutrinos in particles for normalisation with hybrid neutrinos*/
    double nufrac_low;
};
typedef struct _delta_nu_int_params delta_nu_int_params;

/**GSL integration kernel for get_delta_nu*/
double get_delta_nu_int(double logai, void * params)
{
    delta_nu_int_params * p = (delta_nu_int_params *) params;
    double fsl_aia = gsl_interp_eval(p->fs_spline,p->fsscales,p->fslengths,logai,p->fs_acc);
    double delta_tot_at_a = gsl_interp_eval(p->spline,p->scale,p->delta_tot,logai,p->acc);
    double specJ = specialJ(p->k*fsl_aia/p->mnubykT, p->qc, p->nufrac_low);
    double ai = exp(logai);
    return fsl_aia/(ai*hubble_function(p->CP, ai)) * specJ * delta_tot_at_a;
}

/*
Main function: given tables of wavenumbers, total delta at Na earlier times (<= a),
and initial conditions for neutrinos, computes the current delta_nu.
Na is the number of currently stored time steps.
*/
void get_delta_nu(Cosmology * CP, const _delta_tot_table * const d_tot, const double a, double delta_nu_curr[],const double mnu)
{
  double fsl_A0a,deriv_prefac;
  int ik;
  /* Variable is unused unless we have hybrid neutrinos,
   * but we define it anyway to save ifdeffing later.*/
  double qc = 0;
  /*Number of stored power spectra. This includes the initial guess for the next step*/
  const int Na = d_tot->ia;
  const double mnubykT = mnu /d_tot->omnu->kBtnu;
  /*Tolerated integration error*/
  double relerr = 1e-6;
//       message(0,"Start get_delta_nu: a=%g Na =%d wavenum[0]=%g delta_tot[0]=%g m_nu=%g\n",a,Na,wavenum[0],d_tot->delta_tot[0][Na-1],mnu);

  fsl_A0a = fslength(CP, log(d_tot->TimeTransfer), log(a),d_tot->light);
  /*Precompute factor used to get delta_nu_init. This assumes that delta ~ a, so delta-dot is roughly 1.*/
  deriv_prefac = d_tot->TimeTransfer*(hubble_function(CP, d_tot->TimeTransfer)/d_tot->light)* d_tot->TimeTransfer;
  for (ik = 0; ik < d_tot->nk; ik++) {
      /* Initial condition piece, assuming linear evolution of delta with a up to startup redshift */
      /* This assumes that delta ~ a, so delta-dot is roughly 1. */
      /* Also ignores any difference in the transfer functions between species.
       * This will be good if all species have similar masses, or
       * if two species are massless.
       * Also, since at early times the clustering is tiny, it is very unlikely to matter.*/
      /*For zero mass neutrinos just use the initial conditions piece, modulating to zero inside the horizon*/
      const double specJ = specialJ(d_tot->wavenum[ik]*fsl_A0a/(mnubykT > 0 ? mnubykT : 1),qc, d_tot->omnu->hybnu.nufrac_low[0]);
      delta_nu_curr[ik] = specJ*d_tot->delta_nu_init[ik] *(1.+ deriv_prefac*fsl_A0a);
  }
  /* Check whether the particle neutrinos are active at this point.
   * If they are we want to truncate our integration.
   * Only do this is hybrid neutrinos are activated in the param file.*/
  const double partnu = particle_nu_fraction(&d_tot->omnu->hybnu, a, 0);
  if(partnu > 0) {
/*       message(0,"Particle neutrinos gravitating: a=%g partnu: %g qc is: %g\n",a, partnu,qc); */
      /*If the particles are everything, be done now*/
      if(1 - partnu < 1e-3)
          return;
      qc = d_tot->omnu->hybnu.vcrit * mnubykT;
      /*More generous integration error for particle neutrinos*/
      relerr /= (1.+1e-5-particle_nu_fraction(&d_tot->omnu->hybnu,a,0))*0.1;
  }
  /*If only one time given, we are still at the initial time*/
  /*If neutrino mass is zero, we are not accurate, just use the initial conditions piece*/
  if(Na > 1 && mnubykT > 0){
        delta_nu_int_params params;
        params.acc = gsl_interp_accel_alloc();
        /*Use cubic interpolation*/
        if(Na > 2) {
                params.spline=gsl_interp_alloc(gsl_interp_cspline,Na);
        }
        /*Unless we have only two points*/
        else {
                params.spline=gsl_interp_alloc(gsl_interp_linear,Na);
        }
        params.scale=d_tot->scalefact;
        params.mnubykT=mnubykT;
        params.qc = qc;
        params.nufrac_low = d_tot->omnu->hybnu.nufrac_low[0];
        /* Massively over-sample the free-streaming lengths.
         * Interpolation is least accurate where the free-streaming length -> 0,
         * which is exactly where it doesn't matter, but
         * we still want to be safe. */
        int Nfs = Na*16;
        params.fs_acc = gsl_interp_accel_alloc();
        params.fs_spline=gsl_interp_alloc(gsl_interp_cspline,Nfs);
        params.CP = CP;
        /*Pre-compute the free-streaming lengths, which are scale-independent*/
        double * fslengths = (double *) mymalloc("fslengths", Nfs* sizeof(double));
        double * fsscales = (double *) mymalloc("fsscales", Nfs* sizeof(double));
        for(ik=0; ik < Nfs; ik++) {
            fsscales[ik] = log(d_tot->TimeTransfer) + ik*(log(a) - log(d_tot->TimeTransfer))/(Nfs-1.);
            fslengths[ik] = fslength(CP, fsscales[ik], log(a),d_tot->light);
        }
        params.fslengths = fslengths;
        params.fsscales = fsscales;

        if (!params.spline || !params.acc || !params.fs_spline || !params.fs_acc || !fslengths || !fsscales) {
            endrun(2016, "Error initializing and allocating memory for interpolators.\n");
        }

        gsl_interp_init(params.fs_spline,params.fsscales,params.fslengths,Nfs);
        for (ik = 0; ik < d_tot->nk; ik++) {
            double abserr,d_nu_tmp;
            params.k=d_tot->wavenum[ik];
            params.delta_tot=d_tot->delta_tot[ik];
            gsl_interp_init(params.spline,params.scale,params.delta_tot,Na);

            // Define the integrand as a lambda function wrapping get_delta_nu_int
            auto integrand = [&params](double logai) {
                return get_delta_nu_int(logai, (void *)&params);
            };
            d_nu_tmp = tanh_sinh_integrate_adaptive(integrand, log(d_tot->TimeTransfer), log(a), &abserr, relerr);
            delta_nu_curr[ik] += d_tot->delta_nu_prefac * d_nu_tmp;
         }
         gsl_interp_free(params.spline);
         gsl_interp_accel_free(params.acc);
         myfree(fsscales);
         myfree(fslengths);
   }
//     for(ik=0; ik< 3; ik++)
//         message(0,"k %g d_nu %g\n",wavenum[d_tot->nk/8*ik], delta_nu_curr[d_tot->nk/8*ik]);
   return;
}
