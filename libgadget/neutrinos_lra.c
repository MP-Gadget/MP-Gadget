#include <math.h>
#include "neutrinos_lra.h"
#include "utils/endrun.h"

/*Structure which holds the neutrino state*/
_delta_tot_table delta_tot_table;
_transfer_init_table t_init;

/* Constructor for delta_tot. Does some sanity checks and initialises delta_nu_last.*/
void delta_tot_resume(_delta_tot_table * const d_tot, const int nk_in, const double wavenum[])
{
    int ik;
    if(nk_in > d_tot->nk_allocated){
           endrun(2011,"input power of %d is longer than memory of %d\n",nk_in,d_tot->nk_allocated);
    }
    if(d_tot->nk != nk_in)
        endrun(201, "Number of neutrino bins %d != stored value %d\n",nk_in, d_tot->nk);

    /*Set the wave number*/
    for(ik=0;ik<d_tot->nk;ik++){
        d_tot->wavenum[ik] = wavenum[ik];
    }
    /*Initialise delta_nu_last*/
    get_delta_nu_combined(d_tot, exp(d_tot->scalefact[d_tot->ia-1]), wavenum, d_tot->delta_nu_last);
    d_tot->delta_tot_init_done=1;
    return;
}

/* Constructor. transfer_init_tabulate must be called before this function.
 * Initialises delta_tot (including from a file) and delta_nu_init from the transfer functions.
 * read_all_nu_state must be called before this if you want reloading from a snapshot to work
 * Note delta_cdm_curr includes baryons, and is only used if not resuming.*/
void delta_tot_first_init(_delta_tot_table * const d_tot, const int nk_in, const double wavenum[], const double delta_cdm_curr[], const _transfer_init_table * const t_init, const double TimeIC)
{
    int ik;
    if(nk_in > d_tot->nk_allocated){
           endrun(2011,"input power of %d is longer than memory of %d\n",nk_in,d_tot->nk_allocated);
    }
    d_tot->nk=nk_in;
    const double OmegaNua3=get_omega_nu_nopart(d_tot->omnu, d_tot->TimeTransfer)*pow(d_tot->TimeTransfer,3);
    const double OmegaNu1 = get_omega_nu(d_tot->omnu, 1);
    gsl_interp_accel *acc = gsl_interp_accel_alloc();
    gsl_interp * spline=gsl_interp_alloc(gsl_interp_cspline,t_init->NPowerTable);
    gsl_interp_init(spline,t_init->logk,t_init->T_nu,t_init->NPowerTable);
    /*Check we have a long enough power table*/
    if(log(wavenum[d_tot->nk-1]) > t_init->logk[t_init->NPowerTable-1])
        endrun(2,"Want k = %g but maximum in CAMB table is %g\n",wavenum[d_tot->nk-1], exp(t_init->logk[t_init->NPowerTable-1]));
    for(ik=0;ik<d_tot->nk;ik++) {
            /* T_nu contains T_nu / T_cdm.*/
            double T_nubyT_nonu = gsl_interp_eval(spline,t_init->logk,t_init->T_nu,log(wavenum[ik]),acc);
            /*Initialise delta_nu_init to use the first timestep's delta_cdm_curr
             * so that it includes potential Rayleigh scattering. */
            d_tot->delta_nu_init[ik] = delta_cdm_curr[ik]*T_nubyT_nonu;
            const double partnu = particle_nu_fraction(&d_tot->omnu->hybnu, TimeIC, 0);
            /*Initialise the first delta_tot*/
            d_tot->delta_tot[ik][0] = get_delta_tot(d_tot->delta_nu_init[ik], delta_cdm_curr[ik], OmegaNua3, d_tot->Omeganonu, OmegaNu1, partnu);
            d_tot->wavenum[ik] = wavenum[ik];
    }
    gsl_interp_accel_free(acc);
    gsl_interp_free(spline);

    /*If we are not restarting, make sure we set the scale factor*/
    d_tot->scalefact[0]=log(TimeIC);
    d_tot->ia=1;
    /*Initialise delta_nu_last*/
    get_delta_nu_combined(d_tot, exp(d_tot->scalefact[0]), wavenum, d_tot->delta_nu_last);
    d_tot->delta_tot_init_done=1;
    return;
}

