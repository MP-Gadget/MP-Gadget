#ifndef NEUTRINOS_LRA_H
#define NEUTRINOS_LRA_H

#include <bigfile-mpi.h>
#include "powerspectrum.h"
#include "cosmology.h"

/** Now we want to define a static object to store all previous delta_tot.
 * This object needs a constructor, a few private data members, and a way to be read and written from disk.
 * nk is fixed, delta_tot, scalefact and ia are updated in get_delta_nu_update*/
struct _delta_tot_table {
    /** Number of actually non-zero k values stored in each power spectrum*/
    int nk;
    /** Size of arrays allocated to store power spectra*/
    int nk_allocated;
    /** Maximum number of redshifts to store. Redshifts are stored every delta a = 0.01 */
    int namax;
    /** Number of already "recorded" time steps, i.e. scalefact[0...ia-1] is recorded.
    * Current time corresponds to index ia (but is only recorded if sufficiently far from previous time).
    * Caution: ia here is different from Na in get_delta_nu (Na = ia+1).*/
    int ia;
    /** Prefactor for use in get_delta_nu. Should be 3/2 Omega_m H^2 /c */
    double delta_nu_prefac;
    /** Set to unity once the init routine has run.*/
    int delta_tot_init_done;
    /** Pointer to nk arrays of length namax containing the total power spectrum.*/
    double **delta_tot;
    /** Array of length namax containing scale factors at which the power spectrum is stored*/
    double * scalefact;
    /** Pointer to array of length nk storing initial neutrino power spectrum*/
    double * delta_nu_init;
    /** Pointer to array of length nk storing the last neutrino power spectrum we saw, for a first estimate
    * of the new delta_tot */
    double * delta_nu_last;
    /**Pointer to array storing the effective wavenumbers for the above power spectra*/
    double * wavenum;
    /** Pointer to a structure for computing omega_nu*/
    const _omega_nu * omnu;
    /** Matter density excluding neutrinos*/
    double Omeganonu;
    /** Light speed in internal units. C is defined in allvars.h to be lightspeed in cm/s*/
    double light;
    /** The time at which we first start our integrator:
     * NOTE! This is not All.TimeBegin, but the time of the transfer function file,
     * so that we can support restarting from snapshots.*/
    double TimeTransfer;
};
typedef struct _delta_tot_table _delta_tot_table;


/** Allocates memory for delta_tot_table.
 * @param nk_in Number of bins stored in each power spectrum.
 * @param TimeTransfer Scale factor of the transfer functions.
 * @param TimeMax Final scale factor up to which we will need memory.
 * @param Omega0 Matter density at z=0.
 * @param omnu Pointer to structure containing pre-computed tables for evaluating neutrino matter densities.
 * @param UnitTime_in_s Time unit of the simulation in s.
 * @param UnitLength_in_cm Length unit of the simulation in cm*/
void init_neutrinos_lra(const int nk_in, const double TimeTransfer, const double TimeMax, const double Omega0, const _omega_nu * const omnu, const double UnitTime_in_s, const double UnitLength_in_cm);

/*Computes delta_nu from a CDM power spectrum.*/
void delta_nu_from_power(struct _powerspectrum * PowerSpectrum, Cosmology * CP, int Time, int TimeIC);

/*These functions save and load neutrino related data from the snapshots*/
void petaio_save_neutrinos(BigFile * bf, int ThisTask);
void petaio_read_neutrinos(BigFile * bf, int ThisTask);
/*Loads from the ICs*/
void petaio_read_icnutransfer(BigFile * bf, int ThisTask);

#endif
