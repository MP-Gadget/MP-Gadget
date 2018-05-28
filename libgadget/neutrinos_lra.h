#ifndef NEUTRINOS_LRA_H
#define NEUTRINOS_LRA_H

#include <bigfile-mpi.h>
#include "powerspectrum.h"
#include "cosmology.h"

/*Allocates the memory for the neutrino integrator*/
void init_neutrinos_lra(const int nk_in, const double TimeTransfer, const double TimeMax, const double Omega0, const _omega_nu * const omnu, const double UnitTime_in_s, const double UnitLength_in_cm);

/*Computes delta_nu from a CDM power spectrum.*/
void delta_nu_from_power(struct _powerspectrum * PowerSpectrum, Cosmology * CP, int Time, int TimeIC);

/*These functions save and load neutrino related data from the snapshots*/
void petaio_save_neutrinos(BigFile * bf, int ThisTask);
void petaio_read_neutrinos(BigFile * bf, int ThisTask);
/*Loads from the ICs*/
void petaio_read_icnutransfer(BigFile * bf, int ThisTask);

#endif
