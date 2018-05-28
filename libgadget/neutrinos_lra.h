#ifndef NEUTRINOS_LRA_H
#define NEUTRINOS_LRA_H

#include <bigfile-mpi.h>
#include "kspace-neutrinos/delta_tot_table.h"

/*Defined in neutrinos_lra.h*/
extern _delta_tot_table delta_tot_table;

/*Initialises a delta_tot_table*/
void delta_tot_resume(_delta_tot_table * const d_tot, const int nk_in, const double wavenum[]);
void delta_tot_first_init(_delta_tot_table * const d_tot, const int nk_in, const double wavenum[], const double delta_cdm_curr[], const double TimeIC);

/*These functions save and load neutrino related data from the snapshots*/
void petaio_save_neutrinos(BigFile * bf, int ThisTask);
void petaio_read_neutrinos(BigFile * bf, int ThisTask);
/*Loads from the ICs*/
void petaio_read_icnutransfer(BigFile * bf, int ThisTask);

#endif
