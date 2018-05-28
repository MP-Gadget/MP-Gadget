#ifndef NEUTRINOS_LRA_H
#define NEUTRINOS_LRA_H

#include "kspace-neutrinos/delta_tot_table.h"

/*Defined in neutrinos_lra.h*/
extern _delta_tot_table delta_tot_table;
extern _transfer_init_table t_init;

void delta_tot_resume(_delta_tot_table * const d_tot, const int nk_in, const double wavenum[]);
void delta_tot_first_init(_delta_tot_table * const d_tot, const int nk_in, const double wavenum[], const double delta_cdm_curr[], const _transfer_init_table * const t_init, const double TimeIC);

#endif
