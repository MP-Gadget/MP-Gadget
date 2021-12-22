#ifndef _COOLING_QSO_LIGHTUP_H
#define _COOLING_QSO_LIGHTUP_H

#include "forcetree.h"
#include "fof.h"
#include "utils/paramset.h"

void set_qso_lightup_params(ParameterSet * ps);

/* Initialize the helium reionization cooling module*/
void init_qso_lightup(char * reion_hist_file);

/* Starts reionization by selecting the first halo and flagging all particles in the first HeIII bubble*/
void do_heiii_reionization(double atime, FOFGroups * fof, DomainDecomp * ddecomp, Cosmology * CP, double uu_in_cgs, FILE * FdHelium);

/* Get the long mean free path photon heating that applies to not-yet-ionized particles*/
double get_long_mean_free_path_heating(double redshift);

/* Returns 1 if helium reionization is in progress, 0 otherwise*/
int during_helium_reionization(double redshift);

/* Returns 1 if the helium model is enabled*/
int qso_lightup_on(void);

/* Returns 1 if there is any work to do to ionize more particles, 0 otherwise*/
int need_change_helium_ionization_fraction(double atime);
#endif
