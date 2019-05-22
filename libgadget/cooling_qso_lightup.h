#ifndef _COOLING_QSO_LIGHTUP_H
#define _COOLING_QSO_LIGHTUP_H

/* Initialize the helium reionization cooling module*/
void init_qso_lightup(char * reion_hist_file);

/* Starts reionization by selecting the first halo and flagging all particles in the first HeIII bubble*/
void start_reionization(double redshift);

#endif
