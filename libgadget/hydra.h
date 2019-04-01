#ifndef HYDRA_H
#define HYDRA_H

#include "forcetree.h"

/*Function to get the pressure from the entropy and the density*/
double PressurePred(int i);

/*Function to compute hydro accelerations and adiabatic entropy change*/
void hydro_force(ForceTree * tree);

#endif
