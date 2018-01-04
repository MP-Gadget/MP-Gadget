#ifndef HYDRA_H
#define HYDRA_H

/*Function to get the pressure from the entropy and the density*/
double GetPressure(int i);

/*Function to compute hydro accelerations and adiabatic entropy change*/
void hydro_force(void);

#endif
