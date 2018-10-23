#ifndef WINDS_H
#define WINDS_H

#include "treewalk.h"

/*Evolve a wind particle, reducing its DelayTime*/
void wind_evolve(int i);

/*do the treewalk for the wind model*/
void winds_and_feedback(int * NewStars, int NumNewStars);

/*Make a particle a wind particle by changing DelayTime to a positive number*/
int make_particle_wind(MyIDType ID, int i, double v, double vmean[3]);

#endif
