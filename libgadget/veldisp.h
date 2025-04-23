#ifndef VELDISP_H
#define VELDISP_H

#include "timestep.h"
#include "cosmology.h"

/* Find the 1D DM velocity dispersion of all nearly star-forming gas and black hole particles.
 * Gas is done by running a density loop for find Vdisp of nearest 40 DM particles.
 * BH is done by finding VDisp of all DM particles inside the SPH kernel.
 * Stores it in VDisp in the slots structure.*/
void winds_find_vel_disp(const ActiveParticles * act, const double Time, const double hubble, Cosmology * CP, DriftKickTimes * times, DomainDecomp * ddecomp);

#endif
