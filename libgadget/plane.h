#ifndef PLANE_H
#define PLANE_H

#include "cosmology.h"

void set_plane_params(ParameterSet * ps);
void write_plane(int SnapPlaneCount, const double atime, const Cosmology * CP, const char * OutputDir, const double UnitVelocity_in_cm_per_s);

#endif