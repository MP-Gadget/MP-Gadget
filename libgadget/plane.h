#ifndef PLANE_H
#define PLANE_H

#include "cosmology.h"

void set_plane_params(ParameterSet * ps);
void write_plane(int SnapPlaneCount, double atime, const Cosmology * CP, const char * OutputDir);

#endif