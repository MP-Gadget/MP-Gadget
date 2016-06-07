#ifndef COSMOLOGY_H
#define COSMOLOGY_H

/*Hubble function at scale factor a, in dimensions of All.Hubble*/
double hubble_function(double a);
/* Linear theory growth factor normalized to D(a=1.0) = 1.0. */
double GrowthFactor(double a);

#endif
