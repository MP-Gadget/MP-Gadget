#include "cosmology.h"
#include "allvars.h"
/*Hubble function at scale factor a, in dimensions of All.Hubble*/
double hubble_function(double a)
{
  double hubble_a;
  hubble_a = All.Omega0 / (a * a * a) + OMEGAK / (a * a) + All.RadiationOn*OMEGAR/(a*a*a*a) + All.OmegaLambda;
  hubble_a = All.Hubble * sqrt(hubble_a);
  return (hubble_a);
}
