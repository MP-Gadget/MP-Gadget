#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "allvars.h"
#include "proto.h"

double INLINE_FUNC hubble_function(double a)
{
  double hubble_a;

  hubble_a = All.Omega0 / (a * a * a) + OMEGAK / (a * a)
#ifdef INCLUDE_RADIATION
  /*Note OMEGAR is defined to be 0 if INCLUDE_RADIATION is not on*/
   + OMEGAR/(a*a*a*a)
#endif
    + All.OmegaLambda;
  hubble_a = All.Hubble * sqrt(hubble_a);
  return (hubble_a);
}
