#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "allvars.h"
#include "proto.h"

#ifdef VORONOI
#include "voronoi.h"

void voronoi_density(void)
{
  int i;
  int dt_step;
  double dt_entr;

  for(i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i])
    {
      if(P[i].Type == 0)
	{
	  SphP[i].d.Density = P[i].Mass / SphP[i].Volume;
	  SphP[i].v.DivVel = 0;

	  dt_step = (P[i].TimeBin ? (1 << P[i].TimeBin) : 0);
	  dt_entr = (All.Ti_Current - (P[i].Ti_begstep + dt_step / 2)) * All.Timebase_interval;

#if defined(VORONOI_MESHRELAX)

#ifdef VORONOI_MESHRELAX_KEEPRESSURE
	  SphP[i].Entropy = SphP[i].Pressure / (GAMMA_MINUS1 * SphP[i].d.Density);
#else
	  SphP[i].Pressure = GAMMA_MINUS1 * SphP[i].Entropy * SphP[i].d.Density;
#endif

#else
	  SphP[i].Pressure =
	    (SphP[i].Entropy + SphP[i].e.DtEntropy * dt_entr) * pow(SphP[i].d.Density, GAMMA);
#endif

	  SphP[i].MaxSignalVel = sqrt(GAMMA * SphP[i].Pressure / SphP[i].d.Density);
	}
    }
}

#endif
