#if CHEMISTRYNETWORK == 2 || CHEMISTRYNETWORK == 3 || CHEMISTRYNETWORK == 7 || CHEMISTRYNETWORK == 11
#define NEEDS_CO_RATES
#endif

#include <stdio.h>
#include <math.h>
#include <mpi.h>
#include "allvars.h"
#include "proto.h"

#if defined RAYTRACE && defined CHEMCOOL

static double Hcol[NCOL][NCOL];
static double H2col[NCOL][NCOL];
#ifdef CO_SHIELDING
static double COcol[NCOL][NCOL];
#endif

void raytrace(void)
{
  double x, y, z, hydrogen_mass, vol, num_H, num_H2, MH2, MH;
  double num_CO, dx, dy, dz, dx2, dy2, dz2;
  double pos[3];
#ifdef PERIODIC
  double boxsize[3];
#endif
  int i, j, k, ix, iy, iz;

  raytrace_init_regionsize();

#ifdef PERIODIC
  for(j = 0; j < 3; j++)
    boxsize[j] = All.BoxSize;

#ifdef LONG_X
  boxsize[0] *= LONG_X;
#endif
#ifdef LONG_Y
  boxsize[1] *= LONG_Y;
#endif
#ifdef LONG_Z
  boxsize[2] *= LONG_Z;
#endif

#endif /* PERIODIC */

  dx = All.RTdX[0];
  dy = All.RTdX[1];
  dz = All.RTdX[2];

  dx2 = dx / 2.;
  dy2 = dy / 2.;
  dz2 = dz / 2.;
  vol = dx * dy * dz;

  for(ix = 0; ix < NCOL; ix++)
    {
      for(iy = 0; iy < NCOL; iy++)
	{
	  for(iz = 0; iz < NCOL; iz++)
	    {
	      density_H[ix][iy][iz] = 0.0;
	      density_H2[ix][iy][iz] = 0.0;
#ifdef CO_SHIELDING
	      density_CO[ix][iy][iz] = 0.0;
#endif
	    }
	}
    }

  /* Crude interpolation onto grid */
  for(i = 0; i < N_gas; i++)
    {
      for(j = 0; j < 3; j++)
	{
	  pos[j] = P[i].Pos[j];
	}
#ifdef PERIODIC
      for(j = 0; j < 3; j++)
	{
	  while(pos[j] < 0)
	    pos[j] += boxsize[j];
	  while(pos[j] >= boxsize[j])
	    pos[j] -= boxsize[j];
	}
#endif
      x = pos[0];
      y = pos[1];
      z = pos[2];

      hydrogen_mass = P[i].Mass / (1.0 + 4.0 * ABHE);
      MH2 = hydrogen_mass * (2.0 * SphP[i].TracAbund[IH2]) * All.UnitMass_in_g;
      MH = hydrogen_mass * (1.0 - 2.0 * SphP[i].TracAbund[IH2]) * All.UnitMass_in_g;

      num_H2 = MH2 / (2.0 * PROTONMASS);
      num_H = MH / PROTONMASS;
#ifdef CO_SHIELDING
      /* Use the fact that our abundances are defined relative to the total number density
       * of hydrogen nuclei
       */
      num_CO = SphP[i].TracAbund[ICO] * (num_H + 2.0 * num_H2);
#endif

      /* Find grid cell for this particle */
      ix = floor((x - All.RTXmin[0]) / dx);
      if(ix == NCOL)
	ix--;

      iy = floor((y - All.RTXmin[1]) / dy);
      if(iy == NCOL)
	iy--;

      iz = floor((z - All.RTXmin[2]) / dz);
      if(iz == NCOL)
	iz--;

      density_H[ix][iy][iz] += (num_H / vol);
      density_H2[ix][iy][iz] += (num_H2 / vol);
#ifdef CO_SHIELDING
      density_CO[ix][iy][iz] += (num_CO / vol);
#endif
    }

  /* Slow but simple method: use MPI_Allreduce to sum the grids */
  MPI_Allreduce(density_H, RTbuffer, NCOL * NCOL * NCOL, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  for(i = 0; i < NCOL; i++)
    {
      for(j = 0; j < NCOL; j++)
	{
	  for(k = 0; k < NCOL; k++)
	    {
	      density_H[i][j][k] = RTbuffer[i][j][k];
	    }
	}
    }

  MPI_Allreduce(density_H2, RTbuffer, NCOL * NCOL * NCOL, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  for(i = 0; i < NCOL; i++)
    {
      for(j = 0; j < NCOL; j++)
	{
	  for(k = 0; k < NCOL; k++)
	    {
	      density_H2[i][j][k] = RTbuffer[i][j][k];
	    }
	}
    }

#if defined CO_SHIELDING && defined NEEDS_CO_RATES
  MPI_Allreduce(density_CO, RTbuffer, NCOL * NCOL * NCOL, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  for(i = 0; i < NCOL; i++)
    {
      for(j = 0; j < NCOL; j++)
	{
	  for(k = 0; k < NCOL; k++)
	    {
	      density_CO[i][j][k] = RTbuffer[i][j][k];
	    }
	}
    }
#endif

  /* Compute column densities */

  /* X direction */
  for(iy = 0; iy < NCOL; iy++)
    {
      for(iz = 0; iz < NCOL; iz++)
	{
	  column_H[0][0][iy][iz] = dx2 * density_H[0][iy][iz];
	  column_H2[0][0][iy][iz] = dx2 * density_H2[0][iy][iz];
#if defined CO_SHIELDING && defined NEEDS_CO_RATES
	  column_CO[0][0][iy][iz] = dx2 * density_CO[0][iy][iz];
#endif
	  for(ix = 1; ix < NCOL; ix++)
	    {
	      column_H[0][ix][iy][iz] = column_H[0][ix - 1][iy][iz]
		+ dx2 * (density_H[ix - 1][iy][iz] + density_H[ix][iy][iz]);
	      column_H2[0][ix][iy][iz] = column_H2[0][ix - 1][iy][iz]
		+ dx2 * (density_H2[ix - 1][iy][iz] + density_H2[ix][iy][iz]);
#if defined CO_SHIELDING && defined NEEDS_CO_RATES
	      column_CO[0][ix][iy][iz] = column_CO[0][ix - 1][iy][iz]
		+ dx2 * (density_CO[ix - 1][iy][iz] + density_CO[ix][iy][iz]);
#endif
	    }
	  Hcol[iy][iz] = column_H[0][NCOL - 1][iy][iz] + dx2 * density_H[NCOL - 1][iy][iz];
	  H2col[iy][iz] = column_H2[0][NCOL - 1][iy][iz] + dx2 * density_H2[NCOL - 1][iy][iz];
#if defined CO_SHIELDING && defined NEEDS_CO_RATES
	  COcol[iy][iz] = column_CO[0][NCOL - 1][iy][iz] + dx2 * density_CO[NCOL - 1][iy][iz];
#endif
	  for(ix = 0; ix < NCOL; ix++)
	    {
	      column_H[1][ix][iy][iz] = Hcol[iy][iz] - column_H[0][ix][iy][iz];
	      column_H2[1][ix][iy][iz] = H2col[iy][iz] - column_H2[0][ix][iy][iz];
#if defined CO_SHIELDING && defined NEEDS_CO_RATES
	      column_CO[1][ix][iy][iz] = COcol[iy][iz] - column_CO[0][ix][iy][iz];
#endif
	    }
	}
    }

  /* Y direction */
  for(ix = 0; ix < NCOL; ix++)
    {
      for(iz = 0; iz < NCOL; iz++)
	{
	  column_H[2][ix][0][iz] = dy2 * density_H[ix][0][iz];
	  column_H2[2][ix][0][iz] = dy2 * density_H2[ix][0][iz];
#if defined CO_SHIELDING && defined NEEDS_CO_RATES
	  column_CO[2][ix][0][iz] = dy2 * density_CO[ix][0][iz];
#endif
	  for(iy = 1; iy < NCOL; iy++)
	    {
	      column_H[2][ix][iy][iz] = column_H[2][ix][iy - 1][iz]
		+ dy2 * (density_H[ix][iy - 1][iz] + density_H[ix][iy][iz]);
	      column_H2[2][ix][iy][iz] = column_H2[2][ix][iy - 1][iz]
		+ dy2 * (density_H2[ix][iy - 1][iz] + density_H2[ix][iy][iz]);
#if defined CO_SHIELDING && defined NEEDS_CO_RATES
	      column_CO[2][ix][iy][iz] = column_CO[2][ix][iy - 1][iz]
		+ dy2 * (density_CO[ix][iy - 1][iz] + density_CO[ix][iy][iz]);
#endif
	    }
	  Hcol[ix][iz] = column_H[2][ix][NCOL - 1][iz] + dy2 * density_H[ix][NCOL - 1][iz];
	  H2col[ix][iz] = column_H2[2][ix][NCOL - 1][iz] + dy2 * density_H2[ix][NCOL - 1][iz];
#if defined CO_SHIELDING && defined NEEDS_CO_RATES
	  COcol[ix][iz] = column_CO[2][ix][NCOL - 1][iz] + dy2 * density_CO[ix][NCOL - 1][iz];
#endif
	  for(iy = 0; iy < NCOL; iy++)
	    {
	      column_H[3][ix][iy][iz] = Hcol[ix][iz] - column_H[2][ix][iy][iz];
	      column_H2[3][ix][iy][iz] = H2col[ix][iz] - column_H2[2][ix][iy][iz];
#if defined CO_SHIELDING && defined NEEDS_CO_RATES
	      column_CO[3][ix][iy][iz] = COcol[ix][iz] - column_CO[2][ix][iy][iz];
#endif
	    }
	}
    }

  /* Z direction */
  for(ix = 0; ix < NCOL; ix++)
    {
      for(iy = 0; iy < NCOL; iy++)
	{
	  column_H[4][ix][iy][0] = dz2 * density_H[ix][iy][0];
	  column_H2[4][ix][iy][0] = dz2 * density_H2[ix][iy][0];
#if defined CO_SHIELDING && defined NEEDS_CO_RATES
	  column_CO[4][ix][iy][0] = dz2 * density_CO[ix][iy][0];
#endif
	  for(iz = 1; iz < NCOL; iz++)
	    {
	      column_H[4][ix][iy][iz] = column_H[4][ix][iy][iz - 1]
		+ dz2 * (density_H[ix][iy][iz - 1] + density_H[ix][iy][iz]);
	      column_H2[4][ix][iy][iz] = column_H2[4][ix][iy][iz - 1]
		+ dz2 * (density_H2[ix][iy][iz - 1] + density_H2[ix][iy][iz]);
#if defined CO_SHIELDING && defined NEEDS_CO_RATES
	      column_CO[4][ix][iy][iz] = column_CO[4][ix][iy][iz - 1]
		+ dz2 * (density_CO[ix][iy][iz - 1] + density_CO[ix][iy][iz]);
#endif
	    }
	  Hcol[ix][iy] = column_H[4][ix][iy][NCOL - 1] + dz2 * density_H[ix][iy][NCOL - 1];
	  H2col[ix][iy] = column_H2[4][ix][iy][NCOL - 1] + dz2 * density_H2[ix][iy][NCOL - 1];
#if defined CO_SHIELDING && defined NEEDS_CO_RATES
	  COcol[ix][iy] = column_CO[4][ix][iy][NCOL - 1] + dz2 * density_CO[ix][iy][NCOL - 1];
#endif
	  for(iz = 0; iz < NCOL; iz++)
	    {
	      column_H[5][ix][iy][iz] = Hcol[ix][iy] - column_H[4][ix][iy][iz];
	      column_H2[5][ix][iy][iz] = H2col[ix][iy] - column_H2[4][ix][iy][iz];
#if defined CO_SHIELDING && defined NEEDS_CO_RATES
	      column_CO[5][ix][iy][iz] = COcol[ix][iy] - column_CO[4][ix][iy][iz];
#endif
	    }
	}
    }

  /* Crude interpolation onto particles */
  for(i = 0; i < N_gas; i++)
    {
      for(j = 0; j < 3; j++)
	{
	  pos[j] = P[i].Pos[j];
	}
#ifdef PERIODIC
      for(j = 0; j < 3; j++)
	{
	  while(pos[j] < 0)
	    pos[j] += boxsize[j];
	  while(pos[j] >= boxsize[j])
	    pos[j] -= boxsize[j];
	}
#endif
      x = pos[0];
      y = pos[1];
      z = pos[2];

      /* Find grid cell for this particle */
      ix = floor((x - All.RTXmin[0]) / dx);
      if(ix == NCOL)
	ix--;

      iy = floor((y - All.RTXmin[1]) / dy);
      if(iy == NCOL)
	iy--;

      iz = floor((z - All.RTXmin[2]) / dz);
      if(iz == NCOL)
	iz--;

      for(j = 0; j < 6; j++)
	{
	  SphP[i].TotalColumnDensity[j] = column_H[j][ix][iy][iz];
	}

      for(j = 0; j < 6; j++)
	{
	  SphP[i].H2ColumnDensity[j] = column_H2[j][ix][iy][iz];
	  if(SphP[i].H2ColumnDensity[j] < 0.)
	    {
	      printf("%d %d %d %d\n", j, ix, iy, iz);
	      printf("%g %g %g\n", x, y, z);
	    }
	}
#if defined CO_SHIELDING && defined NEEDS_CO_RATES
      for(j = 0; j < 6; j++)
	{
	  SphP[i].COColumnDensity[j] = column_CO[j][ix][iy][iz];
	}
#endif
    }

  return;
}

/* Adapted from  pm_init_regionsize in pm_nonperiodic.c */
void raytrace_init_regionsize(void)
{
  double xmin[3], xmax[3], pos[3];
  double RTXmax[3];
  double RTMeshSize[3];
#ifdef PERIODIC
  double boxsize[3];
#endif
  int i, j;

  /* find enclosing rectangle */

  for(j = 0; j < 3; j++)
    {
      xmin[j] = 1.0e36;
      xmax[j] = -1.0e36;
    }

#ifdef PERIODIC
  for(j = 0; j < 3; j++)
    boxsize[j] = All.BoxSize;

#ifdef LONG_X
  boxsize[0] *= LONG_X;
#endif
#ifdef LONG_Y
  boxsize[1] *= LONG_Y;
#endif
#ifdef LONG_Z
  boxsize[2] *= LONG_Z;
#endif

#endif /* PERIODIC */

  for(i = 0; i < N_gas; i++)
    for(j = 0; j < 3; j++)
      {
	pos[j] = P[i].Pos[j];
#ifdef PERIODIC
	while(pos[j] < 0)
	  pos[j] += boxsize[j];
	while(pos[j] >= boxsize[j])
	  pos[j] -= boxsize[j];
#endif
	if(pos[j] > xmax[j])
	  xmax[j] = pos[j];
	if(pos[j] < xmin[j])
	  xmin[j] = pos[j];
      }

  MPI_Allreduce(xmin, All.RTXmin, 3, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(xmax, RTXmax, 3, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

  for(j = 0; j < 3; j++)
    {
      RTMeshSize[j] = RTXmax[j] - All.RTXmin[j];
      All.RTdX[j] = RTMeshSize[j] / NCOL;
    }
  return;
}

#endif /* RAYTRACE && CHEMCOOL */
#undef NEEDS_CO_RATES
