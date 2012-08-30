/* -----------------------------------------------------------------------------
 *
 *  Copyright (C) 1997-2005 Krzysztof M. Gorski, Eric Hivon, 
 *                          Benjamin D. Wandelt, Anthony J. Banday, 
 *                          Matthias Bartelmann, 
 *                          Reza Ansari & Kenneth M. Ganga 
 *
 *
 *  This file is part of HEALPix.
 *
 *  HEALPix is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  HEALPix is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with HEALPix; if not, write to the Free Software
 *  Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
 *
 *  For more information about HEALPix see http://healpix.jpl.nasa.gov
 *
 *----------------------------------------------------------------------------- */
/* vec2pix_nest.c */

/* Standard Includes */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <signal.h>
#include <unistd.h>

#if defined(HEALPIX)
#include "allvars.h"
#include "proto.h"

void mk_xy2pix(int *x2pix, int *y2pix)
{
  /* =======================================================================
   * subroutine mk_xy2pix
   * =======================================================================
   * sets the array giving the number of the pixel lying in (x,y)
   * x and y are in {1,128}
   * the pixel number is in {0,128**2-1}
   *
   * if  i-1 = sum_p=0  b_p * 2^p
   * then ix = sum_p=0  b_p * 4^p
   * iy = 2*ix
   * ix + iy in {0, 128**2 -1}
   * =======================================================================
   */
  int i, K, IP, I, J, ID;

  for(i = 0; i < 127; i++)
    x2pix[i] = 0;
  for(I = 1; I <= 128; I++)
    {
      J = I - 1;		//            !pixel numbers
      K = 0;			//
      IP = 1;			//
    truc:if(J == 0)
	{
	  x2pix[I - 1] = K;
	  y2pix[I - 1] = 2 * K;
	}
      else
	{
	  ID = (int) fmod(J, 2);
	  J = J / 2;
	  K = IP * ID + K;
	  IP = IP * 4;
	  goto truc;
	}
    }

}

void vec2pix_nest(const long nside, double *vec, long *ipix)
{

  double z, za, z0, tt, tp, tmp, phi;
  int face_num, jp, jm;
  long ifp, ifm;
  int ix, iy, ix_low, ix_hi, iy_low, iy_hi, ipf, ntt;
  double piover2 = 0.5 * M_PI, twopi = 2.0 * M_PI;
  int ns_max = 8192;
  static int x2pix[128], y2pix[128];
  static char setup_done = 0;

  if(nside < 1 || nside > ns_max)
    {
      fprintf(stderr, "AUCH (%d): nside out of range: %ld\n", ns_max, nside);
      printf("Outch \n \n");
      fflush(stdout);

      endrun(73630);
    }
  if(!setup_done)
    {
      mk_xy2pix(x2pix, y2pix);
      //  printf("Outchr1\n");fflush(stdout);
      setup_done = 1;
    }

  z = vec[2] / sqrt(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2]);
  phi = 0.0;
  if(vec[0] != 0.0 || vec[1] != 0.0)
    {
      phi = atan2(vec[1], vec[0]);	/* in ]-pi, pi] */
      if(phi < 0.0)
	phi += twopi;		/* in  [0, 2pi[ */
    }

  za = fabs(z);
  z0 = 2. / 3.;
  tt = phi / piover2;		/* in [0,4[ */

  if(za <= z0)
    {				/* equatorial region */

      /* (the index of edge lines increase when the longitude=phi goes up) */
      jp = (int) floor(ns_max * (0.5 + tt - z * 0.75));	/* ascending edge line index */
      jm = (int) floor(ns_max * (0.5 + tt + z * 0.75));	/* descending edge line index */

      /* finds the face */
      ifp = jp / ns_max;	/* in {0,4} */
      ifm = jm / ns_max;

      if(ifp == ifm)
	face_num = (int) fmod(ifp, 4) + 4;	/* faces 4 to 7 */
      else if(ifp < ifm)
	face_num = (int) fmod(ifp, 4);	/* (half-)faces 0 to 3 */
      else
	face_num = (int) fmod(ifm, 4) + 8;	/* (half-)faces 8 to 11 */

      ix = (int) fmod(jm, ns_max);
      iy = ns_max - (int) fmod(jp, ns_max) - 1;
    }
  else
    {				/* polar region, za > 2/3 */

      ntt = (int) floor(tt);
      if(ntt >= 4)
	ntt = 3;
      tp = tt - ntt;
      tmp = sqrt(3. * (1. - za));	/* in ]0,1] */

      /* (the index of edge lines increase when distance from the closest pole
       * goes up)
       */
      /* line going toward the pole as phi increases */
      jp = (int) floor(ns_max * tp * tmp);

      /* that one goes away of the closest pole */
      jm = (int) floor(ns_max * (1. - tp) * tmp);
      jp = (int) (jp < ns_max - 1 ? jp : ns_max - 1);
      jm = (int) (jm < ns_max - 1 ? jm : ns_max - 1);

      /* finds the face and pixel's (x,y) */
      if(z >= 0)
	{
	  face_num = ntt;	/* in {0,3} */
	  ix = ns_max - jm - 1;
	  iy = ns_max - jp - 1;
	}
      else
	{
	  face_num = ntt + 8;	/* in {8,11} */
	  ix = jp;
	  iy = jm;
	}
    }

  ix_low = (int) fmod(ix, 128);
  ix_hi = ix / 128;
  iy_low = (int) fmod(iy, 128);
  iy_hi = iy / 128;

  ipf = (x2pix[ix_hi] + y2pix[iy_hi]) * (128 * 128) + (x2pix[ix_low] + y2pix[iy_low]);
  ipf = (long) (ipf / pow(ns_max / nside, 2));	/* in {0, nside**2 - 1} */
  *ipix = (long) (ipf + face_num * pow(nside, 2));	/* in {0, 12*nside**2 - 1} */
}

//////////////////////////////////////////////////// END OF STANDARTD HEALPIX

void healpix_halo_cond(float *res)
{
  int i, k;
  double cm[3], r2, r2min, minmap, maxmap;
  long ipix, nsid;
  float *res_pre;

  nsid = (long) All.Nside;
  res_pre = (float *) mymalloc("Healpix_temp",NSIDE2NPIX(All.Nside) * sizeof(float));	//<- just to not use the same in/out buffer in MPI
  for(i = 0; i < NSIDE2NPIX(All.Nside); i++)
    res_pre[i] = res[i];		//<- just in case
// begin of the construction of the map
  r2min=1e10;
  for(i = 0; i < NumPart; i++)
    {
       if((1 << P[i].Type) & (HEALPIX_OUTERBOUND))
	{
	  for(k = 0; k < 3; k++)
	    cm[k] = (double) P[i].Pos[k] - SysState.CenterOfMassComp[0][k];	         // postition realite to the CM of the DM in the HighResRegion
	  r2 = sqrt(cm[0] * cm[0] + cm[1] * cm[1] + cm[2] * cm[2]);	                 // calculate the radious from the CM
          if (r2min > r2) r2min = r2;
	}
    }

  MPI_Allreduce(&r2min, &r2, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);

  for(i = 0; i < NSIDE2NPIX(All.Nside); i++)
    if(res_pre[i] == 0) res_pre[i]=r2;

  for(i = 0; i < NumPart; i++)
    {
       if((1 << P[i].Type) & (HEALPIX_INNERBOUND))
	{
	  for(k = 0; k < 3; k++)
	    cm[k] = (double) P[i].Pos[k] - SysState.CenterOfMassComp[0][k]; // postition realite to the CM of the DM in the HighResRegion
	  r2 = sqrt(cm[0] * cm[0] + cm[1] * cm[1] + cm[2] * cm[2]);	    // calculate the radious from the CM
	  vec2pix_nest(nsid, cm, &ipix);	                            // tell me what's the index on the map

	  /* Not sure if gas is a good proxy, so test later !
	  if(P[i].Type == 0)
	    res_pre[ipix] = res_pre[ipix] > (float) r2/HEALPIX*0.9 ? res_pre[ipix] : (float) r2;	
	  */
	  if(P[i].Type == 1)
	    {
	      if(P[i].Mass == All.Minmass)
		res_pre[ipix] = res_pre[ipix] > (float) r2 ? res_pre[ipix] : (float) r2;	// now are we the most away particle with low mass
	    }
	}
    }
// we all share the same maximum map

  MPI_Allreduce(res_pre, res, (int) NSIDE2NPIX(All.Nside), MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
 
  minmap=1e10;
  maxmap=0;
  for(i = 0; i < NSIDE2NPIX(All.Nside); i++)
    {
      if (minmap > res[i]) minmap=res[i];
      if (maxmap < res[i]) maxmap=res[i];
    }
  if(ThisTask == 0)
    printf("HEALPIX construct: (%f,%f,%f) min=%f max=%f \n",
           SysState.CenterOfMassComp[0][1],SysState.CenterOfMassComp[0][1],SysState.CenterOfMassComp[0][2],minmap,maxmap);

  myfree(res_pre);
  
  res_pre=NULL;


};

#endif
