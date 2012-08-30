#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "allvars.h"
#include "proto.h"

#ifdef VORONOI
#include "voronoi.h"


void voronoi_hydro_force(void)
{
  int i, j, q1, q2, li, ri;
  double length, pressure1, pressure2, cx, cy, cz, c, fac1, fac2, ex, ey, ez, forcex, forcey, forcez;
  double mass1, mass2, dEdt, fviscx, fviscy, fviscz, w, dens1, dens2, entr1, entr2, volume1, volume2;

#ifdef VORONOI_SHAPESCHEME
  double w1, w2;
#endif
  MyFloat *vel1, *vel2, *center1, *center2;
  point *p1, *p2;

  for(i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i])
    {
      if(P[i].Type == 0)
	{
	  SphP[i].e.DtEntropy = 0;

	  for(j = 0; j < 3; j++)
	    SphP[i].a.dHydroAccel[j] = 0;
	}
    }


  voronoi_exchange_ghost_variables();

  for(i = 0; i < Nvf; i++)
    {
      p1 = &DP[VF[i].p1];
      p2 = &DP[VF[i].p2];

      q1 = li = p1->index;
      q2 = ri = p2->index;

      if(!
	 ((li >= 0 && li < N_gas && p1->task == ThisTask) || (ri >= 0 && ri < N_gas && p2->task == ThisTask)))
	continue;

      if(li < 0 || ri < 0)
	continue;

      if(li >= N_gas && p1->task == ThisTask)
	li -= N_gas;

      if(ri >= N_gas && p2->task == ThisTask)
	ri -= N_gas;

      if(p1->task == ThisTask)
	{
	  pressure1 = SphP[li].Pressure;
	  dens1 = SphP[li].d.Density;
	  vel1 = SphP[li].VelPred;
	  mass1 = P[li].Mass;
	  entr1 = SphP[li].Entropy;
	  center1 = SphP[li].Center;
	  volume1 = SphP[li].Volume;
#ifdef VORONOI_SHAPESCHEME
	  w1 = SphP[li].W;
#endif
	}
      else
	{
	  pressure1 = PrimExch[q1].Pressure;
	  dens1 = PrimExch[q1].Density;
	  vel1 = PrimExch[q1].VelPred;
	  mass1 = PrimExch[q1].Mass;
	  entr1 = PrimExch[q1].Entropy;
	  center1 = PrimExch[q1].Center;
	  volume1 = PrimExch[q1].Volume;
#ifdef VORONOI_SHAPESCHEME
	  w1 = PrimExch[q1].W;
#endif
	}

      if(p2->task == ThisTask)
	{
	  pressure2 = SphP[ri].Pressure;
	  dens2 = SphP[ri].d.Density;
	  vel2 = SphP[ri].VelPred;
	  mass2 = P[ri].Mass;
	  entr2 = SphP[ri].Entropy;
	  center2 = SphP[ri].Center;
	  volume2 = SphP[ri].Volume;
#ifdef VORONOI_SHAPESCHEME
	  w2 = SphP[ri].W;
#endif
	}
      else
	{
	  pressure2 = PrimExch[q2].Pressure;
	  dens2 = PrimExch[q2].Density;
	  vel2 = PrimExch[q2].VelPred;
	  mass2 = PrimExch[q2].Mass;
	  entr2 = PrimExch[q2].Entropy;
	  center2 = PrimExch[q2].Center;
	  volume2 = PrimExch[q2].Volume;
#ifdef VORONOI_SHAPESCHEME
	  w2 = PrimExch[q2].W;
#endif
	}


      cx = p2->x - p1->x;
      cy = p2->y - p1->y;
      cz = p2->z - p1->z;

      c = sqrt(cx * cx + cy * cy + cz * cz);	/* distance of the two points */
      length = VF[i].area;	/* length/area of common face */


      fac1 = 0.5 * (pressure2 + pressure1) * length / c;
      fac2 = (pressure2 - pressure1) * length / c;


      ex = VF[i].cx - 0.5 * (p1->x + p2->x);
      ey = VF[i].cy - 0.5 * (p1->y + p2->y);
      ez = VF[i].cz - 0.5 * (p1->z + p2->z);


      forcex = -fac1 * cx - fac2 * ex;
      forcey = -fac1 * cy - fac2 * ey;
      forcez = -fac1 * cz - fac2 * ez;


      /* calculate viscous force */

      w = cx * (vel2[0] - vel1[0]) + cy * (vel2[1] - vel1[1]) + cz * (vel2[2] - vel1[2]);
      if(w < 0)
	{
	  w /= c;

	  /*
	     fviscx = - 0.5 * All.ArtBulkViscConst * mass1 * mass2 / (mass1 + mass2) * cx * w * w / (c * c);
	     fviscy = - 0.5 * All.ArtBulkViscConst * mass1 * mass2 / (mass1 + mass2) * cy * w * w / (c * c);
	     fviscz = - 0.5 * All.ArtBulkViscConst * mass1 * mass2 / (mass1 + mass2) * cz * w * w / (c * c);
	   */

	  double csound, pvisc;

	  /* calculate viscous force */

	  csound = 0.5 * (sqrt(GAMMA * pressure1 / dens1) + sqrt(GAMMA * pressure2 / dens2));
	  pvisc = 0.5 * All.ArtBulkViscConst * (dens1 + dens2) * (-w * csound + 1.5 * w * w);

	  fviscx = -pvisc * length * cx / c;
	  fviscy = -pvisc * length * cy / c;
	  fviscz = -pvisc * length * cz / c;


	  /* rate at which energy is dissipated */
	  dEdt = fviscx * (vel2[0] - vel1[0]) + fviscy * (vel2[1] - vel1[1]) + fviscz * (vel2[2] - vel1[2]);

	  if(dEdt < 0)
	    endrun(88);
	}
      else
	{
	  fviscx = 0;
	  fviscy = 0;
	  fviscz = 0;

	  dEdt = 0;
	}


#ifdef VORONOI_SHAPESCHEME
      double s1x = VF[i].cx - p1->x;
      double s1y = VF[i].cy - p1->y;
      double s1z = VF[i].cz - p1->z;

      double s2x = VF[i].cx - p2->x;
      double s2y = VF[i].cy - p2->y;
      double s2z = VF[i].cz - p2->z;

      double d1x = NEAREST_X(p1->x - center1[0]);
      double d1y = NEAREST_Y(p1->y - center1[1]);
      double d1z = NEAREST_Z(p1->z - center1[2]);

      double d2x = NEAREST_X(p2->x - center2[0]);
      double d2y = NEAREST_Y(p2->y - center2[1]);
      double d2z = NEAREST_Z(p2->z - center2[2]);

      double d1 = d1x * d1x + d1y * d1y + d1z * d1z;
      double d2 = d2x * d2x + d2y * d2y + d2z * d2z;

      double square1 = 1 + All.VoronoiStiffNess * d1 * pow(volume1, -2.0 / DIMS);
      double square2 = 1 + All.VoronoiStiffNess * d2 * pow(volume2, -2.0 / DIMS);

      double curly1 = 1 + All.VoronoiRoundNess * (w1 * pow(volume1, -2.0 / DIMS) - SHAPE_FAC);
      double curly2 = 1 + All.VoronoiRoundNess * (w2 * pow(volume2, -2.0 / DIMS) - SHAPE_FAC);


      double Q1 = All.VoronoiStiffNess / GAMMA_MINUS1 * pressure1 * pow(volume1, 1.0 - 2.0 / DIMS) * curly1;
      double Q2 = All.VoronoiStiffNess / GAMMA_MINUS1 * pressure2 * pow(volume2, 1.0 - 2.0 / DIMS) * curly2;

      double L1 = All.VoronoiRoundNess / GAMMA_MINUS1 * pressure1 * pow(volume1, 1.0 - 2.0 / DIMS) * square1;
      double L2 = All.VoronoiRoundNess / GAMMA_MINUS1 * pressure2 * pow(volume2, 1.0 - 2.0 / DIMS) * square2;

      pressure1 *= (square1 * curly1 +
		    All.VoronoiStiffNess * (2.0 / DIMS) / GAMMA_MINUS1 * d1 / pow(volume1,
										  2.0 / DIMS) * curly1 +
		    All.VoronoiRoundNess * (2.0 / DIMS) / GAMMA_MINUS1 * w1 / pow(volume1,
										  2.0 / DIMS) * square1);

      pressure2 *= (square2 * curly2 +
		    All.VoronoiStiffNess * (2.0 / DIMS) / GAMMA_MINUS1 * d2 / pow(volume2,
										  2.0 / DIMS) * curly2 +
		    All.VoronoiRoundNess * (2.0 / DIMS) / GAMMA_MINUS1 * w2 / pow(volume2,
										  2.0 / DIMS) * square2);

      pressure1 += 2 * d1 * Q1 / volume1 + w1 * L1 / volume1;
      pressure2 += 2 * d2 * Q2 / volume2 + w2 * L2 / volume2;

      fac1 = 0.5 * (pressure2 + pressure1) * length / c;
      fac2 = (pressure2 - pressure1) * length / c;

      forcex = -fac1 * cx - fac2 * ex;
      forcey = -fac1 * cy - fac2 * ey;
      forcez = -fac1 * cz - fac2 * ez;

      double g1x = NEAREST_X(VF[i].cx - center1[0]);
      double g1y = NEAREST_Y(VF[i].cy - center1[1]);
      double g1z = NEAREST_Z(VF[i].cz - center1[2]);

      double g2x = NEAREST_X(VF[i].cx - center2[0]);
      double g2y = NEAREST_Y(VF[i].cy - center2[1]);
      double g2z = NEAREST_Z(VF[i].cz - center2[2]);

      double f1x = L1 / volume1 * g1x;
      double f1y = L1 / volume1 * g1y;
      double f1z = L1 / volume1 * g1z;

      double f2x = L2 / volume2 * g2x;
      double f2y = L2 / volume2 * g2y;
      double f2z = L2 / volume2 * g2z;


      double e1x = Q1 / volume1 * d1x;
      double e1y = Q1 / volume1 * d1y;
      double e1z = Q1 / volume1 * d1z;

      double e2x = Q2 / volume2 * d2x;
      double e2y = Q2 / volume2 * d2y;
      double e2z = Q2 / volume2 * d2z;

      double prod2 = 2 * ((s1x * e1x + s1y * e1y + s1z * e1z) - (s2x * e2x + s2y * e2y + s2z * e2z));

      prod2 += ((g2x * f2x + g2y * f2y + g2z * f2z) - (g1x * f1x + g1y * f1y + g1z * f1z));
      prod2 += (VF[i].T_xx + VF[i].T_yy + VF[i].T_zz) * (L2 / volume2 - L1 / volume1);


      double edx = 2 * (e1x - e2x + f2x - f1x);
      double edy = 2 * (e1y - e2y + f2y - f1y);
      double edz = 2 * (e1z - e2z + f2z - f1z);

      double tfx = VF[i].T_xx * edx + VF[i].T_xy * edy + VF[i].T_xz * edz;
      double tfy = VF[i].T_xy * edx + VF[i].T_yy * edy + VF[i].T_yz * edz;
      double tfz = VF[i].T_xz * edx + VF[i].T_yz * edy + VF[i].T_zz * edz;

#ifndef TWODIMS
      double ggx = (L2 / volume2 - L1 / volume1) * VF[i].g_x;
      double ggy = (L2 / volume2 - L1 / volume1) * VF[i].g_y;
      double ggz = (L2 / volume2 - L1 / volume1) * VF[i].g_z;
#else
      double ggx = 0, ggy = 0, ggz = 0;
#endif

      double fshape1_x = length / c * (tfx + s1x * prod2 + ggx);
      double fshape1_y = length / c * (tfy + s1y * prod2 + ggy);
      double fshape1_z = length / c * (tfz + s1z * prod2 + ggz);

      double fshape2_x = length / c * (-tfx - s2x * prod2 - ggx);
      double fshape2_y = length / c * (-tfy - s2y * prod2 - ggy);
      double fshape2_z = length / c * (-tfz - s2z * prod2 - ggz);
#endif



#ifdef VORONOI_MESHRELAX
      fviscx = fviscy = fviscz = dEdt = 0;
#endif


      if(p1->task == ThisTask && q1 >= 0 && q1 < N_gas)
	{
	  if(TimeBinActive[P[q1].TimeBin])
	    {
	      SphP[q1].a.dHydroAccel[0] += (forcex + fviscx) / mass1;
	      SphP[q1].a.dHydroAccel[1] += (forcey + fviscy) / mass1;
	      SphP[q1].a.dHydroAccel[2] += (forcez + fviscz) / mass1;

#ifdef VORONOI_SHAPESCHEME
	      SphP[q1].a.dHydroAccel[0] += fshape1_x / mass1;
	      SphP[q1].a.dHydroAccel[1] += fshape1_y / mass1;
	      SphP[q1].a.dHydroAccel[2] += fshape1_z / mass1;
	      SphP[q1].e.DtEntropy += 0.5 * dEdt * GAMMA_MINUS1 / pow(dens1, GAMMA_MINUS1) / mass1 /
		(square1 * curly1);
#else
	      SphP[q1].e.DtEntropy += 0.5 * dEdt * GAMMA_MINUS1 / pow(dens1, GAMMA_MINUS1) / mass1;
#endif
	    }
	}

      if(p2->task == ThisTask && q2 >= 0 && q2 < N_gas)
	{
	  if(TimeBinActive[P[q2].TimeBin])
	    {
	      SphP[q2].a.dHydroAccel[0] -= (forcex + fviscx) / mass2;
	      SphP[q2].a.dHydroAccel[1] -= (forcey + fviscy) / mass2;
	      SphP[q2].a.dHydroAccel[2] -= (forcez + fviscz) / mass2;

#ifdef VORONOI_SHAPESCHEME
	      SphP[q2].a.dHydroAccel[0] += fshape2_x / mass2;
	      SphP[q2].a.dHydroAccel[1] += fshape2_y / mass2;
	      SphP[q2].a.dHydroAccel[2] += fshape2_z / mass2;
	      SphP[q2].e.DtEntropy += 0.5 * dEdt * GAMMA_MINUS1 / pow(dens2, GAMMA_MINUS1) / mass2 /
		(square2 * curly2);
#else
	      SphP[q2].e.DtEntropy += 0.5 * dEdt * GAMMA_MINUS1 / pow(dens2, GAMMA_MINUS1) / mass2;
#endif
	    }
	}
    }


#ifdef VORONOI_SHAPESCHEME
  for(i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i])
    {
      if(P[i].Type == 0)
	{
	  double Q =
	    All.VoronoiStiffNess / GAMMA_MINUS1 * SphP[i].Pressure * pow(SphP[i].Volume, 1.0 - 2.0 / DIMS);

	  double dx = P[i].Pos[0] - SphP[i].Center[0];
	  double dy = P[i].Pos[1] - SphP[i].Center[1];
	  double dz = P[i].Pos[2] - SphP[i].Center[2];

	  SphP[i].a.dHydroAccel[0] += (-2 * Q * dx) / P[i].Mass;
	  SphP[i].a.dHydroAccel[1] += (-2 * Q * dy) / P[i].Mass;
	  SphP[i].a.dHydroAccel[2] += (-2 * Q * dz) / P[i].Mass;
	}
    }
#endif




#ifdef VORONOI_MESHRELAX
  voronoi_meshrelax();

  myfree(Grad);
  myfree(GradExch);
#endif

  myfree(PrimExch);

  myfree(List_P);
  myfree(ListExports);

  myfree(DT);
  myfree(DP - 5);
  myfree(VF);			/* free the list of faces */


  /*
     #ifdef VORONOI_SHAPESCHEME
     savepositions(0);
     endrun(0);
     #endif
   */
}

#endif
