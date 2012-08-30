#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "allvars.h"
#include "proto.h"

#ifdef VORONOI
#include "voronoi.h"



#ifdef VORONOI_MESHRELAX

void voronoi_meshrelax(void)
{
  voronoi_meshrelax_calc_displacements();

  voronoi_meshrelax_calculate_gradients();

  voronoi_exchange_gradients();

  voronoi_meshrelax_fluxes();

  voronoi_meshrelax_update_particles();

  voronoi_meshrelax_drift();
}



void voronoi_meshrelax_update_particles(void)
{
  int i, k, p;
  double fac, dir, utherm, rel, maxrel, maxrel_all, limiter, *DeltaM;

#define RELCHANGE_LIMIT 0.025

  DeltaM = mymalloc("DeltaM", N_gas * sizeof(double));

  for(i = 0; i < N_gas; i++)
    {
      utherm = SphP[i].Entropy;

      SphP[i].Entropy = P[i].Mass * (0.5 * (P[i].Vel[0] * P[i].Vel[0] + P[i].Vel[1] * P[i].Vel[1] + P[i].Vel[2] * P[i].Vel[2]) + utherm);	/* converts to energy */
      P[i].Vel[0] *= P[i].Mass;	/* converts vel to momentum */
      P[i].Vel[1] *= P[i].Mass;
      P[i].Vel[2] *= P[i].Mass;

      DeltaM[i] = 0;
    }



  for(i = 0; i < Nvf; i++)
    {
      fac = VF[i].area;

      for(k = 0; k < 2; k++)
	{
	  if(k == 0)
	    {
	      if(VF[i].p1->task != ThisTask)
		continue;

	      p = VF[i].p1->index;
	      if(p < 0 || p >= N_gas)
		continue;

	      dir = -fac;
	    }
	  else
	    {
	      if(VF[i].p2->task != ThisTask)
		continue;

	      p = VF[i].p2->index;
	      if(p < 0 || p >= N_gas)
		continue;

	      dir = +fac;
	    }

	  DeltaM[p] += dir * VF[i].mflux;
	}
    }


  for(i = 0, maxrel = 0; i < N_gas; i++)
    {
      rel = fabs(DeltaM[i]) / P[i].Mass;
      if(rel > maxrel)
	maxrel = rel;
    }

  MPI_Allreduce(&maxrel, &maxrel_all, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

  if(maxrel_all > RELCHANGE_LIMIT)
    limiter = RELCHANGE_LIMIT / maxrel_all;
  else
    limiter = 1.0;

  if(ThisTask == 0)
    printf("max relative change=%g (wanted %g)\n", maxrel_all * limiter, maxrel_all);

  myfree(DeltaM);

  for(i = 0; i < Nvf; i++)
    {
      fac = limiter * VF[i].area;

      for(k = 0; k < 2; k++)
	{
	  if(k == 0)
	    {
	      if(VF[i].p1->task != ThisTask)
		continue;

	      p = VF[i].p1->index;
	      if(p < 0 || p >= N_gas)
		continue;

	      dir = -fac;
	    }
	  else
	    {
	      if(VF[i].p2->task != ThisTask)
		continue;

	      p = VF[i].p2->index;
	      if(p < 0 || p >= N_gas)
		continue;

	      dir = +fac;
	    }

	  P[p].Mass += dir * VF[i].mflux;

	  P[p].Vel[0] += dir * VF[i].qflux_x;
	  P[p].Vel[1] += dir * VF[i].qflux_y;
	  P[p].Vel[2] += dir * VF[i].qflux_z;
	  SphP[p].Entropy += dir * VF[i].eflux;
	}
    }

  for(i = 0; i < N_gas; i++)
    {
      if(P[i].Mass <= 0)
	{
	  printf("i=%d task=%d mass=%g\n", i, ThisTask, P[i].Mass);
	  endrun(1231);
	}

      P[i].Vel[0] /= P[i].Mass;	/* converts vel to momentum */
      P[i].Vel[1] /= P[i].Mass;
      P[i].Vel[2] /= P[i].Mass;

      SphP[i].VelPred[0] = P[i].Vel[0];
      SphP[i].VelPred[1] = P[i].Vel[1];
      SphP[i].VelPred[2] = P[i].Vel[2];

      utherm = SphP[i].Entropy / P[i].Mass - 0.5 * (P[i].Vel[0] * P[i].Vel[0] +
						    P[i].Vel[1] * P[i].Vel[1] + P[i].Vel[2] * P[i].Vel[2]);

      SphP[i].Entropy = utherm;

      SphP[i].a.HydroAccel[0] *= limiter;
      SphP[i].a.HydroAccel[1] *= limiter;
      SphP[i].a.HydroAccel[2] *= limiter;
    }
}




void voronoi_meshrelax_fluxes(void)
{
  int i, ri, li;
  double nx, ny, nz, mx, my, mz, px, py, pz, nn, mm;
  double l_dx, l_dy, l_dz, r_dx, r_dy, r_dz, fac;
  double rho_f, velx_f, vely_f, velz_f, press_f;
  double rho_R, press_R, rho_L, press_L;
  double vxtmp_L, vytmp_L, vztmp_L, vxtmp_R, vytmp_R, vztmp_R;
  double evx_f, evy_f, evz_f;
  double velGas_L[3], velGas_R[3];
  MyFloat vel_vertex_L[3], vel_vertex_R[3];
  struct grad_data *gL, *gR;
  double delta_rho, delta_velx, delta_vely, delta_velz, delta_press;

  for(i = 0; i < Nvf; i++)
    {
      li = VF[i].p1->index;
      ri = VF[i].p2->index;

      if(!((li >= 0 && li < N_gas && VF[i].p1->task == ThisTask) ||
	   (ri >= 0 && ri < N_gas && VF[i].p2->task == ThisTask)) || VF[i].area == 0)
	{
	  VF[i].mflux = 0;
	  VF[i].qflux_x = 0;
	  VF[i].qflux_y = 0;
	  VF[i].qflux_z = 0;
	  VF[i].eflux = 0;
	  continue;
	}

      /* normal vector pointing to "right" state */
      nx = VF[i].p2->x - VF[i].p1->x;
      ny = VF[i].p2->y - VF[i].p1->y;
      nz = VF[i].p2->z - VF[i].p1->z;

      nn = sqrt(nx * nx + ny * ny + nz * nz);
      nx /= nn;
      ny /= nn;
      nz /= nn;

      /* need an ortonormal basis */
      if(nx != 0 || ny != 0)
	{
	  mx = -ny;
	  my = nx;
	  mz = 0;
	}
      else
	{
	  mx = 1;
	  my = 0;
	  mz = 0;
	}

      mm = sqrt(mx * mx + my * my + mz * mz);
      mx /= mm;
      my /= mm;
      mz /= mm;

      px = ny * mz - nz * my;
      py = nz * mx - nx * mz;
      pz = nx * my - ny * mx;


      if(li >= N_gas && VF[i].p1->task == ThisTask)
	li -= N_gas;

      if(ri >= N_gas && VF[i].p2->task == ThisTask)
	ri -= N_gas;

      /* interpolation vector for the left state */
      if(VF[i].p1->task == ThisTask)
	{
	  l_dx = VF[i].cx - SphP[li].Center[0];
	  l_dy = VF[i].cy - SphP[li].Center[1];
	  l_dz = VF[i].cz - SphP[li].Center[2];
	}
      else
	{
	  l_dx = VF[i].cx - PrimExch[li].Center[0];
	  l_dy = VF[i].cy - PrimExch[li].Center[1];
	  l_dz = VF[i].cz - PrimExch[li].Center[2];
	}

      /* interpolation vector for the right state */
      if(VF[i].p2->task == ThisTask)
	{
	  r_dx = VF[i].cx - SphP[ri].Center[0];
	  r_dy = VF[i].cy - SphP[ri].Center[1];
	  r_dz = VF[i].cz - SphP[ri].Center[2];
	}
      else
	{
	  r_dx = VF[i].cx - PrimExch[ri].Center[0];
	  r_dy = VF[i].cy - PrimExch[ri].Center[1];
	  r_dz = VF[i].cz - PrimExch[ri].Center[2];
	}

#ifdef PERIODIC
      if(l_dx < -boxHalf_X)
	l_dx += boxSize_X;
      if(l_dx > boxHalf_X)
	l_dx -= boxSize_X;
      if(r_dx < -boxHalf_X)
	r_dx += boxSize_X;
      if(r_dx > boxHalf_X)
	r_dx -= boxSize_X;

      if(l_dy < -boxHalf_Y)
	l_dy += boxSize_Y;
      if(l_dy > boxHalf_Y)
	l_dy -= boxSize_Y;
      if(r_dy < -boxHalf_Y)
	r_dy += boxSize_Y;
      if(r_dy > boxHalf_Y)
	r_dy -= boxSize_Y;

      if(l_dz < -boxHalf_Z)
	l_dz += boxSize_Z;
      if(l_dz > boxHalf_Z)
	l_dz -= boxSize_Z;
      if(r_dz < -boxHalf_Z)
	r_dz += boxSize_Z;
      if(r_dz > boxHalf_Z)
	r_dz -= boxSize_Z;
#endif

      /* get the vertex velocity and gas velocity of the left state */
      if(VF[i].p1->task == ThisTask)
	{
	  velGas_L[0] = P[li].Vel[0];
	  velGas_L[1] = P[li].Vel[1];
	  velGas_L[2] = P[li].Vel[2];
	  vel_vertex_L[0] = SphP[li].a.HydroAccel[0];
	  vel_vertex_L[1] = SphP[li].a.HydroAccel[1];
	  vel_vertex_L[2] = SphP[li].a.HydroAccel[2];

	  rho_L = SphP[li].d.Density;
	  press_L = SphP[li].Pressure;
	  gL = &Grad[li];
	}
      else
	{
	  velGas_L[0] = PrimExch[li].VelPred[0];
	  velGas_L[1] = PrimExch[li].VelPred[1];
	  velGas_L[2] = PrimExch[li].VelPred[2];
	  vel_vertex_L[0] = PrimExch[li].HydroAccel[0];
	  vel_vertex_L[1] = PrimExch[li].HydroAccel[1];
	  vel_vertex_L[2] = PrimExch[li].HydroAccel[2];

	  rho_L = PrimExch[li].Density;
	  press_L = PrimExch[li].Pressure;
	  gL = &GradExch[li];
	}



      /* get the vertex velocity and gas velocity of the right state */
      if(VF[i].p2->task == ThisTask)
	{
	  velGas_R[0] = P[ri].Vel[0];
	  velGas_R[1] = P[ri].Vel[1];
	  velGas_R[2] = P[ri].Vel[2];
	  vel_vertex_R[0] = SphP[ri].a.HydroAccel[0];
	  vel_vertex_R[1] = SphP[ri].a.HydroAccel[1];
	  vel_vertex_R[2] = SphP[ri].a.HydroAccel[2];

	  rho_R = SphP[ri].d.Density;
	  press_R = SphP[ri].Pressure;
	  gR = &Grad[ri];
	}
      else
	{
	  velGas_R[0] = PrimExch[ri].VelPred[0];
	  velGas_R[1] = PrimExch[ri].VelPred[1];
	  velGas_R[2] = PrimExch[ri].VelPred[2];
	  vel_vertex_R[0] = PrimExch[ri].HydroAccel[0];
	  vel_vertex_R[1] = PrimExch[ri].HydroAccel[1];
	  vel_vertex_R[2] = PrimExch[ri].HydroAccel[2];

	  rho_R = PrimExch[ri].Density;
	  press_R = PrimExch[ri].Pressure;
	  gR = &GradExch[ri];
	}



      /* rough motion of mid-point of edge */
      evx_f = 0.5 * (vel_vertex_L[0] + vel_vertex_R[0]);
      evy_f = 0.5 * (vel_vertex_L[1] + vel_vertex_R[1]);
      evz_f = 0.5 * (vel_vertex_L[2] + vel_vertex_R[2]);

      double cx, cy, cz, facv;

      cx = VF[i].cx - 0.5 * (VF[i].p2->x + VF[i].p1->x);
      cy = VF[i].cy - 0.5 * (VF[i].p2->y + VF[i].p1->y);
      cz = VF[i].cz - 0.5 * (VF[i].p2->z + VF[i].p1->z);

      facv = (cx * (vel_vertex_L[0] - vel_vertex_R[0]) +
	      cy * (vel_vertex_L[1] - vel_vertex_R[1]) + cz * (vel_vertex_L[2] - vel_vertex_R[2])) / nn;

      evx_f += facv * nx;
      evy_f += facv * ny;
      evz_f += facv * nz;

      vxtmp_L = velGas_L[0] - evx_f;
      vytmp_L = velGas_L[1] - evy_f;
      vztmp_L = velGas_L[2] - evz_f;

      vxtmp_R = velGas_R[0] - evx_f;
      vytmp_R = velGas_R[1] - evy_f;
      vztmp_R = velGas_R[2] - evz_f;


      /* calculate the extrapolated left state */

      delta_rho = gL->drho[0] * l_dx + gL->drho[1] * l_dy + gL->drho[2] * l_dz;
      delta_velx = gL->dvel[0][0] * l_dx + gL->dvel[0][1] * l_dy + gL->dvel[0][2] * l_dz;
      delta_vely = gL->dvel[1][0] * l_dx + gL->dvel[1][1] * l_dy + gL->dvel[1][2] * l_dz;
      delta_velz = gL->dvel[2][0] * l_dx + gL->dvel[2][1] * l_dy + gL->dvel[2][2] * l_dz;
      delta_press = gL->dpress[0] * l_dx + gL->dpress[1] * l_dy + gL->dpress[2] * l_dz;

      if(press_L + delta_press < 0 || rho_L + delta_rho < 0)
	{
	  delta_rho = 0;
	  delta_press = 0;
	  delta_velx = 0;
	  delta_vely = 0;
	  delta_velz = 0;
	}

      rho_L += delta_rho;
      vxtmp_L += delta_velx;
      vytmp_L += delta_vely;
      vztmp_L += delta_velz;
      press_L += delta_press;



      /* calculate the extrapolated right state */

      delta_rho = gR->drho[0] * r_dx + gR->drho[1] * r_dy + gR->drho[2] * r_dz;
      delta_velx = gR->dvel[0][0] * r_dx + gR->dvel[0][1] * r_dy + gR->dvel[0][2] * r_dz;
      delta_vely = gR->dvel[1][0] * r_dx + gR->dvel[1][1] * r_dy + gR->dvel[1][2] * r_dz;
      delta_velz = gR->dvel[2][0] * r_dx + gR->dvel[2][1] * r_dy + gR->dvel[2][2] * r_dz;
      delta_press = gR->dpress[0] * r_dx + gR->dpress[1] * r_dy + gR->dpress[2] * r_dz;

      if(press_R + delta_press < 0 || rho_R + delta_rho < 0)
	{
	  delta_rho = 0;
	  delta_press = 0;
	  delta_velx = 0;
	  delta_vely = 0;
	  delta_velz = 0;
	}

      rho_R += delta_rho;
      vxtmp_R += delta_velx;
      vytmp_R += delta_vely;
      vztmp_R += delta_velz;
      press_R += delta_press;

      /* just solve the advection equation */

      double ev = evx_f * nx + evy_f * ny + evz_f * nz;

      if(ev < 0)
	{
	  rho_f = rho_L;
	  velx_f = vxtmp_L;
	  vely_f = vytmp_L;
	  velz_f = vztmp_L;
	  press_f = press_L;
	}
      else
	{
	  rho_f = rho_R;
	  velx_f = vxtmp_R;
	  vely_f = vytmp_R;
	  velz_f = vztmp_R;
	  press_f = press_R;
	}


      /* add the facet's velocity again */
      velx_f += evx_f;
      vely_f += evy_f;
      velz_f += evz_f;


      /* compute net flux with dot-product of outward normal and area of face */
      /* multiplication with area and time-step comes later */


      fac = (-evx_f) * nx + (-evy_f) * ny + (-evz_f) * nz;

      VF[i].mflux = rho_f * fac;

      VF[i].qflux_x = rho_f * velx_f * fac;

      VF[i].qflux_y = rho_f * vely_f * fac;

      VF[i].qflux_z = rho_f * velz_f * fac;

      VF[i].eflux = press_f / GAMMA_MINUS1 * fac;
    }

}


void voronoi_meshrelax_drift(void)
{
  int i;

  for(i = 0; i < N_gas; i++)
    {
      P[i].Pos[0] += SphP[i].a.HydroAccel[0];
      P[i].Pos[1] += SphP[i].a.HydroAccel[1];
      P[i].Pos[2] += SphP[i].a.HydroAccel[2];
    }

  for(i = 0; i < N_gas; i++)
    {
      SphP[i].a.HydroAccel[0] = 0;
      SphP[i].a.HydroAccel[1] = 0;
      SphP[i].a.HydroAccel[2] = 0;
    }
}



void voronoi_meshrelax_calc_displacements(void)
{
  int i;
  double fac, disp2_sum, disp2_sum_all, disp_rms, rad;
  double press, press_sum, press2_sum, press_sum_all, press2_sum_all, press_rms;
  double norm_disp2, max_disp2, max_disp2_all, minmass, maxmass, minmass_all, maxmass_all;

  minmass = maxmass = P[0].Mass;

  for(i = 0, disp2_sum = press_sum = press2_sum = max_disp2 = 0; i < N_gas; i++)
    {

#ifdef TWODIMS
      rad = sqrt(SphP[i].Volume / M_PI);
#else
      rad = pow(SphP[i].Volume * 3.0 / (4.0 * M_PI), 1.0 / 3);
      SphP[i].a.HydroAccel[0] /= rad;
      SphP[i].a.HydroAccel[1] /= rad;
      SphP[i].a.HydroAccel[2] /= rad;
#endif

      norm_disp2 = (SphP[i].a.HydroAccel[0] * SphP[i].a.HydroAccel[0] +
		    SphP[i].a.HydroAccel[1] * SphP[i].a.HydroAccel[1] +
		    SphP[i].a.HydroAccel[2] * SphP[i].a.HydroAccel[2]) / (rad * rad);

      disp2_sum += norm_disp2;

      if(norm_disp2 > max_disp2)
	max_disp2 = norm_disp2;

      press = 1.0 / (P[i].Mass / All.MeanMass);

      press_sum += press;
      press2_sum += press * press;

      if(P[i].Mass < minmass)
	minmass = P[i].Mass;
      if(P[i].Mass > maxmass)
	maxmass = P[i].Mass;
    }


  MPI_Allreduce(&disp2_sum, &disp2_sum_all, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&press_sum, &press_sum_all, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&press2_sum, &press2_sum_all, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&max_disp2, &max_disp2_all, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

  MPI_Allreduce(&minmass, &minmass_all, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(&maxmass, &maxmass_all, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

  press_rms = press2_sum_all / All.TotN_gas - pow(press_sum_all / All.TotN_gas, 2);
  if(press_rms > 0)
    press_rms = sqrt(press_rms);
  else
    press_rms = 0;

  disp_rms = sqrt(disp2_sum_all / All.TotN_gas);

  fac = 0.05;

  if((fac * disp_rms) > 0.01)
    fac = 0.01 / disp_rms;


  if(ThisTask == 0)
    {
      printf
	("\nMESHRELAX: disp_rms=%g  press_rms=%g  actual_disp_rms=%g  max_disp=%g  minmass_all=%g maxmass_all=%g\n",
	 disp_rms, press_rms, disp_rms * fac, sqrt(max_disp2_all) * fac, minmass_all, maxmass_all);
    }

  for(i = 0; i < N_gas; i++)
    {
      SphP[i].a.HydroAccel[0] *= fac;
      SphP[i].a.HydroAccel[1] *= fac;
      SphP[i].a.HydroAccel[2] *= fac;
    }

  /* the vector SphP[i].a.HydroAccel[] is now the displacement we adopt for the current step */
}




void voronoi_meshrelax_calculate_gradients(void)
{
  int i, j, k, t, q, s;
  point *p;
  double fac, n[3], nn, d[3], velGas[3];

  struct max_data
  {
    double max_rho, max_vel[3], max_press;
    double min_rho, min_vel[3], min_press;
  }
   *minmax;


  Grad = mymalloc("Grad", N_gas * sizeof(struct grad_data));

  minmax = mymalloc("minmax", N_gas * sizeof(struct max_data));


  for(i = 0; i < N_gas; i++)
    {
      minmax[i].max_rho = minmax[i].max_vel[0] = minmax[i].max_vel[1] = minmax[i].max_vel[2] =
	minmax[i].max_press = -MAX_REAL_NUMBER;
      minmax[i].min_rho = minmax[i].min_vel[0] = minmax[i].min_vel[1] = minmax[i].min_vel[2] =
	minmax[i].min_press = +MAX_REAL_NUMBER;

      for(j = 0; j < 3; j++)
	{
	  Grad[i].drho[j] = 0;
	  Grad[i].dvel[0][j] = 0;
	  Grad[i].dvel[1][j] = 0;
	  Grad[i].dvel[2][j] = 0;
	  Grad[i].dpress[j] = 0;
	}
    }

  for(i = 0; i < Nvf; i++)
    {
      n[0] = VF[i].p1->x - VF[i].p2->x;
      n[1] = VF[i].p1->y - VF[i].p2->y;
      n[2] = VF[i].p1->z - VF[i].p2->z;

      nn = sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);

      for(j = 0; j < 3; j++)
	n[j] /= nn;

      for(k = 0; k < 2; k++)
	{
	  if(k == 0)
	    {
	      p = VF[i].p2;
	      q = p->index;

	      s = VF[i].p1->index;
	      t = VF[i].p1->task;
	    }
	  else
	    {
	      p = VF[i].p1;
	      q = p->index;

	      s = VF[i].p2->index;
	      t = VF[i].p2->task;
	    }

	  if(p->task == ThisTask && q >= 0 && q < N_gas)
	    {
	      if(k == 0)
		{
		  fac = 0.5 * VF[i].area / SphP[q].Volume;
		}
	      else
		{
		  fac = -0.5 * VF[i].area / SphP[q].Volume;
		}


	      if(t == ThisTask)
		{
		  if(s >= N_gas)
		    s -= N_gas;

		  if(s >= 0)
		    {
		      velGas[0] = P[s].Vel[0];
		      velGas[1] = P[s].Vel[1];
		      velGas[2] = P[s].Vel[2];

		      for(j = 0; j < 3; j++)
			{
			  Grad[q].drho[j] += fac * n[j] * SphP[s].d.Density;
			  Grad[q].dvel[0][j] += fac * n[j] * velGas[0];
			  Grad[q].dvel[1][j] += fac * n[j] * velGas[1];
			  Grad[q].dvel[2][j] += fac * n[j] * velGas[2];
			  Grad[q].dpress[j] += fac * n[j] * SphP[s].Pressure;

			  if(VF[i].area > 0)
			    {
			      if(minmax[q].max_rho < SphP[s].d.Density)
				minmax[q].max_rho = SphP[s].d.Density;

			      if(minmax[q].max_vel[0] < velGas[0])
				minmax[q].max_vel[0] = velGas[0];
			      if(minmax[q].max_vel[1] < velGas[1])
				minmax[q].max_vel[1] = velGas[1];
			      if(minmax[q].max_vel[2] < velGas[2])
				minmax[q].max_vel[2] = velGas[2];

			      if(minmax[q].max_press < SphP[s].Pressure)
				minmax[q].max_press = SphP[s].Pressure;

			      if(minmax[q].min_rho > SphP[s].d.Density)
				minmax[q].min_rho = SphP[s].d.Density;

			      if(minmax[q].min_vel[0] > velGas[0])
				minmax[q].min_vel[0] = velGas[0];
			      if(minmax[q].min_vel[1] > velGas[1])
				minmax[q].min_vel[1] = velGas[1];
			      if(minmax[q].min_vel[2] > velGas[2])
				minmax[q].min_vel[2] = velGas[2];

			      if(minmax[q].min_press > SphP[s].Pressure)
				minmax[q].min_press = SphP[s].Pressure;
			    }
			}
		    }
		}
	      else
		{
		  for(j = 0; j < 3; j++)
		    {
		      velGas[0] = PrimExch[s].VelPred[0];
		      velGas[1] = PrimExch[s].VelPred[1];
		      velGas[2] = PrimExch[s].VelPred[2];

		      Grad[q].drho[j] += fac * n[j] * PrimExch[s].Density;
		      Grad[q].dvel[0][j] += fac * n[j] * velGas[0];
		      Grad[q].dvel[1][j] += fac * n[j] * velGas[1];
		      Grad[q].dvel[2][j] += fac * n[j] * velGas[2];
		      Grad[q].dpress[j] += fac * n[j] * PrimExch[s].Pressure;

		      if(VF[i].area > 0)
			{
			  if(minmax[q].max_rho < PrimExch[s].Density)
			    minmax[q].max_rho = PrimExch[s].Density;

			  if(minmax[q].max_vel[0] < velGas[0])
			    minmax[q].max_vel[0] = velGas[0];
			  if(minmax[q].max_vel[1] < velGas[1])
			    minmax[q].max_vel[1] = velGas[1];
			  if(minmax[q].max_vel[2] < velGas[2])
			    minmax[q].max_vel[2] = velGas[2];

			  if(minmax[q].max_press < PrimExch[s].Pressure)
			    minmax[q].max_press = PrimExch[s].Pressure;

			  if(minmax[q].min_rho > PrimExch[s].Density)
			    minmax[q].min_rho = PrimExch[s].Density;

			  if(minmax[q].min_vel[0] > velGas[0])
			    minmax[q].min_vel[0] = velGas[0];
			  if(minmax[q].min_vel[1] > velGas[1])
			    minmax[q].min_vel[1] = velGas[1];
			  if(minmax[q].min_vel[2] > velGas[2])
			    minmax[q].min_vel[2] = velGas[2];

			  if(minmax[q].min_press > PrimExch[s].Pressure)
			    minmax[q].min_press = PrimExch[s].Pressure;
			}
		    }
		}
	    }
	}
    }


  /* let's now implement a slope limitation if appropriate */

  for(i = 0; i < Nvf; i++)
    {
      for(j = 0; j < 2; j++)
	{
	  if(j == 0)
	    p = VF[i].p1;
	  else
	    p = VF[i].p2;

	  if(p->task == ThisTask && p->index >= 0 && p->index < N_gas)
	    {
	      q = p->index;

	      d[0] = VF[i].cx - SphP[q].Center[0];
	      d[1] = VF[i].cy - SphP[q].Center[1];
	      d[2] = VF[i].cz - SphP[q].Center[2];

#ifdef PERIODIC
	      if(d[0] < -boxHalf_X)
		d[0] += boxSize_X;
	      if(d[0] > boxHalf_X)
		d[0] -= boxSize_X;

	      if(d[1] < -boxHalf_Y)
		d[1] += boxSize_Y;
	      if(d[1] > boxHalf_Y)
		d[1] -= boxSize_Y;

	      if(d[2] < -boxHalf_Z)
		d[2] += boxSize_Z;
	      if(d[2] > boxHalf_Z)
		d[2] -= boxSize_Z;
#endif

	      if(VF[i].area > 0)
		{
		  limit_gradient(d, SphP[q].d.Density, minmax[q].min_rho, minmax[q].max_rho, Grad[q].drho);
		  limit_gradient(d, P[q].Vel[0], minmax[q].min_vel[0], minmax[q].max_vel[0], Grad[q].dvel[0]);
		  limit_gradient(d, P[q].Vel[1], minmax[q].min_vel[1], minmax[q].max_vel[1], Grad[q].dvel[1]);
		  limit_gradient(d, P[q].Vel[2], minmax[q].min_vel[2], minmax[q].max_vel[2], Grad[q].dvel[2]);
		  limit_gradient(d, SphP[q].Pressure, minmax[q].min_press, minmax[q].max_press,
				 Grad[q].dpress);
		}

	    }
	}
    }

  myfree(minmax);
}




void limit_gradient(double *d, double phi, double min_phi, double max_phi, double *dphi)
{
  double dp, fac;

  dp = dphi[0] * d[0] + dphi[1] * d[1] + dphi[2] * d[2];

  if(dp > 0)
    {
      if(phi + dp > max_phi)
	{
	  if(max_phi > phi)
	    fac = (max_phi - phi) / dp;
	  else
	    fac = 0;

	  if(fac < 0 || fac > 1)
	    {
	      printf("fac=%g\n", fac);
	      printf("dp=%g max_phi=%g  phi=%g\n", dp, max_phi, phi);
	      endrun(135);
	    }

	  dphi[0] *= fac;
	  dphi[1] *= fac;
	  dphi[2] *= fac;
	}
    }
  else if(dp < 0)
    {
      if(phi + dp < min_phi)
	{
	  if(min_phi < phi)
	    fac = (min_phi - phi) / dp;
	  else
	    fac = 0;

	  if(fac < 0 || fac > 1)
	    {
	      printf("fac=%g\n", fac);
	      printf("Dp=%g max_phi=%g  phi=%g\n", dp, max_phi, phi);
	      endrun(133);
	    }

	  dphi[0] *= fac;
	  dphi[1] *= fac;
	  dphi[2] *= fac;
	}
    }
}




#endif
#endif
