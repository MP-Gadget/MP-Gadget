#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>


#include "allvars.h"
#include "proto.h"

#ifndef DEBUG
#define NDEBUG
#endif
#include <assert.h>


#ifdef DISTORTIONTENSORPS
  /*
     We need to synchronize the phase-space distortion tensor if we need to access
     all entries, because the velocity part is a half step behind the configuration 
     space part due to leapfrog integration scheme.
     Note that the caustic identification does not need this sync since it only depends
     on the configuration space part. But the maximum density calculation needs it
     because it requires full 6D phase-space distortion information. Also the
     phase-space density calculation (determinant of the full tensor) requires a
     sync. The sync is also called in io.c for the output. This is the same procedure
     that is applied to the position and velocity information (see io.c).
   */

void get_half_kick_distortion(int pindex, MyDouble half_kick_add[6][6])
{
  double dt_gravkick;
  int j1, j2, j;
  int dt_step;
  MyDouble dv_distortion_tensorps[6][6];


  for(j1 = 0; j1 < 6; j1++)
    for(j2 = 0; j2 < 6; j2++)
      half_kick_add[j1][j2] = 0;

  /* get time step of that particle from its TimeBin */
  dt_step = (P[pindex].TimeBin ? (1 << P[pindex].TimeBin) : 0);

  /* get gravkick from dt_step */
  if(All.ComovingIntegrationOn)
    {
      dt_gravkick = get_gravkick_factor(P[pindex].Ti_begstep, All.Ti_Current) -
	get_gravkick_factor(P[pindex].Ti_begstep, P[pindex].Ti_begstep + dt_step / 2);
    }
  else
    dt_gravkick = (All.Ti_Current - (P[pindex].Ti_begstep + dt_step / 2)) * All.Timebase_interval;


  /* now we do the distortiontensor half kick */
  for(j1 = 0; j1 < 3; j1++)
    for(j2 = 0; j2 < 3; j2++)
      {
	dv_distortion_tensorps[j1 + 3][j2] = 0.0;
	dv_distortion_tensorps[j1 + 3][j2 + 3] = 0.0;

	/* the 'acceleration' is given by the product of tidaltensor and distortiontensor */
	for(j = 0; j < 3; j++)
	  {
	    dv_distortion_tensorps[j1 + 3][j2] +=
	      P[pindex].tidal_tensorps[j1][j] * P[pindex].distortion_tensorps[j][j2];
	    dv_distortion_tensorps[j1 + 3][j2 + 3] +=
	      P[pindex].tidal_tensorps[j1][j] * P[pindex].distortion_tensorps[j][j2 + 3];
	  }
	dv_distortion_tensorps[j1 + 3][j2] *= dt_gravkick;
	dv_distortion_tensorps[j1 + 3][j2 + 3] *= dt_gravkick;

	/* add it to the distortiontensor 'velocities' half kick add */
	half_kick_add[j1 + 3][j2] += dv_distortion_tensorps[j1 + 3][j2];
	half_kick_add[j1 + 3][j2 + 3] += dv_distortion_tensorps[j1 + 3][j2 + 3];
      }


#ifdef PMGRID
  double dt_gravkick_pm;

  if(All.ComovingIntegrationOn)
    dt_gravkick_pm = get_gravkick_factor(All.PM_Ti_begstep, All.Ti_Current) -
      get_gravkick_factor(All.PM_Ti_begstep, (All.PM_Ti_begstep + All.PM_Ti_endstep) / 2);
  else
    dt_gravkick_pm = (All.Ti_Current - (All.PM_Ti_begstep + All.PM_Ti_endstep) / 2) * All.Timebase_interval;



  /* now we do the distortiontensor kick for mesh */
  for(j1 = 0; j1 < 3; j1++)
    for(j2 = 0; j2 < 3; j2++)
      {
	dv_distortion_tensorps[j1 + 3][j2] = 0.0;
	dv_distortion_tensorps[j1 + 3][j2 + 3] = 0.0;

	/* the 'acceleration' is given by the product of tidaltensor and distortiontensor */
	for(j = 0; j < 3; j++)
	  {
	    dv_distortion_tensorps[j1 + 3][j2] +=
	      P[pindex].tidal_tensorpsPM[j1][j] * P[pindex].distortion_tensorps[j][j2];
	    dv_distortion_tensorps[j1 + 3][j2 + 3] +=
	      P[pindex].tidal_tensorpsPM[j1][j] * P[pindex].distortion_tensorps[j][j2 + 3];
	  }
	dv_distortion_tensorps[j1 + 3][j2] *= dt_gravkick;
	dv_distortion_tensorps[j1 + 3][j2 + 3] *= dt_gravkick;
	/* add it to the distortiontensor 'velocities' */
	half_kick_add[j1 + 3][j2] += dv_distortion_tensorps[j1 + 3][j2];
	half_kick_add[j1 + 3][j2 + 3] += dv_distortion_tensorps[j1 + 3][j2 + 3];
      }
#endif
}

#ifdef COMOVING_DISTORTION
 /* comoving displacement to physical displacement at 'time' a */
MyDouble **comoving_to_physical_a;

 /* physical displacement to comoving displacement at 'time' a0 */
MyDouble **physical_to_comoving_a0;

/* init comoving <-> physical transformation tensors */
void init_transformation_tensors(double a0)
{
  int i, j;
  MyDouble a = All.Time;
  MyDouble H_a = hubble_function(All.Time);
  MyDouble H_a0 = hubble_function(a0);

  comoving_to_physical_a = matrix(1, 6, 1, 6);
  physical_to_comoving_a0 = matrix(1, 6, 1, 6);

  for(i = 1; i <= 3; i++)
    for(j = 1; j <= 3; j++)
      {
	if(i == j)
	  {
	    comoving_to_physical_a[i][j] = a;
	    comoving_to_physical_a[i][j + 3] = 0;
	    comoving_to_physical_a[i + 3][j] = H_a * a;
	    comoving_to_physical_a[i + 3][j + 3] = 1 / a;
	    physical_to_comoving_a0[i][j] = 1.0 / a0;
	    physical_to_comoving_a0[i][j + 3] = 0;
	    physical_to_comoving_a0[i + 3][j] = -H_a0 * a0;
	    physical_to_comoving_a0[i + 3][j + 3] = a0;

	  }
	else
	  {
	    comoving_to_physical_a[i][j] = 0;
	    comoving_to_physical_a[i][j + 3] = 0;
	    comoving_to_physical_a[i + 3][j] = 0;
	    comoving_to_physical_a[i + 3][j + 3] = 0;
	    physical_to_comoving_a0[i][j] = 0;
	    physical_to_comoving_a0[i][j + 3] = 0;
	    physical_to_comoving_a0[i + 3][j] = 0;
	    physical_to_comoving_a0[i + 3][j + 3] = 0;

	  }
      }
}

/* free comoving <-> physical transformation tensors */
void free_transformation_tensors(void)
{
  free_matrix(physical_to_comoving_a0, 1, 6, 1, 6);
  free_matrix(comoving_to_physical_a, 1, 6, 1, 6);
}

/* transformation from comoving to physical 6D distortiontensor */
void comoving_to_physical_distortion(MyDouble ** comoving_distortiontensorpos, double a0)
{
  int i, j;
  MyDouble **temp1 = matrix(1, 6, 1, 6);
  MyDouble **temp2 = matrix(1, 6, 1, 6);

  init_transformation_tensors(a0);
  mult_matrix(comoving_distortiontensorpos, physical_to_comoving_a0, 6, temp1);
  mult_matrix(comoving_to_physical_a, temp1, 6, temp2);
  free_transformation_tensors();

  for(i = 1; i <= 6; i++)
    for(j = 1; j <= 6; j++)
      comoving_distortiontensorpos[i][j] = temp2[i][j];

  free_matrix(temp2, 1, 6, 1, 6);
  free_matrix(temp1, 1, 6, 1, 6);
}

/* physical accelaration of particle */
MyDouble get_physical_accel(MyIDType pindex, MyDouble a)
{
  MyDouble accel[3];
  int k;

  for(k = 0; k < 3; k++)
    accel[k] = 1.0 / (a * a) * P[pindex].g.GravAccel[k];

#ifdef PMGRID
  for(k = 0; k < 3; k++)
    accel[k] += 1.0 / (a * a) * P[pindex].GravPM[k];
#endif

  return sqrt(accel[0] * accel[0] + accel[1] * accel[1] + accel[2] * accel[2]);
}
#endif /* COMOVING_DISTORTION */

/* 
 This function is only called when a snapshot is written. It calculates the physical stream density determinant
 (taking into account the initial stream density) and the determinant of the full 6x6 phase-space
 distortion tensor. Due to Liouville phase-space volume conservation this should always be one.
*/

void get_current_ps_info(MyIDType pindex, MyDouble * flde, MyDouble * psde)
{
  /* for half kick correction of phase-space distortion tensor */
  MyDouble half_kick_add[6][6];

  /* allocate needed tensors */
  MyDouble **distortion_tensorps = matrix(1, 6, 1, 6);	/* phase space distortion tensor */
  MyDouble **V_q = matrix(1, 3, 1, 3);	/* V_q */

  /* (q,p) <-> (x,v) tensors */
  MyDouble **D_xq = matrix(1, 3, 1, 3);	/* D_xq */
  MyDouble **D_xp = matrix(1, 3, 1, 3);	/* D_xp */

  /* (Q,P) <-> (x,v) tensors */
  MyDouble **D_xQ = matrix(1, 3, 1, 3);	/* D_xQ  */

  MyDouble perm;
  int index[6], index2[3];
  int i, j;


  /* sync phase-space distortion tensor */
  get_half_kick_distortion(pindex, half_kick_add);

  /* copy distortion tensor in temporary memory */
  for(i = 1; i <= 6; i++)
    for(j = 1; j <= 6; j++)
      distortion_tensorps[i][j] = P[pindex].distortion_tensorps[i - 1][j - 1] + half_kick_add[i - 1][j - 1];

  /* phase space density calculation */
  ludcmp(distortion_tensorps, 6, index, &perm);

  *psde = perm * distortion_tensorps[1][1] * distortion_tensorps[2][2] * distortion_tensorps[3][3] *
    distortion_tensorps[4][4] * distortion_tensorps[5][5] * distortion_tensorps[6][6];

  /* copy distortion tensor in temporary memory */
  for(i = 1; i <= 6; i++)
    for(j = 1; j <= 6; j++)
      distortion_tensorps[i][j] = P[pindex].distortion_tensorps[i - 1][j - 1] + half_kick_add[i - 1][j - 1];

  /* direct (q,p) <-> (x,v) tensors */
  for(i = 1; i <= 3; i++)
    for(j = 1; j <= 3; j++)
      {
	/* D_xq */
	D_xq[i][j] = distortion_tensorps[i][j];
	/* D_xp */
	D_xp[i][j] = distortion_tensorps[i][j + 3];
	/* +V_q */
	V_q[i][j] = +P[pindex].V_matrix[i - 1][j - 1];
      }

  /* D_xQ = D_xq  +  D_xp * V_q */
  mult_matrix(D_xp, V_q, 3, D_xQ);
  for(i = 1; i <= 3; i++)
    for(j = 1; j <= 3; j++)
      D_xQ[i][j] += D_xq[i][j];

  /* configuration space density calculation */
  ludcmp(D_xQ, 3, index2, &perm);

  *flde = perm * D_xQ[1][1] * D_xQ[2][2] * D_xQ[3][3];

  /* take correct initial local density and rescale determinant value with it */
  *flde *= 1.0 / P[pindex].init_density;


  /* free temp tensors */
  free_matrix(D_xQ, 1, 3, 1, 3);
  free_matrix(D_xp, 1, 3, 1, 3);
  free_matrix(D_xq, 1, 3, 1, 3);
  free_matrix(V_q, 1, 3, 1, 3);
  free_matrix(distortion_tensorps, 1, 6, 1, 6);

}





void analyse_phase_space(MyIDType pindex,
			 MyDouble * s_1, MyDouble * s_2, MyDouble * s_3,
			 MyDouble * smear_x, MyDouble * smear_y, MyDouble * smear_z,
			 MyDouble * D_xP_proj, MyDouble * second_deriv, MyDouble * sigma)
{
  /* for half kick correction of phase-space distortion tensor */
  MyDouble half_kick_add[6][6];

  MyDouble **distortion_tensorps = matrix(1, 6, 1, 6);	/* phase space distortion tensor */
  MyDouble **inverse_distortion_tensorps = matrix(1, 6, 1, 6);	/* inverse of phase space distortion tensor */
  MyDouble **V_q = matrix(1, 3, 1, 3);	/* V_q */

  /* (q,p) <-> (x,v) tensors */
  /* direct 3x3 sub tensors */
  MyDouble **D_xp = matrix(1, 3, 1, 3);	/* D_xp */
  MyDouble **D_vq = matrix(1, 3, 1, 3);	/* D_vq */
  MyDouble **D_vp = matrix(1, 3, 1, 3);	/* D_vp */

  /* inverse 3x3 sub tensors */
  MyDouble **D_qv = matrix(1, 3, 1, 3);	/* D_qv */
  MyDouble **D_pv = matrix(1, 3, 1, 3);	/* D_pv */

  /* (Q,P) <-> (x,v) tensors */
  MyDouble **D_Pv = matrix(1, 3, 1, 3);	/* D_Pv */
  MyDouble **D_vQ = matrix(1, 3, 1, 3);	/* D_vQ */

  /* symmetric stretch and velocity shift tensors */
  MyDouble **S = matrix(1, 3, 1, 3);	/* S */
  MyDouble **V = matrix(1, 3, 1, 3);	/* V */

  /* transformation matrix to principal axis frame of S (caustic frame) = (eigenvector_1 | eigenvector_2 | eigenvector_3) */
  MyDouble **T = matrix(1, 3, 1, 3);	/* T */

  /* other tensors */
  MyDouble **eivecs = matrix(1, 3, 1, 3);
  MyDouble *S_eigenvals = vector(1, 3);
  MyDouble *V_eigenvals = vector(1, 3);
  MyDouble **temp = matrix(1, 3, 1, 3);

  int nrot;
  int i, j;

  /************************/
  /* START TENSOR ALGEBRA */
  /************************/

  /* sync phase-space distortion tensor */
  get_half_kick_distortion(pindex, half_kick_add);

  /* copy distortion tensor in temporary memory */
  for(i = 1; i <= 6; i++)
    for(j = 1; j <= 6; j++)
      distortion_tensorps[i][j] = P[pindex].distortion_tensorps[i - 1][j - 1] + half_kick_add[i - 1][j - 1];


#ifdef COMOVING_DISTORTION
  /* transform comoving distortion to physical */
  comoving_to_physical_distortion(distortion_tensorps, P[pindex].a0);
#endif

  /* direct (q,p) <-> (x,v) tensors */
  for(i = 1; i <= 3; i++)
    for(j = 1; j <= 3; j++)
      {
	/* D_xp */
	D_xp[i][j] = distortion_tensorps[i][j + 3];
	/* D_vq */
	D_vq[i][j] = distortion_tensorps[i + 3][j];
	/* D_vp */
	D_vp[i][j] = distortion_tensorps[i + 3][j + 3];
	/* +V_q */
	V_q[i][j] = +P[pindex].V_matrix[i - 1][j - 1];
      }

#ifdef COMOVING_DISTORTION
  /* sheet orientation V_matrix comoving -> transform to physical */
  for(i = 1; i <= 3; i++)
    for(j = 1; j <= 3; j++)
      {
	/* remove scalefactor multiplication */
	V_q[i][j] /= P[pindex].a0 * P[pindex].a0;
	/* add again Hubble flow */
	if(i == j)
	  V_q[i][j] += hubble_function(P[pindex].a0);
      }
#endif

  /* D_vQ = D_vq  +  D_vp * V_q */
  mult_matrix(D_vp, V_q, 3, D_vQ);
  for(i = 1; i <= 3; i++)
    for(j = 1; j <= 3; j++)
      D_vQ[i][j] += D_vq[i][j];


  /* get inverse distortion tensor */
  luinvert(distortion_tensorps, 6, inverse_distortion_tensorps);

  /* inverse (q,p) <-> (x,v) tensors */
  for(i = 1; i <= 3; i++)
    for(j = 1; j <= 3; j++)
      {
	/* D_qv */
	D_qv[i][j] = inverse_distortion_tensorps[i][j + 3];
	/* D_pv */
	D_pv[i][j] = inverse_distortion_tensorps[i + 3][j + 3];
	/* -V_q */
	V_q[i][j] = -V_q[i][j];
      }

  /* D_Pv = D_pv  -  V_q * D_qv */
  mult_matrix(V_q, D_qv, 3, D_Pv);
  for(i = 1; i <= 3; i++)
    for(j = 1; j <= 3; j++)
      D_Pv[i][j] += D_pv[i][j];

  /* S TENSOR ALGEBRA */
  /* S = D_Pv^t * D_Pv */
  mult_matrix_transpose_A(D_Pv, D_Pv, 3, S);

  /* S eigensystem */
  jacobi(S, 3, S_eigenvals, eivecs, &nrot, pindex);

  /* sort eigenvalues/eigenvectors in descending order of eigenvalues --> s_3 is always caustic eigenvalue (smallest one) */
  eigsrt(S_eigenvals, eivecs, 3);

  /* stretch factors */
  *s_1 = sqrt(S_eigenvals[1]);
  *s_2 = sqrt(S_eigenvals[2]);
  *s_3 = sqrt(S_eigenvals[3]);

  /* T = (v1|v2|v3) */
  for(i = 1; i <= 3; i++)
    for(j = 1; j <= 3; j++)
      T[i][j] = eivecs[i][j];

  /* V TENSOR ALGEBRA */
  /* V = D_xp^t * D_xp */
  mult_matrix_transpose_A(D_xp, D_xp, 3, V);

  /* V eigensystem */
  jacobi(V, 3, V_eigenvals, eivecs, &nrot, pindex);

  /* sort eigenvalues/eigenvectors in descending order of eigenvalues --> v_3 is always caustic eigenvalue */
  eigsrt(V_eigenvals, eivecs, 3);

  /* shift factors */
  *smear_x = sqrt(V_eigenvals[1]);
  *smear_y = sqrt(V_eigenvals[2]);
  *smear_z = sqrt(V_eigenvals[3]);

#ifdef OUTPUT_LAST_CAUSTIC
  /* write the eigenvectors */
  for(i = 1; i <= 3; i++)
    {
      P[pindex].lc_Dir_x[i - 1] = eivecs[i][1];
      P[pindex].lc_Dir_y[i - 1] = eivecs[i][2];
      P[pindex].lc_Dir_z[i - 1] = eivecs[i][3];
    }
#endif



  /************************************/
  /* START CUTOFF DENSITY CALCULATION */
  /************************************/

  /* D_xP is only used for 1D simulations to compare to alpha_k in Mohayaee and Shandarin (2006) */
  *D_xP_proj = 0;
  /* All.DM_velocity_dispersion is in cm/s => bring it to Gadget units */
  MyDouble vel_dispersion = All.DM_velocity_dispersion / All.UnitVelocity_in_cm_per_s;

#ifdef COMOVING_DISTORTION
  /* get velocity dispersion at a0: sigma(a0) = sigma(a=1)/a */
  vel_dispersion /= P[pindex].a0;
#endif

#ifdef REINIT_AT_TURNAROUND
  /* 
     REINIT_AT_TURNAROUND 
     => GDE calculation starts when particle turns around
     => Lagrangian frame=turnaround coordinates
     => intitial stream density & sheet orientation & velocity dispersion at turnaround
     => stream density & sheet orientation already done in predict.c (drift operation)
     => velocity dispersion needs to be tone
     -> COMOVING: sigma(t_ta)=prefactor * sigma(today)/a(t_ta) [see above + prefactor here] 
     => PHYSICAL: rho(t_ta) / sigma^3(t_ta) = rho(today) / \sigma(today) [
   */
#ifdef COMOVING_DISTORTION
  /* turnaround dispersion != homogeneous dispersion => prefactor */
  vel_dispersion *= pow(9.0 * M_PI * M_PI / (16.0 * (3.0 * All.SIM_epsilon + 1.0)), 1.0 / 3.0);
#else
  /* phase-space density relation */
  vel_dispersion *=
    pow(P[pindex].init_density / (1.0 / (6.0 * M_PI * All.G * All.TimeMax * All.TimeMax)), 1.0 / 3.0);
#endif
#endif


#ifdef SHELL_CODE
  int alpha, beta;

  /* set D_xP_proj in case of 1D simulation -> calculate radial projection */
  for(alpha = 0; alpha <= 2; alpha++)
    for(beta = 0; beta <= 2; beta++)
      *D_xP_proj +=
	P[pindex].Pos[alpha] * P[pindex].distortion_tensorps[alpha][beta + 3] * P[pindex].Pos[beta];
  *D_xP_proj /= P[pindex].Pos[0] * P[pindex].Pos[0] + P[pindex].Pos[1] * P[pindex].Pos[1] +
    P[pindex].Pos[2] * P[pindex].Pos[2];
#endif

  /* particle acceleration */
  MyDouble accel =
    sqrt(P[pindex].g.GravAccel[0] * P[pindex].g.GravAccel[0] +
	 P[pindex].g.GravAccel[1] * P[pindex].g.GravAccel[1] +
	 P[pindex].g.GravAccel[2] * P[pindex].g.GravAccel[2]);


#ifdef COMOVING_DISTORTION
  accel = get_physical_accel(pindex, All.Time);
#endif

  /* White & Vogelsberger (2008) second derivative estimate based on Galilean-invariant quantities */
  /* 1. way to estimate */
  MyDouble D_Px_sum = 0.0;

  for(i = 1; i <= 3; i++)
    for(j = 1; j <= 3; j++)
      D_Px_sum += D_vQ[i][j] * D_vQ[i][j];

  D_Px_sum = sqrt(D_Px_sum);

  *second_deriv = D_Px_sum / accel;

  /* 2. way to estimate */
  /*
     MyDouble D_xP_sum = 0.0;  
     for(i = 1; i <= 3; i++)
     for(j = 1; j <= 3; j++)
     D_xP_sum += D_xp[i][j]*D_xp[i][j];

     D_xP_sum = sqrt(D_xP_sum);
     *second_deriv = 1.0/(D_xP_sum*accel);
   */

  /* calculate the extent of the caustic based on its smearing in all directions (GADGET length units) */
  *smear_x *= vel_dispersion;
  *smear_y *= vel_dispersion;
  *smear_z *= vel_dispersion;


#ifdef OUTPUT_LAST_CAUSTIC
  P[pindex].lc_smear_x = *smear_x;
  P[pindex].lc_smear_y = *smear_y;
  P[pindex].lc_smear_z = *smear_z;
#endif

  /* free all tensors */
  free_matrix(temp, 1, 3, 1, 3);
  free_vector(V_eigenvals, 1, 3);
  free_vector(S_eigenvals, 1, 3);
  free_matrix(eivecs, 1, 3, 1, 3);
  free_matrix(T, 1, 3, 1, 3);
  free_matrix(V, 1, 3, 1, 3);
  free_matrix(S, 1, 3, 1, 3);
  free_matrix(D_vQ, 1, 3, 1, 3);
  free_matrix(D_Pv, 1, 3, 1, 3);
  free_matrix(D_pv, 1, 3, 1, 3);
  free_matrix(D_qv, 1, 3, 1, 3);
  free_matrix(D_vp, 1, 3, 1, 3);
  free_matrix(D_vq, 1, 3, 1, 3);
  free_matrix(D_xp, 1, 3, 1, 3);
  free_matrix(V_q, 1, 3, 1, 3);
  free_matrix(inverse_distortion_tensorps, 1, 6, 1, 6);
  free_matrix(distortion_tensorps, 1, 6, 1, 6);

  *sigma = vel_dispersion;
}


MyDouble get_analytic_annihilation(MyDouble s_1, MyDouble s_2, MyDouble s_3,
				   MyDouble s_1_prime, MyDouble s_2_prime, MyDouble s_3_prime,
				   MyDouble second_deriv, MyDouble second_deriv_prime,
				   MyDouble dt, MyDouble sigma)
{
  /* 2^(9/2) Gamma(5/4)^2 / pi */
  MyDouble GammaFac = 5.917350238;

  MyDouble s_1_tilde = (s_1 + s_1_prime) / 2.0;
  MyDouble s_2_tilde = (s_2 + s_2_prime) / 2.0;
  MyDouble second_deriv_tilde = (second_deriv + second_deriv_prime) / 2.0;
  MyDouble s_3_diff = fabs(s_3 + s_3_prime);
  MyDouble Delta_P =
    dt / (s_1_tilde * s_2_tilde * s_3_diff) * log(GammaFac * s_3 * s_3_prime / (sigma * second_deriv_tilde));

  return Delta_P;
}

MyDouble get_max_caustic_density(MyDouble s_1, MyDouble s_2,
				 MyDouble s_1_prime, MyDouble s_2_prime,
				 MyDouble second_deriv, MyDouble second_deriv_prime,
				 MyDouble sigma, MyIDType pindex)
{
  /* 2^(5/4) Gamma(5/4) / sqrt(pi) */
  MyDouble GammaFac = 1.216280214;

  MyDouble s_1_tilde = (s_1 + s_1_prime) / 2.0;
  MyDouble s_2_tilde = (s_2 + s_2_prime) / 2.0;
  MyDouble second_deriv_tilde = (second_deriv + second_deriv_prime) / 2.0;
  MyDouble rho0_max =
    GammaFac * 1.0 / (s_1_tilde * s_2_tilde) * 1.0 / sqrt(second_deriv_tilde) * 1.0 / sqrt(sigma);


#ifdef COMOVING_DISTORTION
  /* return comoving normed maximum density */
  return pow(All.Time / P[pindex].a0, 3.0) * rho0_max;
#else
  /* return physical normed maximum density */
  return rho0_max;
#endif
}

#endif /* DISTORTIONTENSORPS */
