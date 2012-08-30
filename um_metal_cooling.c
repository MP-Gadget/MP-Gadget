
/*=================== Cooling from metals (C+,Si+,O,Fe+) ===================*/

/*------------------------ [H/e-impact excitations] ------------------------*/

/*==========================================================================*/

/*  For the atomic data, see: Moore 1952 - Osterbrock 1989 -
    Hollenbach & McKee (HM) 1989 - Quinet et al. 1996 - Santoro & Shull 2005 */


#if defined (UM_METAL_COOLING) && defined (LT_METAL_COOLING)

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <gsl/gsl_vector.h>
#include <gsl/gsl_linalg.h>

#include "allvars.h"


static void sq_system_solver(double *, double *, int, double *);

static void set_ndens_and_metallicity(float, int);

static double CII_cooling(double);
static double SiII_cooling(double);
static double OI_cooling(double);
static double OI_cooling_2level_approx(double);
static double FeII_cooling(double);
static double FeII_cooling_2level_approx(double);


/* abundances: */
static double H_ndens, He_ndens;
static double C_ndens, N_ndens, O_ndens, Mg_ndens, Si_ndens, Fe_ndens;
static double CII_ndens, OI_ndens, SiII_ndens, FeII_ndens;
static double dummy_average_atomic_mass = 40.;

static double tiny_value = 1.e-10;

/* interacting particles: */

#ifdef UM_H_MET_IMPACTS
static double n_field_H;
#endif

#ifdef UM_e_MET_IMPACTS
static double n_field_e;
#endif

/*------------ this routine solves n*n square linear systems ------------*/

void sq_system_solver(double *a_data, double *b_data, int n, double sol[])
/*
  a_data, b_data and n are input, sol is autput.
  n         refers to the dimensions of a_data and b_data;
  a_data    is an n*n matrix with indeces running between [0,n-1];
  b_data    is an n-dim vector with index running between [0,n-1];
  sol       is the solution: n-dim vector with index running between [0,n-1];
*/
{
  gsl_matrix_view m = gsl_matrix_view_array(a_data, n, n);

  gsl_vector_view b = gsl_vector_view_array(b_data, n);

  gsl_vector *x = gsl_vector_alloc(n);

  int s;
  gsl_permutation *p = gsl_permutation_alloc(n);

  gsl_linalg_LU_decomp(&m.matrix, p, &s);

  gsl_linalg_LU_solve(&m.matrix, p, &b.vector, x);

  //  printf("\nx =\n");
  //  gsl_vector_fprintf(stdout, x, " %g");

  int i = 0;

  for(i = 0; i < n; i++)
    sol[i] = gsl_vector_get(x, i);


  gsl_permutation_free(p);

  gsl_vector_free(x);

  return;
}

/*
  :: note to umberto : i've changed all double declarations in float declarations
*/

void set_ndens_and_metallicity(float rho, int n)
     /*
      * note: FillEl collects the mass of all the elements present in the yields table
      *       but not explicitly tracked by the stellarevolution.
      *       Elements explicitly tracked are specified in metals.dat.
      */
{
  int k;
  double sum_Zs;

  /* LT_ADD */
  //  double FillEl_ndens;
  double Hmass, invmass;

  for(sum_Zs = 0, k = 0; k < n; k++)
    sum_Zs += um_ZsPoint[k];

  invmass = 1.0 / (double) um_mass;
  Hmass = (double) um_mass - sum_Zs;

  /* note :: FillEl gives the position of the Filling elements in the metal array. This position
   * is the same as that in metals.dat (look for the tag "Ej") */

  /* FillEl_ndens = rho * um_ZsPoint[FillEl] * invmass / dummy_average_atomic_mass; (comment by UM)*/
  /* end of LT */

  /* compute ndens = rho * mass_fraction / mass_element */

  C_ndens = N_ndens = O_ndens = Mg_ndens = Si_ndens = Fe_ndens = 0.;

  H_ndens = rho * (Hmass * invmass) / (1. * PROTONMASS);

  He_ndens = rho * (um_ZsPoint[Hel] * invmass) / (4. * PROTONMASS);

  if(Carbon >= 0)
    C_ndens = rho * (um_ZsPoint[Carbon] * invmass) / (12. * PROTONMASS);

  if(Nitrogen >= 0)
    N_ndens = rho * (um_ZsPoint[Nitrogen] * invmass) / (14. * PROTONMASS);

  if(Oxygen >= 0)
    O_ndens = rho * (um_ZsPoint[Oxygen] * invmass) / (16. * PROTONMASS);

  if(Magnesium >= 0)
    Mg_ndens = rho * (um_ZsPoint[Magnesium] * invmass) / (24. * PROTONMASS);

  if(Silicon >= 0)
    Si_ndens = rho * (um_ZsPoint[Silicon] * invmass) / (28. * PROTONMASS);

  if(Iron >= 0)
    Fe_ndens = rho * (um_ZsPoint[Iron] * invmass) / (56. * PROTONMASS);

  /* Ansatz: C, Si, Fe ionized by the background radiation */

  CII_ndens = C_ndens;
  SiII_ndens = Si_ndens;
  FeII_ndens = Fe_ndens;
  OI_ndens = O_ndens;

  return;
}


/***********************************************************/
/*-----cooling from C+ : 2(J=3/2) -> 1(J=1/2)              */
/*    from Santoro&Shull 2005, Wolfire et al. 1995 + HM89  */
/***********************************************************/
double CII_cooling(double T_gas)
{
#if (defined (UM_H_MET_IMPACTS)) || (defined (UM_e_MET_IMPACTS))

  double E21 = 1.259e-14;
  double A21 = 2.4e-6;
  int g2 = 4;
  int g1 = 2;
  double x2 = 0;

#ifdef UM_H_MET_IMPACTS
  double gammaH21 = 8e-10 * pow(T_gas / 100, 0.07);
  double gammaH12 = gammaH21 * g2 / g1 * exp(-E21 / (BOLTZMANN * T_gas));
#endif
#ifdef UM_e_MET_IMPACTS
  double gammae21 = 2.8e-7 * pow(T_gas / 100, -0.5);	/* HM89 */
  double gammae12 = gammae21 * g2 / g1 * exp(-E21 / (BOLTZMANN * T_gas));
#endif


#if defined (UM_H_MET_IMPACTS) && !(defined (UM_e_MET_IMPACTS))
  x2 = gammaH12 / (gammaH21 + gammaH12 + A21 / n_field_H);
#endif
#if defined (UM_e_MET_IMPACTS) && !defined (UM_H_MET_IMPACTS)
  x2 = gammae12 / (gammae21 + gammae12 + A21 / n_field_e);
#endif
#if defined (UM_H_MET_IMPACTS) && defined (UM_e_MET_IMPACTS)
  x2 =
    (gammaH12 + gammae12 * n_field_e / n_field_H) / (gammaH21 + gammaH12 + A21 / n_field_H +
						     (gammae21 + gammae12) * n_field_e / n_field_H);
#endif

  return (A21 * E21 * x2 * CII_ndens);	//by def. erg/cm3/s

#else
  return 0;
#endif /* close #if/else UM_H_MET_IMPACTS || UM_e_MET_IMPACTS */
}


/***********************************************************/
/*-----cooling from Si+ : 2(J=3/2) -> 1(J=1/2)             */
/*                        from Santoro&Shull 2005 + HM89   */
/***********************************************************/
double SiII_cooling(double T_gas)
{
#if (defined (UM_H_MET_IMPACTS)) || (defined (UM_e_MET_IMPACTS))

  double E21 = 5.71e-14;
  double A21 = 2.1e-4;
  int g2 = 4;
  int g1 = 2;
  double x2 = 0;

#ifdef UM_H_MET_IMPACTS
  double gammaH21 = 8e-10 * pow(T_gas / 100, -0.07);
  double gammaH12 = gammaH21 * g2 / g1 * exp(-E21 / (BOLTZMANN * T_gas));
#endif
#ifdef UM_e_MET_IMPACTS
  double gammae21 = 1.7e-6 * pow(T_gas / 100, -0.5);
  double gammae12 = gammae21 * g2 / g1 * exp(-E21 / (BOLTZMANN * T_gas));
#endif


#if defined (UM_H_MET_IMPACTS) && !(defined (UM_e_MET_IMPACTS))
  x2 = gammaH12 / (gammaH21 + gammaH12 + A21 / n_field_H);
#endif
#if defined (UM_e_MET_IMPACTS) && !(defined (UM_H_MET_IMPACTS))
  x2 = gammae12 / (gammae21 + gammae12 + A21 / n_field_e);
#endif
#if defined (UM_H_MET_IMPACTS) && defined (UM_e_MET_IMPACTS)
  x2 =
    (gammaH12 + gammae12 * n_field_e / n_field_H) / (gammaH21 + gammaH12 + A21 / n_field_H +
						     (gammae21 + gammae12) * n_field_e / n_field_H);
#endif

  return (A21 * E21 * x2 * SiII_ndens);	//by def. erg/cm3/s

#else
  return 0;
#endif /* close #if/else  UM_H_MET_IMPACTS || UM_e_MET_IMPACTS */
}


/*********************************************************************/
/*---- cooling from OI :levels (1,2,3,4,5)=(3P2, 3P1, 3P0, 1D2, 1S0) */
/*                       data from Hollenbach&McKee 1989             */
/*********************************************************************/
double OI_cooling_2level_approx(double T_gas)
{
#if (defined (UM_H_MET_IMPACTS)) || (defined (UM_e_MET_IMPACTS))

  double E21 = 3.14e-14;
  double A21 = 8.9e-5; 	// A[i][j] from Osterbrock 1989
  int g1 = 5;
  int g2 = 3;
  double x2 = 0;

#ifdef UM_H_MET_IMPACTS
  double gammaH21 = 9.2e-11 * pow(T_gas / 100, 0.67);
  double gammaH12 = gammaH21 * g2 / g1 * exp(-E21 / (BOLTZMANN * T_gas));
#endif
#ifdef UM_e_MET_IMPACTS
  double gammae21 = 1.4e-8;  //~2.5e-10; (Bell '98): this is not crucial
  double gammae12 = gammae21 * g2 / g1 * exp(-E21 / (BOLTZMANN * T_gas));
#endif


#if defined (UM_H_MET_IMPACTS) && !(defined (UM_e_MET_IMPACTS))
  x2 = gammaH12 / (gammaH21 + gammaH12 + A21 / n_field_H);
#endif
#if defined (UM_e_MET_IMPACTS) && !(defined (UM_H_MET_IMPACTS))
  x2 = gammae12 / (gammae21 + gammae12 + A21 / n_field_e);
#endif
#if defined (UM_H_MET_IMPACTS) && defined (UM_e_MET_IMPACTS)
  x2 =
    (gammaH12 + gammae12 * n_field_e / n_field_H) / (gammaH21 + gammaH12 + A21 / n_field_H +
						     (gammae21 + gammae12) * n_field_e / n_field_H);
#endif

  return (A21 * E21 * x2 * OI_ndens);	//by def. erg/cm3/s

#else
  return 0;
#endif /* close #if/else  UM_H_MET_IMPACTS || UM_e_MET_IMPACTS */
}


double OI_cooling(double T_gas)
{
#if (defined (UM_H_MET_IMPACTS)) || (defined (UM_e_MET_IMPACTS))

  int i, j;
  int n = 5;			// number of levels
  int g[n];			// level degenarations
  double E[n][n];		// energy transitions
  double A[n][n];		// Einstein coefficients

#ifdef UM_H_MET_IMPACTS
  double gammaH[n][n];		// rates
#endif
#ifdef UM_e_MET_IMPACTS
  double gammae[n][n];		// rates
#endif
#if defined (UM_H_MET_IMPACTS) && defined(UM_e_MET_IMPACTS)
  double n_ratio = n_field_e / n_field_H;
#endif

  double mtrx[n][n];		// population matrix with index 1-5
  double sum_column[n];		// sum in column in population matrix
  double b[n];			// known term vector 
  double x_pop[n];		// fractional population of the levels


  /* set   b=( 1 0 0 0 0 )  */
  b[1] = 1.;
  for(i = 2; i <= n; i++)
    b[i] = 0.;

  for(i = 1; i <= n; i++)
    {
      for(j = 1; j <= n; j++)	/* initialization: */
	{
	  E[i][j] = 0.;
	  A[i][j] = 0.;
#ifdef UM_H_MET_IMPACTS
	  gammaH[i][j] = 0.;
#endif
#ifdef UM_e_MET_IMPACTS
	  gammae[i][j] = 0.;
#endif
	  mtrx[i][j] = 0.;
	}
      g[i] = 0;
      sum_column[i] = 0;
    }

  /* atomic data... */

  E[2][1] = 3.14e-14;
  E[3][2] = 1.29e-14;
  E[4][3] = 3.143e-12;
  E[5][3] = 3.56e-12;
  E[3][1] = E[3][2] + E[2][1];
  E[4][2] = E[4][3] + E[3][2];
  E[4][1] = E[4][3] + E[3][1];
  E[5][2] = E[5][3] + E[3][2];
  E[5][1] = E[5][2] + E[2][1];
  E[5][4] = E[5][3] - E[4][3];
  A[2][1] = 8.9e-5;		// A[i][j] from Osterbrock 1989
  A[3][1] = 1.0e-10;
  A[3][2] = 1.7e-5;
  A[4][1] = 6.3e-3;
  A[4][2] = 2.1e-3;
  A[4][3] = 7.3e-7;
  A[5][1] = 2.9e-4;
  A[5][2] = 7.3e-2;
  A[5][4] = 1.2;
  g[1] = 5;			//    3P2, J = 2 (ground level)
  g[2] = 3;			//    3P1, J = 1
  g[3] = 1;			//    3P0, J = 0
  g[4] = 5;			//    1D2, J = 2
  g[5] = 1;			//    1S0, J = 0


#ifdef UM_H_MET_IMPACTS
  gammaH[2][1] = 9.2e-11 * pow(T_gas / 100, 0.67);
  gammaH[3][1] = 4.3e-11 * pow(T_gas / 100, 0.80);
  gammaH[3][2] = 1.1e-10 * pow(T_gas / 100, 0.44);
  gammaH[4][1] = gammaH[4][2] = gammaH[4][3] = 1.e-12;	// less well determined
  gammaH[5][1] = gammaH[5][2] = gammaH[5][3] = 1.e-12;	//assumed = 10^-12 cm3/s
  gammaH[1][2] = gammaH[2][1] * g[2] / g[1] * exp(-E[2][1] / (BOLTZMANN * T_gas));
  gammaH[1][3] = gammaH[3][1] * g[3] / g[1] * exp(-E[3][1] / (BOLTZMANN * T_gas));
  gammaH[2][3] = gammaH[3][2] * g[3] / g[2] * exp(-E[3][2] / (BOLTZMANN * T_gas));
  gammaH[1][4] = gammaH[4][1] * g[4] / g[1] * exp(-E[4][1] / (BOLTZMANN * T_gas));
  gammaH[2][4] = gammaH[4][2] * g[4] / g[2] * exp(-E[4][2] / (BOLTZMANN * T_gas));
  gammaH[3][4] = gammaH[4][3] * g[4] / g[3] * exp(-E[4][3] / (BOLTZMANN * T_gas));
  gammaH[1][5] = gammaH[5][1] * g[5] / g[1] * exp(-E[5][1] / (BOLTZMANN * T_gas));
  gammaH[2][5] = gammaH[5][2] * g[5] / g[2] * exp(-E[5][2] / (BOLTZMANN * T_gas));
  gammaH[3][5] = gammaH[5][3] * g[5] / g[3] * exp(-E[5][3] / (BOLTZMANN * T_gas));
#endif

#ifdef UM_e_MET_IMPACTS
  gammae[2][1] = 1.4e-8;;//~2.5e-10; (Bell '98): to change this is not crucial
  gammae[3][1] = 1.4e-8; //~2.6e-10; (Bell '98): to change this is not crucial
  gammae[3][2] = 5.0e-9; //~1.0e-11; (Bell '98): to change this is not crucial
  gammae[4][1] = gammae[4][2] = gammae[4][3] = 1.e-10;	// less well determined
  gammae[5][1] = gammae[5][2] = gammae[5][3] = 1.e-10;	// assumed ~1.e-10!!!!
  gammae[1][2] = gammae[2][1] * g[2] / g[1] * exp(-E[2][1] / (BOLTZMANN * T_gas));
  gammae[1][3] = gammae[3][1] * g[3] / g[1] * exp(-E[3][1] / (BOLTZMANN * T_gas));
  gammae[2][3] = gammae[3][2] * g[3] / g[2] * exp(-E[3][2] / (BOLTZMANN * T_gas));
  gammae[1][4] = gammae[4][1] * g[4] / g[1] * exp(-E[4][1] / (BOLTZMANN * T_gas));
  gammae[2][4] = gammae[4][2] * g[4] / g[2] * exp(-E[4][2] / (BOLTZMANN * T_gas));
  gammae[3][4] = gammae[4][3] * g[4] / g[3] * exp(-E[4][3] / (BOLTZMANN * T_gas));
  gammae[1][5] = gammae[5][1] * g[5] / g[1] * exp(-E[5][1] / (BOLTZMANN * T_gas));
  gammae[2][5] = gammae[5][2] * g[5] / g[2] * exp(-E[5][2] / (BOLTZMANN * T_gas));
  gammae[3][5] = gammae[5][3] * g[5] / g[3] * exp(-E[5][3] / (BOLTZMANN * T_gas));
#endif


  /* build the matrix for the system to solve */

  i = 1;
  j = 1;
  for(i = 1; i <= n; i++)
    {
      for(j = 1; j <= n; j++)
	{
#if defined (UM_H_MET_IMPACTS) && !(defined(UM_e_MET_IMPACTS))
	  mtrx[i][j] = gammaH[j][i] + A[j][i] / n_field_H;
#endif
#if defined (UM_e_MET_IMPACTS) && !(defined(UM_H_MET_IMPACTS))
	  mtrx[i][j] = gammae[j][i] + A[j][i] / n_field_e;
#endif
#if defined (UM_e_MET_IMPACTS) && defined(UM_H_MET_IMPACTS)
	  mtrx[i][j] = gammaH[j][i] + A[j][i] / n_field_H + gammae[j][i] * n_ratio;
#endif
	}

      printf("1 - |  %g  %g  %g  %g  %g  |\n",mtrx[i][1], mtrx[i][2], mtrx[i][3], mtrx[i][4], mtrx[i][5] );
    }


  //now, update the diagonal elements(= 0 so far) to -(sum in column)!
  i = 1;
  j = 1;
  for(j = 1; j <= n; j++)
    {
      for(i = 1; i <= n; i++)
	sum_column[j] += mtrx[i][j];
    }

  i = 1;
  for(i = 1; i <= n; i++)
    mtrx[i][i] = -sum_column[i];
  //...done!




  printf("2 - |  %g  %g  %g  %g  %g  |\n",mtrx[1][1], mtrx[1][2], mtrx[1][3], mtrx[1][4], mtrx[1][5] );
  printf("2 - |  %g  %g  %g  %g  %g  |\n",mtrx[2][1], mtrx[2][2], mtrx[2][3], mtrx[2][4], mtrx[2][5] );
  printf("2 - |  %g  %g  %g  %g  %g  |\n",mtrx[3][1], mtrx[3][2], mtrx[3][3], mtrx[3][4], mtrx[3][5] );
  printf("2 - |  %g  %g  %g  %g  %g  |\n",mtrx[4][1], mtrx[4][2], mtrx[4][3], mtrx[4][4], mtrx[4][5] );
  printf("2 - |  %g  %g  %g  %g  %g  |\n",mtrx[5][1], mtrx[5][2], mtrx[5][3], mtrx[5][4], mtrx[5][5] );
  printf("\n");

  /*  now, set the condition "x1 + x2 + ... = 1",  
     replacing one (the first) line of mtrx[][] with: ( 1  1 ... 1  1) 
     because of the linear dependence: */
    mtrx[1][1] = 1.;
    mtrx[1][2] = 1.;
    mtrx[1][3] = 1.;
    mtrx[1][4] = 1.;
    mtrx[1][5] = 1.;

  printf("3 - |  %g  %g  %g  %g  %g  |\n",mtrx[1][1], mtrx[1][2], mtrx[1][3], mtrx[1][4], mtrx[1][5] );
  printf("3 - |  %g  %g  %g  %g  %g  |\n",mtrx[2][1], mtrx[2][2], mtrx[2][3], mtrx[2][4], mtrx[2][5] );
  printf("3 - |  %g  %g  %g  %g  %g  |\n",mtrx[3][1], mtrx[3][2], mtrx[3][3], mtrx[3][4], mtrx[3][5] );
  printf("3 - |  %g  %g  %g  %g  %g  |\n",mtrx[4][1], mtrx[4][2], mtrx[4][3], mtrx[4][4], mtrx[4][5] );
  printf("3 - |  %g  %g  %g  %g  %g  |\n",mtrx[5][1], mtrx[5][2], mtrx[5][3], mtrx[5][4], mtrx[5][5] );
  printf("\n");



  /* find level populations */

  double mtrx_pass[n][n];
  double b_pass[n];
  double x_got[n];

  // matrix/vectors with index 0-4
  /*
     the following 3 assignements are necessary to be consistent with the routine sq_system_solver() and the gsl libraries used there
   */
  i = 0;
  j = 0;
  for(i = 0; i < n; i++)
    {
      for(j = 0; j < n; j++)
	mtrx_pass[i][j] = mtrx[i + 1][j + 1];	//1. ok

      b_pass[i] = b[i + 1];   //2. ok
    }

  sq_system_solver((double *) mtrx_pass, b_pass, 5, x_got);	//..find the solution




  j = 0;
  for(j = 0; j < n; j++){
    x_pop[j + 1] = x_got[j];	//3.
    printf("   x_pop[%d] = %g\n", j+1, x_pop[j + 1]);
  }

  endrun(99999999);   /* I am here:  try runs with 2-level approx */


  /* compute the cooling fct */

  return (x_pop[2] * A[2][1] * E[2][1] + 
	  x_pop[3] * A[3][1] * E[3][1] + 
	  x_pop[3] * A[3][2] * E[3][2] +
	  /* Santoro&Shull2005 do not include the following: */
	  x_pop[4] * A[4][1] * E[4][1] + 
	  x_pop[4] * A[4][2] * E[4][2] + 
	  x_pop[4] * A[4][3] * E[4][3] + 
	  x_pop[5] * A[5][1] * E[5][1] + 
	  x_pop[5] * A[5][2] * E[5][2] + 
	  x_pop[5] * A[5][4] * E[5][4]) * OI_ndens;	//erg/cm3/s

#else
  return 0;
#endif /* close #if/else  UM_H_MET_IMPACTS || UM_e_MET_IMPACTS */
}


/***********************************************************************/
/*-----cooling from Fe+ :level 6D(1,2,3,4,5)=6D(J=9/2,7/2,5/2,3/2,1/2) */
/*                         data from Hollenbach&McKee 1989             */
/***********************************************************************/
double FeII_cooling_2level_approx(double T_gas)
{
#if (defined (UM_H_MET_IMPACTS)) || (defined (UM_e_MET_IMPACTS))

  double E21 = 7.64e-14;
  double A21 =  2.13e-3;
  int g1 = 10;
  int g2 = 8;
  double x2 = 0;

#ifdef UM_H_MET_IMPACTS
  double gammaH21 =  9.5e-10;
  double gammaH12 = gammaH21 * g2 / g1 * exp(-E21 / (BOLTZMANN * T_gas));
#endif
#ifdef UM_e_MET_IMPACTS
  double gammae21 = 1.8e-6 * pow(T_gas / 100, -0.5);
  double gammae12 = gammae21 * g2 / g1 * exp(-E21 / (BOLTZMANN * T_gas));
#endif


#if defined (UM_H_MET_IMPACTS) && !(defined (UM_e_MET_IMPACTS))
  x2 = gammaH12 / (gammaH21 + gammaH12 + A21 / n_field_H);
#endif
#if defined (UM_e_MET_IMPACTS) && !(defined (UM_H_MET_IMPACTS))
  x2 = gammae12 / (gammae21 + gammae12 + A21 / n_field_e);
#endif
#if defined (UM_H_MET_IMPACTS) && defined (UM_e_MET_IMPACTS)
  x2 =
    (gammaH12 + gammae12 * n_field_e / n_field_H) / (gammaH21 + gammaH12 + A21 / n_field_H +
						     (gammae21 + gammae12) * n_field_e / n_field_H);
#endif

  return (A21 * E21 * x2 * OI_ndens);	//by def. erg/cm3/s

#else
  return 0;
#endif /* close #if/else  UM_H_MET_IMPACTS || UM_e_MET_IMPACTS */
}


double FeII_cooling(double T_gas)
{
#if (defined (UM_H_MET_IMPACTS)) || (defined (UM_e_MET_IMPACTS))

  int i, j;
  int n = 5;			// number of levels
  int g[n];			// level degenarations
  double E[n][n];		// energy transitions
  double A[n][n];		// Einstein coefficients

#ifdef UM_H_MET_IMPACTS
  double gammaH[n][n];		// rates
#endif
#ifdef UM_e_MET_IMPACTS
  double gammae[n][n];		// rates
#endif
#if defined (UM_H_MET_IMPACTS) && defined(UM_e_MET_IMPACTS)
  double n_ratio = n_field_e / n_field_H;
#endif

  double mtrx[n][n];		// population matrix with index 1-5 (consistent withthe data in Santoro&Shull 2005)
  double sum_column[n];		// sum in column in population matrix
  double b[n];			// known term vector 
  double x_pop[n];		// fractional population of the levels

  /* set   b=( 1 0 0 0 0 )  */
  b[1] = 1.;
  for(i = 2; i <= n; i++)
    b[i] = 0.;

  for(i = 1; i <= n; i++)
    {
      for(j = 1; j <= n; j++)
	{
	  E[i][j] = 0.;
	  A[i][j] = 0.;
#ifdef UM_H_MET_IMPACTS
	  gammaH[i][j] = 0.;
#endif
#ifdef UM_e_MET_IMPACTS
	  gammae[i][j] = 0.;
#endif
	  mtrx[i][j] = 0.;
	}
      g[i] = 0;
      sum_column[i] = 0;
    }

  /* atomic data... */

  E[2][1] = 7.64e-14;
  E[3][2] = 5.62e-14;
  E[4][3] = 3.87e-14;
  E[5][4] = 2.27e-14;
  E[3][1] = E[3][2] + E[2][1];
  E[4][1] = E[4][3] + E[3][1];
  E[5][1] = E[5][4] + E[4][1];
  E[4][2] = E[4][3] + E[3][2];
  E[5][2] = E[5][4] + E[4][2];
  E[5][3] = E[5][4] + E[4][3];
  A[2][1] = 2.13e-3;
  A[3][2] = 1.57e-3;
  A[3][1] = 1.50e-9;
  A[4][3] = 7.18e-4;
  A[5][4] = 1.88e-4;
  g[1] = 10;			//  J = 9/2    (ground level)
  g[2] = 8;			//  J = 7/2
  g[3] = 6;			//  J = 5/2
  g[4] = 4;			//  J = 3/2
  g[5] = 2;			//  J = 1/2

#ifdef UM_H_MET_IMPACTS
  gammaH[2][1] = 9.5e-10;
  gammaH[3][2] = 4.7e-10;
  gammaH[4][3] = 5.e-10;	/* unknown datum: fiducial value */
  gammaH[5][4] = 5.e-10;	/* unknown datum: fiducial value */
  gammaH[3][1] = 5.7e-10;
  gammaH[4][1] = 5.e-10;	/* unknown datum: fiducial value */
  gammaH[5][1] = 5.e-10;	/* unknown datum: fiducial value */
  gammaH[4][2] = 5.e-10;	/* unknown datum: fiducial value */
  gammaH[5][2] = 5.e-10;	/* unknown datum: fiducial value */
  gammaH[5][3] = 5.e-10;	/* unknown datum: fiducial value */
  gammaH[1][2] = gammaH[2][1] * g[2] / g[1] * exp(-E[2][1] / (BOLTZMANN * T_gas));
  gammaH[2][3] = gammaH[3][2] * g[3] / g[2] * exp(-E[3][2] / (BOLTZMANN * T_gas));
  gammaH[3][4] = gammaH[4][3] * g[4] / g[3] * exp(-E[4][3] / (BOLTZMANN * T_gas));
  gammaH[4][5] = gammaH[5][4] * g[5] / g[4] * exp(-E[5][4] / (BOLTZMANN * T_gas));
  gammaH[1][3] = gammaH[3][1] * g[3] / g[1] * exp(-E[3][1] / (BOLTZMANN * T_gas));
  gammaH[1][4] = gammaH[4][1] * g[4] / g[1] * exp(-E[4][1] / (BOLTZMANN * T_gas));
  gammaH[1][5] = gammaH[5][1] * g[5] / g[1] * exp(-E[5][1] / (BOLTZMANN * T_gas));
  gammaH[2][4] = gammaH[4][2] * g[4] / g[2] * exp(-E[4][2] / (BOLTZMANN * T_gas));
  gammaH[2][5] = gammaH[5][2] * g[5] / g[2] * exp(-E[5][2] / (BOLTZMANN * T_gas));
  gammaH[3][5] = gammaH[5][3] * g[5] / g[3] * exp(-E[5][3] / (BOLTZMANN * T_gas));
#endif


#ifdef UM_e_MET_IMPACTS
  gammae[2][1] = 1.8e-6 * pow(T_gas / 100, -0.5);
  gammae[3][2] = 8.7e-7 * pow(T_gas / 100, -0.5);
  gammae[4][3] = 1.e-5 * pow(T_gas, -0.5);	/* unknown datum: fiducial value */
  gammae[5][4] = 1.e-5 * pow(T_gas, -0.5);	/* unknown datum: fiducial value */
  gammae[3][1] = 1.8e-6 * pow(T_gas / 100, -0.5);
  gammae[4][1] = 1.e-5 * pow(T_gas, -0.5);	/* unknown datum: fiducial value */
  gammae[5][1] = 1.e-5 * pow(T_gas, -0.5);	/* unknown datum: fiducial value */
  gammae[4][2] = 1.e-5 * pow(T_gas, -0.5);	/* unknown datum: fiducial value */
  gammae[5][2] = 1.e-5 * pow(T_gas, -0.5);	/* unknown datum: fiducial value */
  gammae[5][3] = 1.e-5 * pow(T_gas, -0.5);	/* unknown datum: fiducial value */
  gammae[1][2] = gammae[2][1] * g[2] / g[1] * exp(-E[2][1] / (BOLTZMANN * T_gas));
  gammae[2][3] = gammae[3][2] * g[3] / g[2] * exp(-E[3][2] / (BOLTZMANN * T_gas));
  gammae[3][4] = gammae[4][3] * g[4] / g[3] * exp(-E[4][3] / (BOLTZMANN * T_gas));
  gammae[4][5] = gammae[5][4] * g[5] / g[4] * exp(-E[5][4] / (BOLTZMANN * T_gas));
  gammae[1][3] = gammae[3][1] * g[3] / g[1] * exp(-E[3][1] / (BOLTZMANN * T_gas));
  gammae[1][4] = gammae[4][1] * g[4] / g[1] * exp(-E[4][1] / (BOLTZMANN * T_gas));
  gammae[1][5] = gammae[5][1] * g[5] / g[1] * exp(-E[5][1] / (BOLTZMANN * T_gas));
  gammae[2][4] = gammae[4][2] * g[4] / g[2] * exp(-E[4][2] / (BOLTZMANN * T_gas));
  gammae[2][5] = gammae[5][2] * g[5] / g[2] * exp(-E[5][2] / (BOLTZMANN * T_gas));
  gammae[3][5] = gammae[5][3] * g[5] / g[3] * exp(-E[5][3] / (BOLTZMANN * T_gas));
#endif



  /* build the matrix for the system to solve */

  i = 1;
  j = 1;
  for(i = 1; i <= n; i++)
    {
      for(j = 1; j <= n; j++)
	{
#if defined (UM_H_MET_IMPACTS) && !(defined(UM_e_MET_IMPACTS))
	  mtrx[i][j] = gammaH[j][i] + A[j][i] / n_field_H;
#endif
#if defined (UM_e_MET_IMPACTS) && !(defined(UM_H_MET_IMPACTS))
	  mtrx[i][j] = gammae[j][i] + A[j][i] / n_field_e;
#endif
#if defined (UM_e_MET_IMPACTS) && defined(UM_H_MET_IMPACTS)
	  mtrx[i][j] = gammaH[j][i] + A[j][i] / n_field_H + gammae[j][i] * n_ratio;
#endif
	}
    }

  //now, update the diagonal elements(= 0 so far) to -(sum in column)!
  i = 1;
  j = 1;
  for(j = 1; j <= n; j++)
    {
      for(i = 1; i <= n; i++)
	sum_column[j] += mtrx[i][j];
    }
  i = 1;
  for(i = 1; i <= n; i++)
    mtrx[i][i] = -sum_column[i];
  //...done!

  /*  now, set the condition "x1 + x2 + ... = 1",  
     replacing one (the first) line of mtrx[][] with: ( 1  1 ... 1  1) 
     bocouse of the linear dependence: */

  for(j = 1; j <= n; j++)
    mtrx[1][j] = 1.;

  /* find level populations */

  double mtrx_pass[n][n];
  double b_pass[n];
  double x_got[n];

  // matrix/vectors with index 0-4  
  /*
     the following 3 assignements are necessary to be consistent with 
     the routine sq_system_solver() and the gsl libraries used there
   */
  i = 0;
  j = 0;
  for(i = 0; i < n; i++)
    {
      for(j = 0; j < n; j++)
	mtrx_pass[i][j] = mtrx[i + 1][j + 1];	//1.
      b_pass[i] = b[i + 1];	//2.
    }

  sq_system_solver((double *) mtrx_pass, b_pass, 5, x_got);	//..find the solution

  j = 0;
  for(j = 0; j < n; j++)
    x_pop[j + 1] = x_got[j];	//3.

  /* compute cooling fct */

  return (x_pop[2] * A[2][1] * E[2][1] + x_pop[3] * A[3][2] * E[3][2] + x_pop[3] * A[3][1] * E[3][1] + x_pop[4] * A[4][3] * E[4][3] + x_pop[5] * A[5][4] * E[5][4]) * FeII_ndens;	//erg/cm3/s
#else
  return 0;
#endif
}






/*********************************************************************/
/*           Total metal cooling function (in erg/cm3/s)             */
/*                                                                   */
/*********************************************************************/

#ifdef UM_e_MET_IMPACTS
double um_metal_line_cooling(double T_gas, double rho, int n, double e_ndens)
#else
double um_metal_line_cooling(double T_gas, double rho, int n)
#endif
{
#if !(defined (UM_H_MET_IMPACTS)) && !(defined (UM_e_MET_IMPACTS))
  return 0;
#endif

  /* initialization */

  double CoolSum = 0.;

  CII_ndens = 0.;
  SiII_ndens = 0.;
  FeII_ndens = 0.;
  OI_ndens = 0.;

  set_ndens_and_metallicity(rho, n);

  /* now, set the interacting particles: */

#ifdef UM_H_MET_IMPACTS
  n_field_H = H_ndens;
#endif

#ifdef UM_e_MET_IMPACTS
  n_field_e = e_ndens;
#endif


  if(T_gas < T_SUP_INTERPOL_LIMIT)
    {
#ifdef UM_ENABLE_CII_COOLING
      if(CII_ndens > tiny_value)
	CoolSum += CII_cooling(T_gas);
#endif
#ifdef UM_ENABLE_SiII_COOLING
      if(SiII_ndens > tiny_value)
	CoolSum += SiII_cooling(T_gas);
#endif
#ifdef UM_ENABLE_OI_COOLING
      if(OI_ndens > tiny_value){
	//	CoolSum += OI_cooling(T_gas);
	CoolSum += OI_cooling_2level_approx(T_gas);
	/* uses only first two levels with well known rates */
      }
#endif
#ifdef UM_ENABLE_FeII_COOLING
      if(FeII_ndens > tiny_value){
	/*	CoolSum += FeII_cooling(T_gas);*/
      	CoolSum += FeII_cooling_2level_approx(T_gas);
	/* uses only first two levels with well known rates */
      }
#endif
    }

  return CoolSum;
}


#endif
