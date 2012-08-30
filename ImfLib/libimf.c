#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_integration.h>
#include "libimf_vars.h"


/* specify what function from gsl qag should be used */
#define limf_gsl_intkey 6

static int gsl_status;

static gsl_function limf_F;



/* /------------------------------------------------\  
 * | IMF functions section                          |
 * |                                                |
 * | > IntegrateIMF_byMass                          |
 * | > IntegrateIMF_byNum                           |
 * | > IntegrateIMF_byEgy                           |
 * |                                                |
 * | > IMFevaluate_byMass_powerlaw                  |
 * | > IMFevaluate_byMass_Larson                    |
 * | > IMFevaluate_byNum_powerlaw                   |
 * | > IMFevaluate_byNum_Larson                     |
 * | > IMFevaluate_byEgy_powerlaw                   |
 * | > IMFevaluate_byEgy_Larson                     |
 * |                                                |
 * | > myIMF_byMass                                 |
 * | > myIMF_byNum                                  |
 * |                                                |
 * | > get_IMF_SlopeBin                             |
 * \------------------------------------------------/ */

double IntegrateIMF(double, double, IMF_Type*, int);
double IntegrateIMF_byWhat(double, double, IMF_Type*, int, int);


/* ============================================================
 *
 *     ---=={[  IMF functions section  ]}==---
 * ============================================================ */


double  IntegrateIMF(double Inf, double Sup, IMF_Type *IMFv, int mode)
     /*
      * actually integrates the IMF, either by mass or by num
      * (mode = 1 or 0 respectively).
     */
{
  int i;
  double partial_result, Sum;
  double slope;
  double sup, inf;
  double sup_slope;
  double result, err;
  double *params;

  partial_result = 0;

  switch (IMFv->type)
    {
    case power_law:
      if(IMFv->NParams == 0)
        {
          /* pure power-law */
          sup_slope = IMFv->MU;
          for(i = 0, Sum = 0; i < IMFv->NSlopes; i++)
            {
              if( Sup <= IMFv->Slopes.masses[i] ||
                  Inf >= sup_slope )
                /*
                 * no overlap
                 */
                continue;
          
              slope = IMFv->Slopes.slopes[i];
              inf = LIMF_MAX(Inf, IMFv->Slopes.masses[i]);
              sup = LIMF_MIN(Sup, sup_slope);

              if(mode == 0)
                /* by number */
                partial_result += IMFv->A[i] * (1 / slope * (pow(inf, -slope) - pow(sup, -slope)));
              else if(mode == 1)
                /* by mass */
                partial_result += IMFv->A[i] * (1 / (1 - slope) * (pow(sup, 1 - slope) - pow(inf, 1 - slope)));
              else
                {
                  IMFp = IMFv;
                  limf_F.function = IMFv->IMFfunc_byEgy;
                  limf_F.params = IMFv->Params;
                  if((gsl_status =
                      gsl_integration_qag(&limf_F, inf, sup, 1e-6, 1e-4, limf_gsl_intspace_dim, 1, limf_w, &result, &err)))
                    {
                      printf("  >> gsl integration error %i in Sn (iimfe) [%.6g - %.6g] : %.6g %.6g\n",
                             gsl_status, inf, sup, result, err);
                      fflush(stdout);
                      return LIBIMF_ERR_INTEGRATION;
                    }
                  partial_result += result;
                }

              if(i < IMFv->NSlopes-1)
                sup_slope = IMFv->Slopes.masses[i];
            }
        }
      else
        {
          /* ============================  */
          /*   place here your own code    */
          /* i.e. a call to IMFv->func()   */
          /* ============================  */

          /* built_in function: Larson IMF, time-dependent with one parameter */

          if(mode == 0 ||
             mode == 2)
            /* by number or by energy */
            {
              IMFp = IMFv;
              if(mode == 0)
                limf_F.function = IMFv->IMFfunc_byNum;
              else
                limf_F.function = IMFv->IMFfunc_byEgy;
              limf_F.params = IMFv->Params;
              if((gsl_status =
                  gsl_integration_qag(&limf_F, Inf, Sup, 1e-6, 1e-4, limf_gsl_intspace_dim, 1, limf_w, &result, &err)))
                {
                  printf
                    ("  >> gsl integration error %i in Sn (iimf) [%.6g - %.6g] : %.6g %.6g\n",
                     gsl_status, Inf, Sup, result, err);
                  fflush(stdout);
                  return LIBIMF_ERR_INTEGRATION;
                }

              /*****************

               note:: define slope!

              ******************/

              partial_result += IMFv->Atot * 1 / (1 - slope) *
                (IMFv->Params[0] *
                 (pow(1 + Sup / IMFv->Params[0], 1 - slope) - pow(1 + inf / IMFv->Params[0], 1 - slope)));
              
            }
          else
            /* by mass */
            {
              params = IMFv->Params;
              partial_result += IMFv->Atot * 1 / (1 - slope) *
                (params[0] * (pow(1 + sup / params[0], 1 - slope) - pow(1 + inf / params[0], 1 - slope)));
            }
        }
      break;
    case whatever:
      IMFp = IMFv;
      if(mode == 0)
        limf_F.function = IMFv->IMFfunc_byNum;
      else
        limf_F.function = IMFv->IMFfunc_byMass;
      limf_F.params = IMFv->Params;
      if((gsl_status =
          gsl_integration_qag(&limf_F, Inf, Sup, 1e-6, 1e-4, limf_gsl_intspace_dim, 1, limf_w, &result, &err)))
        {
          printf
            ("  >> gsl integration error %i in Sn (iimf) [%.6g - %.6g] : %.6g %.6g\n",
             gsl_status, Inf, Sup, result, err);
          fflush(stdout);
          return LIBIMF_ERR_INTEGRATION;
        }
      partial_result = result;
      break;
    }

  return partial_result;
}


double  IntegrateIMF_byWhat(double Inf, double Sup, IMF_Type * IMFv, int modeBH, int modeWhat)
     /*
      * determine what actually is the interval of integration accounting
      * for the BH ranges; then calls IntegrateIMF so that to integrate by 
      * Number (modeWhat = 0), by Mass (modeWhat = 1) or by Energy (modeWhat = 2)
      */
{
  double inf, sup, partial_result;
  double tmp;
  int i;

  if(Sup <= IMFv->Mm || Inf >= IMFv->MU)
    return 0;
  if(Inf < IMFv->Mm)
    Inf = IMFv->Mm;
  if(Sup > IMFv->MU)
    Sup = IMFv->MU;

  if(modeBH == INC_BH)
    return IntegrateIMF(Inf, Sup, IMFv, 1);
  else
    {
      partial_result = 0;

      for(i = 0; i < IMFv->N_notBH_ranges; i++)
        {
          if( Sup <= IMFv->notBH_ranges.inf[i] ||
              Inf >= IMFv->notBH_ranges.sup[i] )
            /*
             * no overlap
             */
            continue;
          
          inf = LIMF_MAX(Inf, IMFv->notBH_ranges.inf[i]);
          sup = LIMF_MIN(Sup, IMFv->notBH_ranges.sup[i]);
          
          if( (tmp = IntegrateIMF(inf, sup, IMFv, modeWhat)) < 0)
            return tmp;
          partial_result += tmp;
        }
    }
  return partial_result;
}

double IntegrateIMF_byNum(double Inf, double Sup, IMF_Type *IMFv, int mode)
     /*
      * determine what actually is the interval of integration accounting
      * for the BH ranges; then calls IntegrateIMF
      */
{
  return IntegrateIMF_byWhat(Inf, Sup, IMFv, mode, 0);
}

double IntegrateIMF_byMass(double Inf, double Sup, IMF_Type *IMFv, int mode)
     /*
      * determine what actually is the interval of integration accounting
      * for the BH ranges; then calls IntegrateIMF
      */
{
  return IntegrateIMF_byWhat(Inf, Sup, IMFv, mode, 1);
}

double IntegrateIMF_byEgy(double Inf, double Sup, IMF_Type *IMFv)
     /*
      * determine what actually is the interval of integration accounting
      * for the BH ranges; then calls IntegrateIMF
      */
{
  if(IMFv->NEKin == 1)
    return IntegrateIMF_byWhat(Inf, Sup, IMFv, EXC_BH, 0) * IMFv->EKin.ekin[0];
  else
    return IntegrateIMF_byWhat(Inf, Sup, IMFv, EXC_BH, 2);
}


/* ---------------------
 * by Mass IMF functions
 * --------------------- */

double  IMFevaluate_byMass_powerlaw(double arg, void *v)
     /*
	note: this function does not care whether or not the mass
	arg will end in a BH.
      */
{
  int i;

  /* pure power-law */
  if( arg < IMFp->Mm ||
      arg > IMFp->MU )
    return 0;

  for(i = 0; i < IMFp->NSlopes; i++)
    if(arg > IMFp->Slopes.masses[i])
      break;
  return IMFp->A[i] * pow(arg, -IMFp->Slopes.slopes[i]);
}

double  IMFevaluate_byMass_Larson(double arg, void *v)
     /*
	note: this function does not care whether or not the mass
	arg will end in a BH.
      */
{
  /* built_in function: Larson IMF, time-dependent with one parameter */

  if( arg < IMFp->Mm ||
      arg > IMFp->MU )
    return 0;

  return IMFp->Atot * pow(1 + arg / *(double *) v, -(IMFp->Slopes.slopes[0]));
}



/* -----------------------
 * by Number IMF functions
 * ----------------------- */


double  IMFevaluate_byNum_powerlaw(double arg, void *v)
     /*
	note: this function does not care whether or not the mass
	arg will end in a BH.
      */
{
  int i;

  /* pure power-law */
  if( arg < IMFp->Mm ||
      arg > IMFp->MU )
    return 0;


  for(i = 0; i < IMFp->NSlopes; i++)
    if(arg > IMFp->Slopes.masses[i])
      break;
  return IMFp->A[i] * pow(arg, -(1 + IMFp->Slopes.slopes[i]));
}

double  IMFevaluate_byNum_Larson(double arg, void *v)
     /*
	note: this function does not care whether or not the mass
	arg will end in a BH.
      */
{
  /* built_in function: Larson IMF, time-dependent with one parameter */
  if( arg < IMFp->Mm ||
      arg > IMFp->MU )
    return 0;

  return IMFp->Atot * pow(1 + arg / *(double *) v, -(IMFp->Slopes.slopes[0])) / arg;
}

/* -----------------------
 * get energy
 * ----------------------- */


double  IMFevaluate_byEgy_powerlaw(double arg, void *v)
     /*
	note: this function does not care whether the mass arg
	will end in a BH or in a SnII.
      */
{
  int i, j;
  double egy;

  if( arg < IMFp->Mm ||
      arg > IMFp->MU )
    return 0;

  /* pure power-law */
  for(i = 0; i < IMFp->NSlopes; i++)
    if(arg > IMFp->Slopes.masses[i])
      break;
  for(j = 0; j < IMFp->NEKin - 1; j++)
    if(arg < IMFp->EKin.masses[j + 1])
      break;
  egy = (arg - IMFp->EKin.masses[j]) / (IMFp->EKin.masses[j + 1] - IMFp->EKin.masses[j]);
  egy = IMFp->EKin.ekin[j] + (IMFp->EKin.ekin[j + 1] - IMFp->EKin.ekin[j]) * egy;
  return IMFp->A[i] * pow(arg, -(1 + IMFp->Slopes.slopes[i])) * egy;
}

double  IMFevaluate_byEgy_Larson(double arg, void *v)
     /*
	note: this function does not care whether the mass arg
	will end in a BH or in a SnII.
      */
{
  double egy;
  int j;

  if( arg < IMFp->Mm ||
      arg > IMFp->MU )
    return 0;

  /* built_in function: Larson IMF, time-dependent with one parameter */

  for(j = 0; j < IMFp->NEKin; j++)
    if(arg < IMFp->EKin.masses[j + 1])
      break;
  egy = (arg - IMFp->EKin.masses[j]) / (IMFp->EKin.masses[j+1] - IMFp->EKin.masses[j]);
  egy = IMFp->EKin.ekin[j] + (IMFp->EKin.ekin[j + 1] - IMFp->EKin.ekin[j]) * egy;
  return IMFp->Atot * pow(1 + arg / *(double *) v, -(IMFp->Slopes.slopes[0])) / arg;
}


/* ============================================================
 *
 *     ---=={[  IMF parameters section  ]}==---
 * ============================================================ */

/*                                                           */
/*  you should place here your own code giving parameters of */
/*  imfs                                                     */

/*  this example is for Larson IMF with params taken from    */
/*  Boehringer et al, 2003                                   */

double get_imf_params(int I, double *v, double time)
{
#define z_imf_inf 2
#define time_imf_inf (1/(z_imf_inf + 1))
#define z_imf_sup 10
#define time_imf_sup (1/(z_imf_sup + 1))
#define ms_inf 0.4467
#define ms_sup 10
#define logsup  log10(ms_sup)
#define loginf log10(ms_inf)

  double z = 1.0 / time - 1;

  if(z <= z_imf_inf)
    return 0.35;
  else if(z >= z_imf_sup)
    return 10;
  else
    return pow(10, ((z - z_imf_inf) / (z_imf_sup - z_imf_inf) * (logsup - loginf) + loginf));
}


int get_IMF_SlopeBin(double m)
{
  int i;

  if( m < IMFp->Mm ||
      m > IMFp->MU )
    return -1;


  for(i = 0; (i < IMFp->NSlopes) && (m < IMFp->Slopes.masses[i]); i++)
    ;
  return i;
}


int not_in_BHrange(int IMFi, double *mass)
{
  int j;

  if(*mass > IMFs[IMFi].MU)
    return 0;
  if(*mass < IMFs[IMFi].Mm)
    return 0;

  for(j = 0; j < IMFs[IMFi].N_notBH_ranges * 2; j++)
    if(*mass <= IMFs[IMFi].notBH_ranges.list[j + 1])
      break;

  if(!(j && 1))
    return 1;
  else
    return -(j + 1);

}



/* /------------------------------------------------\  
 * | U T I L I T I E S                              |
 * |                                                |
 * | > printf_IMF                                   |
 * | > write_IMF_info                               |
 * | > read_imfs                                    |
 * | > read_ekin                                    |
 * | > initialize_externalIMFs                      |
 * \------------------------------------------------/ */



void print_IMF(int num, char* IMFfilename)
     /*
      * print the IMF in a file named IMFfilename.%02d
      * where IMFfilename is the name of the IMF file
      */
{
  FILE *file;
  char filename[500];

#define Npoints 100
  int npoints, npoints_tot, i, j;
  double sup, inf, delta, *x;

  if(IMFfilename != 0x0)
    sprintf(filename, "%s.%02d", IMFfilename, num);
  else
    sprintf(filename, "IMFs.%02d", num);
  file = fopen(filename, "w");

  x = (double *)malloc(Npoints * sizeof(double));

  if(IMFs[num].NSlopes == 1)
    {
      inf = IMFs[num].Mm;
      sup = IMFs[num].MU;
      delta = log10(sup/inf) / Npoints;

      x[0] = inf;
      for(npoints = Npoints, i = 1;
	  i < npoints;
	  i++)
	x[i] = x[0] * pow(10, delta * i);
      x[Npoints - 1] = sup;
      npoints_tot = Npoints;
    }
  else
    for(sup = IMFs[num].MU, npoints_tot = j = 0; j < IMFs[num].NSlopes; j++)
      {
	inf = IMFs[num].Slopes.masses[j];
	npoints = log10(sup / inf) / log10(IMFs[num].MU / IMFs[num].Mm) * Npoints;
	delta = log10(sup/inf) / npoints;

	x[npoints_tot] = sup;
	for(i = 1;
	    i < npoints;
	    i++)
	  x[npoints_tot + i] = x[npoints_tot] * pow(10, -delta * i);
	x[npoints_tot + npoints - 1] = inf;
	npoints_tot += npoints-1;

	sup = inf;
      }

  IMFp = &IMFs[num];

  for(i = 1;
      i < npoints_tot;
      i++)
    fprintf(file, "%g %g %g\n",
	    x[i], (*IMFs[num].IMFfunc_byMass)(x[i], 0x0), (*IMFs[num].IMFfunc_byNum)(x[i], 0x0));

  fclose(file);
   return;
}


void write_IMF_info(int num, FILE *file)
{
  int i;
  double sup, inf;

  switch (IMFs[num].type)
    {
    case power_law:
      i = 0;
      break;
    case whatever:
      i = 1;
      break;
    default:
      i = -1;
      break;
    }

  if(i == -1)
    {
      fprintf(file, "IMF %d has an unknown type\n", num);
      return;
    }

  fprintf(file,  "::  IMF %3d\n"
	  "    type           : %-10s\n"
          "    name           : %-s\n"
	  "    depends on time: %-3s\n"
	  "    normalization  : %-4.3e\n",
	  num, IMF_Spec_Labels[i], IMFs[num].name, (IMFs[num].timedep) ? "yes" : "no", IMFs[num].Atot);
  if(IMFs[num].NParams > 0)
    {
      fprintf(file, "    Parameters: \n");
      for(i = 0; i < IMFs[num].NParams; i++)
	fprintf(file, "      %-8.6f ", IMFs[num].Params[i]);
      fprintf(file, "\n");
    }
  else
    fprintf(file, "    No parameters\n");

  fprintf(file, "    Slopes: \n");
  for(i = 0, sup = IMFs[num].MU; i < IMFs[num].NSlopes; i++)
    {
      inf = IMFs[num].Slopes.masses[i];
      fprintf(file, "      [%-6.4f:%-6.4f] -> %-6.4f [norm: %-6.4f]\n", sup, inf, IMFs[num].Slopes.slopes[i], IMFs[num].A[i]);
      sup = IMFs[num].Slopes.masses[i];
    }

  fprintf(file, "\n");
  return;
}

double Renormalize_IMF(IMF_Type *IMFv, double inf, double sup, int BHmode)
{
  double A;
  double mleft, mright;
  int i;

  IMFp = IMFv;

  A = IntegrateIMF_byMass(inf, sup, IMFv, BHmode);
  if(A > 0)
    {
      if(fabs(A-1)/A > IMFv->Mm/1000)
        {
          IMFv->A[IMFv->NSlopes-1] = 1;
          for(i = IMFv->NSlopes-2; i>=0; i--)
            {
              IMFv->A[i] = 1;
              mleft = (double)IMFv->Slopes.masses[i]*(1-1e-9);
              mright = (double)IMFv->Slopes.masses[i]*(1+1e-9);
              IMFv->A[i] = IMFv->IMFfunc_byNum(mleft, 0x0) /
                IMFv->IMFfunc_byNum(mright, 0x0);
            }

          A = IntegrateIMF_byMass(inf, sup, IMFv, BHmode);

          for(i = IMFv->NSlopes-1; i>=0; i--)
            IMFv->A[i] /= A;

          A = IntegrateIMF_byMass(inf, sup, IMFs, BHmode);

/*           for(i = IMFv->NSlopes-1; i>=0; i--) */
/*             printf("[%6.4e -> %6.4e] %8.6e\n", IMFv->Slopes.masses[i], (i>0)?IMFv->Slopes.masses[i-1]:IMFv->MU, IMFv->A[i]); */
        }
    }
  return 1.0/A;
}

void allocate_IMF_integration_space()
{
  /* initialize integration */
  if(limf_w == 0x0)
    limf_w = gsl_integration_workspace_alloc(limf_gsl_intspace_dim);
  return;
}

int read_ekin_file(char*, double**, double**);

int read_imfs(char *filename)
     /*
      * this routine read in imfs' data
      */
{
  FILE *myfile = 0x0;
  char *charp, buff[500], param[100], value[100];

  int IMF_index, IMF_Nspec;
  int i, j, n, line_num;

  double usr, p;

  if((myfile = fopen(filename, "r")) == 0x0)
    {
      printf("it's impossible to open the imf file <%s>\nwe terminate here!\n", filename);
      return(LIBIMF_ERR_IMF_FILE);
    }
  
  /* now read in the imfs data.
   * the file is organized as follows:
   * first, 4 lines contain respectively the imf type, the time dependence of
   * the imf (0|1), the maximum and minimum mass.
   * then a line contains the number of slopes, wheter the imf is multi-slope, and
   * as much lines as the slopes follow, each with a pair (limiting mass, slope).
   * finally a line contains the number of parameters: each of the following lines
   * contains a parameter.
   * lines beginning with a "#" are ignored.
   */


  fscanf(myfile, "%d\n", &IMFs_dim);

  IMFs = (IMF_Type *) calloc(IMFs_dim, sizeof(IMF_Type));

  if(IMFs == 0x0)
    {
      printf("[Task 0] memory allocation failed when reading imfs' file\n");
      return(LIBIMF_ERR_IMF_ALLOCATE);
    }

  Nof_TimeDep_IMF = 0;

  i = 0;
  line_num = 0;
  IMF_index = -1;
  IMF_Nspec = 0;
  do
    {
      charp = fgets(buff, 500, myfile);
      line_num++;
      if(charp != 0x0 && buff[0] != '#')
	{
	  ((p = sscanf(buff, "%s %s", &param[0], &value[0])) >= 1) ? n++ : n;

	  if(strcmp(param, "TYPE") == 0)
	    {
	      if(IMF_index >= 0)
		{
		  if(IMF_Nspec < IMF_NSPEC)
		    {
		      printf("error in ifms file format at line %d:"
			     " IMF num %d has not been completely specified\n", line_num, IMF_index);
		      return(LIBIMF_ERR_IMF_FILE_FORMAT);
		    }
		}
	      if(++IMF_index > IMFs_dim)
		{
		  printf("error in ifms file format at line %d\n", line_num);
		  return(LIBIMF_ERR_IMF_FILE_FORMAT);
		}

	      IMF_Nspec = 1;
	      if(strcmp(value, "PowerLaw") == 0)
		{
		  IMFs[IMF_index].type = power_law;
		  IMFs[IMF_index].IMFfunc_byMass = &IMFevaluate_byMass_powerlaw;
		  IMFs[IMF_index].IMFfunc_byNum = &IMFevaluate_byNum_powerlaw;
		}
	      else if(strcmp(value, "Whatever") == 0)
                IMFs[IMF_index].type = whatever;
	      /* else if: place here your own code */
	    }
          else if(strcmp(param, "NAME") == 0)
	    {
	      IMF_Nspec++;
	      IMFs[IMF_index].name = (char*)malloc(strlen(buff + strlen(param) + 1) + 1);
              sprintf(IMFs[IMF_index].name, "%s", buff + strlen(param) + 1);
              for(j = strlen(IMFs[IMF_index].name)-1; j >=0; j--)
                if(IMFs[IMF_index].name[j] == '\n')
                  {
                    IMFs[IMF_index].name[j] = '\0';
                    break;
                  }
              if(IMFs[IMF_index].type == whatever)
                {
                  if(NexternalIMFs > 0)
                    {
                      if( (j = search_externalIMF_name(value)) >= 0)
                        {
                          IMFs[IMF_index].IMFfunc_byMass = externalIMFs_byMass[j];
                          IMFs[IMF_index].IMFfunc_byNum  = externalIMFs_byNum[j];
                        }
                      else
                        {
                          printf("no byMass and byNum functions assigned to an IMF of name <%s>\n", value);
                          return(-11);
                        }
                      
                    }
                  else
                    {
                      printf("no external IMFs specified (try to execute initialize_externalIMFs and set_externalIMF before caling read_imfs() )\n");
                      return(-11);
                    }
                }
	    }
	  else if(strcmp(param, "TDEP") == 0)
	    {
	      IMF_Nspec++;
	      IMFs[IMF_index].timedep = atoi(value);
	    }
	  else if(strcmp(param, "MU") == 0)
	    {
	      IMF_Nspec++;
	      IMFs[IMF_index].MU = atof(value);
	    }
	  else if(strcmp(param, "Mm") == 0)
	    {
	      IMF_Nspec++;
	      IMFs[IMF_index].Mm = atof(value);
	    }
	  else if(strcmp(param, "NPar") == 0)
	    {
	      IMF_Nspec++;

	      if((IMFs[IMF_index].NParams = atoi(value)) > 0)
		{
		  if((IMFs[IMF_index].Params =
		      (double *) malloc(IMFs[IMF_index].NParams * sizeof(double))) == 0x0)
		    {
		      printf("[Task 0][a] memory allocation failed when reading imfs' file\n");
		      return(LIBIMF_ERR_IMF_ALLOCATE);
		    }
		  for(j = 0; j < IMFs[IMF_index].NParams; j++)
		    {
		      i = fscanf(myfile, "%s\n", buff);
		      line_num++;
		      if(i == EOF || i < 1)
			{
			  printf("error in ifms' file format at line %d\n", line_num);
			  return(LIBIMF_ERR_IMF_FILE_FORMAT);
			}
		      if(buff[0] != '#')
			i = sscanf(buff, "%lf\n", &IMFs[IMF_index].Params[j]);
		    }
		}
	      else
		IMFs[IMF_index].Params = 0x0;
	    }
	  else if(strcmp(param, "NSlopes") == 0)
	    {
	      IMF_Nspec++;

	      IMFs[IMF_index].NSlopes = atoi(value);

	      IMFs[IMF_index].Slopes.masses =
		(double *) malloc(IMFs[IMF_index].NSlopes * sizeof(double));
	      IMFs[IMF_index].Slopes.slopes =
		(double *) malloc(IMFs[IMF_index].NSlopes * sizeof(double));
	      IMFs[IMF_index].A =
		(double *) malloc((IMFs[IMF_index].NSlopes) * sizeof(double));
	      if(IMFs[IMF_index].Slopes.masses == 0x0 ||
		 IMFs[IMF_index].Slopes.slopes == 0x0 ||
		 IMFs[IMF_index].A == 0x0)
		{
		  printf("[Task 0][c] memory allocation failed when reading imfs' file\n");
		  return(LIBIMF_ERR_IMF_ALLOCATE);
		}

	      for(j = 0; j < IMFs[IMF_index].NSlopes; j++)
		{
		  charp = fgets(buff, 500, myfile);
		  line_num++;
		  if(charp == 0x0)
		    {
		      printf("error in IMFs' file format at line %d: not enough slopes specified",
			     line_num);
		      return(LIBIMF_ERR_IMF_FILE_FORMAT);
		    }
		  if(buff[0] != '#')
		    {
		      /* the slope file format is as follos:
		       *   inf mass, slope, normalization
		       * where inf mass is the inf limit of the mass interval to which
		       * the slope refers. then, larger masses come first. */
		      i = sscanf(buff, "%lf %lf %lf\n",
				 &IMFs[IMF_index].Slopes.masses[j], &IMFs[IMF_index].Slopes.slopes[j], &usr);
		      if(i ==3)
			IMFs[IMF_index].A[j] = usr;
		      else
                        {
                          IMFs[IMF_index].A[j] = 1.0;
                          if(i == 1)
                            IMFs[IMF_index].Slopes.slopes[j] = 0;
                        }
		    }
		}
/* 	      if(IMFs[IMF_index].NSlopes == 1) */
/* 		IMFs[IMF_index].A[0] = 1; */
	    }
	  else if(strcmp(param, "N_notBH") == 0)
	    {
	      IMF_Nspec++;

	      IMFs[IMF_index].N_notBH_ranges = atoi(value);

	      IMFs[IMF_index].notBH_ranges.sup =
		(double *) malloc((IMFs[IMF_index].N_notBH_ranges + 1) * sizeof(double));
	      IMFs[IMF_index].notBH_ranges.inf =
		(double *) malloc((IMFs[IMF_index].N_notBH_ranges + 1) * sizeof(double));

              IMFs[IMF_index].notBH_ranges.list =
                (double *) malloc((IMFs[IMF_index].N_notBH_ranges + 1) * 2 * sizeof(double));

	      if(IMFs[IMF_index].notBH_ranges.sup == 0x0 || IMFs[IMF_index].notBH_ranges.inf == 0x0)
		{
		  printf("[Task 0][d] memory allocation failed when reading imfs' file\n");
		  return(LIBIMF_ERR_IMF_ALLOCATE);
		}

	      for(j = 0; j < IMFs[IMF_index].N_notBH_ranges; j++)
		{
		  charp = fgets(buff, 500, myfile);
		  line_num++;
		  if(charp == 0x0)
		    {
		      printf("error in IMFs' file format at line %d: not enough slopes specified",
			     line_num);
		      return(LIBIMF_ERR_IMF_FILE_FORMAT);
		    }
		  if(buff[0] != '#')
		    /* the slope file format is as follos:
		     *   inf mass, slope
		     * where inf mass is the inf limit of the mass interval to which
		     * the slope refers. then, larger masses come first. */
		    i = sscanf(buff, "%lf %lf\n",
			       &IMFs[IMF_index].notBH_ranges.sup[j],
			       &IMFs[IMF_index].notBH_ranges.inf[j]);
                  IMFs[IMF_index].notBH_ranges.list[j * 2] = IMFs[IMF_index].notBH_ranges.inf[j];
                  IMFs[IMF_index].notBH_ranges.list[j * 2 + 1] = IMFs[IMF_index].notBH_ranges.sup[j];
		}
	      IMFs[IMF_index].notBH_ranges.sup[j] = 0;
	      IMFs[IMF_index].notBH_ranges.inf[j] = 0;
	    }
	  else if(strcmp(param, "YSet") == 0)
	    {
	      IMF_Nspec++;
	      IMFs[IMF_index].YSet = atoi(value);
	    }
	  else if(strcmp(param, "EKin") == 0)
	    {
	      IMF_Nspec++;

	      if(isdigit(*value))
		{
		  IMFs[IMF_index].EKin.masses = (double *) malloc(sizeof(double));
		  IMFs[IMF_index].EKin.ekin = (double *) malloc(sizeof(double));
		  if(IMFs[IMF_index].MU > 0)
		    *(IMFs[IMF_index].EKin.masses) = IMFs[IMF_index].MU;
		  else
		    {
		      printf("please, specify single-value EKin after MU in the IMF file\n");
		      return(LIBIMF_ERR_IMF_FILE_FORMAT);
		    }
		  *(IMFs[IMF_index].EKin.ekin) = atof(value);
		  IMFs[IMF_index].NEKin = 1;
		  IMFs[IMF_index].IMFfunc_byEgy = 0x0;
		}
	      else
		{
		  if( (IMFs[IMF_index].NEKin =
                       read_ekin_file(value, &IMFs[IMF_index].EKin.masses, &IMFs[IMF_index].EKin.ekin)) <= 0)
                    {
                      printf("some error occurred while reading <%s> energy file\n", value);
                      return(LIBIMF_ERR_IMF_FILE);
                    }
                        
		  if(IMFs[IMF_index].type == power_law)
		    IMFs[IMF_index].IMFfunc_byEgy = IMFevaluate_byEgy_powerlaw;
                  else if(IMFs[IMF_index].type == whatever)
                    {
                      if(NexternalIMFs > 0)
                        {
                          if( (j = search_externalIMF_name(value)) >= 0)
                            IMFs[IMF_index].IMFfunc_byMass = externalIMFs_byEgy[j];
                          else
                            {
                              printf("no byEgy  function assigned to an IMF of name <%s>\n", value);
                              return(-11);
                            }
                      
                        }
                      else
                        {
                          printf("no external IMFs specified (try to execute initialize_externalIMFs and set_externalIMF before caling read_imfs() )\n");
                          return(-11);
                        }
                    }
		  /* place your own else if here */
		}
	    }
	  else
	    {
	      printf("at line %d in IMFs' format file: unknown parameter (ignored)\n", line_num);
	      /*return(LIBIMF_ERR_IMF_FILE_FORMAT);*/
	    }
	}
    }			/* close the while */
  while(charp != 0x0);

  /* performs some checks.. */

  for(i = 0; i < IMFs_dim; i++)
    {
      if(IMFs[i].type == power_law && IMFs[i].NSlopes > 1)
        {
          if(IMFs[i].Slopes.masses[0] > IMFs[i].MU)
            {
              printf("slope%c in IMF %3d have been badly specified:\n"
                     "the inf bound for the first slopes (%g) cannot be larger than the maximum mass allowed (%g)\n",
                     (IMFs[i].NSlopes>1)? 's':' ', i, IMFs[i].Slopes.masses[0], IMFs[i].MU);
              return(LIBIMF_ERR_IMF_FILE_FORMAT);
            }
          if(IMFs[i].Slopes.masses[IMFs[i].NSlopes - 1] != IMFs[i].Mm)
            {
              printf("slope%c in IMF %3d have been badly specified:\n"
                     "the inf bound for the last slopes (%g) cannot be different from the minimum mass allowed (%g)\n",
                     (IMFs[i].NSlopes>1)? 's':' ', i, IMFs[i].Slopes.masses[0], IMFs[i].Mm);
              return(LIBIMF_ERR_IMF_FILE_FORMAT);
            }
          
          for(j = 0; j < IMFs[i].NSlopes; j++)
            if(IMFs[i].A[j] == 1)
              {
                printf("the slope in IMF %3d has normalization 1 in the range %d; if you mean that i should\n"
                       "calculate the right normalization, i advice you that i won't do it. Look at the literature\n"
                       "to find the correct value (an observational guess is needed)\n",
                       i, j);
	    }
        }
    }


  printf("%d IMF%c read\n", IMFs_dim, (IMFs_dim > 1) ? 's' : '\0');
  fflush(stdout);

  return 0;
}


int read_ekin_file(char *string, double **mass_array, double **ekin_array)
     /* 
      * read in kinetic energies from a file; energy should be given
      * in units of 1e51 ergs
      *
      */
{
  double *masses = 0x0, *ekin = 0x0;
  FILE *myfile;
  char *charp, buff[500];
  int i, Num;

  if((myfile = fopen(string, "r")) == 0x0)
    {
      printf("it's impossible to open the kinetic energy file <%s>\nwe terminate here!\n", string);
      fflush(stdout);
      return(-(LIBIMF_ERR_IMF_FILE));
    }

  do
    charp = fgets(buff, 500, myfile);
  while(buff[0] == '#' && charp != 0x0);

  sscanf(buff, "%i %*s\n", &Num);
  masses = (double *) malloc(Num * sizeof(double));
  ekin = (double *) malloc(Num * sizeof(double));

  for(i = 0; i < Num; i++)
    {
      do
	charp = fgets(buff, 500, myfile);
      while((charp != 0x0) && (buff[0] == '#'));
      sscanf(buff, "%lg %lg\n", &masses[i], &ekin[i]);
    }

  *mass_array = masses;
  *ekin_array = ekin;

  fclose(myfile);

  return Num;
}


int initialize_externalIMFs(int N)
{
  NexternalIMFs = N;

  externalIMFs_byMass = (EXTERNALIMF*)malloc(NexternalIMFs * sizeof(EXTERNALIMF));
  externalIMFs_byNum  = (EXTERNALIMF*)malloc(NexternalIMFs * sizeof(EXTERNALIMF));
  externalIMFs_byEgy  = (EXTERNALIMF*)malloc(NexternalIMFs * sizeof(EXTERNALIMF));
  
  externalIMFs_names = (char**)malloc(NexternalIMFs * sizeof(char**));
  return 0;
}

int set_externalIMF(int N, char *name, EXTERNALIMF func_byMass, EXTERNALIMF func_byNum, EXTERNALIMF func_byEgy)
{
  if(externalIMFs_byMass != 0x0 &&
     externalIMFs_byNum  != 0x0 &&
     externalIMFs_byEgy  != 0x0 &&
     N < NexternalIMFs)
    {
      externalIMFs_names[N] = (char*)malloc(strlen(name) + 1);
      sprintf(&externalIMFs_names[N][0], "%s", name);

      externalIMFs_byMass[N] = func_byMass;
      externalIMFs_byNum[N]  = func_byNum;
      externalIMFs_byEgy[N]  = func_byEgy;

      return 0;
    }
  else
    return -1;
}

int search_externalIMF_name(char *name)
{
  int i;

  for(i = 0; i < NexternalIMFs; i++)
    if(strcmp(externalIMFs_names[i], name) == 0)
      break;

  if(i < NexternalIMFs)
    return i;
  else
    return -1;
}
