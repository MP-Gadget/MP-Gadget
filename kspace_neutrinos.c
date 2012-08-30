#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stddef.h>
#include <sys/types.h>
#include <dirent.h>

#include "allvars.h"
#include "proto.h"


#if defined(KSPACE_NEUTRINOS) && defined(PMGRID)


static double R8;
static double Norm;


static int NPowerTable, NTransfer;

static struct pow_table
{
  double logk, logD, logD2nd;
}
 *PowerTable, *PowerTableList;

static struct transfer_func
{
  double atime;
  int index;
}
 *TransFunc;

static int tfunc_left, tfunc_right;
static double tfunc_atime;








static double sigma2_int(double k, void *param)
{
  double kr, kr3, kr2, w, x;
  double r_tophat = *((double *)param);

  kr = r_tophat * k;
  kr2 = kr * kr;
  kr3 = kr2 * kr;

  if(kr < 1e-8)
    return 0;

  w = 3 * (sin(kr) / kr3 - cos(kr) / kr2);
  x = 4 * M_PI * k * k * w * w * get_powerspec(k, 1.0);

  return x;
}


static double TopHatSigma2(double R)
{
#define WORKSIZE 100000

  double result, abserr;

  gsl_function F;
  gsl_integration_workspace *workspace;

  workspace = gsl_integration_workspace_alloc(WORKSIZE);

  F.function = &sigma2_int;
  F.params = &R;

  double kstart = 1.01*pow(10.0, PowerTableList[0].logk) /  (All.InputSpectrum_UnitLength_in_cm / All.UnitLength_in_cm);
  double kend =  0.99*pow(10.0, PowerTableList[NPowerTable-1].logk) /  (All.InputSpectrum_UnitLength_in_cm / All.UnitLength_in_cm);  

  gsl_integration_qag(&F, kstart, kend, 
		      0.001, 1.0e-4, WORKSIZE, GSL_INTEG_GAUSS41, workspace, &result, &abserr);

  gsl_integration_workspace_free(workspace);

  return result;	

#undef WORKSIZE 
}



static void setup_powertable_for_given_time(double a)
{
  double s;
  int i;

  while(TransFunc[tfunc_left].atime > a && tfunc_left > 0)
    {
      tfunc_left--;
      tfunc_right--;
    }

  while(TransFunc[tfunc_right].atime < a && tfunc_right < NTransfer-1)
    {
      tfunc_left++;
      tfunc_right++;
    }

  s = (log10(a)-log10(TransFunc[tfunc_left].atime)) / (log10(TransFunc[tfunc_right].atime) - log10(TransFunc[tfunc_left].atime));

  for(i=0; i< NPowerTable; i++)
    {
      PowerTable[i].logk = PowerTableList[NPowerTable * TransFunc[tfunc_left].index + i].logk;

      PowerTable[i].logD = 
	(1-s)* PowerTableList[NPowerTable * TransFunc[tfunc_left].index + i].logD +
	(s-0)* PowerTableList[NPowerTable * TransFunc[tfunc_right].index + i].logD;

      PowerTable[i].logD2nd = 
	(1-s)* PowerTableList[NPowerTable * TransFunc[tfunc_left].index + i].logD2nd +
	(s-0)* PowerTableList[NPowerTable * TransFunc[tfunc_right].index + i].logD2nd;
    }

}


static double PowerSpec_Tabulated2nd(double k)
{
  double logk, logD, P, kold, u, dlogk, Delta2;
  int binlow, binhigh, binmid;

  kold = k;
  
  k *= (All.InputSpectrum_UnitLength_in_cm / All.UnitLength_in_cm);	/* convert to h/Mpc */

  logk = log10(k);

  if(logk < PowerTable[0].logk || logk > PowerTable[NPowerTable - 1].logk)
    return 0;

  binlow = 0;
  binhigh = NPowerTable - 1;

  while(binhigh - binlow > 1)
    {
      binmid = (binhigh + binlow) / 2;
      if(logk < PowerTable[binmid].logk)
	binhigh = binmid;
      else
	binlow = binmid;
    }

  dlogk = PowerTable[binhigh].logk - PowerTable[binlow].logk;

  if(dlogk == 0)
    terminate("dlokk = 0 found");

  u = (logk - PowerTable[binlow].logk) / dlogk;

  logD = (1 - u) * PowerTable[binlow].logD2nd + u * PowerTable[binhigh].logD2nd;

  Delta2 = pow(10.0, logD);

  P = Norm * Delta2 / (4 * M_PI * kold * kold * kold);

  return P;
}





static double PowerSpec_Tabulated(double k)
{
  double logk, logD, P, kold, u, dlogk, Delta2;
  int binlow, binhigh, binmid;

  kold = k;

  k *= (All.InputSpectrum_UnitLength_in_cm / All.UnitLength_in_cm);	/* convert to h/Mpc */

  logk = log10(k);

  if(logk < PowerTable[0].logk || logk > PowerTable[NPowerTable - 1].logk)
    return 0;

  binlow = 0;
  binhigh = NPowerTable - 1;

  while(binhigh - binlow > 1)
    {
      binmid = (binhigh + binlow) / 2;
      if(logk < PowerTable[binmid].logk)
	binhigh = binmid;
      else
	binlow = binmid;
    }

  dlogk = PowerTable[binhigh].logk - PowerTable[binlow].logk;

  if(dlogk == 0)
    terminate("dlokk = 0 found");

  u = (logk - PowerTable[binlow].logk) / dlogk;

  logD = (1 - u) * PowerTable[binlow].logD + u * PowerTable[binhigh].logD;

  Delta2 = pow(10.0, logD);

  P = Norm * Delta2 / (4 * M_PI * kold * kold * kold);

  return P;
}









static int compare_logk(const void *a, const void *b)
{
  if(((struct pow_table *) a)->logk < (((struct pow_table *) b)->logk))
    return -1;

  if(((struct pow_table *) a)->logk > (((struct pow_table *) b)->logk))
    return +1;

  return 0;
}



static int compare_atime(const void *a, const void *b)
{
  if(((struct transfer_func *) a)->atime < (((struct transfer_func *) b)->atime))
    return -1;

  if(((struct transfer_func *) a)->atime > (((struct transfer_func *) b)->atime))
    return +1;

  return 0;
}



void init_transfer_functions(void)
{
  DIR *dp;
  FILE *fd;
  struct dirent *ep;
  double redshift;
  int Nall = -1, count;
  char buf[1000];

  NTransfer = 0;

  dp = opendir(All.KspaceDirWithTransferfunctions);
  if(dp != NULL)
    {
      while((ep = readdir(dp)))
	{
	  if(strncmp(ep->d_name, All.KspaceBaseNameTransferfunctions, strlen(All.KspaceBaseNameTransferfunctions)) == 0)
	    {
      	      sprintf(buf, "%s/%s", All.KspaceDirWithTransferfunctions, ep->d_name);
	      
	      if(!(fd = fopen(buf, "r")))
		{
		  sprintf(buf, "can't read input spectrum in file '%s' on task %d\n", buf, ThisTask);
		  terminate(buf);
		}
	      
	      NPowerTable = 0;
	      do
		{
		  double k, T_cdm, T_b, dummy, T_nu, T_tot;
		  
		  /* read transfer function file from CAMB */
		  if(fscanf(fd, " %lg %lg %lg %lg %lg %lg %lg", &k, &T_cdm, &T_b, &dummy, &dummy, &T_nu, &T_tot) == 7)
		    NPowerTable++;
		  else
		    break;
		}
	      while(1);
	      
	      fclose(fd);

	      if(NTransfer == 0)
		Nall = NPowerTable;
	      else
		{
		  if(Nall != NPowerTable)
		    terminate("not all transfer functions are of the same length");
		}

	      NTransfer++;
	    }
	}

      closedir(dp);
    }
  else
    terminate("Couldn't open the directory.");



  if(ThisTask == 0)
    printf("Found %d Transfer functions of length %d each.\n", NTransfer, NPowerTable);




  TransFunc = mymalloc("TransFunc", NTransfer * sizeof(struct transfer_func));
  PowerTableList = mymalloc("PowerTableList", NPowerTable * NTransfer * sizeof(struct pow_table));
  PowerTable = mymalloc("PowerTable", NPowerTable *  sizeof(struct pow_table));




  NTransfer = 0;

  dp = opendir(All.KspaceDirWithTransferfunctions);
  if(dp != NULL)
    {
      while((ep = readdir(dp)))
	{
	  if(strncmp(ep->d_name, All.KspaceBaseNameTransferfunctions, strlen(All.KspaceBaseNameTransferfunctions)) == 0)
	    {
	      redshift = atof(ep->d_name + strlen(All.KspaceBaseNameTransferfunctions));

      
	      sprintf(buf, "%s/%s", All.KspaceDirWithTransferfunctions, ep->d_name);
	      
	      if(!(fd = fopen(buf, "r")))
		{
		  sprintf(buf, "can't read input spectrum in file '%s' on task %d\n", buf, ThisTask);
		  terminate(buf);
		}

	      TransFunc[NTransfer].atime = 1.0/(1+redshift);
	      TransFunc[NTransfer].index = NTransfer;

	      if(ThisTask == 0)
		{
		  printf("reading transfer function at a=%8g: '%s'\n",
			 TransFunc[NTransfer].atime, buf);
		}

	      count = 0;
	      do
		{
		  double k, T_cdm, T_b, dummy, T_nu, T_tot, lD, lDNU;
		  
		  /* read transfer function file from CAMB */
		  if(fscanf(fd, " %lg %lg %lg %lg %lg %lg %lg", &k, &T_cdm, &T_b, &dummy, &dummy, &T_nu, &T_tot) == 7)
		    {
		      PowerTableList[NTransfer*NPowerTable + count].logk = log10(k);

		      lD = pow(k, All.PrimordialIndex) * k * k * k * T_cdm * T_cdm;
		      lDNU = pow(k, All.PrimordialIndex) * k * k * k * T_nu * T_nu;
		      
		      PowerTableList[NTransfer*NPowerTable + count].logD = log10(lD);
		      PowerTableList[NTransfer*NPowerTable + count].logD2nd = log10(lDNU);
		      
		      count++;
		    }
		  else
		    break;
		}
	      while(1);
	      
	      fclose(fd);

	      if(NTransfer == 0)
		Nall = NPowerTable;
	      else
		{
		  if(Nall != NPowerTable)
		    terminate("not all transfer functions are of the same length");
		}

	      NTransfer++;
	    }
	}

      closedir(dp);
    }
  else
    terminate("Couldn't open the directory.");



  for(count=0; count<NTransfer; count++)
    qsort(PowerTableList + count * NPowerTable, NPowerTable, sizeof(struct pow_table), compare_logk);
  
  /* now sort the transfer functions / power spectra by time */

  qsort(TransFunc, NTransfer, sizeof(struct transfer_func), compare_atime);

  if(NTransfer < 2)
    terminate("need at list two transfer functions");

  tfunc_left = 0;
  tfunc_right = 1;
  tfunc_atime = -1.0;


  R8 = 8 * (3.085678e24 / All.UnitLength_in_cm);	/* 8 Mpc/h */

 
  Norm = 1.0;

  double res = TopHatSigma2(R8);
  
  if(ThisTask == 0)
    printf("\nNormalization of spectrum in file:  Sigma8 = %g\n", sqrt(res));

  Norm = All.Sigma8 * All.Sigma8 / res;

  printf("Normalization adjusted to  Sigma8=%g   (Normfac=%g)\n\n", All.Sigma8, Norm);
}




double get_powerspec(double k, double ascale)
{
  double power;

  if(ascale != tfunc_atime)
    {
      /* first, need to prepare a new interpolated version */
      setup_powertable_for_given_time(ascale);

      tfunc_atime = ascale;
    }

  power = PowerSpec_Tabulated(k);

  power *= pow(k, All.PrimordialIndex - 1.0);
  
  return power;
}



double get_neutrino_powerspec(double k, double ascale)
{
  if(ascale != tfunc_atime)
    {
      /* first, need to prepare a new interpolated version */
      setup_powertable_for_given_time(ascale);

      tfunc_atime = ascale;
    }

  double power;

  power = PowerSpec_Tabulated2nd(k);

  power *= pow(k, All.PrimordialIndex - 1.0);

  return power;
}





#endif
