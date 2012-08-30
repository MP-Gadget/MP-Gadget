#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>

#include "allvars.h"
#include "proto.h"

#include "lt_error_codes.h"

#ifdef LT_STELLAREVOLUTION


/* 
   A miscelannea of routines are collected here.
   Index of sections:
   > particle metallicity calculations
   > read SF specification file (IMFs are now read in libimf)
   > evaluation of SF/IMF for a particle
   > packing routines
*/



#ifndef INLINE_FUNC
#ifdef INLINE
#define INLINE_FUNC inline
#else
#define INLINE_FUNC
#endif
#endif


/* ============================================================
 *
 *        ---=={[  get metallicity section  ]}==---
 * ============================================================ */


double INLINE_FUNC get_metalmass(float *metals)
{
  double Sum;
  int i;

  for(i = 0, Sum = 0; i < LT_NMetP; i++)
    Sum += metals[i];
  if(Hel >= 0)
    Sum -= metals[Hel];
  return Sum;
}

double INLINE_FUNC get_metallicity_solarunits(MyFloat Z)
{
  if(Z > 0)
    {
      Z *= 381.68;
      Z = log10(Z);
      if(Z <= ZMin)
	return NO_METAL;
      else
	return Z;
    }
  return NO_METAL;
}

double INLINE_FUNC get_metallicity(int i, int mode)
     /*
      * mode < 0 :: return global metallicity
      * mode >= 0 :: return the abundance of element #mode (Mass / Hydrogen_Mass)
      */
{
  /* Sutherland&Dopita 1993 used Anders & Grevesse 1989 as for solar abundances.
     Then,
     Fe / H = 2.62e-3  by mass
     O / H = 1.36e-2 by mass
   */

  float *Metals, Mass;
  double MetalMass = 0, Hmass;
  int j;

  if(P[i].Type & 7)		/* not a gas particle */
    {
      Metals = &MetP[P[i].MetID].Metals[0];
      Mass = MetP[P[i].MetID].iMass;
    }
  else
    {
      Metals = &SphP[i].Metals[0];
      Mass = P[i].Mass + SphP[i].MassRes;
    }

  for(j = 0; j < LT_NMetP; j++)
    /* include also He if present */
    MetalMass += Metals[j];

  Hmass = (Mass - MetalMass);

  if(mode >= 0)
    return (Metals[mode] / Hmass);
  else
    {
      if(Hel >= 0)
	MetalMass -= Metals[Hel];
      return (MetalMass / Hmass);
    }
}

/* ============================================================
 *
 *     ---=={[  SF specification loading section  ]}==---
 * ============================================================ */


int load_SFs_IMFs(void)
     /*
      * this routine read in SFs and IMFs data
      */
{
  FILE *file;
  char *charp, buff[1500], param[100], value[100], buffer[200];
  int SF_index, SF_Nspec;
  int i, j, p, n, line_num;

  if(ThisTask == 0)
    /* Task 0 read all data and then broadcast them */
    {
      sprintf(buff, "%s/%s", All.OutputDir, All.IMFfilename);

      if(read_imfs(buff) != 0)
	endrun(LT_ERR_SF_FILE);

      for(i = 0; i < IMFs_dim; i++)
	for(j = 0; j < IMFs[i].NEKin; j++)
	  IMFs[i].EKin.ekin[j] *= 1e51 / All.UnitEnergy_in_cgs;

      sprintf(buff, "%s%s", All.OutputDir, All.SFfilename);

      if((file = fopen(buff, "r")) == 0x0)
	{
	  printf("it's impossible to open the SF file <%s>\nwe terminate here!\n", buff);
	  endrun(LT_ERR_SF_FILE);
	}

      /* now read in the SF data.
       * lines beginning with a "#" are ignored.
       */

      fscanf(file, "%d\n", &SFs_dim);
      SFs = (SF_Type *) mymalloc("SFS", SFs_dim * sizeof(SF_Type));
      memset(SFs, 0, SFs_dim * sizeof(SF_Type));

      if(SFs == 0x0)
	{
	  printf("[Task 0] memory allocation failed when reading SFs' file\n");
	  endrun(LT_ERR_SF_ALLOCATE);
	}

      IsThere_TimeDep_SF = 0;
      IsThere_ZDep_SF = 0;

      i = 0;
      line_num = 0;
      SF_index = -1;
      SF_Nspec = 0;
      do
	{
	  charp = fgets(buff, 500, file);
	  line_num++;
	  if(charp != 0x0 && buff[0] != '#')
	    {
	      ((p = sscanf(buff, "%s %s", &param[0], &value[0])) >= 1) ? n++ : n;

	      if(strcmp(param, "NAME") == 0)
		{
		  SF_index++;
		  if(SF_index >= SFs_dim)
		    {
		      printf
			("something odd in the %s%s file: while %d SFs were expected, at least %d have been found\n",
			 All.OutputDir, All.SFfilename, SFs_dim, SF_index + 1);
		      endrun(123458);
		    }

		  SFs[SF_index].IMFi = -1;
		  SFs[SF_index].IMFname = 0x0;

		  SFs[SF_index].SFTh_Zdep = 0;

		  SFs[SF_index].MaxSfrTimescale = All.MaxSfrTimescale;
		  SFs[SF_index].MaxSfrTimescale_rescale_by_densityth = 1;

		  sprintf(buffer, "SF_id_%02d", SF_index);
		  SFs[SF_index].identifier = (char *) mymalloc(buffer, strlen(value) + 2);
		  sprintf(SFs[SF_index].identifier, "%s", value);

		  sprintf(buffer, "FEVP_%02d", SF_index);
		  SFs[SF_index].FEVP = (double *) mymalloc(buffer, sizeof(double));
		  memset(SFs[SF_index].FEVP, 0, sizeof(double));

		  sprintf(buffer, "PhysDensTh_%02d", SF_index);
		  SFs[SF_index].PhysDensThresh = (double *) mymalloc(buffer, sizeof(double));
		  memset(SFs[SF_index].PhysDensThresh, 0, sizeof(double));

#ifdef WINDS
		  SFs[SF_index].WindEnergyFraction = -1;
		  SFs[SF_index].WindEfficiency = -1;
#endif

		}
	      else if(strcmp(param, "IMFName") == 0)
		{
		  sprintf(buffer, "SF_IMFname_%02d", SF_index);
		  SFs[SF_index].IMFname = (char *) mymalloc(buffer, strlen(value) + 2);
		  sprintf(SFs[SF_index].IMFname, "%s", value);

		  for(j = 0; j < IMFs_dim; j++)
		    if(strcasecmp(IMFs[j].name, value) == 0)
		      break;
		  if(j < IMFs_dim)
		    SFs[SF_index].IMFi = j;
		  else
		    {
		      free(SFs[SF_index].IMFname);
		      printf("IMF <%s> does not exist\n", value);
		    }
		}
	      else if(strcmp(param, "IMFNum") == 0)
		{
		  SFs[SF_index].IMFi = atoi(value);
		  if(SFs[SF_index].IMFi >= IMFs_dim)
		    {
		      printf("only %d IMF(s) have been specified\n", IMFs_dim);
		      SFs[SF_index].IMFi = -1;
		    }
		  else
		    {
		      if(IMFs[SFs[SF_index].IMFi].name != 0x0)
			{
			  sprintf(buffer, "SF_IMFname_%02d", SF_index);
			  SFs[SF_index].IMFname =
			    (char *) mymalloc(buffer, strlen(IMFs[SFs[SF_index].IMFi].name) + 2);
			  sprintf(SFs[SF_index].IMFname, "%s", IMFs[SFs[SF_index].IMFi].name);
			}
		    }
		}
	      else if(strcmp(param, "MaxSfrTimescale") == 0)
		{
		  SF_Nspec++;
		  SFs[SF_index].MaxSfrTimescale = atof(value);
		}
	      else if(strcmp(param, "ZDEP") == 0)
		{
		  SF_Nspec++;
		  if(atoi(value) == 0)
		    SFs[SF_index].SFTh_Zdep = 0;
		  else
		    {
		      IsThere_ZDep_SF += (SFs[SF_index].SFTh_Zdep = 1);

		      myfree(SFs[SF_index].PhysDensThresh);
		      myfree(SFs[SF_index].FEVP);

		      sprintf(buffer, "FEVP_%02d", SF_index);
		      SFs[SF_index].FEVP = (double *) mymalloc(buffer, ZBins * sizeof(double));
		      memset(SFs[SF_index].FEVP, 0, ZBins * sizeof(double));

		      sprintf(buffer, "PhysDensTh_%02d", SF_index);
		      SFs[SF_index].PhysDensThresh = (double *) mymalloc(buffer, ZBins * sizeof(double));
		      memset(SFs[SF_index].PhysDensThresh, 0, ZBins * sizeof(double));

		    }
		}
	      else if(strcmp(param, "TDEP") == 0)
		{
		  SF_Nspec++;
		  if(atoi(value) == 0)
		    SFs[SF_index].SFTh_Tdep = 0;
		  else
		    {
		      IsThere_TimeDep_SF++;
		      SFs[SF_index].SFTh_Tdep = 1;
		    }
		}
	      else if(strcmp(param, "MaxSfrTimescale") == 0)
		{
		  SF_Nspec++;
		  SFs[SF_index].MaxSfrTimescale = atof(value);
		}
	      else if(strcmp(param, "Z4Th") == 0)
		{
		  SF_Nspec++;
		  SFs[SF_index].referenceZ_toset_SF_DensTh = atof(value);
		  if(SFs[SF_index].referenceZ_toset_SF_DensTh > 0)
		    SFs[SF_index].referenceZ_toset_SF_DensTh =
		      log10(SFs[SF_index].referenceZ_toset_SF_DensTh);
		  else
		    SFs[SF_index].referenceZ_toset_SF_DensTh = NO_METAL;
		}
	      else if(strcmp(param, "SFTh") == 0)
		{
		  SF_Nspec++;
		  SFs[SF_index].PhysDensThresh[0] = atof(value);
		}
	      else if(strcmp(param, "TSFR_rescale") == 0)
		{
		  SF_Nspec++;
		  SFs[SF_index].MaxSfrTimescale_rescale_by_densityth = atof(value);
		}
	      else if(strcmp(param, "EgyMTh") == 0)
		{
		  SF_Nspec++;
		  SFs[SF_index].egyShortLiv_MassTh = atof(value);
		}
	      else if(strcmp(param, "MetMTh") == 0)
		{
		  SF_Nspec++;
		  SFs[SF_index].metShortLiv_MassTh = atof(value);
		}
	      else if(strcmp(param, "NGen") == 0)
		{
		  SF_Nspec++;
		  SFs[SF_index].Generations = atoi(value);
		}
	      else if(strcmp(param, "WindEff") == 0)
		{
		  SF_Nspec++;
		  SFs[SF_index].WindEfficiency = atof(value);
		}
	      else if(strcmp(param, "WindEF") == 0)
		{
		  SF_Nspec++;
		  SFs[SF_index].WindEnergyFraction = atof(value);
		}
	      else
		{
		  printf("error in SFs' format file at line %d: unknown parameter\n", line_num);
		  endrun(LT_ERR_SF_FILE_FORMAT);
		}
	    }
	}			/* close the while */
      while(charp != 0x0);

      fclose(file);

      if(SF_index < SFs_dim - 1)
	{
	  printf("something odd in the %s file: while %d SFs were expefted, %d have been found\n",
		 buff, SFs_dim, SF_index);
	  endrun(123459);
	}

      for(SF_index = 0; SF_index < SFs_dim; SF_index++)
	{
	  /* check that key arguments have been set-up */
#ifdef WINDS
	  if(SFs[SF_index].WindEfficiency <= 0)
	    {
	      printf("SF %d :: WindEfficiency parameter is NOT allowed to be zero\n", SF_index);
	      fflush(stdout);
	      endrun(123456);
	    }
#ifndef LT_WIND_VELOCITY
	  if(SFs[SF_index].WindEnergyFraction <= 0)
	    {
	      printf("SF %d :: WindEenergyFraction parameter is NOT allowed to be either zero or not set\n",
		     SF_index);
	      fflush(stdout);
	      endrun(123457);
	    }
#else
	  if(SFs[SF_index].WindEnergyFraction <= 0)
	    printf("SF %d :: EnergyFraction will be set from the value of LT_WIND_VELOCITY (%8.6e)\n",
		   SF_index, LT_WIND_VELOCITY);
	  fflush(stdout);
#endif
#endif
	}


    }				/* close if(ThisTask == 0) */


  /* -------------------------------------------------------------- *
   * communicate IMFs data                                          *
   * -------------------------------------------------------------- */

  MPI_Bcast(&IMFs_dim, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&SFs_dim, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if(ThisTask != 0)
    {
      if((IMFs = (IMF_Type *) mymalloc("IMFS", IMFs_dim * sizeof(IMF_Type))) == 0x0)
	{
	  printf("[Task %d] memory allocation failed when communicating imfs' file\n", ThisTask);
	  endrun(LT_ERR_IMF_ALLOCATE);
	}
      if((SFs = (SF_Type *) mymalloc("SFS", SFs_dim * sizeof(SF_Type))) == 0x0)
	{
	  printf("[Task %d] memory allocation failed when reading SFs' file\n", ThisTask);
	  endrun(LT_ERR_SF_ALLOCATE);
	}
    }

  for(i = 0; i < IMFs_dim; i++)
    MPI_Bcast(&IMFs[i], sizeof(IMF_Type), MPI_BYTE, 0, MPI_COMM_WORLD);

  for(i = 0; i < SFs_dim; i++)
    MPI_Bcast(&SFs[i], sizeof(SF_Type), MPI_BYTE, 0, MPI_COMM_WORLD);

  if(ThisTask != 0)
    {
      for(i = 0; i < IMFs_dim; i++)
	{
	  sprintf(buffer, "IMF_param_%02d", i);
	  if((IMFs[i].NParams > 0)
	     && (IMFs[i].Params = (double *) mymalloc(buffer, IMFs[i].NParams * sizeof(double))) == 0x0)
	    {
	      printf("[Task %d][b] memory allocation failed when communicating MFs (%g MB requested)\n",
		     ThisTask, (double) IMFs[i].NParams / 1024 * sizeof(double) / 1024);
	      endrun(LT_ERR_IMF_ALLOCATE);
	    }
	  sprintf(buffer, "IMF_slopes_m_%02d", i);
	  IMFs[i].Slopes.masses = (double *) mymalloc(buffer, IMFs[i].NSlopes * sizeof(double));
	  sprintf(buffer, "IMF_slopes_s_%02d", i);
	  IMFs[i].Slopes.slopes = (double *) mymalloc(buffer, IMFs[i].NSlopes * sizeof(double));
	  sprintf(buffer, "IMF_A_%02d", i);
	  IMFs[i].A = (double *) mymalloc(buffer, IMFs[i].NSlopes * sizeof(double));
	  if(IMFs[i].Slopes.masses == 0x0 || IMFs[i].Slopes.slopes == 0x0 || IMFs[i].A == 0x0)
	    {
	      printf("[Task %d][c] memory allocation failed when communicating MFs (%g MB requested)\n",
		     ThisTask, (double) IMFs[i].NSlopes / 1024 * sizeof(double) / 1024);
	      endrun(LT_ERR_IMF_ALLOCATE);
	    }
	  sprintf(buffer, "IMF_notBH_s_%02d", i);
	  IMFs[i].notBH_ranges.sup =
	    (double *) mymalloc(buffer, (IMFs[i].N_notBH_ranges + 1) * sizeof(double));
	  sprintf(buffer, "IMF_notBH_i_%02d", i);
	  IMFs[i].notBH_ranges.inf =
	    (double *) mymalloc(buffer, (IMFs[i].N_notBH_ranges + 1) * sizeof(double));
	  sprintf(buffer, "IMF_notBH_l_%02d", i);
	  IMFs[i].notBH_ranges.list =
	    (double *) mymalloc(buffer, (IMFs[i].N_notBH_ranges) * 2 * sizeof(double));
	  if(IMFs[i].notBH_ranges.sup == 0x0 || IMFs[i].notBH_ranges.inf == 0x0
	     || IMFs[i].notBH_ranges.list == 0x0)
	    {
	      printf("[Task %d][d] memory allocation failed when communicating MFs (%g MB requested)\n",
		     ThisTask, (double) (IMFs[i].N_notBH_ranges + 1) / 1024 * sizeof(double) / 1024);
	      endrun(LT_ERR_IMF_ALLOCATE);
	    }
	  sprintf(buffer, "IMF_Ekin_m_%02d", i);
	  IMFs[i].EKin.masses = (double *) mymalloc(buffer, (IMFs[i].NEKin) * sizeof(double));
	  sprintf(buffer, "IMF_Ekin_%02d", i);
	  IMFs[i].EKin.ekin = (double *) mymalloc(buffer, (IMFs[i].NEKin) * sizeof(double));
	  if(IMFs[i].EKin.masses == 0x0 || IMFs[i].EKin.ekin == 0x0)
	    {
	      printf("[Task %d][e] memory allocation failed when communicating MFs (%g MB requested)\n",
		     ThisTask, (double) IMFs[i].NEKin / 1024 * sizeof(double) / 1024);
	      endrun(LT_ERR_IMF_ALLOCATE);
	    }
	}

      for(i = 0; i < SFs_dim; i++)
	{
	  if(SFs[i].SFTh_Zdep)
	    {
	      sprintf(buffer, "SF_PhysDensTh_%02d", i);
	      if((SFs[i].PhysDensThresh = (double *) mymalloc(buffer, ZBins * sizeof(double))) == 0x0)
		{
		  printf("[Task %d][a] memory allocation failed when communicating SFfs (%g MB requested)\n",
			 ThisTask, (double) ZBins / 1024 * sizeof(double) / 1024);
		  endrun(LT_ERR_SF_ALLOCATE);
		}
	      memset(SFs[i].PhysDensThresh, 0, ZBins * sizeof(double));
	      sprintf(buffer, "SF_FEVP_%02d", i);
	      if((SFs[i].FEVP = (double *) mymalloc(buffer, ZBins * sizeof(double))) == 0x0)
		{
		  printf("[Task %d][a1] memory allocation failed when communicating imfs (%g MB requested)\n",
			 ThisTask, (double) ZBins / 1024 * sizeof(double) / 1024);
		  endrun(LT_ERR_SF_ALLOCATE);
		}
	      memset(SFs[i].FEVP, 0, ZBins * sizeof(double));
	    }
	  else
	    {
	      sprintf(buffer, "SF_PhysDensTh_%02d", i);
	      if((SFs[i].PhysDensThresh = (double *) mymalloc(buffer, sizeof(double))) == 0x0)
		{
		  printf("[Task %d][a] memory allocation failed when communicating SFs (%g MB requested)\n",
			 ThisTask, (double) ZBins / 1024 * sizeof(double) / 1024);
		  endrun(LT_ERR_SF_ALLOCATE);
		}
	      memset(SFs[i].PhysDensThresh, 0, sizeof(double));
	      sprintf(buffer, "SF_FEVP_%02d", i);
	      if((SFs[i].FEVP = (double *) mymalloc(buffer, sizeof(double))) == 0x0)
		{
		  printf("[Task %d][a1] memory allocation failed when communicating SFs  (%g MB requested)\n",
			 ThisTask, (double) ZBins / 1024 * sizeof(double) / 1024);
		  endrun(LT_ERR_SF_ALLOCATE);
		}
	      memset(SFs[i].FEVP, 0, sizeof(double));
	    }

	}
    }

  MPI_Barrier(MPI_COMM_WORLD);

  for(i = 0; i < IMFs_dim; i++)
    {
      MPI_Bcast(&IMFs[i].Params[0], sizeof(double) * IMFs[i].NParams, MPI_BYTE, 0, MPI_COMM_WORLD);
      MPI_Bcast(&IMFs[i].A[0], sizeof(double) * IMFs[i].NSlopes, MPI_BYTE, 0, MPI_COMM_WORLD);
      MPI_Bcast(&IMFs[i].Slopes.slopes[0], sizeof(double) * IMFs[i].NSlopes, MPI_BYTE, 0, MPI_COMM_WORLD);
      MPI_Bcast(&IMFs[i].Slopes.masses[0], sizeof(double) * IMFs[i].NSlopes, MPI_BYTE, 0, MPI_COMM_WORLD);
      MPI_Bcast(&IMFs[i].notBH_ranges.sup[0], sizeof(double) * IMFs[i].N_notBH_ranges, MPI_BYTE, 0,
		MPI_COMM_WORLD);
      MPI_Bcast(&IMFs[i].notBH_ranges.inf[0], sizeof(double) * IMFs[i].N_notBH_ranges, MPI_BYTE, 0,
		MPI_COMM_WORLD);
      MPI_Bcast(&IMFs[i].notBH_ranges.list[0], sizeof(double) * 2 * IMFs[i].N_notBH_ranges, MPI_BYTE, 0,
		MPI_COMM_WORLD);
      MPI_Bcast(&IMFs[i].EKin.masses[0], sizeof(double) * (IMFs[i].NEKin), MPI_BYTE, 0, MPI_COMM_WORLD);
      MPI_Bcast(&IMFs[i].EKin.ekin[0], sizeof(double) * (IMFs[i].NEKin), MPI_BYTE, 0, MPI_COMM_WORLD);
    }

  for(i = 0; i < SFs_dim; i++)
    {
      if(SFs[i].SFTh_Zdep > 0)
	MPI_Bcast(&SFs[i].PhysDensThresh[0], sizeof(double) * ZBins, MPI_BYTE, 0, MPI_COMM_WORLD);
      else
	MPI_Bcast(&SFs[i].PhysDensThresh[0], sizeof(double), MPI_BYTE, 0, MPI_COMM_WORLD);
    }

  if(ThisTask == 0)
    {
      printf("%d IMF%c read\n", IMFs_dim, (IMFs_dim > 1) ? 's' : ' ');
      printf("%d SF%c read\n", SFs_dim, (SFs_dim > 1) ? 's' : ' ');
      fflush(stdout);
    }

  return 0;
}

/* ============================================================
 *
 *     ---=={[  IMF I/O section  ]}==---
 * ============================================================ */

void write_SF_info(int num, FILE * file)
{
  fprintf(file,
	  "::  SF %3d\n"
	  "    name             : %-10s\n"
	  "    SFR timescale    : %-6.4e\n"
	  "    # Generations    : %-d\n",
	  num, SFs[num].identifier, SFs[num].MaxSfrTimescale, SFs[num].Generations);

  if(SFs[num].IMFname != 0x0)
    fprintf(file, "    IMF              : %-s\n", SFs[num].IMFname);
  else
    fprintf(file, "    IMF              : %-d\n", SFs[num].IMFi);

  fprintf(file, "    Winds Efficiency : %-6.4e\n", SFs[num].WindEfficiency);

  fprintf(file, "\n");
  fflush(file);
  return;
}


/* ============================================================
 *
 *     ---=={[  SF selection section  ]}==---
 * ============================================================ */

/* here place your own code for selecting IMF depending on gas
   physical conditions

   prototype:
   int INLINE_FUNC Get_IMF_index(int i)

   where i is the ordinal number of particle in the P and SphP
   structures

   returned integer is the ordinal number of the IMF
*/


/* no selection, only one IMF active */

#ifndef LT_STARBURSTS
#ifndef LT_POPIII_FLAGS
int INLINE_FUNC get_SF_index(int i, int *mySFi, int *IMFi)
{
  *mySFi = 0;
  *IMFi = SFs[0].IMFi;
  return *mySFi;
}

#else
/* POPIII-POPII selection */
int INLINE_FUNC get_SF_index(int i, int *mySFi, int *IMFi)
{
#define INV_SOLAR 50
  int j;
  float *Zs;
  double Z;

#if !defined(LT_SMOOTH_Z) || (defined(LT_SMOOTH_Z) && !defined(LT_SMOOTHZ_IN_SF_SWITCH))
  if(P[i].Type & 4)
    Zs = &MetP[P[i].MetID].Metals[0];
  else
    Zs = &SphP[i].Metals[0];

  for(Z = 0, j = 0; j < LT_NMetP; j++)
    if(j != Hel)
      Z += Zs[j];
  Z /= (P[i].Mass - Z - Zs[Hel]);
#else
  if(P[i].Type & 4)
    Z = MetP[P[i].MetID].Zsmooth;
  else
    Z = SphP[i].Zsmooth;
#endif


/*   int j; */
/*   float *Zs; */

/*   if(P[i].Type & 4) */
/*     Zs = &MetP[P[i].MetID].Metals[0]; */
/*   else */
/*     Zs = &SphP[i].Metals[0]; */

/*   for(Z = 0, j = 0; j < LT_NMetP; j++) */
/*     if(j != Hel) */
/*       Z += Zs[j]; */
/*   Z /= (P[i].Mass - Z - Zs[Hel]); */

/* #ifdef LT_SMOOTH_Z */
/*   if(Z > 0) */
/*     { */
/*       if(P[i].Type & 4) */
/*         Z = (Z + MetP[P[i].MetID].Zsmooth) / 2; */
/*       else */
/*         Z = (Z + SphP[i].Zsmooth) / 2; */
/*     } */
/* #endif   */

  if(Z * INV_SOLAR < All.PopIII_Zlimit)
    *mySFi = All.PopIII_IMF_idx;
  else
    *mySFi = 0;

/*   if(Z * INV_SOLAR > 1e-2) */
/*     return 0; */
/*   else */
/*     return 1; */

  *IMFi = SFs[*mySFi].IMFi;
  return *mySFi;
}
#endif

#else
/* Star Burst selection */
int INLINE_FUNC get_SF_index(int i)
{
  switch (All.StarBurstCondition)
    {
    case SB_DENSITY:
      if(SphP[i].a2.Density < All.SB_Density_Thresh)
	*mySFi = 0;
      else
	*mySFi = 1;
      break;

    case SB_DENTROPY:
      if(SphP[i].e.DtEntropy < All.SB_DEntropy_Thresh)
	*mySFi = 0;
      else
	*mySFi = 1;
      break;

    case (SB_DENSITY + SB_DENTROPY):
      if(SphP[i].a2.Density >= All.SB_Density_Thresh && SphP[i].e.DtEntropy >= All.SB_DEntropy_Thresh)
	*mySFi = 1;
      else
	*mySFi = 0;
      break;
    }
  *IMFi = SFs[*mySFi].IMFi;
  return *mySFi;
}
#endif


/* ============================================================
 *
 *     ---=={[  PACKING section  ]}==---
 * ============================================================ */



#ifdef LT_TRACK_CONTRIBUTES

void init_packing()
{
  int i;

  MPI_Bcast(&PowerBase, 1, MPI_INT, 0, MPI_COMM_WORLD);

  Packing_Factor = (1 << LT_Nbits) - 1;
  MaxError = (1.0 / Packing_Factor * 3 * SFs_dim) * 2.0;
  UnPacking_Factor = (float) (1.0 / Packing_Factor);

  Max_Power = (1 << LT_power10_Nbits) - 1;

  Power_Factors = (unsigned int *) mymalloc("PowerFactors", (Max_Power + 1) * sizeof(int));
  memset(Power_Factors, 0, (Max_Power + 1) * sizeof(int));
  Power_Factors[0] = 1;
  for(i = 1; i <= Max_Power; i++)
    Power_Factors[i] = Power_Factors[i - 1] * PowerBase;

  MinPackableFraction = UnPacking_Factor / Power_Factors[Max_Power];
  MaxRaisableFraction = 1.0 / PowerBase;
  PowerBaseLog_inv = 1.0 / log((double) PowerBase);

  if(ThisTask == 0)
    {
      fprintf(FdSnInit, ">> packing structure has %u bytes\nMaxError is %8g\n",
	      (unsigned int) sizeof(Contrib), MaxError);
      printf(">> packing structure has %u bytes\nMaxError is %8g\n",
	     (unsigned int) sizeof(Contrib), MaxError);
    }

  save_fractionII = (float *) mymalloc("save_fractions", sizeof(float) * (LT_NMetP * SFs_dim) * 15);

  save_fractionIa = save_fractionII + (LT_NMetP * SFs_dim);
  save_fractionAGB = save_fractionIa + (LT_NMetP * SFs_dim);

  fractionII = save_fractionAGB + (LT_NMetP * SFs_dim);
  fractionIa = fractionII + (LT_NMetP * SFs_dim);
  fractionAGB = fractionIa + (LT_NMetP * SFs_dim);

  nfractionII = fractionAGB + (LT_NMetP * SFs_dim);
  nfractionIa = nfractionII + (LT_NMetP * SFs_dim);
  nfractionAGB = nfractionIa + (LT_NMetP * SFs_dim);

  dfractionII = (double *) (nfractionAGB + (LT_NMetP * SFs_dim));
  dfractionIa = dfractionII + (LT_NMetP * SFs_dim);
  dfractionAGB = dfractionIa + (LT_NMetP * SFs_dim);

  return;
}

int INLINE_FUNC get_packing_power(float f)
{
  int i;

  if(f < MinPackableFraction)
    return 0;
  if(f > MaxRaisableFraction)
    return 0;

  for(i = 1; i <= Max_Power; i++)
    if((f *= PowerBase) >= MaxRaisableFraction)
      break;
  return i;

  /*
     f*b^n = 1;
     n = log_b(1/f)
   */
  /*
     if((power = (int)trunc(log(1/f)*PowerBaseLog)) > MaxPower)
     return MaxPower;
     else
     return power;
   */
}

void INLINE_FUNC pack_contrib(Contrib * contrib, int IMFi, float *II, float *Ia, float *AGB)
{
  /* example using LT_NMet = 9 */

  if(IMFi == 0)
    {
      /* Massive stars */
/*       contrib->II_el0_imf0 = II[0] * */
/* 	Power_Factors[contrib->IIexp_el0_imf0 = get_packing_power(II[0])] * */
/* 	Packing_Factor; */
      contrib->II_el1_imf0 = II[1] *
	Power_Factors[contrib->IIexp_el1_imf0 = get_packing_power(II[1])] * Packing_Factor;
      contrib->II_el2_imf0 = II[2] *
	Power_Factors[contrib->IIexp_el2_imf0 = get_packing_power(II[2])] * Packing_Factor;
      contrib->II_el3_imf0 = II[3] *
	Power_Factors[contrib->IIexp_el3_imf0 = get_packing_power(II[3])] * Packing_Factor;
      contrib->II_el4_imf0 = II[4] *
	Power_Factors[contrib->IIexp_el4_imf0 = get_packing_power(II[4])] * Packing_Factor;
      contrib->II_el5_imf0 = II[5] *
	Power_Factors[contrib->IIexp_el5_imf0 = get_packing_power(II[5])] * Packing_Factor;
      contrib->II_el6_imf0 = II[6] *
	Power_Factors[contrib->IIexp_el6_imf0 = get_packing_power(II[6])] * Packing_Factor;
      contrib->II_el7_imf0 = II[7] *
	Power_Factors[contrib->IIexp_el7_imf0 = get_packing_power(II[7])] * Packing_Factor;
/*       contrib->II_el8_imf0 = II[8] * */
/* 	Power_Factors[contrib->IIexp_el8_imf0 = get_packing_power(II[8])] * */
/* 	Packing_Factor; */

      /* Ia supernovae */
/*       contrib->Ia_el0_imf0 = Ia[0] * */
/* 	Power_Factors[contrib->Iaexp_el0_imf0 = get_packing_power(Ia[0])] * */
/* 	Packing_Factor; */
      contrib->Ia_el1_imf0 = Ia[1] *
	Power_Factors[contrib->Iaexp_el1_imf0 = get_packing_power(Ia[1])] * Packing_Factor;
      contrib->Ia_el2_imf0 = Ia[2] *
	Power_Factors[contrib->Iaexp_el2_imf0 = get_packing_power(Ia[2])] * Packing_Factor;
      contrib->Ia_el3_imf0 = Ia[3] *
	Power_Factors[contrib->Iaexp_el3_imf0 = get_packing_power(Ia[3])] * Packing_Factor;
      contrib->Ia_el4_imf0 = Ia[4] *
	Power_Factors[contrib->Iaexp_el4_imf0 = get_packing_power(Ia[4])] * Packing_Factor;
      contrib->Ia_el5_imf0 = Ia[5] *
	Power_Factors[contrib->Iaexp_el5_imf0 = get_packing_power(Ia[5])] * Packing_Factor;
      contrib->Ia_el6_imf0 = Ia[6] *
	Power_Factors[contrib->Iaexp_el6_imf0 = get_packing_power(Ia[6])] * Packing_Factor;
      contrib->Ia_el7_imf0 = Ia[7] *
	Power_Factors[contrib->Iaexp_el7_imf0 = get_packing_power(Ia[7])] * Packing_Factor;
/*       contrib->Ia_el8_imf0 = Ia[8] * */
/* 	Power_Factors[contrib->Iaexp_el8_imf0 = get_packing_power(Ia[8])] * */
/* 	Packing_Factor; */

      /* AGB stars */
/*       contrib->AGB_el0_imf0 = AGB[0] * */
/* 	Power_Factors[contrib->AGBexp_el0_imf0 = get_packing_power(AGB[0])] * */
/* 	Packing_Factor; */
      contrib->AGB_el1_imf0 = AGB[1] *
	Power_Factors[contrib->AGBexp_el1_imf0 = get_packing_power(AGB[1])] * Packing_Factor;
      contrib->AGB_el2_imf0 = AGB[2] *
	Power_Factors[contrib->AGBexp_el2_imf0 = get_packing_power(AGB[2])] * Packing_Factor;
      contrib->AGB_el3_imf0 = AGB[3] *
	Power_Factors[contrib->AGBexp_el3_imf0 = get_packing_power(AGB[3])] * Packing_Factor;
      contrib->AGB_el4_imf0 = AGB[4] *
	Power_Factors[contrib->AGBexp_el4_imf0 = get_packing_power(AGB[4])] * Packing_Factor;
      contrib->AGB_el5_imf0 = AGB[5] *
	Power_Factors[contrib->AGBexp_el5_imf0 = get_packing_power(AGB[5])] * Packing_Factor;
      contrib->AGB_el6_imf0 = AGB[6] *
	Power_Factors[contrib->AGBexp_el6_imf0 = get_packing_power(AGB[6])] * Packing_Factor;
      contrib->AGB_el7_imf0 = AGB[7] *
	Power_Factors[contrib->AGBexp_el7_imf0 = get_packing_power(AGB[7])] * Packing_Factor;
/*       contrib->AGB_el8_imf0 = AGB[8] * */
/* 	Power_Factors[contrib->AGBexp_el8_imf0 = get_packing_power(AGB[8])] * */
/* 	Packing_Factor; */
    }
  /* Add here below more blocks for additional IMFs
   *
   * else if(IMFi == ..)
   */
/*   else */
/*     { */
/* /\*       contrib->II_el0_imf1 = II[0] * *\/ */
/* /\* 	Power_Factors[contrib->IIexp_el0_imf1 = get_packing_power(II[0])] * *\/ */
/* /\* 	Packing_Factor; *\/ */
/*       contrib->II_el1_imf1 = II[1] * */
/* 	Power_Factors[contrib->IIexp_el1_imf1 = get_packing_power(II[1])] * */
/* 	Packing_Factor; */
/*       contrib->II_el2_imf1 = II[2] * */
/* 	Power_Factors[contrib->IIexp_el2_imf1 = get_packing_power(II[2])] * */
/* 	Packing_Factor; */
/*       contrib->II_el3_imf1 = II[3] * */
/* 	Power_Factors[contrib->IIexp_el3_imf1 = get_packing_power(II[3])] * */
/* 	Packing_Factor; */
/*       contrib->II_el4_imf1 = II[4] * */
/* 	Power_Factors[contrib->IIexp_el4_imf1 = get_packing_power(II[4])] * */
/* 	Packing_Factor; */
/*       contrib->II_el5_imf1 = II[5] * */
/* 	Power_Factors[contrib->IIexp_el5_imf1 = get_packing_power(II[5])] * */
/* 	Packing_Factor; */
/*       contrib->II_el6_imf1 = II[6] * */
/* 	Power_Factors[contrib->IIexp_el6_imf1 = get_packing_power(II[6])] * */
/* 	Packing_Factor; */
/*       contrib->II_el7_imf1 = II[7] * */
/* 	Power_Factors[contrib->IIexp_el7_imf1 = get_packing_power(II[7])] * */
/* 	Packing_Factor; */
/* /\*       contrib->II_el8_imf1 = II[8] * *\/ */
/* /\* 	Power_Factors[contrib->IIexp_el8_imf1 = get_packing_power(II[8])] * *\/ */
/* /\* 	Packing_Factor; *\/ */

/* /\*       contrib->Ia_el1_imf1 = Ia[1] * *\/ */
/* /\* 	Power_Factors[contrib->Iaexp_el1_imf1 = get_packing_power(Ia[1])] * *\/ */
/* /\* 	Packing_Factor; *\/ */
/* /\*       contrib->Ia_el2_imf1 = Ia[2] * *\/ */
/* /\* 	Power_Factors[contrib->Iaexp_el2_imf1 = get_packing_power(Ia[2])] * *\/ */
/* /\* 	Packing_Factor; *\/ */
/* /\*       contrib->Ia_el3_imf1 = Ia[3] * *\/ */
/* /\* 	Power_Factors[contrib->Iaexp_el3_imf1 = get_packing_power(Ia[3])] * *\/ */
/* /\* 	Packing_Factor; *\/ */
/* /\*       contrib->Ia_el4_imf1 = Ia[4] * *\/ */
/* /\* 	Power_Factors[contrib->Iaexp_el4_imf1 = get_packing_power(Ia[4])] * *\/ */
/* /\* 	Packing_Factor; *\/ */
/* /\*       contrib->Ia_el5_imf1 = Ia[5] * *\/ */
/* /\* 	Power_Factors[contrib->Iaexp_el5_imf1 = get_packing_power(Ia[5])] * *\/ */
/* /\* 	Packing_Factor; *\/ */
/* /\*       contrib->Ia_el6_imf1 = Ia[6] * *\/ */
/* /\* 	Power_Factors[contrib->Iaexp_el6_imf1 = get_packing_power(Ia[6])] * *\/ */
/* /\* 	Packing_Factor; *\/ */
/* /\*       contrib->Ia_el7_imf1 = Ia[7] * *\/ */
/* /\* 	Power_Factors[contrib->Iaexp_el7_imf1 = get_packing_power(Ia[7])] * *\/ */
/* /\* 	Packing_Factor; *\/ */

/* /\*       contrib->AGB_el1_imf1 = AGB[1] * *\/ */
/* /\* 	Power_Factors[contrib->AGBexp_el1_imf1 = get_packing_power(AGB[1])] * *\/ */
/* /\* 	Packing_Factor; *\/ */
/* /\*       contrib->AGB_el2_imf1 = AGB[2] * *\/ */
/* /\* 	Power_Factors[contrib->AGBexp_el2_imf1 = get_packing_power(AGB[2])] * *\/ */
/* /\* 	Packing_Factor; *\/ */
/* /\*       contrib->AGB_el3_imf1 = AGB[3] * *\/ */
/* /\* 	Power_Factors[contrib->AGBexp_el3_imf1 = get_packing_power(AGB[3])] * *\/ */
/* /\* 	Packing_Factor; *\/ */
/* /\*       contrib->AGB_el4_imf1 = AGB[4] * *\/ */
/* /\* 	Power_Factors[contrib->AGBexp_el4_imf1 = get_packing_power(AGB[4])] * *\/ */
/* /\* 	Packing_Factor; *\/ */
/* /\*       contrib->AGB_el5_imf1 = AGB[5] * *\/ */
/* /\* 	Power_Factors[contrib->AGBexp_el5_imf1 = get_packing_power(AGB[5])] * *\/ */
/* /\* 	Packing_Factor; *\/ */
/* /\*       contrib->AGB_el6_imf1 = AGB[6] * *\/ */
/* /\* 	Power_Factors[contrib->AGBexp_el6_imf1 = get_packing_power(AGB[6])] * *\/ */
/* /\* 	Packing_Factor; *\/ */
/* /\*       contrib->AGB_el7_imf1 = AGB[7] * *\/ */
/* /\* 	Power_Factors[contrib->AGBexp_el7_imf1 = get_packing_power(AGB[7])] * *\/ */
/* /\* 	Packing_Factor; *\/ */
/*      } */
  return;
}


void INLINE_FUNC unpack_contrib(Contrib * contrib, float *II, float *Ia, float *AGB)
{
  /* example using LT_NMet = 9 */

  /* Massive stars */
  //II[0] = contrib->II_el0_imf0 * UnPacking_Factor / Power_Factors[(int) contrib->IIexp_el0_imf0];
  II[1] = contrib->II_el1_imf0 * UnPacking_Factor / Power_Factors[(int) contrib->IIexp_el1_imf0];
  II[2] = contrib->II_el2_imf0 * UnPacking_Factor / Power_Factors[(int) contrib->IIexp_el2_imf0];
  II[3] = contrib->II_el3_imf0 * UnPacking_Factor / Power_Factors[(int) contrib->IIexp_el3_imf0];
  II[4] = contrib->II_el4_imf0 * UnPacking_Factor / Power_Factors[(int) contrib->IIexp_el4_imf0];
  II[5] = contrib->II_el5_imf0 * UnPacking_Factor / Power_Factors[(int) contrib->IIexp_el5_imf0];
  II[6] = contrib->II_el6_imf0 * UnPacking_Factor / Power_Factors[(int) contrib->IIexp_el6_imf0];
  II[7] = contrib->II_el7_imf0 * UnPacking_Factor / Power_Factors[(int) contrib->IIexp_el7_imf0];
/*   II[8] = contrib->II_el8_imf0 * UnPacking_Factor / Power_Factors[(int) contrib->IIexp_el8_imf0]; */

  /* Ia supernovae */
  //Ia[0] = contrib->Ia_el0_imf0 * UnPacking_Factor / Power_Factors[(int) contrib->Iaexp_el0_imf0];
  Ia[1] = contrib->Ia_el1_imf0 * UnPacking_Factor / Power_Factors[(int) contrib->Iaexp_el1_imf0];
  Ia[2] = contrib->Ia_el2_imf0 * UnPacking_Factor / Power_Factors[(int) contrib->Iaexp_el2_imf0];
  Ia[3] = contrib->Ia_el3_imf0 * UnPacking_Factor / Power_Factors[(int) contrib->Iaexp_el3_imf0];
  Ia[4] = contrib->Ia_el4_imf0 * UnPacking_Factor / Power_Factors[(int) contrib->Iaexp_el4_imf0];
  Ia[5] = contrib->Ia_el5_imf0 * UnPacking_Factor / Power_Factors[(int) contrib->Iaexp_el5_imf0];
  Ia[6] = contrib->Ia_el6_imf0 * UnPacking_Factor / Power_Factors[(int) contrib->Iaexp_el6_imf0];
  Ia[7] = contrib->Ia_el7_imf0 * UnPacking_Factor / Power_Factors[(int) contrib->Iaexp_el7_imf0];
/*   Ia[8] = contrib->Ia_el8_imf0 * UnPacking_Factor / Power_Factors[(int) contrib->Iaexp_el8_imf0]; */

  /* AGB stars */
  //AGB[0] = contrib->AGB_el0_imf0 * UnPacking_Factor / Power_Factors[(int) contrib->AGBexp_el0_imf0];
  AGB[1] = contrib->AGB_el1_imf0 * UnPacking_Factor / Power_Factors[(int) contrib->AGBexp_el1_imf0];
  AGB[2] = contrib->AGB_el2_imf0 * UnPacking_Factor / Power_Factors[(int) contrib->AGBexp_el2_imf0];
  AGB[3] = contrib->AGB_el3_imf0 * UnPacking_Factor / Power_Factors[(int) contrib->AGBexp_el3_imf0];
  AGB[4] = contrib->AGB_el4_imf0 * UnPacking_Factor / Power_Factors[(int) contrib->AGBexp_el4_imf0];
  AGB[5] = contrib->AGB_el5_imf0 * UnPacking_Factor / Power_Factors[(int) contrib->AGBexp_el5_imf0];
  AGB[6] = contrib->AGB_el6_imf0 * UnPacking_Factor / Power_Factors[(int) contrib->AGBexp_el6_imf0];
  AGB[7] = contrib->AGB_el7_imf0 * UnPacking_Factor / Power_Factors[(int) contrib->AGBexp_el7_imf0];
/*   AGB[8] = contrib->AGB_el8_imf0 * UnPacking_Factor / Power_Factors[(int) contrib->AGBexp_el8_imf0]; */

  /* Add here below more blocks for additional IMFs; elements will be addressed
   * as lt_NMetP + N_IMF*#_ELEMENT, e.g. for the 2nd IMF the 1st element will
   * have address 9 = 8 + 1*0 etc.
   */
  /* Massive stars */

/*   //II[LT_NMetP + 0] = contrib->II_el0_imf1 * UnPacking_Factor / Power_Factors[(int) contrib->IIexp_el0_imf1]; */
/*   II[LT_NMetP + 1] = contrib->II_el1_imf1 * UnPacking_Factor / Power_Factors[(int) contrib->IIexp_el1_imf1]; */
/*   II[LT_NMetP + 2] = contrib->II_el2_imf1 * UnPacking_Factor / Power_Factors[(int) contrib->IIexp_el2_imf1]; */
/*   II[LT_NMetP + 3] = contrib->II_el3_imf1 * UnPacking_Factor / Power_Factors[(int) contrib->IIexp_el3_imf1]; */
/*   II[LT_NMetP + 4] = contrib->II_el4_imf1 * UnPacking_Factor / Power_Factors[(int) contrib->IIexp_el4_imf1]; */
/*   II[LT_NMetP + 5] = contrib->II_el5_imf1 * UnPacking_Factor / Power_Factors[(int) contrib->IIexp_el5_imf1]; */
/*   II[LT_NMetP + 6] = contrib->II_el6_imf1 * UnPacking_Factor / Power_Factors[(int) contrib->IIexp_el6_imf1]; */
/*   II[LT_NMetP + 7] = contrib->II_el7_imf1 * UnPacking_Factor / Power_Factors[(int) contrib->IIexp_el7_imf1]; */
/*   //II[LT_NMetP + 8] = contrib->II_el8_imf1 * UnPacking_Factor / Power_Factors[(int) contrib->IIexp_el8_imf1]; */

/*   Ia[LT_NMetP + 1] = contrib->Ia_el1_imf1 * UnPacking_Factor / Power_Factors[(int) contrib->Iaexp_el1_imf1]; */
/*   Ia[LT_NMetP + 2] = contrib->Ia_el2_imf1 * UnPacking_Factor / Power_Factors[(int) contrib->Iaexp_el2_imf1]; */
/*   Ia[LT_NMetP + 3] = contrib->Ia_el3_imf1 * UnPacking_Factor / Power_Factors[(int) contrib->Iaexp_el3_imf1]; */
/*   Ia[LT_NMetP + 4] = contrib->Ia_el4_imf1 * UnPacking_Factor / Power_Factors[(int) contrib->Iaexp_el4_imf1]; */
/*   Ia[LT_NMetP + 5] = contrib->Ia_el5_imf1 * UnPacking_Factor / Power_Factors[(int) contrib->Iaexp_el5_imf1]; */
/*   Ia[LT_NMetP + 6] = contrib->Ia_el6_imf1 * UnPacking_Factor / Power_Factors[(int) contrib->Iaexp_el6_imf1]; */
/*   Ia[LT_NMetP + 7] = contrib->Ia_el7_imf1 * UnPacking_Factor / Power_Factors[(int) contrib->Iaexp_el7_imf1]; */

/*   AGB[LT_NMetP + 1] = contrib->AGB_el1_imf1 * UnPacking_Factor / Power_Factors[(int) contrib->AGBexp_el1_imf1]; */
/*   AGB[LT_NMetP + 2] = contrib->AGB_el2_imf1 * UnPacking_Factor / Power_Factors[(int) contrib->AGBexp_el2_imf1]; */
/*   AGB[LT_NMetP + 3] = contrib->AGB_el3_imf1 * UnPacking_Factor / Power_Factors[(int) contrib->AGBexp_el3_imf1]; */
/*   AGB[LT_NMetP + 4] = contrib->AGB_el4_imf1 * UnPacking_Factor / Power_Factors[(int) contrib->AGBexp_el4_imf1]; */
/*   AGB[LT_NMetP + 5] = contrib->AGB_el5_imf1 * UnPacking_Factor / Power_Factors[(int) contrib->AGBexp_el5_imf1]; */
/*   AGB[LT_NMetP + 6] = contrib->AGB_el6_imf1 * UnPacking_Factor / Power_Factors[(int) contrib->AGBexp_el6_imf1]; */
/*   AGB[LT_NMetP + 7] = contrib->AGB_el7_imf1 * UnPacking_Factor / Power_Factors[(int) contrib->AGBexp_el7_imf1]; */


}



void update_contrib(Contrib * current_contrib, float *current_metals, Contrib * new_contrib,
		    float *new_metals)
     /*

      */
{

  double inv_total, Sum, delta;
  int i, j;

  memset((void *) save_fractionII, 0, sizeof(float) * LT_NMetP * SFs_dim * 15);

  /* gets current fractional contributes */
  unpack_contrib(current_contrib, fractionII, fractionIa, fractionAGB);
  unpack_contrib(new_contrib, nfractionII, nfractionIa, nfractionAGB);

  memcpy(save_fractionII, fractionII, LT_NMetP * SFs_dim * sizeof(float));
  memcpy(save_fractionIa, fractionIa, LT_NMetP * SFs_dim * sizeof(float));
  memcpy(save_fractionAGB, fractionAGB, LT_NMetP * SFs_dim * sizeof(float));

  /* updates the fractional contributes */
  for(j = 0; j < SFs_dim; j++)
    {
      for(i = 1; i < LT_NMetP; i++)
	{
	  if(current_metals[i] + new_metals[i] > 0)
	    {
	      inv_total = 1.0 / ((double) current_metals[i] + (double) new_metals[i]);

	      dfractionII[j * LT_NMetP + i] =
		((double) fractionII[j * LT_NMetP + i] * (double) current_metals[i] +
		 (double) nfractionII[j * LT_NMetP + i] * (double) new_metals[i]) * inv_total;

	      dfractionIa[j * LT_NMetP + i] =
		((double) fractionIa[j * LT_NMetP + i] * (double) current_metals[i] +
		 (double) nfractionIa[j * LT_NMetP + i] * (double) new_metals[i]) * inv_total;

	      dfractionAGB[j * LT_NMetP + i] =
		((double) fractionAGB[j * LT_NMetP + i] * (double) current_metals[i] +
		 (double) nfractionAGB[j * LT_NMetP + i] * (double) new_metals[i]) * inv_total;
	    }
	  else
	    dfractionII[j * LT_NMetP + i] =
	      dfractionIa[j * LT_NMetP + i] = dfractionAGB[j * LT_NMetP + i] = 0;
	}
    }

  for(i = 1; i < LT_NMetP; i++)
    {
      Sum = 0;
      for(j = 0; j < SFs_dim; j++)
	{
	  Sum += (fractionII[j * LT_NMetP + i] = (float) dfractionII[j * LT_NMetP + i]);
	  Sum += (fractionIa[j * LT_NMetP + i] = (float) dfractionIa[j * LT_NMetP + i]);
	  Sum += (fractionAGB[j * LT_NMetP + i] = (float) dfractionAGB[j * LT_NMetP + i]);
	}

      delta = fabs(Sum - 1);
/*       if( i != Hel && */
/*           i != FillEl && */
/*           Sum > 1e-3 && */
/*           delta > 0) */
      if(Sum > MinPackableFraction && delta > 0)
	{
	  if(delta > MaxError)
	    {
	      printf("[%d] An error in packing contributes from supernovae occurred (Sum %g).\n"
		     "element: %d\n"
		     "current: %g, new: %g\n", ThisTask, Sum, i, current_metals[i], new_metals[i]);
	      for(j = 0; j < SFs_dim; j++)
		printf("[imf %2d]fsII: %10g  fsIa: %10g fsAGB: %10g\n"
		       "         fII : %10g  fIa : %10g fAGB : %10g\n"
		       "         fnII: %10g  fnIa: %10g fnAGB: %10g\n", j,
		       save_fractionII[j * LT_NMetP + i],
		       save_fractionIa[j * LT_NMetP + i],
		       save_fractionAGB[j * LT_NMetP + i],
		       fractionII[j * LT_NMetP + i],
		       fractionIa[j * LT_NMetP + i],
		       fractionAGB[j * LT_NMetP + i],
		       nfractionII[j * LT_NMetP + i],
		       nfractionIa[j * LT_NMetP + i], nfractionAGB[j * LT_NMetP + i]);
	      printf("\n\n");
	      for(i = 1; i < LT_NMetP; i++)
		{
		  printf("current: %10.8e   new: %10.8e\n", current_metals[i], new_metals[i]);
		  for(j = 0; j < SFs_dim; j++)
		    printf("[imf %2d]fsII: %10g  fsIa: %10g fsAGB: %10g\n"
			   "         fII : %10g  fIa : %10g fAGB : %10g\n"
			   "         fnII: %10g  fnIa: %10g fnAGB: %10g\n", j,
			   save_fractionII[j * LT_NMetP + i],
			   save_fractionIa[j * LT_NMetP + i],
			   save_fractionAGB[j * LT_NMetP + i],
			   dfractionII[j * LT_NMetP + i],
			   dfractionIa[j * LT_NMetP + i],
			   dfractionAGB[j * LT_NMetP + i],
			   nfractionII[j * LT_NMetP + i],
			   nfractionIa[j * LT_NMetP + i], nfractionAGB[j * LT_NMetP + i]);

		}
	      fflush(stdout);
	      endrun(565656);
	    }
	  else
	    {
	      Sum = 0.999 / Sum;
	      for(j = 0; j < SFs_dim; j++)
		{
		  fractionII[j * LT_NMetP + i] *= Sum;
		  fractionIa[j * LT_NMetP + i] *= Sum;
		  fractionAGB[j * LT_NMetP + i] *= Sum;
		}
	    }
	}
    }

  for(j = 0; j < SFs_dim; j++)
    /* store in a packed forme the fractional contributes */
    pack_contrib(current_contrib, j,
		 &fractionII[j * LT_NMetP], &fractionIa[j * LT_NMetP], &fractionAGB[j * LT_NMetP]);



  return;
}
#endif


/* ============================================================
 *
 *     ---=={[  get index                      ]}==---
 * ============================================================ */

/*
 * This is nothing more than a trivial binary-interpolating search routine on arrays of double.
 * In the case you need a search routine for whatever data type you must modify it.
 */

int INLINE_FUNC getindex(double *table, int lower, int greatest, double *tval, int *index)
{
  double vh, maxval, minval;
  int found, half, upi, downi, res;

  downi = 0;

  maxval = table[greatest];
  minval = table[lower];

  upi = greatest;
  found = 0;

  res = 0;

  if(*tval >= maxval)
    {
      *index = greatest;
      if(*tval > maxval)
	res = -1;
      return res;
    }

  if(*tval <= minval)
    {
      *index = lower;
      if(*tval < minval)
	res = -2;
      return res;
    }

  while(found != 1)
    {
      /* non-interpolated binary guess          
       * half = (upi + downi) / 2;
       */

      /* linearly interpolate median index.
       * this work if values are equally linearly distributed as is our case.
       */

      /*if((half = downi + (int)((*tval - table[downi]) * (upi - downi) / (table[upi] - table[downi]))) != downi) */
      if((half = (upi + downi) / 2) != downi)
	{
	  vh = table[half];

	  if(*tval < vh)
	    upi = half;
	  else if(*tval > vh)
	    downi = half;
	  else
	    {
	      downi = half;
	      found = 1;
	    }
	  if((upi - downi) == 1)
	    found = 1;
	}
      else
	found = 1;
    }

  res = 0;
  *index = downi;

  return (res);
}




/* ============================================================
 *
 *     ---=={[  LOG enrichment details section  ]}==---
 * ============================================================ */


#ifdef LT_LOG_ENRICH_DETAILS

int OpenLogEnrichDetails_single(int mode)
{
  char buf[1000];

  if(mode == 1)
    {
      sprintf(buf, "%s%s_%3d.tmp", All.OutputDir, "LogEnrich");
      if((FdLogEnrichDetails_temp = fopen(buf, "w")) == 0x0)
	{
	  printf("error in opening file '%s'\n", buf);
	  endrun(1);
	}
    }
  else
    return fclose(FdLogEnrichDetails_temp);
  return 0;
}

int OpenLogEnrichDetails(int mode)
{
  char buf[1000];

  if(mode == 1)
    {
      sprintf(buf, "%s%s_%3d.tmp", All.OutputDir, "LogEnrich");
      if((FdLogEnrichDetails = fopen(buf, "w")) == 0x0)
	{
	  printf("error in opening file '%s'\n", buf);
	  endrun(1);
	}
    }
  else
    return fclose(FdLogEnrichDetails) return 0;
}


static char LogEnrichBuff[10000];

#ifndef LONGIDS
int LogEnrichDetails(unsigned int IDsource, FLOAT radius, int NNeigh, int *targets, char *Zflags)
#else
int LogEnrichDetails(unsigned long int IDsource, FLOAT radius, int Nneigh, int *targets, char *Zflags)
#endif
{
  int i;

  memset(LogEnrichBuff, 0, 10000);

  sprintf(LogEnrichBuff, "%020lu %8.7lf %4d", (unsigned long) IDsource, (double) radius, Nneigh);

  for(i = 0; i < Nneigh; i++)
    if((Zflag + i / 8) & (1 << (i % 8)))
      sprintf(LogEnrichBuff, "%020lu ", (unsigned long) P[targets[i]].ID);

  return fprintf(FdLogEnrichDetails_temp, "%8.7lf %s", All.Time, LogEnrichBuff);
}


int LogEnrichSource(int i, int IMFtype)
{
  return fprintf(FdLogEnrichDetails_temp, "%8.7lf %020lu %1d", All.Time, P[i].ID, IMFtype);
}

int ReshuffleLogEnrichDetails()
{
  char buf[1000];

  FILE **files;

  int i;

  OpenLogEnrichDetails_single(0);

  if(ThisTask == 0)
    {
      file = (FILE **) malloc(NTask * sizeof(FILE));
      for(i = 0; i < NTask; i++)
	{
	  sprintf(buf, "%s%s_%3d.tmp", All.OutputDir, "LogEnrich");
	  FdLogEnrichDetails_temp = fopen(buf, "w");
	}


    }

  MPI_Barrier(MPI_COMM_WORLD);
  OpenLogEnrichDetails_single(1);
  MPI_Barrier(MPI_COMM_WORLD);

  return 0;
}

#endif


void Initialize_StandardArguments()
{
  All.LocalSpreadFactor = 1.0;


  return;
}


static double aold = -1;

int get_chemstep_bin(double a, double chemstep, int *chem_step, int i)
{
  static double factor;
  int mychemstep;
  int ti_min, mybin;

  if(a != aold)
    {
      factor = (1000 * SEC_PER_MEGAYEAR) / All.UnitTime_in_s * All.HubbleParam / All.Timebase_interval;
      if(All.ComovingIntegrationOn)
	factor *= hubble_function(All.Time);
    }

  mychemstep = (int) (chemstep * factor);
  ti_min = TIMEBASE;
  while(ti_min > mychemstep)
    ti_min >>= 1;
  if((mychemstep = ti_min) == 0)
    return 0;

  if(mychemstep == 1)
    {
      printf("task %d wants to assign a chem step = 1 to particle %d (ID %llu)\n",
	     ThisTask, i, (long long unsigned) P[i].ID);
      endrun(112313);
    }

  *chem_step = mychemstep;

  mybin = -1;
  while(mychemstep)
    {
      mybin++;
      mychemstep >>= 1;
    }

  return mybin;
}

#endif
