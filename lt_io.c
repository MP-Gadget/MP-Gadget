#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>

#include "allvars.h"
#include "proto.h"

#include "lt_error_codes.h"

#ifdef LT_STELLAREVOLUTION

/* | ------------------------------------------------------------------------------------------------------ | 
 * |
 * |  COOLING
 * | ------------------------------------------------------------------------------------------------------ | 
 */

#ifdef LT_METAL_COOLING

int find_cooling_files(char *basename, int *nlines)
{
  char *comment_string = "!#;\0";

  FILE *file;

  int i, nline;

  char buff[500], line[1000], *c;

  nline = 0;
  for(i = 0;; i++)
    {
      sprintf(buff, "%s.%d", basename, i);
      if((file = fopen(buff, "r")) != 0x0)
	{
	  if(i == 0)
	    {
	      while((c = fgets(line, 999, file)) != 0x0)
		{
		  if(strchr(comment_string, line[0]) != 0x0)
		    continue;
		  nline++;
		}
	    }
	  fclose(file);
	}
      else
	break;
    }

  *nlines = nline;
  return i;
}

void reading_thresholds_for_thermal_instability()
{
  FILE *file;

  char buff[500];

  int i;

  printf("reading temperatures for thermal instability thresholds..\n");

  sprintf(buff, "thermalinstability_onset.dat");
  if((file = fopen(buff, "r")) != 0x0)
    {
      for(i = 0; i < ZBins; i++)
	fscanf(file, "%lf", &ThInst_onset[i]);
    }
  else
    {
      printf("file for thermal inistability thresholds not found: using 10^5 K for all the %d metallicities",
	     ZBins);
      for(i = 0; i < ZBins; i++)
	ThInst_onset[i] = 5;
    }

/*   for(i = 0; i < ZBins; i++) */
/*     ThInst_onset[i] = pow(10, ThInst_onset[i]); */

  return;
}

void read_cooling_tables(char *basename)
{
  char comment_string[5];

  FILE *file;

  char buff[500], line[1000];

  char *c;

  int i, j;

  double Z, T, L;

  double *swap_point;

  double InfoGroup[6];

  if(ThisTask == 0)
    {
      sprintf(comment_string, "!#;");

      if((ZBins = find_cooling_files(basename, &TBins)) == 0)
	{
	  printf("!! error: no cooling tables found! better to stop here\n");
	  endrun(909091);
	}
    }

  MPI_Bcast(&ZBins, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&TBins, 1, MPI_INT, 0, MPI_COMM_WORLD);

  CoolTvalue = (double *) mymalloc("CoolTvalue", (TBins + ZBins + TBins * ZBins) * sizeof(double));
  memset(CoolTvalue, 0, (TBins + ZBins + TBins * ZBins) * sizeof(double));

  CoolingTables = (double **) mymalloc("CoolingTables", ZBins * sizeof(double *));
  memset(CoolingTables, 0, ZBins * sizeof(double *));

  CoolZvalue = CoolTvalue + TBins;
  CoolingTables[0] = CoolZvalue + ZBins;
  for(i = 1; i < ZBins; i++)
    CoolingTables[i] = CoolingTables[i - 1] + TBins;

  if(ThisTask == 0)
    printf("\nreading %d cooling tables, each with %d temperature data\n", ZBins, TBins);

  TMin = 1e9;
  TMax = 0;
  ZMin = 10;
  ZMax = 0;

  if(ThisTask == 0)
    {
      for(i = 0; i < ZBins; i++)
	{
	  sprintf(buff, "%s.%d", basename, i);
	  file = fopen(buff, "r");

	  j = 0;
	  Z = 1e9;
	  while((c = fgets(line, 999, file)) != 0x0)
	    {
	      if(strchr(comment_string, line[0]) != 0x0)
		{
		  if((c = strstr(line, "name of tables ")) != 0x0)
		    if(i == 0)
		      printf("%s\n", &line[(int) (c - line)]);
		  if((c = strstr(line, "Z = ")) != 0x0)
		    {
		      sscanf(&line[(int) (c - line)], "%*s = %lf %*s\n", &Z);
		      printf("reading data for Z = %8.6e\n", Z);
		      if(ZMin > Z)
			ZMin = Z;
		      if(ZMax < Z)
			ZMax = Z;
		      CoolZvalue[i] = Z;
		    }
		  continue;
		}

	      if(sscanf(line, "%lf %*f %*f %*f %*f %lf %*s", &T, &L) != 2)
		{
		  printf("some error in reading line %d of table %d\n", j, i);
		  endrun(909092);
		}
	      if(i == 0)
		{
		  CoolTvalue[j] = T;
		  if(TMin > T)
		    TMin = T;
		  if(TMax < T)
		    TMax = T;
		}
	      CoolingTables[i][j] = L;
	      j++;
	      if(feof(file))
		break;
	    }
	  if(Z == 1e9)
	    {
	      printf("this file %s has not a defined metallicity. interrupt.\n", buff);
	      exit(404040);
	    }
	  if(j < TBins)
	    {
	      printf("this file %s has not %d data lines. interrupt.\n", buff, TBins);
	      exit(404040);
	    }
	  fclose(file);
	}

      for(i = 0; i < ZBins; i++)
	for(j = i + 1; j < ZBins; j++)
	  if(CoolZvalue[j] < CoolZvalue[j - 1])
	    {
	      Z = CoolZvalue[j - 1];
	      CoolZvalue[j - 1] = CoolZvalue[j];
	      CoolZvalue[j] = Z;

	      swap_point = CoolingTables[j - 1];
	      CoolingTables[j - 1] = CoolingTables[j];
	      CoolingTables[j] = swap_point;
	    }

      *(int *) &InfoGroup[0] = TBins;
      *(int *) &InfoGroup[1] = ZBins;
      InfoGroup[2] = ZMax;
      InfoGroup[3] = ZMin;
      InfoGroup[4] = TMax;
      InfoGroup[5] = TMin;

    }

  MPI_Bcast(InfoGroup, 6 * sizeof(double), MPI_BYTE, 0, MPI_COMM_WORLD);

  if(ThisTask != 0)
    {
      TBins = *(int *) &InfoGroup[0];
      ZBins = *(int *) &InfoGroup[1];
      ZMax = InfoGroup[2];
      ZMin = InfoGroup[3];
      TMax = InfoGroup[4];
      TMin = InfoGroup[5];

      CoolTvalue = (double *) mymalloc("CoolTvalue", (TBins + ZBins + TBins * ZBins) * sizeof(double));
      memset(CoolTvalue, 0, (TBins + ZBins + TBins * ZBins) * sizeof(double));
    }

  MPI_Bcast(CoolTvalue, (TBins + ZBins + TBins * ZBins) * sizeof(double), MPI_BYTE, 0, MPI_COMM_WORLD);

  ThInst_onset = (double *) mymalloc("ThIns_onset", ZBins * sizeof(double));

  if(ThisTask > 0)
    {
      CoolZvalue = CoolTvalue + TBins;
      CoolingTables[0] = CoolZvalue + ZBins;
      for(i = 1; i < ZBins; i++)
	CoolingTables[i] = CoolingTables[i - 1] + TBins;

      for(i = 1; i < ZBins; i++)
	CoolingTables[i] = CoolingTables[i - 1] + TBins;
    }

  /* note : for S&D 1993 */
/*   if(ThisTask == 0) */
/*     reading_thresholds_for_thermal_instability(); */
/*   MPI_Bcast(&ThInst_onset[0], ZBins * sizeof(double), MPI_BYTE, 0, MPI_COMM_WORLD); */
/*   ThInst_onset[0] = 5; */
/*   ThInst_onset[1] = 5; */
/*   ThInst_onset[2] = 5; */
/*   ThInst_onset[3] = 5.3; */
/*   ThInst_onset[4] = 5.3; */
/*   ThInst_onset[5] = 5.3; */
/*   ThInst_onset[6] = 5.3; */
/*   ThInst_onset[7] = 5.3; */




  if(ThisTask == 11)
    {
      for(i = 0; i < ZBins; i++)
	{

	  sprintf(buff, "coolingtable.%d.%d", ThisTask, i);
	  file = fopen(buff, "w");

	  for(j = 0; j < TBins; j++)
	    fprintf(file, "%10.8e %10.8e\n", CoolTvalue[j], CoolingTables[i][j]);

	  fclose(file);

	}
    }

  MPI_Barrier(MPI_COMM_WORLD);
  if(ThisTask == 0)
    printf("\n\n");
  return;
}

#else

void read_cooling_tables(char *basename)
{
  int i;

  ZBins = 1;
  TBins = 1;
  ZMin = ZMax = -4.0;
  TMin = 4.0;
  TMax = 8.0;

  CoolTvalue = (double *) mymalloc("CoolTvalue", (TBins + ZBins + TBins * ZBins) * sizeof(double));
  memset(CoolTvalue, 0, (TBins + ZBins + TBins * ZBins) * sizeof(double));

  CoolingTables = (double **) mymalloc("CoolingTables", ZBins * sizeof(double *));
  memset(CoolingTables, 0, ZBins * sizeof(double *));

  CoolZvalue = CoolTvalue + TBins;
  CoolingTables[0] = CoolZvalue + ZBins;
  for(i = 1; i < ZBins; i++)
    CoolingTables[i] = CoolingTables[i - 1] + TBins;

  ThInst_onset = (double *) mymalloc("ThInst_onset", ZBins * sizeof(double));
  ThInst_onset[0] = 5;
}


#endif



/* | ------------------------------------------------------------------------------------------------------ | 
 * |
 * |  METALS
 * | ------------------------------------------------------------------------------------------------------ | 
 */

void read_metals(void)
{
  char s[200], name[200];

  int j;

  FILE *file;

#ifdef UM_METAL_COOLING
  char *PT_Symbols;

  float *PT_Masses;

  int NPT;
#endif

  Hel = -1;
  Iron = -1;
  Oxygen = -1;
  FillEl = -1;

  /* :: by UM :: */
#ifdef UM_METAL_COOLING
  Carbon = -1;
  Magnesium = -1;
  Silicon = -1;
  Nitrogen = -1;
#endif

  if(ThisTask == 0)
    {
      if((file = fopen("metals.dat", "r")) == NULL)
	{
	  printf("I can't open metals data input file:" "%s\n", "metals.dat");
	  endrun(88888);
	}

      for(j = 0; j < LT_NMet; j++)
	{
	  if(feof(file))
	    {
	      printf("something wrong with <metals.dat>\n");
	      endrun(88889);
	    }
	  fgets(s, 200, file);
	  if((sscanf(s, "%[a-zA-Z]s %lg", &name[0], &MetSolarValues[j])) == 1)
	    printf("it seems that in <metals.dat> no solar abundance is present for element %s\n", name);

	  sprintf(s, "MetNames_%02d", j);
	  MetNames[j] = (char *) mymalloc(s, strlen(name) + 2);
	  strcpy(MetNames[j], name);
	  if(strcmp(name, "He") == 0)
	    Hel = j;
#ifdef UM_METAL_COOLING
	  else if(strcmp(name, "C") == 0)
	    Carbon = j;
	  else if(strcmp(name, "N") == 0)
	    Nitrogen = j;
	  else if(strcmp(name, "Mg") == 0)
	    Magnesium = j;
	  else if(strcmp(name, "Si") == 0)
	    Silicon = j;
#endif
	  else if(strcmp(name, "O") == 0)
	    Oxygen = j;
	  else if(strcmp(name, "Fe") == 0)
	    Iron = j;
	  else if(strcmp(name, "Ej") == 0)
	    FillEl = j;
	}

      Hyd = LT_NMet - 1;
      sprintf(s, "MetNames_%02d", Hyd);
      MetNames[Hyd] = (char *) mymalloc(s, 3);
      strcpy(MetNames[Hyd], "H");


#ifdef UM_METAL_COOLING
      printf
	("\n:: Hyd %d, Hel %d, Carbon %d, Nitrogen %d, Magnesium %d, Silicon %d, Oxygen %d, Iron %d, FillEl %d\n",
	 Hyd, Hel, Carbon, Nitrogen, Magnesium, Silicon, Oxygen, Iron, FillEl);
      printf(":: -1 means -> not present!\n\n");
#endif


      if(FillEl == -1)
	{
	  printf("you don't have specified the FillEl position.. better to do\n");
	  endrun(10001000);
	}

      fclose(file);

#ifdef LT_METAL_COOLING
      if(Iron == -1)
	{
	  if(Oxygen >= 0)
	    printf("you don't trace IRON. metal cooling will be calculated inferring X_Fe from X_O\n");
	  else
	    {
	      printf("you don't trace neither IRON nor OXYGEN. So far, metal cooling cannot be used\n");
	      endrun(993399);
	    }
	}
#endif
    }				/* close ThisTask = 0 */

  MPI_Bcast(&Iron, sizeof(int), MPI_BYTE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&Oxygen, sizeof(int), MPI_BYTE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&Hel, sizeof(int), MPI_BYTE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&FillEl, sizeof(int), MPI_BYTE, 0, MPI_COMM_WORLD);
#ifdef UM_METAL_COOLING
  MPI_Bcast(&Carbon, sizeof(int), MPI_BYTE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&Nitrogen, sizeof(int), MPI_BYTE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&Magnesium, sizeof(int), MPI_BYTE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&Silicon, sizeof(int), MPI_BYTE, 0, MPI_COMM_WORLD);
#endif


  return;
}


/* | ------------------------------------------------------------------------------------------------------ | 
 * |
 * |  YIELDS
 * | ------------------------------------------------------------------------------------------------------ | 
 */


int *Zbins_dim, *Mbins_dim;

double *Zbins, *Mbins;

double *DataSpace, *Yields[LT_NMet];

/* #ifdef UM_METAL_COOLING */
/* double *FillNDens; */

/* int compare_El_Symbols(const void *a, const void *b) */
/* { */
/*   return strcmp((char *) a, (char *) b); */
/* } */


/* #endif */

#define LL 1500

int read_yields_file(FILE * file, char *DataSpaceName)
{
  char s[LL], name[20];

  int i, j, k, nonproc = 0;

  /* : ................... : */
  /* :[ read Sn Ia yields ]: */


  /*   read the number of metal bins (1 for metal
     independent yields) */
  do
    {
      fgets(s, LL, file);
      if(strstr(s, "nonproc on") != 0x0)
	nonproc = 1;
    }
  while(strchr("%# \n", s[0]) != 0x0);
  /*    allocate space for metal bins and read/store them */
  *Zbins_dim = atoi(s);
  Zbins = (double *) mymalloc("Zbins", *Zbins_dim * sizeof(double));
  memset(Zbins, 0, *Zbins_dim * sizeof(double));
  if(*Zbins_dim > 1)
    {
      do
	fgets(s, LL, file);
      while(strchr("%# \n", s[0]) != 0x0);
      for(j = 0; j < *Zbins_dim; j++)
	sscanf(&s[0], "%lg%[^n]s", &Zbins[j], &s[0]);
    }
  else
    Zbins[0] = 0;
  /*    allocate space for mass bins and read/store them */
  do
    fgets(s, LL, file);
  while(strchr("%# \n", s[0]) != 0x0);
  *Mbins_dim = atoi(s);
  Mbins = (double *) mymalloc("Mbins", *Mbins_dim * sizeof(double));
  memset(Mbins, 0, *Mbins_dim * sizeof(double));

  if(*Mbins_dim > 1)
    {
      do
	fgets(s, LL, file);
      while(strchr("%# \n", s[0]) != 0x0);
      for(j = 0; j < *Mbins_dim; j++)
	sscanf(&s[0], "%lg%[^n]s", &Mbins[j], &s[0]);
    }
  else
    Mbins[0] = 0;

  DataSpace = (double *) mymalloc(DataSpaceName, (LT_NMet * *Zbins_dim * *Mbins_dim) * sizeof(double));
  memset(DataSpace, 0, (LT_NMet * *Zbins_dim * *Mbins_dim) * sizeof(double));

  Yields[0] = &DataSpace[0];
  for(j = 1; j < LT_NMet; j++)
    Yields[j] = Yields[j - 1] + *Zbins_dim * *Mbins_dim;


  /* actually read yields. they are organized in subsequent blocks, one for each
     metal bin. each block is a table, whose rows refer to a single element and
     columns to the mass array */
  j = 0;
  while(j < *Zbins_dim)
    {
      do
	fgets(s, LL, file);
      while(!feof(file) && (strchr("%# \n", s[0]) != 0x0));

      while(!feof(file) && (strchr("%# \n", s[0]) == 0x0))
	{
	  /* find the element name */
	  sscanf(s, "%s %[^\n]s", name, &s[0]);

	  for(k = 0; k < LT_NMet; k++)
	    if(strcmp(name, MetNames[k]) == 0)
	      break;

	  /* this is an element you want to use (as specified in metals.dat) */
	  if(k < LT_NMet)
	    for(i = 0; i < *Mbins_dim; i++)
	      sscanf(s, "%lg%[^\n]s", &Yields[k][*Mbins_dim * j + i], &s[0]);
	  fgets(s, LL, file);
	}
      j++;
    }
  return nonproc;
}


void read_yields_specie(char *Specie,
			int Nset,
			int **myZbins_dim,
			double ***myZbins,
			int **myMbins_dim,
			double ***myMbins,
			double ****YieldsTables, double ***Ej, int **NonProcOn, char *datafile)
{
  char buff[300], name[100];

  int set, i, j, k;

  FILE *file;

  /*
   * read in Sn Ia yields
   */

  sprintf(name, "%s_Zbins_dim", Specie);
  *myZbins_dim = (int *) mymalloc(name, sizeof(int) * Nset);
  memset(*myZbins_dim, 0, sizeof(int) * Nset);

  sprintf(name, "%s_Zbins", Specie);
  *myZbins = (double **) mymalloc(name, sizeof(double *) * Nset);
  memset(*myZbins, 0, sizeof(double *) * Nset);

  sprintf(name, "%s_Mbins_dim", Specie);
  *myMbins_dim = (int *) mymalloc(name, sizeof(int) * Nset);
  memset(*myMbins_dim, 0, sizeof(int) * Nset);

  sprintf(name, "%s_Mbins", Specie);
  *myMbins = (double **) mymalloc(name, sizeof(double *) * Nset);
  memset(*myMbins, 0, sizeof(double *) * Nset);

  sprintf(name, "%s_Yields", Specie);
  *YieldsTables = (double ***) mymalloc(name, sizeof(double **) * Nset);
  memset(*YieldsTables, 0, sizeof(double **) * Nset);

  for(set = 0; set < Nset; set++)
    {
      sprintf(buff, "%s_Yields_set_%02d_rep", Specie, set);
      (*YieldsTables)[set] = (double **) mymalloc(buff, LT_NMet * sizeof(double *));
      //memset((*YieldsTables)[set], 0, LT_NMet * sizeof(double *));
    }

  if(Ej != 0x0)
    {
      sprintf(name, "%s_Ej", Specie);
      *Ej = (double **) mymalloc(name, Nset * sizeof(double *));
      //memset(*Ej, 0, Nset * sizeof(double *));
    }

  sprintf(buff, "%s_NonProcOn", Specie);
  *NonProcOn = (int *) mymalloc(buff, Nset * sizeof(int));
  memset(*NonProcOn, 0, Nset * sizeof(int));

  for(set = 0; set < Nset; set++)
    {
      Zbins_dim = &(*myZbins_dim)[set];
      Mbins_dim = &(*myMbins_dim)[set];

      sprintf(name, "%s_Yields_set_%02d", Specie, set);
      if(ThisTask == 0)
	{
	  if(Nset > 1)
	    sprintf(buff, "%s.%03d", datafile, set);
	  else
	    strcpy(buff, datafile);
	  if((file = fopen(buff, "r")) == NULL)
	    {
	      printf("I can't open %s data input file: <%s>\n", Specie, buff);
	      MPI_Finalize();
	      exit(0);
	    }
	  else
	    {
	      if((*NonProcOn)[set] = read_yields_file(file, name))
		fprintf(FdSnInit, "%s yields in Set %d need to account for non processed metals\n", Specie,
			set);
	      fclose(file);

	      (*myZbins)[set] = Zbins;
	      (*myMbins)[set] = Mbins;
	      for(i = 0; i < LT_NMet; i++)
		(*YieldsTables)[set][i] = Yields[i];

	    }
	}

      MPI_Bcast(&((*NonProcOn)[set]), 1, MPI_INT, 0, MPI_COMM_WORLD);

      MPI_Bcast(&(Zbins_dim[set]), 1, MPI_INT, 0, MPI_COMM_WORLD);
      if(ThisTask != 0)
	{
	  sprintf(buff, "%s_Zbins_set_%02d", Specie, set);
	  (*myZbins)[set] = (double *) mymalloc(buff, Zbins_dim[set] * sizeof(double));
	}
      MPI_Bcast((*myZbins)[set], Zbins_dim[set] * sizeof(double), MPI_BYTE, 0, MPI_COMM_WORLD);

      MPI_Bcast(&(Mbins_dim[set]), 1, MPI_INT, 0, MPI_COMM_WORLD);
      if(ThisTask != 0)
	{
	  sprintf(buff, "%s_Mbins_set_%02d", Specie, set);
	  (*myMbins)[set] = (double *) mymalloc(buff, Mbins_dim[set] * sizeof(double));
	}
      MPI_Bcast((*myMbins)[set], Mbins_dim[set] * sizeof(double), MPI_BYTE, 0, MPI_COMM_WORLD);

      if(ThisTask != 0)
	DataSpace =
	  (double *) mymalloc(name, (LT_NMet * (*myZbins_dim)[set] * (*myMbins_dim[set])) * sizeof(double));
      MPI_Bcast(DataSpace, LT_NMet * (*myZbins_dim)[set] * (*myMbins_dim)[set] * sizeof(double), MPI_BYTE, 0,
		MPI_COMM_WORLD);

      if(ThisTask != 0)
	{
	  (*YieldsTables)[set][0] = DataSpace;
	  for(j = 1; j < LT_NMet; j++)
	    (*YieldsTables)[set][j] = (*YieldsTables)[set][j - 1] + (*myZbins_dim)[set] * (*myMbins_dim[set]);
	}

      MPI_Barrier(MPI_COMM_WORLD);

      /* Ej[set] will contain the total ejected mass in all element present in file for each couple
         (Zbin,Mbin) */
      if(Ej != 0x0)
	{
	  sprintf(buff, "%s_Ej_set_%02d", Specie, set);
	  (*Ej)[set] = (double *) mymalloc(buff, (*myZbins_dim)[set] * (*myMbins_dim)[set] * sizeof(double));

	  for(j = 0; j < (*myZbins_dim)[set]; j++)
	    for(i = 0; i < (*myMbins_dim)[set]; i++)
	      {
		k = j * (*myMbins_dim)[set] + i;
		(*Ej)[set][k] = (*YieldsTables)[set][FillEl][k];
	      }

/*           memset((*Ej)[set], 0, (*myZbins_dim)[set] * (*myMbins_dim)[set] * sizeof(double)); */
/*           memcpy((void *) (*Ej)[set], (void *) (*YieldsTables)[set][FillEl], */
/*                  (size_t) ((*myZbins_dim)[set] * (*myMbins_dim)[set] * sizeof(double))); */

	  /* the last element FillEl will contain the difference between the total ejected mass and the sum
	     of ejecta from the used elements (which can be less than those present in the file) */
	  for(i = 0; i < (*myZbins_dim)[set]; i++)
	    for(j = 0; j < (*myMbins_dim)[set]; j++)
	      {
		(*YieldsTables)[set][FillEl][i * (*myMbins_dim)[set] + j] = 0;
		for(k = 0; k < LT_NMet; k++)
		  if(k != FillEl)
		    (*YieldsTables)[set][FillEl][i * (*myMbins_dim)[set] + j] +=
		      (*YieldsTables)[set][k][i * (*myMbins_dim)[set] + j];
		if(fabs
		   (((*Ej)[set][i * (*myMbins_dim)[set] + j] -
		     (*YieldsTables)[set][FillEl][i * (*myMbins_dim)[set] +
						  j]) / (*Ej)[set][i * (*myMbins_dim)[set] + j]) < 1e-2)
		  (*YieldsTables)[set][FillEl][i * (*myMbins_dim)[set] + j] = 0;
		else
		  {
		    (*YieldsTables)[set][FillEl][i * (*myMbins_dim)[set] + j] =
		      (*Ej)[set][i * (*myMbins_dim)[set] + j] -
		      (*YieldsTables)[set][FillEl][i * (*myMbins_dim)[set] + j];
		    if((*YieldsTables)[set][FillEl][i * (*myMbins_dim)[set] + j] < 0)
		      {
			if(ThisTask == 0 && (*YieldsTables)[set][FillEl][i * (*myMbins_dim)[set] + j] < 1e-2)
			  printf("    --> warning: the fill elements in II set %d for mass bin %d is lower"
				 " than the sum of the collected elements!\n     better to force it to zero\n",
				 set, j);
			fflush(stdout);
			(*YieldsTables)[set][FillEl][i * (*myMbins_dim)[set] + j] = 0;
		      }
		  }
	      }
	}
    }

  return;
}


void ReadYields(int readIa, int readII, int readAGB)
{
  if(readII)
    {
      if(ThisTask == 0)
	printf("reading yields for SnII.. ");
      fflush(stdout);
      read_yields_specie("SnII",
			 All.II_Nset_ofYields,
			 &IIZbins_dim,
			 &IIZbins,
			 &IIMbins_dim, &IIMbins, &SnIIYields, &SnIIEj, &NonProcOn_II, All.SnIIDataFile);
      if(ThisTask == 0)
	printf("done\n");
      fflush(stdout);
    }

  if(readIa)
    {
      if(ThisTask == 0)
	printf("reading yields for SnIa.. ");
      fflush(stdout);
      read_yields_specie("SnIa",
			 All.Ia_Nset_ofYields,
			 &IaZbins_dim,
			 &IaZbins, &IaMbins_dim, &IaMbins, &SnIaYields, 0x0, &NonProcOn_Ia, All.SnIaDataFile);
      if(ThisTask == 0)
	printf("done\n");
      fflush(stdout);
    }

  if(readAGB)
    {
      if(ThisTask == 0)
	printf("reading yields for AGB.. ");
      fflush(stdout);
      read_yields_specie("AGB",
			 All.AGB_Nset_ofYields,
			 &AGBZbins_dim,
			 &AGBZbins,
			 &AGBMbins_dim, &AGBMbins, &AGBYields, &AGBEj, &NonProcOn_AGB, All.AGBDataFile);
      if(ThisTask == 0)
	printf("done\n");
      fflush(stdout);
    }
  return;
}


// -==================================================================================

/* #ifdef LT_SNIa */
/* void read_SnIa_yields(void) */
/* { */
/*   char buff[300], name[100]; */
/*   int  set, i, j; */
/*   FILE *file; */


/*   /\* */
/*    * read in Sn Ia yields */
/*    *\/ */

/*   IaZbins_dim = (int *)     mymalloc("IaZbins_dim", sizeof(int)      * All.Ia_Nset_ofYields); */
/*   memset(IaZbins_dim, 0, sizeof(int) * All.Ia_Nset_ofYields); */
/*   IaZbins     = (double **) mymalloc("IaZbins", sizeof(double *) * All.Ia_Nset_ofYields); */
/*   memset(IaZbins,     0, sizeof(double*) * All.Ia_Nset_ofYields); */
/*   IaMbins_dim = (int *)     mymalloc("IaMbins_dim", sizeof(int)      * All.Ia_Nset_ofYields); */
/*   memset(IaMbins_dim, 0, sizeof(int) * All.Ia_Nset_ofYields); */
/*   IaMbins     = (double **) mymalloc("IaMbins", sizeof(double *) * All.Ia_Nset_ofYields); */
/*   memset(IaMbins,     0, sizeof(double*) * All.Ia_Nset_ofYields); */
/*   SnIaYields  = (double ***)mymalloc("SnIaYields", sizeof(double **)* All.Ia_Nset_ofYields); */
/*   memset(SnIaYields, 0, sizeof(double**) * All.Ia_Nset_ofYields); */

/*   for(set = 0; set < All.Ia_Nset_ofYields; set++) */
/*     { */
/*       sprintf(buff, "SnIaYields_set_%02d_rep", set); */
/*       SnIaYields[set] = (double **) mymalloc(buff, LT_NMet * sizeof(double *)); */
/*       memset(SnIaYields[set], 0, LT_NMet * sizeof(double *)); */
/*     } */
/*   NonProcOn_Ia = (int *) mymalloc("NonProcOn_Ia", All.Ia_Nset_ofYields * sizeof(int)); */
/*   memset(NonProcOn_Ia, 0, All.Ia_Nset_ofYields * sizeof(int)); */

/*   for(set = 0; set < All.Ia_Nset_ofYields; set++) */
/*     { */
/*       sprintf(name, "SnIaYields_set_%02d", set); */
/*       if(ThisTask == 0) */
/* 	{ */
/* 	  if(All.Ia_Nset_ofYields > 1) */
/* 	    sprintf(buff, "%s.%03d", All.SnIaDataFile, set); */
/* 	  else */
/* 	    strcpy(buff, All.SnIaDataFile); */
/* 	  if((file = fopen(buff, "r")) == NULL) */
/* 	    { */
/* 	      printf("I can't open SnIa data input file: <%s>\n", buff); */
/* 	      MPI_Finalize(); */
/* 	      exit(0); */
/* 	    } */
/* 	  else */
/* 	    { */
/* 	      Zbins_dim = &IaZbins_dim[set]; */
/* 	      Mbins_dim = &IaMbins_dim[set]; */
/* 	      if((NonProcOn_Ia[set] = read_yields_file(file, name))) */
/* 		fprintf(FdSnInit, "Ia yields in Set %d need to account for non processed metals\n", set); */
/* 	      fclose(file); */

/* 	      IaZbins[set] = Zbins; */
/* 	      IaMbins[set] = Mbins; */
/* 	      for(i = 0; i < LT_NMet; i++) */
/* 		SnIaYields[set][i] = Yields[i]; */

/* 	    } */
/* 	} */

/*       MPI_Bcast(&NonProcOn_Ia[0], All.Ia_Nset_ofYields, MPI_INT, 0, MPI_COMM_WORLD); */
/*       MPI_Bcast(&IaZbins_dim[set], 1, MPI_INT, 0, MPI_COMM_WORLD); */
/*       if(ThisTask != 0) */
/*         { */
/*           sprintf(buff, "IaZbins_set_%02d", set); */
/*           IaZbins[set] = (double *) mymalloc(buff, IaZbins_dim[set] * sizeof(double)); */
/*           memset(IaZbins[set], 0, IaZbins_dim[set] * sizeof(double)); */
/*         } */
/*       MPI_Bcast(IaZbins[set], IaZbins_dim[set] * sizeof(double), MPI_BYTE, 0, MPI_COMM_WORLD); */

/*       MPI_Bcast(&IaMbins_dim[set], 1, MPI_INT, 0, MPI_COMM_WORLD); */
/*       if(ThisTask != 0) */
/*         { */
/*           sprintf(buff, "IaMbins_set_%02d", set); */
/*           IaMbins[set] = (double *) mymalloc(buff, IaMbins_dim[set] * sizeof(double)); */
/*           memset(IaMbins[set], 0, IaMbins_dim[set] * sizeof(double)); */
/*         } */
/*       MPI_Bcast(IaMbins[set], IaMbins_dim[set] * sizeof(double), MPI_BYTE, 0, MPI_COMM_WORLD); */

/*       if(ThisTask != 0) */
/* 	{ */
/* 	  DataSpace = (double *) mymalloc(name, (LT_NMet * IaZbins_dim[set] * IaMbins_dim[set]) * sizeof(double)); */
/*           memset(DataSpace, 0, (LT_NMet * IaZbins_dim[set] * IaMbins_dim[set]) * sizeof(double)); */

/* 	  SnIaYields[set][0] = DataSpace; */
/* 	  for(j = 1; j < LT_NMet; j++) */
/* 	    SnIaYields[set][j] = SnIaYields[set][j - 1] + IaZbins_dim[set] * IaMbins_dim[set]; */
/* 	} */
/*       MPI_Bcast(&DataSpace[0], LT_NMet * IaZbins_dim[set] * IaMbins_dim[set] * sizeof(double), */
/* 		MPI_BYTE, 0, MPI_COMM_WORLD); */


/*     } */

/*   return; */
/* } */
/* #endif */


/* #ifdef LT_SNII */
/* void read_SnII_yields(void) */
/* { */
/*   char  buff[300], name[100]; */
/*   int   set, i, j, k; */
/*   FILE *file; */


/*   /\* */
/*    * read in Sn II yields */
/*    *\/ */

/*   IIZbins_dim = (int *)     mymalloc("IIZbins_dim", sizeof(int      ) * All.II_Nset_ofYields); */
/*   memset(IIZbins_dim, 0, sizeof(int     ) * All.II_Nset_ofYields); */

/*   IIZbins     = (double **) mymalloc("IIZbins", sizeof(double  *) * All.II_Nset_ofYields); */
/*   memset(IIZbins,     0, sizeof(double *) * All.II_Nset_ofYields); */

/*   IIMbins_dim = (int *)     mymalloc("IIMbins_dim", sizeof(int      ) * All.II_Nset_ofYields); */
/*   memset(IIZbins_dim, 0, sizeof(int     ) * All.II_Nset_ofYields); */

/*   IIMbins     = (double **) mymalloc("IIMBbins", sizeof(double  *) * All.II_Nset_ofYields); */
/*   memset(IIMbins,     0, sizeof(double *) * All.II_Nset_ofYields); */

/*   SnIIYields  = (double ***)mymalloc("SnIIYields", sizeof(double **) * All.II_Nset_ofYields); */
/*   memset(SnIIYields,  0, sizeof(double**) * All.II_Nset_ofYields); */

/*   for(set = 0; set < All.II_Nset_ofYields; set++) */
/*     { */
/*       sprintf(name, "SnIIYields_set_%02d_rep", set); */
/*       SnIIYields[set] = (double **) mymalloc(name, LT_NMet * sizeof(double *)); */
/*       memset(SnIIYields[set], 0, LT_NMet * sizeof(double *)); */
/*     } */
/*   SnIIEj       = (double **) mymalloc("SnIIEj", All.II_Nset_ofYields * sizeof(double *)); */
/*   memset(SnIIEj, 0, All.II_Nset_ofYields * sizeof(double *)); */

/*   NonProcOn_II = (int *) mymalloc("NonProcOn_II", All.II_Nset_ofYields * sizeof(int)); */
/*   memset(NonProcOn_II, 0, All.II_Nset_ofYields * sizeof(int)); */


/*   for(set = 0; set < All.II_Nset_ofYields; set++) */
/*     { */
/*       sprintf(name, "SnIIYields_set_%02d", set); */
/*       if(ThisTask == 0) */
/* 	{ */
/* 	  if(All.II_Nset_ofYields > 1) */
/* 	    sprintf(buff, "%s.%03d", All.SnIIDataFile, set); */
/* 	  else */
/* 	    strcpy(buff, All.SnIIDataFile); */
/* 	  if((file = fopen(buff, "r")) == NULL) */
/* 	    { */
/* 	      printf("I can't open SnII data input file: <%s>\n", buff); */
/* 	      MPI_Finalize(); */
/* 	      exit(0); */
/* 	    } */
/* 	  else */
/* 	    { */
/* 	      Zbins_dim = &IIZbins_dim[set]; */
/* 	      Mbins_dim = &IIMbins_dim[set]; */
/* 	      if((NonProcOn_II[set] = read_yields_file(file, name))) */
/* 		fprintf(FdSnInit, "II yields in Set %d need to account for non processed metals\n", set); */
/* 	      fclose(file); */

/* 	      IIZbins[set] = Zbins; */
/* 	      IIMbins[set] = Mbins; */
/* 	      for(i = 0; i < LT_NMet; i++) */
/* 		SnIIYields[set][i] = Yields[i]; */
/* 	    } */
/* 	} */

/*       MPI_Bcast(&NonProcOn_II[0], All.II_Nset_ofYields, MPI_INT, 0, MPI_COMM_WORLD); */
/*       MPI_Bcast(&IIZbins_dim[set], 1, MPI_INT, 0, MPI_COMM_WORLD); */
/*       if(ThisTask != 0) */
/*         { */
/*           sprintf(buff, "IIZbins_set_%02d", set); */
/*           IIZbins[set] = (double *) mymalloc(buff, IIZbins_dim[set] * sizeof(double)); */
/*           memset(IIZbins[set], 0, IIZbins_dim[set] * sizeof(double)); */
/*         } */
/*       MPI_Bcast(&IIZbins[set][0], IIZbins_dim[set] * sizeof(double), MPI_BYTE, 0, MPI_COMM_WORLD); */

/*       MPI_Bcast(&IIMbins_dim[set], 1, MPI_INT, 0, MPI_COMM_WORLD); */
/*       if(ThisTask != 0) */
/*         { */
/*           sprintf(buff, "IIMbins_set_%02d", set); */
/*           IIMbins[set] = (double *) mymalloc(buff, IIMbins_dim[set] * sizeof(double)); */
/*           memset(IIMbins[set], 0, IIMbins_dim[set] * sizeof(double)); */
/*         } */
/*       MPI_Bcast(&IIMbins[set][0], IIMbins_dim[set] * sizeof(double), MPI_BYTE, 0, MPI_COMM_WORLD); */

/*       if(ThisTask != 0) */
/* 	{ */
/* 	  DataSpace = (double *) mymalloc(name, (LT_NMet * IIZbins_dim[set] * IIMbins_dim[set]) * sizeof(double)); */
/*           memset(DataSpace, 0, (LT_NMet * IIZbins_dim[set] * IIMbins_dim[set]) * sizeof(double)); */
/* 	  SnIIYields[set][0] = DataSpace; */
/* 	  for(j = 1; j < LT_NMet; j++) */
/* 	    SnIIYields[set][j] = SnIIYields[set][j - 1] + IIZbins_dim[set] * IIMbins_dim[set]; */
/* 	} */
/*       MPI_Bcast(&DataSpace[0], LT_NMet * IIZbins_dim[set] * IIMbins_dim[set] * sizeof(double), */
/* 		MPI_BYTE, 0, MPI_COMM_WORLD); */

/*       MPI_Barrier(MPI_COMM_WORLD); */

/*       /\* SnIIEj[set] will contain the total ejected mass in all element present in file for each couple */
/*          (Zbin,Mbin) *\/ */
/*       sprintf(buff, "SnIIEj_set_%02d", set); */
/*       SnIIEj[set] = (double *) mymalloc(buff, IIZbins_dim[set] * IIMbins_dim[set] * sizeof(double)); */
/*       memset(SnIIEj[set], 0, IIZbins_dim[set] * IIMbins_dim[set] * sizeof(double)); */
/*       memcpy((void *) SnIIEj[set], (void *) SnIIYields[set][FillEl], */
/* 	     (size_t) (IIZbins_dim[set] * IIMbins_dim[set] * sizeof(double))); */

/*       /\* the last element FillEl will contain the difference between the total ejected mass and the sum */
/*          of ejecta from the used elements (which can be less than those present in the file) *\/ */
/*       for(i = 0; i < IIZbins_dim[set]; i++) */
/* 	for(j = 0; j < IIMbins_dim[set]; j++) */
/* 	  { */
/* 	    SnIIYields[set][FillEl][i * IIMbins_dim[set] + j] = 0; */
/* 	    for(k = 0; k < LT_NMet; k++) */
/* 	      if(k != FillEl) */
/* 		SnIIYields[set][FillEl][i * IIMbins_dim[set] + j] += */
/* 		  SnIIYields[set][k][i * IIMbins_dim[set] + j]; */
/* 	    if(fabs */
/* 	       ((SnIIEj[set][i * IIMbins_dim[set] + j] - */
/* 		 SnIIYields[set][FillEl][i * IIMbins_dim[set] + j]) / SnIIEj[set][i * IIMbins_dim[set] + j]) < */
/* 	       1e-2) */
/* 	      SnIIYields[set][FillEl][i * IIMbins_dim[set] + j] = 0; */
/* 	    else */
/* 	      { */
/* 		SnIIYields[set][FillEl][i * IIMbins_dim[set] + j] = */
/* 		  SnIIEj[set][i * IIMbins_dim[set] + j] - SnIIYields[set][FillEl][i * IIMbins_dim[set] + j]; */
/* 		if(SnIIYields[set][FillEl][i * IIMbins_dim[set] + j] < 0) */
/* 		  { */
/* 		    if(ThisTask == 0 && SnIIYields[set][FillEl][i * IIMbins_dim[set] + j] < 1e-2) */
/* 		      printf("    --> warning: the fill elements in II set %d for mass bin %d is lower" */
/* 			     " than the sum of the collected elements!\n     better to force it to zero\n", */
/* 			     set, j); */
/* 		    fflush(stdout); */
/* 		    SnIIYields[set][FillEl][i * IIMbins_dim[set] + j] = 0; */
/* 		  } */
/* 	      } */
/* 	  } */
/*     } */
/*   return; */
/* } */
/* #endif */


/* #ifdef LT_AGB */
/* void read_AGB_yields(void) */
/* { */
/*   char buff[300], name[100]; */
/*   int set, i, j, k; */
/*   FILE *file; */


/*   /\* */
/*    * read in Sn AGB yields */
/*    *\/ */

/*   AGBZbins_dim = (int *)     mymalloc("AGBZbins_dim", sizeof(int      ) * All.AGB_Nset_ofYields); */
/*   memset(AGBZbins_dim, 0, sizeof(int      ) * All.AGB_Nset_ofYields); */

/*   AGBZbins     = (double **) mymalloc("AGBZbins", sizeof(double  *) * All.AGB_Nset_ofYields); */
/*   memset(AGBZbins,     0, sizeof(double  *) * All.AGB_Nset_ofYields); */

/*   AGBMbins_dim = (int *)     mymalloc("AGBMbins_dim", sizeof(int      ) * All.AGB_Nset_ofYields); */
/*   memset(AGBMbins_dim, 0, sizeof(int      ) * All.AGB_Nset_ofYields); */

/*   AGBMbins     = (double **) mymalloc("AGBMbins", sizeof(double  *) * All.AGB_Nset_ofYields); */
/*   memset(AGBMbins,     0, sizeof(double  *) * All.AGB_Nset_ofYields); */

/*   AGBYields    = (double ***)mymalloc("AGBYields", sizeof(double **) * All.AGB_Nset_ofYields); */
/*   memset(AGBYields,    0, sizeof(double  **) * All.AGB_Nset_ofYields); */

/*   for(set = 0; set < All.AGB_Nset_ofYields; set++) */
/*     { */
/*       sprintf(name, "ABYields_set_%02d_rep", set); */
/*       AGBYields[set] = (double **) mymalloc(name, LT_NMet * sizeof(double *)); */
/*       memset(AGBYields[set], 0, LT_NMet * sizeof(double *)); */
/*     } */
/*   AGBEj         = (double **) mymalloc("AGBEj", sizeof(double *) * All.AGB_Nset_ofYields); */
/*   memset(AGBEj, 0, sizeof(double *) * All.AGB_Nset_ofYields); */

/*   NonProcOn_AGB = (int *) mymalloc("NonProcOn_AGB", sizeof(int) * All.AGB_Nset_ofYields); */
/*   memset(NonProcOn_AGB, 0, sizeof(int) * All.AGB_Nset_ofYields); */

/*   for(set = 0; set < All.AGB_Nset_ofYields; set++) */
/*     { */
/*       sprintf(name, "AGBYields_set_%02d", set); */
/*       if(ThisTask == 0) */
/* 	{ */
/* 	  if(All.AGB_Nset_ofYields > 1) */
/* 	    sprintf(buff, "%s.%03d", All.AGBDataFile, set); */
/* 	  else */
/* 	    strcpy(buff, All.AGBDataFile); */
/* 	  if((file = fopen(buff, "r")) == NULL) */
/* 	    { */
/* 	      printf("I can't open AGB data input file: <%s>\n", buff); */
/* 	      MPI_Finalize(); */
/* 	      exit(1); */
/* 	    } */
/* 	  else */
/* 	    { */
/* 	      Zbins_dim = &AGBZbins_dim[set]; */
/* 	      Mbins_dim = &AGBMbins_dim[set]; */
/* 	      if((NonProcOn_AGB[set] = read_yields_file(file, name))) */
/* 		fprintf(FdSnInit, "AGB yields in Set %d need to account for non processed metals\n", set); */
/* 	      fclose(file); */

/* 	      AGBZbins[set] = Zbins; */
/* 	      AGBMbins[set] = Mbins; */
/* 	      for(i = 0; i < LT_NMet; i++) */
/* 		AGBYields[set][i] = Yields[i]; */
/* 	    } */
/* 	} */

/*       MPI_Bcast(&NonProcOn_AGB[0], All.AGB_Nset_ofYields, MPI_INT, 0, MPI_COMM_WORLD); */
/*       MPI_Bcast(&AGBZbins_dim[set], 1, MPI_INT, 0, MPI_COMM_WORLD); */
/*       if(ThisTask != 0) */
/*         { */
/*           sprintf(buff, "AGBZbins_set_%02d", set); */
/*           AGBZbins[set] = (double *) mymalloc(buff, AGBZbins_dim[set] * sizeof(double)); */
/*           memset(AGBZbins[set], 0, AGBZbins_dim[set] * sizeof(double)); */
/*         } */
/*       MPI_Bcast(&AGBZbins[set][0], AGBZbins_dim[set] * sizeof(double), MPI_BYTE, 0, MPI_COMM_WORLD); */

/*       MPI_Bcast(&AGBMbins_dim[set], 1, MPI_INT, 0, MPI_COMM_WORLD); */
/*       if(ThisTask != 0) */
/*         { */
/*           sprintf(buff, "AGBMbins_set_%02d", set); */
/*           AGBMbins[set] = (double *) mymalloc(name, AGBMbins_dim[set] * sizeof(double)); */
/*           memset(AGBMbins[set], 0, AGBMbins_dim[set] * sizeof(double)); */
/*         } */
/*       MPI_Bcast(&AGBMbins[set][0], AGBMbins_dim[set] * sizeof(double), MPI_BYTE, 0, MPI_COMM_WORLD); */

/*       if(ThisTask != 0) */
/* 	{ */
/* 	  DataSpace = (double *) mymalloc(name, LT_NMet * AGBZbins_dim[set] * AGBMbins_dim[set] * sizeof(double)); */
/*           memset(DataSpace, 0, LT_NMet * AGBZbins_dim[set] * AGBMbins_dim[set] * sizeof(double)); */
/* 	  AGBYields[set][0] = DataSpace; */
/* 	  for(j = 1; j < LT_NMet; j++) */
/* 	    AGBYields[set][j] = AGBYields[set][j - 1] + AGBZbins_dim[set] * AGBMbins_dim[set]; */
/* 	} */
/*       MPI_Bcast(&DataSpace[0], LT_NMet * AGBZbins_dim[set] * AGBMbins_dim[set] * sizeof(double), */
/* 		MPI_BYTE, 0, MPI_COMM_WORLD); */

/*       /\* AGBEj[set] will contain the total ejected mass in all element present in file for each couple */
/*          (Zbin,Mbin) *\/ */
/*       sprintf(buff, "AGBEj_set_%02d", set); */
/*       AGBEj[set] = (double *) mymalloc(buff, AGBZbins_dim[set] * AGBMbins_dim[set] * sizeof(double)); */
/*       memset(AGBEj[set], 0, AGBZbins_dim[set] * AGBMbins_dim[set] * sizeof(double)); */
/*       memcpy((void *) AGBEj[set], (void *) AGBYields[set][FillEl], */
/* 	     (size_t) (AGBZbins_dim[set] * AGBMbins_dim[set] * sizeof(double))); */

/*       /\* the last element FillEl will contain the difference between the total ejected mass and the sum */
/*          of ejecta from the used elements (which can be less than those present in the file) *\/ */
/*       for(i = 0; i < AGBZbins_dim[set]; i++) */
/* 	for(j = 0; j < AGBMbins_dim[set]; j++) */
/* 	  { */
/* 	    AGBYields[set][FillEl][i * AGBMbins_dim[set] + j] = 0; */
/* 	    for(k = 0; k < LT_NMet; k++) */
/* 	      if(k != FillEl) */
/* 		AGBYields[set][FillEl][i * AGBMbins_dim[set] + j] += */
/* 		  AGBYields[set][k][i * AGBMbins_dim[set] + j]; */
/* 	    if(fabs */
/* 	       ((AGBEj[set][i * AGBMbins_dim[set] + j] - */
/* 		 AGBYields[set][FillEl][i * AGBMbins_dim[set] + j]) / AGBEj[set][i * AGBMbins_dim[set] + j]) < */
/* 	       1e-2) */
/* 	      AGBYields[set][FillEl][i * AGBMbins_dim[set] + j] = 0; */
/* 	    else */
/* 	      { */
/* 		AGBYields[set][FillEl][i * AGBMbins_dim[set] + j] = */
/* 		  AGBEj[set][i * AGBMbins_dim[set] + j] - AGBYields[set][FillEl][i * AGBMbins_dim[set] + j]; */

/* 		if(AGBYields[set][FillEl][i * AGBMbins_dim[set] + j] < 0) */
/* 		  { */
/* 		    if(ThisTask == 0 && AGBYields[set][FillEl][i * AGBMbins_dim[set] + j] < -1e-2) */
/* 		      printf("    --> warning: the fill elements in AGB set %d for mass bin %d is lower" */
/* 			     " than the sum of the collected elements!\n     better to force it to zero\n", */
/* 			     set, j); */
/* 		    fflush(stdout); */
/* 		    AGBYields[set][FillEl][i * AGBMbins_dim[set] + j] = 0; */
/* 		  } */

/* 	      } */
/* 	  } */
/*     } */
/*   return; */
/* } */
/* #endif */



#endif
