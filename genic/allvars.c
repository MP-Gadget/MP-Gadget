#include <stdio.h>
#include <stdint.h>
#include "genic/allvars.h"

int WhichSpectrum;

int SphereMode;
int Nmesh, Nsample, Ngrid;


char * FileWithInputSpectrum;

double Box;
int ProduceGas;
int Seed;
int UsePeculiarVelocity;
int NumPart;
double MaxMemoryPerCore;

struct part_data *P;

double InitTime;
double Redshift;
double MassTable[6];


char * OutputDir, * FileBase;
int NumWriters;
int  NumPartPerFile;


int ThisTask, NTask;

double UnitTime_in_s, UnitLength_in_cm, UnitMass_in_g, UnitVelocity_in_cm_per_s;
double InputSpectrum_UnitLength_in_cm;
double G;
Cosmology CP;
double OmegaDM_2ndSpecies, Sigma8;
double ShapeGamma;
double PrimordialIndex;

int WDM_On;
int WDM_Vtherm_On;
double WDM_PartMass_in_kev;
