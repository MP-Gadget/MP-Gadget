#include <stdio.h>
#include <stdint.h>
#include "genic/allvars.h"

int Nmesh, Ngrid;

double Box;
int ProduceGas;
int Seed;
int Unitary;
int UsePeculiarVelocity;
int NumPart;
double MaxMemSizePerNode;

struct part_data *P;

double InitTime;

char * OutputDir, * FileBase;
int NumWriters;
int  NumPartPerFile;

int ThisTask, NTask;

double UnitTime_in_s, UnitLength_in_cm, UnitMass_in_g, UnitVelocity_in_cm_per_s;
double G;
Cosmology CP;
struct power_params PowerP;
