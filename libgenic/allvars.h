#ifndef _GENIC_ALLVARS_H
#define _GENIC_ALLVARS_H

#include "power.h"
#include <libgadget/utils/unitsystem.h>

struct ic_part_data
{
  double PrePos[3];
  double Pos[3];
  float Vel[3];
  float Disp[3];
  float Density;
  float Mass;
};

struct genic_config {
    int Ngrid, NgridGas, NGridNu;
    int Nmesh;
    double BoxSize;
    int ProduceGas;
    int Seed;
    int UnitaryAmplitude;
    int InvertPhase;
    int PrePosGridCenter;
    double Max_nuvel;
    double WDM_therm_mass;
    int MakeGlassGas;
    int MakeGlassCDM;
    int NumFiles;
    int NumWriters;
    /* Whether to save the pre-displacement positions to the snapshot*/
    int SavePrePos;
    struct power_params PowerP;
    struct UnitSystem units;
    char OutputDir[100];
    char InitCondFile[100];
    double TimeIC;
    int UsePeculiarVelocity;
};

#endif
