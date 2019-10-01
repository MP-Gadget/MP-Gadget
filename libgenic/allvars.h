#include <stdint.h>
#include <libgadget/cosmology.h>
#include <libgadget/allvars.h>
#include "power.h"

struct ic_part_data
{
  double PrePos[3];
  double Pos[3];
  float Vel[3];
  float Disp[3];
  float Density;
  float Mass;
  int RegionInd;
};

struct genic_config {
    int Ngrid, NgridGas, NGridNu;

    int ProduceGas;
    int Seed;
    int UnitaryAmplitude;
    int InvertPhase;
    double Max_nuvel;
    double WDM_therm_mass;
    int MakeGlassGas;
    int MakeGlassCDM;
    int  NumFiles;
    struct power_params PowerP;
} ;


extern struct genic_config All2;

