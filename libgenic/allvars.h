#include <stdint.h>
#include <libgadget/cosmology.h>
#include <libgadget/allvars.h>
#include "power.h"

extern struct ic_part_data
{
  double Pos[3];
  float Vel[3];
  float Density;
  uint64_t ID;
  float Mass;
  int RegionInd;
} * ICP;
extern int NumPart;

struct genic_config {
    int Ngrid, NGridNu;

    int ProduceGas;
    int Seed;
    int UnitaryAmplitude;
    int InvertPhase;
    int DifferentTransferFunctions;
    double Max_nuvel;
    double WDM_therm_mass;

    int  NumFiles;
    struct power_params PowerP;
} ;


extern struct genic_config All2;

