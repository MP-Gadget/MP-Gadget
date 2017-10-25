#include <stdint.h>
#include "cosmology.h"
#include "power.h"

extern double MaxMemSizePerNode;

extern int      Nmesh, Ngrid, NGridNu;

extern int UsePeculiarVelocity;

extern double   Box;
extern int   ProduceGas;
extern int Seed;
extern int Unitary;
extern int DifferentTransferFunctions;

extern int      NumPart;

extern struct part_data 
{
  double Pos[3];
  float Vel[3];
  float Density;
  uint64_t ID;
  float Mass;
  int RegionInd;
} *P;

extern double InitTime;
extern double Max_nuvel;

extern char * OutputDir, * FileBase;
extern int  NumFiles;
extern int  NumWriters;

extern int      ThisTask, NTask;

extern double UnitTime_in_s, UnitLength_in_cm, UnitMass_in_g, UnitVelocity_in_cm_per_s;
extern double G;

extern Cosmology CP;
extern struct power_params PowerP;
