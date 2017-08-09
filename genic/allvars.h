#include <stdint.h>
#include "cosmology.h"

extern int      WhichSpectrum;

extern double MaxMemoryPerCore;

extern int      Nmesh, Ngrid;

extern int UsePeculiarVelocity;
extern int PowerIsRedshiftZero;

extern char   *  FileWithInputSpectrum;
extern double   Box;
extern int   ProduceGas;
extern int Seed;

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
extern double MassTable[6];

extern char * OutputDir, * FileBase;
extern int  NumPartPerFile;
extern int  NumWriters;

extern int      ThisTask, NTask;

extern double UnitTime_in_s, UnitLength_in_cm, UnitMass_in_g, UnitVelocity_in_cm_per_s;
extern double InputSpectrum_UnitLength_in_cm;
extern double G;

extern Cosmology CP;

extern double Sigma8;
extern double PrimordialIndex;
extern double ShapeGamma;
