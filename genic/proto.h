#ifndef GENIC_PROTO_H
#define GENIC_PROTO_H
#include <bigfile.h>
#include <stdint.h>

void   displacement_fields(int Type);
void setup_grid(double shift, int64_t FirstID, int Ngrid);
void   free_ffts(void);

void saveheader(BigFile * bf, int64_t TotNumPart, int64_t TotNuPart, double nufrac);
void  write_particle_data(int Type, BigFile * bf);

void  read_parameterfile(char *fname);
#endif
