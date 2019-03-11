#ifndef GENIC_PROTO_H
#define GENIC_PROTO_H
#include <bigfile.h>
#include <stdint.h>
#include "power.h"

void   displacement_fields(enum TransferType Type);
void setup_grid(double shift, int Ngrid);
void setup_glass(double shift, int Ngrid, int seed);
uint64_t id_offset_from_index(const int i, const int Ngrid);
void   free_ffts(void);

void saveheader(BigFile * bf, int64_t TotNumPart, int64_t TotNuPart, double nufrac);
void  write_particle_data(const int Type, BigFile * bf,  const uint64_t FirstID, const int Ngrid);

void  read_parameterfile(char *fname);
#endif
