#ifndef GENIC_PROTO_H
#define GENIC_PROTO_H
#include <bigfile.h>

void shift_particles(double shift, int64_t FirstID);
void   displacement_fields(int Type);
void   setup_grid(double shift);
void   free_ffts(void);

void saveheader(BigFile * bf, int64_t TotNumPart);
void  write_particle_data(int Type, BigFile * bf);

void  read_parameterfile(char *fname);
#endif
