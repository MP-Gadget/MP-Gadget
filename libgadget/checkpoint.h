#ifndef CHECKPOINT_H
#define CHECKPOINT_H
#include "cosmology.h"

void write_checkpoint(int snapnum, int WriteGroupID, int MetalReturnOn, double Time, const Cosmology * CP, const char * OutputDir, const int OutputDebugFields);
void dump_snapshot(const char * dump, const double Time, const Cosmology * CP, const char * OutputDir);
int find_last_snapnum(const char * OutputDir);

#endif
