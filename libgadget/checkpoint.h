#ifndef CHECKPOINT_H
#define CHECKPOINT_H

#include "forcetree.h"

void write_checkpoint(int snapnum, int WriteSnapshot, int WriteFOF, double Time, const char * OutputDir, const char * SnapshotFileBase, const int OutputDebugFields, ForceTree * tree);
void dump_snapshot(const char * dump, const char * OutputDir);
int find_last_snapnum(const char * OutputDir);

#endif
