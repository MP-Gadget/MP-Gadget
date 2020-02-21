#ifndef CHECKPOINT_H
#define CHECKPOINT_H

#include "forcetree.h"

void write_checkpoint(int WriteSnapshot, int WriteFOF, ForceTree * tree);
void dump_snapshot(const char * dump);
int find_last_snapnum(const char * OutputDir);

#endif
