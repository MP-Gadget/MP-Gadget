#ifndef CHECKPOINT_H
#define CHECKPOINT_H

#include "forcetree.h"

void write_checkpoint(int WriteSnapshot, int WriteFOF, struct OctTree * tree);
void dump_snapshot(void);
int find_last_snapnum();

#endif
