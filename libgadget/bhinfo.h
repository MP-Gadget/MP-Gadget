#ifndef BHINFO_H
#define BHINFO_H

#include "bhdynfric.h"
#include "blackhole.h"

/* Writes a packed binary structure of detailed black hole information to disc*/
void collect_BH_info(int * ActiveBlackHoles, int NumActiveBlackHoles, struct BHPriv *priv, FILE * FdBlackholeDetails);

void write_blackhole_txt(FILE * FdBlackHoles, const struct UnitSystem units, const double atime);

#endif
