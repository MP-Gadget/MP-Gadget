#ifndef BHINFO_H
#define BHINFO_H

#include "bhdynfric.h"
#include "blackhole.h"

/* Writes a packed binary structure of detailed black hole information to disc*/
void collect_BH_info(const int * const ActiveBlackHoles, const int64_t NumActiveBlackHoles, struct BHPriv *priv, const struct part_manager_type * const PartManager, const struct bh_particle_data * const BHManager, FILE * FdBlackholeDetails);

void write_blackhole_txt(FILE * FdBlackHoles, const struct UnitSystem units, const double atime);

#endif
