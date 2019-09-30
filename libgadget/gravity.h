#ifndef LONGRANGE_H
#define LONGRANGE_H

#include "forcetree.h"

/* Fill the short-range gravity table*/
void gravshort_fill_ntab(void);

/*Defined in gravpm.c*/
void gravpm_init_periodic(double BoxSize, int Nmesh);

int
grav_apply_short_range_window(double r, double * fac, double * pot);

/*Note: tree is rebuilt during this function*/
void gravpm_force(ForceTree * tree);

void grav_short_pair(ForceTree * tree);
void grav_short_tree(ForceTree * tree);
#endif
