#ifndef LONGRANGE_H
#define LONGRANGE_H

void grav_init(void);

int
grav_apply_short_range_window(double r, double * fac, double * pot);

void gravpm_force(void);

void grav_short_pair(void);
void grav_short_tree(void);
#endif
