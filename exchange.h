#ifndef __EXCHANGE_H
#define __EXCHANGE_H

void domain_count_particles();
void domain_exchange(int (*layoutfunc)(int p));
void domain_refresh_totals();

#endif
