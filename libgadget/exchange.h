#ifndef __EXCHANGE_H
#define __EXCHANGE_H

int domain_exchange(int (*layoutfunc)(int p), int do_gc);
void domain_test_id_uniqueness();

#endif
