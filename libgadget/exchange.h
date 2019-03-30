#ifndef __EXCHANGE_H
#define __EXCHANGE_H

int domain_exchange(int (*layoutfunc)(int p, const void * userdata), const void * layout_userdata, int do_gc);
void domain_test_id_uniqueness();

#endif
