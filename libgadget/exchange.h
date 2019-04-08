#ifndef __EXCHANGE_H
#define __EXCHANGE_H

typedef int (*ExchangeLayoutFunc) (int p, const void * userdata);

int domain_exchange(ExchangeLayoutFunc, const void * layout_userdata, int do_gc);
void domain_test_id_uniqueness();

#endif
