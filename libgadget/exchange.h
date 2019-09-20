#ifndef __EXCHANGE_H
#define __EXCHANGE_H

#include "partmanager.h"

typedef int (*ExchangeLayoutFunc) (int p, const void * userdata);

int domain_exchange(ExchangeLayoutFunc, const void * layout_userdata, int do_gc, struct part_manager_type * PartManager, MPI_Comm Comm);
void domain_test_id_uniqueness(struct part_manager_type * PartManager);

#endif
