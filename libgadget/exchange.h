#ifndef __EXCHANGE_H
#define __EXCHANGE_H

#include "partmanager.h"
#include "slotsmanager.h"

typedef int (*ExchangeLayoutFunc) (int p, const void * userdata);

int domain_exchange(ExchangeLayoutFunc, const void * layout_userdata, int do_gc, struct part_manager_type * pman, struct slots_manager_type * sman, int maxiter, MPI_Comm Comm);
void domain_test_id_uniqueness(struct part_manager_type * pman);

#endif
