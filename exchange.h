#ifndef __EXCHANGE_H
#define __EXCHANGE_H
enum ExchangeType {
    EXCHANGE_INCREMENTAL = 0,
    EXCHANGE_FULL = 1,
};

int domain_exchange(int (*layoutfunc)(int p), enum ExchangeType exchange_type);

#endif
