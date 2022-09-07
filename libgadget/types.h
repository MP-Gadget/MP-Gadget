#ifndef __TYPES_H
#define __TYPES_H

#include <stdint.h>

/*Define some useful types*/

typedef int64_t inttime_t;

typedef uint64_t MyIDType;

#ifndef LOW_PRECISION
#define LOW_PRECISION double
#endif
#ifndef HIGH_PRECISION
#define HIGH_PRECISION double
#endif

typedef LOW_PRECISION MyFloat;
typedef HIGH_PRECISION MyDouble;

#define HAS(val, flag) ((flag & (val)) == (flag))

#endif
