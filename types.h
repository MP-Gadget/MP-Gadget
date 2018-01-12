#ifndef __TYPES_H
#define __TYPES_H

#include <stdint.h>

/*Define some useful types*/

/* Note on a 32-bit architecture MPI_LONG may be 32-bit,
 * so these should be MPI_LONG_LONG. But in
 * the future MPI_LONG_LONG may become 128-bit.*/
#define MPI_UINT64 MPI_UNSIGNED_LONG
#define MPI_INT64 MPI_LONG

typedef uint32_t binmask_t;
typedef int32_t inttime_t;

#define BINMASK_ALL ((uint32_t) (-1))
#define BINMASK(i) (1u << i)

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
