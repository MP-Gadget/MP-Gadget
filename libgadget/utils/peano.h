#ifndef PEANO_H
#define PEANO_H

#include <stddef.h>
#include <stdint.h>

typedef uint64_t peano_t;

#define  BITS_PER_DIMENSION 21	/* for Peano-Hilbert order. Note: Maximum is 21 to fit in 64-bit integer ! */
#define  PEANOCELLS (((peano_t)1)<<(3*BITS_PER_DIMENSION))

peano_t peano_hilbert_key(const int x, const int y, const int z, const int bits);

static inline peano_t PEANO(const double * const Pos, const double BoxSize)
{
    /*No reason known for the Box/2000 and 1.001 factors*/
    const double DomainFac = 1.0 / (BoxSize*1.001) * (((peano_t) 1) << (BITS_PER_DIMENSION));
    const double spos[3] = {Pos[0] + BoxSize/2000, Pos[1] + BoxSize/2000, Pos[2] + BoxSize/2000};
    return peano_hilbert_key(spos[0]*DomainFac, spos[1]*DomainFac, spos[2]*DomainFac, BITS_PER_DIMENSION);
}

#endif
