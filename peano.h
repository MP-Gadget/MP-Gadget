#ifndef PEANO_H
#define PEANO_H

#include <stddef.h>
#include <stdint.h>

typedef uint64_t peano_t;
typedef uint64_t morton_t;

#define  BITS_PER_DIMENSION 21	/* for Peano-Hilbert order. Note: Maximum is 10 to fit in 32-bit integer ! */
#define  PEANOCELLS (((peano_t)1)<<(3*BITS_PER_DIMENSION))

#define DomainFac(len) ( 1.0 / (len) * (((peano_t) 1) << (BITS_PER_DIMENSION)))

#define PEANO(Pos) peano_hilbert_key((int) ((Pos[0] + All.BoxSize/2000) * DomainFac(All.BoxSize*1.001)), \
        (int) ((Pos[1] + All.BoxSize/2000) * DomainFac(All.BoxSize*1.001)), \
        (int) ((Pos[2] + All.BoxSize/2000) * DomainFac(All.BoxSize*1.001)), \
        BITS_PER_DIMENSION)

#define MORTON(Pos) morton_key((int) ((Pos[0] + All.BoxSize/2000) * DomainFac(All.BoxSize*1.001)), \
        (int) ((Pos[1] + All.BoxSize/2000) * DomainFac(All.BoxSize*1.001)), \
        (int) ((Pos[2] + All.BoxSize/2000) * DomainFac(All.BoxSize*1.001)), \
        BITS_PER_DIMENSION)

void mysort_peano(void *b, size_t n, size_t s, int (*cmp) (const void *, const void *));

void init_peano_map(void);
peano_t peano_hilbert_key(int x, int y, int z, int bits);
morton_t morton_key(int x, int y, int z, int bits);

#endif
