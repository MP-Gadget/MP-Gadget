#ifndef PEANO_H
#define PEANO_H

#include <stddef.h>
#include <stdint.h>

//typedef uint64_t peano_t;
//#define PEANOT_MAX UINT64_MAX
#define  BITS_PER_DIMENSION 64
//#define  PEANOCELLS (((peano_t)1)<<(3*BITS_PER_DIMENSION))

typedef struct {
    /* 'hs'-high significance, 'is'-intermediate, 'ls'-low significance bits */
    uint64_t hs;
    uint64_t is;
    uint64_t ls; 
} peano_t;

static const peano_t PEANOT_MAX = {
    .hs = UINT64_MAX,
    .is = UINT64_MAX,
    .ls = UINT64_MAX
};

static inline peano_t get_peanocells(void) {
    peano_t cells = {0};
    int total_bits = BITS_PER_DIMENSION;
    cells.hs = ((uint64_t)1) << (total_bits);
    return cells;
}

#define PEANOCELLS (get_peanocells())

peano_t peano_hilbert_key(const int x, const int y, const int z, const int bits);

static inline peano_t PEANO(const double * const Pos, const double BoxSize)
{
    /*No reason known for the Box/2000 and 1.001 factors*/
    const double DomainFac = 1.0 / (BoxSize*1.001) * (((uint64_t) 1) << (BITS_PER_DIMENSION - 2));
    const double spos[3] = {Pos[0] + BoxSize/2000, Pos[1] + BoxSize/2000, Pos[2] + BoxSize/2000};
    return peano_hilbert_key(spos[0]*DomainFac, spos[1]*DomainFac, spos[2]*DomainFac, BITS_PER_DIMENSION);
}

unsigned char peano_incremental_key(unsigned char pix, unsigned char *rotation);


inline peano_t get_peanokey_offset(unsigned int j, int bits) /* this returns the peanokey for which  j << bits */
{
  peano_t key = {j, j, j};

  if(bits < BITS_PER_DIMENSION)
    key.ls <<= bits;
  else
    key.ls = 0;

  int is_bits = bits - BITS_PER_DIMENSION;

  if(is_bits <= -BITS_PER_DIMENSION)
    key.is = 0;
  else if(is_bits <= 0)
    key.is >>= -is_bits;
  else if(is_bits < BITS_PER_DIMENSION)
    key.is <<= is_bits;
  else
    key.is = 0;

  int hs_bits = bits - 2 * BITS_PER_DIMENSION;

  if(hs_bits <= -BITS_PER_DIMENSION)
    key.hs = 0;
  else if(hs_bits <= 0)
    key.hs >>= -hs_bits;
  else if(hs_bits < BITS_PER_DIMENSION)
    key.hs <<= hs_bits;
  else
    key.hs = 0;

  return key;
}

inline peano_t add_peano_key(const peano_t a, const peano_t b)
{
  peano_t c;

  c.ls = a.ls + b.ls;
  c.is = a.is + b.is;
  c.hs = a.hs + b.hs;

  if(c.is < a.is || c.is < b.is) /* overflow has occurred */
    {
      c.hs += 1;
    }

  if(c.ls < a.ls || c.ls < b.ls) /* overflow has occurred */
    {
      c.is += 1;
      if(c.is == 0) /* overflown again */
        c.hs += 1;
    }

  /* note: for hs we don't check for overflow explicitly as this would not be represented in the type anyhow */

  return c;
}

inline peano_t subtract_peano_key(const peano_t a, const peano_t b)
{
    peano_t c;
    c.ls = a.ls - b.ls;
    c.is = a.is - b.is;
    c.hs = a.hs - b.hs;

    /* Handle underflow for least significant bits */
    if(a.ls < b.ls) {
        c.is -= 1;
        /* If is underflowed to max value, we need to decrease hs */
        if(c.is == UINT64_MAX)
            c.hs -= 1;
    }

    /* Handle underflow for intermediate significance bits */
    if(a.is < b.is || (a.ls < b.ls && a.is == 0)) {
        c.hs -= 1;
    }

    return c;
}

#endif
