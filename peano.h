#ifndef PEANO_H
#define PEANO_H

typedef uint64_t peanokey;

struct peano_hilbert_data
{
  peanokey key;
  int index;
};

#define  BITS_PER_DIMENSION 21	/* for Peano-Hilbert order. Note: Maximum is 10 to fit in 32-bit integer ! */
#define  PEANOCELLS (((peanokey)1)<<(3*BITS_PER_DIMENSION))

#define KEY(i) peano_hilbert_key((int) ((P[i].Pos[0] - DomainCorner[0]) * DomainFac), \
        (int) ((P[i].Pos[1] - DomainCorner[1]) * DomainFac), \
        (int) ((P[i].Pos[2] - DomainCorner[2]) * DomainFac), \
        BITS_PER_DIMENSION)

void mysort_peano(void *b, size_t n, size_t s, int (*cmp) (const void *, const void *));

void init_peano_map(void);
peanokey peano_hilbert_key(int x, int y, int z, int bits);
peanokey peano_and_morton_key(int x, int y, int z, int bits, peanokey *morton);
peanokey morton_key(int x, int y, int z, int bits);

int peano_compare_key(const void *a, const void *b);
#endif
