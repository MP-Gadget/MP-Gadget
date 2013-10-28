#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "allvars.h"
#include "proto.h"
#include "domain.h"

#include <gsl/gsl_heapsort.h>




static struct peano_hilbert_data
{
  peanokey key;
  int index;
}
 *mp;

static int *Id;

void peano_hilbert_order(void)
{
  int i;

  if(ThisTask == 0)
    printf("begin Peano-Hilbert order...\n");

  if(N_sph)
    {
      mp = (struct peano_hilbert_data *) mymalloc("mp", sizeof(struct peano_hilbert_data) * N_sph);
      Id = (int *) mymalloc("Id", sizeof(int) * N_sph);

      for(i = 0; i < N_sph; i++)
	{
	  mp[i].index = i;
	  mp[i].key = KEY(i);
	}

#ifdef MYSORT
      mysort_peano(mp, N_sph, sizeof(struct peano_hilbert_data), peano_compare_key);
#else
      qsort(mp, N_sph, sizeof(struct peano_hilbert_data), peano_compare_key);
#endif

      for(i = 0; i < N_sph; i++)
	Id[mp[i].index] = i;

      reorder_gas();

      myfree(Id);
      myfree(mp);
    }


  if(NumPart - N_sph > 0)
    {
      mp =
	(struct peano_hilbert_data *) mymalloc("mp", sizeof(struct peano_hilbert_data) * (NumPart - N_sph));
      mp -= (N_sph);

      Id = (int *) mymalloc("Id", sizeof(int) * (NumPart - N_sph));
      Id -= (N_sph);

      for(i = N_sph; i < NumPart; i++)
	{
	  mp[i].index = i;
	  mp[i].key = KEY(i);
	}

#ifdef MYSORT
      mysort_peano(mp + N_sph, NumPart - N_sph, sizeof(struct peano_hilbert_data), peano_compare_key);
#else
      qsort(mp + N_sph, NumPart - N_sph, sizeof(struct peano_hilbert_data), peano_compare_key);
#endif

      for(i = N_sph; i < NumPart; i++)
	Id[mp[i].index] = i;

      reorder_particles();

      Id += N_sph;
      myfree(Id);
      mp += N_sph;
      myfree(mp);
    }

  if(ThisTask == 0)
    printf("Peano-Hilbert done.\n");
}


int peano_compare_key(const void *a, const void *b)
{
  if(((struct peano_hilbert_data *) a)->key < (((struct peano_hilbert_data *) b)->key))
    return -1;

  if(((struct peano_hilbert_data *) a)->key > (((struct peano_hilbert_data *) b)->key))
    return +1;

  return 0;
}

void reorder_gas(void)
{
  int i;
  struct particle_data Psave, Psource;
  struct sph_particle_data SphPsave, SphPsource;
  int idsource, idsave, dest;

  for(i = 0; i < N_sph; i++)
    {
      if(Id[i] != i)
	{
	  Psource = P[i];
	  SphPsource = SPHP(i);

	  idsource = Id[i];
	  dest = Id[i];

	  do
	    {
	      Psave = P[dest];
	      SphPsave = SPHP(dest);
	      idsave = Id[dest];

	      P[dest] = Psource;
	      SPHP(dest) = SphPsource;
	      Id[dest] = idsource;

	      if(dest == i)
		break;

	      Psource = Psave;
	      SphPsource = SphPsave;
	      idsource = idsave;

	      dest = idsource;
	    }
	  while(1);
	}
    }
}


void reorder_particles(void)
{
  int i;
  struct particle_data Psave, Psource;
  int idsource, idsave, dest;

  for(i = N_sph; i < NumPart; i++)
    {
      if(Id[i] != i)
	{
	  Psource = P[i];
	  idsource = Id[i];

	  dest = Id[i];

	  do
	    {
	      Psave = P[dest];
	      idsave = Id[dest];

	      P[dest] = Psource;
	      Id[dest] = idsource;
	      if(dest == i)
		break;

	      Psource = Psave;
	      idsource = idsave;

	      dest = idsource;
	    }
	  while(1);
	}
    }
}





/*  The following rewrite of the original function
 *  peano_hilbert_key_old() has been written by MARTIN REINECKE. 
 *  It is about a factor 2.3 - 2.5 faster than Volker's old routine!
 */
const unsigned char rottable3[48][8] = {
  {36, 28, 25, 27, 10, 10, 25, 27},
  {29, 11, 24, 24, 37, 11, 26, 26},
  {8, 8, 25, 27, 30, 38, 25, 27},
  {9, 39, 24, 24, 9, 31, 26, 26},
  {40, 24, 44, 32, 40, 6, 44, 6},
  {25, 7, 33, 7, 41, 41, 45, 45},
  {4, 42, 4, 46, 26, 42, 34, 46},
  {43, 43, 47, 47, 5, 27, 5, 35},
  {33, 35, 36, 28, 33, 35, 2, 2},
  {32, 32, 29, 3, 34, 34, 37, 3},
  {33, 35, 0, 0, 33, 35, 30, 38},
  {32, 32, 1, 39, 34, 34, 1, 31},
  {24, 42, 32, 46, 14, 42, 14, 46},
  {43, 43, 47, 47, 25, 15, 33, 15},
  {40, 12, 44, 12, 40, 26, 44, 34},
  {13, 27, 13, 35, 41, 41, 45, 45},
  {28, 41, 28, 22, 38, 43, 38, 22},
  {42, 40, 23, 23, 29, 39, 29, 39},
  {41, 36, 20, 36, 43, 30, 20, 30},
  {37, 31, 37, 31, 42, 40, 21, 21},
  {28, 18, 28, 45, 38, 18, 38, 47},
  {19, 19, 46, 44, 29, 39, 29, 39},
  {16, 36, 45, 36, 16, 30, 47, 30},
  {37, 31, 37, 31, 17, 17, 46, 44},
  {12, 4, 1, 3, 34, 34, 1, 3},
  {5, 35, 0, 0, 13, 35, 2, 2},
  {32, 32, 1, 3, 6, 14, 1, 3},
  {33, 15, 0, 0, 33, 7, 2, 2},
  {16, 0, 20, 8, 16, 30, 20, 30},
  {1, 31, 9, 31, 17, 17, 21, 21},
  {28, 18, 28, 22, 2, 18, 10, 22},
  {19, 19, 23, 23, 29, 3, 29, 11},
  {9, 11, 12, 4, 9, 11, 26, 26},
  {8, 8, 5, 27, 10, 10, 13, 27},
  {9, 11, 24, 24, 9, 11, 6, 14},
  {8, 8, 25, 15, 10, 10, 25, 7},
  {0, 18, 8, 22, 38, 18, 38, 22},
  {19, 19, 23, 23, 1, 39, 9, 39},
  {16, 36, 20, 36, 16, 2, 20, 10},
  {37, 3, 37, 11, 17, 17, 21, 21},
  {4, 17, 4, 46, 14, 19, 14, 46},
  {18, 16, 47, 47, 5, 15, 5, 15},
  {17, 12, 44, 12, 19, 6, 44, 6},
  {13, 7, 13, 7, 18, 16, 45, 45},
  {4, 42, 4, 21, 14, 42, 14, 23},
  {43, 43, 22, 20, 5, 15, 5, 15},
  {40, 12, 21, 12, 40, 6, 23, 6},
  {13, 7, 13, 7, 41, 41, 22, 20}
};

const unsigned char subpix3[48][8] = {
  {0, 7, 1, 6, 3, 4, 2, 5},
  {7, 4, 6, 5, 0, 3, 1, 2},
  {4, 3, 5, 2, 7, 0, 6, 1},
  {3, 0, 2, 1, 4, 7, 5, 6},
  {1, 0, 6, 7, 2, 3, 5, 4},
  {0, 3, 7, 4, 1, 2, 6, 5},
  {3, 2, 4, 5, 0, 1, 7, 6},
  {2, 1, 5, 6, 3, 0, 4, 7},
  {6, 1, 7, 0, 5, 2, 4, 3},
  {1, 2, 0, 3, 6, 5, 7, 4},
  {2, 5, 3, 4, 1, 6, 0, 7},
  {5, 6, 4, 7, 2, 1, 3, 0},
  {7, 6, 0, 1, 4, 5, 3, 2},
  {6, 5, 1, 2, 7, 4, 0, 3},
  {5, 4, 2, 3, 6, 7, 1, 0},
  {4, 7, 3, 0, 5, 6, 2, 1},
  {6, 7, 5, 4, 1, 0, 2, 3},
  {7, 0, 4, 3, 6, 1, 5, 2},
  {0, 1, 3, 2, 7, 6, 4, 5},
  {1, 6, 2, 5, 0, 7, 3, 4},
  {2, 3, 1, 0, 5, 4, 6, 7},
  {3, 4, 0, 7, 2, 5, 1, 6},
  {4, 5, 7, 6, 3, 2, 0, 1},
  {5, 2, 6, 1, 4, 3, 7, 0},
  {7, 0, 6, 1, 4, 3, 5, 2},
  {0, 3, 1, 2, 7, 4, 6, 5},
  {3, 4, 2, 5, 0, 7, 1, 6},
  {4, 7, 5, 6, 3, 0, 2, 1},
  {6, 7, 1, 0, 5, 4, 2, 3},
  {7, 4, 0, 3, 6, 5, 1, 2},
  {4, 5, 3, 2, 7, 6, 0, 1},
  {5, 6, 2, 1, 4, 7, 3, 0},
  {1, 6, 0, 7, 2, 5, 3, 4},
  {6, 5, 7, 4, 1, 2, 0, 3},
  {5, 2, 4, 3, 6, 1, 7, 0},
  {2, 1, 3, 0, 5, 6, 4, 7},
  {0, 1, 7, 6, 3, 2, 4, 5},
  {1, 2, 6, 5, 0, 3, 7, 4},
  {2, 3, 5, 4, 1, 0, 6, 7},
  {3, 0, 4, 7, 2, 1, 5, 6},
  {1, 0, 2, 3, 6, 7, 5, 4},
  {0, 7, 3, 4, 1, 6, 2, 5},
  {7, 6, 4, 5, 0, 1, 3, 2},
  {6, 1, 5, 2, 7, 0, 4, 3},
  {5, 4, 6, 7, 2, 3, 1, 0},
  {4, 3, 7, 0, 5, 2, 6, 1},
  {3, 2, 0, 1, 4, 5, 7, 6},
  {2, 5, 1, 6, 3, 4, 0, 7}
};

/*! This function computes a Peano-Hilbert key for an integer triplet (x,y,z),
  *  with x,y,z in the range between 0 and 2^bits-1.
  */
peanokey peano_hilbert_key(int x, int y, int z, int bits)
{
  int mask;
  unsigned char rotation = 0;
  peanokey key = 0;

  for(mask = 1 << (bits - 1); mask > 0; mask >>= 1)
    {
      unsigned char pix = ((x & mask) ? 4 : 0) | ((y & mask) ? 2 : 0) | ((z & mask) ? 1 : 0);

      key <<= 3;
      key |= subpix3[rotation][pix];
      rotation = rottable3[rotation][pix];
    }

  return key;
}



peanokey morton_key(int x, int y, int z, int bits)
{
  int mask;
  peanokey morton = 0;

  for(mask = 1 << (bits - 1); mask > 0; mask >>= 1)
    {
      morton <<= 3;
      morton += ((z & mask) ? 4 : 0) + ((y & mask) ? 2 : 0) + ((x & mask) ? 1 : 0);
    }

  return morton;
}


peanokey peano_and_morton_key(int x, int y, int z, int bits, peanokey * morton_key)
{
  int mask;
  unsigned char rotation = 0;
  peanokey key = 0;
  peanokey morton = 0;


  for(mask = 1 << (bits - 1); mask > 0; mask >>= 1)
    {
      unsigned char pix = ((x & mask) ? 4 : 0) | ((y & mask) ? 2 : 0) | ((z & mask) ? 1 : 0);

      key <<= 3;
      key |= subpix3[rotation][pix];
      rotation = rottable3[rotation][pix];

      morton <<= 3;
      morton += ((z & mask) ? 4 : 0) + ((y & mask) ? 2 : 0) + ((x & mask) ? 1 : 0);
    }

  *morton_key = morton;

  return key;
}




static int quadrants[24][2][2][2] = {
  /* rotx=0, roty=0-3 */
  {{{0, 7}, {1, 6}}, {{3, 4}, {2, 5}}},
  {{{7, 4}, {6, 5}}, {{0, 3}, {1, 2}}},
  {{{4, 3}, {5, 2}}, {{7, 0}, {6, 1}}},
  {{{3, 0}, {2, 1}}, {{4, 7}, {5, 6}}},
  /* rotx=1, roty=0-3 */
  {{{1, 0}, {6, 7}}, {{2, 3}, {5, 4}}},
  {{{0, 3}, {7, 4}}, {{1, 2}, {6, 5}}},
  {{{3, 2}, {4, 5}}, {{0, 1}, {7, 6}}},
  {{{2, 1}, {5, 6}}, {{3, 0}, {4, 7}}},
  /* rotx=2, roty=0-3 */
  {{{6, 1}, {7, 0}}, {{5, 2}, {4, 3}}},
  {{{1, 2}, {0, 3}}, {{6, 5}, {7, 4}}},
  {{{2, 5}, {3, 4}}, {{1, 6}, {0, 7}}},
  {{{5, 6}, {4, 7}}, {{2, 1}, {3, 0}}},
  /* rotx=3, roty=0-3 */
  {{{7, 6}, {0, 1}}, {{4, 5}, {3, 2}}},
  {{{6, 5}, {1, 2}}, {{7, 4}, {0, 3}}},
  {{{5, 4}, {2, 3}}, {{6, 7}, {1, 0}}},
  {{{4, 7}, {3, 0}}, {{5, 6}, {2, 1}}},
  /* rotx=4, roty=0-3 */
  {{{6, 7}, {5, 4}}, {{1, 0}, {2, 3}}},
  {{{7, 0}, {4, 3}}, {{6, 1}, {5, 2}}},
  {{{0, 1}, {3, 2}}, {{7, 6}, {4, 5}}},
  {{{1, 6}, {2, 5}}, {{0, 7}, {3, 4}}},
  /* rotx=5, roty=0-3 */
  {{{2, 3}, {1, 0}}, {{5, 4}, {6, 7}}},
  {{{3, 4}, {0, 7}}, {{2, 5}, {1, 6}}},
  {{{4, 5}, {7, 6}}, {{3, 2}, {0, 1}}},
  {{{5, 2}, {6, 1}}, {{4, 3}, {7, 0}}}
};


static int rotxmap_table[24] = { 4, 5, 6, 7, 8, 9, 10, 11,
  12, 13, 14, 15, 0, 1, 2, 3, 17, 18, 19, 16, 23, 20, 21, 22
};

static int rotymap_table[24] = { 1, 2, 3, 0, 16, 17, 18, 19,
  11, 8, 9, 10, 22, 23, 20, 21, 14, 15, 12, 13, 4, 5, 6, 7
};

static int rotx_table[8] = { 3, 0, 0, 2, 2, 0, 0, 1 };
static int roty_table[8] = { 0, 1, 1, 2, 2, 3, 3, 0 };

static int sense_table[8] = { -1, -1, -1, +1, +1, -1, -1, -1 };


peanokey peano_hilbert_key_old(int x, int y, int z, int bits)
{
  int i, quad, bitx, bity, bitz;
  int mask, rotation, rotx, roty, sense;
  peanokey key;


  mask = 1 << (bits - 1);
  key = 0;
  rotation = 0;
  sense = 1;


  for(i = 0; i < bits; i++, mask >>= 1)
    {
      bitx = (x & mask) ? 1 : 0;
      bity = (y & mask) ? 1 : 0;
      bitz = (z & mask) ? 1 : 0;

      quad = quadrants[rotation][bitx][bity][bitz];

      key <<= 3;
      key += (sense == 1) ? (quad) : (7 - quad);

      rotx = rotx_table[quad];
      roty = roty_table[quad];
      sense *= sense_table[quad];

      while(rotx > 0)
	{
	  rotation = rotxmap_table[rotation];
	  rotx--;
	}

      while(roty > 0)
	{
	  rotation = rotymap_table[rotation];
	  roty--;
	}
    }

  return key;
}


peanokey peano_and_morton_key_old(int x, int y, int z, int bits, peanokey * morton_key)
{
  int i, quad, bitx, bity, bitz;
  int mask, rotation, rotx, roty, sense;
  peanokey key, morton;


  mask = 1 << (bits - 1);
  key = 0;
  rotation = 0;
  sense = 1;
  morton = 0;

  for(i = 0; i < bits; i++, mask >>= 1)
    {
      bitx = (x & mask) ? 1 : 0;
      bity = (y & mask) ? 1 : 0;
      bitz = (z & mask) ? 1 : 0;

      quad = quadrants[rotation][bitx][bity][bitz];

      key <<= 3;
      key += (sense == 1) ? (quad) : (7 - quad);

      rotx = rotx_table[quad];
      roty = roty_table[quad];
      sense *= sense_table[quad];

      while(rotx > 0)
	{
	  rotation = rotxmap_table[rotation];
	  rotx--;
	}

      while(roty > 0)
	{
	  rotation = rotymap_table[rotation];
	  roty--;
	}

      morton <<= 3;
      morton += (bitz << 2) + (bity << 1) + bitx;
    }

  *morton_key = morton;

  return key;
}


peanokey morton_key_old(int x, int y, int z, int bits)
{
  int i, bitx, bity, bitz;
  int mask;
  peanokey morton;


  mask = 1 << (bits - 1);
  morton = 0;

  for(i = 0; i < bits; i++, mask >>= 1)
    {
      bitx = (x & mask) ? 1 : 0;
      bity = (y & mask) ? 1 : 0;
      bitz = (z & mask) ? 1 : 0;

      morton <<= 3;
      morton += (bitz << 2) + (bity << 1) + bitx;
    }

  return morton;
}


static void msort_peano_with_tmp(struct peano_hilbert_data *b, size_t n, struct peano_hilbert_data *t)
{
  struct peano_hilbert_data *tmp;
  struct peano_hilbert_data *b1, *b2;
  size_t n1, n2;

  if(n <= 1)
    return;

  n1 = n / 2;
  n2 = n - n1;
  b1 = b;
  b2 = b + n1;

  msort_peano_with_tmp(b1, n1, t);
  msort_peano_with_tmp(b2, n2, t);

  tmp = t;

  while(n1 > 0 && n2 > 0)
    {
      if(b1->key <= b2->key)
	{
	  --n1;
	  *tmp++ = *b1++;
	}
      else
	{
	  --n2;
	  *tmp++ = *b2++;
	}
    }

  if(n1 > 0)
    memcpy(tmp, b1, n1 * sizeof(struct peano_hilbert_data));
  memcpy(b, t, (n - n2) * sizeof(struct peano_hilbert_data));
}

void mysort_peano(void *b, size_t n, size_t s, int (*cmp) (const void *, const void *))
{
  const size_t size = n * s;

  struct peano_hilbert_data *tmp =
    (struct peano_hilbert_data *) mymalloc("struct peano_hilbert_data *tmp", size);

  msort_peano_with_tmp((struct peano_hilbert_data *) b, n, tmp);

  myfree(tmp);
}
