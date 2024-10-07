/*Simple test for the Peano hilbert key function*/

#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>
#include <math.h>
#include <mpi.h>
#include <stdio.h>

#include <libgadget/utils/peano.h>
#include "stub.h"

/* This is the old Peano Hilbert key routine from Gadget-2.
 * It is 3-4x slower than the one in peano.c*/
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

peano_t peano_hilbert_key_old(const int x, const int y, const int z, const int bits)
{
  int i, quad, bitx, bity, bitz;
  int mask, rotation, rotx, roty, sense;
  peano_t key;


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

peano_t result_keys[] = {6020610249, 483815267677980425, 3870522191621343542, 3767582776533235958, 1132333615783364415, 627930462752356607, 3891110085746456438, 3947726760453646118, 1152921506754357558, 2285255124148295382, 2305843012818375753, 2326430894279092521, 1297036690084656374, 2182315706524411686, 2449958193349321865, 2429370311902976217, 144115192773494591, 288230379909822719, 4230810164830870390, 4271985931826743040, 952189634125453529, 808074446989125401, 4035225267010758950, 4091841947589974246, 1636736773291449206, 1780851966333379734, 2789658282730152585, 2830834052094008169, 1641883751239952166, 1785998938376280294, 2794805254773053145, 2825687080051107609, 9058668959936226486, 8914553775484263254, 5023443698797515263, 4920504278302312767, 8266035424780819318, 8410150617822749846, 5044031583709523465, 5100648263986747097, 7617517078103921737, 7473401895799450537, 6464595579402677686, 6361656159482713174, 7638104960322013321, 7493989779057710809, 6485183462358946294, 6341068276224452902, 9079256844081281216, 8935141656944953088, 4992561872023905417, 4951386105028032767, 8271182402729322278, 8415297589865650406, 5188146769844016857, 5131530089264801561, 7586635263563326601, 7442520070521396073, 6433713754124623222, 6392537984760767638, 7581488285614823641, 7437373098478495513, 6428566782081722662, 6397684956803668198};

static void
test_peano(void **state)
{
    int i,j,k;
    /* Check against some known good results*/
    int Box = 4;
    for(i = 0; i < Box * Box * Box; i++) {
        double Pos[3] = {i % Box, (i / Box) % Box, (i / Box / Box) % Box};
        peano_t Key = PEANO(Pos, Box);
        assert_true(result_keys[i] == Key);
        // printf("K = %ld\n", Key);
    }

    double start = MPI_Wtime();
    for(i = 1; i < BITS_PER_DIMENSION-1; i++) {
        for(j = 1; j < BITS_PER_DIMENSION-1; j++) {
            for(k = 1; k < BITS_PER_DIMENSION-1; k++) {
                peano_t Key = peano_hilbert_key(1<<i, 1<<j, 1<<k, BITS_PER_DIMENSION);
                peano_t Key2 = peano_hilbert_key_old(1<<i, 1<<j, 1<<k, BITS_PER_DIMENSION);
                assert_true(Key == Key2);
            }
        }
    }

    /* Check speed*/
    Box = 100;
    peano_t Key = 0;
    for(i = 0; i < Box * Box * Box; i++) {
        double Pos[3] = {i % Box, (i / Box) % Box, (i / Box / Box) % Box};
        Key += PEANO(Pos, Box);
    }
    double end = MPI_Wtime();
    double ms = (end - start)*1000;
    printf("Computed %d keys in %.3g ms (sum %ld)\n", Box*Box*Box, ms, Key);

    return;
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_peano),
    };
    return cmocka_run_group_tests_mpi(tests, NULL, NULL);
}
