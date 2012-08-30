#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "allvars.h"
#include "proto.h"


#define MAXBLOCKS 256

static unsigned long Nblocks = 0;

static void *Table[MAXBLOCKS];

static size_t SizeTable[MAXBLOCKS];

static size_t TotMem = 0, HighMarkMem = 0;


void *mymalloc(size_t n)
{
  if((n % 8) > 0)
    n = (n / 8 + 1) * 8;

  if(Nblocks >= MAXBLOCKS)
    {
      printf("No blocks left in mymalloc().\n");
      exit(1);
    }

  SizeTable[Nblocks] = n;
  TotMem += n;
  if(TotMem > HighMarkMem)
    {
      HighMarkMem = TotMem;
      /*
         printf("new high mark = %g MB\n", HighMarkMem / (1024.0 * 1024.0));
       */
    }

  if(!(Table[Nblocks] = malloc(n)))
    {
      printf("Failed to allocate memory for %u bytes.\n", (int) n);
      exit(2);
    }

  Nblocks += 1;

  return Table[Nblocks - 1];
}


void myfree(void *p)
{
  if(Nblocks == 0)
    exit(1);

  if(p != Table[Nblocks - 1])
    {
      printf("Wrong call of myfree() - not the last allocated block!\n");
      exit(1);
    }

  free(p);

  Nblocks -= 1;

  TotMem -= SizeTable[Nblocks];
}
