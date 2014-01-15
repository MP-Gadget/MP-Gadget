#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <gsl/gsl_math.h>

#include "allvars.h"
#include "proto.h"


#define MAXBLOCKS 500
#define MAXCHARS  16

static size_t TotBytes;
static void *Base;

static unsigned long Nblocks;

static void **Table;
static size_t *BlockSize;
static char *MovableFlag;
static void ***BasePointers;

static char *VarName;
static char *FunctionName;
static char *FileName;
static int *LineNumber;


void mymalloc_init(void)
{
  size_t n;

  BlockSize = (size_t *) malloc(MAXBLOCKS * sizeof(size_t));
  Table = (void **) malloc(MAXBLOCKS * sizeof(void *));
  MovableFlag = (char *) malloc(MAXBLOCKS * sizeof(char));
  BasePointers = (void ***) malloc(MAXBLOCKS * sizeof(void **));
  VarName = (char *) malloc(MAXBLOCKS * MAXCHARS * sizeof(char));
  FunctionName = (char *) malloc(MAXBLOCKS * MAXCHARS * sizeof(char));
  FileName = (char *) malloc(MAXBLOCKS * MAXCHARS * sizeof(char));
  LineNumber = (int *) malloc(MAXBLOCKS * sizeof(int));

  memset(VarName, 0, MAXBLOCKS * MAXCHARS);
  memset(FunctionName, 0, MAXBLOCKS * MAXCHARS);
  memset(FileName, 0, MAXBLOCKS * MAXCHARS);

  n = All.MaxMemSizePerCore * All.NumThreads * ((size_t) 1024 * 1024);

  if(!(Base = malloc(n)))
    {
      printf("Failed to allocate memory for `Base' (%d Mbytes).\n", All.MaxMemSizePerCore * All.NumThreads);
      endrun(122);
    }

  TotBytes = FreeBytes = n;

  AllocatedBytes = 0;
  Nblocks = 0;
  HighMarkBytes = 0;
}

void report_detailed_memory_usage_of_largest_task(size_t * OldHighMarkBytes, const char *label,
						  const char *func, const char *file, int line)
{
  size_t *sizelist, maxsize, minsize;
  double avgsize;
  int i, task;

  sizelist = (size_t *) mymalloc("sizelist", NTask * sizeof(size_t));
  MPI_Allgather(&AllocatedBytes, sizeof(size_t), MPI_BYTE, sizelist, sizeof(size_t), MPI_BYTE,
		MPI_COMM_WORLD);

  for(i = 1, task = 0, maxsize = minsize = sizelist[0], avgsize = sizelist[0]; i < NTask; i++)
    {
      if(sizelist[i] > maxsize)
	{
	  maxsize = sizelist[i];
	  task = i;
	}
      if(sizelist[i] < minsize)
	{
	  minsize = sizelist[i];
	}
      avgsize += sizelist[i];
    }

  myfree(sizelist);


  if(maxsize > 1.1 * (*OldHighMarkBytes))
    {
      *OldHighMarkBytes = maxsize;

      avgsize /= NTask;

      if(ThisTask == task)
	{
	  printf
	    ("\nAt '%s', %s()/%s/%d: Largest Allocation = %g Mbyte (on task=%d), Smallest = %g Mbyte, Average = %g Mbyte\n\n",
	     label, func, file, line, maxsize / (1024.0 * 1024.0), task, minsize / (1024.0 * 1024.0),
	     avgsize / (1024.0 * 1024.0));
	  dump_memory_table();
	}
      fflush(stdout);
      MPI_Barrier(MPI_COMM_WORLD);
    }
}




void dump_memory_table(void)
{
  int i;
  size_t totBlocksize = 0;

  printf("------------------------ Allocated Memory Blocks----------------------------------------\n");
  printf("Task   Nr F          Variable      MBytes   Cumulative         Function/File/Linenumber\n");
  printf("----------------------------------------------------------------------------------------\n");
  for(i = 0; i < Nblocks; i++)
    {
      totBlocksize += BlockSize[i];

      printf("%4d %4d %d  %16s  %10.4f   %10.4f  %s()/%s/%d\n",
	     ThisTask, i, MovableFlag[i], VarName + i * MAXCHARS, BlockSize[i] / (1024.0 * 1024.0),
	     totBlocksize / (1024.0 * 1024.0), FunctionName + i * MAXCHARS,
	     FileName + i * MAXCHARS, LineNumber[i]);
    }
  printf("----------------------------------------------------------------------------------------\n");
}

void *mymalloc_fullinfo(const char *varname, size_t n, const char *func, const char *file, int line)
{
  if((n % 8) > 0)
    n = (n / 8 + 1) * 8;

  if(n < 8)
    n = 8;

  if(Nblocks >= MAXBLOCKS)
    {
      printf("Task=%d: No blocks left in mymalloc_fullinfo() at %s()/%s/line %d. MAXBLOCKS=%d\n", ThisTask,
	     func, file, line, MAXBLOCKS);
      endrun(813);
    }

  if(n > FreeBytes)
    {
      dump_memory_table();
      printf
	("\nTask=%d: Not enough memory in mymalloc_fullinfo() to allocate %g MB for variable '%s' at %s()/%s/line %d (FreeBytes=%g MB).\n",
	 ThisTask, n / (1024.0 * 1024.0), varname, func, file, line, FreeBytes / (1024.0 * 1024.0));
      endrun(812);
    }
  Table[Nblocks] = Base + (TotBytes - FreeBytes);
  FreeBytes -= n;

  strncpy(VarName + Nblocks * MAXCHARS, varname, MAXCHARS - 1);
  strncpy(FunctionName + Nblocks * MAXCHARS, func, MAXCHARS - 1);
  strncpy(FileName + Nblocks * MAXCHARS, file, MAXCHARS - 1);
  LineNumber[Nblocks] = line;

  AllocatedBytes += n;
  BlockSize[Nblocks] = n;
  MovableFlag[Nblocks] = 0;

  Nblocks += 1;

  if(AllocatedBytes > HighMarkBytes)
    HighMarkBytes = AllocatedBytes;

  return Table[Nblocks - 1];
}


void *mymalloc_movable_fullinfo(void *ptr, const char *varname, size_t n, const char *func, const char *file,
				int line)
{
  if((n % 8) > 0)
    n = (n / 8 + 1) * 8;

  if(n < 8)
    n = 8;

  if(Nblocks >= MAXBLOCKS)
    {
      printf("Task=%d: No blocks left in mymalloc_fullinfo() at %s()/%s/line %d. MAXBLOCKS=%d\n", ThisTask,
	     func, file, line, MAXBLOCKS);
      endrun(816);
    }

  if(n > FreeBytes)
    {
      dump_memory_table();
      printf
	("\nTask=%d: Not enough memory in mymalloc_fullinfo() to allocate %g MB for variable '%s' at %s()/%s/line %d (FreeBytes=%g MB).\n",
	 ThisTask, n / (1024.0 * 1024.0), varname, func, file, line, FreeBytes / (1024.0 * 1024.0));
      endrun(817);
    }
  Table[Nblocks] = Base + (TotBytes - FreeBytes);
  FreeBytes -= n;

  strncpy(VarName + Nblocks * MAXCHARS, varname, MAXCHARS - 1);
  strncpy(FunctionName + Nblocks * MAXCHARS, func, MAXCHARS - 1);
  strncpy(FileName + Nblocks * MAXCHARS, file, MAXCHARS - 1);
  LineNumber[Nblocks] = line;

  AllocatedBytes += n;
  BlockSize[Nblocks] = n;
  MovableFlag[Nblocks] = 1;
  BasePointers[Nblocks] = ptr;

  Nblocks += 1;

  if(AllocatedBytes > HighMarkBytes)
    HighMarkBytes = AllocatedBytes;

  return Table[Nblocks - 1];
}



void myfree_fullinfo(void *p, const char *func, const char *file, int line)
{
  if(Nblocks == 0)
    endrun(76878);

  if(p != Table[Nblocks - 1])
    {
      dump_memory_table();
      printf("Task=%d: Wrong call of myfree() at %s()/%s/line %d: not the last allocated block!\n", ThisTask,
	     func, file, line);
      fflush(stdout);
      endrun(814);
    }

  Nblocks -= 1;
  AllocatedBytes -= BlockSize[Nblocks];
  FreeBytes += BlockSize[Nblocks];
}



void myfree_movable_fullinfo(void *p, const char *func, const char *file, int line)
{
  int i;

  if(Nblocks == 0)
    endrun(768728);

  /* first, let's find the block */
  int nr;

  for(nr = Nblocks - 1; nr >= 0; nr--)
    if(p == Table[nr])
      break;

  if(nr < 0)
    {
      dump_memory_table();
      printf
	("Task=%d: Wrong call of myfree_movable() from %s()/%s/line %d - this block has not been allocated!\n",
	 ThisTask, func, file, line);
      fflush(stdout);
      endrun(8152);
    }

  if(nr < Nblocks - 1)		/* the block is not the last allocated block */
    {
      /* check that all subsequent blocks are actually movable */
      for(i = nr + 1; i < Nblocks; i++)
	if(MovableFlag[i] == 0)
	  {
	    dump_memory_table();
	    printf
	      ("Task=%d: Wrong call of myfree_movable() from %s()/%s/line %d - behind block=%d there are subsequent non-movable allocated blocks\n",
	       ThisTask, func, file, line, nr);
	    fflush(stdout);
	    endrun(81252);
	  }
    }


  AllocatedBytes -= BlockSize[nr];
  FreeBytes += BlockSize[nr];

  size_t offset = -BlockSize[nr];
  size_t length = 0;

  for(i = nr + 1; i < Nblocks; i++)
    length += BlockSize[i];

  if(nr < Nblocks - 1)
    memmove(Table[nr + 1] + offset, Table[nr + 1], length);

  for(i = nr + 1; i < Nblocks; i++)
    {
      Table[i] += offset;
      *BasePointers[i] = *BasePointers[i] + offset;
    }

  for(i = nr + 1; i < Nblocks; i++)
    {
      Table[i - 1] = Table[i];
      BasePointers[i - 1] = BasePointers[i];
      BlockSize[i - 1] = BlockSize[i];
      MovableFlag[i - 1] = MovableFlag[i];

      strncpy(VarName + (i - 1) * MAXCHARS, VarName + i * MAXCHARS, MAXCHARS - 1);
      strncpy(FunctionName + (i - 1) * MAXCHARS, FunctionName + i * MAXCHARS, MAXCHARS - 1);
      strncpy(FileName + (i - 1) * MAXCHARS, FileName + i * MAXCHARS, MAXCHARS - 1);
      LineNumber[i - 1] = LineNumber[i];
    }

  Nblocks -= 1;
}






void *myrealloc_fullinfo(void *p, size_t n, const char *func, const char *file, int line)
{
  if((n % 8) > 0)
    n = (n / 8 + 1) * 8;

  if(n < 8)
    n = 8;

  if(Nblocks == 0)
    endrun(76879);

  if(p != Table[Nblocks - 1])
    {
      dump_memory_table();
      printf("Task=%d: Wrong call of myrealloc() at %s()/%s/line %d - not the last allocated block!\n",
	     ThisTask, func, file, line);
      fflush(stdout);
      endrun(815);
    }

  AllocatedBytes -= BlockSize[Nblocks - 1];
  FreeBytes += BlockSize[Nblocks - 1];

  if(n > FreeBytes)
    {
      dump_memory_table();
      printf
	("Task=%d: Not enough memory in myremalloc(n=%g MB) at %s()/%s/line %d. previous=%g FreeBytes=%g MB\n",
	 ThisTask, n / (1024.0 * 1024.0), func, file, line, BlockSize[Nblocks - 1] / (1024.0 * 1024.0),
	 FreeBytes / (1024.0 * 1024.0));
      endrun(812);
    }
  Table[Nblocks - 1] = Base + (TotBytes - FreeBytes);
  FreeBytes -= n;

  AllocatedBytes += n;
  BlockSize[Nblocks - 1] = n;

  if(AllocatedBytes > HighMarkBytes)
    HighMarkBytes = AllocatedBytes;

  return Table[Nblocks - 1];
}

void *myrealloc_movable_fullinfo(void *p, size_t n, const char *func, const char *file, int line)
{
  int i;

  if((n % 8) > 0)
    n = (n / 8 + 1) * 8;

  if(n < 8)
    n = 8;

  if(Nblocks == 0)
    endrun(768799);

  /* first, let's find the block */
  int nr;

  for(nr = Nblocks - 1; nr >= 0; nr--)
    if(p == Table[nr])
      break;

  if(nr < 0)
    {
      dump_memory_table();
      printf
	("Task=%d: Wrong call of myrealloc_movable() from %s()/%s/line %d - this block has not been allocated!\n",
	 ThisTask, func, file, line);
      fflush(stdout);
      endrun(8151);
    }

  if(nr < Nblocks - 1)		/* the block is not the last allocated block */
    {
      /* check that all subsequent blocks are actually movable */
      for(i = nr + 1; i < Nblocks; i++)
	if(MovableFlag[i] == 0)
	  {
	    dump_memory_table();
	    printf
	      ("Task=%d: Wrong call of myrealloc_movable() from %s()/%s/line %d - behind block=%d there are subsequent non-movable allocated blocks\n",
	       ThisTask, func, file, line, nr);
	    fflush(stdout);
	    endrun(8152);
	  }
    }


  AllocatedBytes -= BlockSize[nr];
  FreeBytes += BlockSize[nr];

  if(n > FreeBytes)
    {
      dump_memory_table();
      printf
	("Task=%d: at %s()/%s/line %d: Not enough memory in myremalloc_movable(n=%g MB). previous=%g FreeBytes=%g MB\n",
	 ThisTask, func, file, line, n / (1024.0 * 1024.0), BlockSize[nr] / (1024.0 * 1024.0),
	 FreeBytes / (1024.0 * 1024.0));
      endrun(812);
    }

  size_t offset = n - BlockSize[nr];
  size_t length = 0;

  for(i = nr + 1; i < Nblocks; i++)
    length += BlockSize[i];

  if(nr < Nblocks - 1)
    memmove(Table[nr + 1] + offset, Table[nr + 1], length);

  for(i = nr + 1; i < Nblocks; i++)
    {
      Table[i] += offset;

      *BasePointers[i] = *BasePointers[i] + offset;
    }

  FreeBytes -= n;
  AllocatedBytes += n;
  BlockSize[nr] = n;

  if(AllocatedBytes > HighMarkBytes)
    HighMarkBytes = AllocatedBytes;

  return Table[nr];
}
