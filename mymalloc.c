#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <gsl/gsl_math.h>

#include "mymalloc.h"
#include "allvars.h"

#define MAXBLOCKS 500
#define MAXCHARS  16
#define ALIGNMENT  4096
static size_t TotBytes;
static char *Base;

static unsigned long Nblocks;

static char *Table[MAXBLOCKS] = {NULL};
static size_t BlockSize[MAXBLOCKS] = {0};
static char MovableFlag[MAXBLOCKS] = {0};

static char VarName[MAXBLOCKS][MAXCHARS] = {0};
static char FunctionName[MAXBLOCKS][MAXCHARS] = {0};
static char FileName[MAXBLOCKS][MAXCHARS] = {0};
static int LineNumber[MAXBLOCKS] = {0};

size_t AllocatedBytes;
size_t HighMarkBytes;
size_t FreeBytes;

static void strncpy_basename(char * target, const char * name, int nchars) {
    int i = 0;
    int len = strlen(name);
    int cpylen;
    for(cpylen = 0; cpylen < nchars && cpylen < len; cpylen ++) {
        if(cpylen < len - 1 && name[len - cpylen - 1] == '/') break;
    }
    
    for(i = 0; i < cpylen; i ++) {
        target[i] = name[len - cpylen + i];
    }
}
static size_t align_size(size_t n) {
    return ((size_t) ((n + ALIGNMENT - 1) / ALIGNMENT)) * ALIGNMENT;
}

void mymalloc_init(void)
{
    size_t n;

    /* n is aligned*/
    n = align_size(All.MaxMemSizePerCore * All.NumThreads * ((size_t) 1024 * 1024));

#ifndef VALGRIND
    /* extra space for aligning Base */
    if(!(Base = malloc(n + ALIGNMENT)))
    {
        printf("Failed to allocate memory for `Base' (%d Mbytes).\n", All.MaxMemSizePerCore * All.NumThreads);
        endrun(122);
    }
    Base = (char*) align_size((size_t) Base);
#else
    Base = NULL;
#endif
    TotBytes = FreeBytes = n;

    AllocatedBytes = 0;
    Nblocks = 0;
    HighMarkBytes = 0;
}

void report_detailed_memory_usage_of_largest_task(const char *label,
        const char *func, const char *file, int line)
{
    size_t *sizelist, maxsize, minsize;
    double avgsize;
    int i, task;
    static size_t highmarks[1024] = {0};
    static char * labels[1024] = {0};
    size_t * OldHighMarkBytes = NULL;
    for(i = 0; i < 1024; i ++) {
        if(labels[i] == NULL) {
            labels[i] = strdup(label);
            break;
        }
        if(!strcmp(labels[i], label)) {
            break;
        }
    }
    if(i == 1024) {
        //"need more label space;"
        endrun(33214);
    }
    OldHighMarkBytes = &highmarks[i];
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

        printf("%4d %4d %d  %16s  %10.4f   %10.4f  %s:%d:%s\n",
                ThisTask, i, MovableFlag[i], VarName[i], BlockSize[i] / (1024.0 * 1024.0),
                totBlocksize / (1024.0 * 1024.0), FileName[i], LineNumber[i], FunctionName[i]);
    }
    printf("----------------------------------------------------------------------------------------\n");
}

void *mymalloc_fullinfo(const char *varname, size_t n, const char *func, const char *file, int line)
{
#ifdef VALGRIND
    return malloc(n);
#endif
    n = align_size(n);

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

    strncpy(VarName[Nblocks], varname, MAXCHARS - 1);
    strncpy(FunctionName[Nblocks], func, MAXCHARS - 1);
    strncpy_basename(FileName[Nblocks], file, MAXCHARS - 1);
    LineNumber[Nblocks] = line;

    AllocatedBytes += n;
    BlockSize[Nblocks] = n;
    MovableFlag[Nblocks] = 0;

    Nblocks += 1;

    if(AllocatedBytes > HighMarkBytes)
        HighMarkBytes = AllocatedBytes;

    return Table[Nblocks - 1];
}



void myfree_fullinfo(void *p, const char *func, const char *file, int line)
{
#ifdef VALGRIND
    free(p);
    return;
#endif
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

void *myrealloc_fullinfo(void *p, size_t n, const char *func, const char *file, int line)
{
#ifdef VALGRIND
    return realloc(p, n);
#endif
    n = align_size(n);
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

