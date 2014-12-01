#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <stddef.h>
#include <string.h>
#include <mpi.h>
#include "allvars.h"
#include "proto.h"
#include "bigfile-mpi.h"

#include "walltime.h"
    
static int ThisColor;
static int ThisKey;
MPI_Comm GROUP;
static int GroupSize;

int64_t NumPartOffset;
int64_t * NumPartPerFileList;
static void saveblock(BigFile * bf, void * baseptr, char * name, char * dtype, int items_per_particle);
void saveheader(BigFile * bf);
int64_t TotNumPart;

void write_particle_data(void) {

    int64_t numpart_64 = NumPart;
    MPI_Allreduce(&numpart_64, &TotNumPart, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    NumPartPerFileList = alloca(sizeof(int64_t) * NumFiles);
    {
        int i;
        for(i = 0; i < NumFiles; i ++) {
            NumPartPerFileList[i] = ((i + 1) * TotNumPart / NumFiles - i * TotNumPart / NumFiles);
        }
    }

    if(NumFilesWrittenInParallel > NTask) {
        NumFilesWrittenInParallel = NTask;
    }
    char buf[4096];
    ThisColor = ThisTask * NumFilesWrittenInParallel / NTask;
    MPI_Comm_split(MPI_COMM_WORLD, ThisColor, 0, &GROUP);
    MPI_Comm_rank(GROUP, &ThisKey);
    MPI_Comm_size(GROUP, &GroupSize);

    walltime_measure("/Misc");
    {
        int i;    
        int NumPartList[NTask];
        int64_t NumPartOffsetList[NTask];
        MPI_Gather(&NumPart, 1, MPI_INT, NumPartList, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if(ThisTask == 0) {
            NumPartOffsetList[0] = 0;
            for(i = 1; i < NTask; i ++) {
                NumPartOffsetList[i] = NumPartOffsetList[i - 1] + NumPartList[i - 1];
            }
        }
        MPI_Scatter(NumPartOffsetList, 1, MPI_LONG, &NumPartOffset, 1, MPI_LONG, 0, MPI_COMM_WORLD);
    }

    sprintf(buf, "%s/%s", OutputDir, FileBase);

    BigFile bf;

    if(0 != big_file_mpi_create(&bf, buf, MPI_COMM_WORLD)) {
        if(ThisTask == 0) {
            fprintf(stderr, "%s\n", big_file_get_error_message());
            abort();
        } 
    }

    saveheader(&bf);

    if (ProduceGas) {
        /* First shift gas */
        double meanspacing = Box / pow(TotNumPart, 1.0 / 3);
        double shift_gas = -0.5 * (Omega - OmegaBaryon) / (Omega) * meanspacing;
        double shift_dm = +0.5 * OmegaBaryon / (Omega) * meanspacing;
        int i;
        for(i = 0; i < NumPart; i ++) {
            int k;
            for(k = 0; k < 3; k ++) {
                P[i].Pos[k] = periodic_wrap(P[i].Pos[k] + shift_gas);
                
            }
        }
        /* Write Gas */
        saveblock(&bf, &P[0].Pos, "0/Position", "f8", 3);
        saveblock(&bf, &P[0].Vel, "0/Velocity", "f4", 3);
        saveblock(&bf, &P[0].ID, "0/ID", "u8", 1);

        /* Then shift back to DM */
        for(i = 0; i < NumPart; i ++) {
            int k;
            for(k = 0; k < 3; k ++) {
                P[i].Pos[k] = periodic_wrap(P[i].Pos[k] + (shift_dm - shift_gas));
            }
            P[i].ID += TotNumPart;
        }
    }
    /* Write DM */
    saveblock(&bf, &P[0].Density, "1/ICDensity", "f4", 1);
    saveblock(&bf, &P[0].Pos, "1/Position", "f8", 3);
    saveblock(&bf, &P[0].Vel, "1/Velocity", "f4", 3);
    saveblock(&bf, &P[0].ID, "1/ID", "u8", 1);
    walltime_measure("/Write");
}

static void saveblock(BigFile * bf, void * baseptr, char * name, char * dtype, int items_per_particle) {
    BigBlock block;
    BigArray array;
    BigBlockPtr ptr;
    size_t dims[2];
    ptrdiff_t strides[2];

    dims[0] = NumPart;
    dims[1] = items_per_particle;
    strides[1] = dtype_itemsize(dtype);
    strides[0] = sizeof(P[0]);
    big_array_init(&array, baseptr, dtype, 2, dims, strides);

    int i;
    if(0 != big_file_mpi_create_block(bf, &block, name, dtype, dims[1], NumFiles, TotNumPart, MPI_COMM_WORLD)) {
        if(ThisTask == 0) {
            fprintf(stderr, "%s:%s\n", big_file_get_error_message(), name);
            abort();
        } 
    }

    for(i = 0; i < GroupSize; i ++) {
        MPI_Barrier(GROUP);
        if(i != ThisKey) continue;
        if(0 != big_block_seek(&block, &ptr, NumPartOffset)) {
            fprintf(stderr, "Task = %d, failed seeking at %td: %s\n", ThisTask, NumPartOffset, 
                    big_file_get_error_message());
            abort();
        }
        if(0 != big_block_write(&block, &ptr, &array)) {
            fprintf(stderr, "Task = %d, failed writing at %td: %s\n", ThisTask, NumPartOffset, 
                    big_file_get_error_message());
            abort();

        }
    }
    if(0 != big_block_mpi_close(&block, MPI_COMM_WORLD)) {
        if(ThisTask == 0) {
            fprintf(stderr, "%s:%s\n", big_file_get_error_message(), name);
            abort();
        } 
    }

}

void saveheader(BigFile * bf) {
    BigBlock bheader;
    if(0 != big_file_mpi_create_block(bf, &bheader, "header", NULL, 0, 0, 0, MPI_COMM_WORLD)) {
        if(ThisTask == 0) {
            fprintf(stderr, "failed to create block %s:%s", "header",
                    big_file_get_error_message());
            abort(); 
        }
    }
    
    int64_t totnumpart[6] = {0};
    double mass[6] = {0};
    totnumpart[1] = TotNumPart;
    if (ProduceGas) {
        totnumpart[0] = TotNumPart;
        mass[0] = (OmegaBaryon) * 3 * Hubble * Hubble / (8 * PI * G) * pow(Box, 3) / TotNumPart;
        mass[1] = (Omega - OmegaBaryon) * 3 * Hubble * Hubble / (8 * PI * G) * pow(Box, 3) / TotNumPart;
    } else {
        mass[1] = (Omega) * 3 * Hubble * Hubble / (8 * PI * G) * pow(Box, 3) / TotNumPart;
    }
    double redshift = 1.0 / InitTime - 1.;

    int rt =(0 != big_block_set_attr(&bheader, "TotNumPart", totnumpart, "i8", 6)) ||
            (0 != big_block_set_attr(&bheader, "MassTable", mass, "f8", 6)) ||
            (big_block_set_attr(&bheader, "Time", &InitTime, "f8", 1)) ||
            (big_block_set_attr(&bheader, "Redshift", &redshift, "f8", 1)) ||
            (big_block_set_attr(&bheader, "BoxSize", &Box, "f8", 1)) ||
            (big_block_set_attr(&bheader, "OmegaM", &Omega, "f8", 1)) ||
            (big_block_set_attr(&bheader, "OmegaB", &OmegaBaryon, "f8", 1)) ||
            (big_block_set_attr(&bheader, "OmegaL", &OmegaLambda, "f8", 1)) ||
            (big_block_set_attr(&bheader, "HubbleParam", &HubbleParam, "f8", 1));
    if(rt && ThisTask == 0) {
        fprintf(stderr, "failed to create attr %s", 
                big_file_get_error_message());
        abort(); 
    }

    if(0 != big_block_mpi_close(&bheader, MPI_COMM_WORLD)) {
        if(ThisTask == 0) {
            fprintf(stderr, "failed to close block %s:%s", "header",
                    big_file_get_error_message());
            abort(); 
        }
    }
}

size_t my_fread(void *ptr, size_t size, size_t nmemb, FILE * stream)
{
  size_t nread;

  if((nread = fread(ptr, size, nmemb, stream)) != nmemb)
    {
      printf("I/O error (fread) on task=%d has occured.\n", ThisTask);
      fflush(stdout);
      FatalError(778);
    }
  return nread;
}
