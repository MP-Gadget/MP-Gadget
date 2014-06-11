#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <string.h>

#include "allvars.h"
#include "proto.h"
#include "petaio.h"
#include "bigfile/bigfile-mpi.h"

static int64_t npartTotal[6];
static int64_t npartLocal[6];
static int64_t offsetLocal[6];

static void count();

static int ThisColor;
static int ThisKey;
MPI_Comm GROUP;
static int GroupSize;
static int NumFiles;
static void saveblock(BigFile * bf, IOTableEntry * ent);
static void write_header(BigFile * bf);
void fof_save_particles(int num) {
    char fname[4096];
    sprintf(fname, "%s/PIG_%03d", All.OutputDir, num);
    /* Split the wolrd into writer groups */
    NumFiles = All.NumFilesWrittenInParallel;
    ThisColor = ThisTask * NumFiles / NTask;
    MPI_Comm_split(MPI_COMM_WORLD, ThisColor, 0, &GROUP);
    MPI_Comm_rank(GROUP, &ThisKey);
    MPI_Comm_size(GROUP, &GroupSize);

    BigFile bf = {0};
    if(0 != big_file_mpi_create(&bf, fname, MPI_COMM_WORLD)) {
        if(ThisTask == 0) {
            fprintf(stderr, "Failed to open IC from %s\n", fname);
        }
        abort();
    }

    count();

    write_header(&bf); 
    int i;
    for(i = 0; i < IOTable.used; i ++) {
        saveblock(&bf, &IOTable.ent[i]);
    }
    big_file_mpi_close(&bf, MPI_COMM_WORLD);
}

static void saveblock(BigFile * bf, IOTableEntry * ent) {
    if(ThisTask == 0) {
        printf("Saving block %s, ptype %d as %s: (%ld, %d)\n", ent->name, ent->ptype, ent->dtype, 
                npartTotal[ent->ptype], ent->items);
    }
    BigBlock bb = {0};
    int i;

    BigArray array;
    BigBlockPtr ptr;
    size_t dims[2];
    ptrdiff_t strides[2];
    size_t fsize[NumFiles];
    int elsize = dtype_itemsize(ent->dtype);

    /* dimensions of the collected array; it's simplly linear */
    dims[0] = npartLocal[ent->ptype];
    dims[1] = ent->items;
    strides[1] = elsize;
    strides[0] = elsize * ent->items;

    for(i = 0; i < NumFiles; i ++) {
        fsize[i] = npartTotal[ent->ptype] * (i + 1) / NumFiles 
                 - npartTotal[ent->ptype] * (i) / NumFiles;
    }

    /* create the block */
    char blockname[128];
    sprintf(blockname, "%d/%s", ent->ptype, ent->name);
    big_file_mpi_create_block(bf, &bb, blockname, ent->dtype, dims[1], NumFiles, fsize, MPI_COMM_WORLD);

    /* create the buffer */
    char * buffer = malloc(dims[0] * dims[1] * elsize);
    big_array_init(&array, buffer, ent->dtype, 2, dims, strides);
    /* fill the buffer */
    char * p = buffer;
    for(i = 0; i < NumPart; i ++) {
        if(P[i].Type != ent->ptype) continue;
        if(P[i].GrNr < 0) continue;
        ent->getter(i, p);
        p += strides[0];
    }

    /* write the buffers one by one in each writer group */
    for(i = 0; i < GroupSize; i ++) {
        MPI_Barrier(GROUP);
        if(i != ThisKey) continue;
        if(0 != big_block_seek(&bb, &ptr, offsetLocal[ent->ptype])) {
            fprintf(stderr, "Failed to seek\n");
            abort();
        }
        //printf("Task = %d, writing at %td\n", ThisTask, offsetLocal[ent->ptype]);
        big_block_write(&bb, &ptr, &array);
    }

    free(buffer);
    big_block_mpi_close(&bb, MPI_COMM_WORLD);
}

static void count() {
    int i;
    int k;
    for (k = 0; k < 6; k ++) {
        npartLocal[k] = 0;
    }
    for (i = 0; i < NumPart; i ++) {
        if(P[i].GrNr < 0) continue; /* skip those not in groups */
        npartLocal[P[i].Type] ++;
    }

    MPI_Allreduce(npartLocal, npartTotal, 6, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);

    int64_t counts[NTask];
    int64_t offsets[NTask];

    for(k = 0; k < 6; k ++) {

        /* gather counts */
        MPI_Gather(&npartLocal[k], 1, MPI_LONG, &counts[0], 1, MPI_LONG, 0, MPI_COMM_WORLD);

        /* count offsets */
        offsets[0] = 0;
        for(i = 1; i < NTask; i ++) {
            offsets[i] = offsets[i - 1] + counts[i - 1];
        }
        MPI_Scatter(&offsets[0], 1, MPI_LONG, &offsetLocal[k], 1, MPI_LONG, 0, MPI_COMM_WORLD); 
    }
}
static void write_header(BigFile * bf) {
    BigBlock bh = {0};
    if(0 != big_file_mpi_create_block(bf, &bh, "header", NULL, 0, 0, NULL, MPI_COMM_WORLD)) {
        fprintf(stderr, "Failed to create header\n");
        abort();
    }
    int i;

    big_block_set_attr(&bh, "NumPartTotal", npartTotal, "i8", 6);
    big_block_set_attr(&bh, "MassTable", All.MassTable, "f8", 6);
    big_block_set_attr(&bh, "Time", &All.Time, "f8", 1);
    big_block_set_attr(&bh, "BoxSize", &All.BoxSize, "f8", 1);
    big_block_set_attr(&bh, "OmegaLambda", &All.OmegaLambda, "f8", 1);
    big_block_set_attr(&bh, "Omega0", &All.Omega0, "f8", 1);
    big_block_set_attr(&bh, "HubbleParam", &All.HubbleParam, "f8", 1);
    big_block_mpi_close(&bh, MPI_COMM_WORLD);
}
