#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "bigfile-mpi.h"
int big_file_mpi_open(BigFile * bf, char * basename, MPI_Comm comm) {
    if(comm == MPI_COMM_NULL) return 0;
    int rank;
    MPI_Comm_rank(comm, &rank);
    big_file_open(bf, basename);
    MPI_Barrier(comm);
    return 0;
}

int big_file_mpi_create(BigFile * bf, char * basename, MPI_Comm comm) {
    if(comm == MPI_COMM_NULL) return 0;
    int rank;
    MPI_Comm_rank(comm, &rank);
    big_file_create(bf, basename);
    MPI_Barrier(comm);
    return 0;
}
int big_file_mpi_open_block(BigFile * bf, BigBlock * block, char * blockname, MPI_Comm comm) {
    if(comm == MPI_COMM_NULL) return 0;
    char * basename = alloca(strlen(bf->basename) + strlen(blockname) + 128);
    sprintf(basename, "%s/%s/", bf->basename, blockname);
    return big_block_mpi_open(block, basename, comm);
}
int big_file_mpi_create_block(BigFile * bf, BigBlock * block, char * blockname, char * dtype, int Nfile, size_t fsize[], MPI_Comm comm) {
    if(comm == MPI_COMM_NULL) return 0;
    char * basename = alloca(strlen(bf->basename) + strlen(blockname) + 128);
    big_file_mksubdir_r(bf->basename, blockname);
    sprintf(basename, "%s/%s/", bf->basename, blockname);
    return big_block_mpi_create(block, basename, dtype, Nfile, fsize, comm);
}

void big_file_mpi_close(BigFile * bf, MPI_Comm comm) {
    if(comm == MPI_COMM_NULL) return ;
    big_file_close(bf);
    MPI_Barrier(comm);
}

static int big_block_mpi_broadcast(BigBlock * bb, int root, MPI_Comm comm);

int big_block_mpi_open(BigBlock * bb, char * basename, MPI_Comm comm) {
    if(comm == MPI_COMM_NULL) return 0;
    int rank;
    MPI_Comm_rank(comm, &rank);
    int rt;
    if(rank == 0) { 
        rt = big_block_open(bb, basename);
    }
    MPI_Bcast(&rt, 1, MPI_INT, 0, comm);
    if(rt) return rt;
    big_block_mpi_broadcast(bb, 0, comm);
    return 0;
}
int big_block_mpi_create(BigBlock * bb, char * basename, char * dtype, int Nfile, size_t fsize[], MPI_Comm comm) {
    if(comm == MPI_COMM_NULL) return 0;
    int rank;
    MPI_Comm_rank(comm, &rank);
    int rt;
    int rt1;
    if(rank == 0) { 
        rt = big_block_create(bb, basename, dtype, Nfile, fsize);
    }
    MPI_Bcast(&rt, 1, MPI_INT, 0, comm);
    if(rt) return rt;
    big_block_mpi_broadcast(bb, 0, comm);
    return 0;
}
int big_block_mpi_close(BigBlock * block, MPI_Comm comm) {
    if(comm == MPI_COMM_NULL) return 0;
    int rank;
    MPI_Comm_rank(comm, &rank);
    if(rank == 0) {
        big_block_close(block);
    }
    MPI_Barrier(comm);
    return 0;
}

static int big_block_mpi_broadcast(BigBlock * bb, int root, MPI_Comm comm) {
    ptrdiff_t i;
    int rank;
    MPI_Comm_rank(comm, &rank);
    int lname = 0;
    char * oldbuf = 0;
    if(rank == root) {
        lname = strlen(bb->basename);
        oldbuf = bb->attrset.attrbuf;
    }
    MPI_Bcast(&lname, 1, MPI_INT, root, comm);
    MPI_Bcast(&oldbuf, sizeof(ptrdiff_t), MPI_BYTE, root, comm);
    MPI_Bcast(bb, sizeof(BigBlock), MPI_BYTE, root, comm);
    if(rank != root) {
        bb->basename = calloc(lname + 1, 1);
        bb->fsize = calloc(bb->Nfile, sizeof(size_t));
        bb->foffset = calloc(bb->Nfile + 1, sizeof(size_t));
        bb->attrset.attrbuf = calloc(bb->attrset.bufsize, 1);
        bb->attrset.attrlist = calloc(bb->attrset.listsize, sizeof(BigBlockAttr));
    }
    MPI_Bcast(bb->basename, lname + 1, MPI_BYTE, root, comm);
    MPI_Bcast(bb->fsize, sizeof(ptrdiff_t) * bb->Nfile, MPI_BYTE, root, comm);
    MPI_Bcast(bb->foffset, sizeof(ptrdiff_t) * bb->Nfile + 1, MPI_BYTE, root, comm);
    MPI_Bcast(bb->attrset.attrbuf, bb->attrset.bufused, MPI_BYTE, root, comm);
    MPI_Bcast(bb->attrset.attrlist, bb->attrset.listused * sizeof(BigBlockAttr), MPI_BYTE, root, comm);
    for(i = 0; i < bb->attrset.listused; i++) {
        BigBlockAttr * a = &bb->attrset.attrlist[i];
        a->name = a->name - oldbuf + bb->attrset.attrbuf;
        a->data = a->data - oldbuf + bb->attrset.attrbuf;
    }
    return 0;
}
