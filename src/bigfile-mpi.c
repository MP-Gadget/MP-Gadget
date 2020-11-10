#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <alloca.h>
#include <string.h>
#include "bigfile-mpi.h"
#include "bigfile-internal.h"
#include "mp-mpiu.h"

/* disable aggregation by default */
static size_t _BigFileAggThreshold = 0;
static int _big_file_mpi_verbose = 0;

static int big_block_mpi_broadcast(BigBlock * bb, int root, MPI_Comm comm);
static int big_file_mpi_broadcast_anyerror(int rt, MPI_Comm comm);

#define BCAST_AND_RAISEIF(rt, comm) \
    if(0 != (rt = big_file_mpi_broadcast_anyerror(rt, comm))) { \
        return rt; \
    } \

void
big_file_mpi_set_verbose(int verbose)
{
    _big_file_mpi_verbose = verbose;
}

void
big_file_mpi_set_aggregated_threshold(size_t bytes)
{
    _BigFileAggThreshold = bytes;
}

size_t
big_file_mpi_get_aggregated_threshold()
{
    return _BigFileAggThreshold;
}

int big_file_mpi_open(BigFile * bf, const char * basename, MPI_Comm comm) {
    if(comm == MPI_COMM_NULL) return 0;
    int rank;
    MPI_Comm_rank(comm, &rank);
    int rt = 0;
    if (rank == 0) {
        rt = big_file_open(bf, basename);
    } else {
        /* FIXME : */
        bf->basename = _strdup(basename);
        rt = 0;
    }

    BCAST_AND_RAISEIF(rt, comm);

    return rt;
}

int big_file_mpi_create(BigFile * bf, const char * basename, MPI_Comm comm) {
    if(comm == MPI_COMM_NULL) return 0;
    int rank;
    MPI_Comm_rank(comm, &rank);
    int rt;
    if (rank == 0) {
        rt = big_file_create(bf, basename);
    } else {
        /* FIXME : */
        bf->basename = _strdup(basename);
        rt = 0;
    }
    BCAST_AND_RAISEIF(rt, comm);

    return rt;
}

/**Helper function for big_file_mpi_create_block, above*/
static int
_big_block_mpi_create(BigBlock * bb,
        const char * basename,
        const char * dtype,
        int nmemb,
        int Nfile,
        const size_t fsize[],
        MPI_Comm comm);

/** Helper function for big_file_mpi_open_block, above*/
static int _big_block_mpi_open(BigBlock * bb, const char * basename, MPI_Comm comm);

int big_file_mpi_open_block(BigFile * bf, BigBlock * block, const char * blockname, MPI_Comm comm) {
    if(comm == MPI_COMM_NULL) return 0;
    if(!bf || !bf->basename || !blockname) return 1;
    char * basename = alloca(strlen(bf->basename) + strlen(blockname) + 128);
    sprintf(basename, "%s/%s/", bf->basename, blockname);
    return _big_block_mpi_open(block, basename, comm);
}

int
big_file_mpi_create_block(BigFile * bf,
        BigBlock * block,
        const char * blockname,
        const char * dtype,
        int nmemb,
        int Nfile,
        size_t size,
        MPI_Comm comm)
{
    size_t fsize[Nfile];
    int i;
    for(i = 0; i < Nfile; i ++) {
        fsize[i] = size * (i + 1) / Nfile 
                 - size * (i) / Nfile;
    }
    return _big_file_mpi_create_block(bf, block, blockname, dtype,
        nmemb, Nfile, fsize, comm);
}

int
_big_file_mpi_create_block(BigFile * bf,
        BigBlock * block,
        const char * blockname,
        const char * dtype,
        int nmemb,
        int Nfile,
        const size_t fsize[],
        MPI_Comm comm)
{
    if(comm == MPI_COMM_NULL) return 0;
    int rank;
    MPI_Comm_rank(comm, &rank);

    int rt = 0;
    if (rank == 0) {
        rt = _big_file_mksubdir_r(bf->basename, blockname);
    } else {
        rt = 0;
    }

    BCAST_AND_RAISEIF(rt, comm);

    char * basename = alloca(strlen(bf->basename) + strlen(blockname) + 128);
    sprintf(basename, "%s/%s/", bf->basename, blockname);
    return _big_block_mpi_create(block, basename, dtype, nmemb, Nfile, fsize, comm);
}

int big_file_mpi_close(BigFile * bf, MPI_Comm comm) {
    if(comm == MPI_COMM_NULL) return 0;
    int rt = big_file_close(bf);
    MPI_Barrier(comm);
    return rt;
}

static int
_big_block_mpi_open(BigBlock * bb, const char * basename, MPI_Comm comm)
{
    if(comm == MPI_COMM_NULL) return 0;
    int rank;
    MPI_Comm_rank(comm, &rank);
    int rt;
    if(rank == 0) { 
        rt = _big_block_open(bb, basename);
    } else {
        rt = 0;
    }

    BCAST_AND_RAISEIF(rt, comm);

    big_block_mpi_broadcast(bb, 0, comm);
    return 0;
}

static int
_big_block_mpi_create(BigBlock * bb,
        const char * basename,
        const char * dtype,
        int nmemb,
        int Nfile,
        const size_t fsize[],
        MPI_Comm comm)
{
    int rank;
    int NTask;
    int rt;

    if(comm == MPI_COMM_NULL) return 0;

    MPI_Comm_size(comm, &NTask);
    MPI_Comm_rank(comm, &rank);

    if(rank == 0) {
        rt = _big_block_create_internal(bb, basename, dtype, nmemb, Nfile, fsize);
    } else {
        rt = 0;
    }

    BCAST_AND_RAISEIF(rt, comm);

    big_block_mpi_broadcast(bb, 0, comm);

    int i;
    for(i = (size_t) bb->Nfile * rank / NTask; i < (size_t) bb->Nfile * (rank + 1) / NTask; i ++) {
        FILE * fp = _big_file_open_a_file(bb->basename, i, "w", 1);
        if(fp == NULL) {
            rt = -1;
            break;
        }
        fclose(fp);
    }

    BCAST_AND_RAISEIF(rt, comm);

    return rt;
}

int
big_block_mpi_grow(BigBlock * bb,
    int Nfile_grow,
    const size_t fsize_grow[],
    MPI_Comm comm) {

    int rank;
    int NTask;
    int rt;

    if(comm == MPI_COMM_NULL) return 0;

    MPI_Comm_size(comm, &NTask);
    MPI_Comm_rank(comm, &rank);

    int oldNfile = bb->Nfile;

    if(rank == 0) {
        rt = _big_block_grow_internal(bb, Nfile_grow, fsize_grow);
    } else {
        rt = 0;
    }

    BCAST_AND_RAISEIF(rt, comm);

    if(rank != 0) {
        /* closed on non-root because we will bcast.*/
        _big_block_close_internal(bb);
    }
    big_block_mpi_broadcast(bb, 0, comm);

    int i;
    for(i = (size_t) Nfile_grow * rank / NTask; i < (size_t) Nfile_grow * (rank + 1) / NTask; i ++) {
        FILE * fp = _big_file_open_a_file(bb->basename, i + oldNfile, "w", 1);
        if(fp == NULL) {
            rt = -1;
            break;
        }
        fclose(fp);
    }

    BCAST_AND_RAISEIF(rt, comm);

    return rt;
}

int
big_block_mpi_grow_simple(BigBlock * bb, int Nfile_grow, size_t size_grow, MPI_Comm comm)
{
    size_t fsize[Nfile_grow];
    int i;
    for(i = 0; i < Nfile_grow; i ++) {
        fsize[i] = size_grow * (i + 1) / Nfile_grow
                 - size_grow * (i) / Nfile_grow;
    }
    int rank;
    MPI_Comm_rank(comm, &rank);

    return big_block_mpi_grow(bb, Nfile_grow, fsize, comm);
}


int
big_block_mpi_flush(BigBlock * block, MPI_Comm comm)
{
    if(comm == MPI_COMM_NULL) return 0;

    int rank;
    MPI_Comm_rank(comm, &rank);

    unsigned int * checksum = alloca(sizeof(int) * block->Nfile);
    MPI_Reduce(block->fchecksum, checksum, block->Nfile, MPI_UNSIGNED, MPI_SUM, 0, comm);
    int dirty;
    MPI_Reduce(&block->dirty, &dirty, 1, MPI_INT, MPI_LOR, 0, comm);
    int rt;
    if(rank == 0) {
        /* only the root rank updates */
        int i;
        big_block_set_dirty(block, dirty);
        for(i = 0; i < block->Nfile; i ++) {
            block->fchecksum[i] = checksum[i];
        }
        rt = big_block_flush(block);
    } else {
        rt = 0;
    }

    BCAST_AND_RAISEIF(rt, comm);
    /* close as we will broadcast the block */
    if(rank != 0) {
        _big_block_close_internal(block);
    }
    big_block_mpi_broadcast(block, 0, comm);
    return 0;

}
int big_block_mpi_close(BigBlock * block, MPI_Comm comm) {

    int rt = big_block_mpi_flush(block, comm);
    _big_block_close_internal(block);

    return rt;
}

static int
big_file_mpi_broadcast_anyerror(int rt, MPI_Comm comm)
{
    int rank;
    MPI_Comm_rank(comm, &rank);
    int root, loc = 0;
    /* Add 1 so we do not swallow errors on rank 0*/
    if(rt != 0)
        loc = rank+1;

    MPI_Allreduce(&loc, &root, 1, MPI_INT, MPI_MAX, comm);

    if (root == 0) {
        /* no errors */
        return 0;
    }

    root -= 1;
    char * error = big_file_get_error_message();

    int errorlen;
    if(rank == root) {
        errorlen = strlen(error);
    }
    MPI_Bcast(&errorlen, 1, MPI_INT, root, comm);

    if(rank != root) {
        error = malloc(errorlen + 1);
    }

    MPI_Bcast(error, errorlen + 1, MPI_BYTE, root, comm);

    if(rank != root) {
        big_file_set_error_message(error);
        free(error);
    }

    MPI_Bcast(&rt, 1, MPI_INT, root, comm);

    return rt;
}

static int
big_block_mpi_broadcast(BigBlock * bb, int root, MPI_Comm comm)
{
    int rank;
    MPI_Comm_rank(comm, &rank);
    void * buf = NULL;
    size_t bytes = 0;

    if(rank == root) {
        buf = _big_block_pack(bb, &bytes);
    }

    MPI_Bcast(&bytes, sizeof(bytes), MPI_BYTE, root, comm);

    if(rank != root) {
        buf = malloc(bytes);
    }

    MPI_Bcast(buf, bytes, MPI_BYTE, root, comm);

    if(rank != root) {
        _big_block_unpack(bb, buf);
    }
    free(buf);
    return 0;
}

static int
_aggregated(
            BigBlock * block,
            BigBlockPtr * ptr,
            ptrdiff_t offset, /* offset of the entire comm */
            size_t localsize,
            BigArray * array,
            int (*action)(BigBlock * bb, BigBlockPtr * ptr, BigArray * array),
            int root,
            MPI_Comm comm);

static int
_throttle_action(MPI_Comm comm, int concurrency, BigBlock * block,
    BigBlockPtr * ptr,
    BigArray * array,
    int (*action)(BigBlock * bb, BigBlockPtr * ptr, BigArray * array)
)
{
    int ThisTask, NTask;

    MPI_Comm_size(comm, &NTask);
    MPI_Comm_rank(comm, &ThisTask);

    MPIU_Segmenter seggrp[1];

    if(concurrency <= 0) {
        concurrency = NTask;
    }

    size_t avgsegsize;
    size_t localsize = array->dims[0];
    size_t myoffset;
    size_t * sizes = malloc(sizeof(sizes[0]) * NTask);

    size_t totalsize = MPIU_Segmenter_collect_sizes(localsize, sizes, &myoffset, comm);

    /* try to create as many segments as number of groups (thus one segment per group) */
    avgsegsize = totalsize / concurrency;

    if(avgsegsize <= 0) avgsegsize = 1;

    /* no segment shall exceed the memory bound set by maxsegsize, since it will be collected to a single rank */
    if(avgsegsize > _BigFileAggThreshold) avgsegsize = _BigFileAggThreshold;

    MPIU_Segmenter_init(seggrp, sizes, NULL, avgsegsize, concurrency, comm);

    free(sizes);

    int rt = 0;
    int segment;

    for(segment = seggrp->segment_start;
        segment < seggrp->segment_end;
        segment ++) {

        MPI_Barrier(seggrp->Group);

        if(0 != (rt = big_file_mpi_broadcast_anyerror(rt, seggrp->Group))) {
            /* failed , abort. */
            continue;
        } 
        if(seggrp->ThisSegment != segment) continue;

        /* use the offset on the first task in the SegGroup */
        size_t offset = myoffset;
        MPI_Bcast(&offset, 1, MPI_LONG, 0, seggrp->Segment);

        rt = _aggregated(block, ptr, offset, localsize, array, action, seggrp->segment_leader_rank, seggrp->Segment);

    }

    if(0 == (rt = big_file_mpi_broadcast_anyerror(rt, comm))) {
        /* no errors*/
        big_block_seek_rel(block, ptr, totalsize);
    }

    MPIU_Segmenter_destroy(seggrp);
    return rt;
}

static int
_aggregated(
            BigBlock * block,
            BigBlockPtr * ptr,
            ptrdiff_t offset, /* offset of the entire comm */
            size_t localsize, /* offset of the entire comm */
            BigArray * array,
            int (*action)(BigBlock * bb, BigBlockPtr * ptr, BigArray * array),
            int root,
            MPI_Comm comm)
{
    size_t elsize = big_file_dtype_itemsize(block->dtype) * block->nmemb;

    /* This will aggregate to the root and write */
    BigBlockPtr ptr1[1];
    /* use memcpy because older compilers doesn't like *ptr assignments */
    memcpy(ptr1, ptr, sizeof(BigBlockPtr));

    int i;
    int e = 0;
    int rank;
    int nrank;

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nrank);

    BigArray garray[1], larray[1];
    BigArrayIter iarray[1], ilarray[1];
    void * lbuf = malloc(elsize * localsize);
    void * gbuf = NULL;

    int recvcounts[nrank];
    int recvdispls[nrank + 1];

    recvdispls[0] = 0;
    recvcounts[rank] = localsize;
    MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, recvcounts, 1, MPI_INT, comm);

    int grouptotalsize = localsize;

    MPI_Allreduce(MPI_IN_PLACE, &grouptotalsize, 1, MPI_INT, MPI_SUM, comm);

    for(i = 0; i < nrank; i ++) {
        recvdispls[i + 1] = recvdispls[i] + recvcounts[i];
    }

    MPI_Datatype mpidtype;
    MPI_Type_contiguous(elsize, MPI_BYTE, &mpidtype);
    MPI_Type_commit(&mpidtype);

    big_array_init(larray, lbuf, block->dtype, 2, (size_t[]){localsize, block->nmemb}, NULL);

    big_array_iter_init(iarray, array);
    big_array_iter_init(ilarray, larray);

    if(rank == root) {
        gbuf = malloc(grouptotalsize * elsize);
        big_array_init(garray, gbuf, block->dtype, 2, (size_t[]){grouptotalsize, block->nmemb}, NULL);
    }

    if(action == big_block_write) {
        _dtype_convert(ilarray, iarray, localsize * block->nmemb);
        MPI_Gatherv(lbuf, recvcounts[rank], mpidtype,
                    gbuf, recvcounts, recvdispls, mpidtype, root, comm);
    }
    if(rank == root) {
        big_block_seek_rel(block, ptr1, offset);
        e = action(block, ptr1, garray);
    }
    if(action == big_block_read) {
        MPI_Scatterv(gbuf, recvcounts, recvdispls, mpidtype,
                    lbuf, localsize, mpidtype, root, comm);
        _dtype_convert(iarray, ilarray, localsize * block->nmemb);
    }

    if(rank == root) {
        free(gbuf);
    }
    free(lbuf);

    MPI_Type_free(&mpidtype);

    return big_file_mpi_broadcast_anyerror(e, comm);
}

int
big_block_mpi_write(BigBlock * block, BigBlockPtr * ptr, BigArray * array, int concurrency, MPI_Comm comm)
{
    int rt = _throttle_action(comm, concurrency, block, ptr, array, big_block_write);
    return rt;
}

int
big_block_mpi_read(BigBlock * block, BigBlockPtr * ptr, BigArray * array, int concurrency, MPI_Comm comm)
{
    int rt = _throttle_action(comm, concurrency, block, ptr, array, big_block_read);
    return rt;
}


int
big_file_mpi_create_records(BigFile * bf,
    const BigRecordType * rtype,
    const char * mode,
    int Nfile,
    const size_t fsize[],
    MPI_Comm comm)
{
    int i;
    for(i = 0; i < rtype->nfield; i ++) {
        BigBlock block[1];
        if (0 == strcmp(mode, "w+")) {
            RAISEIF(0 != _big_file_mpi_create_block(bf, block,
                             rtype->fields[i].name,
                             rtype->fields[i].dtype,
                             rtype->fields[i].nmemb,
                             Nfile,
                             fsize,
                             comm),
                ex_open,
                NULL);
        } else if (0 == strcmp(mode, "a+")) {
            RAISEIF(0 != big_file_mpi_open_block(bf, block, rtype->fields[i].name, comm),
                ex_open,
                NULL);
            RAISEIF(0 != big_block_mpi_grow(block, Nfile, fsize, comm),
                ex_grow,
                NULL);
        } else {
            RAISE(ex_open,
                "Mode string must be `a+` or `w+`, `%s` provided",
                mode);
        }
        RAISEIF(0 != big_block_mpi_close(block, comm),
            ex_close,
            NULL);
        continue;
        ex_grow:
            RAISEIF(0 != big_block_mpi_close(block, comm),
            ex_close,
            NULL);
            return -1;
        ex_open:
        ex_close:
            return -1;
    }
    return 0;
}
int
big_file_mpi_write_records(BigFile * bf,
    const BigRecordType * rtype,
    ptrdiff_t offset,
    size_t size,
    const void * buf,
    int concurrency,
    MPI_Comm comm)
{
    int i;
    for(i = 0; i < rtype->nfield; i ++) {
        BigArray array[1];
        BigBlock block[1];
        BigBlockPtr ptr = {0};

        /* rainwoodman: cast away the const. We don't really modify it.*/
        RAISEIF(0 != big_record_view_field(rtype, i, array, size, (void*) buf),
            ex_array,
            NULL);
        RAISEIF(0 != big_file_mpi_open_block(bf, block, rtype->fields[i].name, comm),
            ex_open,
            NULL);
        RAISEIF(0 != big_block_seek(block, &ptr, offset),
            ex_seek,
            NULL);
        RAISEIF(0 != big_block_mpi_write(block, &ptr, array, concurrency, comm),
            ex_write,
            NULL);
        RAISEIF(0 != big_block_mpi_close(block, comm),
            ex_close,
            NULL);
        continue;
        ex_write:
        ex_seek:
            RAISEIF(0 != big_block_mpi_close(block, comm),
            ex_close,
            NULL);
            return -1;
        ex_open:
        ex_close:
        ex_array:
            return -1;
    }
    return 0;
}


int
big_file_mpi_read_records(BigFile * bf,
    const BigRecordType * rtype,
    ptrdiff_t offset,
    size_t size,
    void * buf,
    int concurrency,
    MPI_Comm comm)
{
    int i;
    for(i = 0; i < rtype->nfield; i ++) {
        BigArray array[1];
        BigBlock block[1];
        BigBlockPtr ptr = {0};

        RAISEIF(0 != big_record_view_field(rtype, i, array, size, buf),
            ex_array,
            NULL);
        RAISEIF(0 != big_file_mpi_open_block(bf, block, rtype->fields[i].name, comm),
            ex_open,
            NULL);
        RAISEIF(0 != big_block_seek(block, &ptr, offset),
            ex_seek,
            NULL);
        RAISEIF(0 != big_block_mpi_read(block, &ptr, array, concurrency, comm),
            ex_read,
            NULL);
        RAISEIF(0 != big_block_mpi_close(block, comm),
            ex_close,
            NULL);
        continue;
        ex_read:
        ex_seek:
            RAISEIF(0 != big_block_mpi_close(block, comm),
            ex_close,
            NULL);
            return -1;
        ex_open:
        ex_close:
        ex_array:
            return -1;
    }
    return 0;
}
