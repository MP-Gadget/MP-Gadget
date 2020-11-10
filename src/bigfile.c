#define _POSIX_C_SOURCE 200809L  // FIXME: scandir needs this
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <alloca.h>
#include <stdint.h>
#include <stddef.h>
#include <limits.h>
#include <stdarg.h>
#include <errno.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/time.h>
#include <unistd.h>
#include <dirent.h>

#include "bigfile.h"
#include "bigfile-internal.h"

#define EXT_HEADER "header"
#define EXT_ATTR "attr"
#define EXT_ATTR_V2 "attr-v2"
#define EXT_DATA   "%06X"
#define FILEID_ATTR -2
#define FILEID_ATTR_V2 -3
#define FILEID_HEADER -1

#if __STDC_VERSION__ >= 201112L && !defined(__STDC_NO_ATOMICS__)
    #include <stdatomic.h>
static char * volatile _Atomic ERRORSTR = NULL;
#else
static char * volatile ERRORSTR = NULL;
#endif

static size_t CHUNK_BYTES = 64 * 1024 * 1024;

/* Internal AttrSet API */

struct BigAttrSet {
    /* All members are readonly */
    int dirty;
    char * attrbuf;
    size_t bufused;
    size_t bufsize;

    BigAttr * attrlist;
    size_t listused;
    size_t listsize;
};

static BigAttrSet *
attrset_create(void);
static void attrset_free(BigAttrSet * attrset);
static int
attrset_read_attr_set_v1(BigAttrSet * attrset, const char * basename);
static int
attrset_read_attr_set_v2(BigAttrSet * attrset, const char * basename);
static int
attrset_write_attr_set_v2(BigAttrSet * attrset, const char * basename);
static BigAttr *
attrset_lookup_attr(BigAttrSet * attrset, const char * attrname);
static int
attrset_remove_attr(BigAttrSet * attrset, const char * attrname);
static BigAttr *
attrset_list_attrs(BigAttrSet * attrset, size_t * count);
static int
attrset_set_attr(BigAttrSet * attrset, const char * attrname, const void * data, const char * dtype, int nmemb);
static int
attrset_get_attr(BigAttrSet * attrset, const char * attrname, void * data, const char * dtype, int nmemb);

/* Internal dtype API */
static int
dtype_convert_simple(void * dst, const char * dstdtype, const void * src, const char * srcdtype, size_t nmemb);

/*Check dtype is valid*/
static int dtype_isvalid(const char * dtype);

/* postfix detection, 1 if postfix is a postfix of str */
static int
endswith(const char * str, const char * postfix)
{
    const char * p = str + strlen(str) - 1;
    const char * q = postfix + strlen(postfix) - 1;
    for(; p >= str && q >= postfix; p --, q--) {
        if(*p != *q) return 0;
    }
    return 1;
}

/* global settings */
int
big_file_set_buffer_size(size_t bytes)
{
    CHUNK_BYTES = bytes;
    return 0;
}

/* Error handling */
char * big_file_get_error_message() {
    return ERRORSTR;
}

void
big_file_set_error_message(char * msg)
{
    char * errorstr;
    if(msg != NULL) msg = _strdup(msg);

#if __STDC_VERSION__ >= 201112L && !defined(__STDC_NO_ATOMICS__)
    errorstr = atomic_exchange(&ERRORSTR, msg);
#elif defined(__ATOMIC_SEQ_CST)
    errorstr = __atomic_exchange_n(&ERRORSTR, msg, __ATOMIC_SEQ_CST);
#elif _OPENMP >= 201307
    /* openmp 4.0 supports swap capture*/
    #pragma omp capture 
     { errorstr = ERRORSTR; ERRORSTR = msg; }
#else
    errorstr = ERRORSTR;
    ERRORSTR = msg;
    /* really should have propagated errors over the stack! */
    #warning "enable -std=c11 or -std=gnu11, or luse gcc for thread friendly error handling"
#endif

    if(errorstr) free(errorstr);
}


static char * _path_join(const char * part1, const char * part2)
{
    size_t l1 = part1?strlen(part1):0;
    size_t l2 = strlen(part2);
    char * result = malloc(l1 + l2 + 10);
    if(l1 == 0) {
        sprintf(result, "%s", part2);
    } else
    if(part1[l1 - 1] == '/') {
        sprintf(result, "%s%s", part1, part2);
    } else {
        sprintf(result, "%s/%s", part1, part2);
    }
    return result;
}

static int
_big_file_path_is_block(const char * basename)
{
    FILE * fheader = _big_file_open_a_file(basename, FILEID_HEADER, "r", 0);
    if(fheader == NULL) {
        return 0;
    }
    fclose(fheader);
    return 1;
}


void
_big_file_raise(const char * msg, const char * file, const int line, ...)
{
    if(!msg) {
        if(ERRORSTR) {
            msg = ERRORSTR;
        } else {
            msg = "UNKNOWN ERROR";
        }
    }

    size_t len = strlen(msg) + strlen(file) + 512;
    char * errorstr = malloc(len);

    char * fmtstr = alloca(len);
    sprintf(fmtstr, "%s @(%s:%d)", msg, file, line);
    va_list va;
    va_start(va, line);
    vsprintf(errorstr, fmtstr, va);
    va_end(va);

    big_file_set_error_message(errorstr);
    free(errorstr);
}

/* BigFile */

int
big_file_open(BigFile * bf, const char * basename)
{
    struct stat st;
    RAISEIF(0 != stat(basename, &st),
            ex_stat,
            "Big File `%s' does not exist (%s)", basename,
            strerror(errno));
    bf->basename = _strdup(basename);
    return 0;
ex_stat:
    return -1;
}

int big_file_create(BigFile * bf, const char * basename) {
    bf->basename = _strdup(basename);
    RAISEIF(0 != _big_file_mksubdir_r(NULL, basename),
        ex_subdir,
        NULL);
    return 0;
ex_subdir:
    return -1;
}

struct bblist {
    struct bblist * next;
    char * blockname;
};
static int (filter)(const struct dirent * ent) {
    if(ent->d_name[0] == '.') return 0;
    return 1;
}

/* taken from glibc not always there. e.g. with c99. */
static int
_alphasort (const struct dirent **a, const struct dirent **b)
{
    return strcoll ((*a)->d_name, (*b)->d_name);
}

static struct
bblist * listbigfile_r(const char * basename, char * blockname, struct bblist * bblist) {
    struct dirent **namelist;
    int n;
    int i;

    char * current;
    current = _path_join(basename, blockname);

    n = scandir(current, &namelist, filter, _alphasort);
    free(current);
    if (n < 0)
        return bblist;

    for(i = 0; i < n ; i ++) {
        char * blockname1 = _path_join(blockname, namelist[i]->d_name);
        char * fullpath1 = _path_join(basename, blockname1);
        if(_big_file_path_is_block(fullpath1)) {
            struct bblist * n = malloc(sizeof(struct bblist) + strlen(blockname1) + 1);
            n->next = bblist;
            n->blockname = (char*) &n[1];
            strcpy(n->blockname, blockname1);
            bblist = n;
        } else {
            bblist = listbigfile_r(basename, blockname1, bblist);
        }
        free(fullpath1);
        free(blockname1);
        free(namelist[i]);
    }
    free(namelist);
    return bblist;
}

int
big_file_list(BigFile * bf, char *** blocknames, int * Nblocks)
{
    struct bblist * bblist = listbigfile_r(bf->basename, "", NULL);
    struct bblist * p;
    int N = 0;
    int i;
    for(p = bblist; p; p = p->next) {
        N ++;
    }
    *Nblocks = N;
    *blocknames = malloc(sizeof(char*) * N);
    for(i = 0; i < N; i ++) {
        (*blocknames)[i] = _strdup(bblist->blockname);
        p = bblist;
        bblist = bblist->next;
        free(p); 
    }
    return 0;
}

int
big_file_open_block(BigFile * bf, BigBlock * block, const char * blockname)
{
    char * basename = _path_join(bf->basename, blockname);
    int rt = _big_block_open(block, basename);
    free(basename);
    return rt;
}

int
big_file_create_block(BigFile * bf, BigBlock * block, const char * blockname, const char * dtype, int nmemb, int Nfile, const size_t fsize[])
{
    RAISEIF(0 != _big_file_mksubdir_r(bf->basename, blockname),
            ex_subdir,
            NULL);
    char * basename = _path_join(bf->basename, blockname);
    int rt = _big_block_create(block, basename, dtype, nmemb, Nfile, fsize);
    free(basename);
    return rt;
ex_subdir:
    return -1;
}

int
big_file_close(BigFile * bf)
{
    free(bf->basename);
    bf->basename = NULL;
    return 0;
}

static void
sysvsum(unsigned int * sum, void * buf, size_t size);

/* Bigblock */

int
_big_block_open(BigBlock * bb, const char * basename)
{
    memset(bb, 0, sizeof(bb[0]));
    if(basename == NULL) basename = "/.";
    bb->basename = _strdup(basename);
    bb->dirty = 0;

    bb->attrset = attrset_create();

    /* COMPATIBLE WITH V1 ATTR FILES */
    RAISEIF(0 != attrset_read_attr_set_v1(bb->attrset, bb->basename),
            ex_readattr,
            NULL);

    RAISEIF(0 != attrset_read_attr_set_v2(bb->attrset, bb->basename),
            ex_readattr,
            NULL);

    if (!endswith(bb->basename, "/.") && 0 != strcmp(bb->basename, ".")) {
        FILE * fheader = _big_file_open_a_file(bb->basename, FILEID_HEADER, "r", 1);
        RAISEIF (!fheader,
                ex_open,
                NULL);

        RAISEIF(
               (1 != fscanf(fheader, " DTYPE: %s", bb->dtype)) ||
               (1 != fscanf(fheader, " NMEMB: %d", &(bb->nmemb))) ||
               (1 != fscanf(fheader, " NFILE: %d", &(bb->Nfile))),
               ex_fscanf,
               "Failed to read header of block `%s' (%s)", bb->basename, strerror(errno));

        RAISEIF(bb->Nfile < 0 || bb->Nfile >= INT_MAX-1, ex_fscanf, 
                "Unreasonable value for Nfile in header of block `%s' (%d)",bb->basename,bb->Nfile);
        RAISEIF(bb->nmemb < 0, ex_fscanf, 
                "Unreasonable value for nmemb in header of block `%s' (%d)",bb->basename,bb->nmemb);
        RAISEIF(!dtype_isvalid(bb->dtype), ex_fscanf, 
                "Unreasonable value for dtype in header of block `%s' (%s)",bb->basename,bb->dtype);
        bb->fsize = calloc(bb->Nfile + 1, sizeof(size_t));
        RAISEIF(!bb->fsize,
                ex_fsize,
                "Failed to alloc memory of block `%s'", bb->basename);
        bb->foffset = calloc(bb->Nfile + 1, sizeof(size_t));
        RAISEIF(!bb->foffset,
                ex_foffset,
                "Failed to alloc memory of block `%s'", bb->basename);
        bb->fchecksum = calloc(bb->Nfile + 1, sizeof(int));
        RAISEIF(!bb->fchecksum,
                ex_fchecksum,
                "Failed to alloc memory `%s'", bb->basename);
        int i;
        for(i = 0; i < bb->Nfile; i ++) {
            int fid; 
            size_t size;
            unsigned int cksum;
            unsigned int sysv;
            RAISEIF(4 != fscanf(fheader, " " EXT_DATA ": %td : %u : %u", &fid, &size, &cksum, &sysv),
                    ex_fscanf1,
                    "Failed to readin physical file layout `%s' %d (%s)", bb->basename, fid,
                    strerror(errno));
            RAISEIF(fid < 0 || fid >= bb->Nfile, ex_fscanf1,
                    "Non-existent file referenced: `%s' (%d)", bb->basename, fid);
            bb->fsize[fid] = size;
            bb->fchecksum[fid] = cksum;
        }
        bb->foffset[0] = 0;
        for(i = 0; i < bb->Nfile; i ++) {
            bb->foffset[i + 1] = bb->foffset[i] + bb->fsize[i];
        }
        bb->size = bb->foffset[bb->Nfile];

        fclose(fheader);

        return 0;

ex_fscanf1:
        free(bb->fchecksum);
ex_fchecksum:
        free(bb->foffset);
ex_foffset:
        free(bb->fsize);
ex_fsize:
ex_fscanf:
        fclose(fheader);
ex_open:
        return -1;
    } else {
        /* The meta block of a big file, has on extra files. */
        bb->Nfile = 0;
        strcpy(bb->dtype, "####");
        return 0;
    }

ex_readattr:
        return -1;
}

int
_big_block_grow_internal(BigBlock * bb, int Nfile_grow, const size_t fsize_grow[])
{
    int Nfile = Nfile_grow + bb->Nfile;

    size_t * fsize = calloc(Nfile + 1, sizeof(size_t));
    size_t * foffset = calloc(Nfile + 1, sizeof(size_t));
    unsigned int * fchecksum = calloc(Nfile + 1, sizeof(unsigned int));

    int i;
    for(i = 0; i < bb->Nfile; i ++) {
        fsize[i] = bb->fsize[i];
        fchecksum[i] = bb->fchecksum[i];
    }

    for(i = bb->Nfile; i < Nfile; i ++) {
        fsize[i] = fsize_grow[i - bb->Nfile];
        fchecksum[i] = 0;
    }

    foffset[0] = 0;
    for(i = 0; i < Nfile; i ++) {
        foffset[i + 1] = foffset[i] + fsize[i];
    }

    free(bb->fsize);
    free(bb->foffset);
    free(bb->fchecksum);

    bb->fsize = fsize;
    bb->foffset = foffset;
    bb->fchecksum = fchecksum;
    bb->Nfile = Nfile;
    bb->size = bb->foffset[Nfile];
    bb->dirty = 1;
    return 0;
}

int
big_block_grow(BigBlock * bb, int Nfile_grow, const size_t fsize_grow[])
{
    int oldNfile = bb->Nfile;

    _big_block_grow_internal(bb, Nfile_grow, fsize_grow);

    int i;

    /* now truncate the new files */
    for(i = oldNfile; i < bb->Nfile; i ++) {
        FILE * fp = _big_file_open_a_file(bb->basename, i, "w", 1);
        RAISEIF(fp == NULL, 
                ex_fileio, 
                NULL);
        fclose(fp);
    }
    return 0;

ex_fileio:
    return -1;
}

int
_big_block_create_internal(BigBlock * bb, const char * basename, const char * dtype, int nmemb, int Nfile, const size_t fsize[])
{
    memset(bb, 0, sizeof(bb[0]));

    if(basename == NULL) basename = "/.";

    RAISEIF (
         strchr(basename, ' ')
      || strchr(basename, '\t')
      || strchr(basename, '\n'),
      ex_name,
      "Column name cannot contain blanks (space, tab or newline)"
    );

    bb->basename = _strdup(basename);

    bb->attrset = attrset_create();
    bb->attrset->dirty = 1;

    if (!endswith(bb->basename, "/.") && 0 != strcmp(bb->basename, ".")) {
        if(dtype == NULL) {
            dtype = "i8";
            Nfile = 0;
            fsize = NULL;
        }
        /* always use normalized dtype in files. */
        _dtype_normalize(bb->dtype, dtype);

        bb->Nfile = Nfile;
        bb->nmemb = nmemb;
        bb->fsize = calloc(bb->Nfile + 1, sizeof(size_t));
        RAISEIF(!bb->fsize, ex_fsize, "No memory"); 
        bb->fchecksum = calloc(bb->Nfile + 1, sizeof(int));
        RAISEIF(!bb->fchecksum, ex_fchecksum, "No memory"); 
        bb->foffset = calloc(bb->Nfile + 1, sizeof(size_t));
        RAISEIF(!bb->foffset, ex_foffset, "No memory"); 
        int i;
        bb->foffset[0] = 0;
        for(i = 0; i < bb->Nfile; i ++) {
            bb->fsize[i] = fsize[i];
            bb->foffset[i + 1] = bb->foffset[i] + bb->fsize[i];
            bb->fchecksum[i] = 0;
        }

        bb->size = bb->foffset[bb->Nfile];
        bb->dirty = 1;

        RAISEIF(0 != big_block_flush(bb), 
                ex_flush, NULL);

        bb->dirty = 0;
        return 0;
ex_flush:
        attrset_free(bb->attrset);
        free(bb->foffset);
ex_foffset:
        free(bb->fchecksum);
ex_fchecksum:
        free(bb->fsize);
ex_fsize:
        return -1;
    } else {
        /* special meta block of a file */
        bb->Nfile = 0;
        /* never flush the header */
        bb->dirty = 0;

        RAISEIF(0 != big_block_flush(bb), 
                ex_flush2, NULL);

        bb->dirty = 0;
        return 0;

ex_flush2:
        attrset_free(bb->attrset);
        return -1;
    }

ex_name:
    return -1;
}

int
_big_block_create(BigBlock * bb, const char * basename, const char * dtype, int nmemb, int Nfile, const size_t fsize[])
{
    int rt = _big_block_create_internal(bb, basename, dtype, nmemb, Nfile, fsize);
    int i;
    RAISEIF(rt != 0,
                ex_internal,
                NULL);

    /* now truncate all files */
    for(i = 0; i < bb->Nfile; i ++) {
        FILE * fp = _big_file_open_a_file(bb->basename, i, "w", 1);
        RAISEIF(fp == NULL, 
                ex_fileio, 
                NULL);
        fclose(fp);
    }
ex_internal:
    return rt;
ex_fileio:
    _big_block_close_internal(bb);
    return -1;
}

void
big_block_set_dirty(BigBlock * block, int value)
{
    block->dirty = value;
}

int
big_block_flush(BigBlock * block)
{
    FILE * fheader = NULL;
    if(block->dirty) {
        int i;
        fheader = _big_file_open_a_file(block->basename, FILEID_HEADER, "w+", 1);
        RAISEIF(fheader == NULL, ex_fileio, NULL);
        RAISEIF(
            (0 > fprintf(fheader, "DTYPE: %s\n", block->dtype)) ||
            (0 > fprintf(fheader, "NMEMB: %d\n", block->nmemb)) ||
            (0 > fprintf(fheader, "NFILE: %d\n", block->Nfile)),
                ex_fprintf,
                "Writing file header");
        for(i = 0; i < block->Nfile; i ++) {
            unsigned int s = block->fchecksum[i];
            unsigned int r = (s & 0xffff) + ((s & 0xffffffff) >> 16);
            unsigned int checksum = (r & 0xffff) + (r >> 16);
            RAISEIF(0 > fprintf(fheader, EXT_DATA ": %td : %u : %u\n", i, block->fsize[i], block->fchecksum[i], checksum),
                ex_fprintf, "Writing file information to header");
        }
        fclose(fheader);
        block->dirty = 0;
    }
    if(block->attrset->dirty) {
        RAISEIF(0 != attrset_write_attr_set_v2(block->attrset, block->basename),
            ex_write_attr,
            NULL);
        block->attrset->dirty = 0;
    }
    return 0;

ex_fprintf:
    fclose(fheader);
ex_write_attr:
ex_fileio:
    return -1;
}

void
_big_block_close_internal(BigBlock * block)
{
    attrset_free(block->attrset);

    free(block->basename);
    free(block->fchecksum);
    free(block->fsize);
    free(block->foffset);
    memset(block, 0, sizeof(BigBlock));
}

int
big_block_close(BigBlock * block)
{
    int rt = 0;
    RAISEIF(0 != big_block_flush(block),
            ex_flush,
            NULL);
finalize:
    _big_block_close_internal(block);
    return rt;

ex_flush:
    rt = -1;
    goto finalize;
}

BigAttr *
big_block_lookup_attr(BigBlock * block, const char * attrname)
{
    BigAttr * attr = attrset_lookup_attr(block->attrset, attrname);
    return attr;
}

int
big_block_remove_attr(BigBlock * block, const char * attrname)
{
    return attrset_remove_attr(block->attrset, attrname);
}

BigAttr *
big_block_list_attrs(BigBlock * block, size_t * count)
{
    return attrset_list_attrs(block->attrset, count);
}

int
big_block_set_attr(BigBlock * block, const char * attrname, const void * data, const char * dtype, int nmemb)
{
    return attrset_set_attr(block->attrset, attrname, data, dtype, nmemb);
}

int
big_block_get_attr(BigBlock * block, const char * attrname, void * data, const char * dtype, int nmemb)
{
    return attrset_get_attr(block->attrset, attrname, data, dtype, nmemb);
}

/* *
 * seek ptr to offset, on bb
 *  
 *      offset: 0 : Start
 *              -1 : End
 *              -2 : End - 1
 *
 * returns 0
 *
 * 0 4 5 10 140  
 * */
int
big_block_seek(BigBlock * bb, BigBlockPtr * ptr, ptrdiff_t offset)
{
    /* handle 0 sized files */
    if(bb->size == 0 && offset == 0) {
        ptr->fileid = 0;
        ptr->roffset = 0;
        ptr->aoffset = 0;
        return 0;
    }
    /* handle negatives */
    if(offset < 0) offset += bb->foffset[bb->Nfile];

    RAISEIF(offset > bb->size, 
            ex_eof,
        /* over the end of file */
        /* note that we allow seeking at the end of file */
            "Over the end of file %td of %td",
            offset, bb->size);
    {
        int left = 0;
        int right = bb->Nfile;
        while(right > left + 1) {
            int mid = ((right - left) >> 1) + left;
            if(bb->foffset[mid] <= offset) {
                left = mid;
            } else {
                right = mid;
            }
        }
        ptr->fileid = left;
        ptr->roffset = offset - bb->foffset[left];
        ptr->aoffset = offset;
        return 0;
    }
ex_eof:
    return -1;
}

int
big_block_seek_rel(BigBlock * bb, BigBlockPtr * ptr, ptrdiff_t rel)
{
    ptrdiff_t abs = bb->foffset[ptr->fileid] + ptr->roffset + rel;
    return big_block_seek(bb, ptr, abs);
}

int
big_block_eof(BigBlock * bb, BigBlockPtr * ptr)
{
    ptrdiff_t abs = bb->foffset[ptr->fileid] + ptr->roffset;
    return abs >= bb->size;
}

/* 
 * this function will alloc memory in array and read from offset start 
 * size of rows from the block. 
 * free(array->data) after using it.
 *
 * at most size rows are read, array.dims[0] has the number that's read.
 *
 * if dtype is NULL use the dtype of the block.
 * otherwise cast the array to the dtype
 * */
int
big_block_read_simple(BigBlock * bb, ptrdiff_t start, ptrdiff_t size, BigArray * array, const char * dtype)
{
    BigBlockPtr ptr = {0};
    if(dtype == NULL) {
        dtype = bb->dtype;
    }
    void * buffer;
    size_t dims[2];

    RAISEIF(0 != big_block_seek(bb, &ptr, start),
       ex_seek,
       "failed to seek"       
    );

    if(start + size > bb->size){
        size = bb->size - start;
    }
    RAISEIF(size < 0,
            ex_seek,
            "failed to seek");

    buffer = malloc(size * big_file_dtype_itemsize(dtype) * bb->nmemb);

    dims[0] = size;
    dims[1] = bb->nmemb;

    big_array_init(array, buffer, dtype, 2, dims, NULL);

    RAISEIF(0 != big_block_read(bb, &ptr, array),
            ex_read,
            "failed to read");
    return 0;
ex_read:
    free(buffer);
    array->data = NULL;
ex_seek:
    return -1;
}

int
big_block_read(BigBlock * bb, BigBlockPtr * ptr, BigArray * array)
{
    char * chunkbuf = malloc(CHUNK_BYTES);

    int nmemb = bb->nmemb ? bb->nmemb : 1;
    int felsize = big_file_dtype_itemsize(bb->dtype) * nmemb;
    size_t CHUNK_SIZE = CHUNK_BYTES / felsize;

    BigArray chunk_array = {0};
    size_t dims[2];
    dims[0] = CHUNK_SIZE;
    dims[1] = bb->nmemb;

    BigArrayIter chunk_iter;
    BigArrayIter array_iter;

    FILE * fp = NULL;
    ptrdiff_t toread = 0;

    RAISEIF(chunkbuf == NULL,
            ex_malloc,
            "Not enough memory for chunkbuf");

    big_array_init(&chunk_array, chunkbuf, bb->dtype, 2, dims, NULL);
    big_array_iter_init(&array_iter, array);

    toread = array->size / nmemb;

    ptrdiff_t abs = bb->foffset[ptr->fileid] + ptr->roffset + toread;
    RAISEIF(abs > bb->size,
                ex_eof,
                "Reading beyond the block `%s` at (%d:%td)",
                bb->basename, ptr->fileid, ptr->roffset * felsize);

    while(toread > 0 && ! big_block_eof(bb, ptr)) {
        size_t chunk_size = CHUNK_SIZE;
        /* remaining items in the file */
        if(chunk_size > bb->fsize[ptr->fileid] - ptr->roffset) {
            chunk_size = bb->fsize[ptr->fileid] - ptr->roffset;
        }
        /* remaining items to read */
        if(chunk_size > toread) {
            chunk_size = toread;
        }
        RAISEIF(chunk_size == 0,
            ex_insuf,
            "Insufficient number of items in file `%s' at (%d:%td)",
            bb->basename, ptr->fileid, ptr->roffset * felsize);

        /* read to the beginning of chunk */
        big_array_iter_init(&chunk_iter, &chunk_array);

        fp = _big_file_open_a_file(bb->basename, ptr->fileid, "r", 1);
        RAISEIF(fp == NULL,
                ex_open,
                NULL);
        RAISEIF(0 > fseek(fp, ptr->roffset * felsize, SEEK_SET),
                ex_seek,
                "Failed to seek in block `%s' at (%d:%td) (%s)", 
                bb->basename, ptr->fileid, ptr->roffset * felsize, strerror(errno));
        RAISEIF(chunk_size != fread(chunkbuf, felsize, chunk_size, fp),
                ex_read,
                "Failed to read in block `%s' at (%d:%td) (%s)",
                bb->basename, ptr->fileid, ptr->roffset * felsize, strerror(errno));
        fclose(fp);
        fp = NULL;

        /* now translate the data from chunkbuf to mptr */
        RAISEIF(0 != _dtype_convert(&array_iter, &chunk_iter, chunk_size * bb->nmemb),
            ex_convert, NULL);

        toread -= chunk_size;
        RAISEIF(0 != big_block_seek_rel(bb, ptr, chunk_size),
                ex_blockseek,
                NULL);
    }

    free(chunkbuf);
    return 0;
ex_read:
ex_seek:
    fclose(fp);
ex_insuf:
ex_convert:
ex_blockseek:
ex_open:
ex_eof:
    free(chunkbuf);
ex_malloc:
    return -1;
}

int
big_block_write(BigBlock * bb, BigBlockPtr * ptr, BigArray * array)
{
    if(array->size == 0) return 0;
    /* the file header is modified */
    bb->dirty = 1;
    char * chunkbuf = malloc(CHUNK_BYTES);
    int nmemb = bb->nmemb ? bb->nmemb : 1;
    int felsize = big_file_dtype_itemsize(bb->dtype) * nmemb;
    size_t CHUNK_SIZE = CHUNK_BYTES / felsize;

    BigArray chunk_array = {0};
    size_t dims[2];
    dims[0] = CHUNK_SIZE;
    dims[1] = bb->nmemb;

    BigArrayIter chunk_iter;
    BigArrayIter array_iter;
    ptrdiff_t towrite = 0;
    FILE * fp;

    RAISEIF(chunkbuf == NULL,
            ex_malloc,
            "not enough memory for chunkbuf of size %d bytes", CHUNK_BYTES);

    big_array_init(&chunk_array, chunkbuf, bb->dtype, 2, dims, NULL);
    big_array_iter_init(&array_iter, array);

    towrite = array->size / nmemb;

    ptrdiff_t abs = bb->foffset[ptr->fileid] + ptr->roffset + towrite;
    RAISEIF(abs > bb->size,
                ex_eof,
                "Writing beyond the block `%s` at (%d:%td)",
                bb->basename, ptr->fileid, ptr->roffset * felsize);

    while(towrite > 0 && ! big_block_eof(bb, ptr)) {
        size_t chunk_size = CHUNK_SIZE;
        /* remaining items in the file */
        if(chunk_size > bb->fsize[ptr->fileid] - ptr->roffset) {
            chunk_size = bb->fsize[ptr->fileid] - ptr->roffset;
        }
        /* remaining items to read */
        if(chunk_size > towrite) {
            chunk_size = towrite;
        }
        /* write from the beginning of chunk */
        big_array_iter_init(&chunk_iter, &chunk_array);

        /* now translate the data to format in the file*/
        RAISEIF(0 != _dtype_convert(&chunk_iter, &array_iter, chunk_size * bb->nmemb),
            ex_convert, NULL);

        sysvsum(&bb->fchecksum[ptr->fileid], chunkbuf, chunk_size * felsize);

        fp = _big_file_open_a_file(bb->basename, ptr->fileid, "r+", 1);
        RAISEIF(fp == NULL,
                ex_open,
                NULL);
        RAISEIF(0 > fseek(fp, ptr->roffset * felsize, SEEK_SET),
                ex_seek,
                "Failed to seek in block `%s' at (%d:%td) (%s)", 
                bb->basename, ptr->fileid, ptr->roffset * felsize, strerror(errno));
        RAISEIF(chunk_size != fwrite(chunkbuf, felsize, chunk_size, fp),
                ex_write,
                "Failed to write in block `%s' at (%d:%td) (%s)",
                bb->basename, ptr->fileid, ptr->roffset * felsize, strerror(errno));
        fclose(fp);

        towrite -= chunk_size;
        RAISEIF(0 != big_block_seek_rel(bb, ptr, chunk_size),
                ex_blockseek, NULL);
    }
    free(chunkbuf);
    return 0;
ex_write:
ex_seek:
    fclose(fp);
ex_convert:
ex_open:
ex_blockseek:
ex_eof:
    free(chunkbuf);
ex_malloc:
    return -1;
}

/**
 * dtype stuff
 * */

#define MACHINE_ENDIANNESS MACHINE_ENDIAN_F()
static char
MACHINE_ENDIAN_F(void)
{
    uint32_t i = 0x01234567;
    return ((*((uint8_t*)(&i))) == 0x67)?'<':'>';
}

int
_dtype_normalize(char * dst, const char * src)
{
/* normalize a dtype, so that
 * dst[0] is the endian-ness
 * dst[1] is the type kind char
 * dtype[2:] is the width
 * */
    memset(dst, 0, 8);
    switch(src[0]) {
        case '<':
        case '>':
        case '|':
        case '=':
            strncpy(dst, src, 8);
        break;
        default:
            dst[0] = '=';
            strncpy(dst + 1, src, 7);
    }
    dst[7]='\0';
    if(dst[0] == '=') {
        dst[0] = MACHINE_ENDIANNESS;
    }
    if(dst[0] == '|') {
        dst[0] = MACHINE_ENDIANNESS;
    }
    return 0;
}

/*Check that the passed dtype is valid.
 * Returns 1 if valid, 0 if invalid*/
static int
dtype_isvalid(const char * dtype)
{
    if(!dtype)
        return 0;
    switch(dtype[0]) {
        case '<':
        case '>':
        case '|':
        case '=':
            break;
        default:
            return 0;
    }
    switch(dtype[1]) {
        case 'S':
        case 'b':
        case 'i':
        case 'f':
        case 'u':
        case 'c':
            break;
        default:
            return 0;
    }
    int width = atoi(&dtype[2]);
    if(width > 16 || width <= 0)
        return 0;
    return 1;
}

int
big_file_dtype_itemsize(const char * dtype)
{
    char ndtype[8];
    _dtype_normalize(ndtype, dtype);
    return atoi(&ndtype[2]);
}

int
big_file_dtype_kind(const char * dtype)
{
    char ndtype[8];
    _dtype_normalize(ndtype, dtype);
    return ndtype[1];
}

int
big_array_init(BigArray * array, void * buf, const char * dtype, int ndim, const size_t dims[], const ptrdiff_t strides[])
{

    memset(array, 0, sizeof(array[0]));

    _dtype_normalize(array->dtype, dtype);
    array->data = buf;
    array->ndim = ndim;
    int i;
    memset(array->dims, 0, sizeof(ptrdiff_t) * 32);
    memset(array->strides, 0, sizeof(ptrdiff_t) * 32);
    array->size = 1;
    for(i = 0; i < ndim; i ++) {
        array->dims[i] = dims[i];
        array->size *= dims[i];
    }
    if(strides != NULL) {
        for(i = 0; i < ndim; i ++) {
            array->strides[i] = strides[i];
        }
    } else {
        array->strides[ndim - 1] = big_file_dtype_itemsize(dtype);
        for(i = ndim - 2; i >= 0; i --) {
            array->strides[i] = array->strides[i + 1] * array->dims[i + 1];
        }
    }
    return 0;
}

int
big_array_iter_init(BigArrayIter * iter, BigArray * array)
{
    memset(iter, 0, sizeof(iter[0]));

    iter->array = array;

    memset(iter->pos, 0, sizeof(ptrdiff_t) * 32);
    iter->dataptr = array->data;

    /* see if the iter is contiguous */
    size_t elsize = big_file_dtype_itemsize(array->dtype);

    int i = 0; 
    ptrdiff_t stride_contiguous = elsize;
    iter->contiguous = 1;
    for(i = array->ndim - 1; i >= 0; i --) {
        if(array->strides[i] != stride_contiguous) {
            iter->contiguous = 0;
            break;
        }
        stride_contiguous *= array->dims[i];
    }
    return 0;
}

void
big_array_iter_advance(BigArrayIter * iter)
{
    BigArray * array = iter->array;

    if(iter->contiguous) {
        iter->dataptr = (char*) iter->dataptr + array->strides[array->ndim - 1];
        return;
    }
    int k;
    iter->pos[array->ndim - 1] ++;
    iter->dataptr = ((char*) iter->dataptr) + array->strides[array->ndim - 1];
    for(k = array->ndim - 1; k >= 0; k --) {
        if(iter->pos[k] == array->dims[k]) {
            iter->dataptr = ((char*) iter->dataptr) - array->strides[k] * iter->pos[k];
            iter->pos[k] = 0;
            if(k > 0) {
                iter->pos[k - 1] ++;
                iter->dataptr = ((char*) iter->dataptr) + array->strides[k - 1];
            }
        } else {
            break;
        }
    }
}
typedef struct {double r; double i;} cplx128_t;
typedef struct {float r; float i;} cplx64_t;
typedef union {
    char *a1;
    char *b1;
    int64_t *i8;
    uint64_t *u8;
    double *f8;
    int32_t *i4;
    uint32_t *u4;
    float *f4;
    cplx128_t * c16;
    cplx64_t * c8;
    void * v;
} variant_t;

/* format data in dtype to a string in buffer */
void
big_file_dtype_format(char * buffer, const char * dtype, const void * data, const char * fmt)
{
    char ndtype[8];
    char ndtype2[8];
    variant_t p;

    /* handle the endianness stuff in case it is not machine */
    char converted[128];

    _dtype_normalize(ndtype2, dtype);
    ndtype2[0] = '=';
    _dtype_normalize(ndtype, ndtype2);
    dtype_convert_simple(converted, ndtype, data, dtype, 1);

    p.v = converted;
#define FORMAT1(dtype, defaultfmt) \
    if(0 == strcmp(ndtype + 1, # dtype)) { \
        if(fmt == NULL) fmt = defaultfmt; \
        sprintf(buffer, fmt, *p.dtype); \
    } else
#define FORMAT2(dtype, defaultfmt) \
    if(0 == strcmp(ndtype + 1, # dtype)) { \
        if(fmt == NULL) fmt = defaultfmt; \
        sprintf(buffer, fmt, p.dtype->r, p.dtype->i); \
    } else

    FORMAT1(a1, "%c")
    FORMAT1(b1, "%d")
    FORMAT1(i8, "%ld")
    FORMAT1(i4, "%d")
    FORMAT1(u8, "%lu")
    FORMAT1(u4, "%u")
    FORMAT1(f8, "%g")
    FORMAT1(f4, "%g")
    FORMAT2(c8, "%g+%gI")
    FORMAT2(c16, "%g+%gI")
    {
        sprintf(buffer, "<%s>", ndtype);
    }
}

/* parse data in dtype to a string in buffer */
int
big_file_dtype_parse(const char * buffer, const char * dtype, void * data, const char * fmt)
{
    char ndtype[8];
    char ndtype2[8];
    variant_t p;

    /* handle the endianness stuff in case it is not machine */
    char converted[128];

    _dtype_normalize(ndtype2, dtype);
    ndtype2[0] = '=';
    _dtype_normalize(ndtype, ndtype2);

    p.v = converted;
#define PARSE1(dtype, defaultfmt) \
    if(0 == strcmp(ndtype + 1, # dtype)) { \
        if(fmt == NULL) fmt = defaultfmt; \
        sscanf(buffer, fmt, p.dtype); \
    }
#define PARSE2(dtype, defaultfmt) \
    if(0 == strcmp(ndtype + 1, # dtype)) { \
        if(fmt == NULL) fmt = defaultfmt; \
        sscanf(buffer, fmt, &p.dtype->r, &p.dtype->i); \
    }
    PARSE1(a1, "%c") else
    PARSE1(i8, "%ld") else
    PARSE1(i4, "%d") else
    PARSE1(u8, "%lu") else
    PARSE1(u4, "%u") else
    PARSE1(f8, "%lf") else
    PARSE1(f4, "%f") else
    PARSE2(c8, "%f + %f I") else
    PARSE2(c16, "%lf + %lf I") else
    return -1;

    dtype_convert_simple(data, dtype, converted, ndtype, 1);

    return 0;
}

static int
dtype_convert_simple(void * dst, const char * dstdtype, const void * src, const char * srcdtype, size_t nmemb)
{
    BigArray dst_array, src_array;
    BigArrayIter dst_iter, src_iter;
    big_array_init(&dst_array, dst, dstdtype, 1, &nmemb, NULL);
    big_array_init(&src_array, (void*) src, srcdtype, 1, &nmemb, NULL);
    big_array_iter_init(&dst_iter, &dst_array);
    big_array_iter_init(&src_iter, &src_array);
    return _dtype_convert(&dst_iter, &src_iter, nmemb);
}

static int cast(BigArrayIter * dst, BigArrayIter * src, size_t nmemb);
static void byte_swap(BigArrayIter * array, size_t nmemb);
int
_dtype_convert(BigArrayIter * dst, BigArrayIter * src, size_t nmemb)
{
    /* cast buf2 of dtype2 into buf1 of dtype1 */
    /* match src to machine endianness */
    if(src->array->dtype[0] != MACHINE_ENDIANNESS) {
        BigArrayIter iter = *src;
        byte_swap(&iter, nmemb);
    }

    BigArrayIter iter1 = *dst;
    BigArrayIter iter2 = *src;

    if(0 != cast(&iter1, &iter2, nmemb)) {
        /* cast is not supported */
        return -1;
    }

    /* match dst to machine endianness */
    if(dst->array->dtype[0] != MACHINE_ENDIANNESS) {
        BigArrayIter iter = *dst;
        byte_swap(&iter, nmemb);
    }
    *dst = iter1;
    *src = iter2;
    return 0;
}


static void
byte_swap(BigArrayIter * iter, size_t nmemb)
{
    /* swap a buffer in-place */
    int elsize = big_file_dtype_itemsize(iter->array->dtype);
    if(elsize == 1) return;
    /* need byte swap; do it now on buf2 */
    /* XXX: this may still be wrong. */
    ptrdiff_t i;
    int half = elsize >> 1;
    for(i = 0; i < nmemb; i ++) {
        int j;
        char * ptr = iter->dataptr;
        for(j = 0; j < half; j ++) {
            char tmp = ptr[j];
            ptr[j] = ptr[elsize - j - 1];
            ptr[elsize - j - 1] = tmp;
        }
        big_array_iter_advance(iter);
    }
}

#define CAST(d1, t1, d2, t2) \
if((0 == strcmp(d1, dst->array->dtype + 1)) && (0 == strcmp(d2, src->array->dtype + 1))) { \
    for(i = 0; i < nmemb; i ++) { \
        t1 * p1 = dst->dataptr; t2 * p2 = src->dataptr; \
        * p1 = * p2; \
        big_array_iter_advance(dst); big_array_iter_advance(src); \
    } \
    return 0; \
}
#define CAST2(d1, t1, d2, t2) \
if((0 == strcmp(d1, dst->array->dtype + 1)) && (0 == strcmp(d2, src->array->dtype + 1))) { \
    for(i = 0; i < nmemb; i ++) { \
        t1 * p1 = dst->dataptr; t2 * p2 = src->dataptr; \
        p1->r = p2->r; p1->i = p2->i; \
        big_array_iter_advance(dst); big_array_iter_advance(src); \
    } \
    return 0; \
}
static int
cast(BigArrayIter * dst, BigArrayIter * src, size_t nmemb)
{
    /* doing cast assuming native byte order */
    /* convert buf2 to buf1, both are native;
     * dtype has no endian-ness prefix
     *   */
    ptrdiff_t i;
    /* same type, no need for casting. */
    if (0 == strcmp(dst->array->dtype + 1, src->array->dtype + 1)) {
        if(dst->contiguous && src->contiguous) {
            /* directly copy of memory chunks; FIXME: use memmove? */
            memcpy(dst->dataptr, src->dataptr, nmemb * dst->array->strides[dst->array->ndim-1]);
            dst->dataptr = (char*) dst->dataptr + nmemb * dst->array->strides[dst->array->ndim - 1];
            src->dataptr = (char*) src->dataptr + nmemb * src->array->strides[src->array->ndim - 1];
            return 0;
        } else {
            /* copy one by one, discontinuous */
            size_t elsize = big_file_dtype_itemsize(dst->array->dtype);
            for(i = 0; i < nmemb; i ++) {
                void * p1 = dst->dataptr;
                void * p2 = src->dataptr;
                memcpy(p1, p2, elsize);
                big_array_iter_advance(dst); big_array_iter_advance(src);
            }
            return 0;
        }
    }
    if(0 == strcmp(dst->array->dtype + 1, "i8")) {
        CAST("i8", int64_t, "i4", int32_t);
        CAST("i8", int64_t, "u4", uint32_t);
        CAST("i8", int64_t, "u8", uint64_t);
        CAST("i8", int64_t, "f8", double);
        CAST("i8", int64_t, "f4", float);
        CAST("i8", int64_t, "b1", char);
    } else
    if(0 == strcmp(dst->array->dtype + 1, "u8")) {
        CAST("u8", uint64_t, "u4", uint32_t);
        CAST("u8", uint64_t, "i4", int32_t);
        CAST("u8", uint64_t, "i8", int64_t);
        CAST("u8", uint64_t, "f8", double);
        CAST("u8", uint64_t, "f4", float);
        CAST("u8", uint64_t, "b1", char);
    } else
    if(0 == strcmp(dst->array->dtype + 1, "f8")) {
        CAST("f8", double, "f4", float);
        CAST("f8", double, "i4", int32_t);
        CAST("f8", double, "i8", int64_t);
        CAST("f8", double, "u4", uint32_t);
        CAST("f8", double, "u8", uint64_t);
        CAST("f8", double, "b1", char);
    } else
    if(0 == strcmp(dst->array->dtype + 1, "i4")) {
        CAST("i4", int32_t, "i8", int64_t);
        CAST("i4", int32_t, "u4", uint32_t);
        CAST("i4", int32_t, "u8", uint64_t);
        CAST("i4", int32_t, "f8", double);
        CAST("i4", int32_t, "f4", float);
        CAST("i4", int32_t, "b1", char);
    } else
    if(0 == strcmp(dst->array->dtype + 1, "u4")) {
        CAST("u4", uint32_t, "u8", uint64_t);
        CAST("u4", uint32_t, "i4", int32_t);
        CAST("u4", uint32_t, "i8", int64_t);
        CAST("u4", uint32_t, "f8", double);
        CAST("u4", uint32_t, "f4", float);
        CAST("u4", uint32_t, "b1", char);
    } else
    if(0 == strcmp(dst->array->dtype + 1, "f4")) {
        CAST("f4", float, "f8", double);
        CAST("f4", float, "i4", int32_t);
        CAST("f4", float, "i8", int64_t);
        CAST("f4", float, "u4", uint32_t);
        CAST("f4", float, "u8", uint64_t);
        CAST("f4", float, "b1", char);
    } else
    if(0 == strcmp(dst->array->dtype + 1, "c8")) {
        CAST2("c8", cplx64_t, "c16", cplx128_t);
    } else
    if(0 == strcmp(dst->array->dtype + 1, "c16")) {
        CAST2("c16", cplx128_t, "c8", cplx64_t);
    }
    RAISE(ex, "Unsupported conversion from %s to %s. ", src->array->dtype, dst->array->dtype);
ex:
    return -1;
}

static void
sysvsum(unsigned int * sum, void * buf, size_t size)
{
    unsigned int thisrun = *sum;
    unsigned char * cp = buf;
    while(size --)    
        thisrun += *(cp++);
    *sum = thisrun;
}

/*
 * Internal API for AttrSet objects;
 * */

static int
attrset_read_attr_set_v1(BigAttrSet * attrset, const char * basename)
{
    attrset->dirty = 0;

    FILE * fattr = _big_file_open_a_file(basename, FILEID_ATTR, "r", 0);
    if(fattr == NULL) {
        return 0;
    }
    int nmemb;
    int lname;
    char dtype[9]={0};
    char * data;
    char * name;
    while(!feof(fattr)) {
        if(1 != fread(&nmemb, sizeof(int), 1, fattr)) break;
        RAISEIF(
            (1 != fread(&lname, sizeof(int), 1, fattr)) ||
            (1 != fread(&dtype, 8, 1, fattr)) ||
            (!dtype_isvalid(dtype)),
            ex_fread,
            "Failed to read from file"
                )
        int ldata = big_file_dtype_itemsize(dtype) * nmemb;
        data = alloca(ldata);
        name = alloca(lname + 1);
        RAISEIF(
            (1 != fread(name, lname, 1, fattr)) ||
            (1 != fread(data, ldata, 1, fattr)),
            ex_fread,
            "Failed to read from file");

        name[lname] = 0;
        RAISEIF(0 != attrset_set_attr(attrset, name, data, dtype, nmemb),
            ex_set_attr,
            NULL);
    }
    attrset->dirty = 0;
    fclose(fattr);
    return 0;
ex_set_attr:
ex_fread:
    attrset->dirty = 0;
    fclose(fattr);
    return -1;
}

static int _isblank(int ch) {
    return ch == ' ' || ch == '\t';
}

static int
attrset_read_attr_set_v2(BigAttrSet * attrset, const char * basename)
{
    attrset->dirty = 0;

    FILE * fattr = _big_file_open_a_file(basename, FILEID_ATTR_V2, "r", 0);
    if(fattr == NULL) {
        return 0;
    }
    fseek(fattr, 0, SEEK_END);
    long size = ftell(fattr);
    /*ftell may fail*/
    RAISEIF(size < 0, ex_init, "ftell error: %s",strerror(errno));
    char * buffer = (char*) malloc(size + 1);
    RAISEIF(!buffer, ex_init, "Could not allocate memory for buffer: %ld bytes",size+1);
    unsigned char * data = (unsigned char * ) malloc(size + 1);
    RAISEIF(!data, ex_data, "Could not allocate memory for data: %ld bytes",size+1);
    fseek(fattr, 0, SEEK_SET);
    RAISEIF(size != fread(buffer, 1, size, fattr),
            ex_read_file,
            "Failed to read attribute file\n");
    buffer[size] = 0;

    /* now parse the v2 attr file.*/
    long i = 0;
    #define ATTRV2_EXPECT(variable) while(_isblank(buffer[i])) i++; \
                    char * variable = buffer + i; \
                    while(!_isblank(buffer[i])) i++; buffer[i] = 0; i++;
    while(buffer[i]) {
        ATTRV2_EXPECT(name);
        ATTRV2_EXPECT(dtype);
        ATTRV2_EXPECT(rawlength);
        ATTRV2_EXPECT(rawdata);
        /* skip the reset of the line */
        while(buffer[i] != '\n' && buffer[i]) i ++;
        if(buffer[i] == '\n') i++;

        int nmemb = atoi(rawlength);
        int itemsize = big_file_dtype_itemsize(dtype);

        RAISEIF(nmemb * itemsize * 2!= strlen(rawdata),
            ex_parse_attr,
            "NMEMB and data mismiatch: %d x %d (%s) * 2 != %d",
            nmemb, itemsize, dtype, strlen(rawdata));

        int j, k;
        for(k = 0, j = 0; k < nmemb * itemsize; k ++, j += 2) {
            char buf[3];
            buf[0] = rawdata[j];
            buf[1] = rawdata[j+1];
            buf[2] = 0;
            unsigned int byte = strtoll(buf, NULL, 16);
            data[k] = byte;
        }
        RAISEIF(0 != attrset_set_attr(attrset, name, data, dtype, nmemb),
            ex_set_attr,
            NULL);
    } 
    fclose(fattr);
    free(data);
    free(buffer);
    attrset->dirty = 0;
    return 0;

ex_read_file:
ex_parse_attr:
ex_set_attr:
    attrset->dirty = 0;
    free(data);
ex_data:
    free(buffer);
ex_init:
    fclose(fattr);
    return -1;
}
static int
attrset_write_attr_set_v2(BigAttrSet * attrset, const char * basename)
{
    static char conv[] = "0123456789ABCDEF";
    attrset->dirty = 0;

    FILE * fattr = _big_file_open_a_file(basename, FILEID_ATTR_V2, "w", 1);
    RAISEIF(fattr == NULL,
            ex_open,
            NULL);

    ptrdiff_t i;
    for(i = 0; i < attrset->listused; i ++) {
        BigAttr * a = & attrset->attrlist[i];
        int itemsize = big_file_dtype_itemsize(a->dtype);
        int ldata = itemsize * a->nmemb;

        char * rawdata = malloc(ldata * 2 + 1);
        unsigned char * adata = (unsigned char*) a->data;
        int j, k; 
        for(j = 0, k = 0; k < ldata; k ++, j+=2) {
            rawdata[j] = conv[adata[k] / 16];
            rawdata[j + 1] = conv[adata[k] % 16];
        }
        rawdata[j] = 0;

        char * textual;
        /* skip textual representation for very long columns */
        if(ldata > 128) {
            textual = _strdup("... (Too Long) ");
        } else {
            textual = malloc(a->nmemb * 32 + 1);
            textual[0] = 0;
            for(j = 0; j < a->nmemb; j ++) {
                if(a->dtype[1] != 'a' &&
                  !(a->dtype[1] == 'S' && big_file_dtype_itemsize(a->dtype) == 1))
                {
                    char buf[128];
                    big_file_dtype_format(buf, a->dtype, &adata[j * itemsize], NULL);
                    strcat(textual, buf);
                    if(j != a->nmemb - 1) {
                        strcat(textual, " ");
                    }
                } else { /* pretty print string encoded as a1 or S1. */
                    char buf[] = {adata[j], 0};
                    if(buf[0] == '\n') {
                        strcat(textual, "...");
                        break;
                    } if(buf[0] == 0) {
                        break;
                    }
                    strcat(textual, buf);
                }
            }
        }
        int rt = fprintf(fattr, "%s %s %d %s #HUMANE [ %s ]\n", 
                a->name, a->dtype, a->nmemb, rawdata, textual
               );
        free(rawdata);
        free(textual);
        RAISEIF(rt <= 0,
            ex_write,
            "Failed to write to file");
    } 
    fclose(fattr);
    return 0;
ex_write:
    fclose(fattr);
ex_open:
    return -1;
}

static int attr_cmp(const void * p1, const void * p2) {
    const BigAttr * c1 = p1;
    const BigAttr * c2 = p2;
    return strcmp(c1->name, c2->name);
}

static BigAttr *
attrset_append_attr(BigAttrSet * attrset)
{
    while(attrset->listsize - attrset->listused < 1) {
        attrset->attrlist = realloc(attrset->attrlist, attrset->listsize * 2 * sizeof(BigAttr));
        attrset->listsize *= 2;
    }
    BigAttr * a = & (attrset->attrlist[attrset->listused++]);
    memset(a, 0, sizeof(BigAttr));
    return a;
}

static int
attrset_add_attr(BigAttrSet * attrset, const char * attrname, const char * dtype, int nmemb)
{
    size_t size = big_file_dtype_itemsize(dtype) * nmemb + strlen(attrname) + 1;
    while(attrset->bufsize - attrset->bufused < size) {
        int i;
        for(i = 0; i < attrset->listused; i ++) {
            attrset->attrlist[i].data -= (ptrdiff_t) attrset->attrbuf;
            attrset->attrlist[i].name -= (ptrdiff_t) attrset->attrbuf;
        }
        attrset->attrbuf = realloc(attrset->attrbuf, attrset->bufsize * 2);
        attrset->bufsize *= 2;
        for(i = 0; i < attrset->listused; i ++) {
            attrset->attrlist[i].data += (ptrdiff_t) attrset->attrbuf;
            attrset->attrlist[i].name += (ptrdiff_t) attrset->attrbuf;
        }
    }
    char * free = attrset->attrbuf + attrset->bufused;
    attrset->bufused += size;

    BigAttr * n = attrset_append_attr(attrset);

    n->nmemb = nmemb;
    memset(n->dtype, 0, 8);
    _dtype_normalize(n->dtype, dtype);

    n->name = free;
    strcpy(free, attrname);
    free += strlen(attrname) + 1;
    n->data = free;

    qsort(attrset->attrlist, attrset->listused, sizeof(BigAttr), attr_cmp);
    return 0;
}

static BigAttr *
attrset_lookup_attr(BigAttrSet * attrset, const char * attrname)
{
    BigAttr lookup = {0};
    lookup.name = (char*) attrname;
    BigAttr * found = bsearch(&lookup, attrset->attrlist, attrset->listused, sizeof(BigAttr), attr_cmp);
    return found;
}

static int
attrset_remove_attr(BigAttrSet * attrset, const char * attrname)
{
    BigAttr *attr = attrset_lookup_attr(attrset, attrname);
    RAISEIF(attr == NULL,
      ex_notfound,
      "Attribute name '%s' is not found.", attrname
    )
    ptrdiff_t ind = attr - attrset->attrlist;
    memmove(&attrset->attrlist[ind], &attrset->attrlist[ind + 1],
        (attrset->listused - ind - 1) * sizeof(BigAttr));
    attrset->listused -= 1;

    return 0;

ex_notfound:
    return -1;
}

static BigAttr *
attrset_list_attrs(BigAttrSet * attrset, size_t * count)
{
    *count = attrset->listused;
    return attrset->attrlist;
}

static int
attrset_set_attr(BigAttrSet * attrset, const char * attrname, const void * data, const char * dtype, int nmemb)
{
    BigAttr * attr;
    attrset->dirty = 1;

    RAISEIF (
         strchr(attrname, ' ')
      || strchr(attrname, '\t')
      || strchr(attrname, '\n'),
      ex_name,
      "Attribute name cannot contain blanks (space, tab or newline)"
    );

    /* Remove it if it exists*/
    attr = attrset_lookup_attr(attrset, attrname);
    if(attr)
        attrset_remove_attr(attrset, attrname);
    /* add ensures the dtype has been normalized! */
    RAISEIF(0 != attrset_add_attr(attrset, attrname, dtype, nmemb),
            ex_add,
            "Failed to add attr");
    attr = attrset_lookup_attr(attrset, attrname);
    RAISEIF(attr->nmemb != nmemb,
            ex_mismatch,
            "attr nmemb mismatch");
    return dtype_convert_simple(attr->data, attr->dtype, data, dtype, attr->nmemb);

ex_name:
ex_mismatch:
ex_add:
    return -1;
}

static int
attrset_get_attr(BigAttrSet * attrset, const char * attrname, void * data, const char * dtype, int nmemb)
{
    BigAttr * found = attrset_lookup_attr(attrset, attrname);
    RAISEIF(!found, ex_notfound, "attr not found");
    RAISEIF(found->nmemb != nmemb, ex_mismatch, "attr nmemb mismatch");
    return dtype_convert_simple(data, dtype, found->data, found->dtype, found->nmemb);

ex_mismatch:
ex_notfound:
    return -1;
}

static BigAttrSet *
attrset_create(void)
{
    BigAttrSet * attrset = calloc(1, sizeof(BigAttrSet));
    attrset->attrbuf = malloc(128);
    attrset->bufsize = 128;
    attrset->bufused = 0;
    attrset->attrlist = malloc(sizeof(BigAttr) * 16);
    attrset->listsize = 16;
    attrset->listused = 0;

    return attrset;
}

static void
attrset_free(BigAttrSet * attrset)
{
    free(attrset->attrbuf);
    free(attrset->attrlist);
    free(attrset);
}

void big_attrset_set_dirty(BigAttrSet * attrset, int dirty)
{
    attrset->dirty = dirty;
}

static void *
_big_attrset_pack(BigAttrSet * attrset, size_t * bytes)
{
    size_t n = 0;
    n += sizeof(BigAttrSet);
    n += attrset->bufused;
    n += sizeof(BigAttr) * attrset->listused;
    char * buf = calloc(n, 1);
    char * p = buf;
    char * attrbuf = (char*) (p + sizeof(BigAttrSet));
    BigAttr * attrlist = (BigAttr *) (attrbuf + attrset->bufused);
    memcpy(p, attrset, sizeof(BigAttrSet));
    memcpy(attrbuf, attrset->attrbuf, attrset->bufused);
    memcpy(attrlist, attrset->attrlist, attrset->listused * sizeof(BigAttr));
    int i = 0;
    for(i = 0; i < attrset->listused; i ++) {
        attrlist[i].data -= (ptrdiff_t) attrset->attrbuf;
        attrlist[i].name -= (ptrdiff_t) attrset->attrbuf;
    }

    *bytes = n;

    return (void*) p;
}

static BigAttrSet *
_big_attrset_unpack(void * p)
{
    BigAttrSet * attrset = calloc(1, sizeof(attrset[0]));
    memcpy(attrset, p, sizeof(BigAttrSet));
    p += sizeof(BigAttrSet);
    attrset->attrbuf = malloc(attrset->bufsize);
    attrset->attrlist = malloc(attrset->listsize * sizeof(BigAttr));
    memcpy(attrset->attrbuf, p, attrset->bufused);
    p += attrset->bufused;
    memcpy(attrset->attrlist, p, attrset->listused * sizeof(BigAttr));
    int i = 0;
    for(i = 0; i < attrset->listused; i ++) {
        attrset->attrlist[i].data += (ptrdiff_t) attrset->attrbuf;
        attrset->attrlist[i].name += (ptrdiff_t) attrset->attrbuf;
    }
    return attrset;
}

void *
_big_block_pack(BigBlock * block, size_t * bytes)
{
    size_t attrsize = 0;
    void * attrset = _big_attrset_pack(block->attrset, &attrsize);
    int Nfile = block->Nfile;

    * bytes =   sizeof(block[0])
              + strlen(block->basename) + 1
              + (Nfile + 1) * sizeof(block->fsize[0])
              + (Nfile + 1) * sizeof(block->foffset[0])
              + (Nfile + 1) * sizeof(block->fchecksum[0])
              + attrsize;

    void * buf = malloc(*bytes);

    char * ptr = (char *) buf;


    memcpy(ptr, block, sizeof(block[0]));
    ptr += sizeof(block[0]);
    memcpy(ptr, block->basename, strlen(block->basename) + 1);
    ptr += strlen(block->basename) + 1;
    if(block->fsize)
        memcpy(ptr, block->fsize, (Nfile + 1) * sizeof(block->fsize[0]));
    ptr += (Nfile + 1) * sizeof(block->fsize[0]);
    if(block->foffset)
        memcpy(ptr, block->foffset, (Nfile + 1) * sizeof(block->foffset[0]));
    ptr += (Nfile + 1) * sizeof(block->foffset[0]);
    if(block->fchecksum)
        memcpy(ptr, block->fchecksum, (Nfile + 1) * sizeof(block->fchecksum[0]));
    ptr += (Nfile + 1) * sizeof(block->fchecksum[0]);
    memcpy(ptr, attrset, attrsize);
    free(attrset);
    ptr += attrsize;

    return buf;
}

void
_big_block_unpack(BigBlock * block, void * buf)
{
    char * ptr = (char*)buf;

    memcpy(block, ptr, sizeof(block[0]));
    ptr += sizeof(block[0]);
    int Nfile = block->Nfile;

    block->fsize = calloc(Nfile + 1, sizeof(size_t));
    block->foffset = calloc(Nfile + 1, sizeof(size_t));
    block->fchecksum = calloc(Nfile + 1, sizeof(int));

    block->basename = _strdup(ptr);
    ptr += strlen(ptr) + 1;
    if(block->fsize)
        memcpy(block->fsize, ptr, (Nfile + 1) * sizeof(block->fsize[0]));
    ptr += (Nfile + 1) * sizeof(block->fsize[0]);
    if(block->foffset)
        memcpy(block->foffset, ptr, (Nfile + 1) * sizeof(block->foffset[0]));
    ptr += (Nfile + 1) * sizeof(block->foffset[0]);
    if(block->fchecksum)
        memcpy(block->fchecksum, ptr, (Nfile + 1) * sizeof(block->fchecksum[0]));
    ptr += (Nfile + 1) * sizeof(block->fchecksum[0]);
    block->attrset = _big_attrset_unpack(ptr);
}


/* File Path */

FILE *
_big_file_open_a_file(const char * basename, int fileid, char * mode, int raise)
{
    char * filename;
    int unbuffered = 0;
    if(fileid == FILEID_HEADER) {
        filename = _path_join(basename, EXT_HEADER);
    } else
    if(fileid == FILEID_ATTR) {
        filename = _path_join(basename, EXT_ATTR);
    } else
    if(fileid == FILEID_ATTR_V2) {
        filename = _path_join(basename, EXT_ATTR_V2);
    } else {
        char d[128];
        sprintf(d, EXT_DATA, fileid);
        filename = _path_join(basename, d);
        unbuffered = 1;
    }
    FILE * fp = fopen(filename, mode);

    if(!raise && fp == NULL) {
        goto ex_fopen;
    }

    RAISEIF(fp == NULL,
        ex_fopen,
        "Failed to open physical file `%s' with mode `%s' (%s)",
        filename, mode, strerror(errno));

    if(unbuffered) {
        setbuf(fp, NULL);
    }
ex_fopen:
    free(filename);
    return fp;
}
static
int _big_file_mkdir(const char * dirname)
{
    struct stat buf;
    int mkdirret;

    mkdirret = mkdir(dirname, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    /* stat is to make sure the dir doesn't exist; it could be owned by others and stopped
     * by a permission error, but we are still OK, as the dir is created*/

    /* already exists; only check stat after mkdir fails with EACCES, avoid meta data calls. */

    RAISEIF((mkdirret !=0 && errno != EEXIST && stat(dirname, &buf)),
        ex_mkdir,
        "Failed to create directory structure at `%s' (%s)",
        dirname,
        strerror(errno)
    );

    /* Attempt to update the time stamp */
    utimes(dirname, NULL);

    return 0;
ex_mkdir:
    return -1;
}

/* make subdir rel to pathname, recursively making parents */
int
_big_file_mksubdir_r(const char * pathname, const char * subdir)
{
    char * subdirname = _strdup(subdir);
    char * p;

    char * mydirname;
    for(p = subdirname; *p; p ++) {
        if(*p != '/') continue;
        *p = 0;
        mydirname = _path_join(pathname, subdirname);
        if(strlen(mydirname) != 0) {
            RAISEIF(0 != _big_file_mkdir(mydirname),
                ex_mkdir, NULL);
        }
        free(mydirname);
        *p = '/';
    }
    mydirname = _path_join(pathname, subdirname);
    RAISEIF(0 != _big_file_mkdir(mydirname),
        ex_mkdir, NULL);

    free(subdirname);
    free(mydirname);
    return 0;
ex_mkdir:
    free(subdirname);
    free(mydirname);
    return -1;
}

