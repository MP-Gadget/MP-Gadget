#ifndef _BIGFILE_H_
#define _BIGFILE_H_
#include <stddef.h>
#include <stdio.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct BigFile {
    /* All members are readonly. Initialize with big_file_open / big_file_create */
    char * basename;
} BigFile;

typedef struct BigAttr BigAttr;
struct BigAttr {
    /* All members are readonly. */
    int nmemb;
    char dtype[8];
    char * name;
    char * data;
};

typedef struct BigAttrSet BigAttrSet;

typedef struct BigBlock {
    /* All members are readonly */
    char dtype[8]; /* numpy style
                        dtype[0] <: little endian, > big endian, = machine endian
                        dtype[1] type char
                        dtype[2:] width in bytes
                    */
    int nmemb;  /* num of dtype typed elements per item */
    char * basename;
    size_t size;
    size_t * fsize; /* Nfile + 1, in units of elements */
    size_t * foffset; /* Nfile + 1, in units of elements */
    unsigned int * fchecksum; /* sysv sums of each file (unreduced) */
    int Nfile;
    BigAttrSet * attrset;
    int dirty;
} BigBlock;

typedef struct BigBlockPtr BigBlockPtr;

struct BigBlockPtr {
    /* All members are readonly */
    int fileid;
    ptrdiff_t roffset; /* offset within a file */
    ptrdiff_t aoffset; /* abs offset */
};

typedef struct BigArray {
    /* All members are readonly */
    int ndim;
    char dtype[8];
    ptrdiff_t dims[32];
    ptrdiff_t strides[32];
    size_t size;
    void * data;
} BigArray;

typedef struct BigArrayIter {
    /* All members are readonly, internally used by bigfile */
    ptrdiff_t pos[32];
    BigArray * array;
    int contiguous;
    void * dataptr;
} BigArrayIter;

int big_file_set_buffer_size(size_t bytes);
char * big_file_get_error_message(void);
void big_file_set_error_message(char * msg);

/** Open a Bigfile: this stats the directory tree, but does not open the file.
 * It initialises the BigFile structure.
 * Arguments:
 * @param BigFile bf - pointer to uninitialised structure.
 * @param const char * basename - String containing directory to put snapshot in.*/
int big_file_open(BigFile * bf, const char * basename); /* raises */

/** Create a Bigfile: this makes the directory tree and initialises the BigFile structure.
 * Arguments:
 * @param BigFile bf - pointer to uninitialised structure.
 * @param const char * basename - String containing directory to put snapshot in.*/
int big_file_create(BigFile * bf, const char * basename); /* raises */
int big_file_list(BigFile * bf, char *** blocknames, int * Nblocks);
int big_file_open_block(BigFile * bf, BigBlock * block, const char * blockname); /* raises*/
int big_file_create_block(BigFile * bf, BigBlock * block, const char * blockname, const char * dtype, int nmemb, int Nfile, const size_t fsize[]); /* raises */
int big_file_close(BigFile * bf); /* raises */

int big_block_close(BigBlock * block); /* raises */
int big_block_grow(BigBlock * bb, int Nfile_grow, const size_t fsize_grow[]); /* raises */
int big_block_flush(BigBlock * block); /* raises */

void big_block_set_dirty(BigBlock * block, int value);
void big_attrset_set_dirty(BigAttrSet * attrset, int value);

/** Initialise BigBlockPtr to the place in the BigBlock offset elements from the beginning of the block.
 * This allows you to write into the BigBlock at a position other than the beginning.
 * @param offset - Position to seek to in units of the size of the array element, eg, 8 bytes for an i8. 
 *                 If offset < 0, seek to size + offset.
 *
 */
int big_block_seek(BigBlock * bb, BigBlockPtr * ptr, ptrdiff_t offset); /* raises */
int big_block_seek_rel(BigBlock * bb, BigBlockPtr * ptr, ptrdiff_t rel); /* raises */

/** Detect for end of file 
 *
 *  Returns non-zero is the ptr is at EOF 
 * */
int big_block_eof(BigBlock * bb, BigBlockPtr * ptr);

/** Read from a block to a BigArray
 *
 * @param ptr - The offset to start reading
 * @param array - An array specifying the number of items to read.
 *
 */
int big_block_read(BigBlock * bb, BigBlockPtr * ptr, BigArray * array); /* raises */

/** Read from a block and create a BigArray 
 *  array->buf shall be freed with the C free() function.
 * 
 *  @param size - Read `size' rows. This will not fail if start + size < bb->size.
 *  @param start - Read from `start'
 *  @param array - an empty BigArray struct, that will be initialized by this function.
 *                 free(array->buf) after use.
 *  @param dtype - If not NULL, cast to the given dtype in the output array.
 *
 *  If there is less than 'size' rows in the file, the true
 *  number of items read will be in array->dims[0].
 *
 * */
int big_block_read_simple(BigBlock * bb, ptrdiff_t start, ptrdiff_t size, BigArray * array, const char * dtype); /* raises */

/** Write data stored in a BigArray to a BigBlock.
 * You cannot write beyond the end of the size of the block.
 * The value may be a (small) array.
 * Arguments:
 * @param block - pointer to opened BigBlock
 * @param ptr - Absolute position to write to in the file. Construct this with a call to big_block_seek.
 * @param array - BigArray containing the data which should be written.
 * @returns 0 if successful. */
int big_block_write(BigBlock * bb, BigBlockPtr * ptr, BigArray * array); /* raisees*/

/** Set an attribute on a BigBlock: attributes are plaintext key-value pairs stored in a special file in the Block directory.
 * The value may be a (small) array.
 * Arguments:
 * @param block - pointer to opened BigBlock
 * @param attrname - name of the attribute to store.
 * @param data - pointer to data to store
 * @param dtype - Type of data array in the format used by dtype.
 * @param nmemb - Number of members in the data array.
 * @returns 0 if successful. */
int big_block_set_attr(BigBlock * block, const char * attrname, const void * data, const char * dtype, int nmemb); /* raises */

/** Get an attribute on a BigBlock: attributes are plaintext key-value pairs stored in a special file in the Block directory.
 * Attribute value is stored in the memory pointed to by data, so make sure it is big enough!
 * Arguments:
 * @param block - pointer to opened BigBlock
 * @param attrname - name of the attribute to store.
 * @param data - pointer to data in which to place attribute
 * @param dtype - Type of data array in the format used by dtype.
 * @param nmemb - Number of members to get. Must be equal to number of members
 * originally stored or an error is raised and -1 returned.*/
int big_block_get_attr(BigBlock * block, const char * attrname, void * data, const char * dtype, int nmemb); /* raises */

/* remove an attribute */
int big_block_remove_attr(BigBlock * block, const char * attrname); /* raises */

BigAttr * big_block_lookup_attr(BigBlock * block, const char * attrname);
BigAttr * big_block_list_attrs(BigBlock * block, size_t * count);

/**
 *
 * dtype: a subset of numpy's dtype descriptor.
 *
 * dtype[0]: endianness, '<' for LE and '>' BE. '=' is native and will be converted to LE or BE during IO.
 * dtype[1]: kind in char: 
 *
 *    'i' :int, 
 *    'f'  float, 
 *    'c'  complex, 
 *    'u'  unsigned int
 *    'b'  boolean / byte
 *    'a'  string bytes
 *
 *    Other kinds are bypassed; the python API will explain them as numpy dtypes.
 *
 * dtype[2:]: is the byte-width, 4 and 8 are supported.
 *
 * dtype[0] can be omitted, in which case native is prepended to form a 'normalized' dtype.
 *
 */

int big_file_dtype_itemsize(const char * dtype);
int big_file_dtype_kind(const char * dtype);
void big_file_dtype_format(char * buffer, const char * dtype, const void * data, const char * flags);
/* Parse a string into a memory location according to dtype; returns non-zero on error */
int big_file_dtype_parse(const char * buffer, const char * dtype, void * data, const char * fmt);

#define dtype_itemsize big_file_dtype_itemsize
#define dtype_format big_file_dtype_format
#define dtype_parse big_file_dtype_parse

/** Create a BigArray from raw memory.
 *
 * A BigArray is a checked, multidimensional, array format. It is used to provide metadata
 * about a raw void pointer to big_block_write. It is an implementation detail of this library, not stored on disc.
 *
 * This routine constructs the BigArray.
 *
 * Data is not copied, so the memory must not be freed before the BigArray is deallocated.
 * @params array - uninitialised BigArray.
 * @params buf - pointer to raw memory to be stored in the BigArray.
 * @params dtype - type of the data array as a string.
 * @params ndim - Number of dimensions to use in the BigArray.
 * Note that a BigArray can have up to 32 dimensions, but only 2 of them can be stored in a BigFile.
 * @params dims - list of integers containing the size of each dimension of the BigArray
 * @params strides - Integers containing the strides of each dimension of the BigArray in bytes, for iterating the array with BigArrayIter.
 *                   NULL for C ordering array.
 *
 * This is a BigArray implementation of numpy strides, so see documentation here: http://www.scipy-lectures.org/advanced/advanced_numpy/#indexing-scheme-strides
 *
 * For a multidimensional array with C ordering and N dimensions,
 * strides[i] is the length of the array in the ith dimension.
 * strides[N-1] is the size of a single element.
 *
 * The logic view of a BigBlock is a two dimensional array, with dims = {size, nmemb}, and strides = {nmemb * itemsize, itemsize}.
 *
 * The array iteration increases the last dimension first. Therefore, the most straight-forward way of creating a BigArray for big_block_write
 * and big_block_read is dims = {chunksize, nmemb}, strides = NULL.
 *
 * The strides representation is quite flexible: the array can be ordered in other ways. 
 * For example, with:
 *
 * struct {  double pos[3];  double vel[3]; } * P;
 *
 * You can use strides[1] = sizeof(double), strides[0] = sizeof(struct P).
 * Then set buf to &P[0].pos[0] to dump position or &P[0].vel[0] to dump velocity.
 */

/* FIXME(rainwoodman): decouple buf from BigArray -- make BigArray a descriptor. */
int big_array_init(BigArray * array,
    void * buf,
    const char * dtype,
    int ndim,
    const size_t dims[],
    const ptrdiff_t strides[]);

int big_array_iter_init(BigArrayIter * iter, BigArray * array);
void big_array_iter_advance(BigArrayIter * iter);

/**
 * Record is a composite of fields. Currently each field must be a scalar dtype.
 *
 * Use the clear/complete session to build a record type.
 * */
typedef struct BigRecordField {
    char * name;
    char dtype[8];
    int nmemb;
    int offset;
    int elsize;
} BigRecordField;

typedef struct BigRecordType {
    BigRecordField * fields;
    size_t nfield;
    size_t itemsize;
} BigRecordType;

void big_record_type_clear(BigRecordType * rtype);
int big_record_type_add(BigRecordType * rtype,
    const char * name,
    const char * dtype,
    int nmemb);
void big_record_type_set(BigRecordType * rtype,
    int i,
    const char * name,
    const char * dtype,
    int nmemb);
void big_record_type_complete(BigRecordType * rtype);

/*
 * Get/Set the value of i-th row and c-th field from a RecordType buffer.
 * */
void big_record_set(const BigRecordType * rtype, void * buf, ptrdiff_t i, int c, const void * data);
void big_record_get(const BigRecordType * rtype, const void * buf, ptrdiff_t i, int c, void * data);

/* View i-th field in a BigRecordType buffer as a BigArray. */
int
big_record_view_field(const BigRecordType * rtype,
    int i,
    BigArray * array,
    size_t size,
    void * buf
);

int
big_file_write_records(BigFile * bf,
    const BigRecordType * rtype,
    ptrdiff_t offset,
    size_t size,
    const void * buf);

/* create or append empty records.
 * mode == "a+": append
 * mode == "w+": create
 * */
int
big_file_create_records(BigFile * bf,
    const BigRecordType * rtype,
    const char * mode,
    int Nfile,
    const size_t fsize[]);

int
big_file_read_records(BigFile * bf,
    const BigRecordType * rtype,
    ptrdiff_t offset,
    size_t size,
    void * buf);

#ifdef __cplusplus
}
#endif
#endif
