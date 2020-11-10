#ifndef _BIGFILE_MPI_H_
#define _BIGFILE_MPI_H_
#include <mpi.h>

#include "bigfile.h"

#ifdef __cplusplus
extern "C" {
#endif
/** Open a Bigfile: this stats the directory tree that constitutes the BigFile format.
 * It initialises the BigFile structure.
 * Arguments:
 * @param BigFile bf - pointer to uninitialised structure.
 * @param const char * basename - String containing directory to put snapshot in.
 * @param MPI_Comm comm - MPI communicator. Does nothing except make sure all tasks exit this function together.
 * @returns 0 if successful. */
int big_file_mpi_open(BigFile * bf, const char * basename, MPI_Comm comm);

/** Create a Bigfile: this creates the directory tree that constitutes the BigFile format.
 * It initialises the BigFile structure.
 * Arguments:
 * @param BigFile bf - pointer to uninitialised structure.
 * @param const char * basename - String containing directory to put snapshot in.
 * @param MPI_Comm comm - MPI communicator. Does nothing except make sure all tasks exit this function together.
 * @returns 0 if successful. */
int big_file_mpi_create(BigFile * bf, const char * basename, MPI_Comm comm);

/** Open a BigBlock:
 * A BigBlock stores a two dimesional table of nmemb columns and size rows. Numerical typed columns are supported.
 * Arguments:
 * @param BigFile bf - pointer to opened BigFile structure.
 * @param BigBlock block - pointer to initialised BigBlock to open.
 * @param const char * basename - Name of the block to open at. eg. "header". Must already exist.
 * @param MPI_Comm comm - MPI communicator to use.
 * @returns 0 if successful, -1 if could not open block. */
int big_file_mpi_open_block(BigFile * bf, BigBlock * block, const char * blockname, MPI_Comm comm);

/** Create a BigBlock:
 * A BigBlock stores a two dimesional table of nmemb columns and size rows. Numerical typed columns are supported.
 * Arguments:
 * @param BigFile bf - pointer to opened BigFile structure.
 * @param BigBlock block - pointer to uninitialised BigBlock.
 * @param const char * basename - Name of the block to initialise at. eg. "header": what happens if this structure exists already?
 * @param dtype - string denoting the type of the block. This has a normalised form, like "i8" for a 64-bit integer.
 * See documentation for dtype_parse. Can be NULL if only attributes will be stored.
 * @param - nmemb - Number of columns that will be stored in this block. Can be zero.
 * @param Nfile - Number of files to use for this block on disc. This is an implementation detail;
 * you will never need it to read the BigFile.
 * @param - size Number of rows of type dtype that will be stored in this block. Can be zero.
 * @param MPI_Comm comm - MPI communicator to use.
 * @returns 0 if successful. */
int big_file_mpi_create_block(BigFile * bf,
        BigBlock * block,
        const char * blockname,
        const char * dtype,
        int nmemb,
        int Nfile,
        size_t size,
        MPI_Comm comm);

int
_big_file_mpi_create_block(BigFile * bf,
        BigBlock * block,
        const char * blockname,
        const char * dtype,
        int nmemb,
        int Nfile,
        const size_t fsize[],
        MPI_Comm comm);

/** Close the BigFile, and free memory associated with it. Once closed, it should not be re-used.*/
int big_file_mpi_close(BigFile * bf, MPI_Comm comm);

/** Close the BigBlock, and free memory associated with it. Once closed, it should not be re-used.*/
int big_block_mpi_close(BigBlock * block, MPI_Comm comm);

/** Grow a BigBlock:
 * Increase the size of a BigBlock. No flushing is implied.
 * Arguments:
 * @param BigBlock block - pointer to initialised BigBlock to open.
 * @param Nfile_grow - Number of files to use for this block on disc. This is an implementation detail;
 * you will never need it to read the BigFile.
 * @param - size_grow Number of rows of type dtype that will be appended in this block.
 * @param MPI_Comm comm - MPI communicator to use.
 * @returns 0 if successful, -1 if could not open block. */
int big_block_mpi_grow(BigBlock * bb, int Nfile_grow, const size_t fsize_grow[], MPI_Comm comm);

int big_block_mpi_grow_simple(BigBlock * block, int Nfile_grow, size_t fsize_grow, MPI_Comm comm);


/** Set the threshold that enables the aggregated IO.
 *
 *  If the total size of data per concurrent writer group is less than the threshold,
 *  the data is aggregated to the leader rank of the writer group for writing, to reduce
 *  the total number of IO requests issued to the file server.
 *
 * */
void big_file_mpi_set_aggregated_threshold(size_t bytes);
size_t big_file_mpi_get_aggregated_threshold();

void big_file_mpi_set_verbose(int verbose);

/** Write data stored in a BigArray to a BigBlock.
 * You cannot write beyond the end of the size of the block.
 * The value may be a (small) array.
 *
 * This is a collective MPI operation. The write operation starts from ptr.
 *
 * Arguments:
 * @param block - pointer to opened BigBlock
 * @param ptr - Absolute position to write to in the file. Construct this with a call to big_block_seek.
 * @param array - BigArray containing the data which should be written.
 * @param concurrency - Max number of MPI ranks that issues write operation at the same time.
 * @param comm - MPI Communicator
 * @returns 0 if successful. */
int big_block_mpi_write(BigBlock * bb, BigBlockPtr * ptr, BigArray * array, int concurrency, MPI_Comm comm);

/** Read from a block to a BigArray
 *
 * This is a collective MPI operation. The read operation will start from ptr.
 *
 * @param ptr - The offset to start reading
 * @param array - An array specifying the number of items to read.
 * @param concurrency - Max number of MPI ranks that issues write operation at the same time.
 * @param comm - MPI Communicator
 *
 * @returns 0 if successful.
 */
int big_block_mpi_read(BigBlock * bb, BigBlockPtr * ptr, BigArray * array, int concurrency, MPI_Comm comm);

/** Flush the BigBlock 
 *
 *  Flush will write the attrset from root rank, and gather the checksums from all ranks.
 *
 * */
int big_block_mpi_flush(BigBlock * block, MPI_Comm comm);

int
big_file_mpi_create_records(BigFile * bf,
    const BigRecordType * rtype,
    const char * mode,
    int Nfile,
    const size_t fsize[],
    MPI_Comm comm);

int
big_file_mpi_write_records(BigFile * bf,
    const BigRecordType * rtype,
    ptrdiff_t offset,
    size_t size,
    const void * buf,
    int concurrency,
    MPI_Comm comm);

int
big_file_mpi_read_records(BigFile * bf,
    const BigRecordType * rtype,
    ptrdiff_t offset,
    size_t size,
    void * buf,
    int concurrency,
    MPI_Comm comm);


#ifdef __cplusplus
}
#endif
#endif
