#include <mpi.h>
#include "bigfile.h"

int big_file_mpi_open(BigFile * bf, char * basename, MPI_Comm comm);
int big_file_mpi_create(BigFile * bf, char * basename, MPI_Comm comm);
int big_file_mpi_open_block(BigFile * bf, BigBlock * block, char * blockname, MPI_Comm comm);
int big_file_mpi_create_block(BigFile * bf, BigBlock * block, char * blockname, char * dtype, int nmemb, int Nfile, size_t size, MPI_Comm comm);
void big_file_mpi_close(BigFile * bf, MPI_Comm comm);

int big_block_mpi_create(BigBlock * bb, char * basename, char * dtype, int nmemb, int Nfile, size_t fsize[], MPI_Comm comm);
int big_block_mpi_close(BigBlock * block, MPI_Comm comm);
int big_block_mpi_open(BigBlock * bb, char * basename, MPI_Comm comm);
