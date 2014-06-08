#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stddef.h>

#include "bigblock-mpi.h"

#if 0
int NTask;
int ThisTask;
int main(int argc, char * argv[]) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
    MPI_Comm_size(MPI_COMM_WORLD, &NTask);

    BigBlock bb = {0};
    BigBlockPtr ptr = {0};
    BigArray ba = {0};
    size_t Nfile = 4;
    size_t Nitem = 99;
    size_t * fsize = alloca(Nfile * sizeof(size_t));
    int i;

    for(i = 0; i < Nfile; i++) {
        fsize[i] = (i + 1)* Nitem / Nfile - i * Nitem / Nfile;
    }

    size_t mysize = (ThisTask + 1) * Nitem / NTask - ThisTask * Nitem / NTask;
    ptrdiff_t mystart = ThisTask * Nitem / NTask;
    int * buffer = malloc(mysize * sizeof(int));

    big_array_init(&ba, buffer, "i4", 1, &mysize, NULL);

    for(i = 0; i < mysize; i ++) {
        buffer[i] = (i + mystart) * 100 + ThisTask;
    }
    big_block_mpi_create(&bb, "testblock", "i4", Nfile, fsize, MPI_COMM_WORLD);
    big_block_set_attr(&bb, "Nitem", &Nitem, "i8", 1);

    big_block_seek(&bb, &ptr, mystart); 
    big_block_write(&bb, &ptr, &ba);

    for(i = 0; i < mysize; i ++) {
        printf("dump %td %d %d\n", i + mystart, buffer[i], ThisTask);
    }

    big_block_mpi_close(&bb, MPI_COMM_WORLD);

    big_block_mpi_open(&bb, "testblock", MPI_COMM_WORLD);
    big_block_get_attr(&bb, "Nitem", &Nitem, "i8", 1);
    printf("attr %d Nitem=%td\n", ThisTask, Nitem);
    MPI_Finalize();
}
#endif
