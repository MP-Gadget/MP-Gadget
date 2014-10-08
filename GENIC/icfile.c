#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "allvars.h"
#include "icfile.h"

static int read_f77(FILE * fd, void * buffer, size_t elsize, size_t N) {
    int dummy = 0, dummy2 = 0;
    fread(&dummy, sizeof(int), 1, fd);
    if(dummy != elsize * N) {
        return -1;
    }
    fread(buffer, elsize, N, fd);
    fread(&dummy2, sizeof(int), 1, fd);
    if(dummy2 != elsize * N) {
        return -1;
    }
    return N;
}
int read_header(struct io_header * header, char * filename) {
    FILE * fd;
    if(!(fd = fopen(filename, "r"))) {
        return -1;
    }

    if(read_f77(fd, header, sizeof(*header), 1) < 0) {
        return -1;
    }
    fclose(fd);
    return 0;
}
int icfile_read(ICFile * icfile, char * filename)
{
  int j, k, n, m, slab, count, type;
  unsigned int dummy, dummy2;
  FILE *fd = 0;

    printf("reading IC file...%s \n", filename);
    fflush(stdout);

    if(!(fd = fopen(filename, "r"))) {
        abort();
    }

    if(read_f77(fd, &icfile->header, sizeof(icfile->header), 1) < 0) {
        abort();
    }

    icfile->NumPart = 0;

	for(k = 0; k < 6; k++)
	    icfile->NumPart += icfile->header.npart[k];

    printf("reading '%s' with %d particles\n", filename, icfile->NumPart);

    icfile->P = malloc(sizeof(ParType) * icfile->NumPart);
    float (* pos)[3] = (float (*) [3]) malloc(sizeof(float) * icfile->NumPart * 3);
    if( read_f77(fd, pos, sizeof(float), 3 * icfile->NumPart) < 0) {

        abort();
    }
            
    int i;
    for(i = 0; i < icfile->NumPart; i ++) {
        for(k = 0; k < 3; k ++) {
            icfile->P[i].Pos[k] = pos[i][k];
        } 
        if(i < icfile->header.npart[0]) {
            icfile->P[i].Mass = icfile->header.mass[0];
        } else {
            icfile->P[i].Mass = icfile->header.mass[1];
        }
    }
    free(pos);
    icfile->filename = strdup(filename);
    fclose(fd);
    return 0;
}

void icfile_destroy(ICFile * icfile) {
    free(icfile->P);
}
