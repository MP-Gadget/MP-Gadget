#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <gsl/gsl_rng.h>
#include <mpi.h>

#include "allvars.h"

#include "cic.h"
#include "icfile.h"

void  read_parameterfile(char *fname);
void   set_units(void);

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
    MPI_Comm_size(MPI_COMM_WORLD, &NTask);

    if(argc < 3)
    {
        if(ThisTask == 0)
        {
            fprintf(stdout, "\nParameters are missing.\n");
            fprintf(stdout, "Call with <ParameterFile> <CICNgrid>\n\n");
        }
        exit(0);
    }

    /* fixme: make this a mpi bcast */
    read_parameterfile(argv[1]);

    int Ngrid = atoi(argv[2]);

    set_units();

    /* read header */
    char buf[4096];
    sprintf(buf, "%s/%s.%d", OutputDir, FileBase, 0);
    struct io_header header;
    if(ThisTask == 0) {
        if(read_header(&header, buf) != 0) {
            sprintf(buf, "%s/%s", OutputDir, FileBase);
            if(read_header(&header, buf) != 0) {
                abort();
            }
            if(header.num_files != 1) {
                abort();
            }
        }
    }
    MPI_Bcast(&header, sizeof(header), MPI_BYTE, 0, MPI_COMM_WORLD);

    CIC cic;
    CIC cic2;
    cic_init(&cic, Ngrid, header.BoxSize);
    cic_init(&cic2, Ngrid, header.BoxSize);

    /* watch out eval of start end may overflow */
    size_t Nfile = header.num_files;

    int start = Nfile * ThisTask / NTask;
    int end = Nfile * (ThisTask + 1) / NTask;
    int i;

    /* only do my process */
    for(i = start; i < end; i ++) {
        ICFile file;
        char buf[4096];
        if(Nfile == 1) {
            sprintf(buf, "%s/%s", OutputDir, FileBase);
        } else{
            sprintf(buf, "%s/%s.%d", OutputDir, FileBase, i);
        }
        icfile_read(&file, buf);
        int k;
#pragma omp parallel for
        for(k = 0; k < file.NumPart; k ++) {
            /* watch out add_particle is thread safe */
            cic_add_particle(&cic, file.P[k].Pos, file.P[k].Mass);
        }
        icfile_destroy(&file);
    }    

    /* now lets gather the CIC buffers to 0 */
    MPI_Barrier(MPI_COMM_WORLD);
    if(ThisTask == 0) {
        printf("Done reading\n");
    }

    
    {
        ptrdiff_t i = 0;
        int chunksize = 1024 * 1024;
        for(i = 0; i < cic2.size; i += chunksize) {
            int size = (i + chunksize <= cic2.size)?
                    chunksize: (cic2.size - i);
            MPI_Reduce(cic.buffer + i,
                    cic2.buffer + i,
                    size, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        }
    }
    if(ThisTask == 0) {
        ptrdiff_t p;
        double sum = 0.0;
#pragma omp parallel for reduction(+: sum)
        for(p = 0; p < cic2.size; p ++) {
            sum += cic2.buffer[p];
        }

        printf("Gas mass = %g. DM mass = %g. \n", header.mass[0], header.mass[1]);
        printf("Total mass = %g. sum of mass = %g. \n", 
                header.mass[0] * (header.npartTotal[0] + (((size_t) header.npartTotalHighWord[0]) << 32))
              + header.mass[1] * (header.npartTotal[1] + (((size_t) header.npartTotalHighWord[1]) << 32)),
                sum
              );
        sprintf(buf, "%s/%s-%d.raw", OutputDir, "cic", Ngrid);
        FILE * fd = fopen(buf, "w");
        fwrite(&cic2, sizeof(cic2), 1, fd);
        fwrite(cic2.buffer, sizeof(double), cic2.size, fd);
        fclose(fd);
        printf("file format of %s:\n", buf);
        printf("%d bytes of header (data follows in double) \n", (int) sizeof(cic2));
        printf("BoxSize (double) at: %d\n", (int) ((char*) &cic2.BoxSize - (char*) &cic2));
        printf("Ngrid (int) at: %d\n", (int) ((char*) &cic2.Ngrid - (char*) &cic2));
    }
    MPI_Finalize();
}

