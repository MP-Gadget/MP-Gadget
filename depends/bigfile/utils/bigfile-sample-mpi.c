#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stddef.h>
#include <unistd.h>
#include <math.h>
#include "bigfile-mpi.h"
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

void usage() {
    fprintf(stderr, "usage: bigfile-sample-mpi [-r ratio] [-N Nfile] [-f newfilepath] filepath block newblock\n");
    exit(1);

}
#define DONE_TAG 1293
#define ERROR_TAG 1295
#define DIE_TAG 1290
#define WORK_TAG 1291

MPI_Datatype MPI_TYPE_WORK;
BigFile bf = {0};
BigFile bfnew = {0};
BigBlock bb = {0};
BigBlock bbnew = {0};
int verbose = 0;
int Nfile = -1;
size_t CHUNKSIZE = 1 * 1024 * 1024;
int ThisTask, NTask;
char * newfilepath = NULL;
void slave(void);
void server(void);

double ratio = 1.0;
struct work {
    int64_t offset;
    int64_t seed;
    int64_t chunksize;
    int64_t offsetnew;
    int64_t nsel;
};

static size_t filesize();

int main(int argc, char * argv[]) {

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
    MPI_Comm_size(MPI_COMM_WORLD, &NTask);

    MPI_Type_contiguous(sizeof(struct work), MPI_BYTE, &MPI_TYPE_WORK);
    MPI_Type_commit(&MPI_TYPE_WORK);

    int ch;
    while(-1 != (ch = getopt(argc, argv, "n:N:vf:r:"))) {
        switch(ch) {
            case 'r':
                ratio = atof(optarg);
                break;
            case 'N':
            case 'n':
                Nfile = atoi(optarg);
                break;
            case 'f':
                newfilepath = optarg;
                break;
            case 'v':
                verbose = 1;
                break;
            default:
                usage();
        }
    }
    if(argc - optind + 1 != 4) {
        usage();
    }
    argv += optind - 1;
    if(0 != big_file_mpi_open(&bf, argv[1], MPI_COMM_WORLD)) {
        fprintf(stderr, "failed to open: %s\n", big_file_get_error_message());
        exit(1);
    }
    if(0 != big_file_mpi_open_block(&bf, &bb, argv[2], MPI_COMM_WORLD)) {
        fprintf(stderr, "failed to open: %s\n", big_file_get_error_message());
        exit(1);
    }
    if(Nfile == -1 || bb.Nfile == 0) {
        Nfile = bb.Nfile;
    }
    if(newfilepath == NULL) {
        newfilepath = argv[1];
    }
    if(0 != big_file_mpi_create(&bfnew, newfilepath, MPI_COMM_WORLD)) {
        fprintf(stderr, "failed to open: %s\n", big_file_get_error_message());
        exit(1);
    }
    size_t newsize = filesize();
    if(0 != big_file_mpi_create_block(&bfnew, &bbnew, argv[3], bb.dtype, bb.nmemb, Nfile, newsize, MPI_COMM_WORLD)) {
        fprintf(stderr, "failed to create temp: %s\n", big_file_get_error_message());
        exit(1);
    }

    /* copy attrs */
    size_t nattr;
    BigAttr * attrs = big_block_list_attrs(&bb, &nattr);
    int i;
    for(i = 0; i < nattr; i ++) {
        BigAttr * attr = &attrs[i];
        big_block_set_attr(&bbnew, attr->name, attr->data, attr->dtype, attr->nmemb);
    }

    if(bb.nmemb > 0 && bb.size > 0) {
    /* copy data */
        if(ThisTask == 0) {
            server();
        } else {
            slave();
        }
    }
    if(0 != big_block_mpi_close(&bbnew, MPI_COMM_WORLD)) {
        fprintf(stderr, "failed to close new: %s\n", big_file_get_error_message());
        exit(1);
    }
    big_block_mpi_close(&bb, MPI_COMM_WORLD);
    big_file_mpi_close(&bf, MPI_COMM_WORLD);
    big_file_mpi_close(&bfnew, MPI_COMM_WORLD);
    return 0;
}
static size_t filesize() {
    gsl_rng * rng = gsl_rng_alloc(gsl_rng_mt19937);
    gsl_rng_set(rng, 1984);
    int64_t offset = 0;
    int64_t offsetnew = 0;
    struct work work;
    for(offset = 0; offset < bb.size; ) {
        int64_t chunksize = CHUNKSIZE;

        /* never read beyond my end (read_simple caps at EOF) */
        if(offset + chunksize >= bb.size) {
            /* this is the last chunk */
            chunksize = bb.size - offset;
        }
        work.offset = offset;
        work.chunksize = chunksize;
        work.seed = gsl_rng_get(rng);
        work.offsetnew = offsetnew;
        if(ratio == 1.0) {
            work.nsel = chunksize;
        } else {
            work.nsel = gsl_ran_poisson(rng, chunksize * ratio);
        }

        offset += chunksize;
        offsetnew += work.nsel;
    }
    return offsetnew;
}
void server() {
    int64_t offset = 0;
    int64_t offsetnew = 0;
    struct work work;
    gsl_rng * rng = gsl_rng_alloc(gsl_rng_mt19937);
    gsl_rng_set(rng, 1984);
    for(offset = 0; offset < bb.size; ) {
        int64_t chunksize = CHUNKSIZE;
        MPI_Status status;
        int result = 0;
        MPI_Recv(&result, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD,
                &status);
        if(status.MPI_TAG == ERROR_TAG) {
            break;
        }

        /* never read beyond my end (read_simple caps at EOF) */
        if(offset + chunksize >= bb.size) {
            /* this is the last chunk */
            chunksize = bb.size - offset;
        }
        work.offset = offset;
        work.chunksize = chunksize;
        work.seed = gsl_rng_get(rng);
        work.offsetnew = offsetnew;
        if(ratio == 1.0) {
            work.nsel = chunksize;
        } else {
            work.nsel = gsl_ran_poisson(rng, chunksize * ratio);
        }
        MPI_Send(&work, 1, MPI_TYPE_WORK, status.MPI_SOURCE, WORK_TAG, MPI_COMM_WORLD);

        offset += chunksize;
        offsetnew += work.nsel;
        if(verbose) {
            fprintf(stderr, "%td / %td done (%0.4g%%)\r", offset, bb.size, (100. / bb.size) * offset);
        }
    }
    int i;
    for(i = 1; i < NTask; i ++) {
        struct work work;
        MPI_Send(&work, 1, MPI_TYPE_WORK, i, DIE_TAG, MPI_COMM_WORLD);
    }

}
void slave() {
    gsl_rng * rng = gsl_rng_alloc(gsl_rng_mt19937);
    int result = 0;
    MPI_Send(&result, 1, MPI_INT, 0, DONE_TAG, MPI_COMM_WORLD);
    while(1) {
        struct work work;
        MPI_Status status;
        MPI_Recv(&work, 1, MPI_TYPE_WORK, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

        if(status.MPI_TAG == DIE_TAG) {
            break;
        }
        gsl_rng_set(rng, work.seed);

        int64_t offset = work.offset;
        int64_t chunksize = work.chunksize;
        int64_t offsetnew = work.offsetnew;
        int64_t nsel = work.nsel;
        BigArray array;
        BigBlockPtr ptrnew;
        BigArray arraynew;

        size_t dims[2];
        void * buffer = malloc(dtype_itemsize(bb.dtype) * bb.nmemb * nsel);
        dims[0] = nsel;
        dims[1] = bb.nmemb;
        big_array_init(&arraynew, buffer, bb.dtype, 2, dims, NULL);

        ptrdiff_t i;
        size_t step = dtype_itemsize(bb.dtype) * bb.nmemb;
        size_t leftover = chunksize;
        char * p = buffer;
        char * q;
        if(0 != big_block_read_simple(&bb, offset, chunksize, &array, NULL)) {
            fprintf(stderr, "failed to read original: %s\n", big_file_get_error_message());
            result = -1;
            goto bad;
        }
        q = array.data;

//        printf("%ld %ld\n", nsel, leftover);
        for(i = 0; i < chunksize; i ++) {
            int64_t r = gsl_rng_uniform_int(rng, leftover);
            if(r < nsel) {
                memcpy(p, q, step);
                p += step;
                nsel --;
            }
            if(nsel == 0) break;
            leftover --;
            q += step;
        }
        if(nsel != 0) abort();
        free(array.data);
        if(0 != big_block_seek(&bbnew, &ptrnew, offsetnew)) {
            fprintf(stderr, "failed to seek new: %s\n", big_file_get_error_message());
            result = -1;
            free(arraynew.data);
            goto bad;
        }

        if(0 != big_block_write(&bbnew, &ptrnew, &arraynew)) {
            fprintf(stderr, "failed to write new: %s\n", big_file_get_error_message());
            result = -1;
            free(arraynew.data);
            goto bad;
        }

        free(arraynew.data);
        MPI_Send(&result, 1, MPI_INT, 0, DONE_TAG, MPI_COMM_WORLD);
        continue;
    bad:
        MPI_Send(&result, 1, MPI_INT, 0, ERROR_TAG, MPI_COMM_WORLD);
        continue;
    }
    return;
}
