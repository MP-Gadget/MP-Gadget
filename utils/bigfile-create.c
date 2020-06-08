#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stddef.h>

#include <unistd.h>
#include <signal.h>
#include "bigfile.h"

static size_t countin;
static size_t count;

static void usr1(int j) {
    fprintf(stderr, "bigfile-create[%d]: %td \n",
        getpid(), 
        countin) ;
}
static void usage() {
    fprintf(stderr, "usage: bigfile-cat [-b] [-n nmemb] filepath block dtype\n");
    fprintf(stderr, "-B blocksize in bytes used in IO \n");
    exit(1);
}

int main(int argc, char * argv[]) {
    BigFile bf = {0};
    BigBlock bb = {0};
    int opt;

    size_t buffersize = 256 * 1024 * 1024;

    char * fmt = NULL;
    char * dtype = NULL;
    int nmemb = 1;

    while(-1 != (opt = getopt(argc, argv, "B:n:"))) {
        switch(opt) {
            case 'B':
                sscanf(optarg, "%td", &buffersize);
                break;
            case 'n':
                sscanf(optarg, "%d", &nmemb);
                break;
            default:
                usage();
        }
    }
    if(argc - optind != 3) {
        usage();
    }
    argv += optind - 1;
    if(0 != big_file_create(&bf, argv[1])) {
        fprintf(stderr, "failed to open: %s: %s\n", argv[1], big_file_get_error_message());
        exit(1);
    }
    dtype = argv[3];
    size_t fsize[1] = {-1};

    if(0 != big_file_create_block(&bf, &bb, argv[2], dtype, nmemb, 1, fsize)) {
        fprintf(stderr, "failed to create: %s: %s\n", argv[2], big_file_get_error_message());
        exit(1);
    }

    signal(SIGUSR1, usr1);

    size_t chunksize = buffersize / (bb.nmemb * dtype_itemsize(bb.dtype));
    BigBlockPtr ptr;
    BigArray array;

    char * buffer = malloc(buffersize);
    ptrdiff_t offset = 0;
    while(!feof(stdin)) {
        size_t nread = fread(buffer, bb.nmemb * dtype_itemsize(bb.dtype), chunksize, stdin);
        size_t dims[2] = {nread, bb.nmemb};
        big_array_init(&array, buffer, bb.dtype, 2, dims, NULL);
        big_block_seek(&bb, &ptr, offset);
        big_block_write(&bb, &ptr, &array);
        offset += nread;
        countin += nread;
    }
    bb.size = offset;
    bb.fsize[0] = offset;
    big_block_close(&bb);
    big_file_close(&bf);
    return 0;
}
