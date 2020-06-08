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
    fprintf(stderr, "bigfile-cat[%d]: %td/%td %g%%\n",
        getpid(), 
        countin,
        count,
        1.0 * countin / count * 100.);
}
static void usage() {
    fprintf(stderr, "usage: bigfile-cat [-b] [-o offset] [-c count] [-B buffersize] filepath block \n");
    fprintf(stderr, "-b direct binary dump \n");
    fprintf(stderr, "-o seek to item at offset \n");
    fprintf(stderr, "-c read item count \n");
    fprintf(stderr, "-B blocksize in bytes used in IO \n");
    exit(1);
}
int main(int argc, char * argv[]) {
    BigFile bf = {0};
    BigBlock bb = {0};
    int opt;

    int humane = 1; /* binary or ascii */
    ptrdiff_t start = 0;
    ptrdiff_t size = -1;

    size_t buffersize = 256 * 1024 * 1024;

    char * fmt = NULL;

    while(-1 != (opt = getopt(argc, argv, "bo:c:B:f:"))) {
        switch(opt) {
            case 'f':
                fmt = optarg;
                break;
            case 'b':
                humane = 0;
                break;
            case 'o':
                sscanf(optarg, "%td", &start);
                break;
            case 'c':
                sscanf(optarg, "%td", &size);
                break;
            case 'B':
                sscanf(optarg, "%td", &buffersize);
                break;
            default:
                usage();
        }
    }
    if(argc - optind != 2) {
        usage();
    }
    argv += optind - 1;
    if(0 != big_file_open(&bf, argv[1])) {
        fprintf(stderr, "failed to open: %s: %s\n", argv[1], big_file_get_error_message());
        exit(1);
    }
    if(0 != big_file_open_block(&bf, &bb, argv[2])) {
        fprintf(stderr, "failed to open: %s: %s\n", argv[2], big_file_get_error_message());
        exit(1);
    }
    if(size == -1 || bb.size < size + start) {
        size = bb.size - start;
    } 

    count = size;

    signal(SIGUSR1, usr1);

    ptrdiff_t end = start + size;

    size_t chunksize = buffersize / (bb.nmemb * big_file_dtype_itemsize(bb.dtype));
    BigBlockPtr ptr;
    BigBlockPtr ptrnew;
    ptrdiff_t offset;
    BigArray array;

    for(offset = start; offset < end; ) {
        if(chunksize > size) chunksize = size;
        if(0 != big_block_read_simple(&bb, offset, chunksize, &array, NULL)) {
            fprintf(stderr, "failed to read original: %s\n", big_file_get_error_message());
            exit(1);
        }

        if(!humane) {
            fwrite(array.data, big_file_dtype_itemsize(bb.dtype), array.size, stdout);
        } else {
            char str[300];
            BigArrayIter iter;
            big_array_iter_init(&iter, &array);
            size_t i;
            for(i = 0; i < array.dims[0]; i++) {
                int j;
                for(j = 0; j < bb.nmemb; j ++) {
                    big_file_dtype_format(str, array.dtype, iter.dataptr, fmt);
                    big_array_iter_advance(&iter);
                    fprintf(stdout, "%s ", str);
                }
                fprintf(stdout, "\n");
            }
        }
        free(array.data);
        offset += array.dims[0];
        countin += array.dims[0];
    }
    big_block_close(&bb);
    big_file_close(&bf);
    return 0;
}
