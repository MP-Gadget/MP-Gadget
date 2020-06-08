#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stddef.h>

#include <unistd.h>
#include "bigfile.h"

int longfmt = 0;

static void usage() {
    fprintf(stderr, "usage: bigfile-set-attr [-t dtype] [-n nmemb] filepath block attr val \n");
    exit(1);
}

int main(int argc, char * argv[]) {
    BigFile bf = {0};
    BigBlock bb = {0};

    int opt;
    char * dtype = NULL;
    char ndtype[8];
    int nmemb = 0;
    int force = 0; 
    while(-1 != (opt = getopt(argc, argv, "ft:n:"))) {
        switch(opt) {
            case 'f':
                force = 1;
                break;
            case 't':
                dtype = optarg;
                break;
            case 'n':
                nmemb = atoi(optarg);
                break;
            default:
                usage();
        }
    }
    argv += optind - 1;
    if(argc - optind < 4) {
        usage();
    }

    if(0 != big_file_open(&bf, argv[1])) {
        fprintf(stderr, "failed to open file : %s %s\n", argv[1], big_file_get_error_message());
        exit(1);
    }
    if(0 != big_file_open_block(&bf, &bb, argv[2])) {
        fprintf(stderr, "failed to open block: %s %s\n", argv[2], big_file_get_error_message());
        exit(1);
    }
    int i; 
    BigAttr * attr;

    if(force) 
        big_block_remove_attr(&bb, argv[3]);

    attr = big_block_lookup_attr(&bb, argv[3]);
    if(attr) {
        if(dtype == NULL) {
            dtype = attr->dtype;
        }
        if(nmemb == 0) {
            nmemb = attr->nmemb;
        }
    }
    char * data = malloc(big_file_dtype_itemsize(dtype) * nmemb);

    memset(data, 0, big_file_dtype_itemsize(dtype) * nmemb);

    if(big_file_dtype_kind(dtype) == 'S') {
        if (nmemb == 0) nmemb = strlen(argv[4]) + 1;
        if (nmemb != strlen(argv[4]) + 1) {
            fprintf(stderr, "nmemb and number of arguments mismatch\n");
            exit(1);
        }
        memcpy(data, argv[4], strlen(argv[4]) + 1); 
    } else {
        if (nmemb == 0) nmemb = argc - optind + 1 - 4;
        if (nmemb != argc - optind + 1 - 4) {
            fprintf(stderr, "nmemb and number of arguments mismatch\n");
            exit(1);
        }
        for(i = 4; i < argc - optind + 1; i ++) {
            char * p = data + (i - 4) * big_file_dtype_itemsize(dtype);
            if(0 != big_file_dtype_parse(argv[i], dtype, p, NULL)) {
                fprintf(stderr, "failed to parse the data `%s`\n", p);
                exit(1);
            }
        }
    }

    big_block_set_attr(&bb, argv[3], data, dtype, nmemb);

    free(data);
    big_block_close(&bb);
    big_file_close(&bf);
    return 0;
}
