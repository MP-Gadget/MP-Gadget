#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stddef.h>

#include <unistd.h>
#include <dirent.h>
#include "bigfile.h"

static void usage() {
    fprintf(stderr, "usage: bigfile-join [-t dtype] [-n nmemb] [-N nfile] filepath block datafile1 ...\n");
    fprintf(stderr, "join files into one block; the files are moved to the block directory\n");
    exit(1);
}

char * dtype;
int nmemb;
int Nfile;

int main(int argc, char * argv[]) {
    int opt;
    while(-1 != (opt = getopt(argc, argv, "t:n:N:"))) {
        switch(opt){
            case 'N':
                Nfile = atoi(optarg);
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
    if(argc - optind < 3) {
        usage();
    }
    argv += optind - 1;
    BigFile bf = {0};
    BigBlock bb = {0};
    if(0 != big_file_create(&bf, argv[1])) {
        fprintf(stderr, "failed to create file : %s\n", big_file_get_error_message());
        return -1;
    }
    int Nfile;
    int Ninput = argc - optind - 2;
    size_t size[Nfile];
    size_t total = 0;
    int i;
    for(i = 0; i < Ninput; i ++) {
        struct stat st;
        stat(argv[i + 3], &st);
        total += st.st_size / dtype_itemsize(dtype) / nmemb;
    }
    for(i = 0; i < Nfile; i ++) {

    }
    
    if(0 != big_file_create_block(&bf, &bb, argv[2], dtype, nmemb, Nfile, fsize)) {
        fprintf(stderr, "failed to create block: %s\n", big_file_get_error_message());
        return -1;
    }
    for(i = 0; i < Nfile; i ++) {
        FILE * fp = fopen
    }
    big_file_close(&bf);
    return 0;
}
