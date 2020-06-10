#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stddef.h>

#include <unistd.h>
#include "bigfile.h"

int longfmt = 0;

static void usage() {
    fprintf(stderr, "usage: bigfile-attrdump [-l] filepath block [attr1] [attr2] ...\n");
    exit(1);
}

static void print_attr(BigAttr * attr, int brief) {
    char buffer[128];
    char * data = attr->data;
    int i;
    if(!brief || longfmt) {
        printf("%-20s %s %3d  ", attr->name, attr->dtype, attr->nmemb);
        printf("[ ");
    }
    for(i = 0; i < attr->nmemb; i ++) {
        char * endl;
        if(i != attr->nmemb - 1 && attr->dtype[1] != 'S') {
            endl = " ";
        } else {
            endl = "";
        }
        big_file_dtype_format(buffer, attr->dtype, data, NULL);
        printf("%s%s", buffer, endl);
        data += dtype_itemsize(attr->dtype);
    }
    if(!brief || longfmt) {
        printf(" ]");
    }
    printf("\n");
}
int main(int argc, char * argv[]) {
    BigFile bf = {0};
    BigBlock bb = {0};

    int opt;
    while(-1 != (opt = getopt(argc, argv, "l"))) {
        switch(opt) {
            case 'l':
                longfmt = 1;
                break;
            default:
                usage();
        }
    }
    argv += optind - 1;
    if(argc - optind < 2) {
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
    if(argc - optind == 2) {
        size_t nattr;
        BigAttr * attrs = big_block_list_attrs(&bb, &nattr);
        for(i = 0; i < nattr; i ++) {
            BigAttr * attr = &attrs[i];
            print_attr(attr, 0);
        }
    }
    for(i = 3; i < argc - optind + 1; i ++) {
        BigAttr * attr = big_block_lookup_attr(&bb, argv[i]);
        if(attr) {
            print_attr(attr, argc - optind == 3);
        } else {
            printf("%s, not attr:\n", argv[i]);
        }
    }
    big_block_close(&bb);
    big_file_close(&bf);
    return 0;
}
