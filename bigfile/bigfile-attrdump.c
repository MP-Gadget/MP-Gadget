#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stddef.h>

#include "bigfile.h"

static void print_attr(BigBlockAttr * attr, int brief) {
    char buffer[128];
    char * data = attr->data;
    int i;
    if(!brief) {
        printf("%s : %s : ", attr->dtype, attr->name, attr->nmemb);
        printf("[");
    }
    for(i = 0; i < attr->nmemb; i ++) {
        char * endl;
        if(i != attr->nmemb - 1) {
            endl = ", ";
        } else {
            endl = "";
        }
        dtype_format(buffer, attr->dtype, data);
        printf("%s%s", buffer, endl);
        data += dtype_itemsize(attr->dtype);
    }
    if(!brief) {
        printf("]");
    }
    printf("\n");
}
int main(int argc, char * argv[]) {
    BigFile bf = {0};
    BigBlock bb = {0};

    big_file_open(&bf, argv[1]);
    big_file_open_block(&bf, &bb, argv[2]);
    int i; 
    if(argc == 3) {
        for(i = 0; i < bb.attrset.listused; i ++) {
            BigBlockAttr * attr = &bb.attrset.attrlist[i];
            print_attr(attr, 0);
        }
    }
    for(i = 3; i < argc; i ++) {
        BigBlockAttr * attr = big_block_lookup_attr(&bb, argv[i]);
        if(attr) {
            print_attr(attr, argc == 4);
        } else {
            printf("%s, not attr:\n", argv[i]);
        }
    }
    big_block_close(&bb);
    big_file_close(&bf);
    return 0;
}
