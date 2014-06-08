#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stddef.h>

#include "bigblock.h"
#ifdef TEST
void test_create() {
    BigBlock bb = {};
    BigBlockPtr ptr = {0};
    size_t fsize[] = {20, 30};
    big_block_create(&bb, "testblock", "f4", 2, fsize);
    big_block_close(&bb);
    if(0 != big_block_open(&bb, "testblock")) {
        abort();
    }
    big_block_seek(&bb, &ptr, 0);
    int32_t data[50];
    int i;
    for(i = 0; i < 50; i ++) {
        data[i] = i;
    }
    big_block_write(&bb, &ptr, data, "i4", 50);
    big_block_close(&bb);
}

void test_open() {
    BigBlock bb = {};
    BigBlockPtr ptr = {0};
    big_block_open(&bb, "testblock");
    int64_t data;
    int64_t i;
    big_block_seek(&bb, &ptr, 0);
    for (i = 0; i < bb.size; i ++) {
        big_block_read(&bb, &ptr, &data, "i8", 1);
        printf("%ld %ld\n", i, data);
        big_block_seek_rel(&bb, &ptr, 1);
    }
    big_block_close(&bb);
}

int main(int argc, char * argv[]) {
    test_create();
    test_open();
}
#endif
