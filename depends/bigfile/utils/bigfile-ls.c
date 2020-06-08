#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stddef.h>

#include <unistd.h>
#include <dirent.h>
#include "bigfile.h"
#include <sys/stat.h>

int longfmt = 0;


static void usage() {
    fprintf(stderr, "usage: bigfile-ls [-l] filepath [block]\n");
    fprintf(stderr, "-l: long format: blockname dtype nmemb counts sysvchecksum nfile abspath\n");
    exit(1);
}

static int listbigblock(BigFile * bf, char * blockname);
static void listbigfile_r(BigFile * bf, char * path);
void listbigfile(char * filename) {
    BigFile bf = {0};
    if(0 != big_file_open(&bf, filename)) {
        fprintf(stderr, "failed to open: %s : %s\n", filename, big_file_get_error_message());
        exit(1);
    }
    listbigfile_r(&bf, "");
    big_file_close(&bf);
}
static int (filter)(const struct dirent * ent) {
    if(ent->d_name[0] == '.') return 0;
//    printf("%s %d\n", ent->d_name, ent->d_type);
// ent->d_type is unknown on COMA.
//if(ent->d_type != DT_DIR) return 0;
    return 1;
}

static void listbigfile_r(BigFile * bf, char * path) {
    struct dirent **namelist;
    struct stat st;
    int n;
    char * buf = alloca(strlen(bf->basename) + strlen(path) + 10);
    if(strlen(path) > 0) 
        sprintf(buf, "%s/%s", bf->basename, path);
    else
        sprintf(buf, "%s", bf->basename);
    stat(buf, &st);
    if(!S_ISDIR(st.st_mode)) return;
    n = scandir(buf, &namelist, filter, alphasort);
    if (n < 0) {
        fprintf(stderr, "cannot open dir: %s\n", buf);
    } else {
        int i;
        for(i = 0; i < n ; i ++) {
            char * blockname = alloca(strlen(namelist[i]->d_name) + strlen(path) + 10);
            if(strlen(path) > 0) 
                sprintf(blockname, "%s/%s", path, namelist[i]->d_name);
            else
                sprintf(blockname, "%s", namelist[i]->d_name);
            listbigblock(bf, blockname);
            listbigfile_r(bf, blockname);
            free(namelist[i]);
        }
        free(namelist);
    }
}

static int listbigblock(BigFile * bf, char * blockname) {
    BigBlock bb = {0};
    if(0 != big_file_open_block(bf, &bb, blockname)) {
        return -1;
    }
    if(!longfmt) {
        fprintf(stdout, "%s\n", blockname);
    } else {
        int i;
        unsigned int sum = 0;
        /* build the sysv sum from sum of each physical file */
        for(i = 0; i < bb.Nfile; i ++) {
            sum += bb.fchecksum[i];
        }
        unsigned int s = sum;
        unsigned int r = (s & 0xffff) + ((s & 0xffffffff) >> 16);
        unsigned int checksum = (r & 0xffff) + (r >> 16);
        fprintf(stdout, "%-28s %s %d %12td %05u %d %s\n", 
                blockname, 
                bb.dtype, bb.nmemb, bb.size, checksum, bb.Nfile,
                bb.basename
                );
    }
    big_block_close(&bb);
    return 0;
}

int main(int argc, char * argv[]) {
    int opt;
    while(-1 != (opt = getopt(argc, argv, "l"))) {
        switch(opt){
            case 'l':
                longfmt = 1;
                break;
            default:
                usage();
        }
    }
    if(argc - optind < 1) {
        usage();
    }
    argv += optind - 1;
    if(argc - optind == 1) {
        listbigfile(argv[1]);
    } else {
        BigFile bf = {0};
        if(0 != big_file_open(&bf, argv[1])) {
            fprintf(stderr, "failed to open: %s : %s\n", argv[1], big_file_get_error_message());
            exit(1);
        }
        if(0 != listbigblock(&bf, argv[2])) {
            fprintf(stderr, "failed to open: %s\n", big_file_get_error_message());
        }
        big_file_close(&bf);
    }
    return 0;
}
