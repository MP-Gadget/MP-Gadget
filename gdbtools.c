#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* these are for debuging in GDB */
#include "allvars.h"

int particle_by_id(MyIDType id) {
    int i;
    for(i = 0; i < NumPart; i++) {
        if(P[i].ID == id) return i;
    }
    return -1;
}
char * particle_by_timebin(int bin) {
    int i;
    static char buf[1024];
    char tmp[1024];
    strcpy(buf, "");
    for(i = 0; i < NumPart; i++) {
        if(P[i].TimeBin == bin) {
            strcpy(tmp, buf);
            snprintf(buf, 1020, "%s %d", tmp, i);
        }
    }
    return buf;
}

