#include <mpi.h>
#include <stdio.h>
#include <string.h>

#include "partmanager.h"


/* these are for debuging in GDB */
int GDB_particle_by_id(MyIDType id, int from) {
    int i;
    for(i = from; i < PartManager->NumPart; i++) {
        if(P[i].ID == id) return i;
    }
    return -1;
}

int GDB_particle_by_type(int type, int from) {
    int i;
    for(i = from; i < PartManager->NumPart; i++) {
        if(P[i].Type == type) return i;
    }
    return -1;
}

int GDB_particle_by_generation(int gen, int from) {
    int i;
    for(i = from; i < PartManager->NumPart; i++) {
        if(P[i].Generation == gen) return i;
    }
    return -1;
}

char * GDB_particle_by_timebin(int bin) {
    int i;
    static char buf[1024];
    char tmp[20] = {'\0'};
    strcpy(buf, "");
    for(i = 0; i < PartManager->NumPart; i++) {
        if(P[i].TimeBinHydro == bin) {
            snprintf(tmp, 15, " %d", i);
            strncat(buf, tmp, 1024-strlen(tmp)-1);
        }
    }
    return buf;
}

int GDB_find_garbage(int from) {
    int i;
    for(i = from; i < PartManager->NumPart; i++) {
        if(P[i].IsGarbage) return i;
    }
    return -1;
}

char * GDB_format_particle(int i) {
    static char buf[1024];
    char * p = buf;
    int n = 1024;

#define add(fmt, ...) \
        snprintf(p, n - 1, fmt, __VA_ARGS__ ); \
        p = buf + strlen(buf); \
        n = 4096 - strlen(buf)

    add("P[%d]: ", i);
    add("ID : %lu ", P[i].ID);
    add("Generation: %d ", (int) P[i].Generation);
    add("Mass : %g ", P[i].Mass);
    add("Pos: %g %g %g ", P[i].Pos[0], P[i].Pos[1], P[i].Pos[2]);
    add("Vel: %g %g %g ", P[i].Vel[0], P[i].Vel[1], P[i].Vel[2]);
    add("FullTreeGravAccel: %g %g %g ", P[i].FullTreeGravAccel[0], P[i].FullTreeGravAccel[1], P[i].FullTreeGravAccel[2]);
    add("GravPM: %g %g %g ", P[i].GravPM[0], P[i].GravPM[1], P[i].GravPM[2]);
    return buf;
}
