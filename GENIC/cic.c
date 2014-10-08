#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <math.h>
#include "cic.h"

void cic_init(CIC * cic, int Ngrid, double BoxSize) {
    int k;
    size_t last = 1;
    for(k = 0; k < 3; k++) {
        cic->strides[2 - k] = last;
        last *= Ngrid;
    }
    cic->Ngrid = Ngrid;
    cic->BoxSize = BoxSize;
    cic->CellSize = BoxSize / Ngrid;
    cic->buffer = malloc(sizeof(double) * last);
    cic->size = last;
}

void cic_add_particle(CIC * cic, double Pos[3], double mass) {
    ptrdiff_t ret = 0;
    int k;
    int iCell[3];
    double Res[3];
    for(k = 0; k < 3; k++) {
        double tmp = Pos[k] / cic->CellSize;
        while(tmp < 0) tmp += cic->Ngrid;
        while(tmp  >= cic->Ngrid) tmp -= cic->Ngrid;
        iCell[k] = tmp;
        Res[k] = tmp - iCell[k];
    }

    int connection = 0;
    double wtsum = 0.0;
    for(connection = 0; connection < 8; connection++) {
        double weight = 1.0;
        ptrdiff_t linear = 0;
        for(k = 0; k < 3; k++) {
            int offset = (connection >> k) & 1;
            int tmp = iCell[k] + offset;
            if(tmp >= cic->Ngrid) tmp -= cic->Ngrid;
            linear += tmp * cic->strides[k];
            weight *= offset?
                /* offset == 1*/ (Res[k])    :
                /* offset == 0*/ (1 - Res[k]);
        }
#pragma omp atomic
        cic->buffer[linear] += weight * mass;
        wtsum += weight;
    }
    if(fabs(wtsum - 1.0) > 1e-6) {
        abort(); 
    }
}
void cic_destroy(CIC * cic) {
    free(cic->buffer);
    cic->buffer = NULL;
}



