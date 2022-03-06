#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <math.h>

#include "interp.h"
#include "mymalloc.h"

void interp_init(Interp * obj, int Ndim, int * dims) {
    ptrdiff_t N = 1;
    int d;
    for(d = 0 ; d < Ndim; d ++) {
        N *= dims[d];     
    }

    /* alloc memory */
    obj->Ndim = Ndim;
    obj->data = mymalloc("interp_data", 0
        +   sizeof(double) * Ndim * 3
        +   sizeof(ptrdiff_t) * Ndim
        +   sizeof(int) * Ndim);

    obj->dims = (int *) obj->data;
    obj->strides = (ptrdiff_t*) (obj->dims + Ndim);
    obj->Min = (double*) (obj->strides + Ndim);
    obj->Step = obj->Min + Ndim;
    obj->Max = obj->Step + Ndim;

    /* fillin strides */
    N = 1;
    for(d = Ndim - 1 ; d >= 0; d --) {
        obj->dims[d] = dims[d];
        /* column first, C ordering */
        obj->strides[d] = N;
        N *= dims[d];     
    }

    int fsize = 1;
    for(d = 0; d < obj->Ndim; d++) {
        fsize *= 2;
    }
    obj->fsize = fsize;
}

/* set up an interpolation dimension.
 * Max is inclusive. aka if dims[d] == 2, Min = 0, Max = 1
 * then the steps are 0, 1.
 * 
 **/
void interp_init_dim(Interp * obj, int d, double Min, double Max) {
    obj->Min[d] = Min;
    obj->Max[d] = Max;
    obj->Step[d] = (Max - Min) / (obj->dims[d] - 1);
}


static ptrdiff_t linearindex(ptrdiff_t * strides, int * xi, int Ndim) {
    int d;
    ptrdiff_t rt = 0;
    for(d = 0; d < Ndim; d++) {
        rt += strides[d] * xi[d] ;
    }
    return rt;
}

/* status:
 * 0 if interplation is good on that axis
 * -1 below
 * +1 above 
 * status needs to be array of Ndim
 * */
double interp_eval(Interp * obj, double * x, double * ydata, int * status) {
    int d;
    if(status == NULL) {
        status = (int *) alloca(sizeof(int) * obj->Ndim);
    }
    int * xi = (int *) alloca(sizeof(int) * obj->Ndim);
    double * f = (double *) alloca(sizeof(double) * obj->Ndim);

    for(d = 0; d < obj->Ndim; d++) {
        double xd = (x[d] - obj->Min[d]) / obj->Step[d];
        if (x[d] < obj->Min[d]) {
            status[d] = -1;
            xi[d] = 0;
            f[d] = 0;
        } else
        if (x[d] > obj->Max[d]) {
            status[d] = 1;
            xi[d] = obj->dims[d] - 1;
            f[d] = 0;
        } else {
            xi[d] = floor(xd);
            f[d] = xd - xi[d];
            status[d] = 0;
        }
    }

    double ret = 0;
    /* the origin, "this point" */
    ptrdiff_t l0 = linearindex(obj->strides, xi, obj->Ndim);

    int i;
    /* for each point covered by the filter */
    for(i = 0; i < obj->fsize; i ++) {
        double filter = 1.0;
        ptrdiff_t l = l0;
        int skip = 0;
        for(d = 0; d < obj->Ndim; d++ ) {
            int foffset = (i & (1 << d))?1:0;
            if(f[d] == 0 && foffset == 1) {
                /* on this dimension the second data point 
                 * is not needed */
                skip = 1;
                break;
            }

            /* 
             * are we on this point or next point?
             *
             * weight on next point is f[d]
             * weight on this point is 1 - f[d] 
             * */
            filter *= foffset?f[d] : (1 - f[d]);
            l += foffset * obj->strides[d];
        }
        if(!skip) {
            ret += ydata[l] * filter;
        }
    }
    return ret;
}

/* interpolation assuming periodic boundary */
double interp_eval_periodic(Interp * obj, double * x, double * ydata) {
    int * xi = (int *) alloca(sizeof(int) * obj->Ndim);
    int * xi1 = (int *) alloca(sizeof(int) * obj->Ndim);
    double * f = (double *) alloca(sizeof(double) * obj->Ndim);

    int d;
    for(d = 0; d < obj->Ndim; d++) {
        double xd = (x[d] - obj->Min[d]) / obj->Step[d];
        xi[d] = floor(xd);
        f[d] = xd - xi[d];
    }

    double ret = 0;
    /* the origin, "this point" */

    int i;
    /* for each point covered by the filter */
    for(i = 0; i < obj->fsize; i ++) {
        double filter = 1.0;
        for(d = 0; d < obj->Ndim; d++ ) {
            int foffset = (i & (1 << d))?1:0;
            xi1[d] = xi[d] + foffset;
            while(xi1[d] >= obj->dims[d]) xi1[d] -= obj->dims[d];
            while(xi1[d] < 0 ) xi1[d] += obj->dims[d];
            /* 
             * are we on this point or next point?
             *
             * weight on next point is f[d]
             * weight on this point is 1 - f[d] 
             * */
            filter *= foffset?f[d] : (1 - f[d]);
        }
        ptrdiff_t l = linearindex(obj->strides, xi1, obj->Ndim);
        ret += ydata[l] * filter;
    }
    return ret;
}

void interp_destroy(Interp * obj) {
    myfree(obj->data);
}

