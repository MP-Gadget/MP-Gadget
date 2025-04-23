#ifndef INTERP_H
#define INTERP_H

#include <stddef.h>
#include <stdint.h>

typedef struct {
    int Ndim;
    int * dims;
    ptrdiff_t * strides;
    double * Min;
    double * Step;
    double * Max;

    void * data; /* internal buffer for all pointer data */
    int fsize;
} Interp;

void interp_init(Interp * obj, int Ndim, int64_t * dims);

/* set the upper and lower limit of dimension d */
void interp_init_dim(Interp * obj, int d, double Min, double Max);

/* interpolate the table at point x;
 * status: array of length dimension,
 * will be -1 if below lower bound
 *         +1 if above upper bound  */
double interp_eval(Interp * obj, double * x, double * ydata, int * status);
double interp_eval_periodic(Interp * obj, double * x, double * ydata);

void interp_destroy(Interp * obj);
#endif
