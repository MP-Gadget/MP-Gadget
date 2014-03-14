#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_integration.h>
#include "allvars.h"
#include "proto.h"

#define NENTRY 1024
static double tab_log10a[NENTRY];
static double dlog10a;
static double tab_Dc[NENTRY];

/*
M, L = self.M, self.L
  logx = numpy.linspace(log10amin, 0, Np)
  def kernel(log10a):
    a = numpy.exp(log10a)
    return 1 / self.Ea(a) * a ** -1 # dz = - 1 / a dlog10a
  y = numpy.array( [romberg(kernel, log10a, 0, vec_func=True, divmax=10) for log10a in logx])
*/
static double kernel(double log10a, void * params) {
    double a = pow(10, log10a);
    return 1 / hubble_function(a) * All.Hubble / a;
} 

static void lightcone_init_entry(int i) {
    tab_log10a[i] = dlog10a * (NENTRY - i - 1);

    gsl_integration_workspace * w = gsl_integration_workspace_alloc (1000); 

    double result, error;

    gsl_function F;
    F.function = &kernel;

    gsl_integration_qags (&F, tab_log10a[i], 0, 0, 1e-7, 1000,
            w, &result, &error); 

    /*
    printf ("result          = % .18f\n", result);
    printf ("exact result    = % .18f\n", expected);
    printf ("estimated error = % .18f\n", error);
    printf ("actual error    = % .18f\n", result - expected);
    printf ("intervals =  %td\n", w->size);
    */
    /* result is in DH, hubble distance */
    /* convert to cm / h */
    result *= C / HUBBLE;
    printf("DH = %g\n", C / HUBBLE / All.UnitLength_in_cm);
    /* convert to Kpc/h or internal units */
    result /= All.UnitLength_in_cm;

    gsl_integration_workspace_free (w);
    tab_Dc[i] = result;
    double a = pow(10.0, tab_log10a[i]);
    double z = 1 / a - 1;
    printf("a = %g z = %g Dc = %g\n", a, z, result);
}

void lightcone_init() {
    int i;
    dlog10a = log10(All.TimeBegin) / (NENTRY - 1);
    for(i = 0; i < NENTRY; i ++) {
        lightcone_init_entry(i);
    };

}
