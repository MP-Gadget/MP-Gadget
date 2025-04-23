#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "gravity.h"
#include "utils/endrun.h"

/*
 * This table is computed by comparing with brute force calculation it matches the full PM exact up to 10 mesh sizes
 * for a point source. it is copied to a tighter array for better cache performance (hopefully)
 *
 * Generated with split = 1.25; check with the assertion above!
 * */
#include "shortrange-kernel.c"
#define NTAB (sizeof(shortrange_force_kernels) / sizeof(shortrange_force_kernels[0]))

/*! variables for short-range lookup table */
static float shortrange_table[NTAB], shortrange_table_potential[NTAB], shortrange_table_tidal[NTAB];

void
gravshort_fill_ntab(const enum ShortRangeForceWindowType ShortRangeForceWindowType, const double Asmth)
{
    if (ShortRangeForceWindowType == SHORTRANGE_FORCE_WINDOW_TYPE_EXACT) {
        if(Asmth != 1.5) {
            endrun(0, "The short range force window is calibrated for Asmth = 1.5, but running with %g\n", Asmth);
        }
    }

    size_t i;
    for(i = 0; i < NTAB; i++)
    {
        /* force_kernels is in units of mesh points; */
        double u = shortrange_force_kernels[i][0] * 0.5 / Asmth;
        switch (ShortRangeForceWindowType) {
            case SHORTRANGE_FORCE_WINDOW_TYPE_EXACT:
                /* Notice that the table is only calibrated for smth of 1.25*/
                shortrange_table[i] = shortrange_force_kernels[i][2]; /* ~ erfc(u) + 2.0 * u / sqrt(M_PI) * exp(-u * u); */
                /* The potential of the calibrated kernel is a bit off, so we still use erfc here; we do not use potential anyways.*/
                shortrange_table_potential[i] = shortrange_force_kernels[i][1];
            break;
            case SHORTRANGE_FORCE_WINDOW_TYPE_ERFC:
                shortrange_table[i] = erfc(u) + 2.0 * u / sqrt(M_PI) * exp(-u * u);
                shortrange_table_potential[i] = erfc(u);
            break;
        }
        /* we don't have a table for that and don't use it anyways. */
        shortrange_table_tidal[i] = 4.0 * u * u * u / sqrt(M_PI) * exp(-u * u);
    }
}

/* multiply force factor (*fac) and potential (*pot) by the shortrange force window function*/
int
grav_apply_short_range_window(double r, double * fac, double * pot, const double cellsize)
{
    const double dx = shortrange_force_kernels[1][0];
    double i = (r / cellsize / dx);
    size_t tabindex = floor(i);
    if(tabindex >= NTAB - 1)
        return 1;
    /* use a linear interpolation; */
    *fac *= (tabindex + 1 - i) * shortrange_table[tabindex] + (i - tabindex) * shortrange_table[tabindex + 1];
    *pot *= (tabindex + 1 - i) * shortrange_table_potential[tabindex] + (i - tabindex) * shortrange_table_potential[tabindex];
    return 0;
}
