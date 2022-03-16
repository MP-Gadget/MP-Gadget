#include <math.h>
#include "densitykernel.h"

#include "utils/endrun.h"

/**
 *
 * We use Price 1012.1885 kernels
 * sml in Gadget is the support big H in Price,
 *
 * u = r / H
 * q = r / h
 *
 * luckily, wk = 1 / H ** 3 W_volker(u)
 *             = 1 / h ** 3 W_price(q)
 * and     dwk = 1 / H ** 4 dw_volker/du
 *             = 1 / h ** 4 dw_price/dq
 *
 * wk_xx is Price eq 6 , 7, 8, without sigma
 *
 * the function density_kernel_wk and _dwk takes u to maintain compatibility
 * with volker's gadget.
 */
double wk_cs(DensityKernel * kernel, double q) {
    if(q < 1.0) {
        return 0.25 * pow(2 - q, 3) - pow(1 - q, 3);
    }
    if(q < 2.0) {
        return 0.25 * pow(2 - q, 3);
    }
    return 0.0;
}
double dwk_cs(DensityKernel * kernel, double q) {
    if(q < 1.0) {
        return - 0.25 * 3 * pow(2 - q, 2) + 3 * pow(1 - q, 2);
    }
    if(q < 2.0) {
        return -0.25 * 3 * pow(2 - q, 2);
    }
    return 0.0;
}
static double wk_qus(DensityKernel * kernel, double q) {
    if(q < 0.5) {
        return pow(2.5 - q, 4) - 5 * pow(1.5 - q, 4) + 10 * pow(0.5 - q, 4);
    }
    if(q < 1.5) {
        return pow(2.5 - q, 4) - 5 * pow(1.5 - q, 4);
    }
    if(q < 2.5) {
        return pow(2.5 - q, 4);
    }
    return 0.0;
}
static double dwk_qus(DensityKernel * kernel, double q) {
    if(q < 0.5) {
        return -4 * pow(2.5 - q, 3) + 20 * pow(1.5 - q, 3) - 40 * pow(0.5 - q, 3);
    }
    if(q < 1.5) {
        return -4 * pow(2.5 - q, 3) + 20 * pow(1.5 - q, 3);
    }
    if(q < 2.5) {
        return -4 * pow(2.5 - q, 3);
    }
    return 0.0;
}
static double wk_qs(DensityKernel * kernel, double q) {
    if(q < 1.0) {
        return pow(3 - q, 5) - 6 * pow(2 - q, 5) + 15 * pow(1 - q, 5);
    }
    if(q < 2.0) {
        return pow(3 - q, 5)- 6 * pow(2 - q, 5);
    }
    if(q < 3.0) {
        return pow(3 - q, 5);
    }
    return 0.0;
}
static double dwk_qs(DensityKernel * kernel, double q) {
    if(q < 1.0) {
        return -5 * pow(3 - q, 4) + 30 * pow(2 - q, 4)
             - 75 * pow (1 - q, 4);
    }
    if(q < 2.0) {
        return -5 * pow(3 - q, 4) + 30 * pow(2 - q, 4);
    }
    if(q < 3.0) {
        return -5 * pow(3 - q, 4);
    }
    return 0.0;
}

static struct {
    const char * name;
    double (*wk)(DensityKernel * kernel, double q);
    double (*dwk)(DensityKernel * kernel, double q);
    double support; /* H / h, see Price 2011: arxiv 1012.1885*/
    double sigma[3];
} KERNELS[] = {
    { "CubicSpline", wk_cs, dwk_cs, 2.,
        {2 / 3., 10 / (7 * M_PI), 1 / M_PI} },
    { "QuinticSpline", wk_qs, dwk_qs, 3.,
        {1 / 120., 7 / (478 * M_PI), 1 / (120 * M_PI)} },
    { "QuarticSpline", wk_qus, dwk_qus, 2.5,
        {1 / 24., 96 / (1199 * M_PI), 1 / (20 * M_PI)} },
};

double
density_kernel_dwk(DensityKernel * kernel, double u)
{
    double support = KERNELS[kernel->type].support;
    return kernel->dWknorm *
        KERNELS[kernel->type].dwk(kernel, u * support);
}

double
density_kernel_wk(DensityKernel * kernel, double u)
{
    double support = KERNELS[kernel->type].support;
    return kernel->Wknorm *
        KERNELS[kernel->type].wk(kernel, u * support);
}

double
density_kernel_desnumngb(DensityKernel * kernel, double eta)
{
    /* this returns the expected number of ngb in for given sph resolution
     * deseta */
    /* See Price: arxiv 1012.1885. eq 12 */
    double support = kernel->support;
    return NORM_COEFF * pow(support * eta, NUMDIMS);
}

double
density_kernel_volume(DensityKernel * kernel)
{
    return NORM_COEFF * pow(kernel->H, NUMDIMS);
}

static void
density_kernel_init_with_type(DensityKernel * kernel, int type, double H)
{
    kernel->H = H;
    kernel->HH = H * H;
    kernel->Hinv = 1. / H;
    kernel->type = type;
    kernel->name = KERNELS[kernel->type].name;
    kernel->support = KERNELS[kernel->type].support;

    double sigma = KERNELS[kernel->type].sigma[NUMDIMS - 1];
    double hinv = kernel->Hinv * kernel->support;

    kernel->Wknorm = sigma * pow(hinv, NUMDIMS);
    kernel->dWknorm = kernel->Wknorm * hinv;
}

void
density_kernel_init(DensityKernel * kernel, double H, enum DensityKernelType type)
{
    int t = -1;
    if(type == DENSITY_KERNEL_CUBIC_SPLINE) {
        t = 0;
    } else
    if(type == DENSITY_KERNEL_QUINTIC_SPLINE) {
        t = 1;
    } else
    if(type == DENSITY_KERNEL_QUARTIC_SPLINE) {
        t = 2;
    } else {
        endrun(1, "Density Kernel type is unknown\n");
    }
    density_kernel_init_with_type(kernel, t, H);
}
