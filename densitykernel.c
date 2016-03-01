#include <math.h>
#include "allvars.h"
#include "densitykernel.h"


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
double wk_cs(density_kernel_t * kernel, double q) {
    if(q < 1.0) {
        return 0.25 * pow(2 - q, 3) - pow(1 - q, 3);
    }
    if(q < 2.0) {
        return 0.25 * pow(2 - q, 3);
    }
    return 0.0;
}
double dwk_cs(density_kernel_t * kernel, double q) {
    if(q < 1.0) {
        return - 0.25 * 3 * pow(2 - q, 2) + 3 * pow(1 - q, 2);
    }
    if(q < 2.0) {
        return -0.25 * 3 * pow(2 - q, 2);
    }
    return 0.0;
}
static double wk_qus(density_kernel_t * kernel, double q) {
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
static double dwk_qus(density_kernel_t * kernel, double q) {
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
static double wk_qs(density_kernel_t * kernel, double q) {
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
static double dwk_qs(density_kernel_t * kernel, double q) {
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
    char * name;
    double (*wk)(density_kernel_t * kernel, double q);
    double (*dwk)(density_kernel_t * kernel, double q);
    double support; /* H / h, see Price 2011: arxiv 1012.1885*/
    double sigma[3];
} KERNELS[] = {
    { "Cubic Spline", wk_cs, dwk_cs, 2., 
        {2 / 3., 10 / (7 * M_PI), 1 / M_PI} },
    { "Quintic Spline", wk_qs, dwk_qs, 3.,
        {1 / 120., 7 / (478 * M_PI), 1 / (120 * M_PI)} },
    { "Quartic Spline", wk_qus, dwk_qus, 2.5,
        {1 / 24., 96 / (1199 * M_PI), 1 / (20 * M_PI)} },
};
char * density_kernel_name(int type) {
    return KERNELS[type].name;
}
int density_kernel_type_end() {
    return sizeof(KERNELS) / sizeof(KERNELS[0]);
}
double density_kernel_dwk(density_kernel_t * kernel, double u) {
    double support = KERNELS[kernel->type].support;
    return kernel->dWknorm * 
        KERNELS[kernel->type].dwk(kernel, u * support);
}

double density_kernel_wk(density_kernel_t * kernel, double u) {
    double support = KERNELS[kernel->type].support;
    return kernel->Wknorm * 
        KERNELS[kernel->type].wk(kernel, u * support);
}
double density_kernel_support(int type) {
    return KERNELS[type].support;
}
int density_kernel_desnumngb(int type, double eta) {
    /* this returns the expected number of ngb in for given sph resolution
     * deseta */
    /* See Price: arxiv 1012.1885. eq 12 */
    double support = KERNELS[type].support;
    return NORM_COEFF * pow(support * eta, NUMDIMS);
}
double density_kernel_volume(density_kernel_t * kernel) {
    /* this returns the expected number of ngb in for given sph resolution
     * deseta */
    /* See Price: arxiv 1012.1885. eq 12 */
    return NORM_COEFF * pow(kernel->H, NUMDIMS);
}
void density_kernel_init(density_kernel_t * kernel, double H) {
    density_kernel_init_with_type(kernel, All.DensityKernelType, H);
}
void density_kernel_init_with_type(density_kernel_t * kernel, int type, double H) {
    kernel->H = H;
    kernel->HH = H * H;
    kernel->Hinv = 1. / H;
    kernel->type = type;
    double support = KERNELS[kernel->type].support;
    double sigma = KERNELS[kernel->type].sigma[NUMDIMS - 1];
    double hinv = kernel->Hinv * support;

    kernel->Wknorm = sigma * pow(hinv, NUMDIMS);
    kernel->dWknorm = kernel->Wknorm * hinv;
}
