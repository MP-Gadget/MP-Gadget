#ifndef _DENSITY_KERNEL_H
#define _DENSITY_KERNEL_H

#if !defined(TWODIMS) && !defined(ONEDIM)
#define  NUMDIMS 3		/*!< For 3D-normalized kernel */
#define  NORM_COEFF      4.188790204786	/*!< Coefficient for kernel normalization. Note:  4.0/3 * PI = 4.188790204786 */
#else
#ifdef  TWODIMS
#define  NUMDIMS 2		/*!< For 2D-normalized kernel */
#define  NORM_COEFF      M_PI	/*!< Coefficient for kernel normalization. */
#else
#define  NUMDIMS 1             /*!< For 1D-normalized kernel */
#define  NORM_COEFF      2.0
#endif
#endif

enum DensityKernelType {
    DENSITY_KERNEL_CUBIC_SPLINE = 1,
    DENSITY_KERNEL_QUINTIC_SPLINE = 2,
    DENSITY_KERNEL_QUARTIC_SPLINE = 4,
};

typedef struct {
    double H;
    double HH;
    double Hinv; /* convert from r to u*/
    int type;
    double support;
    const char * name;
    /* private: */
    double Wknorm;
    double dWknorm;
} DensityKernel;

double
density_kernel_desnumngb(DensityKernel * kernel, double eta);
void
density_kernel_init(DensityKernel * kernel, double H, enum DensityKernelType type);
double
density_kernel_wk(DensityKernel * kernel, double u);
double
density_kernel_dwk(DensityKernel * kernel, double u);
double
density_kernel_volume(DensityKernel * kernel);

static inline double
density_kernel_dW(DensityKernel * kernel, double u, double wk, double dwk)
{
    return - (NUMDIMS * kernel->Hinv * wk + u * dwk);
}

static inline double
dotproduct(const double v1[3], const double v2[3])
{
    double r =0;
    int d;
    for(d = 0; d < 3; d ++) {
        r += v1[d] * v2[d];
    }
    return r;
}

static inline void crossproduct(const double v1[3], const double v2[3], double out[3])
{
    static const int D2[3] = {1, 2, 0};
    static const int D3[3] = {2, 0, 1};

    int d1;
    for(d1 = 0; d1 < 3; d1++)
    {
        const int d2 = D2[d1];
        const int d3 = D3[d1];
        out[d1] = (v1[d2] * v2[d3] -  v2[d2] * v1[d3]);
    }
}

#endif
