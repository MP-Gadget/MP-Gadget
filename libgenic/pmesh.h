#ifndef PMESH_H
#define PMESH_H
#include <gsl/gsl_rng.h>
#include <libgadget/petapm.h>
#include <libgadget/utils.h>

/*
 * The following functions are from fastpm/libfastpm/initialcondition.c.
 * Agrees with nbodykit's pmesh/whitenoise.c, which agrees with n-genic.
 * */
typedef struct {
    struct {
        ptrdiff_t start[3];
        ptrdiff_t size[3];
        ptrdiff_t strides[3];
        ptrdiff_t total;
    } ORegion;
    int Nmesh[3];
} PMDesc;

static inline void
SETSEED(PMDesc * pm, unsigned int * table[2][2], int i, int j, gsl_rng * rng)
{
    unsigned int seed = 0x7fffffff * gsl_rng_uniform(rng);

    int ii[2] = {i, (pm->Nmesh[0] - i) % pm->Nmesh[0]};
    int jj[2] = {j, (pm->Nmesh[1] - j) % pm->Nmesh[1]};
    int d1, d2;
    for(d1 = 0; d1 < 2; d1++) {
        ii[d1] -= pm->ORegion.start[0];
        jj[d1] -= pm->ORegion.start[1];
    }
    for(d1 = 0; d1 < 2; d1++)
    for(d2 = 0; d2 < 2; d2++) {
        if( ii[d1] >= 0 &&
            ii[d1] < pm->ORegion.size[0] &&
            jj[d2] >= 0 &&
            jj[d2] < pm->ORegion.size[1]
        ) {
            table[d1][d2][ii[d1] * pm->ORegion.size[1] + jj[d2]] = seed;
        }
    }
}
static inline unsigned int
GETSEED(PMDesc * pm, unsigned int * table[2][2], int i, int j, int d1, int d2)
{
    i -= pm->ORegion.start[0];
    j -= pm->ORegion.start[1];
    if(i < 0) abort();
    if(j < 0) abort();
    if(i >= pm->ORegion.size[0]) abort();
    if(j >= pm->ORegion.size[1]) abort();
    return table[d1][d2][i * pm->ORegion.size[1] + j];
}

static void
SAMPLE(gsl_rng * rng, double * ampl, double * phase)
{
    *phase = gsl_rng_uniform(rng) * 2 * M_PI;
    *ampl = 0;
    do *ampl = gsl_rng_uniform(rng); while(*ampl == 0);
}

static void
pmic_fill_gaussian_gadget(PMDesc * pm, double * delta_k, int seed, int setUnitaryAmplitude, int setInvertPhase)
{
    /* Fill delta_k with gadget scheme */
    int d;
    int i, j, k;

    gsl_rng * rng = gsl_rng_alloc(gsl_rng_ranlxd1);
    gsl_rng_set(rng, seed);

    unsigned int * seedtable[2][2];
    for(i = 0; i < 2; i ++)
    for(j = 0; j < 2; j ++) {
            seedtable[i][j] = (unsigned int *) mymalloc("seedtable", pm->ORegion.size[0] * pm->ORegion.size[1] * sizeof(int));
            memset(seedtable[i][j], 0, pm->ORegion.size[0] * pm->ORegion.size[1] * sizeof(int));
    }

    for(i = 0; i < pm->Nmesh[0] / 2; i++) {
        for(j = 0; j < i; j++) SETSEED(pm, seedtable, i, j, rng);
        for(j = 0; j < i + 1; j++) SETSEED(pm, seedtable, j, i, rng);
        for(j = 0; j < i; j++) SETSEED(pm, seedtable, pm->Nmesh[0] - 1 - i, j, rng);
        for(j = 0; j < i + 1; j++) SETSEED(pm, seedtable, pm->Nmesh[1] - 1 - j, i, rng);
        for(j = 0; j < i; j++) SETSEED(pm, seedtable, i, pm->Nmesh[1] - 1 - j, rng);
        for(j = 0; j < i + 1; j++) SETSEED(pm, seedtable, j, pm->Nmesh[0] - 1 - i, rng);
        for(j = 0; j < i; j++) SETSEED(pm, seedtable, pm->Nmesh[0] - 1 - i, pm->Nmesh[1] - 1 - j, rng);
        for(j = 0; j < i + 1; j++) SETSEED(pm, seedtable, pm->Nmesh[1] - 1 - j, pm->Nmesh[0] - 1 - i, rng);
    }
    gsl_rng_free(rng);

    ptrdiff_t irel[3];
    for(i = pm->ORegion.start[0];
        i < pm->ORegion.start[0] + pm->ORegion.size[0];
        i ++) {

        gsl_rng * lower_rng = gsl_rng_alloc(gsl_rng_ranlxd1);
        gsl_rng * this_rng = gsl_rng_alloc(gsl_rng_ranlxd1);

        int ci = pm->Nmesh[0] - i;
        if(ci >= pm->Nmesh[0]) ci -= pm->Nmesh[0];

        for(j = pm->ORegion.start[1];
            j < pm->ORegion.start[1] + pm->ORegion.size[1];
            j ++) {
            /* always pull the gaussian from the lower quadrant plane for k = 0
             * plane*/
            /* always pull the whitenoise from the lower quadrant plane for k = 0
             * plane and k == All.Nmesh / 2 plane*/
            int d1 = 0, d2 = 0;
            int cj = pm->Nmesh[1] - j;
            if(cj >= pm->Nmesh[1]) cj -= pm->Nmesh[1];

            /* d1, d2 points to the conjugate quandrant */
            if( (ci == i && cj < j)
             || (ci < i && cj != j)
             || (ci < i && cj == j)) {
                d1 = 1;
                d2 = 1;
            }

            unsigned int seed_conj, seed_this;
            /* the lower quadrant generator */
            seed_conj = GETSEED(pm, seedtable, i, j, d1, d2);
            gsl_rng_set(lower_rng, seed_conj);

            seed_this = GETSEED(pm, seedtable, i, j, 0, 0);
            gsl_rng_set(this_rng, seed_this);

            for(k = 0; k <= pm->Nmesh[2] / 2; k ++) {
                int use_conj = (d1 != 0 || d2 != 0) && (k == 0 || k == pm->Nmesh[2] / 2);

                double ampl, phase;
                if(use_conj) {
                    /* on k = 0 and All.Nmesh/2 plane, we use the lower quadrant generator,
                     * then hermit transform the result if it is nessessary */
                    SAMPLE(this_rng, &ampl, &phase);
                    SAMPLE(lower_rng, &ampl, &phase);
                } else {
                    SAMPLE(lower_rng, &ampl, &phase);
                    SAMPLE(this_rng, &ampl, &phase);
                }

                ptrdiff_t iabs[3] = {i, j, k};
                ptrdiff_t ip = 0;
                for(d = 0; d < 3; d ++) {
                    irel[d] = iabs[d] - pm->ORegion.start[d];
                    ip += pm->ORegion.strides[d] * irel[d];
                }

                if(irel[2] < 0) continue;
                if(irel[2] >= pm->ORegion.size[2]) continue;

                /* we want two numbers that are of std ~ 1/sqrt(2) */
                ampl = sqrt(- log(ampl));

                if (setUnitaryAmplitude) ampl = 1.0; /* cos and sin gives 1/sqrt(2)*/


                if (setInvertPhase){
                  phase += M_PI; /*invert phase*/
                }

                (delta_k + 2 * ip)[0] = ampl * cos(phase);
                (delta_k + 2 * ip)[1] = ampl * sin(phase);

                if(use_conj) {
                    (delta_k + 2 * ip)[1] *= -1;
                }

                if((pm->Nmesh[0] - iabs[0]) % pm->Nmesh[0] == iabs[0] &&
                   (pm->Nmesh[1] - iabs[1]) % pm->Nmesh[1] == iabs[1] &&
                   (pm->Nmesh[2] - iabs[2]) % pm->Nmesh[2] == iabs[2]) {
                    /* The mode is self conjuguate, thus imaginary mode must be zero */
                    (delta_k + 2 * ip)[1] = 0;
                    (delta_k + 2 * ip)[0] = ampl * cos(phase);
                }

                if(iabs[0] == 0 && iabs[1] == 0 && iabs[2] == 0) {
                    /* the mean is zero */
                    (delta_k + 2 * ip)[0] = 0;
                    (delta_k + 2 * ip)[1] = 0;
                }
            }
        }
        gsl_rng_free(lower_rng);
        gsl_rng_free(this_rng);
    }
    for(i = 1; i >= 0; i --)
    for(j = 1; j >= 0; j --) {
        myfree(seedtable[i][j]);
    }
/*
    char * fn[1000];
    sprintf(fn, "canvas.dump.f4.%d", pm->ThisTask);
    fwrite(pm->canvas, sizeof(pm->canvas[0]), pm->ORegion.total * 2, fopen(fn, "w"));
*/
}
#endif
