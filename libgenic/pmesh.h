#ifndef PMESH_H
#define PMESH_H
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#include <libgadget/petapm.h>
#include <libgadget/utils.h>

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
SETSEED(PMDesc * pm, unsigned int * table[2][2], int i, int j, boost::random::mt19937 & rng)
{
    boost::random::uniform_real_distribution<double> dist(0, 1);
    unsigned int seed = static_cast<unsigned int>(0x7fffffff * dist(rng));

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
SAMPLE(boost::random::mt19937 & rng, double * ampl, double * phase)
{
    boost::random::uniform_real_distribution<double> dist(0, 1);
    *phase = dist(rng) * 2 * M_PI;
    *ampl = 0;
    do *ampl = dist(rng); while(*ampl == 0);
}

static void
pmic_fill_gaussian_gadget(PMDesc * pm, double * delta_k, int seed, int setUnitaryAmplitude, int setInvertPhase)
{
    /* Fill delta_k with gadget scheme */
    int d;
    int i, j, k;

    // Initialize the Boost RNG
    boost::random::mt19937 rng(seed);

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

    ptrdiff_t irel[3];
    for(i = pm->ORegion.start[0];
        i < pm->ORegion.start[0] + pm->ORegion.size[0];
        i ++) {

        boost::random::mt19937 lower_rng, this_rng;

        int ci = pm->Nmesh[0] - i;
        if(ci >= pm->Nmesh[0]) ci -= pm->Nmesh[0];

        for(j = pm->ORegion.start[1];
            j < pm->ORegion.start[1] + pm->ORegion.size[1];
            j ++) {
            int d1 = 0, d2 = 0;
            int cj = pm->Nmesh[1] - j;
            if(cj >= pm->Nmesh[1]) cj -= pm->Nmesh[1];

            if( (ci == i && cj < j)
             || (ci < i && cj != j)
             || (ci < i && cj == j)) {
                d1 = 1;
                d2 = 1;
            }

            unsigned int seed_conj, seed_this;
            seed_conj = GETSEED(pm, seedtable, i, j, d1, d2);
            lower_rng.seed(seed_conj);

            seed_this = GETSEED(pm, seedtable, i, j, 0, 0);
            this_rng.seed(seed_this);

            for(k = 0; k <= pm->Nmesh[2] / 2; k ++) {
                int use_conj = (d1 != 0 || d2 != 0) && (k == 0 || k == pm->Nmesh[2] / 2);

                double ampl, phase;
                if(use_conj) {
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

                ampl = sqrt(- log(ampl));

                if (setUnitaryAmplitude) ampl = 1.0;

                if (setInvertPhase){
                  phase += M_PI;
                }

                (delta_k + 2 * ip)[0] = ampl * cos(phase);
                (delta_k + 2 * ip)[1] = ampl * sin(phase);

                if(use_conj) {
                    (delta_k + 2 * ip)[1] *= -1;
                }

                if((pm->Nmesh[0] - iabs[0]) % pm->Nmesh[0] == iabs[0] &&
                   (pm->Nmesh[1] - iabs[1]) % pm->Nmesh[1] == iabs[1] &&
                   (pm->Nmesh[2] - iabs[2]) % pm->Nmesh[2] == iabs[2]) {
                    (delta_k + 2 * ip)[1] = 0;
                    (delta_k + 2 * ip)[0] = ampl * cos(phase);
                }

                if(iabs[0] == 0 && iabs[1] == 0 && iabs[2] == 0) {
                    (delta_k + 2 * ip)[0] = 0;
                    (delta_k + 2 * ip)[1] = 0;
                }
            }
        }
    }
    for(i = 1; i >= 0; i --)
    for(j = 1; j >= 0; j --) {
        myfree(seedtable[i][j]);
    }
}

#endif