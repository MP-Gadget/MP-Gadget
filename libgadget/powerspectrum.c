#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

#include "types.h"
#include "powerspectrum.h"
#include "physconst.h"
#include "utils.h"

/*Power spectrum related functions*/

/*Allocate memory for the power spectrum*/
void powerspectrum_alloc(Power * ps, const int nbins, const int nthreads, const int MassiveNuLinResp, const double BoxSize_in_cm)
{
    ps->size = nbins;
    const int nalloc = nbins*nthreads;
    ps->nalloc = nalloc;
    ps->kk = (double *) mymalloc("Powerspectrum", sizeof(double) * 2*nalloc);
    ps->Power = ps->kk + nalloc;
    ps->BoxSize_in_MPC = BoxSize_in_cm / CM_PER_MPC;
    ps->logknu = NULL;
    if(MassiveNuLinResp) {
        /*These arrays are stored separately to make interpolation more accurate*/
        ps->logknu = (double *) mymalloc("PowerNu", sizeof(double) * 2*nbins);
        ps->delta_nu_ratio = ps-> logknu + nbins;
    }
    ps->Nmodes = (int64_t *) mymalloc("Powermodes", sizeof(int64_t) * nalloc);
    powerspectrum_zero(ps);
}

/*Zero memory for the power spectrum*/
void powerspectrum_zero(Power * ps)
{
    memset(ps->kk, 0, sizeof(double) * ps->nalloc);
    memset(ps->Power, 0, sizeof(double) * ps->nalloc);
    memset(ps->Nmodes, 0, sizeof(int64_t) * ps->nalloc);
    ps->Norm = 0;
}

/*Free power spectrum memory*/
void powerspectrum_free(Power * ps)
{
    myfree(ps->Nmodes);
    if(ps->logknu)
        myfree(ps->logknu);
    myfree(ps->kk);
}

/* Sum the different modes on each thread and processor together to get a power spectrum,
 * and fix the units. */
void powerspectrum_sum(Power * ps)
{
    /*Sum power spectrum thread-local storage*/
    int i,j;
    for(i = 0; i < ps->size; i ++) {
        for(j = 1; j < ps->nalloc/ps->size; j++) {
            ps->Power[i] += ps->Power[i+ ps->size*j];
            ps->kk[i] += ps->kk[i+ ps->size*j];
            ps->Nmodes[i] += ps->Nmodes[i +ps->size*j];
        }
    }

    /*Now sum power spectrum MPI storage*/
    MPI_Allreduce(MPI_IN_PLACE, &(ps->Norm), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, ps->kk, ps->size, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, ps->Power, ps->size, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, ps->Nmodes, ps->size, MPI_INT64, MPI_SUM, MPI_COMM_WORLD);

    int nk_nz = 0;
    /*Now fix power spectrum units and remove zero entries.*/
    for(i = 0; i < ps->size; i ++) {
        if(ps->Nmodes[i] == 0) continue;
        ps->Power[i] /= ps->Nmodes[i];
        ps->Power[i] /= ps->Norm;
        ps->kk[i] /= ps->Nmodes[i];
        /* Mpc/h units */
        ps->kk[i] *= 2 * M_PI / (ps->BoxSize_in_MPC);
        ps->Power[i] *= pow(ps->BoxSize_in_MPC , 3.0);
        /*Move the power spectrum earlier, removing zero modes*/
        ps->Power[nk_nz] = ps->Power[i];
        ps->kk[nk_nz] = ps->kk[i];
        ps->Nmodes[nk_nz] = ps->Nmodes[i];
        nk_nz++;
    }
    ps->nonzero = nk_nz;
}

/*Save the power spectrum to a file*/
void powerspectrum_save(Power * ps, const char * OutputDir, const char * filename, const double Time, const double D1)
{
        int i;
        int ThisTask;
        MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
        if(ThisTask != 0)
            return;
        char * fname = fastpm_strdup_printf("%s/%s-%0.4f.txt", OutputDir, filename, Time);
        /* Avoid -0.0000.txt at high z*/
        if(Time <= 1e-4) {
            myfree(fname);
            fname = fastpm_strdup_printf("%s/%s-%0.4e.txt", OutputDir, filename, Time);
        }
        message(1, "Writing Power Spectrum to %s\n", fname);
        FILE * fp = fopen(fname, "w");
        if(!fp)
            message(1, "Could not open %s for writing\n", fname);
        else {
            fprintf(fp, "# in Mpc/h Units \n");
            fprintf(fp, "# D1 = %g \n", D1);
            fprintf(fp, "# k P N P(z=0)\n");
            for(i = 0; i < ps->nonzero; i ++) {
                fprintf(fp, "%g %g %ld %g\n", ps->kk[i], ps->Power[i], ps->Nmodes[i],
                            ps->Power[i] / (D1 * D1));
            }
            fclose(fp);
        }
        myfree(fname);
}
