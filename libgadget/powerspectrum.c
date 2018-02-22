#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

#include "types.h"
#include "powerspectrum.h"

#include "utils.h"

/*Power spectrum related functions*/

/*Allocate memory for the power spectrum*/
void powerspectrum_alloc(struct _powerspectrum * PowerSpectrum, const int nbins, const int nthreads)
{
    PowerSpectrum->size = nbins;
    const int nalloc = nbins*nthreads;
    PowerSpectrum->nalloc = nalloc;
    PowerSpectrum->kk = mymalloc("Powerspectrum", sizeof(double) * (2*nalloc + 3*nbins));
    PowerSpectrum->Power = PowerSpectrum->kk + nalloc;
    PowerSpectrum->logknu = PowerSpectrum->kk + 2*nalloc;
    PowerSpectrum->Pnuratio = PowerSpectrum-> logknu + nbins;
    PowerSpectrum->Nmodes = mymalloc("Powermodes", sizeof(int64_t) * nalloc);
}

/*Zero memory for the power spectrum*/
void powerspectrum_zero(struct _powerspectrum * PowerSpectrum)
{
    memset(PowerSpectrum->kk, 0, sizeof(double) * PowerSpectrum->nalloc);
    memset(PowerSpectrum->Power, 0, sizeof(double) * PowerSpectrum->nalloc);
    memset(PowerSpectrum->Nmodes, 0, sizeof(double) * PowerSpectrum->nalloc);
    PowerSpectrum->Norm = 0;
}

/* Sum the different modes on each thread and processor together to get a power spectrum,
 * and fix the units.*/
void powerspectrum_sum(struct _powerspectrum * PowerSpectrum, const double BoxSize_in_cm)
{
    /*Sum power spectrum thread-local storage*/
    int i,j;
    for(i = 0; i < PowerSpectrum->size; i ++) {
        for(j = 1; j < PowerSpectrum->nalloc/PowerSpectrum->size; j++) {
            PowerSpectrum->Power[i] += PowerSpectrum->Power[i+ PowerSpectrum->size*j];
            PowerSpectrum->kk[i] += PowerSpectrum->kk[i+ PowerSpectrum->size*j];
            PowerSpectrum->Nmodes[i] += PowerSpectrum->Nmodes[i +PowerSpectrum->size*j];
        }
    }

    /*Now sum power spectrum MPI storage*/
    MPI_Allreduce(MPI_IN_PLACE, &(PowerSpectrum->Norm), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, PowerSpectrum->kk, PowerSpectrum->size, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, PowerSpectrum->Power, PowerSpectrum->size, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, PowerSpectrum->Nmodes, PowerSpectrum->size, MPI_INT64, MPI_SUM, MPI_COMM_WORLD);

    /*Now fix power spectrum units*/
    for(i = 0; i < PowerSpectrum->size; i ++) {
        if(PowerSpectrum->Nmodes[i] == 0) continue;
        PowerSpectrum->Power[i] /= PowerSpectrum->Nmodes[i];
        PowerSpectrum->Power[i] /= PowerSpectrum->Norm;
        PowerSpectrum->kk[i] /= PowerSpectrum->Nmodes[i];
        /* Mpc/h units */
        PowerSpectrum->kk[i] *= 2 * M_PI / (BoxSize_in_cm / 3.085678e24 );
        PowerSpectrum->Power[i] *= pow(BoxSize_in_cm / 3.085678e24 , 3.0);
    }
}

/*Save the power spectrum to a file*/
void powerspectrum_save(struct _powerspectrum * PowerSpectrum, const char * OutputDir, const double Time, const double D1)
{
        int i;
        char fname[1024];
        sprintf(fname, "%s/powerspectrum-%0.4f.txt", OutputDir, Time);
        message(1, "Writing Power Spectrum to %s\n", fname);
        FILE * fp = fopen(fname, "w");
        if(!fp)
            message(1, "Could not open %s for writing\n", fname);
        else {
            fprintf(fp, "# in Mpc/h Units \n");
            fprintf(fp, "# D1 = %g \n", D1);
            fprintf(fp, "# k P N P(z=0)\n");
            for(i = 0; i < PowerSpectrum->size; i ++) {
                if(PowerSpectrum->Nmodes[i] == 0) continue;
                fprintf(fp, "%g %g %ld %g\n", PowerSpectrum->kk[i], PowerSpectrum->Power[i], PowerSpectrum->Nmodes[i],
                            PowerSpectrum->Power[i] / (D1 * D1));
            }
            fclose(fp);
        }
}
