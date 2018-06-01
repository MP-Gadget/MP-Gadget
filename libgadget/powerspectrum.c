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
void powerspectrum_alloc(struct _powerspectrum * PowerSpectrum, const int nbins, const int nthreads, const int MassiveNuLinResp)
{
    PowerSpectrum->size = nbins;
    const int nalloc = nbins*nthreads;
    PowerSpectrum->nalloc = nalloc;
    PowerSpectrum->kk = mymalloc("Powerspectrum", sizeof(double) * 2*nalloc);
    PowerSpectrum->Power = PowerSpectrum->kk + nalloc;
    if(MassiveNuLinResp) {
        /*These arrays are stored separately to make interpolation more accurate*/
        PowerSpectrum->logknu = mymalloc("PowerNu", sizeof(double) * 3*nbins);
        PowerSpectrum->delta_nu_ratio = PowerSpectrum-> logknu + nbins;
        PowerSpectrum->delta_nu = PowerSpectrum-> logknu + 2*nbins;
    }
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

    int nk_nz = 0;
    /*Now fix power spectrum units and remove zero entries.*/
    for(i = 0; i < PowerSpectrum->size; i ++) {
        if(PowerSpectrum->Nmodes[i] == 0) continue;
        PowerSpectrum->Power[i] /= PowerSpectrum->Nmodes[i];
        PowerSpectrum->Power[i] /= PowerSpectrum->Norm;
        PowerSpectrum->kk[i] /= PowerSpectrum->Nmodes[i];
        /* Mpc/h units */
        PowerSpectrum->kk[i] *= 2 * M_PI / (BoxSize_in_cm / CM_PER_MPC );
        PowerSpectrum->Power[i] *= pow(BoxSize_in_cm / CM_PER_MPC , 3.0);
        /*Move the power spectrum earlier, removing zero modes*/
        PowerSpectrum->Power[nk_nz] = PowerSpectrum->Power[i];
        PowerSpectrum->kk[nk_nz] = PowerSpectrum->kk[i];
        PowerSpectrum->Nmodes[nk_nz] = PowerSpectrum->Nmodes[i];
        nk_nz++;
    }
    PowerSpectrum->nonzero = nk_nz;
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
            for(i = 0; i < PowerSpectrum->nonzero; i ++) {
                fprintf(fp, "%g %g %ld %g\n", PowerSpectrum->kk[i], PowerSpectrum->Power[i], PowerSpectrum->Nmodes[i],
                            PowerSpectrum->Power[i] / (D1 * D1));
            }
            fclose(fp);
        }
}

/*Save the neutrino power spectrum to a file*/
void powerspectrum_nu_save(struct _powerspectrum * PowerSpectrum, const char * OutputDir, const double Time)
{
    int i;
    char fname[1024];
    /* Now save the neutrino power spectrum*/
    snprintf(fname, 1024,"%s/powerspectrum-nu-%0.4f.txt", OutputDir, Time);
    FILE * fp = fopen(fname, "w");
    fprintf(fp, "# in Mpc/h Units \n");
    fprintf(fp, "# k P_nu(k) Nmodes\n");
    fprintf(fp, "# a= %g\n", Time);
    fprintf(fp, "# nk = %ld\n", PowerSpectrum->nonzero);
    /* Pnuratio contains O_nu d_nu / (O_cdm d_cdm). Power contains P_t.
     * We have that O_0 d_t = O_cdm d_cdm (1 + Pnuratio)
     * so that d_nu = Pnuratio /O_nu (O_cdm d_cdm) = Pnuratio/(1+Pnuratio) d_t O_0/O_nu
     * and O_0 / O_nu = 1 + O_nonu/O_nu = 1 + 1/n_prefac*/
    for(i = 0; i < PowerSpectrum->nonzero; i++){
        fprintf(fp, "%g %g %ld\n", PowerSpectrum->kk[i], pow(PowerSpectrum->delta_nu[i],2), PowerSpectrum->Nmodes[i]);
    }
    fclose(fp);
    /*Clean up the neutrino memory now we saved the power spectrum.*/
    gsl_interp_free(PowerSpectrum->nu_spline);
    gsl_interp_accel_free(PowerSpectrum->nu_acc);
}
