/**************
 * This file (C) Yu Feng 2017 implements support for a fluctuating UVB, using an external table.
 * This table is generated following the model explained in Battaglia & Trac 2010.
 ************** */

#include <mpi.h>
#include <string.h>
#include <math.h>
#include "cooling_rates.h"
#include "bigfile.h"
#include "utils/mymalloc.h"
#include "utils/interp.h"
#include "utils/endrun.h"

static struct {
    double UVRedshiftThreshold;
    int disabled;
    Interp interp;
    Interp Finterp;
    double * Table;
    ptrdiff_t Nside;
    double * Fraction;
    double * Zbins;
    int N_Zbins;
} UVF;

/*Global UVbackground stored to avoid extra interpolations.*/
struct UVBG GlobalUVBG = {0};
double GlobalUVRed = -1;

/*Sets the global variable corresponding to the uniform part of the UV background.*/
void
set_global_uvbg(double redshift)
{
    GlobalUVBG = get_global_UVBG(redshift);
    GlobalUVRed = redshift;
}

/* Read a big array from filename/dataset into an array, allocating memory in buffer.
 * which is returned. Nread argument is set equal to number of elements read.*/
static double *
read_big_array(const char * filename, char * dataset, int * Nread)
{
    int N;
    void * buffer=NULL;
    int ThisTask;
    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);

    if(ThisTask == 0) {
        BigFile bf[1];
        BigBlockPtr ptr;
        BigBlock bb[1];
        BigArray array[1];
        size_t dims[2];
        big_file_open(bf, filename);
        if(0 != big_file_open_block(bf, bb, dataset)) {
            endrun(1, "Cannot open %s %s: %s\n", filename, dataset, big_file_get_error_message());
        }

        N = bb->size;

        buffer = mymalloc("cooling_data", N * dtype_itemsize(bb->dtype) * bb->nmemb);

        dims[0] = N;
        dims[1] = bb->nmemb;

        big_array_init(array, buffer, bb->dtype, 2, dims, NULL);
        if(0 != big_block_seek(bb, &ptr, 0))
            endrun(1, "Failed to seek block %s %s: %s\n", filename, dataset, big_file_get_error_message());

        if(0 != big_block_read(bb, &ptr, array))
            endrun(1, "Failed to read %s %s: %s", filename, dataset, big_file_get_error_message());
        /* steal the buffer */
        big_block_close(bb);
        big_file_close(bf);
    }

    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if(ThisTask != 0)
        buffer = mymalloc("cooling_data",N * sizeof(double));

    MPI_Bcast(buffer, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    *Nread = N;
    return buffer;
}

/* The UV fluctation file is a bigfile with these tables:
 * ReionizedFraction: values of the reionized fraction as function of
 * redshift.
 * Redshift_Bins: uniform redshifts of the reionized fraction values
 *
 * XYZ_Bins: the uniform XYZ points where Z_reion is tabulated. (length of Nside)
 *
 * Zreion_Table: a Nside (X) x Nside (Y)x Nside (z) C ordering double array,
 * the reionization redshift as function of space, on a grid give by
 * XYZ_Bins.
 *
 * Notice that this table is broadcast to all MPI ranks, thus it can't be
 * too big. (400x400x400 is around 400 MBytes)
 *
 * */
void
init_uvf_table(const char * UVFluctuationFile, double UVRedshiftThreshold)
{
    if(strlen(UVFluctuationFile) == 0) {
        UVF.disabled = 1;
        return;
    }

    UVF.UVRedshiftThreshold = UVRedshiftThreshold;

    message(0, "Using NON-UNIFORM UV BG fluctuations from %s\n", UVFluctuationFile);
    UVF.disabled = 0;

    {
        /* read the reionized fraction */
        UVF.Zbins = read_big_array(UVFluctuationFile, "Redshift_Bins", &UVF.N_Zbins);
        UVF.Fraction = read_big_array(UVFluctuationFile, "ReionizedFraction", &UVF.N_Zbins);
        int dims[] = {UVF.N_Zbins};
        interp_init(&UVF.Finterp, 1, dims);
        interp_init_dim(&UVF.Finterp, 0, UVF.Zbins[0], UVF.Zbins[UVF.N_Zbins - 1]);
    }

    int Nside;
    double * XYZ_Bins = read_big_array(UVFluctuationFile, "XYZ_Bins", &Nside);
    int dims[] = {Nside, Nside, Nside};
    interp_init(&UVF.interp, 3, dims);
    interp_init_dim(&UVF.interp, 0, XYZ_Bins[0], XYZ_Bins[Nside - 1]);
    interp_init_dim(&UVF.interp, 1, XYZ_Bins[0], XYZ_Bins[Nside - 1]);
    interp_init_dim(&UVF.interp, 2, XYZ_Bins[0], XYZ_Bins[Nside - 1]);
    myfree(XYZ_Bins);
    UVF.Nside = Nside;

    int size;
    UVF.Table = read_big_array(UVFluctuationFile, "Zreion_Table", &size);
    if(UVF.Table[0] < 0.01 || UVF.Table[0] > 100.0) {
        endrun(123, "UV Fluctuation out of range: %g\n", UVF.Table[0]);
    }
}

#if 0
/* Fraction of total universe that is ionized.
 * currently unused. Unclear if the UVBG in Treecool shall be adjusted
 * by the factor or not. seems to be NOT after reading Giguere's paper.
 * */
static double GetReionizedFraction(double time) {
    if(UVF.disabled) {
        return 1.0;
    }
    int status[1];
    double redshift = 1 / time - 1;
    double x[] = {redshift};
    double fraction = interp_eval(&UVF.Finterp, x, UVF.Fraction, status);
    if(status[0] < 0) return 0.0;
    if(status[0] > 0) return 1.0;
    return fraction;
}

#endif

/*
 * returns the spatial dependent UVBG if UV fluctuation is enabled.
 * Otherwise returns the global UVBG passed in.
 *
 * */
struct UVBG get_particle_UVBG(double redshift, double * Pos)
{
    if(fabs(redshift - GlobalUVRed) > 1e-4)
        endrun(1, "Called with redshift %g not %g expected by the UVBG cache.\n", redshift, GlobalUVRed);
    struct UVBG uvbg = {0};
    /* if a threshold is set, disable UV bg above that redshift */
    if(UVF.UVRedshiftThreshold >= 0.0 && redshift > UVF.UVRedshiftThreshold) {
        return uvbg;
    }

    if(UVF.disabled) {
        /* directly use the TREECOOL table if UVF is disabled */
        memcpy(&uvbg, &GlobalUVBG, sizeof(struct UVBG));
        return uvbg;
    }

    double zreion = interp_eval_periodic(&UVF.interp, Pos, UVF.Table);
    if(zreion < redshift) {
        return uvbg;
    }

    memcpy(&uvbg, &GlobalUVBG, sizeof(struct UVBG));
    return uvbg;
}


/*Here comes the Metal Cooling code*/
struct {
    int CoolingNoMetal;
    int NRedshift_bins;
    double * Redshift_bins;

    int NHydrogenNumberDensity_bins;
    double * HydrogenNumberDensity_bins;

    int NTemperature_bins;
    double * Temperature_bins;

    double * Lmet_table; /* metal cooling @ one solar metalicity*/

    Interp interp;
} MetalCool;

void
InitMetalCooling(const char * MetalCoolFile)
{
    /* now initialize the metal cooling table from cloudy; we got this file
     * from vogelsberger's Arepo simulations; it is supposed to be
     * cloudy + UVB - H and He; look so.
     * the table contains only 1 Z_sun values. Need to be scaled to the
     * metallicity.
     *
     * */
    /* let's see if the Metal Cool File is magic NoMetal */
    if(strlen(MetalCoolFile) == 0) {
        MetalCool.CoolingNoMetal = 1;
        return;
    } else {
        MetalCool.CoolingNoMetal = 0;
    }

    int size;
    //This is never used if MetalCoolFile == ""
    double * tabbedmet = read_big_array(MetalCoolFile, "MetallicityInSolar_bins", &size);

    if(size != 1 || tabbedmet[0] != 0.0) {
        endrun(123, "MetalCool file %s is wrongly tabulated\n", MetalCoolFile);
    }
    myfree(tabbedmet);

    MetalCool.Redshift_bins = read_big_array(MetalCoolFile, "Redshift_bins", &MetalCool.NRedshift_bins);
    MetalCool.HydrogenNumberDensity_bins = read_big_array(MetalCoolFile, "HydrogenNumberDensity_bins", &MetalCool.NHydrogenNumberDensity_bins);
    MetalCool.Temperature_bins = read_big_array(MetalCoolFile, "Temperature_bins", &MetalCool.NTemperature_bins);
    MetalCool.Lmet_table = read_big_array(MetalCoolFile, "NetCoolingRate", &size);

    int dims[] = {MetalCool.NRedshift_bins, MetalCool.NHydrogenNumberDensity_bins, MetalCool.NTemperature_bins};

    interp_init(&MetalCool.interp, 3, dims);
    interp_init_dim(&MetalCool.interp, 0, MetalCool.Redshift_bins[0], MetalCool.Redshift_bins[MetalCool.NRedshift_bins - 1]);
    interp_init_dim(&MetalCool.interp, 1, MetalCool.HydrogenNumberDensity_bins[0],
                    MetalCool.HydrogenNumberDensity_bins[MetalCool.NHydrogenNumberDensity_bins - 1]);
    interp_init_dim(&MetalCool.interp, 2, MetalCool.Temperature_bins[0],
                    MetalCool.Temperature_bins[MetalCool.NTemperature_bins - 1]);
}

double
TableMetalCoolingRate(double redshift, double temp, double nHcgs)
{
    if(MetalCool.CoolingNoMetal)
        return 0;

    double lognH = log10(nHcgs);
    double logT = log10(temp);

    double x[] = {redshift, lognH, logT};
    int status[3];
    double rate = interp_eval(&MetalCool.interp, x, MetalCool.Lmet_table, status);
    /* XXX: in case of very hot / very dense we just use whatever the table says at
     * the limit. should be OK. */
    return rate;
}
