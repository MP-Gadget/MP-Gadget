/**************
 * This file (C) Yu Feng 2017 implements support for a fluctuating UVB, using an external table.
 * This table is generated following the model explained in Battaglia & Trac 2010.
 ************** */

#include <mpi.h>
#include <string.h>
#include <math.h>
#include "cooling_rates.h"
#include "physconst.h"
#include "bigfile.h"
#include "bigfile-mpi.h"
#include "utils/mymalloc.h"
#include "utils/interp.h"
#include "utils/endrun.h"

static struct {
    int enabled;
    Interp interp;
    double * Table;
    ptrdiff_t Nside;
} UVF;

/*Global UVbackground stored to avoid extra interpolations.*/
static struct UVBG GlobalUVBG = {0};
static double GlobalUVRed = -1;

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
init_uvf_table(const char * UVFluctuationFile, const double BoxSize, const double UnitLength_in_cm)
{
    if(strlen(UVFluctuationFile) == 0) {
        UVF.enabled = 0;
        return;
    }

    /* Open and validate the UV fluctuation file*/
    BigFile bf;
    BigBlock bh;
    if(0 != big_file_mpi_open(&bf, UVFluctuationFile, MPI_COMM_WORLD)) {
        endrun(0, "Failed to open snapshot at %s:%s\n", UVFluctuationFile,
                    big_file_get_error_message());
    }

    if(0 != big_file_mpi_open_block(&bf, &bh, "Zreion_Table", MPI_COMM_WORLD)) {
        endrun(0, "Failed to create block at %s:%s\n", "Header",
                    big_file_get_error_message());
    }
    double TableBoxSize;
    double ReionRedshift;
    if ((0 != big_block_get_attr(&bh, "Nmesh", &UVF.Nside, "u8", 1)) ||
        (0 != big_block_get_attr(&bh, "BoxSize", &TableBoxSize, "f8", 1)) ||
        (0 != big_block_get_attr(&bh, "Redshift", &ReionRedshift, "f8", 1)) ||
        (0 != big_block_mpi_close(&bh, MPI_COMM_WORLD))) {
        endrun(0, "Failed to close block: %s\n",
                    big_file_get_error_message());
    }
    big_file_mpi_close(&bf, MPI_COMM_WORLD);
    double BoxMpc = BoxSize * UnitLength_in_cm / CM_PER_MPC;
    if(TableBoxSize != BoxMpc)
        endrun(0, "Wrong UV fluctuation file! %s is for box size %g Mpc/h, but current box is %g Mpc/h\n", TableBoxSize, BoxMpc);

    message(0, "Using NON-UNIFORM UV BG fluctuations from %s. Median reionization redshift is %g\n", UVFluctuationFile, ReionRedshift);
    UVF.enabled = 1;

    int size;
    UVF.Table = read_big_array(UVFluctuationFile, "Zreion_Table", &size);

    if(UVF.Nside * UVF.Nside * UVF.Nside != size)
        endrun(0, "Corrupt UV Fluctuation table: Nside = %ld, but table is %ld != %ld^3\n", UVF.Nside, size, UVF.Nside);

    int dims[] = {UVF.Nside, UVF.Nside, UVF.Nside};
    interp_init(&UVF.interp, 3, dims);
    interp_init_dim(&UVF.interp, 0, 0, BoxSize);
    interp_init_dim(&UVF.interp, 1, 0, BoxSize);
    interp_init_dim(&UVF.interp, 2, 0, BoxSize);

    if(UVF.Table[0] < 0.01 || UVF.Table[0] > 100.0) {
        endrun(0, "UV Fluctuation out of range: %g\n", UVF.Table[0]);
    }
}

/*
 * returns the spatial dependent UVBG if UV fluctuation is enabled.
 * Otherwise returns the global UVBG passed in.
 *
 * */
struct UVBG get_local_UVBG(double redshift, double * Pos, const double * PosOffset)
{
    if(fabs(redshift - GlobalUVRed) > 1e-4)
        endrun(1, "Called with redshift %g not %g expected by the UVBG cache.\n", redshift, GlobalUVRed);

    if(!UVF.enabled) {
        /* directly use the TREECOOL table if UVF is disabled */
        return GlobalUVBG;
    }

    struct UVBG uvbg = {0};

    uvbg.self_shield_dens = GlobalUVBG.self_shield_dens;

    double corrpos[3];
    int i;
    for(i = 0; i < 3; i++)
        corrpos[i] = Pos[i] - PosOffset[i];
    double zreion = interp_eval_periodic(&UVF.interp, corrpos, UVF.Table);
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
