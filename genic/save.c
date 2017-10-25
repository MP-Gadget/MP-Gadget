#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <stddef.h>
#include <string.h>
#include <mpi.h>
#include "genic/allvars.h"
#include "genic/proto.h"
#include "endrun.h"
#include "bigfile-mpi.h"
#include "cosmology.h"

#include "walltime.h"

static void saveblock(BigFile * bf, void * baseptr, int ptype, char * bname, char * dtype, int items_per_particle, int64_t TotNumPart) {
    BigBlock block;
    BigArray array;
    BigBlockPtr ptr;
    size_t dims[2];
    ptrdiff_t strides[2];
    char name[128];
    snprintf(name, 128, "%d/%s", ptype, bname);

    dims[0] = NumPart;
    dims[1] = items_per_particle;
    strides[1] = dtype_itemsize(dtype);
    strides[0] = sizeof(P[0]);
    big_array_init(&array, baseptr, dtype, 2, dims, strides);

    if(0 != big_file_mpi_create_block(bf, &block, name, dtype, dims[1], NumFiles, TotNumPart, MPI_COMM_WORLD)) {
        endrun(0, "%s:%s\n", big_file_get_error_message(), name);
    }

    if(0 != big_block_seek(&block, &ptr, 0)) {
        endrun(0, "Failed to seek:%s\n", big_file_get_error_message());
    }

    if(0 != big_block_mpi_write(&block, &ptr, &array, NumWriters, MPI_COMM_WORLD)) {
        endrun(0, "Failed to write :%s\n", big_file_get_error_message());
    }

    if(0 != big_block_mpi_close(&block, MPI_COMM_WORLD)) {
        endrun(0, "%s:%s\n", big_file_get_error_message(), name);
    }

}

void write_particle_data(int Type, BigFile * bf) {
    int64_t numpart_64 = NumPart, TotNumPart;
    MPI_Allreduce(&numpart_64, &TotNumPart, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);

    /* Write particles */
    saveblock(bf, &P[0].Density, Type, "ICDensity", "f4", 1, TotNumPart);
    saveblock(bf, &P[0].Pos, Type, "Position", "f8", 3, TotNumPart);
    saveblock(bf, &P[0].Vel, Type, "Velocity", "f4", 3, TotNumPart);
    saveblock(bf, &P[0].ID, Type, "ID", "u8", 1, TotNumPart);
    walltime_measure("/Write");
}

void saveheader(BigFile * bf, int64_t TotNumPart) {
    BigBlock bheader;
    if(0 != big_file_mpi_create_block(bf, &bheader, "Header", NULL, 0, 0, 0, MPI_COMM_WORLD)) {
        endrun(0, "failed to create block %s:%s", "Header",
                big_file_get_error_message());
    }

    int64_t totnumpart[6] = {0};
    double mass[6] = {0};
    totnumpart[1] = TotNumPart;
    double OmegaCDM = CP.Omega0;
    if (ProduceGas) {
        totnumpart[0] = TotNumPart;
        mass[0] = (CP.OmegaBaryon) * 3 * CP.Hubble * CP.Hubble / (8 * M_PI * G) * pow(Box, 3) / TotNumPart;
        OmegaCDM -= CP.OmegaBaryon;
    }
    if(CP.MNu[0] + CP.MNu[1] + CP.MNu[2] > 0)
        OmegaCDM -= get_omega_nu(&CP.ONu, 1);
    mass[1] = OmegaCDM * 3 * CP.Hubble * CP.Hubble / (8 * M_PI * G) * pow(Box, 3) / TotNumPart;
    double redshift = 1.0 / InitTime - 1.;

    int rt =(0 != big_block_set_attr(&bheader, "TotNumPart", totnumpart, "i8", 6)) ||
            (0 != big_block_set_attr(&bheader, "MassTable", mass, "f8", 6)) ||
            (big_block_set_attr(&bheader, "Time", &InitTime, "f8", 1)) ||
            (big_block_set_attr(&bheader, "Redshift", &redshift, "f8", 1)) ||
            (big_block_set_attr(&bheader, "BoxSize", &Box, "f8", 1)) ||
            (big_block_set_attr(&bheader, "UsePeculiarVelocity", &UsePeculiarVelocity, "i4", 1)) ||
            (big_block_set_attr(&bheader, "Omega0", &CP.Omega0, "f8", 1)) ||
            (big_block_set_attr(&bheader, "OmegaBaryon", &CP.OmegaBaryon, "f8", 1)) ||
            (big_block_set_attr(&bheader, "OmegaLambda", &CP.OmegaLambda, "f8", 1)) ||
            (big_block_set_attr(&bheader, "UnitLength_in_cm", &UnitLength_in_cm, "f8", 1)) ||
            (big_block_set_attr(&bheader, "UnitMass_in_g", &UnitMass_in_g, "f8", 1)) ||
            (big_block_set_attr(&bheader, "UnitVelocity_in_cm_per_s", &UnitVelocity_in_cm_per_s, "f8", 1)) ||
            (big_block_set_attr(&bheader, "HubbleParam", &CP.HubbleParam, "f8", 1));
    if(rt) {
        endrun(0, "failed to create attr %s", 
                big_file_get_error_message());
    }

    if(0 != big_block_mpi_close(&bheader, MPI_COMM_WORLD)) {
        endrun(0, "failed to close block %s:%s", "Header",
                    big_file_get_error_message());
    }
}
