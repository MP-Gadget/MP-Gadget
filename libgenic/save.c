#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <stddef.h>
#include <string.h>
#include <mpi.h>
#include "allvars.h"
#include "proto.h"
#include <bigfile-mpi.h>

#include <libgadget/types.h>
#include <libgadget/physconst.h>
#include <libgadget/utils.h>
#include <libgadget/cosmology.h>
#include <libgadget/walltime.h>
#include <libgadget/utils/mymalloc.h>

void _bigfile_utils_create_block_from_c_array(BigFile * bf, void * baseptr, const char * name, const char * dtype, size_t dims[], ptrdiff_t elsize, int NumFiles, int NumWriters, MPI_Comm comm)
{
    BigBlock block;
    BigArray array;
    BigBlockPtr ptr;
    ptrdiff_t strides[2];
    int64_t TotNumPart;
    MPI_Allreduce(&dims[0], &TotNumPart, 1, MPI_INT64, MPI_SUM, comm);

    strides[1] = dtype_itemsize(dtype);
    strides[0] = elsize;

    big_array_init(&array, baseptr, dtype, 2, dims, strides);

    if(0 != big_file_mpi_create_block(bf, &block, name, dtype, dims[1], NumFiles, TotNumPart, comm)) {
        endrun(0, "%s:%s\n", big_file_get_error_message(), name);
    }

    if(0 != big_block_seek(&block, &ptr, 0)) {
        endrun(0, "Failed to seek:%s\n", big_file_get_error_message());
    }

    if(0 != big_block_mpi_write(&block, &ptr, &array, NumWriters, comm)) {
        endrun(0, "Failed to write :%s\n", big_file_get_error_message());
    }

    if(0 != big_block_mpi_close(&block, comm)) {
        endrun(0, "%s:%s\n", big_file_get_error_message(), name);
    }
}

static void saveblock(BigFile * bf, void * baseptr, int ptype, const char * bname, const char * dtype, int items_per_particle, const int NumPart, ptrdiff_t elsize, int NumFiles, int NumWriters) {
    size_t dims[2];
    char name[128];
    snprintf(name, 128, "%d/%s", ptype, bname);

    dims[0] = NumPart;
    dims[1] = items_per_particle;
    _bigfile_utils_create_block_from_c_array(bf, baseptr, name, dtype, dims, elsize, NumFiles, NumWriters, MPI_COMM_WORLD);
}


void
write_particle_data(IDGenerator * idgen,
                    const int Type,
                    BigFile * bf,
                    const uint64_t FirstID,
                    const int SavePrePos,
                    int NumFiles, int NumWriters,
                    struct ic_part_data * curICP)
{
    /* Write particles */
    if(SavePrePos)
        saveblock(bf, &curICP[0].PrePos, Type, "PrePosition", "f8", 3, idgen->NumPart, sizeof(curICP[0]), NumFiles, NumWriters);
    saveblock(bf, &curICP[0].Density, Type, "ICDensity", "f4", 1, idgen->NumPart, sizeof(curICP[0]), NumFiles, NumWriters);
    saveblock(bf, &curICP[0].Pos, Type, "Position", "f8", 3, idgen->NumPart, sizeof(curICP[0]), NumFiles, NumWriters);
    saveblock(bf, &curICP[0].Vel, Type, "Velocity", "f4", 3, idgen->NumPart, sizeof(curICP[0]), NumFiles, NumWriters);
    /*Generate and write IDs*/
    uint64_t * ids = (uint64_t *) mymalloc("IDs", idgen->NumPart * sizeof(uint64_t));
    memset(ids, 0, idgen->NumPart * sizeof(uint64_t));
    int i;
    #pragma omp parallel for
    for(i = 0; i < idgen->NumPart; i++)
    {
        ids[i] = idgen_create_id_from_index(idgen, i) + FirstID;
    }
    saveblock(bf, ids, Type, "ID", "u8", 1, idgen->NumPart, sizeof(uint64_t), NumFiles, NumWriters);
    myfree(ids);
    walltime_measure("/Write");
}

/*Compute the mass array from the cosmology and the total number of particles.*/
void compute_mass(double * mass, int64_t TotNumPartCDM, int64_t TotNumPartGas, int64_t TotNuPart, double nufrac, const double BoxSize, Cosmology * CP, const struct genic_config GenicConfig)
{
    double UnitTime_in_s = GenicConfig.units.UnitLength_in_cm / GenicConfig.units.UnitVelocity_in_cm_per_s;
    double Grav = GRAVITY / pow(GenicConfig.units.UnitLength_in_cm, 3) * GenicConfig.units.UnitMass_in_g * pow(UnitTime_in_s, 2);
    const double OmegatoMass = 3 * CP->Hubble * CP->Hubble / (8 * M_PI * Grav) * pow(BoxSize, 3);
    double OmegaCDM = CP->Omega0;
    mass[0] = mass[2] = mass[3] = mass[4] = mass[5] = 0;
    if (TotNumPartGas > 0) {
        mass[0] = (CP->OmegaBaryon) * 3 * CP->Hubble * CP->Hubble / (8 * M_PI * Grav) * pow(BoxSize, 3) / TotNumPartGas;
        OmegaCDM -= CP->OmegaBaryon;
    }
    if(CP->MNu[0] + CP->MNu[1] + CP->MNu[2] > 0) {
        double OmegaNu = get_omega_nu(&CP->ONu, 1);
        OmegaCDM -= OmegaNu;
        if(TotNuPart > 0) {
            mass[2] = nufrac * OmegaNu * OmegatoMass / TotNuPart;
        }
    }
    mass[1] = OmegaCDM * OmegatoMass / TotNumPartCDM;
}


void saveheader(BigFile * bf, int64_t TotNumPartCDM, int64_t TotNumPartGas, int64_t TotNuPart, double nufrac, const double BoxSize, Cosmology * CP, const struct genic_config GenicConfig) {
    BigBlock bheader;
    if(0 != big_file_mpi_create_block(bf, &bheader, "Header", NULL, 0, 0, 0, MPI_COMM_WORLD)) {
        endrun(0, "failed to create block %s:%s\n", "Header",
                big_file_get_error_message());
    }

    int64_t totnumpart[6] = {TotNumPartGas, TotNumPartCDM, TotNuPart, 0, 0, 0};
    double mass[6] = {0};
    compute_mass(mass, TotNumPartCDM, TotNumPartGas, TotNuPart, nufrac, BoxSize, CP, GenicConfig);

    double redshift = 1.0 / GenicConfig.TimeIC - 1.;

    int rt =(0 != big_block_set_attr(&bheader, "TotNumPart", totnumpart, "i8", 6)) ||
            (0 != big_block_set_attr(&bheader, "MassTable", mass, "f8", 6)) ||
            (big_block_set_attr(&bheader, "Time", &GenicConfig.TimeIC, "f8", 1)) ||
            (big_block_set_attr(&bheader, "Redshift", &redshift, "f8", 1)) ||
            (big_block_set_attr(&bheader, "BoxSize", &BoxSize, "f8", 1)) ||
            (big_block_set_attr(&bheader, "UsePeculiarVelocity", &GenicConfig.UsePeculiarVelocity, "i4", 1)) ||
            (big_block_set_attr(&bheader, "Omega0", &CP->Omega0, "f8", 1)) ||
            (big_block_set_attr(&bheader, "FractionNuInParticles", &nufrac, "f8", 1)) ||
            (big_block_set_attr(&bheader, "OmegaBaryon", &CP->OmegaBaryon, "f8", 1)) ||
            (big_block_set_attr(&bheader, "OmegaLambda", &CP->OmegaLambda, "f8", 1)) ||
            (big_block_set_attr(&bheader, "UnitLength_in_cm", &GenicConfig.units.UnitLength_in_cm, "f8", 1)) ||
            (big_block_set_attr(&bheader, "UnitMass_in_g", &GenicConfig.units.UnitMass_in_g, "f8", 1)) ||
            (big_block_set_attr(&bheader, "UnitVelocity_in_cm_per_s", &GenicConfig.units.UnitVelocity_in_cm_per_s, "f8", 1)) ||
            (big_block_set_attr(&bheader, "HubbleParam", &CP->HubbleParam, "f8", 1)) ||
            (big_block_set_attr(&bheader, "InvertPhase", &GenicConfig.InvertPhase, "i4", 1)) ||
            (big_block_set_attr(&bheader, "Seed", &GenicConfig.Seed, "i8", 1)) ||
            (big_block_set_attr(&bheader, "UnitaryAmplitude", &GenicConfig.UnitaryAmplitude, "i4", 1));
    if(rt) {
        endrun(0, "failed to create attr %s",
                big_file_get_error_message());
    }

    if(0 != big_block_mpi_close(&bheader, MPI_COMM_WORLD)) {
        endrun(0, "failed to close block %s:%s", "Header",
                    big_file_get_error_message());
    }
}
