#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <string.h>
#include <stdarg.h>
#include <omp.h>

#include <bigfile-mpi.h>

#include "sfr_eff.h"
#include "cooling.h"
#include "timestep.h"

#include "petaio.h"
#include "slotsmanager.h"
#include "hydra.h"
#include "density.h"
#include "partmanager.h"
#include "config.h"
#include "neutrinos_lra.h"

#include "utils.h"
/************
 *
 * The IO api , intented to replace io.c and read_ic.c
 * currently we have a function to register the blocks and enumerate the blocks
 *
 */
static struct petaio_params {
    size_t BytesPerFile;   /* Number of bytes per physical file; this decides how many files bigfile creates each block */
    int WritersPerFile;    /* Number of concurrent writers per file; this decides number of writers */
    int NumWriters;        /* Number of concurrent writers, this caps number of writers */
    int MinNumWriters;        /* Min Number of concurrent writers, this caps number of writers */
    int EnableAggregatedIO;  /* Enable aggregated IO policy for small files.*/
    size_t AggregatedIOThreshold; /* bytes per writer above which to use non-aggregated IO (avoid OOM)*/
    /* Changes the comoving factors of the snapshot outputs. Set in the ICs.
     * If UsePeculiarVelocity = 1 then snapshots save to the velocity field the physical peculiar velocity, v = a dx/dt (where x is comoving distance).
     * If UsePeculiarVelocity = 0 then the velocity field is a * v = a^2 dx/dt in snapshots
     * and v / sqrt(a) = sqrt(a) dx/dt in the ICs. Note that snapshots never match Gadget-2, which
     * saves physical peculiar velocity / sqrt(a) in both ICs and snapshots. */
    int UsePeculiarVelocity;
    int OutputPotential;        /*!< Flag whether to include the potential in snapshots*/
    int OutputHeliumFractions;  /*!< Flag whether to output the helium ionic fractions in snapshots*/
    int OutputTimebins;         /* Flag whether to save the timebins*/
    char SnapshotFileBase[100]; /* Snapshots are written to OutputDir/SnapshotFileBase_$n*/
    char InitCondFile[100]; /* Path to read ICs from is InitCondFile */

    int ExcursionSetReionOn;

} IO;

/* Struct to store constant information written to each snapshot header*/
static struct header_data Header;

/*Set the IO parameters*/
void
set_petaio_params(ParameterSet * ps)
{
    int ThisTask;
    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
    if(ThisTask == 0) {
        IO.BytesPerFile = param_get_int(ps, "BytesPerFile");
        IO.UsePeculiarVelocity = 0; /* Will be set by the Initial Condition File */
        IO.NumWriters = param_get_int(ps, "NumWriters");
        IO.MinNumWriters = param_get_int(ps, "MinNumWriters");
        IO.WritersPerFile = param_get_int(ps, "WritersPerFile");
        IO.AggregatedIOThreshold = param_get_int(ps, "AggregatedIOThreshold");
        /* Convert from MB to bytes*/
        IO.AggregatedIOThreshold *= 1024L * 1024L;
        IO.EnableAggregatedIO = param_get_int(ps, "EnableAggregatedIO");
        IO.OutputPotential = param_get_int(ps, "OutputPotential");
        IO.OutputTimebins = param_get_int(ps, "OutputTimebins");
        IO.OutputHeliumFractions = param_get_int(ps, "OutputHeliumFractions");
        param_get_string2(ps, "SnapshotFileBase", IO.SnapshotFileBase, sizeof(IO.SnapshotFileBase));
        param_get_string2(ps, "InitCondFile", IO.InitCondFile, sizeof(IO.InitCondFile));
        IO.ExcursionSetReionOn = param_get_int(ps,"ExcursionSetReionOn");
    }
    MPI_Bcast(&IO, sizeof(struct petaio_params), MPI_BYTE, 0, MPI_COMM_WORLD);
}

int GetUsePeculiarVelocity(void)
{
    return IO.UsePeculiarVelocity;
}

static void petaio_write_header(BigFile * bf, const double atime, const int64_t * NTotal, const Cosmology * CP, const struct header_data * data);
static void petaio_read_header_internal(BigFile * bf, Cosmology * CP, struct header_data * data);

/* these are only used in reading in */
void petaio_init(void) {
    /* Smaller files will do aggregated IO.*/
    if(IO.EnableAggregatedIO) {
        message(0, "Aggregated IO is enabled\n");
        big_file_mpi_set_aggregated_threshold(IO.AggregatedIOThreshold);
    } else {
        message(0, "Aggregated IO is disabled.\n");
        big_file_mpi_set_aggregated_threshold(0);
    }
    if(IO.NumWriters == 0)
        MPI_Comm_size(MPI_COMM_WORLD, &IO.NumWriters);
}

/* Build a list of the first particle of each type on the current processor.
 * This assumes that all particles are sorted!*/
/**
 * Create a Selection array for the buffers. This array indirectly sort
 * the particles by the type.
 *
 * The offset for the starting of each type is stored in ptype_offset.
 *
 * if select_func is provided, it shall return 1 for those that shall be
 * included in the output.
 */
void
petaio_build_selection(int * selection,
    int64_t * ptype_offset,
    int64_t * ptype_count,
    const struct particle_data * Parts,
    const int64_t NumPart,
    int (*select_func)(int i, const struct particle_data * Parts)
    )
{
    int64_t i;

    #pragma omp parallel for reduction(+: ptype_count[:6])
    for(i = 0; i < NumPart; i ++) {
        if(P[i].IsGarbage)
            continue;
        if((select_func == NULL) || (select_func(i, Parts) != 0)) {
            int ptype = Parts[i].Type;
            ptype_count[ptype] ++;
        }
    }

    ptype_offset[0] = 0;
    for(i = 1; i < 6; i ++) {
        ptype_offset[i] = ptype_offset[i-1] + ptype_count[i-1];
        ptype_count[i-1] = 0;
    }

    ptype_count[5] = 0;
    for(i = 0; i < NumPart; i ++) {
        int ptype = Parts[i].Type;
        if(P[i].IsGarbage)
            continue;
        if((select_func == NULL) || (select_func(i, Parts) != 0)) {
            selection[ptype_offset[ptype] + ptype_count[ptype]] = i;
            ptype_count[ptype]++;
        }
    }
}

void
petaio_save_snapshot(const char * fname, struct IOTable * IOTable, int verbose, const double atime, const Cosmology * CP)
{
    message(0, "saving snapshot into %s\n", fname);

    BigFile bf = {0};
    if(0 != big_file_mpi_create(&bf, fname, MPI_COMM_WORLD)) {
        endrun(0, "Failed to create snapshot at %s:%s\n", fname,
                    big_file_get_error_message());
    }

    int64_t ptype_offset[6]={0};
    int64_t ptype_count[6]={0};
    int64_t NTotal[6]={0};

    int * selection = (int *) mymalloc("Selection", sizeof(int) * PartManager->NumPart);

    petaio_build_selection(selection, ptype_offset, ptype_count, P, PartManager->NumPart, NULL);

    MPI_Allreduce(ptype_count, NTotal, 6, MPI_INT64, MPI_SUM, MPI_COMM_WORLD);
    struct conversions conv = {0};
    conv.atime = atime;
    conv.hubble = hubble_function(CP, atime);

    petaio_write_header(&bf, atime, NTotal, CP, &Header);

    int i;
    for(i = 0; i < IOTable->used; i ++) {
        /* only process the particle blocks */
        char blockname[128];
        int ptype = IOTable->ent[i].ptype;
        BigArray array = {0};
        /*This exclude FOF blocks*/
        if(!(ptype < 6 && ptype >= 0)) {
            continue;
        }
        /* No need to write empty folders for particle types we don't have.
         * But do still write them for stars and BHs as someone might expect them.*/
        if(ptype_count[ptype] == 0 && ptype < 4)
            continue;
        sprintf(blockname, "%d/%s", ptype, IOTable->ent[i].name);
        petaio_build_buffer(&array, &IOTable->ent[i], selection + ptype_offset[ptype], ptype_count[ptype], P, SlotsManager, &conv);
        petaio_save_block(&bf, blockname, &array, verbose);
        petaio_destroy_buffer(&array);
    }

    if(CP->MassiveNuLinRespOn) {
        int ThisTask;
        MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
        petaio_save_neutrinos(&bf, ThisTask);
    }
    if(0 != big_file_mpi_close(&bf, MPI_COMM_WORLD)){
        endrun(0, "Failed to close snapshot at %s:%s\n", fname,
                    big_file_get_error_message());
    }

    MPI_Barrier(MPI_COMM_WORLD);
    message(0, "Finished saving snapshot into %s\n", fname);
    myfree(selection);
}

char *
petaio_get_snapshot_fname(int num, const char * OutputDir)
{
    char * fname;
    if(num == -1) {
        fname = fastpm_strdup_printf("%s", IO.InitCondFile);
    } else {
        fname = fastpm_strdup_printf("%s/%s_%03d", OutputDir, IO.SnapshotFileBase, num);
    }
    return fname;
}

struct header_data
    petaio_read_header(int num, const char * OutputDir, Cosmology * CP)
{
    BigFile bf = {0};

    char * fname = petaio_get_snapshot_fname(num, OutputDir);
    message(0, "Probing Header of snapshot file: %s\n", fname);

    if(0 != big_file_mpi_open(&bf, fname, MPI_COMM_WORLD)) {
        endrun(0, "Failed to open snapshot at %s:%s\n", fname,
                    big_file_get_error_message());
    }

    struct header_data head;
    petaio_read_header_internal(&bf, CP, &head);
    head.neutrinonk = -1;
    /* Try to read the neutrino header data from the snapshot.
     * If this fails then neutrinonk will be zero.*/
    if(num >= 0) {
        BigBlock bn;
        if(0 == big_file_mpi_open_block(&bf, &bn, "Neutrino", MPI_COMM_WORLD)) {
            if(0 != big_block_get_attr(&bn, "Nkval", &head.neutrinonk, "u8", 1))
                endrun(0, "Failed to read attr: %s\n", big_file_get_error_message());
            big_block_mpi_close(&bn, MPI_COMM_WORLD);
        }
    }

    if(0 != big_file_mpi_close(&bf, MPI_COMM_WORLD)) {
        endrun(0, "Failed to close snapshot at %s:%s\n", fname,
                    big_file_get_error_message());
    }
    myfree(fname);
    Header = head;
    return head;
}

void
petaio_read_snapshot(int num, const char * OutputDir, Cosmology * CP, struct header_data * header, struct part_manager_type * PartManager, struct slots_manager_type * SlotsManager, MPI_Comm Comm)
{
    char * fname = petaio_get_snapshot_fname(num, OutputDir);
    int i;
    const int ic = (num == -1);
    BigFile bf = {0};
    message(0, "Reading snapshot %s\n", fname);

    if(0 != big_file_mpi_open(&bf, fname, Comm)) {
        endrun(0, "Failed to open snapshot at %s:%s\n", fname,
                    big_file_get_error_message());
    }

    /*Read neutrinos from the snapshot if necessary*/
    if(CP->MassiveNuLinRespOn) {
        int ThisTask;
        MPI_Comm_rank(Comm, &ThisTask);
        /*Read the neutrino transfer function from the ICs: init_neutrinos_lra should have been called before this!*/
        if(ic)
            petaio_read_icnutransfer(&bf, ThisTask);
        else
            petaio_read_neutrinos(&bf, ThisTask);
    }

    struct conversions conv = {0};
    conv.atime = header->TimeSnapshot;
    conv.hubble = hubble_function(CP, header->TimeSnapshot);

    struct IOTable IOTable[1] = {0};
    /* Always try to read the metal tables.
     * This lets us turn it off for a short period and then re-enable it.
     * Note the metal fields are non-fatal so this does not break resuming without metals.*/
    register_io_blocks(IOTable, 0, 1);

    for(i = 0; i < IOTable->used; i ++) {
        /* only process the particle blocks */
        char blockname[128];
        int ptype = IOTable->ent[i].ptype;
        BigArray array = {0};
        if(!(ptype < 6 && ptype >= 0)) {
            continue;
        }
        if(header->NTotal[ptype] == 0) continue;
        if(ic) {
            /* for IC read in only three blocks */
            int keep = 0;
            keep |= (0 == strcmp(IOTable->ent[i].name, "Position"));
            keep |= (0 == strcmp(IOTable->ent[i].name, "Velocity"));
            keep |= (0 == strcmp(IOTable->ent[i].name, "ID"));
            if (ptype == 5) {
                keep |= (0 == strcmp(IOTable->ent[i].name, "Mass"));
                keep |= (0 == strcmp(IOTable->ent[i].name, "BlackholeMass"));
                keep |= (0 == strcmp(IOTable->ent[i].name, "MinPotPos"));
            }
            if(!keep) continue;
        }
        if(IOTable->ent[i].setter == NULL) {
            /* FIXME: do not know how to read this block; assume the fucker is
             * internally intialized; */
            continue;
        }
        sprintf(blockname, "%d/%s", ptype, IOTable->ent[i].name);
        petaio_alloc_buffer(&array, &IOTable->ent[i], header->NLocal[ptype]);
        if(0 == petaio_read_block(&bf, blockname, &array, IOTable->ent[i].required))
            petaio_readout_buffer(&array, &IOTable->ent[i], &conv, PartManager, SlotsManager);
        petaio_destroy_buffer(&array);
    }
    destroy_io_blocks(IOTable);

    if(0 != big_file_mpi_close(&bf, Comm)) {
        endrun(0, "Failed to close snapshot at %s:%s\n", fname,
                    big_file_get_error_message());
    }
    /* now we have IDs, set up the ID consistency between slots. */
    slots_setup_id(PartManager, SlotsManager);
    myfree(fname);

    if(ic) {
        /*
         *  IC doesn't have entropy or energy; always use the
         *  InitTemp in paramfile, then use init.c to convert to
         *  entropy.
         * */
        struct particle_data * parts = PartManager->Base;
        int i;
        /* touch up the mass -- IC files save mass in header */
        #pragma omp parallel for
        for(i = 0; i < PartManager->NumPart; i++)
        {
            parts[i].Mass = header->MassTable[parts[i].Type];
        }

        if (!IO.UsePeculiarVelocity ) {
            /* fixing the unit of velocity from Legacy GenIC IC */
            #pragma omp parallel for
            for(i = 0; i < PartManager->NumPart; i++) {
                int k;
                /* for GenIC's Gadget-1 snapshot Unit to Gadget-2 Internal velocity unit */
                for(k = 0; k < 3; k++)
                    parts[i].Vel[k] *= sqrt(header->TimeSnapshot) * header->TimeSnapshot;
            }
        }
    }
}

/* write a header block */
static void petaio_write_header(BigFile * bf, const double atime, const int64_t * NTotal, const Cosmology * CP, const struct header_data * data) {
    BigBlock bh;
    if(0 != big_file_mpi_create_block(bf, &bh, "Header", NULL, 0, 0, 0, MPI_COMM_WORLD)) {
        endrun(0, "Failed to create block at %s:%s\n", "Header",
                big_file_get_error_message());
    }

    /* conversion from peculiar velocity to RSD */
    const double hubble = hubble_function(CP, atime);
    double RSD = 1.0 / (atime * hubble);

    if(!IO.UsePeculiarVelocity) {
        RSD /= atime; /* Conversion from internal velocity to RSD */
    }

    int dk = GetDensityKernelType();
    if(
    (0 != big_block_set_attr(&bh, "TotNumPart", NTotal, "u8", 6)) ||
    (0 != big_block_set_attr(&bh, "TotNumPartInit", &data->NTotalInit, "u8", 6)) ||
    (0 != big_block_set_attr(&bh, "MassTable", &data->MassTable, "f8", 6)) ||
    (0 != big_block_set_attr(&bh, "Time", &atime, "f8", 1)) ||
    (0 != big_block_set_attr(&bh, "TimeIC", &data->TimeIC, "f8", 1)) ||
    (0 != big_block_set_attr(&bh, "BoxSize", &data->BoxSize, "f8", 1)) ||
    (0 != big_block_set_attr(&bh, "OmegaLambda", &CP->OmegaLambda, "f8", 1)) ||
    (0 != big_block_set_attr(&bh, "OmegaFld", &CP->Omega_fld, "f8", 1)) ||
    (0 != big_block_set_attr(&bh, "W0_Fld", &CP->w0_fld, "f8", 1)) ||
    (0 != big_block_set_attr(&bh, "WA_Fld", &CP->wa_fld, "f8", 1)) ||
    (0 != big_block_set_attr(&bh, "RSDFactor", &RSD, "f8", 1)) ||
    (0 != big_block_set_attr(&bh, "UsePeculiarVelocity", &IO.UsePeculiarVelocity, "i4", 1)) ||
    (0 != big_block_set_attr(&bh, "Omega0", &CP->Omega0, "f8", 1)) ||
    (0 != big_block_set_attr(&bh, "OmegaUR", &CP->Omega_ur, "f8", 1)) ||
    (0 != big_block_set_attr(&bh, "OmegaK", &CP->OmegaK, "f8", 1)) ||
    (0 != big_block_set_attr(&bh, "class_radiation_convention", &CP->use_class_radiation_convention, "i4", 1)) ||
    (0 != big_block_set_attr(&bh, "CMBTemperature", &CP->CMBTemperature, "f8", 1)) ||
    (0 != big_block_set_attr(&bh, "OmegaBaryon", &CP->OmegaBaryon, "f8", 1)) ||
    (0 != big_block_set_attr(&bh, "UnitLength_in_cm", &data->UnitLength_in_cm, "f8", 1)) ||
    (0 != big_block_set_attr(&bh, "UnitMass_in_g", &data->UnitMass_in_g, "f8", 1)) ||
    (0 != big_block_set_attr(&bh, "UnitVelocity_in_cm_per_s", &data->UnitVelocity_in_cm_per_s, "f8", 1)) ||
    (0 != big_block_set_attr(&bh, "CodeVersion", GADGET_VERSION, "S1", strlen(GADGET_VERSION))) ||
    (0 != big_block_set_attr(&bh, "CompilerSettings", GADGET_COMPILER_SETTINGS, "S1", strlen(GADGET_COMPILER_SETTINGS))) ||
    (0 != big_block_set_attr(&bh, "DensityKernel", &dk, "i4", 1)) ||
    (0 != big_block_set_attr(&bh, "HubbleParam", &CP->HubbleParam, "f8", 1)) ) {
        endrun(0, "Failed to write attributes %s\n",
                    big_file_get_error_message());
    }

    if(0 != big_block_mpi_close(&bh, MPI_COMM_WORLD)) {
        endrun(0, "Failed to close block %s\n",
                    big_file_get_error_message());
    }
}
static double
_get_attr_double(BigBlock * bh, const char * name, const double def)
{
    double foo;
    if(0 != big_block_get_attr(bh, name, &foo, "f8", 1)) {
        foo = def;
    }
    return foo;
}
static int
_get_attr_int(BigBlock * bh, const char * name, const int def)
{
    int foo;
    if(0 != big_block_get_attr(bh, name, &foo, "i4", 1)) {
        foo = def;
    }
    return foo;
}

static void
petaio_read_header_internal(BigFile * bf, Cosmology * CP, struct header_data * Header) {
    BigBlock bh;
    if(0 != big_file_mpi_open_block(bf, &bh, "Header", MPI_COMM_WORLD)) {
        endrun(0, "Failed to create block at %s:%s\n", "Header",
                    big_file_get_error_message());
    }
    double Time = 0.;
    if(
    (0 != big_block_get_attr(&bh, "TotNumPart", Header->NTotal, "u8", 6)) ||
    (0 != big_block_get_attr(&bh, "MassTable", Header->MassTable, "f8", 6)) ||
    (0 != big_block_get_attr(&bh, "BoxSize", &Header->BoxSize, "f8", 1)) ||
    (0 != big_block_get_attr(&bh, "Time", &Time, "f8", 1))
    ) {
        endrun(0, "Failed to read attr: %s\n",
                    big_file_get_error_message());
    }

    Header->TimeSnapshot = Time;
    if(0!= big_block_get_attr(&bh, "TimeIC", &Header->TimeIC, "f8", 1))
        Header->TimeIC = Time;

    /* Set a reasonable MassTable entry for stars and BHs*/
    if(Header->MassTable[4] == 0)
        Header->MassTable[4] = Header->MassTable[0] / get_generations();
    if(Header->MassTable[5] == 0)
        Header->MassTable[5] = Header->MassTable[4];

    /* fall back to traditional MP-Gadget Units if not given in the snapshot file. */
    Header->UnitVelocity_in_cm_per_s = _get_attr_double(&bh, "UnitVelocity_in_cm_per_s", 1e5); /* 1 km/sec */
    Header->UnitLength_in_cm = _get_attr_double(&bh, "UnitLength_in_cm",  3.085678e21); /* 1.0 Kpc /h */
    Header->UnitMass_in_g = _get_attr_double(&bh, "UnitMass_in_g", 1.989e43); /* 1e10 Msun/h */

    double OmegaBaryon = CP->OmegaBaryon;
    double HubbleParam = CP->HubbleParam;
    double OmegaLambda = CP->OmegaLambda;
    double OmegaFld = -1, WA_Fld = -100, W0_Fld = -100;

    if(0 == big_block_get_attr(&bh, "OmegaBaryon", &CP->OmegaBaryon, "f8", 1) &&
            OmegaBaryon > 0 && fabs(CP->OmegaBaryon - OmegaBaryon) > 1e-3)
        message(0,"IC Header has Omega_b = %g but paramfile wants %g\n", CP->OmegaBaryon, OmegaBaryon);
    if(0 == big_block_get_attr(&bh, "HubbleParam", &CP->HubbleParam, "f8", 1) &&
            HubbleParam > 0 && fabs(CP->HubbleParam - HubbleParam) > 1e-3)
        message(0,"IC Header has HubbleParam = %g but paramfile wants %g\n", CP->HubbleParam, HubbleParam);
    if(0 == big_block_get_attr(&bh, "OmegaLambda", &CP->OmegaLambda, "f8", 1) &&
            OmegaLambda > 0 && fabs(CP->OmegaLambda - OmegaLambda) > 1e-3)
        message(0,"IC Header has Omega_L = %g but paramfile wants %g\n", CP->OmegaLambda, OmegaLambda);
    /* Validate the DE fluid model is the same as the ICs.*/
    if(0 == big_block_get_attr(&bh, "OmegaFld", &OmegaFld, "f8", 1) &&
        0 == big_block_get_attr(&bh, "WA_Fld", &WA_Fld, "f8", 1) &&
        0 == big_block_get_attr(&bh, "W0_Fld", &W0_Fld, "f8", 1) &&
            (OmegaFld >= 0 && (fabs(CP->Omega_fld - OmegaFld) > 1e-3 || fabs(CP->wa_fld - WA_Fld) > 1e-3 || fabs(CP->w0_fld - W0_Fld) > 1e-3)))
        message(0,"IC Header has Omega_fld = %g w0 = %g wa = %g but paramfile wants Omega_fld = %g w0 = %g wa = %g\n", OmegaFld, W0_Fld, WA_Fld, CP->Omega_fld, CP->w0_fld, CP->wa_fld);

    /* Validate that Omega_UR is the same as the ICs*/
    double OmegaUR;
    if(0 == big_block_get_attr(&bh, "OmegaUR", &OmegaUR, "f8", 1) && fabs(CP->Omega_ur - OmegaUR) > 1e-3)
        message(0, "IC Header has Omega_ur = %g but paramfile wants Omega_ur = %g\n", CP->Omega_ur, OmegaUR);
    /* Read in the radiation convention we are using (small differences to Omega_k). May fail, in which case we continue with CAMB.*/
    CP->use_class_radiation_convention = 0;
    big_block_get_attr(&bh, "class_radiation_convention", &CP->use_class_radiation_convention, "i4", 1);
    /* If UsePeculiarVelocity = 1 then snapshots save to the velocity field the physical peculiar velocity, v = a dx/dt (where x is comoving distance).
     * If UsePeculiarVelocity = 0 then the velocity field is a * v = a^2 dx/dt in snapshots
     * and v / sqrt(a) = sqrt(a) dx/dt in the ICs. Note that snapshots never match Gadget-2, which
     * saves physical peculiar velocity / sqrt(a) in both ICs and snapshots. */
    IO.UsePeculiarVelocity = _get_attr_int(&bh, "UsePeculiarVelocity", 0);

    if(0 != big_block_get_attr(&bh, "TotNumPartInit", Header->NTotalInit, "u8", 6)) {
        int ptype;
        for(ptype = 0; ptype < 6; ptype ++) {
            Header->NTotalInit[ptype] = Header->NTotal[ptype];
        }
    }

    if(0 != big_block_mpi_close(&bh, MPI_COMM_WORLD)) {
        endrun(0, "Failed to close block: %s\n",
                    big_file_get_error_message());
    }
}

void petaio_alloc_buffer(BigArray * array, IOTableEntry * ent, int64_t localsize) {
    size_t dims[2];
    ptrdiff_t strides[2];
    int elsize = dtype_itemsize(ent->dtype);

    dims[0] = localsize;
    dims[1] = ent->items;
    strides[1] = elsize;
    strides[0] = elsize * ent->items;
    char * buffer = (char *) mymalloc("IOBUFFER", dims[0] * dims[1] * elsize);

    big_array_init(array, buffer, ent->dtype, 2, dims, strides);
}

/* readout array into P struct with setters */
void petaio_readout_buffer(BigArray * array, IOTableEntry * ent, struct conversions * conv, struct part_manager_type * PartManager, struct slots_manager_type * SlotsManager) {
    int i;
    /* fill the buffer */
    char * p = (char *) array->data;
    for(i = 0; i < PartManager->NumPart; i ++) {
        if(PartManager->Base[i].Type != ent->ptype) continue;
        ent->setter(i, p, PartManager->Base, SlotsManager, conv);
        p += array->strides[0];
    }
}
/* build an IO buffer for block, based on selection
 * only check P[ selection[i]]. If selection is NULL, just use P[i].
 * NOTE: selected range should contain only one particle type!
*/
void
petaio_build_buffer(BigArray * array, IOTableEntry * ent, const int * selection, const int NumSelection, struct particle_data * Parts, struct slots_manager_type * SlotsManager, struct conversions * conv)
{
    if(selection == NULL) {
        endrun(-1, "NULL selection is not supported\n");
    }

    /* don't forget to free buffer after its done*/
    petaio_alloc_buffer(array, ent, NumSelection);

    /* Fast code path if there are no such particles */
    if(NumSelection == 0) {
        return;
    }

#pragma omp parallel
    {
        int i;
        const int tid = omp_get_thread_num();
        const int NT = omp_get_num_threads();
        const int start = NumSelection * (size_t) tid / NT;
        const int end = NumSelection * ((size_t) tid + 1) / NT;
        /* fill the buffer */
        char * p = (char *) array->data;
        p += array->strides[0] * start;
        for(i = start; i < end; i ++) {
            const int j = selection[i];
            if(Parts[j].Type != ent->ptype) {
                endrun(2, "Selection %d has type = %d != %d\n", j, Parts[j].Type, ent->ptype);
            }
            ent->getter(j, p, Parts, SlotsManager, conv);
            p += array->strides[0];
        }
    }
}

/* destroy a buffer, freeing its memory */
void petaio_destroy_buffer(BigArray * array) {
    myfree(array->data);
}

/* read a block from disk, spread the values to memory with setters  */
int petaio_read_block(BigFile * bf, const char * blockname, BigArray * array, int required) {
    BigBlock bb;
    BigBlockPtr ptr;

    /* open the block */
    if(0 != big_file_mpi_open_block(bf, &bb, blockname, MPI_COMM_WORLD)) {
        if(required)
            endrun(0, "Failed to open block at %s:%s\n", blockname, big_file_get_error_message());
        else
            return 1;
    }
    if(0 != big_block_seek(&bb, &ptr, 0)) {
            endrun(1, "Failed to seek block %s: %s\n", blockname, big_file_get_error_message());
    }
    if(0 != big_block_mpi_read(&bb, &ptr, array, IO.NumWriters, MPI_COMM_WORLD)) {
        endrun(1, "Failed to read from block %s: %s\n", blockname, big_file_get_error_message());
    }
    if(0 != big_block_mpi_close(&bb, MPI_COMM_WORLD)) {
        endrun(0, "Failed to close block at %s:%s\n", blockname,
                    big_file_get_error_message());
    }
    return 0;
}

/* save a block to disk */
void petaio_save_block(BigFile * bf, const char * blockname, BigArray * array, int verbose)
{

    BigBlock bb;
    BigBlockPtr ptr;

    int elsize = big_file_dtype_itemsize(array->dtype);

    int NumWriters = IO.NumWriters;

    size_t size = count_sum(array->dims[0]);
    int NumFiles;

    if(IO.EnableAggregatedIO) {
        NumFiles = (size * elsize + IO.BytesPerFile - 1) / IO.BytesPerFile;
        if(NumWriters > NumFiles * IO.WritersPerFile) {
            NumWriters = NumFiles * IO.WritersPerFile;
        }
        if(NumWriters < IO.MinNumWriters) {
            message(0, "Throttling to %d NumWriters but could throttle to %d.\n", IO.MinNumWriters, NumWriters);
            NumWriters = IO.MinNumWriters;
            NumFiles = (NumWriters + IO.WritersPerFile - 1) / IO.WritersPerFile ;
        }
    } else {
        NumFiles = NumWriters;
    }
    /*Do not write empty files*/
    if(size == 0) {
        NumFiles = 0;
    }

    if(verbose && size > 0) {
        message(0, "Will write %td particles to %d Files with %d writers for %s. \n", size, NumFiles, NumWriters, blockname);
    }
    /* create the block */
    /* dims[1] is the number of members per item */
    if(0 != big_file_mpi_create_block(bf, &bb, blockname, array->dtype, array->dims[1], NumFiles, size, MPI_COMM_WORLD)) {
        endrun(0, "Failed to create block at %s:%s\n", blockname,
                    big_file_get_error_message());
    }
    if(0 != big_block_seek(&bb, &ptr, 0)) {
        endrun(0, "Failed to seek:%s\n", big_file_get_error_message());
    }
    if(0 != big_block_mpi_write(&bb, &ptr, array, NumWriters, MPI_COMM_WORLD)) {
        endrun(0, "Failed to write :%s\n", big_file_get_error_message());
    }

    if(verbose && size > 0)
        message(0, "Done writing %td particles to %d Files\n", size, NumFiles);

    if(0 != big_block_mpi_close(&bb, MPI_COMM_WORLD)) {
        endrun(0, "Failed to close block at %s:%s\n", blockname,
                big_file_get_error_message());
    }
}

/*
 * register an IO block of name for particle type ptype.
 *
 * use IO_REG wrapper.
 *
 * with getter function getter
 * getter(i, output)
 * will fill the property of particle i to output.
 *
 * NOTE: dtype shall match the format of output of getter
 *
 * NOTE: currently there is a hard limit (4096 blocks ).
 *
 * */
void io_register_io_block(const char * name,
        const char * dtype,
        int items,
        int ptype,
        property_getter getter,
        property_setter setter,
        int required,
        struct IOTable * IOTable
        ) {
    if (IOTable->used == IOTable->allocated) {
        IOTable->ent = (IOTableEntry *) myrealloc(IOTable->ent, 2*IOTable->allocated*sizeof(IOTableEntry));
        IOTable->allocated *= 2;
    }
    IOTableEntry * ent = &IOTable->ent[IOTable->used];
    strncpy(ent->name, name, 63);
    ent->name[63] = '\0';
    ent->zorder = IOTable->used;
    ent->ptype = ptype;
    strncpy(ent->dtype, dtype, 7);
    ent->dtype[7] = '\0';
    ent->getter = getter;
    ent->setter = setter;
    ent->items = items;
    ent->required = required;
    IOTable->used ++;
}

static void GTPosition(int i, double * out, void * baseptr, void * smanptr, const struct conversions * params) {
    /* Remove the particle offset before saving*/
    struct particle_data * part = (struct particle_data *) baseptr;
    int d;
    for(d = 0; d < 3; d ++) {
        out[d] = part[i].Pos[d] - PartManager->CurrentParticleOffset[d];
        while(out[d] > PartManager->BoxSize) out[d] -= PartManager->BoxSize;
        while(out[d] <= 0) out[d] += PartManager->BoxSize;
    }
}

static void STPosition(int i, double * out, void * baseptr, void * smanptr, const struct conversions * params) {
    int d;
    struct particle_data * part = (struct particle_data *) baseptr;
    for(d = 0; d < 3; d ++) {
        part[i].Pos[d] = out[d];
    }
}

#define SIMPLE_PROPERTY(name, field, type, items) \
    SIMPLE_GETTER(GT ## name , field, type, items, struct particle_data) \
    SIMPLE_SETTER(ST ## name , field, type, items, struct particle_data)
/*A property with getters and setters that are type specific*/
#define SIMPLE_PROPERTY_TYPE(name, ptype, field, type, items) \
    SIMPLE_GETTER(GT ## ptype ## name , field, type, items, struct particle_data) \
    SIMPLE_SETTER(ST ## ptype ## name , field, type, items, struct particle_data)

/* A property that uses getters and setters via the PI of a particle data array.*/
#define SIMPLE_GETTER_PI(name, field, dtype, items, slottype) \
static void name(int i, dtype * out, void * baseptr, void * smanptr, const struct conversions * params) { \
    int PI = ((struct particle_data *) baseptr)[i].PI; \
    int ptype = ((struct particle_data *) baseptr)[i].Type; \
    struct slot_info * info = &(((struct slots_manager_type *) smanptr)->info[ptype]); \
    slottype * sl = (slottype *) info->ptr; \
    int k; \
    for(k = 0; k < items; k ++) { \
        out[k] = *(&(sl[PI].field) + k); \
    } \
}

#define SIMPLE_SETTER_PI(name, field, dtype, items, slottype) \
static void name(int i, dtype * out, void * baseptr, void * smanptr, const struct conversions * params) { \
    int PI = ((struct particle_data *) baseptr)[i].PI; \
    int ptype = ((struct particle_data *) baseptr)[i].Type; \
    struct slot_info * info = &(((struct slots_manager_type *) smanptr)->info[ptype]); \
    slottype * sl = (slottype *) info->ptr; \
    int k; \
    for(k = 0; k < items; k ++) { \
        *(&(sl[PI].field) + k) = out[k]; \
    } \
}

#define SIMPLE_PROPERTY_PI(name, field, type, items, slottype) \
    SIMPLE_GETTER_PI(GT ## name , field, type, items, slottype) \
    SIMPLE_SETTER_PI(ST ## name , field, type, items, slottype)
#define SIMPLE_PROPERTY_TYPE_PI(name, ptype, field, type, items, slottype) \
    SIMPLE_GETTER_PI(GT ## ptype ## name , field, type, items, slottype) \
    SIMPLE_SETTER_PI(ST ## ptype ## name , field, type, items, slottype)

static void GTVelocity(int i, float * out, void * baseptr, void * smanptr, const struct conversions * params) {
    /* Convert to Peculiar Velocity if UsePeculiarVelocity is set */
    double fac;
    struct particle_data * part = (struct particle_data *) baseptr;
    if (IO.UsePeculiarVelocity) {
        fac = 1.0 / params->atime;
    } else {
        fac = 1.0;
    }

    int d;
    for(d = 0; d < 3; d ++) {
        out[d] = fac * part[i].Vel[d];
    }
}
static void STVelocity(int i, float * out, void * baseptr, void * smanptr, const struct conversions * params) {
    double fac;
    struct particle_data * part = (struct particle_data *) baseptr;
    if (IO.UsePeculiarVelocity) {
        fac = params->atime;
    } else {
        fac = 1.0;
    }

    int d;
    for(d = 0; d < 3; d ++) {
        part[i].Vel[d] = out[d] * fac;
    }
}
SIMPLE_PROPERTY(Mass, Mass, float, 1)
SIMPLE_PROPERTY(ID, ID, uint64_t, 1)
SIMPLE_GETTER(GTPotential, Potential, float, 1, struct particle_data)
SIMPLE_GETTER(GTTimeBinHydro, TimeBinHydro, int, 1, struct particle_data)
SIMPLE_GETTER(GTTimeBinGravity, TimeBinGravity, int, 1, struct particle_data)
SIMPLE_PROPERTY(SmoothingLength, Hsml, float, 1)
SIMPLE_PROPERTY_PI(Density, Density, float, 1, struct sph_particle_data)
SIMPLE_PROPERTY_PI(EgyWtDensity, EgyWtDensity, float, 1, struct sph_particle_data)
SIMPLE_PROPERTY_PI(ElectronAbundance, Ne, float, 1, struct sph_particle_data)
SIMPLE_PROPERTY_PI(DelayTime, DelayTime, float, 1, struct sph_particle_data)
SIMPLE_PROPERTY_TYPE_PI(StarFormationTime, 4, FormationTime, float, 1, struct star_particle_data)
SIMPLE_PROPERTY_PI(BirthDensity, BirthDensity, float, 1, struct star_particle_data)
SIMPLE_PROPERTY_TYPE_PI(Metallicity, 4, Metallicity, float, 1, struct star_particle_data)
SIMPLE_PROPERTY_TYPE_PI(LastEnrichmentMyr, 4, LastEnrichmentMyr, float, 1, struct star_particle_data)
SIMPLE_PROPERTY_TYPE_PI(TotalMassReturned, 4, TotalMassReturned, float, 1, struct star_particle_data)
SIMPLE_PROPERTY_TYPE_PI(Metallicity, 0, Metallicity, float, 1, struct sph_particle_data)
SIMPLE_PROPERTY_TYPE_PI(Metals, 4, Metals[0], float, NMETALS, struct star_particle_data)
SIMPLE_PROPERTY_TYPE_PI(Metals, 0, Metals[0], float, NMETALS, struct sph_particle_data)

SIMPLE_GETTER_PI(GTStarFormationRate, Sfr, float, 1, struct sph_particle_data)
SIMPLE_PROPERTY_TYPE_PI(StarFormationTime, 5, FormationTime, float, 1, struct bh_particle_data)
SIMPLE_PROPERTY_PI(BlackholeMass, Mass, float, 1, struct bh_particle_data)
SIMPLE_PROPERTY_PI(BlackholeDensity, Density, float, 1, struct bh_particle_data)
SIMPLE_PROPERTY_PI(BlackholeAccretionRate, Mdot, float, 1, struct bh_particle_data)
SIMPLE_PROPERTY_PI(BlackholeProgenitors, CountProgs, int, 1, struct bh_particle_data)
SIMPLE_PROPERTY_PI(BlackholeSwallowID, SwallowID, uint64_t, 1, struct bh_particle_data)
SIMPLE_PROPERTY_PI(BlackholeSwallowTime, SwallowTime, float, 1, struct bh_particle_data)
SIMPLE_PROPERTY_PI(BlackholeJumpToMinPot, JumpToMinPot, int, 1, struct bh_particle_data)
SIMPLE_PROPERTY_PI(BlackholeMtrack, Mtrack, float, 1, struct bh_particle_data)
SIMPLE_PROPERTY_PI(BlackholeMseed, Mseed, float, 1, struct bh_particle_data)
SIMPLE_PROPERTY_PI(BlackholeKineticFdbkEnergy, KineticFdbkEnergy, float, 1, struct bh_particle_data)

SIMPLE_SETTER_PI(STBlackholeMinPotPos , MinPotPos[0], double, 3, struct bh_particle_data)

/* extra properties from excursion set addition */
#ifdef EXCUR_REION
SIMPLE_PROPERTY_PI(J21, local_J21, float, 1, struct sph_particle_data)
SIMPLE_PROPERTY_PI(ZReionized, zreion, float, 1, struct sph_particle_data)
#endif

static void GTBlackholeMinPotPos(int i, double * out, void * baseptr, void * smanptr, const struct conversions * params) {
    /* Remove the particle offset before saving*/
    struct particle_data * part = (struct particle_data *) baseptr;
    int PI = part[i].PI;
    struct slot_info * info = &(((struct slots_manager_type *) smanptr)->info[5]);
    struct bh_particle_data * sl = (struct bh_particle_data *) info->ptr;
    int d;
    for(d = 0; d < 3; d ++) {
        out[d] = sl[PI].MinPotPos[d] - PartManager->CurrentParticleOffset[d];
        while(out[d] > PartManager->BoxSize) out[d] -= PartManager->BoxSize;
        while(out[d] <= 0) out[d] += PartManager->BoxSize;
    }
}

/*This is only used if FoF is enabled*/
SIMPLE_GETTER(GTGroupID, GrNr, uint32_t, 1, struct particle_data)
static void GTNeutralHydrogenFraction(int i, float * out, void * baseptr, void * smanptr, const struct conversions * params) {
    double redshift = 1./params->atime - 1;
    struct particle_data * pl = ((struct particle_data *) baseptr)+i;
    int PI = pl->PI;
    struct slot_info * info = &(((struct slots_manager_type *) smanptr)->info[0]);
    struct sph_particle_data * sl = (struct sph_particle_data *) info->ptr;
    *out = get_neutral_fraction_sfreff(redshift, params->hubble, pl, sl+PI);
}

static void GTHeliumIFraction(int i, float * out, void * baseptr, void * smanptr, const struct conversions * params) {
    double redshift = 1./params->atime - 1;
    struct particle_data * pl = ((struct particle_data *) baseptr)+i;
    int PI = pl->PI;
    struct slot_info * info = &(((struct slots_manager_type *) smanptr)->info[0]);
    struct sph_particle_data * sl = (struct sph_particle_data *) info->ptr;
    *out = get_helium_neutral_fraction_sfreff(0, redshift, params->hubble, pl, sl+PI);
}
static void GTHeliumIIFraction(int i, float * out, void * baseptr, void * smanptr, const struct conversions * params) {
    double redshift = 1./params->atime - 1;
    struct particle_data * pl = ((struct particle_data *) baseptr)+i;
    int PI = pl->PI;
    struct slot_info * info = &(((struct slots_manager_type *) smanptr)->info[0]);
    struct sph_particle_data * sl = (struct sph_particle_data *) info->ptr;
    *out = get_helium_neutral_fraction_sfreff(1, redshift, params->hubble, pl, sl+PI);
}
static void GTHeliumIIIFraction(int i, float * out, void * baseptr, void * smanptr, const struct conversions * params) {
    double redshift = 1./params->atime - 1;
    struct particle_data * pl = ((struct particle_data *) baseptr)+i;
    int PI = pl->PI;
    struct slot_info * info = &(((struct slots_manager_type *) smanptr)->info[0]);
    struct sph_particle_data * sl = (struct sph_particle_data *) info->ptr;
    *out = get_helium_neutral_fraction_sfreff(2, redshift, params->hubble, pl, sl+PI);
}
static void GTInternalEnergy(int i, float * out, void * baseptr, void * smanptr, const struct conversions * params) {
    int PI = ((struct particle_data *) baseptr)[i].PI;
    struct slot_info * info = &(((struct slots_manager_type *) smanptr)->info[0]);
    struct sph_particle_data * sl = (struct sph_particle_data *) info->ptr;
    double a3inv = 1/(params->atime * params->atime * params->atime);
    *out = sl[PI].Entropy / GAMMA_MINUS1 * pow(sl[PI].Density * a3inv, GAMMA_MINUS1);
}

static void STInternalEnergy(int i, float * out, void * baseptr, void * smanptr, const struct conversions * params) {
    float u = *out;
    int PI = ((struct particle_data *) baseptr)[i].PI;
    struct slot_info * info = &(((struct slots_manager_type *) smanptr)->info[0]);
    struct sph_particle_data * sl = (struct sph_particle_data *) info->ptr;
    double a3inv = 1/(params->atime * params->atime * params->atime);
    sl[PI].Entropy  = GAMMA_MINUS1 * u / pow(sl[PI].Density * a3inv, GAMMA_MINUS1);
}

/* Can't use the macros because cannot take address of a bitfield*/
static void GTHeIIIIonized(int i, unsigned char * out, void * baseptr, void * smanptr, const struct conversions * params) {
    struct particle_data * part = (struct particle_data *) baseptr;
    *out = part[i].HeIIIionized;
}

static void STHeIIIIonized(int i, unsigned char * out, void * baseptr, void * smanptr, const struct conversions * params) {
    struct particle_data * part = (struct particle_data *) baseptr;
    part[i].HeIIIionized = *out;
}
static void GTSwallowed(int i, unsigned char * out, void * baseptr, void * smanptr, const struct conversions * params) {
    struct particle_data * part = (struct particle_data *) baseptr;
    *out = part[i].Swallowed;
}

static void STSwallowed(int i, unsigned char * out, void * baseptr, void * smanptr, const struct conversions * params) {
    struct particle_data * part = (struct particle_data *) baseptr;
    part[i].Swallowed = *out;
}
static void GTGeneration(int i, unsigned char * out, void * baseptr, void * smanptr, const struct conversions * params) {
    struct particle_data * part = (struct particle_data *) baseptr;
    *out = part[i].Generation;
}

static void STGeneration(int i, unsigned char * out, void * baseptr, void * smanptr, const struct conversions * params) {
    struct particle_data * part = (struct particle_data *) baseptr;
    part[i].Generation = *out;
}

static int order_by_type(const void *a, const void *b)
{
    const struct IOTableEntry * pa  = (const struct IOTableEntry *) a;
    const struct IOTableEntry * pb  = (const struct IOTableEntry *) b;

    if(pa->ptype < pb->ptype)
        return -1;
    if(pa->ptype > pb->ptype)
        return +1;
    if(pa->zorder < pb->zorder)
        return -1;
    if(pa->zorder > pb->zorder)
        return 1;

    return 0;
}

void register_io_blocks(struct IOTable * IOTable, int WriteGroupID, int MetalReturnOn)
{
    int i;
    IOTable->used = 0;
    IOTable->allocated = 100;
    IOTable->ent = (IOTableEntry *) mymalloc2("IOTable", IOTable->allocated * sizeof(IOTableEntry));
    /* Bare Bone Gravity*/
    for(i = 0; i < 6; i ++) {
        /* We put Mass first because sometimes there is
         * corruption in the first array and we can recover from Mass corruption*/
        IO_REG(Mass,     "f4", 1, i, IOTable);
        IO_REG(Position, "f8", 3, i, IOTable);
        IO_REG(Velocity, "f4", 3, i, IOTable);
        IO_REG(ID,       "u8", 1, i, IOTable);
        if(IO.OutputPotential)
            IO_REG_WRONLY(Potential, "f4", 1, i, IOTable);
        if(WriteGroupID)
            IO_REG_WRONLY(GroupID, "u4", 1, i, IOTable);
        if(IO.OutputTimebins) {
            IO_REG_WRONLY(TimeBinHydro,       "u4", 1, i, IOTable);
            IO_REG_WRONLY(TimeBinGravity,       "u4", 1, i, IOTable);
        }
    }

    IO_REG(Generation,       "u1", 1, 0, IOTable);
    IO_REG(Generation,       "u1", 1, 4, IOTable);
    IO_REG(Generation,       "u1", 1, 5, IOTable);
    /* Bare Bone SPH*/
    IO_REG(SmoothingLength,  "f4", 1, 0, IOTable);
    IO_REG(Density,          "f4", 1, 0, IOTable);

    if(DensityIndependentSphOn())
        IO_REG(EgyWtDensity,          "f4", 1, 0, IOTable);

    /* On reload this sets the Entropy variable, need the densities.
     * Register this after Density and EgyWtDensity will ensure density is read
     * before this. */
    IO_REG(InternalEnergy,   "f4", 1, 0, IOTable);

    /* Cooling */
    IO_REG(ElectronAbundance,       "f4", 1, 0, IOTable);
    IO_REG_WRONLY(NeutralHydrogenFraction, "f4", 1, 0, IOTable);

    if(IO.OutputHeliumFractions) {
        IO_REG_WRONLY(HeliumIFraction, "f4", 1, 0, IOTable);
        IO_REG_WRONLY(HeliumIIFraction, "f4", 1, 0, IOTable);
        IO_REG_WRONLY(HeliumIIIFraction, "f4", 1, 0, IOTable);
    }
    /* Marks whether a particle has been HeIII ionized yet*/
    IO_REG_NONFATAL(HeIIIIonized, "u1", 1, 0, IOTable);

    /* SF */
    IO_REG_WRONLY(StarFormationRate, "f4", 1, 0, IOTable);
    /* Another new addition: save the DelayTime for wind particles*/
    IO_REG_NONFATAL(DelayTime,  "f4", 1, 0, IOTable);

    IO_REG_NONFATAL(BirthDensity, "f4", 1, 4, IOTable);
    IO_REG_TYPE(StarFormationTime, "f4", 1, 4, IOTable);
    IO_REG_TYPE(Metallicity,       "f4", 1, 0, IOTable);
    IO_REG_TYPE(Metallicity,       "f4", 1, 4, IOTable);
    if(MetalReturnOn) {
        IO_REG_TYPE(Metals,       "f4", NMETALS, 0, IOTable);
        IO_REG_TYPE(Metals,       "f4", NMETALS, 4, IOTable);
        IO_REG_TYPE(LastEnrichmentMyr, "f4", 1, 4, IOTable);
        IO_REG_TYPE(TotalMassReturned, "f4", 1, 4, IOTable);
        IO_REG_NONFATAL(SmoothingLength,  "f4", 1, 4, IOTable);
    }
    /* end SF */

    /* Black hole */
    IO_REG_TYPE(StarFormationTime, "f4", 1, 5, IOTable);
    IO_REG(BlackholeMass,          "f4", 1, 5, IOTable);
    IO_REG(BlackholeDensity,          "f4", 1, 5, IOTable);
    IO_REG(BlackholeAccretionRate, "f4", 1, 5, IOTable);
    IO_REG(BlackholeProgenitors,   "i4", 1, 5, IOTable);
    IO_REG(BlackholeMinPotPos, "f8", 3, 5, IOTable);
    IO_REG(BlackholeJumpToMinPot,   "i4", 1, 5, IOTable);
    IO_REG(BlackholeMtrack,         "f4", 1, 5, IOTable);
    IO_REG_NONFATAL(BlackholeMseed,         "f4", 1, 5, IOTable);
    IO_REG_NONFATAL(BlackholeKineticFdbkEnergy, "f4", 1, 5, IOTable);

    /* Smoothing lengths for black hole: this is a new addition*/
    IO_REG_NONFATAL(SmoothingLength,  "f4", 1, 5, IOTable);
    /* Marks whether a BH particle has been swallowed*/
    IO_REG_NONFATAL(Swallowed, "u1", 1, 5, IOTable);
    /* ID of the swallowing black hole particle. If == -1, then particle is live*/
    IO_REG_NONFATAL(BlackholeSwallowID, "u8", 1, 5, IOTable);
    /* Time the BH was swallowed*/
    IO_REG_NONFATAL(BlackholeSwallowTime, "f4", 1, 5, IOTable);

    /* excursion set */
#ifdef EXCUR_REION
    if(IO.ExcursionSetReionOn){
        IO_REG_NONFATAL(J21,"f4",1,0,IOTable);
        IO_REG_NONFATAL(ZReionized,"f4",1,0,IOTable);
    }
#endif
    /* end excursion set*/

    /*Sort IO blocks so similar types are together; then ordered by the sequence they are declared. */
    qsort_openmp(IOTable->ent, IOTable->used, sizeof(struct IOTableEntry), order_by_type);
}

/* Add extra debug blocks to the output*/
/* Write (but don't read) them, only useful for debugging the particle structures.
 * Warning: future code versions may change the units!*/
SIMPLE_GETTER(GTGravAccel, FullTreeGravAccel[0], float, 3, struct particle_data)
SIMPLE_GETTER(GTGravPM, GravPM[0], float, 3, struct particle_data)
SIMPLE_GETTER_PI(GTHydroAccel, HydroAccel[0], float, 3, struct sph_particle_data)
SIMPLE_GETTER_PI(GTMaxSignalVel, MaxSignalVel, float, 1, struct sph_particle_data)
SIMPLE_GETTER_PI(GTEntropy, Entropy, float, 1, struct sph_particle_data)
SIMPLE_GETTER_PI(GTDtEntropy, DtEntropy, float, 1, struct sph_particle_data)
SIMPLE_GETTER_PI(GTDhsmlEgyDensityFactor, DhsmlEgyDensityFactor, float, 1, struct sph_particle_data)
SIMPLE_GETTER_PI(GTDivVel, DivVel, float, 1, struct sph_particle_data)
SIMPLE_GETTER_PI(GTCurlVel, CurlVel, float, 1, struct sph_particle_data)
SIMPLE_GETTER_PI(GTVelDisp, VDisp, float, 1, struct sph_particle_data)
SIMPLE_GETTER_PI(GTBHVelDisp, VDisp, float, 1, struct bh_particle_data)
SIMPLE_GETTER_PI(GTStarVelDisp, VDisp, float, 1, struct star_particle_data)

void register_debug_io_blocks(struct IOTable * IOTable)
{
    int ptype;
    for(ptype = 0; ptype < 6; ptype++) {
        IO_REG_WRONLY(GravAccel,       "f4", 3, ptype, IOTable);
        IO_REG_WRONLY(GravPM,       "f4", 3, ptype, IOTable);
        if(!IO.OutputTimebins) { /* Otherwise it is output in the regular blocks*/
            IO_REG_WRONLY(TimeBinHydro,       "u4", 1, ptype, IOTable);
            IO_REG_WRONLY(TimeBinGravity,       "u4", 1, ptype, IOTable);
        }
    }
    IO_REG_WRONLY(HydroAccel,       "f4", 3, 0, IOTable);
    IO_REG_WRONLY(MaxSignalVel,       "f4", 1, 0, IOTable);
    IO_REG_WRONLY(Entropy,       "f4", 1, 0, IOTable);
    IO_REG_WRONLY(DtEntropy,       "f4", 1, 0, IOTable);
    IO_REG_WRONLY(DhsmlEgyDensityFactor,       "f4", 1, 0, IOTable);
    IO_REG_WRONLY(DivVel,       "f4", 1, 0, IOTable);
    IO_REG_WRONLY(CurlVel,       "f4", 1, 0, IOTable);
    IO_REG_WRONLY(VelDisp,       "f4", 1, 0, IOTable);
    IO_REG_WRONLY(BHVelDisp,       "f4", 1, 5, IOTable);
    IO_REG_WRONLY(StarVelDisp,       "f4", 1, 4, IOTable);

    /*Sort IO blocks so similar types are together; then ordered by the sequence they are declared. */
    qsort_openmp(IOTable->ent, IOTable->used, sizeof(struct IOTableEntry), order_by_type);
}

void destroy_io_blocks(struct IOTable * IOTable) {
    myfree(IOTable->ent);
    IOTable->allocated = 0;
}
