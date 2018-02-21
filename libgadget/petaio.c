#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <string.h>
#include <stdarg.h>

#include <bigfile-mpi.h>

#include "allvars.h"
#include "sfr_eff.h"
#include "cooling.h"
#include "timestep.h"

#include "petaio.h"
#include "slotsmanager.h"
#include "partmanager.h"
#include "config.h"
#include "kspace-neutrinos/delta_tot_table.h"

#include "utils.h"

/*Defined in fofpetaio.c and only used here*/
void fof_register_io_blocks();

/************
 *
 * The IO api , intented to replace io.c and read_ic.c
 * currently we have a function to register the blocks and enumerate the blocks
 *
 */

struct IOTable IOTable = {0};

static void petaio_write_header(BigFile * bf, const int64_t * NTotal);
static void petaio_read_header_internal(BigFile * bf);

static void register_io_blocks();

/* these are only used in reading in */
void petaio_init() {
    /* Smaller files will do aggregareted IO.*/
    if(All.IO.EnableAggregatedIO) {
        message(0, "Aggregated IO is enabled\n");
        big_file_mpi_set_aggregated_threshold(All.IO.AggregatedIOThreshold);
    } else {
        message(0, "Aggregated IO is disabled.\n");
        big_file_mpi_set_aggregated_threshold(0);
    }
    register_io_blocks();
}

/* save a snapshot file */
static void petaio_save_internal(char * fname);

void
petaio_save_snapshot(const char *fmt, ...)
{
    va_list va;
    va_start(va, fmt);

    char fname[4096];
    vsprintf(fname, fmt, va);
    va_end(va);
    message(0, "saving snapshot into %s\n", fname);

    petaio_save_internal(fname);
}

/* this is unused.
void petaio_save_restart() {
    char fname[4096];
    sprintf(fname, "%s/RESTART", All.OutputDir);
    if(ThisTask == 0) {
        printf("saving restart into %s\n", fname);
        fflush(stdout);
    }
    petaio_save_internal(fname);
}
*/

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
    int * ptype_offset,
    int * ptype_count,
    const int NumPart,
    int (*select_func)(int i)
    )
{
    int i;
    ptype_offset[0] = 0;
    ptype_count[0] = 0;

    for(i = 0; i < NumPart; i ++) {
        if((select_func == NULL) || (select_func(i) != 0)) {
            int ptype = P[i].Type;
            ptype_count[ptype] ++;
        }
    }
    for(i = 1; i < 6; i ++) {
        ptype_offset[i] = ptype_offset[i-1] + ptype_count[i-1];
        ptype_count[i-1] = 0;
    }

    ptype_count[5] = 0;
    for(i = 0; i < NumPart; i ++) {
        int ptype = P[i].Type;
        if((select_func == NULL) || (select_func(i) != 0)) {
            selection[ptype_offset[ptype] + ptype_count[ptype]] = i;
            ptype_count[ptype]++;
        }
    }
}

/*These two functions are only ued for the semi-linear neutrino implementation,
 * and store data specific to that in the snapshots*/
/*Defined in gravpm.c*/
extern _delta_tot_table delta_tot_table;

void petaio_save_neutrinos(BigFile * bf)
{
#pragma omp master
    {
    double * scalefact = delta_tot_table.scalefact;
    size_t nk = delta_tot_table.nk, ia = delta_tot_table.ia;
    size_t ik, i;
    double * delta_tot = mymalloc("tmp_delta",nk * ia * sizeof(double));
    /*Save a flat memory block*/
    for(ik=0;ik< nk;ik++)
        for(i=0;i< ia;i++)
            delta_tot[ik*ia+i] = delta_tot_table.delta_tot[ik][i];

    BigBlock bn = {0};
    if(0 != big_file_mpi_create_block(bf, &bn, "Neutrino", NULL, 0, 0, 0, MPI_COMM_WORLD)) {
        endrun(0, "Failed to create block at %s:%s\n", "Neutrino",
                big_file_get_error_message());
    }
    if ( (0 != big_block_set_attr(&bn, "Nscale", &ia, "u8", 1)) ||
       (0 != big_block_set_attr(&bn, "scalefact", scalefact, "f8", ia)) ||
        (0 != big_block_set_attr(&bn, "Nkval", &nk, "u8", 1)) ) {
        endrun(0, "Failed to write neutrino attributes %s\n",
                    big_file_get_error_message());
    }
    if(0 != big_block_mpi_close(&bn, MPI_COMM_WORLD)) {
        endrun(0, "Failed to close block %s\n",
                    big_file_get_error_message());
    }
    BigArray deltas = {0};
    size_t dims[2] = {0 , ia};
    /*The neutrino state is shared between all processors,
     *so only write on master task*/
    if(ThisTask == 0) {
        dims[0] = nk;
    }
    ptrdiff_t strides[2] = {8 * ia, 8};
    big_array_init(&deltas, delta_tot, "=f8", 2, dims, strides);
    petaio_save_block(bf, "Neutrino/Deltas", &deltas);
    myfree(delta_tot);
    }
}

/*Read the neutrino data from the snapshot*/
void petaio_read_neutrinos(BigFile * bf)
{
#pragma omp master
    {
    size_t nk, ia, ik, i;
    BigBlock bn = {0};
    if(0 != big_file_mpi_open_block(bf, &bn, "Neutrino", MPI_COMM_WORLD)) {
        endrun(0, "Failed to open block at %s:%s\n", "Neutrino",
                    big_file_get_error_message());
    }
    if(
    (0 != big_block_get_attr(&bn, "Nscale", &ia, "u8", 1)) ||
    (0 != big_block_get_attr(&bn, "Nkval", &nk, "u8", 1))) {
        endrun(0, "Failed to read attr: %s\n",
                    big_file_get_error_message());
    }
    double *delta_tot = (double *) mymalloc("tmp_nusave",ia*nk*sizeof(double));
    /*Allocate list of scale factors, and space for delta_tot, in one operation.*/
    if(0 != big_block_get_attr(&bn, "scalefact", delta_tot_table.scalefact, "f8", ia))
        endrun(0, "Failed to read attr: %s\n", big_file_get_error_message());
    if(0 != big_block_mpi_close(&bn, MPI_COMM_WORLD)) {
        endrun(0, "Failed to close block %s\n",
                    big_file_get_error_message());
    }
    BigArray deltas = {0};
    size_t dims[2] = {0, ia};
    ptrdiff_t strides[2] = {8*ia, 8};
    /*The neutrino state is shared between all processors,
     *so only read on master task and broadcast*/
    if(ThisTask == 0) {
        dims[0] = nk;
    }
    big_array_init(&deltas, delta_tot, "=f8", 2, dims, strides);
    petaio_read_block(bf, "Neutrino/Deltas", &deltas, 1);
    /*Save a flat memory block*/
    for(ik=0;ik<nk;ik++)
        for(i=0;i<ia;i++)
            delta_tot_table.delta_tot[ik][i] = delta_tot[ik*ia+i];
    delta_tot_table.nk = nk;
    delta_tot_table.ia = ia;
    myfree(delta_tot);
    /*Broadcast the arrays.*/
    MPI_Bcast(&(delta_tot_table.ia), 1,MPI_INT,0,MPI_COMM_WORLD);
    if(delta_tot_table.ia > 0) {
        MPI_Bcast(&(delta_tot_table.nk), 1,MPI_INT,0,MPI_COMM_WORLD);
        /*Broadcast data for scalefact and delta_tot, Delta_tot is allocated as the same block of memory as scalefact.
          Not all this memory will actually have been used, but it is easiest to bcast all of it.*/
        MPI_Bcast(delta_tot_table.scalefact,delta_tot_table.namax*(delta_tot_table.nk+1),MPI_DOUBLE,0,MPI_COMM_WORLD);
    }
    }
}
/*End of massive neutrino functions*/


static void petaio_save_internal(char * fname) {
    BigFile bf = {0};
    if(0 != big_file_mpi_create(&bf, fname, MPI_COMM_WORLD)) {
        endrun(0, "Failed to create snapshot at %s:%s\n", fname,
                    big_file_get_error_message());
    }

    int ptype_offset[6]={0};
    int ptype_count[6]={0};
    int64_t NTotal[6]={0};

    int * selection = mymalloc("Selection", sizeof(int) * PartManager->NumPart);

    petaio_build_selection(selection, ptype_offset, ptype_count, PartManager->NumPart, NULL);

    sumup_large_ints(6, ptype_count, NTotal);

    petaio_write_header(&bf, NTotal);

    int i;
    for(i = 0; i < IOTable.used; i ++) {
        /* only process the particle blocks */
        char blockname[128];
        int ptype = IOTable.ent[i].ptype;
        BigArray array = {0};
        /*This exclude FOF blocks*/
        if(!(ptype < 6 && ptype >= 0)) {
            continue;
        }
        sprintf(blockname, "%d/%s", ptype, IOTable.ent[i].name);
        petaio_build_buffer(&array, &IOTable.ent[i], selection + ptype_offset[ptype], ptype_count[ptype]);
        petaio_save_block(&bf, blockname, &array);
        petaio_destroy_buffer(&array);
    }

    if(All.MassiveNuLinRespOn) {
        petaio_save_neutrinos(&bf);
    }
    if(0 != big_file_mpi_close(&bf, MPI_COMM_WORLD)){
        endrun(0, "Failed to close snapshot at %s:%s\n", fname,
                    big_file_get_error_message());
    }

    myfree(selection);
}

void petaio_read_internal(char * fname, int ic) {
    int ptype;
    int i;
    BigFile bf = {0};
    BigBlock bh = {0};
    message(0, "Reading snapshot %s\n", fname);

    if(0 != big_file_mpi_open(&bf, fname, MPI_COMM_WORLD)) {
        endrun(0, "Failed to open snapshot at %s:%s\n", fname,
                    big_file_get_error_message());
    }

    int64_t NTotal[6];
    if(0 != big_file_mpi_open_block(&bf, &bh, "Header", MPI_COMM_WORLD)) {
        endrun(0, "Failed to create block at %s:%s\n", "Header",
                    big_file_get_error_message());
    }
    if ((0 != big_block_get_attr(&bh, "TotNumPart", NTotal, "u8", 6)) ||
        (0 != big_block_mpi_close(&bh, MPI_COMM_WORLD))) {
        endrun(0, "Failed to close block: %s\n",
                    big_file_get_error_message());
    }

    /* sets the maximum number of particles that may reside on a processor */
    int MaxPart = (int) (All.PartAllocFactor * All.TotNumPartInit / NTask);

    /*Allocates ActiveParticle array*/
    timestep_allocate_memory(MaxPart);

    /*Allocate the particle memory*/
    particle_alloc_memory(MaxPart);

    int NLocal[6];
    for(ptype = 0; ptype < 6; ptype ++) {
        int64_t start = ThisTask * NTotal[ptype] / NTask;
        int64_t end = (ThisTask + 1) * NTotal[ptype] / NTask;
        NLocal[ptype] = end - start;
        PartManager->NumPart += NLocal[ptype];

    }

    /* Allocate enough memory for stars and black holes.
     * This will be dynamically increased as needed.*/

    if(PartManager->NumPart >= PartManager->MaxPart) {
        endrun(1, "Overwhelmed by part: %d > %d\n", PartManager->NumPart, PartManager->MaxPart);
    }

    int newSlots[6];

    for(ptype = 0; ptype < 6; ptype++) {
        /* initialize MaxSlots to zero, such that grow don't fail. */
        newSlots[ptype] = 0;
        if(NLocal[ptype] > 0) {
            newSlots[ptype] = All.PartAllocFactor * NLocal[ptype];
        }
    }

    /* Now allocate memory for the secondary particle data arrays.
     * This may be dynamically resized later!*/

    /*Ensure all processors have initially the same number of particle slots*/
    MPI_Allreduce(MPI_IN_PLACE, newSlots, 6, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    slots_reserve(0, newSlots);

    /* initialize particle types */
    int offset = 0;
    for(ptype = 0; ptype < 6; ptype ++) {
        for(i = 0; i < NLocal[ptype]; i++)
        {
            int j = offset + i;
            P[j].Type = ptype;
        }
        offset += NLocal[ptype];
    }

    /* so we can set up the memory topology of secondary slots */
    slots_setup_topology();

    for(i = 0; i < IOTable.used; i ++) {
        /* only process the particle blocks */
        char blockname[128];
        int ptype = IOTable.ent[i].ptype;
        BigArray array = {0};
        if(!(ptype < 6 && ptype >= 0)) {
            continue;
        }
        if(NTotal[ptype] == 0) continue;
        if(ic) {
            /* for IC read in only three blocks */
            int keep = 0;
            keep |= (0 == strcmp(IOTable.ent[i].name, "Position"));
            keep |= (0 == strcmp(IOTable.ent[i].name, "Velocity"));
            keep |= (0 == strcmp(IOTable.ent[i].name, "ID"));
            if (ptype == 5) {
                keep |= (0 == strcmp(IOTable.ent[i].name, "Mass"));
                keep |= (0 == strcmp(IOTable.ent[i].name, "BlackholeMass"));
                keep |= (0 == strcmp(IOTable.ent[i].name, "MinPotPos"));
                keep |= (0 == strcmp(IOTable.ent[i].name, "MinPotVel"));
            }
            if(!keep) continue;
        }
        if(IOTable.ent[i].setter == NULL) {
            /* FIXME: do not know how to read this block; assume the fucker is
             * internally intialized; */
            continue;
        }
        sprintf(blockname, "%d/%s", ptype, IOTable.ent[i].name);
        petaio_alloc_buffer(&array, &IOTable.ent[i], NLocal[ptype]);
        if(0 == petaio_read_block(&bf, blockname, &array, IOTable.ent[i].required))
            petaio_readout_buffer(&array, &IOTable.ent[i]);
        petaio_destroy_buffer(&array);
    }

    /*Read neutrinos from the snapshot if necessary*/
    if((!ic) && All.MassiveNuLinRespOn)
        petaio_read_neutrinos(&bf);

    if(0 != big_file_mpi_close(&bf, MPI_COMM_WORLD)) {
        endrun(0, "Failed to close snapshot at %s:%s\n", fname,
                    big_file_get_error_message());
    }

    /* now we have IDs, set up the ID consistency between slots. */
    slots_setup_id();
}
void
petaio_read_header(int num)
{
    BigFile bf = {0};

    char * fname;
    if(num == -1) {
        fname = fastpm_strdup_printf("%s", All.InitCondFile);
    } else {
        fname = fastpm_strdup_printf("%s/%s_%03d", All.OutputDir, All.SnapshotFileBase, num);
    }
    message(0, "Probing Header of snapshot file: %s\n", fname);

    if(0 != big_file_mpi_open(&bf, fname, MPI_COMM_WORLD)) {
        endrun(0, "Failed to open snapshot at %s:%s\n", fname,
                    big_file_get_error_message());
    }

    petaio_read_header_internal(&bf);

    if(0 != big_file_mpi_close(&bf, MPI_COMM_WORLD)) {
        endrun(0, "Failed to close snapshot at %s:%s\n", fname,
                    big_file_get_error_message());
    }
    free(fname);
}

void
petaio_read_snapshot(int num)
{
    char * fname;
    if(num == -1) {
        fname = fastpm_strdup_printf("%s", All.InitCondFile);
        /*
         *  IC doesn't have entropy or energy; always use the
         *  InitTemp in paramfile, then use init.c to convert to
         *  entropy.
         * */
        petaio_read_internal(fname, 1);

        int i;
        /* touch up the mass -- IC files save mass in header */
        for(i = 0; i < PartManager->NumPart; i++)
        {
            P[i].Mass = All.MassTable[P[i].Type];
        }

        if (!All.IO.UsePeculiarVelocity ) {

            /* fixing the unit of velocity from Legacy GenIC IC */
            #pragma omp parallel for
            for(i = 0; i < PartManager->NumPart; i++) {
                int k;
                /* for GenIC's Gadget-1 snapshot Unit to Gadget-2 Internal velocity unit */
                for(k = 0; k < 3; k++)
                    P[i].Vel[k] *= sqrt(All.cf.a) * All.cf.a;
            }

        }
    } else {
        fname = fastpm_strdup_printf("%s/%s_%03d", All.OutputDir, All.SnapshotFileBase, num);
        /*
         * we always save the Entropy, init.c will not mess with the entropy
         * */
        petaio_read_internal(fname, 0);
    }
    free(fname);
}


/* write a header block */
static void petaio_write_header(BigFile * bf, const int64_t * NTotal) {
    BigBlock bh = {0};
    if(0 != big_file_mpi_create_block(bf, &bh, "Header", NULL, 0, 0, 0, MPI_COMM_WORLD)) {
        endrun(0, "Failed to create block at %s:%s\n", "Header",
                big_file_get_error_message());
    }

    /* conversion from peculiar velocity to RSD */
    double RSD = 1.0 / (All.cf.a * All.cf.hubble);

    if(!All.IO.UsePeculiarVelocity) {
        RSD /= All.cf.a; /* Conversion from internal velocity to RSD */
    }

    if(
    (0 != big_block_set_attr(&bh, "TotNumPart", NTotal, "u8", 6)) ||
    (0 != big_block_set_attr(&bh, "TotNumPartInit", All.NTotalInit, "u8", 6)) ||
    (0 != big_block_set_attr(&bh, "MassTable", All.MassTable, "f8", 6)) ||
    (0 != big_block_set_attr(&bh, "Time", &All.Time, "f8", 1)) ||
    (0 != big_block_set_attr(&bh, "TimeIC", &All.TimeIC, "f8", 1)) ||
    (0 != big_block_set_attr(&bh, "BoxSize", &All.BoxSize, "f8", 1)) ||
    (0 != big_block_set_attr(&bh, "OmegaLambda", &All.CP.OmegaLambda, "f8", 1)) ||
    (0 != big_block_set_attr(&bh, "RSDFactor", &RSD, "f8", 1)) ||
    (0 != big_block_set_attr(&bh, "UsePeculiarVelocity", &All.IO.UsePeculiarVelocity, "i4", 1)) ||
    (0 != big_block_set_attr(&bh, "Omega0", &All.CP.Omega0, "f8", 1)) ||
    (0 != big_block_set_attr(&bh, "CMBTemperature", &All.CP.CMBTemperature, "f8", 1)) ||
    (0 != big_block_set_attr(&bh, "OmegaBaryon", &All.CP.OmegaBaryon, "f8", 1)) ||
    (0 != big_block_set_attr(&bh, "UnitLength_in_cm", &All.UnitLength_in_cm, "f8", 1)) ||
    (0 != big_block_set_attr(&bh, "UnitMass_in_g", &All.UnitMass_in_g, "f8", 1)) ||
    (0 != big_block_set_attr(&bh, "UnitVelocity_in_cm_per_s", &All.UnitVelocity_in_cm_per_s, "f8", 1)) ||
    (0 != big_block_set_attr(&bh, "CodeVersion", GADGET_VERSION, "S1", strlen(GADGET_VERSION))) ||
    (0 != big_block_set_attr(&bh, "CompilerSettings", GADGET_COMPILER_SETTINGS, "S1", strlen(GADGET_COMPILER_SETTINGS))) ||
    (0 != big_block_set_attr(&bh, "DensityKernel", &All.DensityKernelType, "u8", 1)) ||
    (0 != big_block_set_attr(&bh, "HubbleParam", &All.CP.HubbleParam, "f8", 1)) ) {
        endrun(0, "Failed to write attributes %s\n",
                    big_file_get_error_message());
    }

    if(0 != big_block_mpi_close(&bh, MPI_COMM_WORLD)) {
        endrun(0, "Failed to close block %s\n",
                    big_file_get_error_message());
    }
}
static double
_get_attr_double(BigBlock * bh, char * name, double def)
{
    double foo;
    if(0 != big_block_get_attr(bh, name, &foo, "f8", 1)) {
        foo = def;
    }
    return foo;
}
static int
_get_attr_int(BigBlock * bh, char * name, int def)
{
    int foo;
    if(0 != big_block_get_attr(bh, name, &foo, "i4", 1)) {
        foo = def;
    }
    return foo;
}

static void
petaio_read_header_internal(BigFile * bf) {
    BigBlock bh = {0};
    if(0 != big_file_mpi_open_block(bf, &bh, "Header", MPI_COMM_WORLD)) {
        endrun(0, "Failed to create block at %s:%s\n", "Header",
                    big_file_get_error_message());
    }
    double Time;
    int ptype;
    int64_t NTotal[6];
    if(
    (0 != big_block_get_attr(&bh, "TotNumPart", NTotal, "u8", 6)) ||
    (0 != big_block_get_attr(&bh, "MassTable", All.MassTable, "f8", 6)) ||
    (0 != big_block_get_attr(&bh, "BoxSize", &All.BoxSize, "f8", 1)) ||
    (0 != big_block_get_attr(&bh, "Time", &Time, "f8", 1))
    ) {
        endrun(0, "Failed to read attr: %s\n",
                    big_file_get_error_message());
    }

    All.TimeInit = Time;
    if(0!= big_block_get_attr(&bh, "TimeIC", &All.TimeIC, "f8", 1))
        All.TimeIC = Time;

    /* fall back to traditional MP-Gadget Units if not given in the snapshot file. */
    All.UnitVelocity_in_cm_per_s = _get_attr_double(&bh, "UnitVelocity_in_cm_per_s", 1e5); /* 1 km/sec */
    All.UnitLength_in_cm = _get_attr_double(&bh, "UnitLength_in_cm",  3.085678e21); /* 1.0 Kpc /h */
    All.UnitMass_in_g = _get_attr_double(&bh, "UnitMass_in_g", 1.989e43); /* 1e10 Msun/h */

    /* Fall back to use a**2 * dx/dt if UsePeculiarVelocity is not set in IC */
    All.IO.UsePeculiarVelocity = _get_attr_int(&bh, "UsePeculiarVelocity", 0);

    if(0 != big_block_get_attr(&bh, "TotNumPartInit", All.NTotalInit, "u8", 6)) {
        int ptype;
        for(ptype = 0; ptype < 6; ptype ++) {
            All.NTotalInit[ptype] = NTotal[ptype];
        }
    }

    int64_t TotNumPart = 0;
    All.TotNumPartInit = 0;
    for(ptype = 0; ptype < 6; ptype ++) {
        TotNumPart += NTotal[ptype];
        All.TotNumPartInit += All.NTotalInit[ptype];
        if(All.NTotalInit[ptype] > 0) {
            All.MeanSeparation[ptype] = All.BoxSize / pow(All.NTotalInit[ptype], 1.0 / 3);
        } else {
            All.MeanSeparation[ptype] = 0;
        }
    }

    message(0, "Total number of particles: %018ld\n", TotNumPart);

    const char * PARTICLE_TYPE_NAMES [] = {"Gas", "DarkMatter", "Neutrino", "Unknown", "Star", "BlackHole"};

    for(ptype = 0; ptype < 6; ptype ++) {
        message(0, "% 11s: Total: %018ld Init: %018ld Mean-Sep %g \n",
                PARTICLE_TYPE_NAMES[ptype], NTotal[ptype], All.NTotalInit[ptype], All.MeanSeparation[ptype]);
    }

    /*FIXME: check others as well */
    /*
    big_block_get_attr(&bh, "OmegaLambda", &All.OmegaLambda, "f8", 1);
    big_block_get_attr(&bh, "Omega0", &All.Omega0, "f8", 1);
    big_block_get_attr(&bh, "HubbleParam", &All.HubbleParam, "f8", 1);
    */
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
    char * buffer = mymalloc("IOBUFFER", dims[0] * dims[1] * elsize);

    big_array_init(array, buffer, ent->dtype, 2, dims, strides);
}

/* readout array into P struct with setters */
void petaio_readout_buffer(BigArray * array, IOTableEntry * ent) {
    int i;
    /* fill the buffer */
    char * p = array->data;
    for(i = 0; i < PartManager->NumPart; i ++) {
        if(P[i].Type != ent->ptype) continue;
        ent->setter(i, p);
        p += array->strides[0];
    }
}
/* build an IO buffer for block, based on selection
 * only check P[ selection[i]]. If selection is NULL, just use P[i].
 * NOTE: selected range should contain only one particle type!
*/
void
petaio_build_buffer(BigArray * array, IOTableEntry * ent, const int * selection, const int NumSelection)
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
        char * p = array->data;
        p += array->strides[0] * start;
        for(i = start; i < end; i ++) {
            const int j = selection[i];
            if(P[j].Type != ent->ptype) {
                endrun(2, "Selection %d has type = %d != %d\n", j, P[j].Type, ent->ptype);
            }
            ent->getter(j, p);
            p += array->strides[0];
        }
    }
}

/* destropy a buffer, freeing its memory */
void petaio_destroy_buffer(BigArray * array) {
    myfree(array->data);
}

/* read a block from disk, spread the values to memory with setters  */
int petaio_read_block(BigFile * bf, char * blockname, BigArray * array, int required) {
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
    if(0 != big_block_mpi_read(&bb, &ptr, array, All.IO.NumWriters, MPI_COMM_WORLD)) {
        endrun(1, "Failed to read from block %s: %s\n", blockname, big_file_get_error_message());
    }
    if(0 != big_block_mpi_close(&bb, MPI_COMM_WORLD)) {
        endrun(0, "Failed to close block at %s:%s\n", blockname,
                    big_file_get_error_message());
    }
    return 0;
}

/* save a block to disk */
void petaio_save_block(BigFile * bf, char * blockname, BigArray * array)
{

    BigBlock bb;
    BigBlockPtr ptr;

    int elsize = dtype_itemsize(array->dtype);

    int NumWriters = All.IO.NumWriters;

    size_t size = count_sum(array->dims[0]);
    int NumFiles;

    if(All.IO.EnableAggregatedIO) {
        NumFiles = (size * elsize + All.IO.BytesPerFile - 1) / All.IO.BytesPerFile;
        if(NumWriters > NumFiles * All.IO.WritersPerFile) {
            NumWriters = NumFiles * All.IO.WritersPerFile;
            message(0, "Throttling NumWriters to %d.\n", NumWriters);
        }
        if(NumWriters < All.IO.MinNumWriters) {
            NumWriters = All.IO.MinNumWriters;
            NumFiles = (NumWriters + All.IO.WritersPerFile - 1) / All.IO.WritersPerFile ;
            message(0, "Throttling NumWriters to %d.\n", NumWriters);
        }
    } else {
        NumFiles = NumWriters;
    }
    /*Do not write empty files*/
    if(size == 0) {
        NumFiles = 0;
    }

    if(size > 0) {
        message(0, "Will write %td particles to %d Files for %s\n", size, NumFiles, blockname);
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

    if(size > 0)
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
void io_register_io_block(char * name,
        char * dtype,
        int items,
        int ptype,
        property_getter getter,
        property_setter setter,
        int required
        ) {
    IOTableEntry * ent = &IOTable.ent[IOTable.used];
    strcpy(ent->name, name);
    ent->ptype = ptype;
    dtype_normalize(ent->dtype, dtype);
    ent->getter = getter;
    ent->setter = setter;
    ent->items = items;
    ent->required = required;
    IOTable.used ++;
}

SIMPLE_PROPERTY(Position, P[i].Pos[0], double, 3)
static void GTVelocity(int i, float * out) {
    /* Convert to Peculiar Velocity if UsePeculiarVelocity is set */
    double fac;
    if (All.IO.UsePeculiarVelocity) {
        fac = 1.0 / All.cf.a;
    } else {
        fac = 1.0;
    }

    int d;
    for(d = 0; d < 3; d ++) {
        out[d] = fac * P[i].Vel[d];
    }
}
static void STVelocity(int i, float * out) {
    double fac;
    if (All.IO.UsePeculiarVelocity) {
        fac = All.cf.a;
    } else {
        fac = 1.0;
    }

    int d;
    for(d = 0; d < 3; d ++) {
        P[i].Vel[d] = out[d] * fac;
    }
}
SIMPLE_PROPERTY(Mass, P[i].Mass, float, 1)
SIMPLE_PROPERTY(ID, P[i].ID, uint64_t, 1)
SIMPLE_PROPERTY(Generation, P[i].Generation, unsigned char, 1)
SIMPLE_GETTER(GTPotential, P[i].Potential, float, 1)
SIMPLE_PROPERTY(SmoothingLength, P[i].Hsml, float, 1)
SIMPLE_PROPERTY(Density, SPHP(i).Density, float, 1)
#ifdef DENSITY_INDEPENDENT_SPH
SIMPLE_PROPERTY(EgyWtDensity, SPHP(i).EgyWtDensity, float, 1)
SIMPLE_PROPERTY(Entropy, SPHP(i).Entropy, float, 1)
#endif
SIMPLE_PROPERTY(ElectronAbundance, SPHP(i).Ne, float, 1)
#ifdef SFR
SIMPLE_PROPERTY_TYPE(StarFormationTime, 4, STARP(i).FormationTime, float, 1)
SIMPLE_PROPERTY(BirthDensity, STARP(i).BirthDensity, float, 1)
SIMPLE_PROPERTY_TYPE(Metallicity, 4, STARP(i).Metallicity, float, 1)
SIMPLE_PROPERTY_TYPE(Metallicity, 0, SPHP(i).Metallicity, float, 1)
static void GTStarFormationRate(int i, float * out) {
    /* Convert to Solar/year */
    *out = get_starformation_rate(i)
        * ((All.UnitMass_in_g / SOLAR_MASS) / (All.UnitTime_in_s / SEC_PER_YEAR));
}
#endif
#ifdef BLACK_HOLES
SIMPLE_PROPERTY_TYPE(StarFormationTime, 5, BHP(i).FormationTime, float, 1)
SIMPLE_PROPERTY(BlackholeMass, BHP(i).Mass, float, 1)
SIMPLE_PROPERTY(BlackholeAccretionRate, BHP(i).Mdot, float, 1)
SIMPLE_PROPERTY(BlackholeProgenitors, BHP(i).CountProgs, float, 1)
SIMPLE_PROPERTY(BlackholeMinPotPos, BHP(i).MinPotPos[0], double, 3)
SIMPLE_PROPERTY(BlackholeMinPotVel, BHP(i).MinPotVel[0], float, 3)
SIMPLE_PROPERTY(BlackholeJumpToMinPot, BHP(i).JumpToMinPot, int, 1)
#endif
/*This is only used if FoF is enabled*/
SIMPLE_GETTER(GTGroupID, P[i].GrNr, uint32_t, 1)
static void GTNeutralHydrogenFraction(int i, float * out) {
    double ne, nh0, nHeII;
    ne = SPHP(i).Ne;
    struct UVBG uvbg;
    GetParticleUVBG(i, &uvbg);
    AbundanceRatios(DMAX(All.MinEgySpec,
                SPHP(i).Entropy / GAMMA_MINUS1 * pow(SPHP(i).EOMDensity *
                    All.cf.a3inv,
                    GAMMA_MINUS1)),
            SPHP(i).Density * All.cf.a3inv, &uvbg, &ne, &nh0, &nHeII);
    *out = nh0;
}

static void GTInternalEnergy(int i, float * out) {
    *out = DMAX(All.MinEgySpec,
        SPHP(i).Entropy / GAMMA_MINUS1 * pow(SPHP(i).EOMDensity * All.cf.a3inv, GAMMA_MINUS1));
}

static void GTJUV(int i, float * out) {
    struct UVBG uvbg;
    GetParticleUVBG(i, &uvbg);
    *out = uvbg.J_UV;
}

static int order_by_type(const void *a, const void *b)
{
    const struct IOTableEntry * pa  = (const struct IOTableEntry *) a;
    const struct IOTableEntry * pb  = (const struct IOTableEntry *) b;

    if(pa->ptype < pb->ptype)
        return -1;
    if(pa->ptype > pb->ptype)
        return +1;
    return 0;
}

static void register_io_blocks() {
    int i;
    /* Bare Bone Gravity*/
    for(i = 0; i < 6; i ++) {
        IO_REG(Position, "f8", 3, i);
        IO_REG(Velocity, "f4", 3, i);
        IO_REG(Mass,     "f4", 1, i);
        IO_REG(ID,       "u8", 1, i);
        if(All.OutputPotential)
            IO_REG_WRONLY(Potential, "f4", 1, i);
        if(All.SnapshotWithFOF)
            IO_REG_WRONLY(GroupID, "u4", 1, i);
    }

    IO_REG(Generation,       "u1", 1, 0);
    IO_REG(Generation,       "u1", 1, 4);
    IO_REG(Generation,       "u1", 1, 5);
    /* Bare Bone SPH*/
    IO_REG(SmoothingLength,  "f4", 1, 0);
    IO_REG(Density,          "f4", 1, 0);
#ifdef DENSITY_INDEPENDENT_SPH
    IO_REG(EgyWtDensity,          "f4", 1, 0);
    IO_REG(Entropy,          "f4", 1, 0);
#endif

    /* Cooling */
    IO_REG(ElectronAbundance,       "f4", 1, 0);
    IO_REG_WRONLY(NeutralHydrogenFraction, "f4", 1, 0);
    IO_REG_WRONLY(InternalEnergy,   "f4", 1, 0);
    IO_REG_WRONLY(JUV,   "f4", 1, 0);

    /* SF */
#ifdef SFR
    IO_REG_WRONLY(StarFormationRate, "f4", 1, 0);
    IO_REG_NONFATAL(BirthDensity, "f4", 1, 4);
    IO_REG_TYPE(StarFormationTime, "f4", 1, 4);
    IO_REG_TYPE(Metallicity,       "f4", 1, 0);
    IO_REG_TYPE(Metallicity,       "f4", 1, 4);
#endif /* SFR */
#ifdef BLACK_HOLES
    /* Blackhole */
    IO_REG_TYPE(StarFormationTime, "f4", 1, 5);
    IO_REG(BlackholeMass,          "f4", 1, 5);
    IO_REG(BlackholeAccretionRate, "f4", 1, 5);
    IO_REG(BlackholeProgenitors,   "i4", 1, 5);
    IO_REG(BlackholeMinPotPos, "f8", 3, 5);
    IO_REG(BlackholeMinPotVel,   "f4", 3, 5);
    IO_REG(BlackholeJumpToMinPot,   "i4", 1, 5);
#endif
    if(All.SnapshotWithFOF)
        fof_register_io_blocks();

    /*Sort IO blocks so similar types are together*/
    qsort_openmp(IOTable.ent, IOTable.used, sizeof(struct IOTableEntry), order_by_type);
}

