#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <string.h>
#include <stdarg.h>

#include "bigfile-mpi.h"
#include "allvars.h"
#include "proto.h"
#include "system.h"
#include "sfr_eff.h"
#include "cooling.h"

#include "petaio.h"
#include "mymalloc.h"
#include "openmpsort.h"
#include "utils-string.h"
#include "endrun.h"

/*Defined in fofpetaio.c and only used here*/
void fof_register_io_blocks();

/************
 *
 * The IO api , intented to replace io.c and read_ic.c
 * currently we have a function to register the blocks and enumerate the blocks
 *
 */

struct IOTable IOTable = {0};

static void petaio_write_header(BigFile * bf);
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

    for(i = 1; i < 6; i ++) {
        ptype_offset[i] = ptype_offset[i-1] + NLocal[i - 1];
        ptype_count[i] = 0;
    }

    for(i = 0; i < NumPart; i ++) {
        int ptype = P[i].Type;
        if((select_func == NULL) || (select_func(i) != 0)) {
            selection[ptype_offset[ptype] + ptype_count[ptype]] = i;
            ptype_count[ptype] ++;
        }
    }
}

static void petaio_save_internal(char * fname) {
    BigFile bf = {0};
    if(0 != big_file_mpi_create(&bf, fname, MPI_COMM_WORLD)) {
        endrun(0, "Failed to create snapshot at %s:%s\n", fname,
                    big_file_get_error_message());
    }
    petaio_write_header(&bf); 

    int ptype_offset[6]={0};
    int ptype_count[6]={0};

    int * selection = mymalloc("Selection", sizeof(int) * NumPart);

    petaio_build_selection(selection, ptype_offset, ptype_count, NumPart, NULL);
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
    message(0, "Reading snapshot %s\n", fname);

    if(0 != big_file_mpi_open(&bf, fname, MPI_COMM_WORLD)) {
        endrun(0, "Failed to open snapshot at %s:%s\n", fname,
                    big_file_get_error_message());
    }

    allocate_memory();

    /* set up the memory topology */
    int offset = 0;
    for(ptype = 0; ptype < 6; ptype ++) {
#pragma omp parallel for
        for(i = 0; i < NLocal[ptype]; i++)
        {
            int j = offset + i;
            P[j].Type = ptype;
            P[j].PI = i;
        }
        offset += NLocal[ptype];
    }

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
            if( strcmp(IOTable.ent[i].name, "Position") &&
                strcmp(IOTable.ent[i].name, "Velocity") &&
                strcmp(IOTable.ent[i].name, "ID")) continue;
        }
        if(IOTable.ent[i].setter == NULL) {
            /* FIXME: do not know how to read this block; assume the fucker is
             * internally intialized; */
            continue;
        }
        sprintf(blockname, "%d/%s", ptype, IOTable.ent[i].name);
        petaio_alloc_buffer(&array, &IOTable.ent[i], NLocal[ptype]);
        petaio_read_block(&bf, blockname, &array);
        petaio_readout_buffer(&array, &IOTable.ent[i]);
        petaio_destroy_buffer(&array);
    }
    if(0 != big_file_mpi_close(&bf, MPI_COMM_WORLD)) {
        endrun(0, "Failed to close snapshot at %s:%s\n", fname,
                    big_file_get_error_message());
    }

    /* set up the cross check for BH ID */
#pragma omp parallel for
    for(i = 0; i < NumPart; i++)
    {
        if(P[i].Type == 5) {
            BhP[P[i].PI].base.ID = P[i].ID;
        }
        if(P[i].Type == 0) {
            SPHP(i).base.ID = P[i].ID;
        }
    }
}
void
petaio_read_header(int num)
{
    BigFile bf = {0};

    char * fname;
    if(num == -1) {
        fname = fastpm_strdup_printf("%s", All.InitCondFile);
    } else {
        fname = fastpm_strdup_printf("%s/PART_%03d", All.OutputDir, num);
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
        for(i = 0; i < NumPart; i++)
        {
            P[i].Mass = All.MassTable[P[i].Type];
        }

        if (!All.IO.UsePeculiarVelocity ) {

            /* fixing the unit of velocity from Legacy GenIC IC */
            #pragma omp parallel for
            for(i = 0; i < NumPart; i++) {
                int k;
                /* for GenIC's Gadget-1 snapshot Unit to Gadget-2 Internal velocity unit */
                for(k = 0; k < 3; k++)
                    P[i].Vel[k] *= sqrt(All.cf.a) * All.cf.a;
            }

        }
    } else {
        fname = fastpm_strdup_printf("%s/PART_%03d", All.OutputDir, num);
        /*
         * we always save the Entropy, init.c will not mess with the entropy
         * */
        petaio_read_internal(fname, 0);
    }
    free(fname);
}


/*Defined in config.h and pulled in via main.c*/
extern const char * COMPILETIMESETTINGS;
extern const char * GADGETVERSION;

/* write a header block */
static void petaio_write_header(BigFile * bf) {
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
    (0 != big_block_set_attr(&bh, "CodeVersion", GADGETVERSION, "S1", strlen(GADGETVERSION))) ||
    (0 != big_block_set_attr(&bh, "CompileSettings", COMPILETIMESETTINGS, "S1", strlen(COMPILETIMESETTINGS))) ||
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
    /* sets the maximum number of particles that may reside on a processor */
    All.MaxPart = (int) (All.PartAllocFactor * All.TotNumPartInit / NTask);	
    /* at most 10% of particles can form BH*/
    All.MaxPartBh = (int) (0.1 * All.MaxPart);

    for(ptype = 0; ptype < 6; ptype ++) {
        int64_t start = ThisTask * NTotal[ptype] / NTask;
        int64_t end = (ThisTask + 1) * NTotal[ptype] / NTask;
        NLocal[ptype] = end - start;
        NumPart += end - start;
    }
    N_sph_slots = NLocal[0];
    N_bh_slots = NLocal[5];

    if(N_bh_slots > All.MaxPartBh) {
        endrun(1, "Overwhelmed by bh: %d > %d\n", N_bh_slots, All.MaxPartBh);
    }

    if(NumPart >= All.MaxPart) {
        endrun(1, "Overwhelmed by part: %d > %d\n", NumPart, All.MaxPart);
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
    for(i = 0; i < NumPart; i ++) {
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
        endrun(-1, "NULL seletion is not supported\n");
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
void petaio_read_block(BigFile * bf, char * blockname, BigArray * array) {
    BigBlock bb;
    BigBlockPtr ptr;

    /* open the block */
    if(0 != big_file_mpi_open_block(bf, &bb, blockname, MPI_COMM_WORLD)) {
        endrun(0, "Failed to open block at %s:%s\n", blockname,
                big_file_get_error_message());
    }
    
    if(0 != big_block_seek(&bb, &ptr, 0)) {
        endrun(1, "Failed to seek: %s\n", big_file_get_error_message());
    }

    if(0 != big_block_mpi_read(&bb, &ptr, array, All.IO.NumWriters, MPI_COMM_WORLD)) {
        endrun(1, "Failed to read from block: %s\n", big_file_get_error_message());
    }

    if(0 != big_block_mpi_close(&bb, MPI_COMM_WORLD)) {
        endrun(0, "Failed to close block at %s:%s\n", blockname,
                    big_file_get_error_message());
    }
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
        property_setter setter
        ) {
    IOTableEntry * ent = &IOTable.ent[IOTable.used];
    strcpy(ent->name, name);
    ent->ptype = ptype;
    dtype_normalize(ent->dtype, dtype);
    ent->getter = getter;
    ent->setter = setter;
    ent->items = items;
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
SIMPLE_PROPERTY(StarFormationTime, P[i].StarFormationTime, float, 1)
#ifdef METALS
SIMPLE_PROPERTY(Metallicity, P[i].Metallicity, float, 1)
#endif
static void GTStarFormationRate(int i, float * out) {
    /* Convert to Solar/year */
    *out = get_starformation_rate(i) 
        * ((All.UnitMass_in_g / SOLAR_MASS) / (All.UnitTime_in_s / SEC_PER_YEAR));
}
#endif
#ifdef BLACK_HOLES
SIMPLE_PROPERTY(BlackholeMass, BHP(i).Mass, float, 1)
SIMPLE_PROPERTY(BlackholeAccretionRate, BHP(i).Mdot, float, 1)
SIMPLE_PROPERTY(BlackholeProgenitors, BHP(i).CountProgs, float, 1)
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
        IO_REG(Generation,       "u1", 1, i);
        if(All.OutputPotential)
            IO_REG_WRONLY(Potential, "f4", 1, i);
        if(All.SnapshotWithFOF)
            IO_REG_WRONLY(GroupID, "u4", 1, i);
    }

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
#ifdef WINDS
    IO_REG(StarFormationTime, "f4", 1, 4);
#endif
#ifdef METALS
    IO_REG(Metallicity,       "f4", 1, 0);
    IO_REG(Metallicity,       "f4", 1, 4);
#endif /* METALS */
#endif /* SFR */
#ifdef BLACK_HOLES
    /* Blackhole */
    IO_REG(BlackholeMass,          "f4", 1, 5);
    IO_REG(StarFormationTime, "f4", 1, 5);
    IO_REG(BlackholeAccretionRate, "f4", 1, 5);
    IO_REG(BlackholeProgenitors,   "i4", 1, 5);
#endif
    if(All.SnapshotWithFOF)
        fof_register_io_blocks();

    /*Sort IO blocks so similar types are together*/
    qsort_openmp(IOTable.ent, IOTable.used, sizeof(struct IOTableEntry), order_by_type);
}

