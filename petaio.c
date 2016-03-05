#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <string.h>

#include "bigfile-mpi.h"
#include "allvars.h"
#include "proto.h"

#include "petaio.h"
#ifdef FOF
    /*Defined in fofpetaio.c and only used here*/
    void fof_register_io_blocks();
#endif


/************
 *
 * The IO api , intented to replace io.c and read_ic.c
 * currently we have a function to register the blocks and enumerate the blocks
 *
 */

struct IOTable IOTable = {0};

static void petaio_write_header(BigFile * bf);
static void petaio_read_header(BigFile * bf);

static void register_io_blocks();

/* these are only used in reading in */
static int64_t npartTotal[6];
static int64_t npartLocal[6];

void petaio_init() {
    register_io_blocks();
}

/* save a snapshot file */
static void petaio_save_internal(char * fname);
void petaio_save_snapshot(int num) {
    char fname[4096];
    sprintf(fname, "%s/PART_%03d", All.OutputDir, num);
    if(ThisTask == 0) {
        printf("saving snapshot into %s\n", fname);
        fflush(stdout);
    }
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


static void petaio_save_internal(char * fname) {
    BigFile bf = {0};
    if(0 != big_file_mpi_create(&bf, fname, MPI_COMM_WORLD)) {
        if(ThisTask == 0) {
            fprintf(stderr, "Failed to create snapshot at %s:%s\n", fname,
                    big_file_get_error_message());
        }
        abort();
    }
    petaio_write_header(&bf); 

    int i;
    for(i = 0; i < IOTable.used; i ++) {
        /* only process the particle blocks */
        char blockname[128];
        int ptype = IOTable.ent[i].ptype;
        BigArray array = {0};
        if(!(ptype < 6 && ptype >= 0)) {
            continue;
        }
        sprintf(blockname, "%d/%s", ptype, IOTable.ent[i].name);
        if(ThisTask == 0) {
            printf("Writing Block %s\n", blockname);
            fflush(stdout);
        }
        petaio_build_buffer(&array, &IOTable.ent[i], NULL, 0);
        petaio_save_block(&bf, blockname, &array, All.NumFilesPerSnapshot, All.NumWritersPerSnapshot);
        petaio_destroy_buffer(&array);
    }
    if(0 != big_file_mpi_close(&bf, MPI_COMM_WORLD)){
        if(ThisTask == 0) {
            fprintf(stderr, "Failed to close snapshot at %s:%s\n", fname,
                    big_file_get_error_message());
        }
        abort();
    }
}

void petaio_read_internal(char * fname, int ic) {
    int ptype;
    int i;
    BigFile bf = {0};
    if(0 != big_file_mpi_open(&bf, fname, MPI_COMM_WORLD)) {
        if(ThisTask == 0) {
            fprintf(stderr, "Failed to open snapshot at %s:%s\n", fname,
                    big_file_get_error_message());
        }
        abort();
    }
    petaio_read_header(&bf); 

    allocate_memory();

    for(ptype = 0; ptype < 6; ptype ++) {
        ptrdiff_t offset = ThisTask * npartTotal[ptype] / NTask;
        ptrdiff_t num = (ThisTask + 1) * npartTotal[ptype] / NTask - offset;
        npartLocal[ptype] = num;
        NumPart += num;
    }
    N_sph = npartLocal[0];
    N_bh = npartLocal[5];

    /* check */
    if(N_sph > All.MaxPartSph) {
        fprintf(stderr, "Task %d overwhelmed by sph: %d > %d\n", ThisTask, N_sph, All.MaxPartSph);
        abort();
    }
    if(N_bh > All.MaxPartBh) {
        fprintf(stderr, "Task %d overwhelmed by bh: %d > %d\n", ThisTask, N_bh, All.MaxPartBh);
        abort();
    }
    if(NumPart >= All.MaxPart) {
        fprintf(stderr, "Task %d overwhelmed by part: %d > %d\n", ThisTask, N_bh, All.MaxPart);
        abort();
    }


    /* set up the memory topology */
    int offset = 0;
    for(ptype = 0; ptype < 6; ptype ++) {
#pragma omp parallel for
        for(i = 0; i < npartLocal[ptype]; i++)
        {
            int j = offset + i;
            P[j].Type = ptype;
            P[j].PI = i;
        }
        offset += npartLocal[ptype];
    }

    for(i = 0; i < IOTable.used; i ++) {
        /* only process the particle blocks */
        char blockname[128];
        int ptype = IOTable.ent[i].ptype;
        BigArray array = {0};
        if(!(ptype < 6 && ptype >= 0)) {
            continue;
        }
        if(npartTotal[ptype] == 0) continue;
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
        petaio_alloc_buffer(&array, &IOTable.ent[i], npartLocal[ptype]);
        petaio_read_block(&bf, blockname, &array);
        petaio_readout_buffer(&array, &IOTable.ent[i]);
        petaio_destroy_buffer(&array);
    }
    if(0 != big_file_mpi_close(&bf, MPI_COMM_WORLD)) {
        if(ThisTask == 0) {
            fprintf(stderr, "Failed to close snapshot at %s:%s\n", fname,
                    big_file_get_error_message());
        }
        abort();
    }

    /* set up the cross check for BH ID */
#pragma omp parallel for
    for(i = 0; i < NumPart; i++)
    {
        if(P[i].Type == 5) {
            BhP[P[i].PI].ID = P[i].ID;
        }
    }
}


void petaio_read_snapshot(int num) {
    char fname[4096];
    sprintf(fname, "%s/PART_%03d", All.OutputDir, num);

    memset(&header, 0, sizeof(header));
    /* 
     * we always save the Entropy, notify init.c not to mess with the entorpy
     * */
    header.flag_entropy_instead_u = 1;
    petaio_read_internal(fname, 0);
}

/* Notice that IC follows the Gadget-1/2 unit, but snapshots use
 * internal units. Old code converts at init.c; we now convert 
 * here in read_ic.
 * */
void petaio_read_ic() {
    int i;
    memset(&header, 0, sizeof(header));
    /* 
     *  IC doesn't have entropy or energy; always use the
     *  InitTemp in paramfile, then use init.c to convert to
     *  entropy.
     * */
    header.flag_entropy_instead_u = 0;
    petaio_read_internal(All.InitCondFile, 1);

    /* touch up the mass -- IC files save mass in header */
    for(i = 0; i < NumPart; i++)
    {
        P[i].Mass = All.MassTable[P[i].Type];
    }

    double u_init = (1.0 / GAMMA_MINUS1) * (BOLTZMANN / PROTONMASS) * All.InitGasTemp;
    u_init *= All.UnitMass_in_g / All.UnitEnergy_in_cgs;	/* unit conversion */

    double molecular_weight;
    if(All.InitGasTemp > 1.0e4)	/* assuming FULL ionization */
        molecular_weight = 4 / (8 - 5 * (1 - HYDROGEN_MASSFRAC));
    else				/* assuming NEUTRAL GAS */
        molecular_weight = 4 / (1 + 3 * HYDROGEN_MASSFRAC);

    u_init /= molecular_weight;

    All.InitGasU = u_init;

    for(i = 0; i < N_sph; i++)
        SPHP(i).Entropy = 0;

    if(All.InitGasTemp > 0)
    {
        for(i = 0; i < N_sph; i++)
        {
            if(SPHP(i).Entropy == 0)
                SPHP(i).Entropy = All.InitGasU;

            /* Note: the coversion to entropy will be done in the function init(),
               after the densities have been computed */
        }
    }

#pragma omp parallel for
    for(i = 0; i < NumPart; i++) {
        int k;
        /* for GenIC's Gadget-1 snapshot Unit to Gadget-2 Internal velocity unit */
        for(k = 0; k < 3; k++)
            P[i].Vel[k] *= sqrt(All.TimeBegin) * All.TimeBegin;
    }

}


/* write a header block */
static void petaio_write_header(BigFile * bf) {
    BigBlock bh = {0};
    if(0 != big_file_mpi_create_block(bf, &bh, "header", NULL, 0, 0, 0, MPI_COMM_WORLD)) {
        if(ThisTask == 0) {
            fprintf(stderr, "Failed to create block at %s:%s\n", "header",
                    big_file_get_error_message());
        }
        abort();
    }
    int i;
    int k;
    int64_t npartLocal[6];
    int64_t npartTotal[6];

    for (k = 0; k < 6; k ++) {
        npartLocal[k] = 0;
    }
    for (i = 0; i < NumPart; i ++) {
        npartLocal[P[i].Type] ++;
    }

    MPI_Allreduce(npartLocal, npartTotal, 6, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
    if( 
    (0 != big_block_set_attr(&bh, "TotNumPart", npartTotal, "u8", 6)) ||
    (0 != big_block_set_attr(&bh, "MassTable", All.MassTable, "f8", 6)) ||
    (0 != big_block_set_attr(&bh, "Time", &All.Time, "f8", 1)) ||
    (0 != big_block_set_attr(&bh, "BoxSize", &All.BoxSize, "f8", 1)) ||
    (0 != big_block_set_attr(&bh, "OmegaLambda", &All.OmegaLambda, "f8", 1)) ||
    (0 != big_block_set_attr(&bh, "Omega0", &All.Omega0, "f8", 1)) ||
    (0 != big_block_set_attr(&bh, "HubbleParam", &All.HubbleParam, "f8", 1)) ) {
        if(ThisTask == 0) {
            fprintf(stderr, "Failed to write attributes %s\n",
                    big_file_get_error_message());
        }
        abort();
    }

    if(0 != big_block_mpi_close(&bh, MPI_COMM_WORLD)) {
        if(ThisTask == 0) {
            fprintf(stderr, "Failed to close block %s\n",
                    big_file_get_error_message());
        }
        abort();
    }
}

static void petaio_read_header(BigFile * bf) {
    BigBlock bh = {0};
    if(0 != big_file_mpi_open_block(bf, &bh, "header", MPI_COMM_WORLD)) {
        if(ThisTask == 0) {
            fprintf(stderr, "Failed to create block at %s:%s\n", "header",
                    big_file_get_error_message());
        }
        abort();
    }
    double Time;
    double BoxSize;
    int k;
    if(
    (0 != big_block_get_attr(&bh, "TotNumPart", npartTotal, "u8", 6)) ||
    (0 != big_block_get_attr(&bh, "MassTable", All.MassTable, "f8", 6)) ||
    (0 != big_block_get_attr(&bh, "Time", &Time, "f8", 1)) ||
    (0 != big_block_get_attr(&bh, "BoxSize", &BoxSize, "f8", 1))) {
        if(ThisTask == 0) {
            fprintf(stderr, "Failed to read attr: %s\n",
                    big_file_get_error_message());
        }
        abort();
    }

    All.TotNumPart = 0;
    for(k = 0; k < 6; k ++) {
        All.TotNumPart += npartTotal[k];
    }
    All.TotN_sph = npartTotal[0];
    All.TotN_bh = npartTotal[5];
    if(ThisTask == 0) {
        printf("Total number of particles: %018ld\n", All.TotNumPart);
        printf("Total number of gas particles: %018ld\n", All.TotN_sph);
        printf("Total number of bh particles: %018ld\n", All.TotN_bh);
    }

    if(RestartFlag >= 2) {
        All.TimeBegin = Time;
        set_global_time(All.TimeBegin);
    }

    if(fabs(BoxSize - All.BoxSize) / All.BoxSize > 1e-6) {
        if(ThisTask == 0) {
            fprintf(stderr, "BoxSize mismatch %g, snapfile has %g\n", All.BoxSize, BoxSize);
        }
        abort();
    }
    /*FIXME: check others as well */
    /*
    big_block_get_attr(&bh, "OmegaLambda", &All.OmegaLambda, "f8", 1);
    big_block_get_attr(&bh, "Omega0", &All.Omega0, "f8", 1);
    big_block_get_attr(&bh, "HubbleParam", &All.HubbleParam, "f8", 1);
    */
    if(0 != big_block_mpi_close(&bh, MPI_COMM_WORLD)) {
        if(ThisTask == 0) {
            fprintf(stderr, "Failed to close block: %s\n",
                    big_file_get_error_message());
        }
        abort();
    }
    /* sets the maximum number of particles that may reside on a processor */
    All.MaxPart = (int) (All.PartAllocFactor * (All.TotNumPart / NTask));	
    All.MaxPartSph = (int) (All.PartAllocFactor * (All.TotN_sph / NTask));	

#ifdef INHOMOG_GASDISTR_HINT
    if(All.TotN_sph > 0) {
        All.MaxPartSph = All.MaxPart;
    }
#endif
    /* at most 10% of SPH can form BH*/
    All.MaxPartBh = (int) (0.1 * All.MaxPartSph);	

}

void petaio_alloc_buffer(BigArray * array, IOTableEntry * ent, int64_t npartLocal) {
    size_t dims[2];
    ptrdiff_t strides[2];
    int elsize = dtype_itemsize(ent->dtype);

    dims[0] = npartLocal;
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
 * only check P[ selection[i]]
*/
void petaio_build_buffer(BigArray * array, IOTableEntry * ent, int * selection, int NumSelection) {

/* This didn't work with CRAY:
 * always has npartLocal = 0
 * after the loop if openmp is used;
 * but I can't reproduce this with a striped version
 * of code. need to investigate.
 * #pragma omp parallel for reduction(+: npartLocal)
 */
    int npartThread[All.NumThreads];
    int offsetThread[All.NumThreads];
#pragma omp parallel
    {
        int i;
        int tid = omp_get_thread_num();
        int NT = omp_get_num_threads();
        if(NT > All.NumThreads) abort();
        int start = (selection?((size_t) NumSelection):((size_t)NumPart)) * tid / NT;
        int end = (selection?((size_t) NumSelection):((size_t)NumPart)) * (tid + 1) / NT;
        int npartLocal = 0;
        npartThread[tid] = 0;
        for(i = start; i < end; i ++) {
            int j = selection?selection[i]:i;
            if(P[j].Type != ent->ptype) continue;
            npartThread[tid] ++;
        }
#pragma omp barrier
        offsetThread[0] = 0;
        for(i = 1; i < NT; i ++) {
            offsetThread[i] = offsetThread[i - 1] + npartThread[i - 1];
        }
        for(i = 0; i < NT; i ++) {
            npartLocal += npartThread[i];
        }
#pragma omp master 
        {
        /* don't forget to free buffer after its done*/
            petaio_alloc_buffer(array, ent, npartLocal);
        }
#pragma omp barrier
#if 0
        printf("Thread = %d offset=%d count=%d start=%d end=%d %d\n", tid, offsetThread[tid], npartThread[tid], start, end, npartLocal);
#endif
        /* fill the buffer */
        char * p = array->data;
        p += array->strides[0] * offsetThread[tid];
        for(i = start; i < end; i ++) {
            int j = selection?selection[i]:i;
            if(P[j].Type != ent->ptype) continue;
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
    MPI_Comm GROUP;
    int GroupSize;
    int ThisColor;
    int ThisKey;
    /* Split the wolrd into writer groups */
    ThisColor = ThisTask * All.NumWritersPerSnapshot/ NTask;
    MPI_Comm_split(MPI_COMM_WORLD, ThisColor, 0, &GROUP);
    MPI_Comm_rank(GROUP, &ThisKey);
    MPI_Comm_size(GROUP, &GroupSize);

    BigBlock bb = {0};
    int i;
    BigBlockPtr ptr;

    int64_t offset = count_to_offset(array->dims[0]);

    /* open the block */
    if(0 != big_file_mpi_open_block(bf, &bb, blockname, MPI_COMM_WORLD)) {
        if(ThisTask == 0) {
            fprintf(stderr, "Failed to open block at %s:%s\n", blockname,
                    big_file_get_error_message());
        }
        abort();
    }
    

    /* read the buffers one by one in each writer group */
    for(i = 0; i < GroupSize; i ++) {
        MPI_Barrier(GROUP);
        if(i != ThisKey) continue;
        if(0 != big_block_seek(&bb, &ptr, offset)) {
            fprintf(stderr, "Failed to seek: %s\n", big_file_get_error_message());
            abort();
        }
        //printf("Task = %d, writing at %td\n", ThisTask, offsetLocal);
        if(0 != big_block_read(&bb, &ptr, array)) {
            fprintf(stderr, "Failed to readform  block, ThisTask = %d: %s\n", ThisTask, big_file_get_error_message());
            endrun(12345);
        }
    }

    if(0 != big_block_mpi_close(&bb, MPI_COMM_WORLD)) {
        if(ThisTask == 0) {
            fprintf(stderr, "Failed to close block at %s:%s\n", blockname,
                    big_file_get_error_message());
        }
        abort();
    }
    MPI_Comm_free(&GROUP);
}

/* save a block to disk */
void petaio_save_block(BigFile * bf, char * blockname, BigArray * array, int NumFiles, int NumWriters) {

    MPI_Comm GROUP;
    int GroupSize;
    int ThisColor;
    int ThisKey;
    /* Split the wolrd into writer groups */
    ThisColor = ThisTask * NumWriters / NTask;
    MPI_Comm_split(MPI_COMM_WORLD, ThisColor, 0, &GROUP);
    MPI_Comm_rank(GROUP, &ThisKey);
    MPI_Comm_size(GROUP, &GroupSize);

    BigBlock bb = {0};
    int i;
    BigBlockPtr ptr;

    int64_t offset = count_to_offset(array->dims[0]);
    size_t size = count_sum(array->dims[0]);

    if(ThisTask == 0) {
        printf("Will write %td particles to %d Files\n", size, NumFiles);
    }
    /* skip the 0 size blocks */
    if(size == 0) return;
    /* create the block */
    /* dims[1] is the number of members per item */
    if(0 != big_file_mpi_create_block(bf, &bb, blockname, array->dtype, array->dims[1], NumFiles, size, MPI_COMM_WORLD)) {
        if(ThisTask == 0) {
            fprintf(stderr, "Failed to create block at %s:%s\n", blockname,
                    big_file_get_error_message());
        }
        abort();
    }
    
    /* write the buffers one by one in each writer group */
    for(i = 0; i < GroupSize; i ++) {
        MPI_Barrier(GROUP);
        if(i != ThisKey) continue;
        if(0 != big_block_seek(&bb, &ptr, offset)) {
            fprintf(stderr, "Failed to seek:%s\n", big_file_get_error_message());
            endrun(124455);
        }
        //printf("Task = %d, writing at %td\n", ThisTask, offsetLocal);
        if(0 != big_block_write(&bb, &ptr, array)) {
            fprintf(stderr, "Failed to write, ThisTask=%d:%s\n", ThisTask, big_file_get_error_message());
            endrun(124455);
        }
    }

    if(ThisTask == 0) {
        printf("Done writing %td particles to %d Files\n", size, NumFiles);
    }

    if(0 != big_block_mpi_close(&bb, MPI_COMM_WORLD)) {
        if(ThisTask == 0) {
            fprintf(stderr, "Failed to close block at %s:%s\n", blockname,
                    big_file_get_error_message());
        }
        abort();
    }
    MPI_Comm_free(&GROUP);
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
SIMPLE_PROPERTY(Velocity, P[i].Vel[0], float, 3)
SIMPLE_PROPERTY(Mass, P[i].Mass, float, 1)
SIMPLE_PROPERTY(ID, P[i].ID, uint64_t, 1)
SIMPLE_PROPERTY(Generation, P[i].Generation, unsigned char, 1)
SIMPLE_PROPERTY(Potential, P[i].Potential, float, 1)
SIMPLE_PROPERTY(SmoothingLength, P[i].Hsml, float, 1)
SIMPLE_PROPERTY(Density, SPHP(i).Density, float, 1)
#ifdef DENSITY_INDEPENDENT_SPH
SIMPLE_PROPERTY(EgyWtDensity, SPHP(i).EgyWtDensity, float, 1)
SIMPLE_PROPERTY(Entropy, SPHP(i).Entropy, float, 1)
SIMPLE_GETTER(GTPressure, SPHP(i).Pressure, float, 1)
#endif
#ifdef COOLING
SIMPLE_PROPERTY(ElectronAbundance, SPHP(i).Ne, float, 1)
#endif
#ifdef SFR
#ifdef STELLARAGE
SIMPLE_PROPERTY(StarFormationTime, P[i].StellarAge, float, 1)
#endif
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
#ifdef BH_ACCRETION
SIMPLE_PROPERTY(BlackholeAccretionRate, BHP(i).Mdot, float, 1)
#endif
#ifdef BH_COUNTPROGS
SIMPLE_PROPERTY(BlackholeProgenitors, BHP(i).CountProgs, float, 1)
#endif
#endif
#ifdef FOF
SIMPLE_GETTER(GTGroupID, P[i].GrNr, uint32_t, 1)
#endif
#ifdef COOLING
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
#endif
static void register_io_blocks() {
    int i;
    /* Bare Bone Gravity*/
    for(i = 0; i < 6; i ++) {
        IO_REG(Position, "f8", 3, i);
        IO_REG(Velocity, "f4", 3, i);
        IO_REG(Mass,     "f4", 1, i);
        IO_REG(ID,       "u8", 1, i);
        IO_REG(Generation,       "u1", 1, i);
        IO_REG(Potential, "f4", 1, i);
#ifdef FOF
        IO_REG_WRONLY(GroupID, "u4", 1, i);
#endif
    }

    /* Bare Bone SPH*/
    IO_REG(SmoothingLength,  "f4", 1, 0);
    IO_REG(Density,          "f4", 1, 0);
#ifdef DENSITY_INDEPENDENT_SPH
    IO_REG(EgyWtDensity,          "f4", 1, 0);
    IO_REG(Entropy,          "f4", 1, 0);
    IO_REG_WRONLY(Pressure,         "f4", 1, 0);
#endif

    /* Cooling */
#ifdef COOLING
    IO_REG(ElectronAbundance,       "f4", 1, 0);
    IO_REG_WRONLY(NeutralHydrogenFraction, "f4", 1, 0);
    IO_REG_WRONLY(InternalEnergy,   "f4", 1, 0);
    IO_REG_WRONLY(JUV,   "f4", 1, 0);
#endif
//    IO_REG_WRONLY(JUV,   "f4", 1, 0);

    /* SF */
#ifdef SFR
    IO_REG_WRONLY(StarFormationRate, "f4", 1, 0);
#ifdef STELLARAGE
    IO_REG(StarFormationTime, "f4", 1, 4);
#endif
#ifdef METALS
    IO_REG(Metallicity,       "f4", 1, 0);
    IO_REG(Metallicity,       "f4", 1, 4);
#endif /* METALS */
#endif /* SFR */
#ifdef BLACK_HOLES
    /* Blackhole */
    IO_REG(BlackholeMass,          "f8", 1, 5);
    IO_REG(StarFormationTime, "f4", 1, 5);
#ifdef BH_ACCRETION
    IO_REG(BlackholeAccretionRate, "f4", 1, 5);
#endif
#ifdef BH_COUNTPROGS
    IO_REG(BlackholeProgenitors,   "i4", 1, 5);
#endif
#endif
#ifdef FOF
    fof_register_io_blocks();
#endif
}

