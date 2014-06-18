#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <string.h>

#include "bigfile/bigfile-mpi.h"
#include "allvars.h"
#include "proto.h"

#include "petaio.h"

/************
 *
 * The IO api , intented to replace io.c and read_ic.c
 * currently we have a function to register the blocks and enumerate the blocks
 *
 */

struct IOTable IOTable = {0};

static void petaio_write_header(BigFile * bf);

static int ThisColor;
static int ThisKey;
static MPI_Comm GROUP;
static int GroupSize;
static int NumFiles;
static void register_io_blocks();

void petaio_init() {
    /* Split the wolrd into writer groups */
    NumFiles = All.NumFilesWrittenInParallel;
    ThisColor = ThisTask * NumFiles / NTask;
    MPI_Comm_split(MPI_COMM_WORLD, ThisColor, 0, &GROUP);
    MPI_Comm_rank(GROUP, &ThisKey);
    MPI_Comm_size(GROUP, &GroupSize);

    register_io_blocks();
}

/* save a snapshot file */
void petaio_save_snapshot(int num) {
    char fname[4096];
    sprintf(fname, "%s/PART_%03d", All.OutputDir, num);

    BigFile bf = {0};
    if(0 != big_file_mpi_create(&bf, fname, MPI_COMM_WORLD)) {
        if(ThisTask == 0) {
            fprintf(stderr, "Failed to create snapshot at %s\n", fname);
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
        petaio_build_buffer(&array, &IOTable.ent[i], NULL);
        petaio_save_block(&bf, blockname, &array);
        petaio_destroy_buffer(&array);
    }
    big_file_mpi_close(&bf, MPI_COMM_WORLD);
}



/* write a header block */
static void petaio_write_header(BigFile * bf) {
    BigBlock bh = {0};
    if(0 != big_file_mpi_create_block(bf, &bh, "header", NULL, 0, 0, 0, MPI_COMM_WORLD)) {
        fprintf(stderr, "Failed to create header\n");
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
    
    big_block_set_attr(&bh, "NumPartTotal", npartTotal, "u8", 6);
    big_block_set_attr(&bh, "MassTable", All.MassTable, "f8", 6);
    big_block_set_attr(&bh, "Time", &All.Time, "f8", 1);
    big_block_set_attr(&bh, "BoxSize", &All.BoxSize, "f8", 1);
    big_block_set_attr(&bh, "OmegaLambda", &All.OmegaLambda, "f8", 1);
    big_block_set_attr(&bh, "Omega0", &All.Omega0, "f8", 1);
    big_block_set_attr(&bh, "HubbleParam", &All.HubbleParam, "f8", 1);
    big_block_mpi_close(&bh, MPI_COMM_WORLD);
}

/* build an IO buffer for block, based on selection function select
 * 0 is to exclude
 * !0 is to include */
void petaio_build_buffer(BigArray * array, IOTableEntry * ent, petaio_selection select) {
    size_t dims[2];
    ptrdiff_t strides[2];
    int elsize = dtype_itemsize(ent->dtype);

    int64_t npartLocal = 0;
    int i;

    for(i = 0; i < NumPart; i ++) {
        if(P[i].Type != ent->ptype) continue;
        if(select && !select(i)) continue;
        npartLocal ++;
    }

    dims[0] = npartLocal;
    dims[1] = ent->items;
    strides[1] = elsize;
    strides[0] = elsize * ent->items;

    /* create the buffer */
    char * buffer = malloc(dims[0] * dims[1] * elsize);
    /* don't forget to free buffer after its done*/
    big_array_init(array, buffer, ent->dtype, 2, dims, strides);

    /* fill the buffer */
    char * p = buffer;
    for(i = 0; i < NumPart; i ++) {
        if(P[i].Type != ent->ptype) continue;
        if(select && !select(i)) continue;
        ent->getter(i, p);
        p += strides[0];
    }
}

/* destropy a buffer, freeing its memory */
void petaio_destroy_buffer(BigArray * array) {
    free(array->data);
}

/* save a block to disk */
void petaio_save_block(BigFile * bf, char * blockname, BigArray * array) {
    BigBlock bb = {0};
    int i;
    int k;
    BigBlockPtr ptr;

    int64_t offset = count_to_offset(array->dims[0]);
    size_t size = count_sum(array->dims[0]);

    /* create the block */
    /* dims[1] is the number of members per item */
    big_file_mpi_create_block(bf, &bb, blockname, array->dtype, array->dims[1], NumFiles, size, MPI_COMM_WORLD);
    
    if(ThisTask == 0) {
        printf("Saving block %s  as %s: (%ld, %td)\n", blockname, array->dtype, 
                size, array->dims[1]);
    }

    /* write the buffers one by one in each writer group */
    for(i = 0; i < GroupSize; i ++) {
        MPI_Barrier(GROUP);
        if(i != ThisKey) continue;
        if(0 != big_block_seek(&bb, &ptr, offset)) {
            fprintf(stderr, "Failed to seek\n");
            abort();
        }
        //printf("Task = %d, writing at %td\n", ThisTask, offsetLocal);
        big_block_write(&bb, &ptr, array);
    }

    big_block_mpi_close(&bb, MPI_COMM_WORLD);
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
        property_getter getter) {
    IOTableEntry * ent = &IOTable.ent[IOTable.used];
    strcpy(ent->name, name);
    ent->ptype = ptype;
    dtype_normalize(ent->dtype, dtype);
    ent->getter = getter;
    ent->items = items;
    IOTable.used ++;
}

SIMPLE_GETTER(GTPosition, P[i].Pos[0], double, 3)
SIMPLE_GETTER(GTGroupID, P[i].GrNr, uint32_t, 1)
SIMPLE_GETTER(GTVelocity, P[i].Vel[0], float, 3)
SIMPLE_GETTER(GTMass, P[i].Mass, float, 1)
SIMPLE_GETTER(GTID, P[i].ID, uint64_t, 1)
SIMPLE_GETTER(GTPotential, P[i].p.Potential, float, 1)
SIMPLE_GETTER(GTSmoothingLength, P[i].Hsml, float, 1)
SIMPLE_GETTER(GTDensity, SPHP(i).d.Density, float, 1)
SIMPLE_GETTER(GTPressure, SPHP(i).Pressure, float, 1)
SIMPLE_GETTER(GTEntropy, SPHP(i).Entropy, float, 1)
SIMPLE_GETTER(GTElectronAbundance, SPHP(i).Ne, float, 1)
SIMPLE_GETTER(GTStarFormationTime, P[i].StellarAge, float, 1)
SIMPLE_GETTER(GTMetallicity, P[i].Metallicity, float, 1)
SIMPLE_GETTER(GTBlackholeMass, BHP(i).Mass, float, 1)
SIMPLE_GETTER(GTBlackholeAccretionRate, BHP(i).Mdot, float, 1)
SIMPLE_GETTER(GTBlackholeProgenitors, BHP(i).CountProgs, float, 1)

static void GTStarFormationRate(int i, float * out) {
    /* Convert to Solar/year */
    *out = get_starformation_rate(i) 
        * ((All.UnitMass_in_g / SOLAR_MASS) / (All.UnitTime_in_s / SEC_PER_YEAR));
}
static void GTNeutralHydrogenFraction(int i, float * out) {
    double ne, nh0, nHeII;
    ne = SPHP(i).Ne;

    AbundanceRatios(DMAX(All.MinEgySpec,
                SPHP(i).Entropy / GAMMA_MINUS1 * pow(SPHP(i).EOMDensity *
                    All.cf.a3inv,
                    GAMMA_MINUS1)),
            SPHP(i).d.Density * All.cf.a3inv, &ne, &nh0, &nHeII);
    *out = nh0;
} 
static void GTInternalEnergy(int i, float * out) {
    *out = DMAX(All.MinEgySpec,
        SPHP(i).Entropy / GAMMA_MINUS1 * pow(SPHP(i).EOMDensity * All.cf.a3inv, GAMMA_MINUS1));
}

static void register_io_blocks() {
    int i;
    /* Bare Bone Gravity*/
    for(i = 0; i < 6; i ++) {
        IO_REG(Position, "f8", 3, i);
        IO_REG(Velocity, "f4", 3, i);
        IO_REG(Mass,     "f4", 1, i);
        IO_REG(ID,       "u8", 1, i);
        IO_REG(Potential, "f4", 1, i);
        IO_REG(GroupID, "u4", 1, i);
    }

    /* Bare Bone SPH*/
    IO_REG(SmoothingLength,  "f4", 1, 0);
    IO_REG(Density,          "f4", 1, 0);
    IO_REG(Pressure,         "f4", 1, 0);
    IO_REG(Entropy,          "f4", 1, 0);

    /* Cooling */
    IO_REG(ElectronAbundance,       "f4", 1, 0);
    IO_REG(NeutralHydrogenFraction, "f4", 1, 0);
    IO_REG(InternalEnergy,   "f4", 1, 0);

    /* SF */
    IO_REG(StarFormationRate, "f4", 1, 0);
    IO_REG(StarFormationTime, "f4", 1, 0);
    IO_REG(Metallicity,       "f4", 1, 0);
    IO_REG(Metallicity,       "f4", 1, 4);

    /* Blackhole */
    IO_REG(BlackholeMass,          "f8", 1, 5);
    IO_REG(BlackholeAccretionRate, "f4", 1, 5);
    IO_REG(BlackholeProgenitors,   "i4", 1, 5);

    fof_register_io_blocks();
}

