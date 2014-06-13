#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <string.h>

#include "allvars.h"
#include "proto.h"
#include "bigfile/bigfile-mpi.h"

#include "petaio.h"

/************
 *
 * The IO api , intented to replace io.c and read_ic.c
 * currently we have a function to register the blocks and enumerate the blocks
 *
 */

struct IOTable IOTable = {0};

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

void register_io_blocks() {
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


#if 0

struct {
    int blocknr;
    int ptype;
    char dtype[8];
    void (*readout) (int i, void * value);
} IO_BLOCK_DESCR[1024];

void io_block_register(int blocknr, int ptype, char * dtype,
        readout_func readout,
        writein_func writein) {

}

static void read_ic_header(BigFile * bf);
static int64_t npartTotal[6];
static int64_t npartLocal[6];
static int ptypeOffsetLocal[6];
static int64_t partOffset[6];
void read_ic(char * fname) {
    walltime_measure("/Misc");
    BigFile bf = {0};
    BigBlock bb = {0};
    BigBlockPtr ptr = {0};
    if(!big_file_mpi_open(&bf, fname, MPI_COMM_WORLD)) {
        if(ThisTask == 0) {
            fprintf(stderr, "Failed to open IC from %s\n", fname);
        abort();
    }
    read_ic_header(&bf);
    int ptype;
    int blocknr;
    for(blocknr = 0; blocknr < IO_LASTENTRY; i ++) {
        char blockname[128];
        if(!block_present(blocknr)) continue;

        if(RestartFlag == 0 && blocknr == IO_U) continue;

        get_dataset_name(blocknr, blockname + 2);
        for(ptype = 0; ptype < 6; ptype ++) {
            if(npartTotal[ptype] == 0) continue;
            blockname[0] = ptype + '0';
            blockname[1] = '/'
            readblock(blocknr, ptype);

        }
    }
    big_file_mpi_close(&bf, MPI_COMM_WORLD);
}

static void read_ic_header(BigFile * bf) {
    BigBlock bh = {0};
    big_file_mpi_open_block(bf, &bh, "header", MPI_COMM_WORLD);
    int i;

    /* set all flags to 0; we do not use header otherwise in petaio */
    memset(&header, 0, sizeof(header));    

    big_block_get_attr(&bh, "NumPartTotal", npartTotal, "i8", 6);
    big_block_get_attr(&bh, "MassTable", All.MassTable, "f8", 6);
    big_block_get_attr(&bh, "Time", &All.TimeBegin, "f8", 1);
    big_block_mpi_close(&bh, MPI_COMM_WORLD);

    /* now initialialize all, this is donw on every body */
    set_global_time(All.TimeBegin);

    All.TotN_sph = npartTotal[0];
    All.TotN_bh = npartTotal[5];
    NumPart = 0;
    for(i = 0; i < 6; i ++) {
        All.TotNumPart += npartTotal[i];
        partOffset[i] = npartTotal[i] * ThisTask / NTask;
        npartLocal[i] = npartTotal[i] * (ThisTask + 1) / NTask - partOffset[i];
        NumPart += npartLocal[i];
    }
    ptypeOffsetLocal[0] = 0;
    for(i = 1; i< 6; i ++) {
        ptypeOffsetLocal[i] = ptypeOffsetLocal[i - 1] + npartLocal[i - 1];
    }

    All.MaxPart = (int) (All.PartAllocFactor * (All.TotNumPart / NTask));	/* sets the maximum number of particles that may */
    All.MaxPartSph = (int) (All.PartAllocFactor * (All.TotN_sph / NTask));	/* sets the maximum number of particles that may 
                                                                               reside on a processor */
    All.MaxPartBh = (int) (0.1 * All.PartAllocFactor * (All.TotN_sph / NTask));	/* sets the maximum number of particles that may 
                                                                                   reside on a processor */
    /* allocate memory */
    allocate_memory();

    /* setup the particle storage bits,
     * so that SPHP, P, BHP are valid */
    N_bh = 0;
    N_sph = 0;
    for(ptype = 0; ptype < 6; ptype ++) {
        int j;
        for(j = 0; j < npartLocal[ptype]; j ++) {
            P[i].Type = ptype;
            if(type == 0) {
                N_sph ++;
            } else
            if(ptype == 5) {
                P[i].PI = N_bh;
                N_bh ++;
            }
            i ++; 
        }
    }
}
#endif
