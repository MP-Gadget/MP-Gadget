#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <string.h>

#include "allvars.h"
#include "proto.h"
#include "petaio.h"
#include "bigfile/bigfile-mpi.h"

#include "fof.h"

/* from FoF*/
extern int Ngroups, TotNgroups;
extern struct group_properties *Group;

static int ThisColor;
static int ThisKey;
MPI_Comm GROUP;
static int GroupSize;
static int NumFiles;
static void saveblock(BigFile * bf, char * blockname, BigArray * array);
static void write_header(BigFile * bf);
static void build_buffer_particle(BigArray * array, IOTableEntry * ent);
static void build_buffer_fof(BigArray * array, IOTableEntry * ent);
static int64_t count_to_offset(int64_t countLocal);
static int64_t count_sum(int64_t countLocal);

static void fof_return_particles();
static void fof_distribute_particles();
void fof_save_particles(int num) {
    char fname[4096];
    sprintf(fname, "%s/PIG_%03d", All.OutputDir, num);
    /* Split the wolrd into writer groups */
    NumFiles = All.NumFilesWrittenInParallel;
    ThisColor = ThisTask * NumFiles / NTask;
    MPI_Comm_split(MPI_COMM_WORLD, ThisColor, 0, &GROUP);
    MPI_Comm_rank(GROUP, &ThisKey);
    MPI_Comm_size(GROUP, &GroupSize);

    fof_distribute_particles();

    BigFile bf = {0};
    if(0 != big_file_mpi_create(&bf, fname, MPI_COMM_WORLD)) {
        if(ThisTask == 0) {
            fprintf(stderr, "Failed to open IC from %s\n", fname);
        }
        abort();
    }

    write_header(&bf); 

    int i;
    for(i = 0; i < IOTable.used; i ++) {
        /* only process the particle blocks */
        char blockname[128];
        int ptype = IOTable.ent[i].ptype;
        BigArray array = {0};
        if(ptype < 6 && ptype >= 0) {
            sprintf(blockname, "%d/%s", ptype, IOTable.ent[i].name);
            build_buffer_particle(&array, &IOTable.ent[i]);
        } else 
        if(ptype == PTYPE_FOF_GROUP) {
            sprintf(blockname, "FOFGroups/%s", IOTable.ent[i].name);
            build_buffer_fof(&array, &IOTable.ent[i]);
        } else {
            abort();
        }
        saveblock(&bf, blockname, &array);
        free(array.data);
    }
    big_file_mpi_close(&bf, MPI_COMM_WORLD);

    fof_return_particles();
}

struct PartIndex {
    uint64_t origin;
    union {
        int64_t sortKey;
        int targetTask;
    };
};

static int fof_sorted_layout(int i) {
    return P[i].targettask;
}
static int fof_origin_layout(int i) {
    return P[i].origintask;
}
static int fof_cmp_sortkey(struct PartIndex * p1, struct PartIndex * p2) {
    return (p1->sortKey > p2->sortKey) - (p1->sortKey < p2->sortKey);
}
static int fof_cmp_origin(struct PartIndex * p1, struct PartIndex * p2) {
    return (p1->origin > p2->origin) - (p1->origin < p2->origin);
}
static void fof_distribute_particles() {
    int i;
    for(i = 0; i < NumPart; i ++) {
        P[i].origintask = ThisTask;
        P[i].targettask = P[i].ID % NTask; /* default target */
    }

    struct PartIndex * pi = mymalloc("PartIndex", sizeof(struct PartIndex) * NumPart);

    int64_t NpigLocal = 0;
    for(i = 0; i < NumPart; i ++) {
        int j = NpigLocal;
        if(P[i].GrNr < 0) continue;
        pi[j].origin = ThisTask * All.MaxPart + i;
        pi[j].sortKey = P[i].GrNr;
        NpigLocal ++;
    }
    /* sort pi to decide targetTask */
    parallel_sort(pi, NpigLocal, sizeof(struct PartIndex), fof_cmp_sortkey);

    int64_t Npig = count_sum(NpigLocal);
    int64_t offsetLocal = count_to_offset(NpigLocal);

    size_t chunksize = (Npig / NTask) + (Npig % NTask != 0);

    for(i = 0; i < NpigLocal; i ++) {
        ptrdiff_t offset = offsetLocal + i;
        pi[i].targetTask = DMIN(offset / chunksize, NTask - 1);
    }
    /* return pi to the original processors */
    parallel_sort(pi, NpigLocal, sizeof(struct PartIndex), fof_cmp_origin);
    for(i = 0; i < NpigLocal; i ++) {
        int index = pi[i].origin % All.MaxPart;
        P[index].targettask = pi[i].targetTask;
    }

    domain_exchange(fof_sorted_layout);

    myfree(pi);
}

static void fof_return_particles() {
    domain_exchange(fof_origin_layout);
}

static void build_buffer_particle(BigArray * array, IOTableEntry * ent) {
    size_t dims[2];
    ptrdiff_t strides[2];
    int elsize = dtype_itemsize(ent->dtype);

    int64_t npartLocal = 0;
    int i;

    for(i = 0; i < NumPart; i ++) {
        if(P[i].Type != ent->ptype) continue;
        if(P[i].GrNr < 0) continue;
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
        if(P[i].GrNr < 0) continue;
        ent->getter(i, p);
        p += strides[0];
    }
}
static void build_buffer_fof(BigArray * array, IOTableEntry * ent) {
    size_t dims[2];
    ptrdiff_t strides[2];
    int elsize = dtype_itemsize(ent->dtype);

    int64_t npartLocal = Ngroups;

    dims[0] = npartLocal;
    dims[1] = ent->items;
    strides[1] = elsize;
    strides[0] = elsize * ent->items;

    /* create the buffer */
    char * buffer = malloc(dims[0] * dims[1] * elsize);
    big_array_init(array, buffer, ent->dtype, 2, dims, strides);
    /* fill the buffer */
    char * p = buffer;
    int i;
    for(i = 0; i < Ngroups; i ++) {
        ent->getter(i, p);
        p += strides[0];
    }
}

static void saveblock(BigFile * bf, char * blockname, BigArray * array) {
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

static void write_header(BigFile * bf) {
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
        if(P[i].GrNr < 0) continue; /* skip those not in groups */
        npartLocal[P[i].Type] ++;
    }

    MPI_Allreduce(npartLocal, npartTotal, 6, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
    
    big_block_set_attr(&bh, "NumPartTotal", npartTotal, "u8", 6);
    big_block_set_attr(&bh, "NumFOFGroupsTotal", &TotNgroups, "u8", 1);
    big_block_set_attr(&bh, "MassTable", All.MassTable, "f8", 6);
    big_block_set_attr(&bh, "Time", &All.Time, "f8", 1);
    big_block_set_attr(&bh, "BoxSize", &All.BoxSize, "f8", 1);
    big_block_set_attr(&bh, "OmegaLambda", &All.OmegaLambda, "f8", 1);
    big_block_set_attr(&bh, "Omega0", &All.Omega0, "f8", 1);
    big_block_set_attr(&bh, "HubbleParam", &All.HubbleParam, "f8", 1);
    big_block_mpi_close(&bh, MPI_COMM_WORLD);
}

SIMPLE_GETTER(GTGroupID, Group[i].GrNr, uint32_t, 1)
SIMPLE_GETTER(GTMassCenterPosition, Group[i].CM[0], double, 3)
SIMPLE_GETTER(GTMassCenterVelocity, Group[i].Vel[0], float, 3)
SIMPLE_GETTER(GTMass, Group[i].Mass, float, 1)
SIMPLE_GETTER(GTMassByType, Group[i].MassType[0], float, 6)
SIMPLE_GETTER(GTLengthByType, Group[i].LenType[0], uint32_t , 6)
SIMPLE_GETTER(GTStarFormationRate, Group[i].Sfr, float, 1)
SIMPLE_GETTER(GTBlackholeMass, Group[i].BH_Mass, float, 1)
SIMPLE_GETTER(GTBlackholeAccretionRate, Group[i].BH_Mdot, float, 1)

void fof_register_io_blocks() {
    IO_REG(GroupID, "u4", 1, PTYPE_FOF_GROUP);
    IO_REG(Mass, "f4", 1, PTYPE_FOF_GROUP);
    IO_REG(MassCenterPosition, "f8", 3, PTYPE_FOF_GROUP);
    IO_REG(MassCenterVelocity, "f4", 3, PTYPE_FOF_GROUP);
    IO_REG(LengthByType, "u4", 6, PTYPE_FOF_GROUP);
    IO_REG(MassByType, "f4", 6, PTYPE_FOF_GROUP);
    IO_REG(StarFormationRate, "f4", 1, PTYPE_FOF_GROUP);
    IO_REG(BlackholeMass, "f4", 1, PTYPE_FOF_GROUP);
    IO_REG(BlackholeAccretionRate, "f4", 1, PTYPE_FOF_GROUP);
}

static int64_t count_sum(int64_t countLocal) {
    int64_t sum = 0;
    MPI_Allreduce(&countLocal, &sum, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
    return sum;
}
static int64_t count_to_offset(int64_t countLocal) {
    int64_t offsetLocal;
    int64_t count[NTask];
    int64_t offset[NTask];
    MPI_Gather(&countLocal, 1, MPI_LONG, &count[0], 1, MPI_LONG, 0, MPI_COMM_WORLD);
    if(ThisTask == 0) {
        offset[0] = 0;
        int i;
        for(i = 1; i < NTask; i ++) {
            offset[i] = offset[i-1] + count[i-1];
        }
    }
    MPI_Scatter(&offset[0], 1, MPI_LONG, &offsetLocal, 1, MPI_LONG, 0, MPI_COMM_WORLD);
    return offsetLocal;
}
