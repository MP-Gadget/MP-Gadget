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

#include "fof.h"

/* from FoF*/
extern int Ngroups, TotNgroups;
extern struct group_properties *Group;

static void fof_write_header(BigFile * bf);
static void build_buffer_fof(BigArray * array, IOTableEntry * ent);

static void fof_return_particles();
static void fof_distribute_particles();

static int fof_select_particle(int i) {
    return P[i].GrNr > 0;
}

void fof_save_particles(int num) {
    char fname[4096];
    sprintf(fname, "%s/PIG_%03d", All.OutputDir, num);
    if(ThisTask == 0) {
        printf("saving particle in group into %s\n", fname);
        fflush(stdout);
    }
    fof_distribute_particles();
    
    BigFile bf = {0};
    if(0 != big_file_mpi_create(&bf, fname, MPI_COMM_WORLD)) {
        if(ThisTask == 0) {
            fprintf(stderr, "Failed to open IC from %s\n", fname);
        }
        abort();
    }

    fof_write_header(&bf); 

    int i;
    for(i = 0; i < IOTable.used; i ++) {
        /* only process the particle blocks */
        char blockname[128];
        int ptype = IOTable.ent[i].ptype;
        BigArray array = {0};
        if(ptype < 6 && ptype >= 0) {
            sprintf(blockname, "%d/%s", ptype, IOTable.ent[i].name);
            petaio_build_buffer(&array, &IOTable.ent[i], fof_select_particle);
        } else 
        if(ptype == PTYPE_FOF_GROUP) {
            sprintf(blockname, "FOFGroups/%s", IOTable.ent[i].name);
            build_buffer_fof(&array, &IOTable.ent[i]);
        } else {
            abort();
        }
        petaio_save_block(&bf, blockname, &array);
        petaio_destroy_buffer(&array);
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
static int fof_cmp_sortkey(const void * c1, const void * c2) {
    const struct PartIndex * p1 = c1;
    const struct PartIndex * p2 = c2;
    return (p1->sortKey > p2->sortKey) - (p1->sortKey < p2->sortKey);
}
static int fof_cmp_origin(const void * c1, const void * c2) {
    const struct PartIndex * p1 = c1;
    const struct PartIndex * p2 = c2;
    return (p1->origin > p2->origin) - (p1->origin < p2->origin);
}
static int p_cmp_GrNr(const void * c1, const void * c2) {
    const struct particle_data * p1 = c1;
    const struct particle_data * p2 = c2;
    return (p1->GrNr > p2->GrNr) - (p1->GrNr < p2->GrNr);
}
static void fof_distribute_particles() {
    int i;
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

#pragma omp parallel for
    for(i = 0; i < NumPart; i ++) {
        P[i].origintask = ThisTask;
        P[i].targettask = ThisTask; //P[i].ID % NTask; /* default target */
    }

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
    /* sort SPH and Others independently */

#pragma omp parallel for
    for(i = 0; i < N_sph; i ++) {
        P[i].PI = i;
    }

    qsort(P, N_sph, sizeof(P[0]), p_cmp_GrNr);

    /* permute the SphP struct to follow P */
    for(i = 0; i < N_sph; i ++) {
        if(P[i].PI == i) continue;
        int j;
        struct sph_particle_data s;
        /* save the first one */
        s = SphP[i];
        j = i;
        while(P[j].PI != i) {
            int freeslot = P[j].PI;
            /* move the freeslot to the right position */
            SphP[j] = SphP[freeslot];
            P[j].PI = j;
            /* now fix P at freeslot */
            j = freeslot;
        }
        /* this guy uses the very first one*/
        SphP[j] = s;
        P[j].PI = j;
    }

    /* sort rest */
    qsort(P + N_sph, NumPart - N_sph, sizeof(P[0]), p_cmp_GrNr);

}

static void fof_return_particles() {
    domain_exchange(fof_origin_layout);
}

static void build_buffer_fof(BigArray * array, IOTableEntry * ent) {

    int64_t npartLocal = Ngroups;

    petaio_alloc_buffer(array, ent, npartLocal);
    /* fill the buffer */
    char * p = array->data;
    int i;
    for(i = 0; i < Ngroups; i ++) {
        ent->getter(i, p);
        p += array->strides[0];
    }
}

static void fof_write_header(BigFile * bf) {
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
    
    big_block_set_attr(&bh, "NumPartInGroupTotal", npartTotal, "u8", 6);
    big_block_set_attr(&bh, "NumFOFGroupsTotal", &TotNgroups, "u8", 1);
    big_block_set_attr(&bh, "MassTable", All.MassTable, "f8", 6);
    big_block_set_attr(&bh, "Time", &All.Time, "f8", 1);
    big_block_set_attr(&bh, "BoxSize", &All.BoxSize, "f8", 1);
    big_block_set_attr(&bh, "OmegaLambda", &All.OmegaLambda, "f8", 1);
    big_block_set_attr(&bh, "Omega0", &All.Omega0, "f8", 1);
    big_block_set_attr(&bh, "HubbleParam", &All.HubbleParam, "f8", 1);
    big_block_mpi_close(&bh, MPI_COMM_WORLD);
}

SIMPLE_PROPERTY(GroupID, Group[i].GrNr, uint32_t, 1)
SIMPLE_PROPERTY(MassCenterPosition, Group[i].CM[0], double, 3)
SIMPLE_PROPERTY(MassCenterVelocity, Group[i].Vel[0], float, 3)
SIMPLE_PROPERTY(Mass, Group[i].Mass, float, 1)
SIMPLE_PROPERTY(MassByType, Group[i].MassType[0], float, 6)
SIMPLE_PROPERTY(LengthByType, Group[i].LenType[0], uint32_t , 6)
SIMPLE_PROPERTY(StarFormationRate, Group[i].Sfr, float, 1)
SIMPLE_PROPERTY(BlackholeMass, Group[i].BH_Mass, float, 1)
SIMPLE_PROPERTY(BlackholeAccretionRate, Group[i].BH_Mdot, float, 1)

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

