#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <string.h>

#include <bigfile-mpi.h>
#include <mpsort.h>

#include "utils.h"

#include "allvars.h"
#include "partmanager.h"
#include "petaio.h"
#include "exchange.h"
#include "fof.h"

static void fof_register_io_blocks(struct IOTable * IOTable);
static void fof_write_header(BigFile * bf, int64_t TotNgroups, MPI_Comm Comm);
static void build_buffer_fof(int Ngroups, BigArray * array, IOTableEntry * ent);

static void fof_return_particles(MPI_Comm Comm);
static void fof_distribute_particles(MPI_Comm Comm);

static int fof_cmp_selection_by_grnr(const void *p1, const void * p2) {
    const int * i1 = p1;
    const int * i2 = p2;
    return (P[*i1].GrNr > P[*i2].GrNr) - (P[*i1].GrNr < P[*i2].GrNr);
}

static void fof_radix_Group_GrNr(const void * a, void * radix, void * arg);
static void fof_radix_Group_GrNr(const void * a, void * radix, void * arg) {
    uint64_t * u = (uint64_t *) radix;
    struct BaseGroup * f = (struct BaseGroup*) a;
    u[0] = f->GrNr;
}

static int
fof_petaio_select_func(int i)
{
    if(P[i].GrNr < 0) return 0;
    return 1;
}

/* Global for the particle saving routines*/
struct Group * Group;

void fof_save_particles(FOFGroups * fof, int num, int SaveParticles, MPI_Comm Comm)
{
    int i;
    struct IOTable FOFIOTable = {0};
    char * fname = fastpm_strdup_printf("%s/%s_%03d", All.OutputDir, All.FOFFileBase, num);
    message(0, "saving particle in group into %s\n", fname);

    fof_register_io_blocks(&FOFIOTable);
    /* sort the groups according to group-number */
    mpsort_mpi(fof->Group, fof->Ngroups, sizeof(struct Group),
            fof_radix_Group_GrNr, 8, NULL, Comm);

    Group = fof->Group;

    BigFile bf = {0};
    if(0 != big_file_mpi_create(&bf, fname, Comm)) {
        endrun(0, "Failed to open file at %s\n", fname);
    }
    myfree(fname);

    MPIU_Barrier(Comm);
    fof_write_header(&bf, fof->TotNgroups, Comm);

    for(i = 0; i < FOFIOTable.used; i ++) {
        /* only process the particle blocks */
        char blockname[128];
        int ptype = FOFIOTable.ent[i].ptype;
        BigArray array = {0};
        if(ptype == PTYPE_FOF_GROUP) {
            sprintf(blockname, "FOFGroups/%s", FOFIOTable.ent[i].name);
            build_buffer_fof(fof->Ngroups, &array, &FOFIOTable.ent[i]);
            message(0, "Writing Block %s\n", blockname);

            petaio_save_block(&bf, blockname, &array, 1);
            petaio_destroy_buffer(&array);
        }
    }
    destroy_io_blocks(&FOFIOTable);
    walltime_measure("/FOF/IO/WriteFOF");

    if(SaveParticles) {
        struct IOTable IOTable = {0};
        register_io_blocks(&IOTable);
        fof_distribute_particles(Comm);
        walltime_measure("/FOF/IO/Distribute");

        int * selection = mymalloc("Selection", sizeof(int) * PartManager->NumPart);

        int ptype_offset[6]={0};
        int ptype_count[6]={0};
        petaio_build_selection(selection, ptype_offset, ptype_count, PartManager->NumPart, fof_petaio_select_func);

        /*Sort each type individually*/
        for(i = 0; i < 6; i++)
            qsort_openmp(selection + ptype_offset[i], ptype_count[i], sizeof(int), fof_cmp_selection_by_grnr);

        walltime_measure("/FOF/IO/argind");

        for(i = 0; i < IOTable.used; i ++) {
            /* only process the particle blocks */
            char blockname[128];
            int ptype = IOTable.ent[i].ptype;
            BigArray array = {0};
            sprintf(blockname, "%d/%s", ptype, IOTable.ent[i].name);
            petaio_build_buffer(&array, &IOTable.ent[i], selection + ptype_offset[ptype], ptype_count[ptype]);
            message(0, "Writing Block %s\n", blockname);
            petaio_save_block(&bf, blockname, &array, 1);
            petaio_destroy_buffer(&array);
        }
        myfree(selection);
        walltime_measure("/FOF/IO/WriteParticles");

        fof_return_particles(Comm);
        walltime_measure("/FOF/IO/Return");
        destroy_io_blocks(&IOTable);
    }

    big_file_mpi_close(&bf, Comm);
}

struct PartIndex {
    uint64_t origin;
    union {
        int64_t sortKey;
        int targetTask;
    };
};

static int fof_sorted_layout(int i, const void * userdata) {
    int * targetTask = (int *) userdata;
    return targetTask[i];
}

static int fof_origin_layout(int i, const void * userdata) {
    return P[i].origintask;
}
static void fof_radix_sortkey(const void * c1, void * out, void * arg) {
    uint64_t * u = out;
    const struct PartIndex * pi = c1;
    *u = pi->sortKey;
}
static void fof_radix_origin(const void * c1, void * out, void * arg) {
    uint64_t * u = out;
    const struct PartIndex * pi = c1;
    *u = pi->origin;
}
#if 0
/*Unused functions*/
static int fof_select_particle(int i) {
    return P[i].GrNr > 0;
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
#endif

static void fof_distribute_particles(MPI_Comm Comm) {
    int i, ThisTask;
    MPI_Comm_rank(Comm, &ThisTask);
    struct PartIndex * pi = mymalloc("PartIndex", sizeof(struct PartIndex) * PartManager->NumPart);

    int64_t NpigLocal = 0;
    int GrNrMax = -1;	/* will mark particles that are not in any group */
    int GrNrMaxGlobal = 0;
    for(i = 0; i < PartManager->NumPart; i ++) {
        int j = NpigLocal;
        if(P[i].GrNr < 0) continue;
        if(P[i].GrNr > GrNrMax) GrNrMax = P[i].GrNr;
        /* Yu: found it! this shall be int64 */
        // pi[j].origin =  ThisTask * PartManager->MaxPart + i;
        pi[j].origin = ((uint64_t) ThisTask) * PartManager->MaxPart + i;
        pi[j].sortKey = P[i].GrNr;
        NpigLocal ++;
    }
    MPI_Allreduce(&GrNrMax, &GrNrMaxGlobal, 1, MPI_INT, MPI_MAX, Comm);
    message(0, "GrNrMax before exchange is %d\n", GrNrMaxGlobal);
    /* sort pi to decide targetTask */
    mpsort_mpi(pi, NpigLocal, sizeof(struct PartIndex),
            fof_radix_sortkey, 8, NULL, Comm);

#pragma omp parallel for
    for(i = 0; i < PartManager->NumPart; i ++) {
        P[i].origintask = ThisTask;
    }

    //int64_t Npig = count_sum(NpigLocal);
    //int64_t offsetLocal = MPIU_cumsum(NpigLocal, Comm);

    //size_t chunksize = (Npig / NTask) + (Npig % NTask != 0);

    for(i = 0; i < NpigLocal; i ++) {
/* YU: A typo error here, should be IMIN, DMIN is for double but this should have tainted TargetTask,
   offset and chunksize are int  */
        //ptrdiff_t offset = offsetLocal + i;
        //pi[i].targetTask = IMIN(offset / chunksize, NTask - 1);
    /* YU: let's see if we keep the FOF particle load on the processes, IO would be faster
           (as at high z many ranks has no FOF), communication becomes sparse. */
        pi[i].targetTask = ThisTask;
    }
    /* return pi to the original processors */
    mpsort_mpi(pi, NpigLocal, sizeof(struct PartIndex), fof_radix_origin, 8, NULL, Comm);
    /* Copy the target task into a temporary array for all particles*/
    int * targettask = mymalloc2("targettask", sizeof(int) * PartManager->NumPart);

    /*By default particles stay where they are*/
    #pragma omp parallel for
    for(i = 0; i < PartManager->NumPart; i ++) {
        targettask[i] = ThisTask;
    }
    for(i = 0; i < NpigLocal; i ++) {
        int index = pi[i].origin % PartManager->MaxPart;
        targettask[index] = pi[i].targetTask;
    }
    myfree(pi);

    if(domain_exchange(fof_sorted_layout, targettask, 1, Comm))
        endrun(1930,"Could not exchange particles\n");
    /* sort SPH and Others independently */

    myfree(targettask);
    GrNrMax = -1;
    for(i = 0; i < PartManager->NumPart; i ++) {
        if(P[i].GrNr < 0) continue;
        if(P[i].GrNr > GrNrMax) GrNrMax = P[i].GrNr;
    }
    MPI_Allreduce(&GrNrMax, &GrNrMaxGlobal, 1, MPI_INT, MPI_MAX, Comm);
    message(0, "GrNrMax after exchange is %d\n", GrNrMaxGlobal);

}
static void fof_return_particles(MPI_Comm Comm) {
    if(domain_exchange(fof_origin_layout, NULL, 1, Comm))
        endrun(1931,"Could not exchange particles\n");
}

static void build_buffer_fof(int Ngroups, BigArray * array, IOTableEntry * ent) {

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

static void fof_write_header(BigFile * bf, int64_t TotNgroups, MPI_Comm Comm) {
    BigBlock bh;
    if(0 != big_file_mpi_create_block(bf, &bh, "Header", NULL, 0, 0, 0, Comm)) {
        endrun(0, "Failed to create header\n");
    }
    int i;
    int k;
    int64_t npartLocal[6];
    int64_t npartTotal[6];

    for (k = 0; k < 6; k ++) {
        npartLocal[k] = 0;
    }
    for (i = 0; i < PartManager->NumPart; i ++) {
        if(P[i].GrNr < 0) continue; /* skip those not in groups */
        npartLocal[P[i].Type] ++;
    }

    MPI_Allreduce(npartLocal, npartTotal, 6, MPI_INT64, MPI_SUM, Comm);

    /* conversion from peculiar velocity to RSD */
    double RSD = 1.0 / (All.cf.a * All.cf.hubble);

    if(!All.IO.UsePeculiarVelocity) {
        RSD /= All.cf.a; /* Conversion from internal velocity to RSD */
    }
    big_block_set_attr(&bh, "NumPartInGroupTotal", npartTotal, "u8", 6);
    big_block_set_attr(&bh, "NumFOFGroupsTotal", &TotNgroups, "u8", 1);
    big_block_set_attr(&bh, "RSDFactor", &RSD, "f8", 1);
    big_block_set_attr(&bh, "MassTable", All.MassTable, "f8", 6);
    big_block_set_attr(&bh, "Time", &All.Time, "f8", 1);
    big_block_set_attr(&bh, "BoxSize", &All.BoxSize, "f8", 1);
    big_block_set_attr(&bh, "OmegaLambda", &All.CP.OmegaLambda, "f8", 1);
    big_block_set_attr(&bh, "Omega0", &All.CP.Omega0, "f8", 1);
    big_block_set_attr(&bh, "HubbleParam", &All.CP.HubbleParam, "f8", 1);
    big_block_set_attr(&bh, "CMBTemperature", &All.CP.CMBTemperature, "f8", 1);
    big_block_set_attr(&bh, "OmegaBaryon", &All.CP.OmegaBaryon, "f8", 1);
    big_block_mpi_close(&bh, Comm);
}

SIMPLE_PROPERTY(GroupID, Group[i].base.GrNr, uint32_t, 1)
SIMPLE_PROPERTY(MinID, Group[i].base.MinID, uint64_t, 1)
SIMPLE_PROPERTY(FirstPos, Group[i].base.FirstPos[0], float, 3)
SIMPLE_PROPERTY(MassCenterPosition, Group[i].CM[0], double, 3)
SIMPLE_PROPERTY(Imom, Group[i].Imom[0][0], float, 9)
/* FIXME: set Jmom to use peculiar velocity */
SIMPLE_PROPERTY(Jmom, Group[i].Jmom[0], float, 3)
static void GTMassCenterVelocity(int i, float * out) {
    double fac;
    if (All.IO.UsePeculiarVelocity) {
        fac = 1.0 / All.cf.a;
    } else {
        fac = 1.0;
    }

    int d;
    for(d = 0; d < 3; d ++) {
        out[d] = fac * Group[i].Vel[d];
    }
}
SIMPLE_PROPERTY(Mass, Group[i].Mass, float, 1)
SIMPLE_PROPERTY(MassByType, Group[i].MassType[0], float, 6)
SIMPLE_PROPERTY(LengthByType, Group[i].LenType[0], uint32_t , 6)
SIMPLE_PROPERTY(StarFormationRate, Group[i].Sfr, float, 1)
SIMPLE_PROPERTY(BlackholeMass, Group[i].BH_Mass, float, 1)
SIMPLE_PROPERTY(BlackholeAccretionRate, Group[i].BH_Mdot, float, 1)

static void fof_register_io_blocks(struct IOTable * IOTable) {
    IOTable->used = 0;
    IOTable->allocated = 100;
    /* Allocate high so we can do a domain exchange,
     * potentially increasing the slots, around this*/
    IOTable->ent = mymalloc2("IOTable", IOTable->allocated* sizeof(IOTableEntry));

    IO_REG(GroupID, "u4", 1, PTYPE_FOF_GROUP, IOTable);
    IO_REG(Mass, "f4", 1, PTYPE_FOF_GROUP, IOTable);
    IO_REG(MassCenterPosition, "f8", 3, PTYPE_FOF_GROUP, IOTable);
    IO_REG(FirstPos, "f4", 3, PTYPE_FOF_GROUP, IOTable);
    IO_REG(MinID, "u8", 1, PTYPE_FOF_GROUP, IOTable);
    IO_REG(Imom, "f4", 9, PTYPE_FOF_GROUP, IOTable);
    IO_REG(Jmom, "f4", 3, PTYPE_FOF_GROUP, IOTable);
    IO_REG_WRONLY(MassCenterVelocity, "f4", 3, PTYPE_FOF_GROUP, IOTable);
    IO_REG(LengthByType, "u4", 6, PTYPE_FOF_GROUP, IOTable);
    IO_REG(MassByType, "f4", 6, PTYPE_FOF_GROUP, IOTable);
    if(All.StarformationOn)
        IO_REG(StarFormationRate, "f4", 1, PTYPE_FOF_GROUP, IOTable);
    if(All.BlackHoleOn) {
        IO_REG(BlackholeMass, "f4", 1, PTYPE_FOF_GROUP, IOTable);
        IO_REG(BlackholeAccretionRate, "f4", 1, PTYPE_FOF_GROUP, IOTable);
    }
}
