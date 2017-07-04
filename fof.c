#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <math.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <gsl/gsl_math.h>
#include <inttypes.h>
#include "allvars.h"
#include "sfr_eff.h"
#include "blackhole.h"
#include "drift.h"
#include "domain.h"
#include "mpsort.h"
#include "mymalloc.h"
#include "endrun.h"
#include "treewalk.h"
#include "system.h"
#include "densitykernel.h"

/*! \file fof.c
 *  \brief parallel FoF group finder
 */

#include "fof.h"

/* Never change the primary link it is always DM. */
#define FOF_PRIMARY_LINK_TYPES 2

/* FIXME: convert this to a parameter */
#define FOF_SECONDARY_LINK_TYPES (1+16+32)    // 2^type for the types linked to nearest primaries
#define LARGE 1e29
void fof_init()
{
    All.FOFHaloComovingLinkingLength = All.FOFHaloLinkingLength * All.MeanSeparation[1];
    All.TimeNextSeedingCheck = All.Time;
}

static double fof_periodic(double x)
{
    if(x >= 0.5 * All.BoxSize)
        x -= All.BoxSize;
    if(x < -0.5 * All.BoxSize)
        x += All.BoxSize;
    return x;
}


static double fof_periodic_wrap(double x)
{
    while(x >= All.BoxSize)
        x -= All.BoxSize;
    while(x < 0)
        x += All.BoxSize;
    return x;
}

static void fof_label_secondary(void);
static int fof_compare_HaloLabel_MinID(const void *a, const void *b);
static int fof_compare_Group_MinIDTask(const void *a, const void *b);
static int fof_compare_Group_OriginalIndex(const void *a, const void *b);
static int fof_compare_Group_MinID(const void *a, const void *b);
static void fof_reduce_groups(
    void * groups, 
    size_t nmemb, 
    size_t elsize, 
    void (*reduce_group)(void * gdst, void * gsrc));

static void fof_finish_group_properties(void);
static void fof_compile_base(void);
static void fof_compile_catalogue(void);
static void fof_assign_grnr();

void fof_label_primary(void);
extern void fof_save_particles(int num);
extern void fof_save_groups(int num);

static void fof_seed(void);

uint64_t Ngroups, TotNgroups, NgroupsExt;
int64_t TotNids;

struct Group *Group;
struct BaseGroup *BaseGroup;

typedef struct {
    TreeWalkQueryBase base;
    MyFloat Hsml;
    MyIDType MinID;
    MyIDType MinIDTask;
} TreeWalkQueryFOF;

typedef struct {
    TreeWalkResultBase base;
    MyFloat Distance;
    MyIDType MinID;
    MyIDType MinIDTask;
} TreeWalkResultFOF;

typedef struct {
    TreeWalkNgbIterBase base;
} TreeWalkNgbIterFOF;

static struct fof_particle_list
{
    MyIDType MinID;
    MyIDType MinIDTask;
    int Pindex;
}
*HaloLabel;

static float *fof_secondary_distance;
static float *fof_secondary_hsml;

static MPI_Datatype MPI_TYPE_GROUP;

void fof_fof(int num)
{
    int i;
    double t0, t1;

    MPI_Type_contiguous(sizeof(Group[0]), MPI_BYTE, &MPI_TYPE_GROUP);
    MPI_Type_commit(&MPI_TYPE_GROUP);

    message(0, "Begin to compute FoF group catalogues...  (presently allocated=%g MB)\n",
            AllocatedBytes / (1024.0 * 1024.0));

    walltime_measure("/Misc");

    message(0, "Comoving linking length: %g    ", All.FOFHaloComovingLinkingLength);
    message(0, "(presently allocated=%g MB)\n", AllocatedBytes / (1024.0 * 1024.0));

    HaloLabel = (struct fof_particle_list *) mymalloc("HaloLabel", NumPart * sizeof(struct fof_particle_list));

    /* HaloLabel stores the MinID and MinIDTask of particles, this pair serves as a halo label. */
    #pragma omp parallel for
    for(i = 0; i < NumPart; i++) {
        HaloLabel[i].Pindex = i;
    }
    /* Fill FOFP_List of primary */
    t0 = second();
    fof_label_primary();
    t1 = second();

    message(0, "group finding took = %g sec\n", timediff(t0, t1));
    walltime_measure("/FOF/Primary");

    /* Fill FOFP_List of secondary */
    t0 = second();
    fof_label_secondary();
    walltime_measure("/FOF/Secondary");
    t1 = second();

    message(0, "attaching gas and star particles to nearest dm particles took = %g sec\n", timediff(t0, t1));

    /* sort HaloLabel according to MinID, because we need that for compiling catalogues */
    qsort(HaloLabel, NumPart, sizeof(struct fof_particle_list), fof_compare_HaloLabel_MinID);

    t0 = second();

    fof_compile_base();

    fof_assign_grnr();

    fof_compile_catalogue();

    t1 = second();

    message(0, "compiling local group data and catalogue took = %g sec\n", timediff(t0, t1));

    walltime_measure("/FOF/Compile");

    t0 = second();

    message(0, "group properties are now allocated.. (presently allocated=%g MB)\n",
            AllocatedBytes / (1024.0 * 1024.0));

    walltime_measure("/FOF/Prop");
    t1 = second();

    message(0, "computation of group properties took = %g sec\n", timediff(t0, t1));

    if(num < 0)
        fof_seed();

    walltime_measure("/FOF/Misc");

    if(num >= 0)
    {
        fof_save_groups(num);
    }

    myfree(Group);
    myfree(BaseGroup);
    myfree(HaloLabel);

    if(num >= 0)
    {
        /* I am not sure why we need a domain decomposition here.
         * But simple peano reorder will produce
         * a tree that misses a few particles and crash PM. */
        domain_Decomposition_short();
    }

    message(0, "Finished computing FoF groups.  (presently allocated=%g MB)\n",
            AllocatedBytes / (1024.0 * 1024.0));

    walltime_measure("/FOF/MISC");

    MPI_Type_free(&MPI_TYPE_GROUP);
}

static MyIDType * FOFOldMinID;
static char * FOFPrimaryActive;
static int * FOFHead;

static int HEADl(int stop, int i) {
    int r;
    if (i == stop) {
        return -1;
    }
//    printf("locking %d by %d in HEADl stop = %d\n", i, omp_get_thread_num(), stop);
    lock_particle(i);
//    printf("locked %d by %d in HEADl, next = %d\n", i, omp_get_thread_num(), FOFHead[i]);

    if(FOFHead[i] == i) {
        /* return locked */
        return i;
    }
    /* this is not the root, keep going, but unlock first, since even if the root is modified by
     * another thread, what we get here is on the path, */
    int next = FOFHead[i];
    unlock_particle(i);
//    printf("unlocking %d by %d in HEADl\n", i, omp_get_thread_num());
    r = HEADl(stop, next);
    return r;
}
static void update_root(int i, int r)
{
    while(FOFHead[i] != i) {
        int t = FOFHead[i];
        FOFHead[i]= r;
        i = t;
    }
}

static int HEAD(int i) {
    /* accelerate with a splay: see https://arxiv.org/abs/1607.03224 */
    int r;
    r = i;
    while(FOFHead[r] != r) {
        r = FOFHead[r];
    }
    while(FOFHead[i] != i) {
        int t = FOFHead[i];
        FOFHead[i]= r;
        i = t;
    }
    return r;
}

static void fof_primary_copy(int place, TreeWalkQueryFOF * I, TreeWalk * tw) {
    int head = HEAD(place);
    I->MinID = HaloLabel[head].MinID;
    I->MinIDTask = HaloLabel[head].MinIDTask;
}

static int fof_primary_isactive(int n, TreeWalk * tw) {
    return (((1 << P[n].Type) & (FOF_PRIMARY_LINK_TYPES))) && FOFPrimaryActive[n];
}

static void
fof_primary_ngbiter(TreeWalkQueryFOF * I,
        TreeWalkResultFOF * O,
        TreeWalkNgbIterFOF * iter,
        LocalTreeWalk * lv);

void fof_label_primary(void)
{
    int i;
    int64_t link_across;
    int64_t link_across_tot;
    double t0, t1;

    message(0, "Start linking particles (presently allocated=%g MB)\n", AllocatedBytes / (1024.0 * 1024.0));

    TreeWalk tw[1] = {0};
    tw->ev_label = "FOF_FIND_GROUPS";
    tw->visit = (TreeWalkVisitFunction) treewalk_visit_ngbiter;
    tw->ngbiter = (TreeWalkNgbIterFunction) fof_primary_ngbiter;
    tw->ngbiter_type_elsize = sizeof(TreeWalkNgbIterFOF);

    tw->isactive = fof_primary_isactive;
    tw->fill = (TreeWalkFillQueryFunction) fof_primary_copy;
    tw->reduce = NULL;
    tw->UseNodeList = 1;
    tw->UseAllParticles = 1;
    tw->query_type_elsize = sizeof(TreeWalkQueryFOF);
    tw->result_type_elsize = sizeof(TreeWalkResultFOF);

    FOFHead = (int*) mymalloc("FOF_Links", NumPart * sizeof(int));
    FOFPrimaryActive = (char*) mymalloc("FOFActive", NumPart * sizeof(char));
    FOFOldMinID = (MyIDType *) mymalloc("FOFActive", NumPart * sizeof(MyIDType));

    /* allocate buffers to arrange communication */

    t0 = second();

    for(i = 0; i < NumPart; i++)
    {
        FOFHead[i] = i;
        FOFOldMinID[i]= P[i].ID;
        FOFPrimaryActive[i] = 1;

        HaloLabel[i].MinID = P[i].ID;
        HaloLabel[i].MinIDTask = ThisTask;
    }

    do
    {
        t0 = second();

        treewalk_run(tw);

        t1 = second();

        /* let's check out which particles have changed their MinID,
         * mark them for next round. */
        link_across = 0;
#pragma omp parallel for
        for(i = 0; i < NumPart; i++) {
            MyIDType newMinID = HaloLabel[HEAD(i)].MinID;
            if(newMinID != FOFOldMinID[i]) {
                FOFPrimaryActive[i] = 1;
#pragma omp atomic
                link_across += 1;
            } else {
                FOFPrimaryActive[i] = 0;
            }
            FOFOldMinID[i] = newMinID;
        }
        MPI_Allreduce(&link_across, &link_across_tot, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
        message(0, "Linked %ld particles %g seconds\n", link_across_tot, t1 - t0);
    }
    while(link_across_tot > 0);

    /* Update MinID of all linked (primary-linked) particles */
    for(i = 0; i < NumPart; i++)
    {
        HaloLabel[i].MinID = HaloLabel[HEAD(i)].MinID;
        HaloLabel[i].MinIDTask = HaloLabel[HEAD(i)].MinIDTask;
    }

    message(0, "Local groups found.\n");

    myfree(FOFOldMinID);
    myfree(FOFPrimaryActive);
    myfree(FOFHead);
}

static void fofp_merge(int target, int other)
{
    /* this will lock h1 */
    int h1 = HEADl(-1, target);
    /* stop looking if we find h1 along the path (because it is already owned by us) */
    int h2 = HEADl(h1, other);

    if(h2 == -1) {
        /* h1 is along the path of h2, already merged.  **/
        /* h1 must be the root of other and target both */
        //printf("unlocking %d by %d in merge\n", h1, omp_get_thread_num());
        update_root(target, h1);
        update_root(other, h1);
        unlock_particle(h1);
        return;
    }

    /* h2 as a sub-tree of h1 */
    FOFHead[h2] = h1;

    /* update MinID of h1 */
    if(HaloLabel[h1].MinID > HaloLabel[h2].MinID)
    {
        HaloLabel[h1].MinID = HaloLabel[h2].MinID;
        HaloLabel[h1].MinIDTask = HaloLabel[h2].MinIDTask;
    }
    //printf("unlocking %d by %d in merge\n", h2, omp_get_thread_num());
    unlock_particle(h2);

    update_root(target, h1);
    update_root(other, h1);

    //printf("unlocking %d by %d in merge\n", h1, omp_get_thread_num());
    unlock_particle(h1);
}

static void
fof_primary_ngbiter(TreeWalkQueryFOF * I,
        TreeWalkResultFOF * O,
        TreeWalkNgbIterFOF * iter,
        LocalTreeWalk * lv)
{
    if(iter->base.other == -1) {
        iter->base.Hsml = All.FOFHaloComovingLinkingLength;
        iter->base.symmetric = NGB_TREEFIND_ASYMMETRIC;
        iter->base.mask = FOF_PRIMARY_LINK_TYPES;
        return;
    }
    int other = iter->base.other;

#pragma omp critical (_fofp_merge_)
    {
        if(lv->mode == 0) {
            /* Local FOF */
            if(lv->target <= other) {
                // printf("locked merge %d %d by %d\n", lv->target, other, omp_get_thread_num());
                fofp_merge(lv->target, other);
            }
        } else /* mode is 1, target is a ghost */
        {
//            printf("locking %d by %d in ngbiter\n", other, omp_get_thread_num());
            lock_particle(other);
            if(HaloLabel[HEAD(other)].MinID > I->MinID)
            {
                HaloLabel[HEAD(other)].MinID = I->MinID;
                HaloLabel[HEAD(other)].MinIDTask = I->MinIDTask;
            }
//            printf("unlocking %d by %d in ngbiter\n", other, omp_get_thread_num());
            unlock_particle(other);
        }
    }
}

static void fof_reduce_base_group(void * pdst, void * psrc) {
    struct BaseGroup * gdst = pdst;
    struct BaseGroup * gsrc = psrc;
    gdst->Length += gsrc->Length;
    /* preserve the dst FirstPos so all other base group gets the same FirstPos */
}

static void fof_reduce_group(void * pdst, void * psrc) {
    struct Group * gdst = pdst;
    struct Group * gsrc = psrc;
    int j;
    gdst->Length += gsrc->Length;
    gdst->Mass += gsrc->Mass;

    for(j = 0; j < 6; j++)
    {
        gdst->LenType[j] += gsrc->LenType[j];
        gdst->MassType[j] += gsrc->MassType[j];
    }

#ifdef SFR
    gdst->Sfr += gsrc->Sfr;
#endif
#ifdef BLACK_HOLES
    gdst->BH_Mdot += gsrc->BH_Mdot;
    gdst->BH_Mass += gsrc->BH_Mass;
    if(gsrc->MaxDens > gdst->MaxDens)
    {
        gdst->MaxDens = gsrc->MaxDens;
        gdst->seed_index = gsrc->seed_index;
        gdst->seed_task = gsrc->seed_task;
    }
#endif

    int d1, d2;
    for(d1 = 0; d1 < 3; d1++)
    {
        gdst->CM[d1] += gsrc->CM[d1];
        gdst->Vel[d1] += gsrc->Vel[d1];
        gdst->Jmom[d1] += gsrc->Jmom[d1];
        for(d2 = 0; d2 < 3; d2 ++) {
            gdst->Imom[d1][d2] += gsrc->Imom[d1][d2];
        }
    }

}

static void add_particle_to_group(struct Group * gdst, int i) {

    /* My local number of particles contributing to the full catalogue. */
    const int index = i;
    if(gdst->Length == 0) {
        struct BaseGroup base = gdst->base;
        memset(gdst, 0, sizeof(gdst[0]));
        gdst->base = base;
        gdst->seed_index = gdst->seed_task = -1;
    }

    gdst->Length ++;
    gdst->Mass += P[index].Mass;
    gdst->LenType[P[index].Type]++;
    gdst->MassType[P[index].Type] += P[index].Mass;


#ifdef SFR
    if(P[index].Type == 0) {
        gdst->Sfr += get_starformation_rate(index);
    }
#endif
#ifdef BLACK_HOLES
    if(P[index].Type == 5)
    {
        gdst->BH_Mdot += BHP(index).Mdot;
        gdst->BH_Mass += BHP(index).Mass;
    }
    if(P[index].Type == 0)
    {
#ifdef WINDS
        /* make bh in non wind gas on bh wind*/
        if(SPHP(index).DelayTime <= 0)
#endif
            if(SPHP(index).Density > gdst->MaxDens)
            {
                gdst->MaxDens = SPHP(index).Density;
                gdst->seed_index = index;
                gdst->seed_task = ThisTask;
            }
    }
#endif

    int d1, d2;
    double xyz[3];
    double rel[3];
    double vel[3];
    double jmom[3];

    for(d1 = 0; d1 < 3; d1++)
    {
        double first = gdst->base.FirstPos[d1];
        rel[d1] = fof_periodic(P[index].Pos[d1] - first) ;
        xyz[d1] = rel[d1] + first;
        vel[d1] = P[index].Vel[d1];
    }

    crossproduct(rel, vel, jmom);

    for(d1 = 0; d1 < 3; d1++) {
        gdst->CM[d1] += P[index].Mass * xyz[d1];
        gdst->Vel[d1] += P[index].Mass * vel[d1];
        gdst->Jmom[d1] += P[index].Mass * jmom[d1];

        for(d2 = 0; d2 < 3; d2++) {
            gdst->Imom[d1][d2] += P[index].Mass * rel[d1] * rel[d2];
        }
    }
}

static void fof_finish_group_properties(void)
{
    int i;

    for(i = 0; i < Ngroups; i++)
    {
        int d1, d2;
        double cm[3];
        double rel[3];
        double jcm[3];
        double vcm[3];

        struct Group * gdst = &Group[i];
        for(d1 = 0; d1 < 3; d1++)
        {
            gdst->Vel[d1] /= gdst->Mass;
            vcm[d1] = gdst->Vel[d1];
            cm[d1] = gdst->CM[d1] / gdst->Mass;

            rel[d1] = fof_periodic(cm[d1] - gdst->base.FirstPos[d1]);

            cm[d1] = fof_periodic_wrap(cm[d1]);
            gdst->CM[d1] = cm[d1];

        }
        crossproduct(rel, vcm, jcm);

        for(d1 = 0; d1 < 3; d1 ++) {
            gdst->Jmom[d1] -= jcm[d1] * gdst->Mass;
        }

        for(d1 = 0; d1 < 3; d1 ++) {
            for(d2 = 0; d2 < 3; d2++) {
                /* Parallel Axis theorem:
                 * https://en.wikipedia.org/wiki/Parallel_axis_theorem ;
                 * J was relative to FirstPos, I is relative to CM.
                 *
                 * Note that our definition of Imom follows the astronomy one,
                 *
                 * I_ij = sum x_i x_j (where x_i x_j is relative displacement)
                 * */

                double diff = rel[d1] * rel[d2];

                gdst->Imom[d1][d2] -= gdst->Mass * diff;
            }
        }
    }

}

static void fof_compile_base(void)
{
    NgroupsExt = 0;

    int i;
    int start;

    for(i = 0; i < NumPart; i ++) {
        if(i == 0 || HaloLabel[i].MinID != HaloLabel[i - 1].MinID) NgroupsExt ++;
    }
    /* The first round is to eliminate groups that are too short. */
    /* We create the smaller 'BaseGroup' data set for this. */
    BaseGroup = (struct BaseGroup *) mymalloc("BaseGroup", sizeof(struct BaseGroup) * NgroupsExt);

    memset(BaseGroup, 0, sizeof(BaseGroup[0]) * NgroupsExt);

    start = 0;
    for(i = 0; i < NumPart; i++)
    {
        if(i == 0 || HaloLabel[i].MinID != HaloLabel[i - 1].MinID) {
            BaseGroup[start].MinID = HaloLabel[i].MinID;
            BaseGroup[start].MinIDTask = HaloLabel[i].MinIDTask;
            int d;
            for(d = 0; d < 3; d ++) {
                BaseGroup[start].FirstPos[d] = P[HaloLabel[i].Pindex].Pos[d];
            }
            start ++;
        }
    }

    /* count local lengths */
    /* This works because BaseGroup is sorted by MinID by construction. */
    start = 0;
    for(i = 0; i < NgroupsExt; i++)
    {
        /* find the first particle */
        for(;start < NumPart; start++) {
            if(HaloLabel[start].MinID >= BaseGroup[i].MinID) break;
        }
        /* count particles */
        for(;start < NumPart; start++) {
            if(HaloLabel[start].MinID != BaseGroup[i].MinID) {
                break;
            }
            BaseGroup[i].Length ++;
        }
    }

    /* update global attributes */
    fof_reduce_groups(BaseGroup, NgroupsExt, sizeof(BaseGroup[0]), fof_reduce_base_group);

    /* eliminate all groups that are too small */
    for(i = 0; i < NgroupsExt; i++)
    {
        if(BaseGroup[i].Length < All.FOFHaloMinLength)
        {
            BaseGroup[i] = BaseGroup[NgroupsExt - 1];
            NgroupsExt--;
            i--;
        }
    }

}


static void fof_compile_catalogue(void)
{
    int i, start;
    Group = (struct Group *) mymalloc("Group", sizeof(struct Group) * NgroupsExt);
    memset(Group, 0, sizeof(Group[0]) * NgroupsExt);

    /* copy in the base properties */

    /* at this point base group shall be sorted by MinID */
    for(i = 0; i < NgroupsExt; i ++) {
        Group[i].base = BaseGroup[i];
    }

    start = 0;
    for(i = 0; i < NgroupsExt; i++)
    {
        /* find the first particle */
        for(;start < NumPart; start++) {
            if(HaloLabel[start].MinID >= Group[i].base.MinID) break;
        }
        /* add particles */
        for(;start < NumPart; start++) {
            if(HaloLabel[start].MinID != Group[i].base.MinID) {
                break;
            }
            add_particle_to_group(&Group[i], HaloLabel[start].Pindex);
        }
    }

    /* collect global properties */
    fof_reduce_groups(Group, NgroupsExt, sizeof(Group[0]), fof_reduce_group);

    /* count Groups and number of particles hosted by me */
    Ngroups = 0;
    int64_t Nids = 0;
    for(i = 0; i < NgroupsExt; i ++) {
        if(Group[i].base.MinIDTask != ThisTask) continue;

        Ngroups++;
        Nids += Group[i].base.Length;

        if(Group[i].base.Length != Group[i].Length) {
            /* These two shall be consistent */
            endrun(3333, "Group base Length mismatch with Group Length");
        }
    }

    fof_finish_group_properties();

    MPI_Allreduce(&Ngroups, &TotNgroups, 1, MPI_UINT64, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&Nids, &TotNids, 1, MPI_INT64, MPI_SUM, MPI_COMM_WORLD);

    /* report some statictics */
    int largestgroup;
    if(TotNgroups > 0)
    {
        int largestloc = 0;

        for(i = 0; i < NgroupsExt; i++)
            if(Group[i].Length > largestloc)
                largestloc = Group[i].Length;
        MPI_Allreduce(&largestloc, &largestgroup, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    }
    else
        largestgroup = 0;

    message(0, "Total number of groups with at least %d particles: %ld\n", All.FOFHaloMinLength, TotNgroups);
    if(TotNgroups > 0)
    {
        message(0, "Largest group has %d particles.\n", largestgroup);
        message(0, "Total number of particles in groups: %012ld\n", TotNids);
    }
}


static void fof_reduce_groups(
    void * groups, 
    size_t nmemb, 
    size_t elsize, 
    void (*reduce_group)(void * gdst, void * gsrc))
{

    /* slangs: 
     *   prime: groups hosted by ThisTask
     *   ghosts: groups that spans into ThisTask but not hosted by ThisTask;
     *           part of the local catalogue
     *   images: ghosts that are sent from another rank.
     *           images are reduced to prime, then the prime attributes
     *           are copied to images, and sent back to the ghosts.
     *
     *   in the begining, prime and ghosts contains local group attributes.
     *   in the end, prime and ghosts all contain full group attributes.
     **/

    int * Send_count = alloca(sizeof(int) * NTask);
    int * Recv_count = alloca(sizeof(int) * NTask);
    void * images = NULL;
    void * ghosts = NULL;
    int i;
    int start;

    MPI_Datatype dtype;

    MPI_Type_contiguous(elsize, MPI_BYTE, &dtype);
    MPI_Type_commit(&dtype);

    /* local groups will be moved to the beginning, we skip them with offset */
    qsort(groups, nmemb, elsize, fof_compare_Group_MinIDTask);
    /* count how many we have of each task */
    memset(Send_count, 0, sizeof(int) * NTask);

    for(i = 0; i < nmemb; i++) {
        struct BaseGroup * gi = (struct BaseGroup *) (((char*) groups) + i * elsize);
        Send_count[gi->MinIDTask]++;
    }

    /* Skip local groups */
    int Nmine = Send_count[ThisTask];
    Send_count[ThisTask] = 0;

    MPI_Alltoall(Send_count, 1, MPI_INT, Recv_count, 1, MPI_INT, MPI_COMM_WORLD);

    int nimport = 0;
    for(i = 0; i < NTask; i ++) {
        nimport += Recv_count[i];
    }

    images = mymalloc("images", nimport * elsize);
    ghosts = ((char*) groups) + elsize * Nmine;

    MPI_Alltoallv_smart(ghosts, Send_count, NULL, dtype, 
                        images, Recv_count, NULL, dtype, MPI_COMM_WORLD);

    for(i = 0; i < nimport; i++) {
        struct BaseGroup * gi = (struct BaseGroup*) ((char*) images + i * elsize);
        gi->OriginalIndex = i;
    }
        
    /* sort the groups according to MinID */
    qsort(groups, Nmine, elsize, fof_compare_Group_MinID);
    qsort(images, nimport, elsize, fof_compare_Group_MinID);

    /* merge the imported ones with the local ones */
    start = 0;
    for(i = 0; i < Nmine; i++) {
        for(;start < nimport; start++) {
            struct BaseGroup * prime = (struct BaseGroup*) ((char*) groups + i * elsize);
            struct BaseGroup * image = (struct BaseGroup*) ((char*) images + start  * elsize);
            if(image->MinID >= prime->MinID) {
                break;
            }
        }
        for(;start < nimport; start++) {
            struct BaseGroup * prime = (struct BaseGroup*) ((char*) groups + i * elsize);
            struct BaseGroup * image = (struct BaseGroup*) ((char*) images + start * elsize);
            if(image->MinID != prime->MinID) {
                break;
            }
            reduce_group(prime, image);
        } 
    }

    /* update the images, such that they can be send back to the ghosts */
    start = 0;
    for(i = 0; i < Nmine; i++)
    {
        for(;start < nimport; start++) {
            struct BaseGroup * prime = (struct BaseGroup*) ((char*) groups + i * elsize);
            struct BaseGroup * image = (struct BaseGroup*) ((char*) images + start * elsize);
            if(image->MinID >= prime->MinID) {
                break;
            }
        }
        for(;start < nimport; start++) {
            struct BaseGroup * prime = (struct BaseGroup*) ((char*) groups + i * elsize);
            struct BaseGroup * image = (struct BaseGroup*) ((char*) images + start * elsize);
            if(image->MinID != prime->MinID) {
                break;
            }
            int save = image->OriginalIndex;
            memcpy(image, prime, elsize);
            image->OriginalIndex = save;
        } 
    }

    /* reset the ordering of imported list, such that it can be properly returned */
    qsort(images, nimport, elsize, fof_compare_Group_OriginalIndex);

    for(i = 0; i < nimport; i++) {
        struct BaseGroup * gi = (struct BaseGroup*) ((char*) images + i * elsize);
        if(gi->MinIDTask != ThisTask) {
            abort();
        }
    }
    void * ghosts2 = mymalloc("TMP", NgroupsExt * elsize);

    MPI_Alltoallv_smart(images, Recv_count, NULL, dtype, 
                        ghosts2, Send_count, NULL, dtype, 
                        MPI_COMM_WORLD);
    for(i = 0; i < NgroupsExt - Nmine; i ++) {
        struct BaseGroup * g1 = (struct BaseGroup*) ((char*) ghosts + i * elsize);
        struct BaseGroup * g2 = (struct BaseGroup*) ((char*) ghosts2 + i* elsize);
        if(g1->MinID != g2->MinID) {
            abort();
        }
        if(g1->MinIDTask != g2->MinIDTask) {
            abort();
        }
    }
    memcpy(ghosts, ghosts2, elsize * (NgroupsExt - Nmine));
    myfree(ghosts2);

    myfree(images);

    MPI_Type_free(&dtype);

    /* At this point, each Group entry has the reduced attribute of the full group */
    /* And the local groups (MinIDTask == ThisTask) are placed at the begining of the list*/
}

static void fof_radix_Group_TotalCountTaskDiffMinID(const void * a, void * radix, void * arg);
static void fof_radix_Group_OriginalTaskMinID(const void * a, void * radix, void * arg);

static void fof_assign_grnr()
{
    int i, j;
    int64_t ngr;

    for(i = 0; i < NgroupsExt; i++)
    {
        BaseGroup[i].OriginalTask = ThisTask;	/* original task */
    }

    mpsort_mpi(BaseGroup, NgroupsExt, sizeof(BaseGroup[0]),
            fof_radix_Group_TotalCountTaskDiffMinID, 24, NULL, MPI_COMM_WORLD);

    /* assign group numbers 
     * at this point, both Group are is sorted by length,
     * and the every time OriginalTask == MinIDTask, a list of ghost BaseGroup is stored.
     * they shall get the same GrNr.
     * */
    ngr = 0;
    for(i = 0; i < NgroupsExt; i++)
    {
        if(BaseGroup[i].OriginalTask == BaseGroup[i].MinIDTask) {
            ngr++;
        }
        BaseGroup[i].GrNr = ngr;
    }

    int64_t * ngra;
    ngra = alloca(sizeof(ngra[0]) * NTask);
    MPI_Allgather(&ngr, 1, MPI_INT64, ngra, 1, MPI_INT64, MPI_COMM_WORLD);

    /* shift to the global grnr. */
    int64_t groffset = 0;
    for(j = 0; j < ThisTask; j++)
        groffset += ngra[j];
    for(i = 0; i < NgroupsExt; i++)
        BaseGroup[i].GrNr += groffset;

    /* bring the group list back into the original task, sorted by MinID */
    mpsort_mpi(BaseGroup, NgroupsExt, sizeof(BaseGroup[0]), 
            fof_radix_Group_OriginalTaskMinID, 16, NULL, MPI_COMM_WORLD);

    for(i = 0; i < NumPart; i++)
        P[i].GrNr = -1;	/* will mark particles that are not in any group */

    int start = 0;
    for(i = 0; i < NgroupsExt; i++)
    {
        for(;start < NumPart; start++) {
            if (HaloLabel[start].MinID >= BaseGroup[i].MinID) 
                break;
        }

        for(;start < NumPart; start++) {
            if (HaloLabel[start].MinID != BaseGroup[i].MinID) 
                break;
            P[HaloLabel[start].Pindex].GrNr = BaseGroup[i].GrNr;
        }
    }
}


void fof_save_groups(int num)
{
    double t0, t1;

    message(0, "start global sorting of group catalogues\n");

    t0 = second();

    fof_save_particles(num);

    t1 = second();

    message(0, "Group catalogues saved. took = %g sec\n", timediff(t0, t1));
}

static void fof_secondary_copy(int place, TreeWalkQueryFOF * I, TreeWalk * tw) {
    I->Hsml = fof_secondary_hsml[place];
}
static int fof_secondary_isactive(int n, TreeWalk * tw) {
    return (((1 << P[n].Type) & (FOF_SECONDARY_LINK_TYPES)));
}
static void fof_secondary_reduce(int place, TreeWalkResultFOF * O, enum TreeWalkReduceMode mode, TreeWalk * tw) {
    if(O->Distance < fof_secondary_distance[place])
    {
        fof_secondary_distance[place] = O->Distance;
        HaloLabel[place].MinID = O->MinID;
        HaloLabel[place].MinIDTask = O->MinIDTask;
    }
}
static void
fof_secondary_ngbiter(TreeWalkQueryFOF * I,
        TreeWalkResultFOF * O,
        TreeWalkNgbIterFOF * iter,
        LocalTreeWalk * lv);

static void fof_label_secondary(void)
{
    int i, n, iter;
    int64_t ntot;
    TreeWalk tw[1] = {0};
    tw->ev_label = "FOF_FIND_NEAREST";
    tw->visit = (TreeWalkVisitFunction) treewalk_visit_ngbiter;
    tw->ngbiter = (TreeWalkNgbIterFunction) fof_secondary_ngbiter;
    tw->ngbiter_type_elsize = sizeof(TreeWalkNgbIterFOF);
    tw->isactive = fof_secondary_isactive;
    tw->fill = (TreeWalkFillQueryFunction) fof_secondary_copy;
    tw->reduce = (TreeWalkReduceResultFunction) fof_secondary_reduce;
    tw->UseNodeList = 1;
    tw->UseAllParticles = 1;
    tw->query_type_elsize = sizeof(TreeWalkQueryFOF);
    tw->result_type_elsize = sizeof(TreeWalkResultFOF);

    message(0, "Start finding nearest dm-particle (presently allocated=%g MB)\n",
            AllocatedBytes / (1024.0 * 1024.0));

    fof_secondary_distance = (float *) mymalloc("fof_secondary_distance", sizeof(float) * NumPart);
    fof_secondary_hsml = (float *) mymalloc("fof_secondary_hsml", sizeof(float) * NumPart);

    for(n = 0; n < NumPart; n++)
    {
        if(((1 << P[n].Type) & (FOF_SECONDARY_LINK_TYPES)))
        {
            fof_secondary_distance[n] = LARGE;
            if(P[n].Type == 0) {
                /* use gas sml as a hint (faster convergence than 0.1 All.FOFHaloComovingLinkingLength at high-z */
                fof_secondary_hsml[n] = 0.5 * P[n].Hsml;
            } else {
                fof_secondary_hsml[n] = 0.1 * All.FOFHaloComovingLinkingLength;
            }
        }
    }

    /* allocate buffers to arrange communication */

    iter = 0;
    /* we will repeat the whole thing for those particles where we didn't find enough neighbours */

    message(0, "fof-nearest iteration started\n");

    do 
    {
        treewalk_run(tw);

        int queuesize;
        int * queue = treewalk_get_queue(tw, &queuesize);

        /* do final operations on results */
        int npleft = 0;
        int count = 0;
        int64_t counttot = 0;
/* CRAY cc doesn't do this one right */
//#pragma omp parallel for reduction(+: npleft)
        for(i = 0; i < queuesize; i++)
        {
            int p = queue[i];
            count ++;
            if(fof_secondary_distance[p] > 0.5 * LARGE)
            {
                if(fof_secondary_hsml[p] < 4 * All.FOFHaloComovingLinkingLength)  /* we only search out to a maximum distance */
                {
                    /* need to redo this particle */
                    npleft++;
                    fof_secondary_hsml[p] *= 2.0;
/*
                    if(iter >= MAXITER - 10)
                    {
                        endrun(1, "i=%d task=%d ID=%llu Hsml=%g  pos=(%g|%g|%g)\n",
                                p, ThisTask, P[p].ID, fof_secondary_hsml[p],
                                P[p].Pos[0], P[p].Pos[1], P[p].Pos[2]);
                    }
*/
                } else {
                    fof_secondary_distance[p] = 0;  /* we not continue to search for this particle */
                }
            }
        }
        sumup_large_ints(1, &npleft, &ntot);
        sumup_large_ints(1, &count, &counttot);

        message(0, "fof-nearest iteration %d: need to repeat for %010ld /%010ld particles.\n", iter, ntot, counttot);

        if(ntot < 0) abort();
        if(ntot > 0)
        {
            iter++;
            if(iter > MAXITER)
            {
                endrun(1159, "Failed to converge in fof-nearest");
            }
        }
        myfree(queue);
    }
    while(ntot > 0);

    myfree(fof_secondary_hsml);
    myfree(fof_secondary_distance);

    message(0, "done finding nearest dm-particle\n");
}

static void
fof_secondary_ngbiter( TreeWalkQueryFOF * I,
        TreeWalkResultFOF * O,
        TreeWalkNgbIterFOF * iter,
        LocalTreeWalk * lv)
{
    if(iter->base.other == -1) {
        O->Distance = LARGE;
        iter->base.Hsml = I->Hsml;
        iter->base.mask = FOF_PRIMARY_LINK_TYPES;
        iter->base.symmetric = NGB_TREEFIND_ASYMMETRIC;
        return;
    }
    int other = iter->base.other;
    double r = iter->base.r;
    if(r < O->Distance && r < I->Hsml)
    {
        O->Distance = r;
        O->MinID = HaloLabel[other].MinID;
        O->MinIDTask = HaloLabel[other].MinIDTask;
    }
}

/* 
 * Deal with seeding of particles At each FOF stage,
 * if seed_index is >= 0,  then that particle on seed_task
 * will be converted to a seed.
 *
 * */
static int cmp_seed_task(const void * c1, const void * c2) {
    const struct Group * g1 = c1;
    const struct Group * g2 = c2;

    return g1->seed_task - g2->seed_task;
}
static void fof_seed_make_one(struct Group * g);

static void fof_seed(void)
{
    int i, j, n, ntot;

    int * Send_count = alloca(sizeof(int) * NTask);
    int * Recv_count = alloca(sizeof(int) * NTask);

    for(n = 0; n < NTask; n++)
        Send_count[n] = 0;

    char * Marked = mymalloc("SeedMark", Ngroups);
    
    int Nexport = 0;
    for(i = 0; i < Ngroups; i++)
    {
        Marked[i] = 
            (Group[i].Mass >= All.MinFoFMassForNewSeed)
        &&  (Group[i].LenType[5] == 0)
        &&  (Group[i].seed_index >= 0);

        if(Marked[i]) Nexport ++;
    }
    struct Group * ExportGroups = mymalloc("Export", sizeof(Group[0]) * Nexport);
    j = 0;
    for(i = 0; i < Ngroups; i ++) {
        if(Marked[i]) {
            ExportGroups[j] = Group[i];
            j++;
        }
    }
    qsort(ExportGroups, Nexport, sizeof(ExportGroups[0]), cmp_seed_task);

    for(i = 0; i < Nexport; i++) {
        Send_count[ExportGroups[i].seed_task]++;
    }

    MPI_Alltoall(Send_count, 1, MPI_INT, Recv_count, 1, MPI_INT, MPI_COMM_WORLD);

    int Nimport = 0;

    for(j = 0;  j < NTask; j++)
    {
        Nimport += Recv_count[j];
    }

    struct Group * ImportGroups = (struct Group *) 
            mymalloc("ImportGroups", Nimport * sizeof(struct Group));

    MPI_Alltoallv_smart(ExportGroups, Send_count, NULL, MPI_TYPE_GROUP, 
                        ImportGroups, Recv_count, NULL, MPI_TYPE_GROUP,
                        MPI_COMM_WORLD);

    MPI_Allreduce(&Nimport, &ntot, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    message(0, "Making %d new black hole particles.\n", ntot);

    for(n = 0; n < Nimport; n++)
    {
        fof_seed_make_one(&ImportGroups[n]);
    }

    myfree(ImportGroups);
    myfree(ExportGroups);
    myfree(Marked);
}

static void fof_seed_make_one(struct Group * g) {
    if(g->seed_task != ThisTask) {
        endrun(7771, "Seed does not belong to the right task");
    }
#ifdef BLACK_HOLES
    int index = g->seed_index;
    blackhole_make_one(index);
#endif
}

static int fof_compare_HaloLabel_MinID(const void *a, const void *b)
{
    if(((struct fof_particle_list *) a)->MinID < ((struct fof_particle_list *) b)->MinID)
        return -1;

    if(((struct fof_particle_list *) a)->MinID > ((struct fof_particle_list *) b)->MinID)
        return +1;

    return 0;
}

static int fof_compare_Group_MinID(const void *a, const void *b)

{
    if(((struct BaseGroup *) a)->MinID < ((struct BaseGroup *) b)->MinID)
        return -1;

    if(((struct BaseGroup *) a)->MinID > ((struct BaseGroup *) b)->MinID)
        return +1;

    return 0;
}

static int fof_compare_Group_MinIDTask(const void *a, const void *b)
{
    const struct BaseGroup * p1 = a;
    const struct BaseGroup * p2 = b;
    int t1 = p1->MinIDTask;
    int t2 = p2->MinIDTask;
    if(t1 == ThisTask) t1 = -1;
    if(t2 == ThisTask) t2 = -1;

    if(t1 < t2) return -1;
    if(t1 > t2) return +1;
    return 0;

}

static int fof_compare_Group_OriginalIndex(const void *a, const void *b)

{
    return ((struct BaseGroup *) a)->OriginalIndex - ((struct BaseGroup *) b)->OriginalIndex;
}

static void fof_radix_Group_TotalCountTaskDiffMinID(const void * a, void * radix, void * arg) {
    uint64_t * u = (uint64_t *) radix;
    struct BaseGroup * f = (struct BaseGroup *) a;
    u[0] = labs(f->OriginalTask - f->MinIDTask);
    u[1] = f->MinID;
    u[2] = UINT64_MAX - (f->Length);
}

static void fof_radix_Group_OriginalTaskMinID(const void * a, void * radix, void * arg) {
    uint64_t * u = (uint64_t *) radix;
    struct BaseGroup * f = (struct BaseGroup *) a;
    u[0] = f->MinID;
    u[1] = f->OriginalTask;
}
