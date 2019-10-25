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
#include <mpsort.h>

#include "utils.h"

#include "walltime.h"
#include "sfr_eff.h"
#include "blackhole.h"
#include "domain.h"

#include "forcetree.h"
#include "treewalk.h"
#include "slotsmanager.h"
#include "partmanager.h"
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
#define MAXITER 400

struct FOFParams
{
    int FOFSaveParticles ; /* saving particles in the fof group */
    double MinFoFMassForNewSeed;	/* Halo mass required before new seed is put in */
    double FOFHaloLinkingLength;
    double FOFHaloComovingLinkingLength; /* in code units */
    int FOFHaloMinLength;
} fof_params;

/*Set the parameters of the BH module*/
void set_fof_params(ParameterSet * ps)
{
    int ThisTask;
    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
    if(ThisTask == 0) {
        fof_params.FOFSaveParticles = param_get_int(ps, "FOFSaveParticles");
        fof_params.FOFHaloLinkingLength = param_get_double(ps, "FOFHaloLinkingLength");
        fof_params.FOFHaloMinLength = param_get_int(ps, "FOFHaloMinLength");
        fof_params.MinFoFMassForNewSeed = param_get_double(ps, "MinFoFMassForNewSeed");
    }
    MPI_Bcast(&fof_params, sizeof(struct FOFParams), MPI_BYTE, 0, MPI_COMM_WORLD);
}

void fof_init(double DMMeanSeparation)
{
    fof_params.FOFHaloComovingLinkingLength = fof_params.FOFHaloLinkingLength * DMMeanSeparation;
}

static double fof_periodic(double x, double BoxSize)
{
    if(x >= 0.5 * BoxSize)
        x -= BoxSize;
    if(x < -0.5 * BoxSize)
        x += BoxSize;
    return x;
}


static double fof_periodic_wrap(double x, double BoxSize)
{
    while(x >= BoxSize)
        x -= BoxSize;
    while(x < 0)
        x += BoxSize;
    return x;
}

static void fof_label_secondary(ForceTree * tree);
static int fof_compare_HaloLabel_MinID(const void *a, const void *b);
static int _fof_compare_Group_MinIDTask_ThisTask;
static int fof_compare_Group_MinIDTask(const void *a, const void *b);
static int fof_compare_Group_OriginalIndex(const void *a, const void *b);
static int fof_compare_Group_MinID(const void *a, const void *b);
static void fof_reduce_groups(
    void * groups,
    int nmemb,
    size_t elsize,
    void (*reduce_group)(void * gdst, void * gsrc), MPI_Comm Comm);

static void fof_finish_group_properties(FOFGroups * fof, double BoxSize);

static int fof_compile_base(struct BaseGroup * base, int NgroupsExt, MPI_Comm Comm);
static void fof_compile_catalogue(FOFGroups * fof, const int NgroupsExt, double BoxSize, int BlackHoleInfo, MPI_Comm Comm);

static struct Group *
fof_alloc_group(const struct BaseGroup * base, const int NgroupsExt);

static void fof_assign_grnr(struct BaseGroup * base, const int NgroupsExt, MPI_Comm Comm);

void fof_label_primary(ForceTree * tree, MPI_Comm Comm);
extern void fof_save_particles(FOFGroups * fof, int num, int SaveParticles, MPI_Comm Comm);

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

static MPI_Datatype MPI_TYPE_GROUP;

/*
 * The FOF finder will produce Group[], which is allocated to the top side of the
 * main heap.
 *
 **/

FOFGroups
fof_fof(ForceTree * tree, double BoxSize, int BlackHoleInfo, MPI_Comm Comm)
{
    int i;

    message(0, "Begin to compute FoF group catalogues. (allocated: %g MB)\n",
            mymalloc_usedbytes() / (1024.0 * 1024.0));

    walltime_measure("/Misc");

    message(0, "Comoving linking length: %g\n", fof_params.FOFHaloComovingLinkingLength);

    HaloLabel = (struct fof_particle_list *) mymalloc("HaloLabel", PartManager->NumPart * sizeof(struct fof_particle_list));

    /* HaloLabel stores the MinID and MinIDTask of particles, this pair serves as a halo label. */
    #pragma omp parallel for
    for(i = 0; i < PartManager->NumPart; i++) {
        HaloLabel[i].Pindex = i;
    }
    /* Fill FOFP_List of primary */
    fof_label_primary(tree, Comm);

    MPIU_Barrier(Comm);
    message(0, "Group finding done.\n");
    walltime_measure("/FOF/Primary");

    /* Fill FOFP_List of secondary */
    fof_label_secondary(tree);
    MPIU_Barrier(Comm);
    message(0, "Attached gas and star particles to nearest dm particles.\n");

    walltime_measure("/FOF/Secondary");

    /* sort HaloLabel according to MinID, because we need that for compiling catalogues */
    qsort_openmp(HaloLabel, PartManager->NumPart, sizeof(struct fof_particle_list), fof_compare_HaloLabel_MinID);

    int NgroupsExt = 0;

    for(i = 0; i < PartManager->NumPart; i ++) {
        if(i == 0 || HaloLabel[i].MinID != HaloLabel[i - 1].MinID) NgroupsExt ++;
    }

    /* The first round is to eliminate groups that are too short. */
    /* We create the smaller 'BaseGroup' data set for this. */
    struct BaseGroup * base = (struct BaseGroup *) mymalloc("BaseGroup", sizeof(struct BaseGroup) * NgroupsExt);

    NgroupsExt = fof_compile_base(base, NgroupsExt, Comm);

    MPIU_Barrier(Comm);
    message(0, "Compiled local group data and catalogue.\n");

    walltime_measure("/FOF/Compile");

    fof_assign_grnr(base, NgroupsExt, Comm);

    /*Initialise the Group object from the BaseGroup*/
    FOFGroups fof;
    MPI_Type_contiguous(sizeof(fof.Group[0]), MPI_BYTE, &MPI_TYPE_GROUP);
    MPI_Type_commit(&MPI_TYPE_GROUP);

    fof.Group = fof_alloc_group(base, NgroupsExt);

    myfree(base);

    fof_compile_catalogue(&fof, NgroupsExt, BoxSize, BlackHoleInfo, Comm);

    MPIU_Barrier(Comm);
    message(0, "Finished FoF. Group properties are now allocated.. (presently allocated=%g MB)\n",
            mymalloc_usedbytes() / (1024.0 * 1024.0));

    walltime_measure("/FOF/Prop");

    myfree(HaloLabel);

    return fof;
}

void
fof_finish(FOFGroups * fof)
{
    myfree(fof->Group);

    message(0, "Finished computing FoF groups.  (presently allocated=%g MB)\n",
            mymalloc_usedbytes() / (1024.0 * 1024.0));

    walltime_measure("/FOF/MISC");

    MPI_Type_free(&MPI_TYPE_GROUP);
}

struct FOFPrimaryPriv {
    int * Head;
    struct SpinLocks * spin;
    char * PrimaryActive;
    MyIDType * OldMinID;
};
#define FOF_PRIMARY_GET_PRIV(tw) ((struct FOFPrimaryPriv *) (tw->priv))

/* This function walks the particle tree starting at particle i until it reaches
 * a particle which has Head[i] = i, the root node (particles are initialised in
 * this state, so this is equivalent to finding a particle which has yet to be merged).
 * Each particle is locked along the way, and unlocked once it is clear that it is not a root.
 * Once it reaches a root, it returns that particle number with the lock still held.
 * There are two other arguments:
 *
 * stop: When this particle is reached, return -1. We use this to find an already merged tree.
 *
 * locked: This particle is already locked (setting locked = -1 means no other locks are taken).
 *         When we try to lock a particle locked by another thread, the code checks whether
 *         it can safely take a second lock. If it cannot, it returns -2, with the expectation
 *         that the first lock is released before a retry.
 *
 * Returns:
 *      root particle if found
 *      -1 if stop particle reached
 *      -2 if locking might have deadlocked.
 */
static int
HEADl(int stop, int i, int locked, int * Head, struct SpinLocks * spin)
{
    int r, next;
    if (i == stop) {
        return -1;
    }
//    printf("locking %d by %d in HEADl stop = %d\n", i, omp_get_thread_num(), stop);
    /* Try to lock the particle. Wait on the lock if it is less than the already locked particle,
     * or if such a particle does not exist. We could use atomic to avoid the lock, but this
     * makes the locking code less clear, and doesn't lead to much of a speedup: the splay
     * means that the tree is shallow.*/
    if(locked < 0 || i < locked) {
        lock_spinlock(i, spin);
    }
    else{
        if(try_lock_spinlock(i, spin)) {
            /*This means some other thread already has the lock.
             *To avoid deadlocks we need to back off, unlock both particles and then retry.*/
            return -2;
        }
    }
//    printf("locked %d by %d in HEADl, next = %d\n", i, omp_get_thread_num(), FOF_PRIMARY_GET_PRIV(tw)->Head[i]);
    /* atomic read because we may change
     * this in update_root without the lock: not necessary on x86_64, but avoids tears elsewhere*/
    #pragma omp atomic read
    next = Head[i];

    if(next == i) {
        /* return locked */
        return i;
    }
    /* this is not the root, keep going, but unlock first, since even if the root is modified by
     * another thread, what we get here is on the path, */
    unlock_spinlock(i, spin);
//    printf("unlocking %d by %d in HEADl\n", i, omp_get_thread_num());
    r = HEADl(stop, next, locked, Head, spin);
    return r;
}

/* Rewrite a tree so that all values in it point directly to the true root.
 * This means that the trees are O(1) deep and speeds up future accesses. */
static void
update_root(int i, const int r, int * Head)
{
    int t = i;
    do {
        i = t;
        #pragma omp atomic capture
        {
            t = Head[i];
            Head[i]= r;
        }
    } while(t != i);
}

static int
HEAD(int i, TreeWalk * tw)
{
    /* accelerate with a splay: see https://arxiv.org/abs/1607.03224 */
    int r;
    r = i;
    while(FOF_PRIMARY_GET_PRIV(tw)->Head[r] != r) {
        r = FOF_PRIMARY_GET_PRIV(tw)->Head[r];
    }
    while(FOF_PRIMARY_GET_PRIV(tw)->Head[i] != i) {
        int t = FOF_PRIMARY_GET_PRIV(tw)->Head[i];
        FOF_PRIMARY_GET_PRIV(tw)->Head[i]= r;
        i = t;
    }
    return r;
}

static void fof_primary_copy(int place, TreeWalkQueryFOF * I, TreeWalk * tw) {
    int head = HEAD(place, tw);
    I->MinID = HaloLabel[head].MinID;
    I->MinIDTask = HaloLabel[head].MinIDTask;
}

static int fof_primary_haswork(int n, TreeWalk * tw) {
    return (((1 << P[n].Type) & (FOF_PRIMARY_LINK_TYPES))) && FOF_PRIMARY_GET_PRIV(tw)->PrimaryActive[n];
}

static void
fof_primary_ngbiter(TreeWalkQueryFOF * I,
        TreeWalkResultFOF * O,
        TreeWalkNgbIterFOF * iter,
        LocalTreeWalk * lv);

void fof_label_primary(ForceTree * tree, MPI_Comm Comm)
{
    int i;
    int64_t link_across;
    int64_t link_across_tot;
    double t0, t1;
    int ThisTask;
    MPI_Comm_rank(Comm, &ThisTask);

    message(0, "Start linking particles (presently allocated=%g MB)\n", mymalloc_usedbytes() / (1024.0 * 1024.0));

    TreeWalk tw[1] = {{0}};
    tw->ev_label = "FOF_FIND_GROUPS";
    tw->visit = (TreeWalkVisitFunction) treewalk_visit_ngbiter;
    tw->ngbiter = (TreeWalkNgbIterFunction) fof_primary_ngbiter;
    tw->ngbiter_type_elsize = sizeof(TreeWalkNgbIterFOF);

    tw->haswork = fof_primary_haswork;
    tw->fill = (TreeWalkFillQueryFunction) fof_primary_copy;
    tw->reduce = NULL;
    tw->type = TREEWALK_ALL;
    tw->query_type_elsize = sizeof(TreeWalkQueryFOF);
    tw->result_type_elsize = sizeof(TreeWalkResultFOF);
    tw->tree = tree;
    struct FOFPrimaryPriv priv[1];
    tw->priv = priv;

    FOF_PRIMARY_GET_PRIV(tw)->Head = (int*) mymalloc("FOF_Links", PartManager->NumPart * sizeof(int));
    FOF_PRIMARY_GET_PRIV(tw)->PrimaryActive = (char*) mymalloc("FOFActive", PartManager->NumPart * sizeof(char));
    FOF_PRIMARY_GET_PRIV(tw)->OldMinID = (MyIDType *) mymalloc("FOFActive", PartManager->NumPart * sizeof(MyIDType));

    /* allocate buffers to arrange communication */

    t0 = second();

    for(i = 0; i < PartManager->NumPart; i++)
    {
        FOF_PRIMARY_GET_PRIV(tw)->Head[i] = i;
        FOF_PRIMARY_GET_PRIV(tw)->OldMinID[i]= P[i].ID;
        FOF_PRIMARY_GET_PRIV(tw)->PrimaryActive[i] = 1;

        HaloLabel[i].MinID = P[i].ID;
        HaloLabel[i].MinIDTask = ThisTask;
    }

    priv[0].spin = init_spinlocks(PartManager->NumPart);
    do
    {
        t0 = second();

        treewalk_run(tw, NULL, PartManager->NumPart);

        t1 = second();

        /* let's check out which particles have changed their MinID,
         * mark them for next round. */
        link_across = 0;
#pragma omp parallel for reduction(+: link_across)
        for(i = 0; i < PartManager->NumPart; i++) {
            MyIDType newMinID = HaloLabel[HEAD(i, tw)].MinID;
            if(newMinID != FOF_PRIMARY_GET_PRIV(tw)->OldMinID[i]) {
                FOF_PRIMARY_GET_PRIV(tw)->PrimaryActive[i] = 1;
                link_across ++;
            } else {
                FOF_PRIMARY_GET_PRIV(tw)->PrimaryActive[i] = 0;
            }
            FOF_PRIMARY_GET_PRIV(tw)->OldMinID[i] = newMinID;
        }
        MPI_Allreduce(&link_across, &link_across_tot, 1, MPI_INT64, MPI_SUM, Comm);
        message(0, "Linked %ld particles %g seconds\n", link_across_tot, t1 - t0);
    }
    while(link_across_tot > 0);

    free_spinlocks(priv[0].spin);

    /* Update MinID of all linked (primary-linked) particles */
    for(i = 0; i < PartManager->NumPart; i++)
    {
        HaloLabel[i].MinID = HaloLabel[HEAD(i, tw)].MinID;
        HaloLabel[i].MinIDTask = HaloLabel[HEAD(i, tw)].MinIDTask;
    }

    message(0, "Local groups found.\n");

    myfree(FOF_PRIMARY_GET_PRIV(tw)->OldMinID);
    myfree(FOF_PRIMARY_GET_PRIV(tw)->PrimaryActive);
    myfree(FOF_PRIMARY_GET_PRIV(tw)->Head);
}

static void
fofp_merge(int target, int other, TreeWalk * tw)
{
    /* this will lock h1 */
    int * Head = FOF_PRIMARY_GET_PRIV(tw)->Head;
    struct SpinLocks * spin = FOF_PRIMARY_GET_PRIV(tw)->spin;
    int h1, h2;

    do {
        h1 = HEADl(-1, target, -1, Head, spin);
        /* stop looking if we find h1 along the path (because it is already owned by us) */
        h2 = HEADl(h1, other, h1, Head, spin);
        /* We had a lock already taken on h2 by another thread.
         * We need to unlock h1 and retry to avoid deadlock loops.*/
        if(h2 == -2)
            unlock_spinlock(h1, spin);
    } while(h2 == -2);

    if(h2 >=0)
    {
        /* h2 as a sub-tree of h1 */
        Head[h2] = h1;

        /* update MinID of h1 */
        if(HaloLabel[h1].MinID > HaloLabel[h2].MinID)
        {
            HaloLabel[h1].MinID = HaloLabel[h2].MinID;
            HaloLabel[h1].MinIDTask = HaloLabel[h2].MinIDTask;
        }
        //printf("unlocking %d by %d in merge\n", h2, omp_get_thread_num());
        unlock_spinlock(h2, spin);
    }

    /* h1 must be the root of other and target both:
     * do the splay to speed up future accesses.
     * We do not need to have h2 locked, because h2 is
     * now just another child of h1: these do not change the root,
     * they make the tree shallow.*/
    update_root(target, h1, Head);
    update_root(other, h1, Head);

    //printf("unlocking %d by %d in merge\n", h1, omp_get_thread_num());
    unlock_spinlock(h1, spin);
}

static void
fof_primary_ngbiter(TreeWalkQueryFOF * I,
        TreeWalkResultFOF * O,
        TreeWalkNgbIterFOF * iter,
        LocalTreeWalk * lv)
{
    TreeWalk * tw = lv->tw;
    if(iter->base.other == -1) {
        iter->base.Hsml = fof_params.FOFHaloComovingLinkingLength;
        iter->base.symmetric = NGB_TREEFIND_ASYMMETRIC;
        iter->base.mask = FOF_PRIMARY_LINK_TYPES;
        return;
    }
    int other = iter->base.other;

    if(lv->mode == 0) {
        /* Local FOF */
        if(lv->target <= other) {
            // printf("locked merge %d %d by %d\n", lv->target, other, omp_get_thread_num());
            fofp_merge(lv->target, other, tw);
        }
    } else /* mode is 1, target is a ghost */
    {
            struct SpinLocks * spin = FOF_PRIMARY_GET_PRIV(tw)->spin;
//        printf("locking %d by %d in ngbiter\n", other, omp_get_thread_num());
        lock_spinlock(other, spin);
        if(HaloLabel[HEAD(other, tw)].MinID > I->MinID)
        {
            HaloLabel[HEAD(other, tw)].MinID = I->MinID;
            HaloLabel[HEAD(other, tw)].MinIDTask = I->MinIDTask;
        }
//        printf("unlocking %d by %d in ngbiter\n", other, omp_get_thread_num());
        unlock_spinlock(other, spin);
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

    gdst->Sfr += gsrc->Sfr;
    gdst->BH_Mdot += gsrc->BH_Mdot;
    gdst->BH_Mass += gsrc->BH_Mass;
    if(gsrc->MaxDens > gdst->MaxDens)
    {
        gdst->MaxDens = gsrc->MaxDens;
        gdst->seed_index = gsrc->seed_index;
        gdst->seed_task = gsrc->seed_task;
    }

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

static void add_particle_to_group(struct Group * gdst, int i, double BoxSize, int ThisTask, int BlackHoleOn) {

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


    if(P[index].Type == 0) {
        gdst->Sfr += SPHP(index).Sfr;
    }
    if(BlackHoleOn && P[index].Type == 5)
    {
        gdst->BH_Mdot += BHP(index).Mdot;
        gdst->BH_Mass += BHP(index).Mass;
    }
    /*This used to depend on black holes being enabled, but I do not see why.
     * I think because it is only useful for seeding*/
    if(P[index].Type == 0)
    {
        /* make bh in non wind gas on bh wind*/
        if(SPHP(index).DelayTime <= 0)
            if(SPHP(index).Density > gdst->MaxDens)
            {
                gdst->MaxDens = SPHP(index).Density;
                gdst->seed_index = index;
                gdst->seed_task = ThisTask;
            }
    }

    int d1, d2;
    double xyz[3];
    double rel[3];
    double vel[3];
    double jmom[3];

    for(d1 = 0; d1 < 3; d1++)
    {
        double first = gdst->base.FirstPos[d1];
        rel[d1] = fof_periodic(P[index].Pos[d1] - first, BoxSize) ;
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

static void
fof_finish_group_properties(struct FOFGroups * fof, double BoxSize)
{
    int i;

    for(i = 0; i < fof->Ngroups; i++)
    {
        int d1, d2;
        double cm[3];
        double rel[3];
        double jcm[3];
        double vcm[3];

        struct Group * gdst = &fof->Group[i];
        for(d1 = 0; d1 < 3; d1++)
        {
            gdst->Vel[d1] /= gdst->Mass;
            vcm[d1] = gdst->Vel[d1];
            cm[d1] = gdst->CM[d1] / gdst->Mass;

            rel[d1] = fof_periodic(cm[d1] - gdst->base.FirstPos[d1], BoxSize);

            cm[d1] = fof_periodic_wrap(cm[d1], BoxSize);
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

static int
fof_compile_base(struct BaseGroup * base, int NgroupsExt, MPI_Comm Comm)
{
    memset(base, 0, sizeof(base[0]) * NgroupsExt);

    int i;
    int start;

    start = 0;
    for(i = 0; i < PartManager->NumPart; i++)
    {
        if(i == 0 || HaloLabel[i].MinID != HaloLabel[i - 1].MinID) {
            base[start].MinID = HaloLabel[i].MinID;
            base[start].MinIDTask = HaloLabel[i].MinIDTask;
            int d;
            for(d = 0; d < 3; d ++) {
                base[start].FirstPos[d] = P[HaloLabel[i].Pindex].Pos[d];
            }
            start ++;
        }
    }

    /* count local lengths */
    /* This works because base is sorted by MinID by construction. */
    start = 0;
    for(i = 0; i < NgroupsExt; i++)
    {
        /* find the first particle */
        for(;start < PartManager->NumPart; start++) {
            if(HaloLabel[start].MinID >= base[i].MinID) break;
        }
        /* count particles */
        for(;start < PartManager->NumPart; start++) {
            if(HaloLabel[start].MinID != base[i].MinID) {
                break;
            }
            base[i].Length ++;
        }
    }

    /* update global attributes */
    fof_reduce_groups(base, NgroupsExt, sizeof(base[0]), fof_reduce_base_group, Comm);

    /* eliminate all groups that are too small */
    for(i = 0; i < NgroupsExt; i++)
    {
        if(base[i].Length < fof_params.FOFHaloMinLength)
        {
            base[i] = base[NgroupsExt - 1];
            NgroupsExt--;
            i--;
        }
    }
    return NgroupsExt;
}

/* Allocate memory for and initialise a Group object
 * from a BaseGroup object.*/
static struct Group *
fof_alloc_group(const struct BaseGroup * base, const int NgroupsExt)
{
    int i;
    struct Group * Group = (struct Group *) mymalloc2("Group", sizeof(struct Group) * NgroupsExt);
    memset(Group, 0, sizeof(Group[0]) * NgroupsExt);

    /* copy in the base properties */
    /* at this point base group shall be sorted by MinID */
    #pragma omp parallel for
    for(i = 0; i < NgroupsExt; i ++) {
        Group[i].base = base[i];
    }
    return Group;
}

static void
fof_compile_catalogue(struct FOFGroups * fof, const int NgroupsExt, double BoxSize, int BlackHoleInfo, MPI_Comm Comm)
{
    int i, start, ThisTask;

    MPI_Comm_rank(Comm, &ThisTask);

    start = 0;
    for(i = 0; i < NgroupsExt; i++)
    {
        /* find the first particle */
        for(;start < PartManager->NumPart; start++) {
            if(HaloLabel[start].MinID >= fof->Group[i].base.MinID) break;
        }
        /* add particles */
        for(;start < PartManager->NumPart; start++) {
            if(HaloLabel[start].MinID != fof->Group[i].base.MinID) {
                break;
            }
            add_particle_to_group(&fof->Group[i], HaloLabel[start].Pindex, BoxSize, ThisTask, BlackHoleInfo);
        }
    }

    /* collect global properties */
    fof_reduce_groups(fof->Group, NgroupsExt, sizeof(fof->Group[0]), fof_reduce_group, Comm);

    /* count Groups and number of particles hosted by me */
    fof->Ngroups = 0;
    int64_t Nids = 0;
    for(i = 0; i < NgroupsExt; i ++) {
        if(fof->Group[i].base.MinIDTask != ThisTask) continue;

        fof->Ngroups++;
        Nids += fof->Group[i].base.Length;

        if(fof->Group[i].base.Length != fof->Group[i].Length) {
            /* These two shall be consistent */
            endrun(3333, "Group base Length mismatch with Group Length");
        }
    }

    fof_finish_group_properties(fof, BoxSize);

    int64_t TotNids;
    sumup_large_ints(1, &fof->Ngroups, &fof->TotNgroups);
    MPI_Allreduce(&Nids, &TotNids, 1, MPI_INT64, MPI_SUM, Comm);

    /* report some statistics */
    int largestgroup = 0;
    if(fof->TotNgroups > 0)
    {
        int largestloc = 0;

        for(i = 0; i < NgroupsExt; i++)
            if(fof->Group[i].Length > largestloc)
                largestloc = fof->Group[i].Length;
        MPI_Allreduce(&largestloc, &largestgroup, 1, MPI_INT, MPI_MAX, Comm);
    }

    message(0, "Total number of groups with at least %d particles: %ld\n", fof_params.FOFHaloMinLength, fof->TotNgroups);
    if(fof->TotNgroups > 0)
    {
        message(0, "Largest group has %d particles, mass %g.\n", largestgroup, fof->Group[largestgroup].Mass);
        message(0, "Total number of particles in groups: %012ld\n", TotNids);
    }
}


static void fof_reduce_groups(
    void * groups,
    int nmemb,
    size_t elsize,
    void (*reduce_group)(void * gdst, void * gsrc), MPI_Comm Comm)
{

    int NTask, ThisTask;
    MPI_Comm_size(Comm, &NTask);
    MPI_Comm_rank(Comm, &ThisTask);
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
    int * Send_count = ta_malloc("Send_count", int, NTask);
    int * Recv_count = ta_malloc("Recv_count", int, NTask);

    void * images = NULL;
    void * ghosts = NULL;
    int i;
    int start;

    MPI_Datatype dtype;

    MPI_Type_contiguous(elsize, MPI_BYTE, &dtype);
    MPI_Type_commit(&dtype);

    /*Set global data for the comparison*/
    _fof_compare_Group_MinIDTask_ThisTask = ThisTask;
    /* local groups will be moved to the beginning, we skip them with offset */
    qsort_openmp(groups, nmemb, elsize, fof_compare_Group_MinIDTask);
    /* count how many we have of each task */
    memset(Send_count, 0, sizeof(int) * NTask);

    for(i = 0; i < nmemb; i++) {
        struct BaseGroup * gi = (struct BaseGroup *) (((char*) groups) + i * elsize);
        Send_count[gi->MinIDTask]++;
    }

    /* Skip local groups */
    int Nmine = Send_count[ThisTask];
    Send_count[ThisTask] = 0;

    MPI_Alltoall(Send_count, 1, MPI_INT, Recv_count, 1, MPI_INT, Comm);

    int nimport = 0;
    for(i = 0; i < NTask; i ++) {
        nimport += Recv_count[i];
    }

    images = mymalloc("images", nimport * elsize);
    ghosts = ((char*) groups) + elsize * Nmine;

    MPI_Alltoallv_smart(ghosts, Send_count, NULL, dtype,
                        images, Recv_count, NULL, dtype, Comm);

    for(i = 0; i < nimport; i++) {
        struct BaseGroup * gi = (struct BaseGroup*) ((char*) images + i * elsize);
        gi->OriginalIndex = i;
    }

    /* sort the groups according to MinID */
    qsort_openmp(groups, Nmine, elsize, fof_compare_Group_MinID);
    qsort_openmp(images, nimport, elsize, fof_compare_Group_MinID);

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
    qsort_openmp(images, nimport, elsize, fof_compare_Group_OriginalIndex);

    for(i = 0; i < nimport; i++) {
        struct BaseGroup * gi = (struct BaseGroup*) ((char*) images + i * elsize);
        if(gi->MinIDTask != ThisTask) {
            abort();
        }
    }
    void * ghosts2 = mymalloc("TMP", nmemb * elsize);

    MPI_Alltoallv_smart(images, Recv_count, NULL, dtype,
                        ghosts2, Send_count, NULL, dtype,
                        Comm);
    for(i = 0; i < nmemb - Nmine; i ++) {
        struct BaseGroup * g1 = (struct BaseGroup*) ((char*) ghosts + i * elsize);
        struct BaseGroup * g2 = (struct BaseGroup*) ((char*) ghosts2 + i* elsize);
        if(g1->MinID != g2->MinID) {
            abort();
        }
        if(g1->MinIDTask != g2->MinIDTask) {
            abort();
        }
    }
    memcpy(ghosts, ghosts2, elsize * (nmemb - Nmine));
    myfree(ghosts2);

    myfree(images);

    MPI_Type_free(&dtype);

    /* At this point, each Group entry has the reduced attribute of the full group */
    /* And the local groups (MinIDTask == ThisTask) are placed at the begining of the list*/
    ta_free(Recv_count);
    ta_free(Send_count);
}

static void fof_radix_Group_TotalCountTaskDiffMinID(const void * a, void * radix, void * arg);
static void fof_radix_Group_OriginalTaskMinID(const void * a, void * radix, void * arg);

static void fof_assign_grnr(struct BaseGroup * base, const int NgroupsExt, MPI_Comm Comm)
{
    int i, j, NTask, ThisTask;
    int64_t ngr;
    MPI_Comm_size(Comm, &NTask);
    MPI_Comm_rank(Comm, &ThisTask);

    for(i = 0; i < NgroupsExt; i++)
    {
        base[i].OriginalTask = ThisTask;	/* original task */
    }

    mpsort_mpi(base, NgroupsExt, sizeof(base[0]),
            fof_radix_Group_TotalCountTaskDiffMinID, 24, NULL, Comm);

    /* assign group numbers
     * at this point, both Group are is sorted by length,
     * and the every time OriginalTask == MinIDTask, a list of ghost base is stored.
     * they shall get the same GrNr.
     * */
    ngr = 0;
    for(i = 0; i < NgroupsExt; i++)
    {
        if(base[i].OriginalTask == base[i].MinIDTask) {
            ngr++;
        }
        base[i].GrNr = ngr;
    }

    int64_t * ngra = ta_malloc("NGRA", int64_t, NTask);

    MPI_Allgather(&ngr, 1, MPI_INT64, ngra, 1, MPI_INT64, Comm);

    /* shift to the global grnr. */
    int64_t groffset = 0;
    for(j = 0; j < ThisTask; j++)
        groffset += ngra[j];
    for(i = 0; i < NgroupsExt; i++)
        base[i].GrNr += groffset;

    ta_free(ngra);

    /* bring the group list back into the original task, sorted by MinID */
    mpsort_mpi(base, NgroupsExt, sizeof(base[0]),
            fof_radix_Group_OriginalTaskMinID, 16, NULL, Comm);

    for(i = 0; i < PartManager->NumPart; i++)
        P[i].GrNr = -1;	/* will mark particles that are not in any group */

    int start = 0;
    for(i = 0; i < NgroupsExt; i++)
    {
        for(;start < PartManager->NumPart; start++) {
            if (HaloLabel[start].MinID >= base[i].MinID)
                break;
        }

        for(;start < PartManager->NumPart; start++) {
            if (HaloLabel[start].MinID != base[i].MinID)
                break;
            P[HaloLabel[start].Pindex].GrNr = base[i].GrNr;
        }
    }
}


void
fof_save_groups(FOFGroups * fof, int num, MPI_Comm Comm)
{
    message(0, "start global sorting of group catalogues\n");

    fof_save_particles(fof, num, fof_params.FOFSaveParticles, Comm);

    message(0, "Group catalogues saved.\n");
}

/* FIXME: these shall goto the private member of secondary tree walk */
struct FOFSecondaryPriv {
    float *distance;
    float *hsml;
    int count;
    int npleft;
};

#define FOF_SECONDARY_GET_PRIV(tw) ((struct FOFSecondaryPriv *) (tw->priv))

static void fof_secondary_copy(int place, TreeWalkQueryFOF * I, TreeWalk * tw) {

    I->Hsml = FOF_SECONDARY_GET_PRIV(tw)->hsml[place];
}
static int fof_secondary_haswork(int n, TreeWalk * tw) {
    return (((1 << P[n].Type) & (FOF_SECONDARY_LINK_TYPES)));
}
static void fof_secondary_reduce(int place, TreeWalkResultFOF * O, enum TreeWalkReduceMode mode, TreeWalk * tw) {
    if(O->Distance < FOF_SECONDARY_GET_PRIV(tw)->distance[place])
    {
        FOF_SECONDARY_GET_PRIV(tw)->distance[place] = O->Distance;
        HaloLabel[place].MinID = O->MinID;
        HaloLabel[place].MinIDTask = O->MinIDTask;
    }
}
static void
fof_secondary_ngbiter(TreeWalkQueryFOF * I,
        TreeWalkResultFOF * O,
        TreeWalkNgbIterFOF * iter,
        LocalTreeWalk * lv);

static void
fof_secondary_postprocess(int p, TreeWalk * tw)
{
#pragma omp atomic
    FOF_SECONDARY_GET_PRIV(tw)->count ++;

    if(FOF_SECONDARY_GET_PRIV(tw)->distance[p] > 0.5 * LARGE)
    {
        if(FOF_SECONDARY_GET_PRIV(tw)->hsml[p] < 4 * fof_params.FOFHaloComovingLinkingLength)  /* we only search out to a maximum distance */
        {
            /* need to redo this particle */
#pragma omp atomic
            FOF_SECONDARY_GET_PRIV(tw)->npleft++;
            FOF_SECONDARY_GET_PRIV(tw)->hsml[p] *= 2.0;
/*
            if(iter >= MAXITER - 10)
            {
                endrun(1, "i=%d task=%d ID=%llu Hsml=%g  pos=(%g|%g|%g)\n",
                        p, ThisTask, P[p].ID, FOF_SECONDARY_GET_PRIV(tw)->hsml[p],
                        P[p].Pos[0], P[p].Pos[1], P[p].Pos[2]);
            }
*/
        } else {
            FOF_SECONDARY_GET_PRIV(tw)->distance[p] = 0;  /* we not continue to search for this particle */
        }
    }
}
static void fof_label_secondary(ForceTree * tree)
{
    int n, iter;

    TreeWalk tw[1] = {{0}};
    tw->ev_label = "FOF_FIND_NEAREST";
    tw->visit = (TreeWalkVisitFunction) treewalk_visit_ngbiter;
    tw->ngbiter = (TreeWalkNgbIterFunction) fof_secondary_ngbiter;
    tw->ngbiter_type_elsize = sizeof(TreeWalkNgbIterFOF);
    tw->haswork = fof_secondary_haswork;
    tw->fill = (TreeWalkFillQueryFunction) fof_secondary_copy;
    tw->reduce = (TreeWalkReduceResultFunction) fof_secondary_reduce;
    tw->postprocess = (TreeWalkProcessFunction) fof_secondary_postprocess;
    tw->type = TREEWALK_ALL;
    tw->query_type_elsize = sizeof(TreeWalkQueryFOF);
    tw->result_type_elsize = sizeof(TreeWalkResultFOF);
    tw->tree = tree;
    struct FOFSecondaryPriv priv[1];

    tw->priv = priv;

    message(0, "Start finding nearest dm-particle (presently allocated=%g MB)\n",
            mymalloc_usedbytes() / (1024.0 * 1024.0));

    FOF_SECONDARY_GET_PRIV(tw)->distance = (float *) mymalloc("FOF_SECONDARY->distance", sizeof(float) * PartManager->NumPart);
    FOF_SECONDARY_GET_PRIV(tw)->hsml = (float *) mymalloc("FOF_SECONDARY->hsml", sizeof(float) * PartManager->NumPart);

    for(n = 0; n < PartManager->NumPart; n++)
    {
        if(fof_secondary_haswork(n, tw))
        {
            FOF_SECONDARY_GET_PRIV(tw)->distance[n] = LARGE;
            if(P[n].Type == 0) {
                /* use gas sml as a hint (faster convergence than 0.1 fof_params.FOFHaloComovingLinkingLength at high-z */
                FOF_SECONDARY_GET_PRIV(tw)->hsml[n] = 0.5 * P[n].Hsml;
            } else {
                FOF_SECONDARY_GET_PRIV(tw)->hsml[n] = 0.1 * fof_params.FOFHaloComovingLinkingLength;
            }
        }
    }

    iter = 0;
    int64_t counttot, ntot;

    /* we will repeat the whole thing for those particles where we didn't find enough neighbours */

    message(0, "fof-nearest iteration started\n");

    do
    {
        FOF_SECONDARY_GET_PRIV(tw)->npleft = 0;
        FOF_SECONDARY_GET_PRIV(tw)->count = 0;

        treewalk_run(tw, NULL, PartManager->NumPart);

        sumup_large_ints(1, &FOF_SECONDARY_GET_PRIV(tw)->npleft, &ntot);
        sumup_large_ints(1, &FOF_SECONDARY_GET_PRIV(tw)->count, &counttot);

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
    }
    while(ntot > 0);

    myfree(FOF_SECONDARY_GET_PRIV(tw)->hsml);
    myfree(FOF_SECONDARY_GET_PRIV(tw)->distance);
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
static void fof_seed_make_one(struct Group * g, int ThisTask);

void fof_seed(FOFGroups * fof, MPI_Comm Comm)
{
    int i, j, n, ntot;

    int NTask;
    MPI_Comm_size(Comm, &NTask);
    int * Send_count = ta_malloc("Send_count", int, NTask);
    int * Recv_count = ta_malloc("Recv_count", int, NTask);

    for(n = 0; n < NTask; n++)
        Send_count[n] = 0;

    char * Marked = mymalloc("SeedMark", fof->Ngroups);

    int Nexport = 0;
    for(i = 0; i < fof->Ngroups; i++)
    {
        Marked[i] =
            (fof->Group[i].Mass >= fof_params.MinFoFMassForNewSeed)
        &&  (fof->Group[i].LenType[5] == 0)
        &&  (fof->Group[i].seed_index >= 0);

        if(Marked[i]) Nexport ++;
    }
    struct Group * ExportGroups = mymalloc("Export", sizeof(fof->Group[0]) * Nexport);
    j = 0;
    for(i = 0; i < fof->Ngroups; i ++) {
        if(Marked[i]) {
            ExportGroups[j] = fof->Group[i];
            j++;
        }
    }
    qsort_openmp(ExportGroups, Nexport, sizeof(ExportGroups[0]), cmp_seed_task);

    for(i = 0; i < Nexport; i++) {
        Send_count[ExportGroups[i].seed_task]++;
    }

    MPI_Alltoall(Send_count, 1, MPI_INT, Recv_count, 1, MPI_INT, Comm);

    int Nimport = 0;

    for(j = 0;  j < NTask; j++)
    {
        Nimport += Recv_count[j];
    }

    struct Group * ImportGroups = (struct Group *)
            mymalloc("ImportGroups", Nimport * sizeof(struct Group));

    MPI_Alltoallv_smart(ExportGroups, Send_count, NULL, MPI_TYPE_GROUP,
                        ImportGroups, Recv_count, NULL, MPI_TYPE_GROUP,
                        Comm);

    MPI_Allreduce(&Nimport, &ntot, 1, MPI_INT, MPI_SUM, Comm);

    message(0, "Making %d new black hole particles.\n", ntot);

    int ThisTask;
    MPI_Comm_rank(Comm, &ThisTask);

    for(n = 0; n < Nimport; n++)
    {
        fof_seed_make_one(&ImportGroups[n], ThisTask);
    }

    myfree(ImportGroups);
    myfree(ExportGroups);
    myfree(Marked);

    ta_free(Recv_count);
    ta_free(Send_count);

    walltime_measure("/FOF/Seeding");
}

static void fof_seed_make_one(struct Group * g, int ThisTask) {
   if(g->seed_task != ThisTask) {
        endrun(7771, "Seed does not belong to the right task");
    }
    int index = g->seed_index;
    blackhole_make_one(index);
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
    if(t1 == _fof_compare_Group_MinIDTask_ThisTask) t1 = -1;
    if(t2 == _fof_compare_Group_MinIDTask_ThisTask) t2 = -1;

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
