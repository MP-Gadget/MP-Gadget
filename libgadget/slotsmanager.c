#include <string.h>
#include "slotsmanager.h"
#include "partmanager.h"

#include "utils.h"

struct slots_manager_type SlotsManager[1];

#define SLOTS_ENABLED(ptype) (SlotsManager->info[ptype].enabled)

MPI_Datatype MPI_TYPE_PARTICLE = 0;
MPI_Datatype MPI_TYPE_PLAN_ENTRY = 0;
MPI_Datatype MPI_TYPE_SLOT[6] = {0};

static struct sph_particle_data * GDB_SphP;
static struct star_particle_data * GDB_StarP;
static struct bh_particle_data * GDB_BhP;

static int
slots_gc_base();

static int
slots_gc_slots(int * compact_slots);

/* Initialise a new slot with type at index pi
 * for the particle at index i.
 * This will modify both P[i] and the slot at pi in type.*/
static void
slots_connect_new_slot(int i, int pi, int type)
{
    /* Fill slot with a meaningless
     * poison value ('e') so we will recognise
     * if it is uninitialised.*/
    memset(BASESLOT_PI(pi, type), 101, SlotsManager->info[type].elsize);
    /* book keeping ID: debug only */
    BASESLOT_PI(pi, type)->ID = P[i].ID;
    /*Update the particle's pointer*/
    P[i].PI = pi;
}

/* This will change a particle type. The original particle_data structure is preserved,
 * but the old slot is made garbage and a new one (with the new type) is created.
 * No data is copied, but the slot is created.
 *
 * Assumes the particle is protected by locks in threaded env.
 *
 * Note that the 'new particle' event is not emitted, as there is no new particle!
 * If you do something on a new slot, this needs a new event.
 *
 * Arguments:
 * parent - particle whose type is changing.
 * ptype - type to change it to
 * placement - if this is not -1, we use a specific numbered slot.
 *             if this is -1, get a new slot atomically from the pre-allocated heap.
 * discardold - if this is true, the pre-conversion slot will be marked as garbage.
 *              If you are really converting then this should be true.
 *              If you are using this function on the output of slots_split_particle, it should be false.
 * */
int
slots_convert(int parent, int ptype, int placement)
{
    /*Explicitly mark old slot as garbage*/
    if(P[parent].PI >= 0 && SLOTS_ENABLED(P[parent].Type))
        BASESLOT_PI(P[parent].PI, P[parent].Type)->ReverseLink = PartManager->MaxPart + 100;

    /*Make a new slot*/
    if(SLOTS_ENABLED(ptype)) {
        int PI = placement;
        /* if enabled, alloc a new Slot for secondary data */
        if(placement < 0)
            PI = atomic_fetch_and_add(&SlotsManager->info[ptype].size, 1);

        /* There is no way clearly to safely grow the slots during this, because the memory may be deep in the heap.*/
        if(PI >= SlotsManager->info[ptype].maxsize) {
            endrun(1, "Tried to use non-allocated slot %d (> %d)\n", PI, SlotsManager->info[ptype].maxsize);
        }
        slots_connect_new_slot(parent, PI, ptype);
    }
    /*Type changed after slot updated*/
    P[parent].Type = ptype;
    return parent;
}

/* This will split a new particle out from an existing one, conserving mass.
 * The type is the same and the slot PI on the new particle is set to -1.
 * You should call slots_convert on the child afterwards to create a new slot.
 *
 * Assumes the particle is protected by locks in threaded env.
 *
 * The Generation of parent is incremented.
 * The child carries the incremented generation number.
 * The ID of the child is modified, with the new generation number set
 * at the highest 8 bits.
 *
 * the new particle's index is returned.
 *
 * Its mass and ptype can be then adjusted using slots_convert.
 * The 'new particle' event is emitted.
 * */
int
slots_split_particle(int parent, double childmass)
{
    int child = atomic_fetch_and_add(&PartManager->NumPart, 1);

    if(child >= PartManager->MaxPart)
        endrun(8888, "Tried to spawn: NumPart=%d MaxPart = %d. Sorry, no space left.\n", child, PartManager->MaxPart);

    P[parent].Generation ++;
    uint64_t g = P[parent].Generation;
    P[child] = P[parent];

    /* change the child ID according to the generation. */
    P[child].ID = (P[parent].ID & 0x00ffffffffffffffL) + (g << 56L);
    if(g >= (1 << (64-56L)))
        endrun(1, "Particle %d (ID: %ld) generated too many particles: generation %d wrapped.\n", parent, P[parent].ID, g);

    P[child].Mass = childmass;
    P[parent].Mass -= childmass;

    /*Invalidate the slot of the child. Call slots_convert soon afterwards!*/
    P[child].PI = -1;

    /*! When a new additional star particle is created, we can put it into the
     *  tree at the position of the spawning gas particle. This is possible
     *  because the Nextnode[] array essentially describes the full tree walk as a
     *  link list. Multipole moments of tree nodes need not be changed.
     */
    /* emit event for forcetree to deal with the new particle */
    EISlotsFork event = {
        .parent = parent,
        .child = child,
    };

    event_emit(&EventSlotsFork, (EIBase *) &event);

    return child;
}

/* remove garbage particles, holes in sph chunk and holes in bh buffer.
 * This algorithm is O(n), and shifts particles over the holes.
 * compact_slots is a 6-member array, 1 if that slot should be compacted, 0 otherwise.
 * As slots_gc_base preserves the order of the slots, one may usually skip compaction.*/
int
slots_gc(int * compact_slots)
{
    /* tree is invalidated if the sequence on P is reordered; */

    int tree_invalid = 0;

    /* TODO: in principle we can track this change and modify the tree nodes;
     * But doing so requires cleaning up the TimeBin link lists, and the tree
     * link lists first. likely worth it, since GC happens only in domain decompose
     * and snapshot IO, both take far more time than rebuilding the tree. */
    tree_invalid |= slots_gc_base();
    tree_invalid |= slots_gc_slots(compact_slots);

    MPI_Allreduce(MPI_IN_PLACE, &tree_invalid, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    return tree_invalid;
}


#define GARBAGE(i, ptype) (ptype >= 0 ? BASESLOT_PI(i,ptype)->ReverseLink > PartManager->MaxPart : P[i].IsGarbage)
#define PART(i, ptype) (ptype >= 0 ? (void *) BASESLOT_PI(i, ptype) : (void *) &P[i])

/*Find the next garbage particle*/
static int
slots_find_next_garbage(int start, int used, int ptype)
{
    int i, nextgc = used;
    /*Find another garbage particle*/
    for(i = start; i < used; i++)
        if(GARBAGE(i, ptype)) {
            nextgc = i;
            break;
        }
    return nextgc;
}

/*Find the next non-garbage particle*/
static int
slots_find_next_nongarbage(int start, int used, int ptype)
{
    int i, nextgc = used;
    /*Find another garbage particle*/
    for(i = start; i < used; i++)
        if(!GARBAGE(i, ptype)) {
            nextgc = i;
            break;
        }
    return nextgc;
}

/*Compaction algorithm*/
static int
slots_gc_compact(int used, int ptype, size_t size)
{
    /*Find first garbage particle: can't use bisection here as not sorted.*/
    int nextgc = slots_find_next_garbage(0, used, ptype);

    int ngc = 0;
    /*Note each particle is tested exactly once*/
    while(nextgc < used) {
        /*Now lastgc contains a garbage*/
        int lastgc = nextgc;
        /*Find a non-garbage after it*/
        int src = slots_find_next_nongarbage(lastgc+1, used, ptype);
        /*If no more non-garbage particles, don't bother copying, just add a skip*/
        if(src == used) {
            ngc += src - lastgc;
            break;
        }
        /*Destination is shifted already*/
        int dest = lastgc - ngc;

        /*Find another garbage particle*/
        nextgc = slots_find_next_garbage(src + 1, used, ptype);

        /*Add number of particles we skipped*/
        ngc += src - lastgc;
        int nmove = nextgc - src +1;
//         message(1,"i = %d, PI = %d-> %d, nm=%d\n",i, src, dest, nmove);
        memmove(PART(dest, ptype),PART(src, ptype),nmove*size);
    }
    if(ngc > used)
        endrun(1, "ngc = %d > used = %d!\n", ngc, used);
    return ngc;
}

static int
slots_gc_base()
{
    int64_t total0, total;

    sumup_large_ints(1, &PartManager->NumPart, &total0);

    /*Compactify the P array: this invalidates the ReverseLink, so
        * that ReverseLink is valid only within gc.*/
    int ngc = slots_gc_compact(PartManager->NumPart, -1, sizeof(struct particle_data));

    PartManager->NumPart -= ngc;

    sumup_large_ints(1, &PartManager->NumPart, &total);

    if(total != total0) {
        message(0, "GC : Reducing Particle slots from %ld to %ld\n", total0, total);
        return 1;
    }
    return 0;
}

static int slot_cmp_reverse_link(const void * b1in, const void * b2in) {
    const struct particle_data_ext * b1 = (struct particle_data_ext *) b1in;
    const struct particle_data_ext * b2 = (struct particle_data_ext *) b2in;
    return (b1->ReverseLink > b2->ReverseLink) - (b1->ReverseLink < b2->ReverseLink);
}

static int
slots_gc_mark(const struct slots_manager_type * SlotsManager)
{
    int i;
    if(!(SlotsManager->info[0].enabled ||
       SlotsManager->info[1].enabled ||
       SlotsManager->info[2].enabled ||
       SlotsManager->info[3].enabled ||
       SlotsManager->info[4].enabled ||
       SlotsManager->info[5].enabled))
        return 0;

#ifdef DEBUG
    int ptype;
    /*Initially set all reverse links to an obviously invalid value*/
    for(ptype = 0; ptype < 6; ptype++)
    {
        struct slot_info info = SlotsManager->info[ptype];
        if(!info.enabled)
            continue;
        #pragma omp parallel for
        for(i = 0; i < info.size; i++) {
            struct particle_data_ext * sdata = (struct particle_data_ext * )(info.ptr + info.elsize * i);
            sdata->ReverseLink = PartManager->MaxPart + 100;
        }
    }
#endif

#pragma omp parallel for
    for(i = 0; i < PartManager->NumPart; i++) {
        struct slot_info info = SlotsManager->info[P[i].Type];
        if(!info.enabled)
            continue;
        int sind = P[i].PI;
        if(sind >= info.size || sind < 0)
            endrun(1, "Particle %d, type %d has PI index %d beyond max slot size %d.\n", i, P[i].Type, sind, info.size);
        struct particle_data_ext * sdata = (struct particle_data_ext * )(info.ptr + info.elsize * sind);
        sdata->ReverseLink = i;
        /* Make the PI of garbage particles invalid*/
        if(P[i].IsGarbage)
            sdata->ReverseLink = PartManager->MaxPart + 100;
    }
    return 0;
}

/* sweep removes unused entries in the slot list. */
static int
slots_gc_sweep(int ptype)
{
    if(!SLOTS_ENABLED(ptype)) return 0;
    int used = SlotsManager->info[ptype].size;

    int ngc = slots_gc_compact(used, ptype, SlotsManager->info[ptype].elsize);

    SlotsManager->info[ptype].size -= ngc;

    return ngc;
}

/* update new pointers. */
static void
slots_gc_collect(int ptype)
{
    int i;
    if(!SLOTS_ENABLED(ptype)) return;

    /* Now update the link in BhP */
#pragma omp parallel for
    for(i = 0;
        i < SlotsManager->info[ptype].size;
        i ++) {

#ifdef DEBUG
        if(BASESLOT_PI(i, ptype)->ReverseLink >= PartManager->MaxPart) {
            endrun(1, "Shall not happen: i=%d ptype = %d\n", i,ptype);
        }
#endif

        P[BASESLOT_PI(i, ptype)->ReverseLink].PI = i;
    }
}


static int
slots_gc_slots(int * compact_slots)
{
    int ptype;

    int64_t total0[6];
    int64_t total1[6];

    int disabled = 1;
    for(ptype = 0; ptype < 6; ptype ++) {
        sumup_large_ints(1, &SlotsManager->info[ptype].size, &total0[ptype]);
        if(compact_slots[ptype])
            disabled = 0;
    }

#ifdef DEBUG
    slots_check_id_consistency(SlotsManager);
#endif

    if(!disabled) {
        slots_gc_mark(SlotsManager);

        for(ptype = 0; ptype < 6; ptype++) {
            if(!compact_slots[ptype])
                continue;
            slots_gc_sweep(ptype);
            slots_gc_collect(ptype);
        }
    }
#ifdef DEBUG
    slots_check_id_consistency(SlotsManager);
#endif
    for(ptype = 0; ptype < 6; ptype ++) {
        sumup_large_ints(1, &SlotsManager->info[ptype].size, &total1[ptype]);

        if(total1[ptype] != total0[ptype])
            message(0, "GC: Reducing number of slots for %d from %ld to %ld\n", ptype, total0[ptype], total1[ptype]);
    }

    /* slot gc never invalidates the tree */
    return 0;
}

static int
order_by_type_and_key(const void *a, const void *b)
{
    const struct particle_data * pa  = (const struct particle_data *) a;
    const struct particle_data * pb  = (const struct particle_data *) b;

    if(pa->IsGarbage && !pb->IsGarbage)
        return +1;
    if(!pa->IsGarbage && pb->IsGarbage)
        return -1;
    if(pa->Type < pb->Type)
        return -1;
    if(pa->Type > pb->Type)
        return +1;
    if(pa->Key < pb->Key)
        return -1;
    if(pa->Key > pb->Key)
        return +1;

    return 0;
}

/*Returns the number of non-Garbage particles in an array with garbage sorted to the end.
 * The index returned always points to a garbage particle.
 * If ptype < 0, find the last garbage particle in the P array.
 * If ptype >= 0, find the last garbage particle in the slot associated with ptype. */
int slots_get_last_garbage(int nfirst, int nlast, int ptype)
{
    /* nfirst is always not garbage, nlast is always garbage*/
    if(GARBAGE(nfirst, ptype))
        return nfirst;
    if(!GARBAGE(nlast, ptype))
        return nlast+1;
    /*Bisection*/
    do {
        int nmid = (nfirst + nlast)/2;
        if(GARBAGE(nmid, ptype))
            nlast = nmid;
        else
            nfirst = nmid;
    }
    while(nlast - nfirst > 1);

    return nlast;
}

/* Sort the particles and their slots by type and peano order.
 * This does a gc by sorting the Garbage to the end of the array and then trimming.
 * It is a different algorithm to slots_gc, somewhat slower,
 * but delivers a spatially compact sort. It always compacts the slots*/
void
slots_gc_sorted()
{
    int ptype;
    /* Resort the particles such that those of the same type and key are close by.
     * The locality is broken by the exchange. */
    qsort_openmp(P, PartManager->NumPart, sizeof(struct particle_data), order_by_type_and_key);

    /*Remove garbage particles*/
    PartManager->NumPart = slots_get_last_garbage(0, PartManager->NumPart -1 , -1);

    /*Set up ReverseLink*/
    slots_gc_mark(SlotsManager);

    for(ptype = 0; ptype < 6; ptype++) {
        if(!SLOTS_ENABLED(ptype))
            continue;
        /* sort the used ones
         * by their location in the P array */
        qsort_openmp(SlotsManager->info[ptype].ptr,
                 SlotsManager->info[ptype].size,
                 SlotsManager->info[ptype].elsize,
                 slot_cmp_reverse_link);

        /*Reduce slots used*/
        SlotsManager->info[ptype].size = slots_get_last_garbage(0, SlotsManager->info[ptype].size-1, ptype);
        slots_gc_collect(ptype);
    }
#ifdef DEBUG
    slots_check_id_consistency(SlotsManager);
#endif
}

void
slots_reserve(int where, int atleast[6])
{
    int newMaxSlots[6];
    int ptype;
    int good = 1;

    if(SlotsManager->Base == NULL) {
        SlotsManager->Base = (char*) mymalloc("SlotsBase", sizeof(struct sph_particle_data));
        /* This is so the ptr is never null! Avoid undefined behaviour. */
        for(ptype = 5; ptype >= 0; ptype--) {
            SlotsManager->info[ptype].ptr = SlotsManager->Base;
        }
    }

    int add = SlotsManager->increase * PartManager->MaxPart;
    if (add < 128) add = 128;

    /* FIXME: allow shrinking; need to tweak the memmove later. */
    for(ptype = 0; ptype < 6; ptype ++) {
        newMaxSlots[ptype] = SlotsManager->info[ptype].maxsize;
        if(!SLOTS_ENABLED(ptype)) continue;
        /* if current empty slots is less than half of add, need to grow */
        if (newMaxSlots[ptype] <= atleast[ptype] + add / 2) {
            newMaxSlots[ptype] = atleast[ptype] + add;
            good = 0;
        }
    }
    /* no need to grow, already have enough */
    if (good) {
        return;
    }

    size_t total_bytes = 0;
    size_t offsets[6];
    size_t bytes[6] = {0};

    for(ptype = 0; ptype < 6; ptype++) {
        offsets[ptype] = total_bytes;
        bytes[ptype] = SlotsManager->info[ptype].elsize * newMaxSlots[ptype];
        total_bytes += bytes[ptype];
    }
    char * newSlotsBase = myrealloc(SlotsManager->Base, total_bytes);

    /* If we are using VALGRIND the allocator is system malloc, and so realloc may move the base pointer.
     * Thus we need to also move the slots pointers before doing the memmove. If we are using our own
     * memory allocator the base address never moves, so this is unnecessary (but we do it anyway).*/
    for(ptype = 0; ptype < 6; ptype++) {
        SlotsManager->info[ptype].ptr = SlotsManager->info[ptype].ptr - SlotsManager->Base + newSlotsBase;
    }

    message(where, "SLOTS: Reserved %g MB for %d sph, %d stars and %d BHs (disabled: %d %d %d)\n", total_bytes / (1024.0 * 1024.0),
            newMaxSlots[0], newMaxSlots[4], newMaxSlots[5], newMaxSlots[1], newMaxSlots[2], newMaxSlots[3]);

    /* move the last block first since we are only increasing sizes, moving items forward.
     * No need to move the 0 block, since it is already moved to newSlotsBase in realloc.*/
    for(ptype = 5; ptype > 0; ptype--) {
        if(!SLOTS_ENABLED(ptype)) continue;
        memmove(newSlotsBase + offsets[ptype],
            SlotsManager->info[ptype].ptr,
            SlotsManager->info[ptype].elsize * SlotsManager->info[ptype].size);
    }

    SlotsManager->Base = newSlotsBase;

    for(ptype = 0; ptype < 6; ptype++) {
        SlotsManager->info[ptype].ptr = newSlotsBase + offsets[ptype];
        SlotsManager->info[ptype].maxsize = newMaxSlots[ptype];
    }
    GDB_SphP = (struct sph_particle_data *) SlotsManager->info[0].ptr;
    GDB_StarP = (struct star_particle_data *) SlotsManager->info[4].ptr;
    GDB_BhP = (struct bh_particle_data *) SlotsManager->info[5].ptr;
}

void
slots_init(double increase)
{
    memset(SlotsManager, 0, sizeof(SlotsManager[0]));

    MPI_Type_contiguous(sizeof(struct particle_data), MPI_BYTE, &MPI_TYPE_PARTICLE);
    MPI_Type_commit(&MPI_TYPE_PARTICLE);
    SlotsManager->increase = increase;
}

void
slots_set_enabled(int ptype, size_t elsize)
{
    SlotsManager->info[ptype].enabled = 1;
    SlotsManager->info[ptype].elsize = elsize;
    MPI_Type_contiguous(SlotsManager->info[ptype].elsize, MPI_BYTE, &MPI_TYPE_SLOT[ptype]);
    MPI_Type_commit(&MPI_TYPE_SLOT[ptype]);
}


void
slots_free(struct slots_manager_type * SlotsManager)
{
    myfree(SlotsManager->Base);
}

/* mark the i-th base particle as a garbage. */
void
slots_mark_garbage(int i)
{
    P[i].IsGarbage = 1;
    if(SLOTS_ENABLED(P[i].Type)) {
        BASESLOT(i)->ReverseLink = PartManager->MaxPart + 100;
    }
}

void
slots_check_id_consistency(struct slots_manager_type * SlotsManager)
{
    int used[6] = {0};
    int i;

    for(i = 0; i < PartManager->NumPart; i++) {
        struct slot_info info = SlotsManager->info[P[i].Type];
        if(!info.enabled)
            continue;

        if(P[i].PI >= info.size) {
            endrun(1, "slot PI consistency failed2\n");
        }
        if(BASESLOT(i)->ID != P[i].ID) {
            endrun(1, "slot id consistency failed2: i=%d P.ID = %ld SLOT.ID=%ld\n",i, P[i].ID, BASESLOT(i)->ID);
        }
        used[P[i].Type] ++;
    }
    int64_t NTotal[6];

    sumup_large_ints(6, used, NTotal);
    int ptype;
    for(ptype = 0; ptype < 6; ptype ++) {
        if(NTotal[ptype] > 0) {
            /* Watch out: we print per rank here, but the condition must be global*/
            message(0, "Task 0: GC: Used slots for type %d is %ld\n", ptype, used[ptype]);
        }
    }
}

/* this function needs the Type of P[i] to be setup */
void
slots_setup_topology(struct slots_manager_type * SlotsManager)
{
    int NLocal[6] = {0};

    int i;
/* not bothering making this OMP */
    for(i = 0; i < PartManager->NumPart; i ++) {
        int ptype = P[i].Type;
        /* atomic fetch add */
        P[i].PI = NLocal[ptype];
        NLocal[ptype] ++;
    }

    int ptype;
    for(ptype = 0; ptype < 6; ptype ++) {
        struct slot_info info = SlotsManager->info[P[i].Type];
        if(!info.enabled)
            continue;
        SlotsManager->info[ptype].size = NLocal[ptype];
    }
}
void
slots_setup_id(const struct slots_manager_type * SlotsManager)
{
    int i;
    /* set up the cross check for child IDs */
    /* not bothering making this OMP */
    for(i = 0; i < PartManager->NumPart; i++)
    {
        struct slot_info info = SlotsManager->info[P[i].Type];
        if(!info.enabled)
            continue;

        int sind = P[i].PI;
        if(sind >= info.size || sind < 0)
            endrun(1, "Particle %d, type %d has PI index %d beyond max slot size %d.\n", i, P[i].Type, sind, info.size);
        struct particle_data_ext * sdata = (struct particle_data_ext * )(info.ptr + info.elsize * sind);
        sdata->ReverseLink = i;
        sdata->ID = P[i].ID;
        if(P[i].IsGarbage)
            sdata->ReverseLink = PartManager->MaxPart + 100;
    }
}

/*Small structure of pointers*/
static struct sph_scratch_data SphScratch;

void
slots_allocate_sph_scratch_data(int sph_grad_rho, int nsph)
{
    /*Data is allocated high so that we can free the tree around it*/
    if(sph_grad_rho)
        SphScratch.GradRho = mymalloc2("SPH_GradRho", sizeof(MyFloat) * 3 * nsph);
    else
        SphScratch.GradRho = NULL;
    /* Allocated in black hole, freed in sfr.*/
    SphScratch.Injected_BH_Energy = NULL;
    SphScratch.EntVarPred = mymalloc2("EntVarPred", sizeof(MyFloat) * nsph);
    SphScratch.VelPred = mymalloc2("VelPred", sizeof(MyFloat) * 3 * nsph);
    SlotsManager->info[0].scratchdata = (char *) &SphScratch;
}

void
slots_free_sph_scratch_data(struct sph_scratch_data * SphScratch)
{
    myfree(SphScratch->VelPred);
    SphScratch->VelPred = NULL;
    myfree(SphScratch->EntVarPred);
    SphScratch->EntVarPred = NULL;
    if(SphScratch->GradRho) {
        myfree(SphScratch->GradRho);
        SphScratch->GradRho = NULL;
    }
}
