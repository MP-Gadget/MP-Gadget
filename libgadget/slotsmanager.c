#include <string.h>
#include "slotsmanager.h"
#include "partmanager.h"

#include "utils.h"

struct slots_manager_type SlotsManager[1] = {0};

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

/*Initialise a new slot for the particle at index i.*/
static void
slots_connect_new_slot(int i, size_t size)
{
    /* Fill slot with a meaningless
     * poison value ('e') so we will recognise
     * if it is uninitialised.*/
    memset(BASESLOT(i), 101, size);
    /* book keeping ID: debug only */
    BASESLOT(i)->ID = P[i].ID;
    BASESLOT(i)->IsGarbage = P[i].IsGarbage;
}

int
slots_fork(int parent, int ptype)
{
    /* this will fork a zero mass particle at the given location of parent of the given type.
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
     * Its mass and ptype can be then adjusted. (watchout detached BH /SPH
     * data!)
     * PI will point to a new slot for this type.
     * if the slots runs out, this will trigger a slots growth
     * */

    if(PartManager->NumPart >= PartManager->MaxPart)
    {
        endrun(8888, "Tried to spawn: NumPart=%d MaxPart = %d. Sorry, no space left.\n",
                PartManager->NumPart, PartManager->MaxPart);
    }
    /*This is all racy if ActiveParticle or P is accessed from another thread*/
    int child = atomic_fetch_and_add(&PartManager->NumPart, 1);

    P[parent].Generation ++;
    uint64_t g = P[parent].Generation;
    /* change the child ID according to the generation. */
    P[child] = P[parent];
    P[child].ID = (P[parent].ID & 0x00ffffffffffffffL) + (g << 56L);

    P[child].Mass = 0;
    P[child].Type = ptype;

    if(SLOTS_ENABLED(ptype)) {
        /* if enabled, alloc a new Slot for secondary data */
        int PI = atomic_fetch_and_add(&SlotsManager->info[ptype].size, 1);

        if(PI >= SlotsManager->info[ptype].maxsize) {
            endrun(1, "This is currently unsupported; because SlotsManager.Base can be deep in the heap\n");
            /* there is no way clearly to safely grow the slots during this.
             * Another thread may be accessing the slots; growth will invalidate these indices.
             * making the read atomic will be too expensive I suspect.
             * */
        }

        P[child].PI = PI;

        if(P[child].PI >= SlotsManager->info[ptype].maxsize) {
            /* this shall not happen because we grow automatically in the critical section above! */
            endrun(1, "Assertion Failure more PI than available slots : %d > %d\n",P[child].PI, SlotsManager->info[ptype].maxsize);
        }

        slots_connect_new_slot(child, SlotsManager->info[ptype].elsize);
    }

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


#define GARBAGE(i, ptype) (ptype >= 0 ? BASESLOT_PI(i,ptype)->IsGarbage : P[i].IsGarbage)
#define PART(i, ptype) (ptype >= 0 ? (void *) BASESLOT_PI(i, ptype) : (void *) &P[i])

/*Compaction algorithm*/
static int
slots_gc_compact(int used, int ptype, size_t size)
{
    /*Find first garbage particle*/
    int i, nextgc = used;
    for(i = 0; i < used; i++)
        if(GARBAGE(i,ptype)) {
            nextgc = i;
            break;
        }

    int ngc = 0;
    /*Note each particle is tested exactly once*/
    while(nextgc < used) {
        int i;
        /*Now lastgc contains a garbage*/
        int lastgc = nextgc;
        /*Find a non-garbage after it*/
        int src = used;
        for(i = lastgc + 1; i < used; i++)
            if(!GARBAGE(i, ptype)) {
                src = i;
                break;
            }
        /*If no more non-garbage particles, don't both copying, just add a skip*/
        if(src == used) {
            ngc += src - lastgc;
            break;
        }
        /*Destination is shifted already*/
        int dest = lastgc - ngc;

        nextgc = used;
        /*Find another garbage particle*/
        for(i = src+1; i < used; i++)
            if(GARBAGE(i, ptype)) {
                nextgc = i;
                break;
            }
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
    if(b1->IsGarbage && b2->IsGarbage) {
        return 0;
    }
    if(b1->IsGarbage) return 1;
    if(b2->IsGarbage) return -1;
    return (b1->gc.ReverseLink > b2->gc.ReverseLink) - (b1->gc.ReverseLink < b2->gc.ReverseLink);

}

static int
slots_gc_mark()
{
    int i;
#ifdef DEBUG
    int ptype;
    /*Initially set all reverse links to an obviously invalid value*/
    for(ptype = 0; ptype < 6; ptype++)
    {
        if(!SLOTS_ENABLED(ptype))
            continue;
        #pragma omp parallel for
        for(i = 0; i < SlotsManager->info[ptype].size; i++) {
            BASESLOT_PI(i, ptype)->gc.ReverseLink = PartManager->MaxPart + 100;
        }
    }
#endif

#pragma omp parallel for
    for(i = 0; i < PartManager->NumPart; i++) {
        if(!SLOTS_ENABLED(P[i].Type)) continue;

        BASESLOT(i)->gc.ReverseLink = i;

        /* two consistency checks.*/
#ifdef DEBUG
        if(P[i].IsGarbage && !BASESLOT(i)->IsGarbage) {
            endrun(1, "IsGarbage flag inconsistent between base and secondary: P[%d].Type=%d\n", i, P[i].Type);
        }
        if(!P[i].IsGarbage && BASESLOT(i)->IsGarbage) {
            endrun(1, "IsGarbage flag inconsistent between secondary and base: P[%d].Type=%d\n", i, P[i].Type);
        }
#endif
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
        if(BASESLOT_PI(i, ptype)->IsGarbage) {
            endrun(1, "Shall not happen: i=%d ptype = %d\n", i,ptype);
        }
#endif

        P[BASESLOT_PI(i, ptype)->gc.ReverseLink].PI = i;
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
    slots_check_id_consistency();
#endif

    if(!disabled) {
        slots_gc_mark();

        for(ptype = 0; ptype < 6; ptype++) {
            if(!compact_slots[ptype])
                continue;
            slots_gc_sweep(ptype);
            slots_gc_collect(ptype);
        }
    }
#ifdef DEBUG
    slots_check_id_consistency();
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

    /*Reduce NumPart*/
    while(PartManager->NumPart > 0 && P[PartManager->NumPart-1].IsGarbage) {
        PartManager->NumPart--;
    }

    /*Set up ReverseLink*/
    slots_gc_mark();

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
        while(SlotsManager->info[ptype].size > 0 && BASESLOT_PI(SlotsManager->info[ptype].size-1, ptype)->IsGarbage) {
            SlotsManager->info[ptype].size--;
        }
        slots_gc_collect(ptype);
    }
#ifdef DEBUG
    slots_check_id_consistency();
#endif
}

void
slots_reserve(int where, int atleast[6])
{
    int newMaxSlots[6];
    int ptype;
    int good = 1;

    if(SlotsManager->Base == NULL) {
        SlotsManager->Base = (char*) mymalloc("SlotsBase", 0);
        /* This is so the ptr is never null! Avoid undefined behaviour. */
        for(ptype = 5; ptype >= 0; ptype--) {
            SlotsManager->info[ptype].ptr = SlotsManager->Base;
        }
    }

    /* FIXME: change 0.01 to a parameter. The experience is
     * this works out fine, since the number of time steps increases
     * (hence the number of growth increases
     * when the star formation rate does)*/
    int add = 0.01 * PartManager->MaxPart;
    if (add < 128) add = 128;

    /* FIXME: allow shrinking; need to tweak the memmove later. */
    for(ptype = 0; ptype < 6; ptype ++) {
        newMaxSlots[ptype] = SlotsManager->info[ptype].maxsize;
        if(!SLOTS_ENABLED(ptype)) continue;
        /* if current empty slots is less than half of add, need to grow */
        if (newMaxSlots[ptype] < atleast[ptype] + add / 2) {
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
slots_init()
{
    memset(SlotsManager, 0, sizeof(SlotsManager[0]));

    MPI_Type_contiguous(sizeof(struct particle_data), MPI_BYTE, &MPI_TYPE_PARTICLE);
    MPI_Type_commit(&MPI_TYPE_PARTICLE);
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
slots_free()
{
    myfree(SlotsManager->Base);
}

/* mark the i-th base particle as a garbage. */
void
slots_mark_garbage(int i)
{
    P[i].IsGarbage = 1;
    if(SLOTS_ENABLED(P[i].Type)) {
        BASESLOT(i)->IsGarbage = 1;
    }
}

void
slots_check_id_consistency()
{
    int used[6] = {0};
    int i;

    for(i = 0; i < PartManager->NumPart; i++) {
        if(!SLOTS_ENABLED(P[i].Type)) continue;

        if(P[i].PI >= SlotsManager->info[P[i].Type].size) {
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
    MPI_Barrier(MPI_COMM_WORLD);
}

/* this function needs the Type of P[i] to be setup */
void
slots_setup_topology()
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
        if(!SLOTS_ENABLED(P[i].Type)) continue;
        SlotsManager->info[ptype].size = NLocal[ptype];
    }
}
void
slots_setup_id()
{
    int i;
    /* set up the cross check for child IDs */
/* not bothering making this OMP */
    for(i = 0; i < PartManager->NumPart; i++)
    {
        if(!SLOTS_ENABLED(P[i].Type)) continue;
        BASESLOT(i)->ID = P[i].ID;
        BASESLOT(i)->IsGarbage = P[i].IsGarbage;
    }
}
