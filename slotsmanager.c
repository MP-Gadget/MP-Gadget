#include <string.h>
#include "allvars.h"
#include "event.h"
#include "slotsmanager.h"
#include "mymalloc.h"
#include "system.h"
#include "endrun.h"
#include "openmpsort.h"

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
slots_gc_slots(double defrag_frac);

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

    if(NumPart >= All.MaxPart)
    {
        endrun(8888,
                "On Task=%d with NumPart=%d we try to spawn. Sorry, no space left...(All.MaxPart=%d)\n",
                ThisTask, NumPart, All.MaxPart);
    }
    /*This is all racy if ActiveParticle or P is accessed from another thread*/
    int child = atomic_fetch_and_add(&NumPart, 1);

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

/* remove garbage particles, holes in sph chunk and holes in bh buffer. */
int
slots_gc(double defrag_frac)
{
    /* tree is invalidated if the sequence on P is reordered; */

    int tree_invalid = 0;

    /* TODO: in principle we can track this change and modify the tree nodes;
     * But doing so requires cleaning up the TimeBin link lists, and the tree
     * link lists first. likely worth it, since GC happens only in domain decompose
     * and snapshot IO, both take far more time than rebuilding the tree. */
    tree_invalid |= slots_gc_base();
    tree_invalid |= slots_gc_slots(defrag_frac);

    MPI_Allreduce(MPI_IN_PLACE, &tree_invalid, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    EISlotsAfterGC event = {{tree_invalid}};

    event_emit(&EventSlotsAfterGC, (EIBase*) &event);

    return tree_invalid;
}

static int
slots_gc_base()
{
    int64_t total0, total;

    sumup_large_ints(1, &NumPart, &total0);

    if(SlotsManager->garbage) {
        int i, pc = 0;
        int * partBufcnt = mymalloc("parttmp", (SlotsManager->garbage+1) * sizeof(int));
        for(i = 0; i < NumPart; i++)
            if(P[i].IsGarbage) {
                /* mark the particle for removal in base slot.*/
                partBufcnt[pc++] = i;
            }

        /*Set final elements*/
        partBufcnt[pc] = NumPart;

        /*Compactify the P array: this invalidates the ReverseLink, so
         * that ReverseLink is valid only within gc.*/
        for(i = 0; i < pc; i++) {
            int src = partBufcnt[i]+1;
            int dest = partBufcnt[i] - i;
            int nmove = partBufcnt[i+1] - partBufcnt[i];
            memmove(&P[dest],&P[src],nmove*sizeof(struct particle_data));
        }
        myfree(partBufcnt);
        NumPart -= pc;
        SlotsManager->garbage = 0;
    }

    sumup_large_ints(1, &NumPart, &total);

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
            BASESLOT_PI(i, ptype)->gc.ReverseLink = All.MaxPart + 100;
        }
    }
#endif

#pragma omp parallel for
    for(i = 0; i < NumPart; i++) {
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
    int garbage = SlotsManager->info[ptype].garbage;
    if(!garbage) return 0;

    int used = SlotsManager->info[ptype].size;
    int i, sc = 0;
    /* Allocate memory for the compatification lists: need one extra element.
     * slotBufcnt is a list of BASESLOT indices. Element i is the ith entry to be removed.
     * The final entry points to the last slot.*/
    int * slotBufcnt = mymalloc("slottmp", (garbage+1) * sizeof(int));
    for(i = 0; i < used; i++)
        if(BASESLOT_PI(i,ptype)->IsGarbage)
            slotBufcnt[sc++] = i;
    slotBufcnt[sc] = used;
    for(i = 0; i < sc; i++) {
        int src = slotBufcnt[i]+1;
        int dest = slotBufcnt[i] - i;
        int nmove = slotBufcnt[i+1] - slotBufcnt[i]-1;
//             message(1,"ptype = %d i = %d, PI = %d-> %d, nm=%d\n",ptype, i, src, dest, nmove);
        memmove(BASESLOT_PI(dest, ptype), BASESLOT_PI(src, ptype), nmove*SlotsManager->info[ptype].elsize);
    }
    SlotsManager->info[ptype].size -= sc;
    SlotsManager->info[ptype].garbage = 0;

    myfree(slotBufcnt);
    return 0;
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
            endrun(1, "Shall not happend\n");
        }
#endif

        P[BASESLOT_PI(i, ptype)->gc.ReverseLink].PI = i;
    }
}


static int
slots_gc_slots(double defrag_frac)
{
    int ptype;

    int64_t total0[6];
    int64_t total1[6];

    int disabled = 1;
    for(ptype = 0; ptype < 6; ptype ++) {
        sumup_large_ints(1, &SlotsManager->info[ptype].size, &total0[ptype]);
    }

#ifdef DEBUG
    slots_check_id_consistency();
#endif
    /*Disable gc if there is insufficient garbage (or no slots are used)*/
    for(ptype = 0; ptype < 6; ptype ++) {
        int defrag = 1 + defrag_frac * SlotsManager->info[ptype].size;
        if(SlotsManager->info[ptype].garbage >= defrag)
            disabled = 0;
    }

    if(!disabled) {
        slots_gc_mark();

        for(ptype = 0; ptype < 6; ptype++) {
            int defrag = 1 + defrag_frac * SlotsManager->info[ptype].size;
            if(SlotsManager->info[ptype].garbage < defrag)
                continue;
            slots_gc_sweep(ptype);
            slots_gc_collect(ptype);
        }
#ifdef DEBUG
        slots_check_id_consistency();
#endif
    }

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

/* Sort the particles and their slots by type and peano order.*/
void
slots_gc_sorted()
{
    int ptype;
    /* Resort the particles such that those of the same type and key are close by.
     * The locality is broken by the exchange. */
    qsort_openmp(P, NumPart, sizeof(struct particle_data), order_by_type_and_key);

    /*Reduce NumPart*/
    while(P[NumPart-1].IsGarbage) {
        NumPart--;
    }
    SlotsManager->garbage = 0;

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
        while(BASESLOT_PI(SlotsManager->info[ptype].size-1, ptype)->IsGarbage) {
            SlotsManager->info[ptype].size--;
        }
        SlotsManager->info[ptype].garbage = 0;
        slots_gc_collect(ptype);
    }
#ifdef DEBUG
    slots_check_id_consistency();
#endif

    /*Rebuild the tree if still allocated*/
    EISlotsAfterGC event = {{1}};
    event_emit(&EventSlotsAfterGC, (EIBase*) &event);
}

void
slots_reserve(int atleast[6])
{
    int newMaxSlots[6];
    int ptype;
    int good = 1;

    if(SlotsManager->Base == NULL)
        SlotsManager->Base = (char*) mymalloc("SlotsBase", 0);

    /* FIXME: change 0.01 to a parameter. The experience is
     * this works out fine, since the number of time steps increases
     * (hence the number of growth increases
     * when the star formation ra*/
    int add = 0.01 * All.MaxPart;
    if (add < 128) add = 128;

    /* FIXME: allow shrinking; need to tweak the memmove later. */
    for(ptype = 0; ptype < 6; ptype ++) {
        newMaxSlots[ptype] = SlotsManager->info[ptype].maxsize;
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

    message(1, "Allocated %g MB for %d sph, %d stars and %d BHs.\n", total_bytes / (1024.0 * 1024.0),
            newMaxSlots[0], newMaxSlots[4], newMaxSlots[5]);

    /* move the last block first since we are only increasing sizes, moving items forward */
    for(ptype = 5; ptype >= 0; ptype--) {
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

void slots_init()
{
    int ptype;
    memset(SlotsManager, 0, sizeof(SlotsManager[0]));

    SlotsManager->info[0].elsize = sizeof(struct sph_particle_data);
    SlotsManager->info[0].enabled = 1;
    SlotsManager->info[4].elsize = sizeof(struct star_particle_data);
    SlotsManager->info[4].enabled = 1;
    SlotsManager->info[5].elsize = sizeof(struct bh_particle_data);
    SlotsManager->info[5].enabled = 1;

    MPI_Type_contiguous(sizeof(struct particle_data), MPI_BYTE, &MPI_TYPE_PARTICLE);
    MPI_Type_commit(&MPI_TYPE_PARTICLE);

    for(ptype = 0; ptype < 6; ptype++) {
        if(!SLOTS_ENABLED(ptype)) continue;

        MPI_Type_contiguous(SlotsManager->info[ptype].elsize, MPI_BYTE, &MPI_TYPE_SLOT[ptype]);
        MPI_Type_commit(&MPI_TYPE_SLOT[ptype]);
    }
}

void slots_free()
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
        atomic_add_and_fetch(&SlotsManager->info[P[i].Type].garbage, 1);
    }
    /*Increment the counter*/
    atomic_add_and_fetch(&SlotsManager->garbage, 1);
}

void
slots_check_id_consistency()
{
    int used[6] = {0};
    int i;

    for(i = 0; i < NumPart; i++) {
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
    for(i = 0; i < NumPart; i ++) {
        int ptype = P[i].Type;
        /* atomic fetch add */
        P[i].PI = NLocal[ptype];
        NLocal[ptype] ++;
    }

    int ptype;
    for(ptype = 0; ptype < 6; ptype ++) {
        SlotsManager->info[ptype].size = NLocal[ptype];
    }
}
void
slots_setup_id()
{
    int i;
    /* set up the cross check for child IDs */
/* not bothering making this OMP */
    for(i = 0; i < NumPart; i++)
    {
        if(!SLOTS_ENABLED(P[i].Type)) continue;
        BASESLOT(i)->ID = P[i].ID;
        BASESLOT(i)->IsGarbage = P[i].IsGarbage;
    }
}
