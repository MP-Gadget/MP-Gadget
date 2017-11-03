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
slots_gc_slots();

EventSpec EventSlotsFork = {"SlotsFork", 0};
EventSpec EventSlotsAfterGC = {"SlotsAfterGC", 0};

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

        /* book keeping ID FIXME: debug only */
        BASESLOT(child)->ID = P[child].ID;
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
slots_gc(void)
{
    /* tree is invalidated if the sequence on P is reordered; */

    int tree_invalid = 0;

    /* TODO: in principle we can track this change and modify the tree nodes;
     * But doing so requires cleaning up the TimeBin link lists, and the tree
     * link lists first. likely worth it, since GC happens only in domain decompose
     * and snapshot IO, both take far more time than rebuilding the tree. */
    tree_invalid |= slots_gc_base();
    tree_invalid |= slots_gc_slots();

    MPI_Allreduce(MPI_IN_PLACE, &tree_invalid, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    EISlotsAfterGC event = {0};

    event_emit(&EventSlotsAfterGC, (EIBase*) &event);

    return tree_invalid;
}

static int
slots_gc_base()
{
    int i, tree_invalid = 0; 
    int count_elim;
    int64_t total0, total;

    sumup_large_ints(1, &NumPart, &total0);

    count_elim = 0;

    for(i = 0; i < NumPart; i++)
        if(P[i].IsGarbage)
        {
            P[i] = P[NumPart - 1];

            NumPart--;
            i--;

            count_elim++;
        }

    sumup_large_ints(1, &NumPart, &total);

    if(total != total0) {
        message(0, "GC : Reducing Particle slots from %ld to %ld\n", total0, total);
        tree_invalid = 1;
    }
    return tree_invalid;
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

#pragma omp parallel for
    for(i = 0; i < NumPart; i++) {
        if(!SLOTS_ENABLED(P[i].Type)) continue;

        BASESLOT(i)->gc.ReverseLink = i;

        /* two consistency checks.*/
#ifdef DEBUG
        if(P[i].IsGarbage && !BASESLOT(i)->IsGarbage) {
            endrun(1, "IsGarbage flag inconsistent between base and secondary\n");
        }
        if(!P[i].IsGarbage && BASESLOT(i)->IsGarbage) {
            endrun(1, "IsGarbage flag inconsistent between base and secondary\n");
        }
#endif
    }
}

/* sweep removed unused elements. */
static int
slots_gc_sweep(int ptype)
{
    if(!SLOTS_ENABLED(ptype)) return 0;

    int used = SlotsManager->info[ptype].size;
    size_t elsize = SlotsManager->info[ptype].elsize;
    int i = 0;
    /* put unused guys to the end */
    while(i < used)
    {
        while(i < used
                &&
        BASESLOT_PI(i, ptype)->IsGarbage) {

            memcpy(BASESLOT_PI(i, ptype),
                BASESLOT_PI(used - 1, ptype), elsize);
            used -- ;
        }
        i ++;
    }

    SlotsManager->info[ptype].size = used;
    return 0;
}

/* defrags ensures locality. */
static void
slots_gc_defrag(int ptype)
{

    if(!SLOTS_ENABLED(ptype)) return;
    /* measure the fragmentation */
    int i;
    int frag = 0;
    for(i = 1;
        i < SlotsManager->info[ptype].size;
        i ++) {

        if( BASESLOT_PI(i, ptype)->gc.ReverseLink <
            BASESLOT_PI(i - 1, ptype)->gc.ReverseLink
            ) {
            frag++;
        }
    }
    int defrag = MPIU_Any(
            frag > SlotsManager->info[ptype].size * 0.1,
            MPI_COMM_WORLD);

    /* if any rank is too fragmented, add a sorting to defrag. */
    if(defrag)  {
        /* sort the used ones
         * by their location in the P array */
        qsort_openmp(SlotsManager->info[ptype].ptr,
                     SlotsManager->info[ptype].size,
                     SlotsManager->info[ptype].elsize, 
                     slot_cmp_reverse_link);
    }
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
        BASESLOT_PI(i, ptype)->IsGarbage = 0;
    }
}

static int
slots_gc_slots()
{
    int ptype;

    int64_t total0[6];
    int64_t total1[6];

    int disabled = 1;
    for(ptype = 0; ptype < 6; ptype ++) {
        sumup_large_ints(1, &SlotsManager->info[ptype].size, &total0[ptype]);
        if(total0[ptype] != 0) disabled = 0;
    }
    /* disabled this if no slots are used */
    if(disabled) {
        return 0;
    }

#ifdef DEBUG
    slots_check_id_consistency();
#endif

    slots_gc_mark();

    for(ptype = 0; ptype < 6; ptype++) {
        slots_gc_sweep(ptype);
        slots_gc_defrag(ptype);
        slots_gc_collect(ptype);
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


void
slots_reserve(int atleast[6])
{
    int newMaxSlots[6];
    int ptype;
    int good = 1;

    if(SlotsManager->Base == NULL)
        SlotsManager->Base = (char*) mymalloc("SlotsBase", 0);

    /* FIXME: change 0.005 to a parameter. The expericence is 
     * this works out fine, since the number of time steps increases
     * (hence the number of growth increases
     * when the star formation ra*/
    int add = 0.005 * All.MaxPart;
    if (add < 128) add = 128;

    /* FIXME: allow shrinking; need to tweak the memmove later. */
    for(ptype = 0; ptype < 6; ptype ++) {
        newMaxSlots[ptype] = SlotsManager->info[ptype].maxsize;
        if (newMaxSlots[ptype] < atleast[ptype] + add) {
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

#pragma omp parallel for
    for(i = 0; i < NumPart; i++) {
        if(!SLOTS_ENABLED(P[i].Type)) continue;

        if(P[i].PI >= SlotsManager->info[P[i].Type].size) {
            endrun(1, "slot PI consistency failed2\n");
        }
        if(BASESLOT(i)->ID != P[i].ID) {
            endrun(1, "slot id consistency failed2\n");
        }
#pragma omp atomic
        used[P[i].Type] ++;
    }
    int ptype;
    for(ptype = 0; ptype < 6; ptype ++) {
        if(used[ptype] > 0) {
            message(0, "GC: Used slots for type %d is %d\n", ptype, used[ptype]);
        }
    }
}

void
slots_setup_topology(int NLocal[6])
{
    int offset = 0;
    int ptype;
    for(ptype = 0; ptype < 6; ptype ++) {
        /* actually allocate this many slots; FIXME: encapsulate this */
        SlotsManager->info[ptype].size = NLocal[ptype];
        int i;
#pragma omp parallel for
        for(i = 0; i < NLocal[ptype]; i++)
        {
            int j = offset + i;
            P[j].Type = ptype;
            P[j].PI = i;
        }

        offset += NLocal[ptype];
    }
}
