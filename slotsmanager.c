#include <string.h>
#include "allvars.h"
#include "slotsmanager.h"
#include "mymalloc.h"
#include "timestep.h"
#include "system.h"
#include "endrun.h"
#include "openmpsort.h"
#include "forcetree.h"

struct slots_manager_type SlotsManager[1] = {0};

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
    /*Update the active particle list:
     * if the parent is active the child should also be active.
     * Stars must always be active on formation, but
     * BHs need not be: a halo can be seeded when the particle in question is inactive.*/
    if(is_timebin_active(P[parent].TimeBin, All.Ti_Current)) {
        int childactive = atomic_fetch_and_add(&NumActiveParticle, 1);
        ActiveParticle[childactive] = child;
    }

    P[parent].Generation ++;
    uint64_t g = P[parent].Generation;
    /* change the child ID according to the generation. */
    P[child] = P[parent];
    P[child].ID = (P[parent].ID & 0x00ffffffffffffffL) + (g << 56L);

    P[child].Mass = 0;
    P[child].Type = ptype;

    if(SlotsManager->info[ptype].enabled) {
        /* if enabled, alloc a new Slot for secondary data */
        int PI = atomic_fetch_and_add(&SlotsManager->info[ptype].size, 1);

        if(PI >= SlotsManager->info[ptype].maxsize) {
            /* rare case, use an expensive critical section */
            #pragma omp critical
            {
                int N_slots[6];
                int ptype;
                for(ptype = 0; ptype < 6; ptype++) {
                    N_slots[ptype] = SlotsManager->info[ptype].size;
                }
                /* slots_grow will do the second check to ensure it is not grown twice */
                endrun(1, "This is currently unsupported; because SlotsManager.Base can be deep in the heap\n");
                slots_reserve(N_slots);
            }
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

    /* we do this only if there is an active force tree 
     * checking Nextnode is not the best way of doing so though.
     * */
    if(force_tree_allocated()) {
        int no;
        no = Nextnode[parent];
        Nextnode[parent] = child;
        Nextnode[child] = no;
        Father[child] = Father[parent];
    }
    return child;
}

/* remove garbage particles, holes in sph chunk and holes in bh buffer. */
int
slots_gc(void)
{
    if (force_tree_allocated()) {
        endrun(0, "GC breaks ForceTree invariance. ForceTree must be freed before calling GC.\n");
    }
    /* tree is invalidated if the sequence on P is reordered; */

    int tree_invalid = 0;

    /* TODO: in principle we can track this change and modify the tree nodes;
     * But doing so requires cleaning up the TimeBin link lists, and the tree
     * link lists first. likely worth it, since GC happens only in domain decompose
     * and snapshot IO, both take far more time than rebuilding the tree. */
    tree_invalid |= slots_gc_base();
    tree_invalid |= slots_gc_slots();

    MPI_Allreduce(MPI_IN_PLACE, &tree_invalid, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

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
    if(b1->gc.ReverseLink == -1 && b2->gc.ReverseLink == -1) {
        return 0;
    }
    if(b1->gc.ReverseLink == -1) return 1;
    if(b2->gc.ReverseLink == -1) return -1;
    return (b1->gc.ReverseLink > b2->gc.ReverseLink) - (b1->gc.ReverseLink < b2->gc.ReverseLink);

}

static int
slots_gc_mark()
{
    int ptype, i;

    for(ptype = 0; ptype < 6; ptype ++) {
        if(!SlotsManager->info[ptype].enabled) continue;
#pragma omp parallel for
        for(i = 0; i < SlotsManager->info[ptype].size; i++) {
            BASESLOT_PI(i, ptype)->gc.ReverseLink = -1;
        }
    }

#pragma omp parallel for
    for(i = 0; i < NumPart; i++) {
        if(!SlotsManager->info[P[i].Type].enabled) continue;

        BASESLOT(i)->gc.ReverseLink = i;
    }
}

/* sweep removed unused elements. */
static int
slots_gc_sweep(int ptype)
{
    if(!SlotsManager->info[ptype].enabled) ;

    int used = SlotsManager->info[ptype].size;
    size_t elsize = SlotsManager->info[ptype].elsize;
    int i = 0;
    /* put unused guys to the end */
    while(i < used)
    {
        while(i < used
                &&
        BASESLOT_PI(i, ptype)->gc.ReverseLink == -1) {

            memcpy(BASESLOT_PI(i, ptype),
                BASESLOT_PI(used - 1, ptype), elsize);
            used -- ;
        }
        i ++;
    }

    SlotsManager->info[ptype].size = used;
}

/* defrags ensures locality. */
static void
slots_gc_defrag(int ptype)
{

    if(!SlotsManager->info[ptype].enabled) return;
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
    if(!SlotsManager->info[ptype].enabled) return;

    /* Now update the link in BhP */
#pragma omp parallel for
    for(i = 0;
        i < SlotsManager->info[ptype].size;
        i ++) {

        P[BASESLOT_PI(i, ptype)->gc.ReverseLink].PI = i;
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

    for(ptype = 0; ptype < 6; ptype ++) {
        newMaxSlots[ptype] = SlotsManager->info[ptype].maxsize;
        while(newMaxSlots[ptype] < atleast[ptype]) {
            int add = 0.2 * newMaxSlots[ptype];
            if (add < 128) add = 128;
            newMaxSlots[ptype] += add;
            good = 0;
        }
    }
    /* no need to grow, already have enough */
    if (good) {
        return;
    }
    /* FIXME: do a global max; because all variables in All.* are synced between ranks. */

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
        if(SlotsManager->info[ptype].enabled) {
            MPI_Type_contiguous(SlotsManager->info[ptype].elsize, MPI_BYTE, &MPI_TYPE_SLOT[ptype]);
            MPI_Type_commit(&MPI_TYPE_SLOT[ptype]);
        }
    }
}

void
slots_check_id_consistency()
{
    int used[6] = {0};
    int i;

#pragma omp parallel for
    for(i = 0; i < NumPart; i++) {
        if(!SlotsManager->info[P[i].Type].enabled) continue;

        if(P[i].PI >= SlotsManager->info[P[i].Type].size) {
            endrun(1, "slot PI consistency failed2\n");
        }
        if(BASESLOT(i)->ID != P[i].ID) {
            endrun(1, "slot id consistency failed2\n");
        }
#pragma omp atomic
        used[P[i].Type] ++;
    }
}
