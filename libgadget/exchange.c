#include <mpi.h>
#include <omp.h>
#include <string.h>
#include "exchange.h"
#include "slotsmanager.h"
#include "partmanager.h"
#include "walltime.h"

#include "utils/endrun.h"
#include "utils/mpsort.h"
#include "utils/mymalloc.h"
#include "utils/system.h"

/*Number of structure types for particles*/
typedef struct {
    int64_t base;
    int64_t slots[6];
} ExchangePlanEntry;

static MPI_Datatype MPI_TYPE_PLAN_ENTRY = 0;
static MPI_Datatype MPI_TYPE_PARTICLE = 0;
static MPI_Datatype MPI_TYPE_SLOT[6] = {0};

/*Small struct to cache the layout function and particle data*/
typedef struct {
    unsigned int ptype;
    unsigned int target ;
} ExchangePartCache;

typedef struct {
    ExchangePlanEntry * toGo;
    ExchangePlanEntry * toGoOffset;
    ExchangePlanEntry * toGet;
    ExchangePlanEntry * toGetOffset;
    ExchangePlanEntry toGoSum;
    ExchangePlanEntry toGetSum;
    int NTask;
    /*List of particles to exchange*/
    int * ExchangeList;
    /*Total number of exchanged particles*/
    size_t nexchange;
    /*Number of garbage particles*/
    int64_t ngarbage;
    /* last particle in current batch of the exchange.
     * Exchange stops when last == nexchange.*/
    size_t last;
    ExchangePartCache * layouts;
} ExchangePlan;
/*
 *
 * exchange particles according to layoutfunc.
 * layoutfunc gives the target task of particle p.
*/
static int domain_exchange_once(ExchangePlan * plan, int do_gc, struct part_manager_type * pman, struct slots_manager_type * sman, MPI_Comm Comm);
static void domain_build_plan(int iter, ExchangeLayoutFunc layoutfunc, const void * layout_userdata, ExchangePlan * plan, struct part_manager_type * pman, MPI_Comm Comm);
static size_t domain_find_iter_space(ExchangePlan * plan, const struct part_manager_type * pman, const struct slots_manager_type * sman);
static void domain_build_exchange_list(ExchangeLayoutFunc layoutfunc, const void * layout_userdata, ExchangePlan * plan, struct part_manager_type * pman, struct slots_manager_type * sman, MPI_Comm Comm);

/* This function builds the count/displ arrays from
 * the rows stored in the entry struct of the plan.
 * MPI expects these numbers to be tightly packed in memory,
 * but our struct stores them as different columns.
 *
 * Technically speaking, the operation is therefore a transpose.
 * */
static void
_transpose_plan_entries(ExchangePlanEntry * entries, int * count, int ptype, int NTask)
{
    int i;
    for(i = 0; i < NTask; i ++) {
        if(ptype == -1) {
            count[i] = entries[i].base;
        } else {
            count[i] = entries[i].slots[ptype];
        }
    }
}

static ExchangePlan
domain_init_exchangeplan(MPI_Comm Comm)
{
    ExchangePlan plan;
    MPI_Comm_size(Comm, &plan.NTask);
    /*! toGo[0][task*NTask + partner] gives the number of particles in task 'task'
     *  that have to go to task 'partner'
     *  toGo[1] is SPH, toGo[2] is BH and toGo[3] is stars
     */
    plan.toGo = ta_malloc("toGo", ExchangePlanEntry, plan.NTask);
    plan.toGoOffset = ta_malloc("toGoOffSet", ExchangePlanEntry, plan.NTask);
    plan.toGet = ta_malloc("toGet", ExchangePlanEntry, plan.NTask);
    plan.toGetOffset = ta_malloc("toGetOffset", ExchangePlanEntry, plan.NTask);
    return plan;
}

static void
domain_free_exchangeplan(ExchangePlan * plan)
{
    myfree(plan->toGetOffset);
    myfree(plan->toGet);
    myfree(plan->toGoOffset);
    myfree(plan->toGo);
}

/*Plan and execute a domain exchange, also performing a garbage collection if requested*/
int domain_exchange(ExchangeLayoutFunc layoutfunc, const void * layout_userdata, PreExchangeList * preexch, struct part_manager_type * pman, struct slots_manager_type * sman, int maxiter, MPI_Comm Comm) {
    int failure = 0;

    /* register the MPI types used in communication if not yet. */
    if (MPI_TYPE_PLAN_ENTRY == 0) {
        MPI_Type_contiguous(sizeof(ExchangePlanEntry), MPI_BYTE, &MPI_TYPE_PLAN_ENTRY);
        MPI_Type_commit(&MPI_TYPE_PLAN_ENTRY);
    }
    if (MPI_TYPE_PARTICLE == 0) {
        MPI_Type_contiguous(sizeof(struct particle_data), MPI_BYTE, &MPI_TYPE_PARTICLE);
        MPI_Type_commit(&MPI_TYPE_PARTICLE);
        int ptype;
        for(ptype = 0; ptype < 6; ptype++) {
            if(!sman->info[ptype].enabled)
                continue;
            MPI_Type_contiguous(sman->info[ptype].elsize, MPI_BYTE, &MPI_TYPE_SLOT[ptype]);
            MPI_Type_commit(&MPI_TYPE_SLOT[ptype]);
        }
    }



    /*Structure for building a list of particles that will be exchanged*/
    ExchangePlan plan = domain_init_exchangeplan(Comm);

    int iter = 0;

    do {
        if(iter >= maxiter) {
            failure = 1;
            break;
        }
        /* Use the pre-exchange list if we can*/
        if(preexch && preexch->ExchangeList) {
            plan.ngarbage= preexch->ngarbage;
            plan.nexchange = preexch->nexchange;
            plan.ExchangeList = preexch->ExchangeList;
            /* We only use this once.*/
            preexch->ExchangeList = NULL;
        }
        else {
            domain_build_exchange_list(layoutfunc, layout_userdata, &plan, pman, sman, Comm);
        }
        walltime_measure("/Domain/exchange/togo");

        /*Exit early if nothing to do*/
        if(!MPIU_Any(plan.nexchange > 0, Comm))
        {
            myfree(plan.ExchangeList);
            break;
        }

        /* determine for each rank how many particles have to be shifted to other ranks */
        plan.last = domain_find_iter_space(&plan, pman, sman);
        domain_build_plan(iter, layoutfunc, layout_userdata, &plan, pman, Comm);

        /* Do a GC if this isn't the last iteration.
         * The gc decision is made collective in domain_exchange_once,
         * and a gc will also be done if we have no space for particles.*/
        int do_gc = (plan.last < plan.nexchange);
        failure = domain_exchange_once(&plan, do_gc, pman, sman, Comm);

        myfree(plan.ExchangeList);

        if(failure)
            break;
        iter++;
    }
    while(MPIU_Any(plan.last < plan.nexchange, Comm));
#ifdef DEBUG
    /* This does not apply for the FOF code, where the exchange list is pre-assigned
     * and we only get one iteration. */
    if(!failure && maxiter > 1) {
        ExchangePlan plan9 = domain_init_exchangeplan(Comm);
        /* Do not drift again*/
        domain_build_exchange_list(layoutfunc, layout_userdata, &plan9, pman, sman, Comm);
        if(plan9.nexchange > 0)
            endrun(5, "Still have %ld particles in exchange list\n", plan9.nexchange);
        myfree(plan9.ExchangeList);
        domain_free_exchangeplan(&plan9);
    }
#endif
    domain_free_exchangeplan(&plan);

    return failure;
}

/*Function decides whether the GC will compact slots.
 * Sets compact[6]. Is collective.*/
static void
shall_we_compact_slots(int * compact, ExchangePlan * plan, const struct slots_manager_type * sman, MPI_Comm Comm)
{
    int ptype;
    int lcompact[6] = {0};
    for(ptype = 0; ptype < 6; ptype++) {
        /* gc if we are low on slot memory. */
        if (sman->info[ptype].size + plan->toGetSum.slots[ptype] > 0.95 * sman->info[ptype].maxsize)
            lcompact[ptype] = 1;
        /* gc if we had a very large exchange. */
        if(plan->toGoSum.slots[ptype] > 0.1 * sman->info[ptype].size)
            lcompact[ptype] = 1;
    }
    /*Make the slot compaction collective*/
    MPI_Allreduce(lcompact, compact, 6, MPI_INT, MPI_LOR, Comm);
}

static int domain_exchange_once(ExchangePlan * plan, int do_gc, struct part_manager_type * pman, struct slots_manager_type * sman, MPI_Comm Comm)
{
    size_t n;
    int ptype;
    struct particle_data *partBuf;
    char * slotBuf[6] = {NULL, NULL, NULL, NULL, NULL, NULL};

    /* Check whether the domain exchange will succeed.
     * Garbage particles will be collected after the particles are exported, so do not need to count.*/
    int64_t needed = pman->NumPart + plan->toGetSum.base - plan->toGoSum.base  - plan->ngarbage;
    if(needed > pman->MaxPart)
        message(1,"Too many particles for exchange: NumPart=%ld count_get = %ld count_togo=%ld garbage = %ld MaxPart=%ld\n",
                pman->NumPart, plan->toGetSum.base, plan->toGoSum.base, plan->ngarbage, pman->MaxPart);
    if(MPIU_Any(needed > pman->MaxPart, Comm)) {
        myfree(plan->layouts);
        return 1;
    }

    for(ptype = 0; ptype < 6; ptype++) {
        if(!sman->info[ptype].enabled) continue;
        slotBuf[ptype] = (char *) mymalloc2("SlotBuf", plan->toGoSum.slots[ptype] * sman->info[ptype].elsize);
    }

    partBuf = (struct particle_data *) mymalloc2("partBuf", plan->toGoSum.base * sizeof(struct particle_data));

    ExchangePlanEntry * toGoPtr = ta_malloc("toGoPtr", ExchangePlanEntry, plan->NTask);
    memset(toGoPtr, 0, sizeof(toGoPtr[0]) * plan->NTask);

    for(n = 0; n < plan->last; n++)
    {
        const int i = plan->ExchangeList[n];
        /* preparing for export */
        const int target = plan->layouts[n].target;

        int type = plan->layouts[n].ptype;

        /* watch out thread unsafe */
        int bufPI = toGoPtr[target].slots[type];
        toGoPtr[target].slots[type] ++;
        size_t elsize = sman->info[type].elsize;
        if(sman->info[type].enabled)
            memcpy(slotBuf[type] + (bufPI + plan->toGoOffset[target].slots[type]) * elsize,
                (char*) sman->info[type].ptr + pman->Base[i].PI * elsize, elsize);
        /* now copy the base P; after PI has been updated */
        memcpy(&(partBuf[plan->toGoOffset[target].base + toGoPtr[target].base]), pman->Base+i, sizeof(struct particle_data));
        toGoPtr[target].base ++;
        /* mark the particle for removal. Both secondary and base slots will be marked. */
        slots_mark_garbage(i, pman, sman);
    }

    myfree(plan->layouts);
    ta_free(toGoPtr);
    walltime_measure("/Domain/exchange/makebuf");

    /* Do a gc if we were asked to, or if we need one
     * to have enough space for the incoming material*/
    int shall_we_gc = do_gc || (pman->NumPart + plan->toGetSum.base > pman->MaxPart);
    if(MPIU_Any(shall_we_gc, Comm)) {
        /*Find which slots to gc*/
        int compact[6] = {0};
        shall_we_compact_slots(compact, plan, sman, Comm);
        slots_gc(compact, pman, sman);

        walltime_measure("/Domain/exchange/garbage");
    }

    int64_t newNumPart;
    int64_t newSlots[6] = {0};
    newNumPart = pman->NumPart + plan->toGetSum.base;

    for(ptype = 0; ptype < 6; ptype ++) {
        if(!sman->info[ptype].enabled) continue;
        newSlots[ptype] = sman->info[ptype].size + plan->toGetSum.slots[ptype];
    }

    if(newNumPart > pman->MaxPart) {
        endrun(787878, "NumPart=%ld MaxPart=%ld\n", newNumPart, pman->MaxPart);
    }

    int * sendcounts = (int*) ta_malloc("sendcounts", int, plan->NTask);
    int * senddispls = (int*) ta_malloc("senddispls", int, plan->NTask);
    int * recvcounts = (int*) ta_malloc("recvcounts", int, plan->NTask);
    int * recvdispls = (int*) ta_malloc("recvdispls", int, plan->NTask);

    _transpose_plan_entries(plan->toGo, sendcounts, -1, plan->NTask);
    _transpose_plan_entries(plan->toGoOffset, senddispls, -1, plan->NTask);
    _transpose_plan_entries(plan->toGet, recvcounts, -1, plan->NTask);
    _transpose_plan_entries(plan->toGetOffset, recvdispls, -1, plan->NTask);

#ifdef DEBUG
    message(0, "Starting particle data exchange\n");
#endif
    /* recv at the end */
    MPI_Alltoallv_sparse(partBuf, sendcounts, senddispls, MPI_TYPE_PARTICLE,
                 pman->Base + pman->NumPart, recvcounts, recvdispls, MPI_TYPE_PARTICLE,
                 Comm);

    /* Do not need Particle buffer any more, make space for more slots*/
    myfree(partBuf);

    message(0, "Done particle data exchange\n");

    slots_reserve(1, newSlots, sman);
    /* Ensure the reservations are finished on all tasks before we start sending the data*/
    MPI_Barrier(Comm);

    for(ptype = 0; ptype < 6; ptype ++) {
        /* skip unused slot types */
        if(!sman->info[ptype].enabled) continue;

        size_t elsize = sman->info[ptype].elsize;
        int N_slots = sman->info[ptype].size;
        char * ptr = sman->info[ptype].ptr;
        _transpose_plan_entries(plan->toGo, sendcounts, ptype, plan->NTask);
        _transpose_plan_entries(plan->toGoOffset, senddispls, ptype, plan->NTask);
        _transpose_plan_entries(plan->toGet, recvcounts, ptype, plan->NTask);
        _transpose_plan_entries(plan->toGetOffset, recvdispls, ptype, plan->NTask);

#ifdef DEBUG
        message(0, "Starting exchange for slot %d\n", ptype);
#endif

        /* recv at the end */
        MPI_Alltoallv_sparse(slotBuf[ptype], sendcounts, senddispls, MPI_TYPE_SLOT[ptype],
                     ptr + N_slots * elsize,
                     recvcounts, recvdispls, MPI_TYPE_SLOT[ptype],
                     Comm);
    }

#ifdef DEBUG
        message(0, "Done with AlltoAllv\n");
#endif
    int src;
    for(src = 0; src < plan->NTask; src++) {
        /* unpack each source rank */
        int64_t newPI[6];
        int64_t i;
        for(ptype = 0; ptype < 6; ptype ++) {
            newPI[ptype] = sman->info[ptype].size + plan->toGetOffset[src].slots[ptype];
        }

        for(i = pman->NumPart + plan->toGetOffset[src].base;
            i < pman->NumPart + plan->toGetOffset[src].base + plan->toGet[src].base;
            i++) {

            int ptype = pman->Base[i].Type;


            pman->Base[i].PI = newPI[ptype];

            newPI[ptype]++;

            if(!sman->info[ptype].enabled) continue;

#ifdef DEBUG
            int PI = pman->Base[i].PI;
            if(BASESLOT_PI(PI, ptype, sman)->ID != pman->Base[i].ID) {
                endrun(1, "Exchange: P[%ld].ID = %ld (type %d) != SLOT ID = %ld. garbage: %d ReverseLink: %d\n",i,pman->Base[i].ID, pman->Base[i].Type, BASESLOT_PI(PI, ptype, sman)->ID, pman->Base[i].IsGarbage, BASESLOT_PI(PI, ptype, sman)->ReverseLink);
            }
#endif
        }
        for(ptype = 0; ptype < 6; ptype ++) {
            if(newPI[ptype] !=
                sman->info[ptype].size + plan->toGetOffset[src].slots[ptype]
              + plan->toGet[src].slots[ptype]) {
                endrun(1, "N_slots mismatched\n");
            }
        }
    }

    walltime_measure("/Domain/exchange/alltoall");

    myfree(recvdispls);
    myfree(recvcounts);
    myfree(senddispls);
    myfree(sendcounts);
    for(ptype = 5; ptype >=0; ptype --) {
        if(!sman->info[ptype].enabled) continue;
        myfree(slotBuf[ptype]);
    }

    pman->NumPart = newNumPart;

    for(ptype = 0; ptype < 6; ptype++) {
        if(!sman->info[ptype].enabled) continue;
        sman->info[ptype].size = newSlots[ptype];
    }

#ifdef DEBUG
    domain_test_id_uniqueness(pman);
    slots_check_id_consistency(pman, sman);
    walltime_measure("/Domain/exchange/finalize");
#endif

    return 0;
}

/* This function builds the list of particles to be exchanged.
 * All particles are processed every time, space is not considered.
 * The exchange list needs to be rebuilt every time gc is run. */
static void
domain_build_exchange_list(ExchangeLayoutFunc layoutfunc, const void * layout_userdata, ExchangePlan * plan, struct part_manager_type * pman, struct slots_manager_type * sman, MPI_Comm Comm)
{
    /*Garbage particles are counted so we have an accurate memory estimate*/
    int ngarbage = 0;
    gadget_thread_arrays gthread = gadget_setup_thread_arrays("exchangelist", 1, pman->NumPart);
    int ThisTask;
    MPI_Comm_rank(Comm, &ThisTask);
    /* flag the particles that need to be exported */
    #pragma omp parallel
    {
        int i;
        size_t nexthr_local = 0;
        const int tid = omp_get_thread_num();
        int * threx_local = gthread.srcs[tid];
        #pragma omp for schedule(static, gthread.schedsz) reduction(+: ngarbage)
        for(i=0; i < pman->NumPart; i++)
        {
            if(pman->Base[i].IsGarbage) {
                ngarbage++;
                continue;
            }
            int target = layoutfunc(i, layout_userdata);
            if(target != ThisTask) {
                threx_local[nexthr_local] = i;
                nexthr_local++;
            }
        }
        gthread.sizes[tid] = nexthr_local;
    }
    plan->ngarbage = ngarbage;
    /*Merge step for the queue.*/
    plan->nexchange = gadget_compact_thread_arrays(&plan->ExchangeList, &gthread);
    /*Shrink memory*/
    plan->ExchangeList = (int *) myrealloc(plan->ExchangeList, sizeof(int) * plan->nexchange);
}

/*Find how many particles we can transfer in current exchange iteration*/
static size_t
domain_find_iter_space(ExchangePlan * plan, const struct part_manager_type * pman, const struct slots_manager_type * sman)
{
    int ptype;
    size_t n, nlimit = mymalloc_freebytes();

    if (nlimit <  4096L * 6 + plan->NTask * 2 * sizeof(MPI_Request))
        endrun(1, "Not enough memory free to store requests!\n");

    nlimit -= 4096 * 2L + plan->NTask * 2 * sizeof(MPI_Request);

    /* Save some memory for memory headers and wasted space at the end of each allocation.
     * Need max. 2*4096 for each heap-allocated array.*/
    nlimit -= 4096 * 4L;

    size_t maxsize = 0;
    for(ptype = 0; ptype < 6; ptype ++ ) {
        if(!sman->info[ptype].enabled) continue;
        if (maxsize < sman->info[ptype].elsize)
            maxsize = sman->info[ptype].elsize;
        /*Reserve space for slotBuf header*/
        nlimit -= 4096 * 2L;
    }

    /* We want to avoid doing an alltoall with
     * more than 2GB of material as this hangs.*/
    const size_t maxexch = 2040L*1024L*1024L;
    if(nlimit > maxexch)
        nlimit = maxexch;
    message(0, "Using %td bytes for exchange.\n", nlimit);

    size_t package = sizeof(pman->Base[0]) + maxsize;
    if(package >= nlimit || nlimit > mymalloc_freebytes())
        endrun(212, "Package is too large, no free memory: package = %lu nlimit = %lu.", package, nlimit);

    /* Fast path: if we have enough space no matter what type the particles
     * are we don't need to check them.*/
    if(plan->nexchange * (sizeof(pman->Base[0]) + maxsize + sizeof(ExchangePartCache)) < nlimit) {
        return plan->nexchange;
    }

    /*Find how many particles we have space for.*/
    for(n = 0; n < plan->nexchange; n++)
    {
        const int i = plan->ExchangeList[n];
        const int ptype = pman->Base[i].Type;
        package += sizeof(pman->Base[0]) + sman->info[ptype].elsize + sizeof(ExchangePartCache);
        if(package >= nlimit) {
//             message(1,"Not enough space for particles: nlimit=%d, package=%d\n",nlimit,package);
            break;
        }
    }
    return n;
}

/*This function populates the toGo and toGet arrays*/
static void
domain_build_plan(int iter, ExchangeLayoutFunc layoutfunc, const void * layout_userdata, ExchangePlan * plan, struct part_manager_type * pman, MPI_Comm Comm)
{
    int ptype;
    size_t n;

    memset(plan->toGo, 0, sizeof(plan->toGo[0]) * plan->NTask);

    plan->layouts = (ExchangePartCache *) mymalloc("layoutcache",sizeof(ExchangePartCache) * plan->last);

    #pragma omp parallel for
    for(n = 0; n < plan->last; n++)
    {
        const int i = plan->ExchangeList[n];
        const int target = layoutfunc(i, layout_userdata);
        plan->layouts[n].ptype = pman->Base[i].Type;
        plan->layouts[n].target = target;
        if(target >= plan->NTask || target < 0)
            endrun(4, "layoutfunc for %d returned unreasonable %d for %d tasks\n", i, target, plan->NTask);
    }

    /*Do the sum*/
    for(n = 0; n < plan->last; n++)
    {
        plan->toGo[plan->layouts[n].target].base++;
        plan->toGo[plan->layouts[n].target].slots[plan->layouts[n].ptype]++;
    }

    MPI_Alltoall(plan->toGo, 1, MPI_TYPE_PLAN_ENTRY, plan->toGet, 1, MPI_TYPE_PLAN_ENTRY, Comm);

    memset(&plan->toGoOffset[0], 0, sizeof(plan->toGoOffset[0]));
    memset(&plan->toGetOffset[0], 0, sizeof(plan->toGetOffset[0]));
    memcpy(&plan->toGoSum, &plan->toGo[0], sizeof(plan->toGoSum));
    memcpy(&plan->toGetSum, &plan->toGet[0], sizeof(plan->toGetSum));

    int rank;
    int64_t maxbasetogo=-1, maxbasetoget=-1;
    for(rank = 1; rank < plan->NTask; rank ++) {
        /* Direct assignment breaks compilers like icc */
        memcpy(&plan->toGoOffset[rank], &plan->toGoSum, sizeof(plan->toGoSum));
        memcpy(&plan->toGetOffset[rank], &plan->toGetSum, sizeof(plan->toGetSum));

        plan->toGoSum.base += plan->toGo[rank].base;
        plan->toGetSum.base += plan->toGet[rank].base;
        if(plan->toGo[rank].base > maxbasetogo)
            maxbasetogo = plan->toGo[rank].base;
        if(plan->toGet[rank].base > maxbasetoget)
            maxbasetoget = plan->toGet[rank].base;

        for(ptype = 0; ptype < 6; ptype++) {
            plan->toGoSum.slots[ptype] += plan->toGo[rank].slots[ptype];
            plan->toGetSum.slots[ptype] += plan->toGet[rank].slots[ptype];
        }
    }

    int64_t maxbasetogomax, maxbasetogetmax, sumtogo;
    MPI_Reduce(&maxbasetogo, &maxbasetogomax, 1, MPI_INT64, MPI_MAX, 0, Comm);
    MPI_Reduce(&maxbasetoget, &maxbasetogetmax, 1, MPI_INT64, MPI_MAX, 0, Comm);
    MPI_Reduce(&plan->toGoSum.base, &sumtogo, 1, MPI_INT64, MPI_SUM, 0, Comm);
    message(0, "iter = %d Total particles in flight: %ld Largest togo: %ld, toget %ld\n", iter, sumtogo, maxbasetogomax, maxbasetogetmax);
}

/* used only by test uniqueness */
static void
mp_order_by_id(const void * data, void * radix, void * arg) {
    ((uint64_t *) radix)[0] = ((MyIDType*) data)[0];
}

void
domain_test_id_uniqueness(struct part_manager_type * pman)
{
    int64_t i;
    MyIDType *ids;
    int NTask, ThisTask;
    MPI_Comm_size(MPI_COMM_WORLD, &NTask);
    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);

    message(0, "Testing ID uniqueness...\n");

    ids = (MyIDType *) mymalloc("ids", pman->NumPart * sizeof(MyIDType));

    #pragma omp parallel for
    for(i = 0; i < pman->NumPart; i++) {
        ids[i] = pman->Base[i].ID;
        if(pman->Base[i].IsGarbage)
            ids[i] = (MyIDType) -1;
    }

    mpsort_mpi(ids, pman->NumPart, sizeof(MyIDType), mp_order_by_id, 8, NULL, MPI_COMM_WORLD);

    /*Remove garbage from the end*/
    int64_t nids = pman->NumPart;
    while(nids > 0 && (ids[nids-1] == (MyIDType)-1)) {
        nids--;
    }

    #pragma omp parallel for
    for(i = 1; i < nids; i++) {
        if(ids[i] <= ids[i - 1])
        {
            endrun(12, "non-unique (or non-ordered) ID=%013ld found on task=%d (i=%ld NumPart=%ld)\n",
                    ids[i], ThisTask, i, nids);
        }
    }

    MyIDType * prev = ta_malloc("prev", MyIDType, 1);
    memset(prev, 0, sizeof(MyIDType));
    const int TAG = 0xdead;

    if(NTask > 1) {
        if(ThisTask == 0) {
            MyIDType * ptr = prev;
            if(nids > 0) {
                ptr = ids;
            }
            MPI_Send(ptr, sizeof(MyIDType), MPI_BYTE, ThisTask + 1, TAG, MPI_COMM_WORLD);
        }
        else if(ThisTask == NTask - 1) {
            MPI_Recv(prev, sizeof(MyIDType), MPI_BYTE,
                    ThisTask - 1, TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        else if(nids == 0) {
            /* simply pass through whatever we get */
            MPI_Recv(prev, sizeof(MyIDType), MPI_BYTE, ThisTask - 1, TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Send(prev, sizeof(MyIDType), MPI_BYTE, ThisTask + 1, TAG, MPI_COMM_WORLD);
        }
        else
        {
            MPI_Sendrecv(
                    ids+(nids - 1), sizeof(MyIDType), MPI_BYTE,
                    ThisTask + 1, TAG,
                    prev, sizeof(MyIDType), MPI_BYTE,
                    ThisTask - 1, TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }

    if(ThisTask > 1) {
        if(nids > 0) {
            if(ids[0] <= *prev && ids[0])
                endrun(13, "non-unique ID=%ld found on task=%d\n", ids[0], ThisTask);
        }
    }

    myfree(prev);
    myfree(ids);
}
