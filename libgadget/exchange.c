#include <mpi.h>
#include <string.h>
#include "mpsort.h"
/* #include "domain.h" */
#include "allvars.h"
#include "exchange.h"
#include "slotsmanager.h"
#include "partmanager.h"

#include "utils.h"

/*Number of structure types for particles*/
typedef struct {
    int base;
    int slots[6];
} ExchangePlanEntry;

static MPI_Datatype MPI_TYPE_PLAN_ENTRY = 0;

/*Small bitfield struct to cache the layout function and particle data*/
typedef struct {
    unsigned int ptype : 3;
    unsigned int target : 8 * sizeof(int) - 3;
} ExchangePartCache;

typedef struct {
    ExchangePlanEntry * toGo;
    ExchangePlanEntry * toGoOffset;
    ExchangePlanEntry * toGet;
    ExchangePlanEntry * toGetOffset;
    ExchangePlanEntry toGoSum;
    ExchangePlanEntry toGetSum;
    /*List of particles to exchange*/
    int * ExchangeList;
    /*Total number of exchanged particles*/
    int nexchange;
    /* last particle in current batch of the exchange.
     * Exchange stops when last == nexchange.*/
    int last;
    ExchangePartCache * layouts;
} ExchangePlan;
/*
 *
 * exchange particles according to layoutfunc.
 * layoutfunc gives the target task of particle p.
*/
static int domain_exchange_once(int (*layoutfunc)(int p), ExchangePlan * plan, int do_gc);
static void domain_build_plan(int (*layoutfunc)(int p), ExchangePlan * plan);
static int domain_find_iter_space(ExchangePlan * plan);
static void domain_build_exchange_list(int (*layoutfunc)(int p), ExchangePlan * plan);

/* This function builts the count/displ arrays from
 * the rows stored in the entry struct of the plan.
 * MPI expects a these numbers to be tightly packed in memory,
 * but our struct stores them as different columns.
 *
 * Technically speaking, the operation is therefore a transpose.
 * */
static void
_transpose_plan_entries(ExchangePlanEntry * entries, int * count, int ptype)
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

/*Plan and execute a domain exchange, also performing a garbage collection if requested*/
int domain_exchange(int (*layoutfunc)(int p), int do_gc) {
    int64_t sumtogo;
    int failure = 0;

    /* register the mpi types used in communication if not yet. */
    if (MPI_TYPE_PLAN_ENTRY == 0) {
        MPI_Type_contiguous(sizeof(ExchangePlanEntry), MPI_BYTE, &MPI_TYPE_PLAN_ENTRY);
        MPI_Type_commit(&MPI_TYPE_PLAN_ENTRY);
    }

    /*Structure for building a list of particles that will be exchanged*/
    ExchangePlan plan;
    plan.last = 0;
    plan.nexchange = PartManager->NumPart;

    /*! toGo[0][task*NTask + partner] gives the number of particles in task 'task'
     *  that have to go to task 'partner'
     *  toGo[1] is SPH, toGo[2] is BH and toGo[3] is stars
     */
    plan.toGo = (ExchangePlanEntry *) mymalloc2("toGo", sizeof(plan.toGo[0]) * NTask);
    plan.toGoOffset = (ExchangePlanEntry *) mymalloc2("toGo", sizeof(plan.toGo[0]) * NTask);
    plan.toGet = (ExchangePlanEntry *) mymalloc2("toGet", sizeof(plan.toGo[0]) * NTask);
    plan.toGetOffset = (ExchangePlanEntry *) mymalloc2("toGet", sizeof(plan.toGo[0]) * NTask);

    walltime_measure("/Domain/exchange/init");

    int iter = 0;

    do {
        domain_build_exchange_list(layoutfunc, &plan);

        /*Exit early if nothing to do*/
        if(!MPIU_Any(plan.nexchange > 0, MPI_COMM_WORLD))
        {
            myfree(plan.ExchangeList);
            break;
        }

        /* determine for each rank how many particles have to be shifted to other ranks */
        plan.last = domain_find_iter_space(&plan);
        domain_build_plan(layoutfunc, &plan);
        walltime_measure("/Domain/exchange/togo");

        sumup_large_ints(1, &plan.toGoSum.base, &sumtogo);

        message(0, "iter=%d exchange of %013ld particles\n", iter, sumtogo);

        /* Do a GC if we are asked to, if this isn't the last iteration,
         * or if we are exchanging a peculiarly large number of particles,
         * indicating large garbage. In practice, if memory is not tight,
         * this means we will usually only gc on PM steps. The gc decision
         * is made collective in domain_exchange_once*/
        int really_do_gc = do_gc || (plan.last < plan.nexchange)
            || (plan.nexchange > PartManager->NumPart / 1000);

        failure = domain_exchange_once(layoutfunc, &plan, really_do_gc);

        myfree(plan.ExchangeList);

        if(failure)
            break;
        iter++;
    }
    while(MPIU_Any(plan.last < plan.nexchange, MPI_COMM_WORLD));

    myfree(plan.toGetOffset);
    myfree(plan.toGet);
    myfree(plan.toGoOffset);
    myfree(plan.toGo);

    return failure;
}

/*Function decides whether the GC will compact slots.
 * Sets compact[6]. Is collective.*/
static void
shall_we_compact_slots(int * compact, ExchangePlan * plan)
{
    int ptype;
    for(ptype = 0; ptype < 6; ptype++) {
        /* gc if we are low on slot memory. */
        if (SlotsManager->info[ptype].size + plan->toGetSum.slots[ptype] > 0.95 * SlotsManager->info[ptype].maxsize)
            compact[ptype] = 1;
        /* gc if we had a very large exchange. */
        if(plan->toGoSum.slots[ptype] > 0.1 * SlotsManager->info[ptype].size)
            compact[ptype] = 1;
    }
    /*Make the slot compaction collective*/
    MPI_Allreduce(MPI_IN_PLACE, compact, 6, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
}

static int domain_exchange_once(int (*layoutfunc)(int p), ExchangePlan * plan, int do_gc)
{
    int n, ptype;
    struct particle_data *partBuf;
    char * slotBuf[6] = {NULL, NULL, NULL, NULL, NULL, NULL};

    int bad_exh=0;

    /*Check whether the domain exchange will succeed. If not, bail*/
    if(PartManager->NumPart + plan->toGetSum.base - plan->toGoSum.base > PartManager->MaxPart){
        message(1,"Too many particles for exchange: NumPart=%d count_get = %d count_togo=%d MaxPart=%d\n",
                PartManager->NumPart, plan->toGetSum.base, plan->toGoSum.base, PartManager->MaxPart);
        bad_exh = 1;
    }

    MPI_Allreduce(MPI_IN_PLACE, &bad_exh, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);

    if(bad_exh) {
        return bad_exh;
    }

    partBuf = (struct particle_data *) mymalloc2("partBuf", plan->toGoSum.base * sizeof(struct particle_data));

    for(ptype = 0; ptype < 6; ptype++) {
        if(!SlotsManager->info[ptype].enabled) continue;
        slotBuf[ptype] = mymalloc2("SlotBuf", plan->toGoSum.slots[ptype] * SlotsManager->info[ptype].elsize);
    }

    ExchangePlanEntry * toGoPtr = ta_malloc("toGoPtr", ExchangePlanEntry, NTask);
    memset(toGoPtr, 0, sizeof(toGoPtr[0]) * NTask);

    for(n = 0; n < plan->last; n++)
    {
        const int i = plan->ExchangeList[n];
        /* preparing for export */
        const int target = plan->layouts[n].target;

        int type = plan->layouts[n].ptype;

        /* watch out thread unsafe */
        int bufPI = toGoPtr[target].slots[type];
        toGoPtr[target].slots[type] ++;
        size_t elsize = SlotsManager->info[type].elsize;
        memcpy(slotBuf[type] + (bufPI + plan->toGoOffset[target].slots[type]) * elsize,
                (char*) SlotsManager->info[type].ptr + P[i].PI * elsize, elsize);

        /* now copy the base P; after PI has been updated */
        partBuf[plan->toGoOffset[target].base + toGoPtr[target].base] = P[i];
        toGoPtr[target].base ++;
        /* mark the particle for removal. Both secondary and base slots will be marked. */
        slots_mark_garbage(i);
    }

    myfree(plan->layouts);
    ta_free(toGoPtr);
    walltime_measure("/Domain/exchange/makebuf");

    /* Do a gc if we were asked to, or if we need one
     * to have enough space for the incoming material*/
    int shall_we_gc = do_gc || (PartManager->NumPart + plan->toGetSum.base > PartManager->MaxPart);
    if(MPIU_Any(shall_we_gc, MPI_COMM_WORLD)) {
        /*Find which slots to gc*/
        int compact[6] = {0};
        shall_we_compact_slots(compact, plan);
        slots_gc(compact);

        walltime_measure("/Domain/exchange/garbage");
    }

    int newNumPart;
    int newSlots[6] = {0};
    newNumPart = PartManager->NumPart + plan->toGetSum.base;

    for(ptype = 0; ptype < 6; ptype ++) {
        if(!SlotsManager->info[ptype].enabled) continue;
        newSlots[ptype] = SlotsManager->info[ptype].size + plan->toGetSum.slots[ptype];
    }

    if(newNumPart > PartManager->MaxPart) {
        endrun(787878, "NumPart=%d All.MaxPart=%d\n", newNumPart, PartManager->MaxPart);
    }

    slots_reserve(1, newSlots);

    int * sendcounts = (int*) ta_malloc("sendcounts", int, NTask);
    int * senddispls = (int*) ta_malloc("senddispls", int, NTask);
    int * recvcounts = (int*) ta_malloc("recvcounts", int, NTask);
    int * recvdispls = (int*) ta_malloc("recvdispls", int, NTask);

    _transpose_plan_entries(plan->toGo, sendcounts, -1);
    _transpose_plan_entries(plan->toGoOffset, senddispls, -1);
    _transpose_plan_entries(plan->toGet, recvcounts, -1);
    _transpose_plan_entries(plan->toGetOffset, recvdispls, -1);

    /* recv at the end */
    MPI_Alltoallv_sparse(partBuf, sendcounts, senddispls, MPI_TYPE_PARTICLE,
                 P + PartManager->NumPart, recvcounts, recvdispls, MPI_TYPE_PARTICLE,
                 MPI_COMM_WORLD);

    for(ptype = 0; ptype < 6; ptype ++) {
        /* skip unused slot types */
        if(!SlotsManager->info[ptype].enabled) continue;

        size_t elsize = SlotsManager->info[ptype].elsize;
        int N_slots = SlotsManager->info[ptype].size;
        char * ptr = SlotsManager->info[ptype].ptr;
        _transpose_plan_entries(plan->toGo, sendcounts, ptype);
        _transpose_plan_entries(plan->toGoOffset, senddispls, ptype);
        _transpose_plan_entries(plan->toGet, recvcounts, ptype);
        _transpose_plan_entries(plan->toGetOffset, recvdispls, ptype);

        /* recv at the end */
        MPI_Alltoallv_sparse(slotBuf[ptype], sendcounts, senddispls, MPI_TYPE_SLOT[ptype],
                     ptr + N_slots * elsize,
                     recvcounts, recvdispls, MPI_TYPE_SLOT[ptype],
                     MPI_COMM_WORLD);
    }

    int src;
    for(src = 0; src < NTask; src++) {
        /* unpack each source rank */
        int newPI[6];
        int i;
        for(ptype = 0; ptype < 6; ptype ++) {
            newPI[ptype] = SlotsManager->info[ptype].size + plan->toGetOffset[src].slots[ptype];
        }

        for(i = PartManager->NumPart + plan->toGetOffset[src].base;
            i < PartManager->NumPart + plan->toGetOffset[src].base + plan->toGet[src].base;
            i++) {

            int ptype = P[i].Type;


            P[i].PI = newPI[ptype];

            newPI[ptype]++;

            if(!SlotsManager->info[ptype].enabled) continue;

            if(BASESLOT(i)->ID != P[i].ID) {
                endrun(1, "Exchange: P[%d].ID = %ld (type %d) != SLOT ID = %ld. garbage: %d slot: %d\n",i,P[i].ID, P[i].Type, BASESLOT(i)->ID, P[i].IsGarbage, BASESLOT(i)->IsGarbage);
            }
        }
        for(ptype = 0; ptype < 6; ptype ++) {
            if(newPI[ptype] !=
                SlotsManager->info[ptype].size + plan->toGetOffset[src].slots[ptype]
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
        if(!SlotsManager->info[ptype].enabled) continue;
        myfree(slotBuf[ptype]);
    }
    myfree(partBuf);

    MPI_Barrier(MPI_COMM_WORLD);

    PartManager->NumPart = newNumPart;

    for(ptype = 0; ptype < 6; ptype++) {
        if(!SlotsManager->info[ptype].enabled) continue;
        SlotsManager->info[ptype].size = newSlots[ptype];
    }

#ifdef DEBUG
    domain_test_id_uniqueness();
    slots_check_id_consistency();
#endif
    walltime_measure("/Domain/exchange/finalize");

    return 0;
}

/* This function builds the list of particles to be exchanged.
 * All particles are processed every time, space is not considered.
 * The exchange list needs to be rebuilt every time gc is run. */
static void
domain_build_exchange_list(int (*layoutfunc)(int p), ExchangePlan * plan)
{
    int i;
    int numthreads = omp_get_max_threads();
    /*static schedule below so we only need this much memory*/
    int narr = plan->nexchange/numthreads+2;
    plan->ExchangeList = mymalloc2("exchangelist", sizeof(int) * narr * numthreads);
    size_t *nexthr = ta_malloc("nexthr", size_t, numthreads);
    int **threx = ta_malloc("threx", int *, numthreads);
    gadget_setup_thread_arrays(plan->ExchangeList, threx, nexthr,narr,numthreads);

    /* flag the particles that need to be exported */
    #pragma omp parallel for schedule(static)
    for(i=0; i < PartManager->NumPart; i++)
    {
        if(P[i].IsGarbage)
            continue;
        int target = layoutfunc(i);
        if(target != ThisTask) {
            const int tid = omp_get_thread_num();
            threx[tid][nexthr[tid]] = i;
            nexthr[tid]++;
        }
    }
    /*Merge step for the queue.*/
    plan->nexchange = gadget_compact_thread_arrays(plan->ExchangeList, threx, nexthr, numthreads);
    ta_free(threx);
    ta_free(nexthr);

    /*Shrink memory*/
    plan->ExchangeList = myrealloc(plan->ExchangeList, sizeof(int) * plan->nexchange);
}

/*Find how many particles we can transfer in current exchange iteration*/
static int
domain_find_iter_space(ExchangePlan * plan)
{
    int n, ptype;
    size_t nlimit = mymalloc_freebytes();

    if (nlimit <  4096 * 2 + NTask * 2 * sizeof(MPI_Request))
        endrun(1, "Not enough memory free to store requests!\n");

    nlimit -= 4096 * 2 + NTask * 2 * sizeof(MPI_Request);

    /* Save some memory for memory headers and wasted space at the end of each allocation.
     * Need max. 2*4096 for each heap-allocated array.*/
    nlimit -= 4096 * 4;

    message(0, "Using %td bytes for exchange.\n", nlimit);

    size_t maxsize = 0;
    for(ptype = 0; ptype < 6; ptype ++ ) {
        if(!SlotsManager->info[ptype].enabled) continue;
        if (maxsize < SlotsManager->info[ptype].elsize)
            maxsize = SlotsManager->info[ptype].elsize;
        /*Reserve space for slotBuf header*/
        nlimit -= 4096 * 2;
    }
    size_t package = sizeof(P[0]) + maxsize;
    if(package >= nlimit)
        endrun(212, "Package is too large, no free memory.");

    /* Fast path: if we have enough space no matter what type the particles
     * are we don't need to check them.*/
    if(plan->nexchange * (sizeof(P[0]) + maxsize + sizeof(ExchangePartCache)) < nlimit) {
        return plan->nexchange;
    }
    /*Find how many particles we have space for.*/
    for(n = 0; n < plan->nexchange; n++)
    {
        const int i = plan->ExchangeList[n];
        const int ptype = P[i].Type;

        package += sizeof(P[0]) + SlotsManager->info[ptype].elsize + sizeof(ExchangePartCache);
        if(package >= nlimit) {
//             message(1,"Not enough space for particles: nlimit=%d, package=%d\n",nlimit,package);
            break;
        }
    }
    return n;
}

/*This function populates the toGo and toGet arrays*/
static void
domain_build_plan(int (*layoutfunc)(int p), ExchangePlan * plan)
{
    int ptype, n;

    memset(plan->toGo, 0, sizeof(plan->toGo[0]) * NTask);

    plan->layouts = mymalloc("layoutcache",sizeof(ExchangePartCache) * plan->last);

    #pragma omp parallel for
    for(n = 0; n < plan->last; n++)
    {
        const int i = plan->ExchangeList[n];
        const int target = layoutfunc(i);
        plan->layouts[n].ptype = P[i].Type;
        plan->layouts[n].target = target;
    }

    /*Do the sum*/
    for(n = 0; n < plan->last; n++)
    {
        plan->toGo[plan->layouts[n].target].base++;
        plan->toGo[plan->layouts[n].target].slots[plan->layouts[n].ptype]++;
    }

    MPI_Alltoall(plan->toGo, 1, MPI_TYPE_PLAN_ENTRY, plan->toGet, 1, MPI_TYPE_PLAN_ENTRY, MPI_COMM_WORLD);

    memset(&plan->toGoOffset[0], 0, sizeof(plan->toGoOffset[0]));
    memset(&plan->toGetOffset[0], 0, sizeof(plan->toGetOffset[0]));
    memcpy(&plan->toGoSum, &plan->toGo[0], sizeof(plan->toGoSum));
    memcpy(&plan->toGetSum, &plan->toGet[0], sizeof(plan->toGetSum));

    int rank;
    for(rank = 1; rank < NTask; rank ++) {
        /* Direct assignment breaks compilers like icc */
        memcpy(&plan->toGoOffset[rank], &plan->toGoSum, sizeof(plan->toGoSum));
        memcpy(&plan->toGetOffset[rank], &plan->toGetSum, sizeof(plan->toGetSum));

        plan->toGoSum.base += plan->toGo[rank].base;
        plan->toGetSum.base += plan->toGet[rank].base;

        for(ptype = 0; ptype < 6; ptype++) {
            plan->toGoSum.slots[ptype] += plan->toGo[rank].slots[ptype];
            plan->toGetSum.slots[ptype] += plan->toGet[rank].slots[ptype];
        }
    }
}

/* used only by test uniqueness */
static void
mp_order_by_id(const void * data, void * radix, void * arg) {
    ((uint64_t *) radix)[0] = ((MyIDType*) data)[0];
}

void
domain_test_id_uniqueness(void)
{
    int i;
    double t0, t1;
    MyIDType *ids, *ids_first;

    message(0, "Testing ID uniqueness...\n");

    if(PartManager->NumPart == 0)
    {
        endrun(8, "need at least one particle per cpu\n");
    }

    t0 = second();

    ids = (MyIDType *) mymalloc("ids", PartManager->NumPart * sizeof(MyIDType));
    ids_first = (MyIDType *) mymalloc("ids_first", NTask * sizeof(MyIDType));

    #pragma omp parallel for
    for(i = 0; i < PartManager->NumPart; i++) {
        ids[i] = P[i].ID;
        if(P[i].IsGarbage)
            ids[i] = (MyIDType) -1;
    }

    mpsort_mpi(ids, PartManager->NumPart, sizeof(MyIDType), mp_order_by_id, 8, NULL, MPI_COMM_WORLD);

    /*Remove garbage from the end*/
    int nids = PartManager->NumPart;
    while(nids > 0 && (ids[nids-1] == (MyIDType)-1)) {
        nids--;
    }

    #pragma omp parallel for
    for(i = 1; i < nids; i++) {
        if(ids[i] == ids[i - 1])
        {
            endrun(12, "non-unique ID=%013ld found on task=%d (i=%d NumPart=%d)\n",
                    ids[i], ThisTask, i, nids);
        }
    }

    MPI_Allgather(&ids[0], sizeof(MyIDType), MPI_BYTE, ids_first, sizeof(MyIDType), MPI_BYTE, MPI_COMM_WORLD);

    if(ThisTask < NTask - 1)
        if(ids[nids - 1] == ids_first[ThisTask + 1])
        {
            endrun(13, "non-unique ID=%d found on task=%d\n", (int) ids[nids - 1], ThisTask);
        }

    myfree(ids_first);
    myfree(ids);

    t1 = second();

    message(0, "success.  took=%g sec\n", timediff(t0, t1));
}

