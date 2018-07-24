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
    /*First and last particles in this exchange*/
    int first;
    int last;
} ExchangePlan;
/*
 *
 * exchange particles according to layoutfunc.
 * layoutfunc gives the target task of particle p.
*/
static int domain_exchange_once(int (*layoutfunc)(int p), ExchangePlan * plan, int do_gc);
static void domain_build_plan(int (*layoutfunc)(int p), ExchangePlan * plan);
static int domain_find_iter_space(ExchangePlan * plan);

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
int domain_exchange(int (*layoutfunc)(int p), int do_gc) {
    int i;
    int64_t sumtogo;
    int failure = 0;

    /* register the mpi types used in communication if not yet. */
    if (MPI_TYPE_PLAN_ENTRY == 0) {
        MPI_Type_contiguous(sizeof(ExchangePlanEntry), MPI_BYTE, &MPI_TYPE_PLAN_ENTRY);
        MPI_Type_commit(&MPI_TYPE_PLAN_ENTRY);
    }

    /* flag the particles that need to be exported */
    ExchangePlan plan;
    /*Build a list of particles that will be exchanged*/
    plan.first = 0;
    plan.last = 0;
    plan.ExchangeList = mymalloc2("exchangelist", sizeof(int) * PartManager->NumPart * omp_get_max_threads());
    memset(plan.ExchangeList, 0, sizeof(int) * PartManager->NumPart * omp_get_max_threads());
    size_t *nexthr = ta_malloc("nexthr", size_t, omp_get_max_threads());
    int **threx = ta_malloc("threx", int *, omp_get_max_threads());
    threx[0] = plan.ExchangeList;
    for(i=0; i < omp_get_max_threads(); i++) {
        threx[i] = plan.ExchangeList+ i * PartManager->NumPart;
        nexthr[i] = 0;
    }

    #pragma omp parallel for
    for(i = 0; i < PartManager->NumPart; i++)
    {
        const int tid = omp_get_thread_num();
        if(P[i].IsGarbage)
            continue;
        int target = layoutfunc(i);
        if(target != ThisTask) {
            threx[tid][nexthr[tid]] = i;
            nexthr[tid]++;
        }
    }
    /*Merge step for the queue.*/
    plan.nexchange = gadget_compact_thread_arrays(plan.ExchangeList, threx, nexthr, omp_get_max_threads());
    ta_free(threx);
    ta_free(nexthr);
    /*Shrink memory*/
    plan.ExchangeList = myrealloc(plan.ExchangeList, sizeof(int) * plan.nexchange);

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

    while(MPIU_Any(plan.last < plan.nexchange, MPI_COMM_WORLD))
    {
        /* determine for each rank how many particles have to be shifted to other ranks */
        plan.last = domain_find_iter_space(&plan);
        domain_build_plan(layoutfunc, &plan);
        walltime_measure("/Domain/exchange/togo");

        sumup_large_ints(1, &plan.toGoSum.base, &sumtogo);

        message(0, "iter=%d exchange of %013ld particles\n", iter, sumtogo);

        failure = domain_exchange_once(layoutfunc, &plan, do_gc || (plan.last < plan.nexchange));

        if(failure)
            break;
        iter++;
    }

    myfree(plan.toGetOffset);
    myfree(plan.toGet);
    myfree(plan.toGoOffset);
    myfree(plan.toGo);
    myfree(plan.ExchangeList);

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
    int n, target, ptype;
    struct particle_data *partBuf;
    char * slotBuf[6];

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
        slotBuf[ptype] = mymalloc2("SlotBuf", plan->toGoSum.slots[ptype] * SlotsManager->info[ptype].elsize);
    }

    ExchangePlanEntry * toGoPtr = ta_malloc("toGoPtr", ExchangePlanEntry, NTask);
    memset(toGoPtr, 0, sizeof(toGoPtr[0]) * NTask);

    for(n = plan->first; n < plan->last; n++)
    {
        const int i = plan->ExchangeList[n];
        /* preparing for export */
        target = layoutfunc(i);

        int ptype = P[i].Type;

        /* watch out thread unsafe */
        int bufPI = toGoPtr[target].slots[ptype];
        toGoPtr[target].slots[ptype] ++;
        size_t elsize = SlotsManager->info[ptype].elsize;
        memcpy(slotBuf[ptype] + (bufPI + plan->toGoOffset[target].slots[ptype]) * elsize,
                (char*) SlotsManager->info[ptype].ptr + P[i].PI * elsize, elsize);

        /* now copy the base P; after PI has been updated */
        partBuf[plan->toGoOffset[target].base + toGoPtr[target].base] = P[i];
        toGoPtr[target].base ++;
        /* mark the particle for removal. Both secondary and base slots will be marked. */
        slots_mark_garbage(i);
    }
    /*Update done marker*/
    plan->first = plan->last;

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
        int newPI[6], i;
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
                endrun(1, "Exchange: P[%d].ID = %ld != SLOT ID = %ld\n",i,P[i].ID, BASESLOT(i)->ID);
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

/*Find how many particles we can transfer in current exchange iteration*/
static int
domain_find_iter_space(ExchangePlan * plan)
{
    int n, ptype;
    size_t nlimit = FreeBytes;

    if (nlimit <  NTask * 2 * sizeof(MPI_Request))
        endrun(1, "Not enough memory free to store requests!\n");

    nlimit -= NTask * 2 * sizeof(MPI_Request);

    message(0, "Using %td bytes for exchange.\n", nlimit);

    size_t maxsize = 0;
    for(ptype = 0; ptype < 6; ptype ++ ) {
        if(!SlotsManager->info[ptype].enabled) continue;
        if (maxsize < SlotsManager->info[ptype].elsize)
            maxsize = SlotsManager->info[ptype].elsize;
    }
    size_t package = sizeof(P[0]) + maxsize;
    if(package >= nlimit)
        endrun(212, "Package is too large, no free memory.");

    /* Fast path: if we have enough space no matter what type the particles
     * are we don't need to check them.*/
    if(plan->nexchange * (sizeof(P[0]) + maxsize) < nlimit) {
        return plan->nexchange;
    }
    /*Find how many particles we have space for.*/
    for(n = plan->first; n < plan->nexchange; n++)
    {
        const int i = plan->ExchangeList[n];
        const int ptype = P[i].Type;

        package += sizeof(P[0]) + SlotsManager->info[ptype].elsize;
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

    for(n = plan->first; n < plan->last; n++)
    {
        const int i = plan->ExchangeList[n];
        const int target = layoutfunc(i);
        const int ptype = P[i].Type;
        plan->toGo[target].base++;
        plan->toGo[target].slots[ptype]++;
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

