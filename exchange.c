#include <mpi.h>
#include <string.h>
/* #include "domain.h" */
#include "mymalloc.h"
#include "allvars.h"
#include "endrun.h"
#include "system.h"
#include "exchange.h"
#include "slotsmanager.h"

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
} ExchangePlan;
/* 
 * 
 * exchange particles according to layoutfunc.
 * layoutfunc gives the target task of particle p.
*/
static int domain_exchange_once(int (*layoutfunc)(int p), ExchangePlan * plan);
static int domain_build_plan(ptrdiff_t nlimit, int (*layoutfunc)(int p), ExchangePlan * plan);

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
int domain_exchange(int (*layoutfunc)(int p), int failfast) {
    int i;
    int64_t sumtogo;
    int failure = 0;

    /* register the mpi types used in communication if not yet. */
    if (MPI_TYPE_PLAN_ENTRY == 0) {
        MPI_Type_contiguous(sizeof(ExchangePlanEntry), MPI_BYTE, &MPI_TYPE_PLAN_ENTRY);
        MPI_Type_commit(&MPI_TYPE_PLAN_ENTRY);
    }

    /*! toGo[0][task*NTask + partner] gives the number of particles in task 'task'
     *  that have to go to task 'partner'
     *  toGo[1] is SPH, toGo[2] is BH and toGo[3] is stars
     */
    /* flag the particles that need to be exported */
    ExchangePlan plan;
    plan.toGo = (ExchangePlanEntry *) mymalloc2("toGo", sizeof(plan.toGo[0]) * NTask);
    plan.toGoOffset = (ExchangePlanEntry *) mymalloc2("toGo", sizeof(plan.toGo[0]) * NTask);
    plan.toGet = (ExchangePlanEntry *) mymalloc2("toGet", sizeof(plan.toGo[0]) * NTask);
    plan.toGetOffset = (ExchangePlanEntry *) mymalloc2("toGet", sizeof(plan.toGo[0]) * NTask);

#pragma omp parallel for
    for(i = 0; i < NumPart; i++)
    {
        int target = layoutfunc(i);
        if(target != ThisTask)
            P[i].OnAnotherDomain = 1;
        P[i].WillExport = 0;
    }

    walltime_measure("/Domain/exchange/init");

    int iter = 0, ret;
    ptrdiff_t exchange_limit;

    do
    {
        exchange_limit = FreeBytes - NTask * (24 * sizeof(int) + 16 * sizeof(MPI_Request));

        if(exchange_limit <= 0)
        {
            endrun(1, "exchange_limit=%d < 0\n", (int) exchange_limit);
        }

        /* determine for each rank how many particles have to be shifted to other ranks */
        ret = domain_build_plan(exchange_limit, layoutfunc, &plan);
        walltime_measure("/Domain/exchange/togo");
        if(ret && failfast) {
            failure = 1;
            break;
        }

        sumup_large_ints(1, &plan.toGoSum.base, &sumtogo);

        message(0, "iter=%d exchange of %013ld particles\n", iter, sumtogo);

        failure = domain_exchange_once(layoutfunc, &plan);
        if(failure)
            break;
        iter++;
    }
    while(ret > 0);

    myfree(plan.toGetOffset);
    myfree(plan.toGet);
    myfree(plan.toGoOffset);
    myfree(plan.toGo);

    return failure;
}

static int domain_exchange_once(int (*layoutfunc)(int p), ExchangePlan * plan)
{
    int i, target, ptype;
    struct particle_data *partBuf;
    char * slotBuf[6];

    ExchangePlanEntry * toGoPtr = ta_malloc("toGoPtr", ExchangePlanEntry, NTask);
    ExchangePlanEntry * toGetPtr = ta_malloc("toGetPtr", ExchangePlanEntry, NTask);

    memset(toGoPtr, 0, sizeof(toGoPtr[0]) * NTask);
    memset(toGetPtr, 0, sizeof(toGetPtr[0]) * NTask);

    int bad_exh=0;

    /*Check whether the domain exchange will succeed. If not, bail*/
    if(NumPart + plan->toGetSum.base - plan->toGetSum.base > All.MaxPart){
        message(1,"Too many particles for exchange: NumPart=%d count_get = %d count_togo=%d All.MaxPart=%d\n",
                NumPart, plan->toGetSum.base, plan->toGoSum.base, All.MaxPart);
        bad_exh = 1;
    }

    MPI_Allreduce(MPI_IN_PLACE, &bad_exh, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);

    if(bad_exh) {
        myfree(toGetPtr);
        myfree(toGoPtr);
        return bad_exh;
    }

    partBuf = (struct particle_data *) mymalloc2("partBuf", plan->toGoSum.base * sizeof(struct particle_data));

    for(ptype = 0; ptype < 6; ptype++) {
        slotBuf[ptype] = mymalloc2("SlotBuf", plan->toGoSum.slots[ptype] * SlotsManager->info[ptype].elsize);
    }

    /*FIXME: make this omp ! */
    for(i = 0; i < NumPart; i++)
    {
        if(!(P[i].OnAnotherDomain && P[i].WillExport)) continue;
        /* preparing for export */
        P[i].OnAnotherDomain = 0;
        P[i].WillExport = 0;
        target = layoutfunc(i);

        /* mark this particle as a garbage for removal later */
        int ptype = P[i].Type;

        /* watch out thread unsafe */
        int bufPI = toGoPtr[target].slots[ptype];
        toGoPtr[target].slots[ptype] ++;
        size_t elsize = SlotsManager->info[ptype].elsize;
        memcpy(slotBuf[ptype] + (bufPI + plan->toGoOffset[target].slots[ptype]) * elsize,
                (char*) SlotsManager->info[ptype].ptr + P[i].PI * elsize, elsize);

        /* now PI points to the communicate buffer location for unpacking at the target */
        P[i].PI = bufPI;

        /* now copy the base P; after PI has been updated */
        partBuf[plan->toGoOffset[target].base + toGoPtr[target].base] = P[i];
        toGoPtr[target].base ++;

        P[i].IsGarbage = 1;
    }

    walltime_measure("/Domain/exchange/makebuf");

    /* now remove the garbage particles because they have already been copied.
     * eventually we want to fill in the garbage gap or defer the gc, because it breaks the tree.
     * invariance . */

    domain_garbage_collection();
    walltime_measure("/Domain/exchange/garbage");


    int newNumPart;
    int newSlots[6];
    newNumPart = NumPart + plan->toGetSum.base;

    for(ptype = 0; ptype < 6; ptype ++) {
        newSlots[ptype] = SlotsManager->info[ptype].size + plan->toGetSum.slots[ptype];
    }

    if(newNumPart > All.MaxPart) {
        endrun(787878, "Task=%d NumPart=%d All.MaxPart=%d\n", ThisTask, newNumPart, All.MaxPart);
    }

    domain_slots_grow(newSlots);

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
                 P + NumPart, recvcounts, recvdispls, MPI_TYPE_PARTICLE,
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
        for(ptype = 0; ptype < 6; ptype ++) {
            newPI[ptype] = SlotsManager->info[ptype].size + plan->toGetOffset[src].slots[ptype];
        }

        for(i = NumPart + plan->toGetOffset[src].base;
            i < NumPart + plan->toGetOffset[src].base + plan->toGet[src].base;
            i++) {

            int ptype = P[i].Type;


            P[i].PI = newPI[ptype];

            newPI[ptype]++;

            if(!SlotsManager->info[ptype].enabled) continue;

            if(BASESLOT(i)->ID != P[i].ID) {
                endrun(1, "ID mismatched\n");
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

    NumPart = newNumPart;

    for(ptype = 0; ptype < 6; ptype++) {
        SlotsManager->info[ptype].size = newSlots[ptype];
    }

    walltime_measure("/Domain/exchange/finalize");

#ifdef DEBUG
    domain_test_id_uniqueness();
    slots_check_id_consistency();
#endif
    return 0;
}

/*This function populates the toGo and toGet arrays*/
static int
domain_build_plan(ptrdiff_t nlimit, int (*layoutfunc)(int p), ExchangePlan * plan)
{
    int n;
    size_t package;
    int ptype;

    memset(plan->toGo, 0, sizeof(plan->toGo[0]) * NTask);

    package = sizeof(P[0]);
    for(ptype = 0; ptype < 6; ptype ++ ) {
        package += SlotsManager->info[ptype].elsize;
    }
    if(package >= nlimit)
        endrun(212, "Package is too large, no free memory.");

    int insuf = 0;
    for(n = 0; n < NumPart; n++)
    {
        if(package >= nlimit) {insuf = 1; break; }
        if(!P[n].OnAnotherDomain) continue;

        int target = layoutfunc(n);
        if (target == ThisTask) continue;

        plan->toGo[target].base += 1;
        nlimit -= sizeof(P[0]);

        int ptype = P[n].Type;
        plan->toGo[target].slots[ptype] += 1;
        nlimit -= SlotsManager->info[ptype].elsize;

        P[n].WillExport = 1;	/* flag this particle for export */
    }

    MPI_Alltoall(plan->toGo, 1, MPI_TYPE_PLAN_ENTRY, plan->toGet, 1, MPI_TYPE_PLAN_ENTRY, MPI_COMM_WORLD);

    memset(&plan->toGoOffset[0], 0, sizeof(plan->toGoOffset[0]));
    memset(&plan->toGetOffset[0], 0, sizeof(plan->toGetOffset[0]));
    memcpy(&plan->toGoSum, &plan->toGo[0], sizeof(plan->toGoSum));
    memcpy(&plan->toGetSum, &plan->toGet[0], sizeof(plan->toGetSum));

    int rank;
    for(rank = 1; rank < NTask; rank ++) {
        plan->toGoOffset[rank] = plan->toGoSum;
        plan->toGetOffset[rank] = plan->toGetSum;

        plan->toGoSum.base += plan->toGo[rank].base;
        plan->toGetSum.base += plan->toGet[rank].base;

        for(ptype = 0; ptype < 6; ptype++) {
            plan->toGoSum.slots[ptype] += plan->toGo[rank].slots[ptype];
            plan->toGetSum.slots[ptype] += plan->toGet[rank].slots[ptype];
        }
    }

    return MPIU_Any(insuf, MPI_COMM_WORLD);;
}

