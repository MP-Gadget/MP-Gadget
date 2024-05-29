#include <mpi.h>
#include <omp.h>
#include <string.h>
#include "exchange.h"
#include "forcetree.h"
#include "slotsmanager.h"
#include "partmanager.h"
#include "walltime.h"
#include "timefac.h"

#include "utils.h"
#include "utils/mpsort.h"

/*Number of structure types for particles*/
typedef struct {
    size_t totalbytes;
    int64_t base;
    int64_t slots[6];
} ExchangePlanEntry;

static MPI_Datatype MPI_TYPE_PLAN_ENTRY = 0;

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
    int ThisTask;
    /*List of particles to exchange*/
    int * ExchangeList;
    /*Total number of exchanged particles*/
    size_t nexchange;
    /*Number of garbage particles*/
    int64_t ngarbage;
    size_t total_togetbytes;
    size_t total_togobytes;
    ExchangePartCache * layouts;
} ExchangePlan;
/*
 *
 * exchange particles according to layoutfunc.
 * layoutfunc gives the target task of particle p.
*/
static int domain_exchange_once(ExchangePlan * plan, struct part_manager_type * pman, struct slots_manager_type * sman, int tag, MPI_Comm Comm);
static void domain_build_plan(ExchangeLayoutFunc layoutfunc, const void * layout_userdata, ExchangePlan * plan, struct part_manager_type * pman, struct slot_info * sinfo, MPI_Comm Comm);
static int domain_check_iter_space(ExchangePlan * plan);
static void domain_build_exchange_list(ExchangeLayoutFunc layoutfunc, const void * layout_userdata, ExchangePlan * plan, struct part_manager_type * pman, struct slots_manager_type * sman, MPI_Comm Comm);

static ExchangePlan
domain_init_exchangeplan(MPI_Comm Comm)
{
    ExchangePlan plan = {0};
    MPI_Comm_size(Comm, &plan.NTask);
    MPI_Comm_rank(Comm, &plan.ThisTask);
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
    myfree(plan->ExchangeList);
    myfree(plan->toGetOffset);
    myfree(plan->toGet);
    myfree(plan->toGoOffset);
    myfree(plan->toGo);
}

/*Plan and execute a domain exchange, also performing a garbage collection if requested*/
int domain_exchange(ExchangeLayoutFunc layoutfunc, const void * layout_userdata, PreExchangeList * preexch, struct part_manager_type * pman, struct slots_manager_type * sman, int maxiter, MPI_Comm Comm) {
    /* register the MPI types used in communication if not yet. */
    if (MPI_TYPE_PLAN_ENTRY == 0) {
        MPI_Type_contiguous(sizeof(ExchangePlanEntry), MPI_BYTE, &MPI_TYPE_PLAN_ENTRY);
        MPI_Type_commit(&MPI_TYPE_PLAN_ENTRY);
    }

    /*Structure for building a list of particles that will be exchanged*/
    ExchangePlan plan = domain_init_exchangeplan(Comm);
    /* Use the pre-exchange list if we can*/
    if(preexch && preexch->ExchangeList) {
        plan.ngarbage= preexch->ngarbage;
        plan.nexchange = preexch->nexchange;
        plan.ExchangeList = preexch->ExchangeList;
        /* We only use this once.*/
        preexch->ExchangeList = NULL;
    }
    else {
        /* This needs to be re-run if there is a gc*/
        domain_build_exchange_list(layoutfunc, layout_userdata, &plan, pman, sman, Comm);
    }
    walltime_measure("/Domain/exchange/togo");

    int failure = 0;
    /*If we have work to do, do it */
    if(MPIU_Any(plan.nexchange > 0, Comm))
    {
        domain_build_plan(layoutfunc, layout_userdata, &plan, pman, sman->info, Comm);
        /* determine for each rank how many particles have to be shifted to other ranks */
        domain_check_iter_space(&plan);
        /* Inside domain_exchange_once, we do send/recv for each processor individually and loop until we are done.*/
        failure = domain_exchange_once(&plan, pman, sman, 123000, Comm);
    }
#ifdef DEBUG
    /* This does not apply for the FOF code, where the exchange list is pre-assigned
     * and we only get one iteration. */
    if(!failure && maxiter > 1) {
        ExchangePlan plan9 = domain_init_exchangeplan(Comm);
        /* Do not drift again*/
        domain_build_exchange_list(layoutfunc, layout_userdata, &plan9, pman, sman, Comm);
        if(plan9.nexchange > 0)
            endrun(5, "Still have %ld particles in exchange list\n", plan9.nexchange);
        domain_free_exchangeplan(&plan9);
    }
#endif
    domain_free_exchangeplan(&plan);
    return failure;
}

/* Move some particle and slot data into an exchange buffer for sending.
   The format is:
   (base particle data 1 - n)
   (slot data for particle i: may be variable size as not necessarily ordered by type)
   This routine is not parallel.
 */
static int
exchange_pack_buffer(char * exch, int task, ExchangePlan * plan, struct part_manager_type * pman, struct slots_manager_type * sman)
{
    char * slotexch = exch + plan->toGo[task].base * sizeof(struct particle_data);
    char * partexch = exch;
    int64_t copybase = {0};
    int64_t copyslots[6] = {0};
    size_t n;
    for(n = 0; n < plan->nexchange; n++)
    {
        const int i = plan->ExchangeList[n];
        /* preparing for export */
        const int target = plan->layouts[n].target;
        if(target != task)
            continue;
        /* watch out thread unsafe */
        memcpy(partexch, pman->Base+i, sizeof(struct particle_data));
        partexch += sizeof(struct particle_data);
        copybase++;
        const int type = pman->Base[i].Type;
        copyslots[type]++;
        if(sman->info[type].enabled) {
            size_t elsize = sman->info[type].elsize;
            memcpy(slotexch,(char*) sman->info[type].ptr + pman->Base[i].PI * elsize, elsize);
            slotexch += elsize;
        }
        /* mark the particle for removal. Both secondary and base slots will be marked. */
        slots_mark_garbage(i, pman, sman);
    }
    if(copybase != plan->toGo[task].base)
        endrun(3, "Copied %ld particles for send to task %d but expected %ld\n", copybase, task, plan->toGo[task].base);
    for(n = 0; n < 6; n++)
        if(copyslots[n]!= plan->toGo[task].slots[n])
            endrun(3, "Copied %ld slots of type %ld for send to task %d but expected %ld\n", copyslots[n], n, task, plan->toGo[task].slots[n]);
    walltime_measure("/Domain/exchange/makebuf");
    return 0;
}

/* Take a received buffer and move the particle data back into the particle table.
 * If possible a garbage particle of the same type will be used for the new memory,
 * This routine is not parallel.
*/
static int
exchange_unpack_buffer(char * exch, int task, ExchangePlan * plan, struct part_manager_type * pman, struct slots_manager_type * sman)
{
    char * slotexch = exch + plan->toGet[task].base * sizeof(struct particle_data);
    char * partexch = exch;
    int64_t copybase = {0};
    int64_t copyslots[6] = {0};
    int64_t i;
    for(i = 0; i < plan->toGet[task].base; i++)
    {
        /* Extract type*/
        const unsigned int type = ((struct particle_data *) partexch)->Type;
        int PI = sman->info[type].size;
        /* Find a destination place in the particle table*/
        int64_t dest = pman->NumPart;
        size_t n;
        for(n = 0; n < plan->nexchange; n++) {
            if(plan->layouts[n].ptype != type)
                continue;
            const int d = plan->ExchangeList[n];
            if(pman->Base[d].IsGarbage) {
                dest = d;
                /* Copy PI so it is not over-written*/
                PI = pman->Base[dest].PI;
                break;
            }
        }
        /* watch out thread unsafe */
        /* If we are copying to the end of the table, increment the counter*/
        if(dest == pman->NumPart) {
            pman->NumPart++;
            if(pman->NumPart > pman->MaxPart)
                endrun(6, "Not enough room for particles after exchange\n");
        }
        memcpy(pman->Base+dest, partexch, sizeof(struct particle_data));

        partexch += sizeof(struct particle_data);
        copybase++;
        /* Copy the slot if needed*/
        if(sman->info[type].enabled) {
            /* Enforce that we have enough slots*/
            if(PI == sman->info[type].size) {
                sman->info[type].size++;
                if(sman->info[type].size >= sman->info[type].maxsize) {
                    int ptype;
                    int64_t newSlots[6] = {0};
                    for(ptype = 0; ptype < 6; ptype ++) {
                        if(!sman->info[ptype].enabled) continue;
                        newSlots[ptype] = sman->info[ptype].size + plan->toGet[task].slots[ptype] - copyslots[type];
                    }
                    /* This will likely fail here because memory ordering restrictions:
                    * need to do it before buffer allocation or alloc buffers high.*/
                    slots_reserve(1, newSlots, sman);
                }
            }
            size_t elsize = sman->info[type].elsize;
            memcpy((char*) sman->info[type].ptr + PI * elsize, slotexch, elsize);
            slotexch += elsize;
            /* Update the PI to be correct*/
            pman->Base[dest].PI = PI;
#ifdef DEBUG
            if(BASESLOT_PI(PI, type, sman)->ID != pman->Base[dest].ID) {
                endrun(1, "Exchange: P[%ld].ID = %ld (type %d) != SLOT ID = %ld. garbage: %d ReverseLink: %d\n",dest,pman->Base[dest].ID, pman->Base[dest].Type, BASESLOT_PI(PI, type, sman)->ID, pman->Base[dest].IsGarbage, BASESLOT_PI(PI, type, sman)->ReverseLink);
            }
#endif
        }
        copyslots[type]++;
    }
    if(copybase != plan->toGet[task].base)
        endrun(3, "Copied %ld particles received from task %d but expected %ld\n", copybase, task, plan->toGet[task].base);
    for(i = 0; i < 6; i++)
        if(copyslots[i]!= plan->toGet[task].slots[i])
            endrun(3, "Copied %ld slots of type %ld received from task %d but expected %ld\n", copyslots[i], i, task, plan->toGet[task].slots[i]);
    walltime_measure("/Domain/exchange/unpack");
    return 0;
}

static int domain_exchange_once(ExchangePlan * plan, struct part_manager_type * pman, struct slots_manager_type * sman, int tag, MPI_Comm Comm)
{
    /* First post receives*/
    struct CommBuffer recvs;
    alloc_commbuffer(&recvs, plan->NTask, 0);
    recvs.databuf = mymalloc("recvbuffer",plan->total_togetbytes * sizeof(char));

    size_t displs = 0;
    int task;
    for(task = 0; task < plan->NTask-1; task++) {
        const int recvtask = (plan->ThisTask - task + plan->NTask - 1) % plan->NTask;
        MPI_Irecv(recvs.databuf + displs, plan->toGet[recvtask].totalbytes, MPI_BYTE, recvtask, tag, Comm, &recvs.rdata_all[task]);
        recvs.rqst_task[task] = recvtask;
        recvs.displs[task] = displs;
        displs += plan->toGet[recvtask].totalbytes;
        recvs.nrequest_all ++;
    }
    if(displs != plan->total_togetbytes)
        endrun(3, "Posted receives for %lu bytes expected %lu bytes.\n", displs, plan->total_togetbytes);

    /* Now post sends: note that the sends are done in reverse order to the receives.
     * This ensures that partial sends and receives can complete early.*/
    struct CommBuffer sends;
    alloc_commbuffer(&sends, plan->NTask, 0);
    sends.databuf = mymalloc("sendbuffer",plan->total_togobytes * sizeof(char));

    displs = 0;
    for(task = 0; task < plan->NTask-1; task++) {
        const int sendtask = (plan->ThisTask + task + 1) % plan->NTask;
        /* Move the data into the buffer*/
        exchange_pack_buffer(sends.databuf + displs, sendtask, plan, pman, sman);
        MPI_Isend(sends.databuf + displs, plan->toGo[sendtask].totalbytes, MPI_BYTE, sendtask, tag, Comm, &sends.rdata_all[task]);
        sends.rqst_task[task] = sendtask;
        sends.displs[task] = displs;
        displs += plan->toGo[sendtask].totalbytes;
        sends.nrequest_all ++;
    }
    if(displs != plan->total_togobytes)
        endrun(3, "Packed %lu bytes for sending but expected %lu bytes.\n", displs, plan->total_togobytes);

    /* Now wait for and unpack the receives as they arrive.*/
    int * completed = ta_malloc("completes", int, recvs.nrequest_all);
    memset(completed, 0, recvs.nrequest_all * sizeof(int) );
    int totcomplete = 0;
    // message(3, "reqs: %d\n", recvs.nrequest_all);
    /* Test each request in turn until it completes*/
    do {
        for(task = 0; task < recvs.nrequest_all; task++) {
            /* If we already completed, no need to test again*/
            if(completed[task])
                continue;
            /* Check for a completed request: note that cleanup is performed if the request is complete.*/
            MPI_Test(&recvs.rdata_all[task], completed+task, MPI_STATUS_IGNORE);
            // message(3, "complete : %d task %d\n", completed[task], recvs.rqst_task[task]);

            /* Try the next one*/
            if (!completed[task])
                continue;
            totcomplete++;
            exchange_unpack_buffer(recvs.databuf+recvs.displs[task], recvs.rqst_task[task], plan, pman, sman);
        }
    } while(totcomplete < recvs.nrequest_all);
    myfree(completed);

    /* Now wait for the sends before we free the buffers*/
    MPI_Waitall(sends.nrequest_all, sends.rdata_all, MPI_STATUSES_IGNORE);
    /* Free the buffers*/
    free_commbuffer(&sends);
    free_commbuffer(&recvs);

    myfree(plan->layouts);

    walltime_measure("/Domain/exchange/alltoall");

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

/*Find how many particles we can transfer in current exchange iteration. TODO: Split requests that need too much.*/
static int
domain_check_iter_space(ExchangePlan * plan)
{
    size_t nlimit = mymalloc_freebytes();

    if (nlimit <  4096L * 6 + plan->NTask * 2 * sizeof(MPI_Request))
        endrun(1, "Not enough memory free to store requests!\n");

    nlimit -= 4096 * 2L + plan->NTask * 2 * sizeof(MPI_Request);

    /* Save some memory for memory headers and wasted space at the end of each allocation.
     * Need max. 2*4096 for each heap-allocated array.*/
    nlimit -= 4096 * 4L;

    /* We want to avoid doing an alltoall with
     * more than 2GB of material as this hangs.*/
    const size_t maxexch = 2040L*1024L*1024L;
    if(nlimit > maxexch)
        nlimit = maxexch;

    message(0, "Using %td bytes for exchange.\n", nlimit);

    /* Maximum size we need for a single send/recv pair*/
    size_t maxsize = 0;
    /* Total size needed for all send/recv pairs*/
    size_t totalsize = 0;
    int ntasks_exchange = 0;
    int task;
    for(task = 1; task < plan->NTask; task++) {
        const int sendtask = (plan->ThisTask + task) % plan->NTask;
        const int recvtask = (plan->ThisTask - task + plan->NTask) % plan->NTask;
        size_t singleiter = plan->toGo[sendtask].totalbytes + plan->toGet[recvtask].totalbytes;
        plan->total_togetbytes += plan->toGet[recvtask].totalbytes;
        plan->total_togobytes += plan->toGo[sendtask].totalbytes;
        if(singleiter > maxsize)
            maxsize = singleiter;
        totalsize += singleiter;
        if(totalsize < nlimit)
            ntasks_exchange ++;
    }

    if(maxsize > nlimit) {
        endrun(5, "Maxsize %ld > %ld limit. Not enough space for a single pair exchange. FIXME: This should be handled.\n", maxsize, nlimit);
    }

    if(ntasks_exchange < plan->NTask-1) {
        endrun(5, "Only enough space for %d tasks of %d to exchange. FIXME: This should be handled.\n", ntasks_exchange, plan->NTask-1);
    }

    return ntasks_exchange;
}

/*This function populates the toGo and toGet arrays*/
static void
domain_build_plan(ExchangeLayoutFunc layoutfunc, const void * layout_userdata, ExchangePlan * plan, struct part_manager_type * pman, struct slot_info * sinfo, MPI_Comm Comm)
{
    int ptype;
    size_t n;

    memset(plan->toGo, 0, sizeof(plan->toGo[0]) * plan->NTask);

    plan->layouts = (ExchangePartCache *) mymalloc("layoutcache",sizeof(ExchangePartCache) * plan->nexchange);

    #pragma omp parallel for
    for(n = 0; n < plan->nexchange; n++)
    {
        const int i = plan->ExchangeList[n];
        const int target = layoutfunc(i, layout_userdata);
        plan->layouts[n].ptype = pman->Base[i].Type;
        plan->layouts[n].target = target;
        if(target >= plan->NTask || target < 0)
            endrun(4, "layoutfunc for %d returned unreasonable %d for %d tasks\n", i, target, plan->NTask);
    }

    /*Do the sum*/
    for(n = 0; n < plan->nexchange; n++)
    {
        plan->toGo[plan->layouts[n].target].base++;
        plan->toGo[plan->layouts[n].target].slots[plan->layouts[n].ptype]++;
        /* Compute total size being sent on this process*/
        plan->toGo[plan->layouts[n].target].totalbytes += sizeof(struct particle_data);
        if(sinfo[plan->layouts[n].ptype].enabled){
            plan->toGo[plan->layouts[n].target].totalbytes += sinfo[plan->layouts[n].ptype].elsize;
        }
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
    message(0, "Total particles in flight: %ld Largest togo: %ld, toget %ld\n", sumtogo, maxbasetogomax, maxbasetogetmax);
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

