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

typedef struct {
    ExchangePlanEntry * toGo;
    ExchangePlanEntry * toGet;
    ExchangePlanEntry toGoSum;
    ExchangePlanEntry toGetSum;
    int NTask;
    int ThisTask;
    /*Number of garbage particles of each type*/
    int64_t ngarbage[6];
    /* List of entries in particle table which are garbage particles of each type*/
    int * garbage_list[6];
    /* Per-task list of particles to send.*/
    int ** target_list;
} ExchangePlan;

/* Structure to store the tasks that need to be exchanged this iteration*/
struct ExchangeIter
{
    /* Start and end of the tasks for this iteration. Sends start from the current task and move forwards.
     * Recvs start from before the current task and move backwards.*/
    int StartTask;
    /* Note that this task should not be sent so the condition is < for send and > for recv.*/
    int EndTask;
    /* If we only have one task this iteration, which particles do we send/recv?*/
    size_t StartPart;
    size_t EndPart;
    /* Total transfer this time*/
    size_t transferbytes;
};

/*
 *
 * exchange particles according to layoutfunc.
 * layoutfunc gives the target task of particle p.
*/
static int domain_exchange_once(ExchangePlan * plan, struct part_manager_type * pman, struct slots_manager_type * sman, int tag, const size_t maxexch, MPI_Comm Comm);
static int domain_build_plan(ExchangeLayoutFunc layoutfunc, const void * layout_userdata, ExchangePlan * plan, PreExchangeList * preplan, struct part_manager_type * pman, struct slots_manager_type * sman, MPI_Comm Comm);
static void domain_build_exchange_list(ExchangeLayoutFunc layoutfunc, const void * layout_userdata, PreExchangeList * preplan, struct part_manager_type * pman, struct slots_manager_type * sman, MPI_Comm Comm);

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
    plan.toGet = ta_malloc("toGet", ExchangePlanEntry, plan.NTask);
    return plan;
}

static void
domain_free_exchangeplan(ExchangePlan * plan)
{
    /* Free the lists.*/
    int n;
    if(plan->target_list) {
        for(n = plan->NTask -1; n >=0; n--) {
            if(plan->target_list[n])
                myfree(plan->target_list[n]);
        }
        myfree(plan->target_list);
        for(n = 5; n >=0; n--) {
            if(plan->garbage_list[n])
                myfree(plan->garbage_list[n]);
        }
    }
    myfree(plan->toGet);
    myfree(plan->toGo);
}

/* We want to avoid doing an alltoall with
    * more than 2GB of material as this hangs.*/
static size_t MaxExch = 2040L*1024L*1024L;
/* For tests*/
void
domain_set_max_exchange(const size_t maxexch)
{
    MaxExch = maxexch;
}

/* Init the exchange iteration. Note that for sends offset is +1 for recvs it is -1*/
void
domain_init_exchange_iter(struct ExchangeIter * iter, const int offset, const ExchangePlan * const plan)
{
    /* + plan->NTask ensures we get a positive number for offset = -1*/
    iter->StartTask = (plan->ThisTask + offset + plan->NTask) % plan->NTask;
    iter->EndTask = (plan->ThisTask + offset + plan->NTask) % plan->NTask;
    iter->StartPart = 0;
    iter->EndPart = 0;
}

/* Find the amount of space available for the domain exchange.
 * Collective so we can make the same decisions on all ranks.*/
size_t
domain_get_exchange_space(const int NTask, MPI_Comm Comm)
{
    size_t nlimit = mymalloc_freebytes();
    /* Save some memory for memory headers and wasted space at the end of each allocation.
     * Need max. 2*4096 for each heap-allocated array, max 1 send, 1 recv.*/
    nlimit -= 4096 * 4L * NTask;
    if(nlimit <= 0)
        endrun(1, "Not enough memory free to store requests!\n");

    if(nlimit > MaxExch)
        nlimit = MaxExch;
    MPI_Allreduce(MPI_IN_PLACE, &nlimit, 1, MPI_UINT64, MPI_MIN, Comm);
    return nlimit;
}

/*Plan and execute a domain exchange, also performing a garbage collection if requested*/
int domain_exchange(ExchangeLayoutFunc layoutfunc, const void * layout_userdata, PreExchangeList * preexch, struct part_manager_type * pman, struct slots_manager_type * sman, MPI_Comm Comm) {
    /* register the MPI types used in communication if not yet. */
    if (MPI_TYPE_PLAN_ENTRY == 0) {
        MPI_Type_contiguous(sizeof(ExchangePlanEntry), MPI_BYTE, &MPI_TYPE_PLAN_ENTRY);
        MPI_Type_commit(&MPI_TYPE_PLAN_ENTRY);
    }
    PreExchangeList preplan = {0};
    /* Use the pre-exchange list if we can*/
    if(!preexch || !preexch->ExchangeList) {
        preexch = &preplan;
        domain_build_exchange_list(layoutfunc, layout_userdata, preexch, pman, sman, Comm);
        walltime_measure("/Domain/exchange/exchangelist");
    }

    /*Structure for building a list of particles that will be exchanged*/
    ExchangePlan plan = domain_init_exchangeplan(Comm);
    int failure = domain_build_plan(layoutfunc, layout_userdata, &plan, preexch, pman, sman, Comm);
    /* Done with the pre-exchange list*/
    myfree(preexch->ExchangeList);
    walltime_measure("/Domain/exchange/plan");
    /* Do this after domain_build_plan so the target lists are already allocated*/
    size_t maxexch = domain_get_exchange_space(plan.NTask, Comm);
    /* Now to do an exchange if it will succeed*/
    if(!failure)
        failure = domain_exchange_once(&plan, pman, sman, 123000, maxexch, Comm);
    domain_free_exchangeplan(&plan);

#ifdef DEBUG
    domain_test_id_uniqueness(pman);
    slots_check_id_consistency(pman, sman);

    if(!failure) {
        PreExchangeList plan9 = {0};
        domain_build_exchange_list(layoutfunc, layout_userdata, &plan9, pman, sman, Comm);
        if(plan9.nexchange - plan9.ngarbage > 0)
            endrun(5, "Still have %ld particles in exchange list\n", plan9.nexchange - plan9.ngarbage);
        myfree(plan9.ExchangeList);
    }
    walltime_measure("/Domain/exchange/finalize");
#endif
    return failure;
}

/* Move some particle and slot data into an exchange buffer for sending.
   The format is:
   (base particle data 1 - n)
   (slot data for particle i: may be variable size as not necessarily ordered by type)
    Returns: size of buffer packed. Writes first not-packed particle to endpart
   */
static size_t
exchange_pack_buffer(char * exch, const int task, const size_t StartPart, ExchangePlan * const plan, struct part_manager_type * pman, struct slots_manager_type * sman, const size_t maxsendexch, size_t * endpart)
{
    char * exchptr = exch;
    int64_t copybase = 0;
    int64_t copyslots[6] = {0};
    size_t n;
    for(n = StartPart; n < plan->toGo[task].base; n++)
    {
        const int i = plan->target_list[task][n];
        /* preparing for export */
        const int type = pman->Base[i].Type;
        size_t elsize = 0;
        if(sman->info[type].enabled)
             elsize = sman->info[type].elsize;
        if(exchptr - exch + sizeof(struct particle_data) + elsize > maxsendexch) {
            *endpart = n;
            break;
        }
        memcpy(exchptr, pman->Base+i, sizeof(struct particle_data));
        exchptr += sizeof(struct particle_data);
        if(sman->info[type].enabled) {
            memcpy(exchptr,(char*) sman->info[type].ptr + pman->Base[i].PI * elsize, elsize);
        }
        exchptr += elsize;
        copybase++;
        copyslots[type]++;
        /* Add this particle to the garbage list so we can unpack something else into it.*/
        int64_t gslot = plan->ngarbage[type];
        plan->garbage_list[type][gslot] = i;
        plan->ngarbage[type]++;
        /* mark the particle for removal. Both secondary and base slots will be marked. */
        slots_mark_garbage(i, pman, sman);
    }
    if(copybase != plan->toGo[task].base)
        endrun(3, "Copied %ld particles for send to task %d but expected %ld\n", copybase, task, plan->toGo[task].base);
    for(n = 0; n < 6; n++)
        if(copyslots[n]!= plan->toGo[task].slots[n])
            endrun(3, "Copied %ld slots of type %ld for send to task %d but expected %ld\n", copyslots[n], n, task, plan->toGo[task].slots[n]);
    return exchptr - exch;
}

/* Take a received buffer and move the particle data back into the particle table.
 * If possible a garbage particle of the same type will be used for the new memory,
 * This routine is not parallel.
*/
static int
exchange_unpack_buffer(char * exch, int task, ExchangePlan * plan, struct part_manager_type * pman, struct slots_manager_type * sman)
{
    char * exchptr = exch;
    int64_t copybase = 0;
    int64_t copyslots[6] = {0};

    int64_t i;
    for(i = 0; i < plan->toGet[task].base; i++)
    {
        /* Extract type*/
        const unsigned int type = ((struct particle_data *) exchptr)->Type;
        int PI = sman->info[type].size;
        /* Find a garbage place in the particle table*/
        int64_t dest = pman->NumPart;
        if(plan->ngarbage[type] >= 1) {
            dest = plan->garbage_list[type][plan->ngarbage[type]-1];
            /* Copy PI so it is not over-written*/
            PI = pman->Base[dest].PI;
            /* No longer garbage!*/
            plan->ngarbage[type]--;
        }
        /* If we are copying to the end of the table, increment the counter*/
        else{
            dest = pman->NumPart;
            pman->NumPart++;
            if(pman->NumPart > pman->MaxPart)
                endrun(6, "Not enough room for particles after exchange\n");
        }
        memcpy(pman->Base+dest, exchptr, sizeof(struct particle_data));

        exchptr += sizeof(struct particle_data);
        copybase++;
        /* Copy the slot if needed*/
        if(sman->info[type].enabled) {
            /* Enforce that we have enough slots*/
            if(PI == sman->info[type].size) {
                sman->info[type].size++;
                if(sman->info[type].size >= sman->info[type].maxsize)
                    endrun(6, "Not enough room for slot %d after exchange\n",type);
            }
            size_t elsize = sman->info[type].elsize;
            memcpy((char*) sman->info[type].ptr + PI * elsize, exchptr, elsize);
            exchptr += elsize;
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
    return 0;
}


/*Find how many tasks we can send in current exchange iteration.
 If the current task does not fit in the buffer, work out how many particles can be sent. */
static void
domain_find_send_iter(ExchangePlan * plan, struct ExchangeIter * senditer,  size_t * expected_freeslots, const size_t maxsendexch)
{
    /* Last loop was the final part of a task, need to reset and do this again*/
    if(senditer->EndPart > 0 && senditer->EndPart == plan->toGo[senditer->StartTask].base) {
        senditer->EndPart = 0;
        senditer->StartPart = 0;
        senditer->EndTask ++;
    }
    /* Last loop was a subtask*/
    if(senditer->EndPart > 0) {
        senditer->StartPart = senditer->EndPart;
        return;
    }
    senditer->StartTask = senditer->EndTask;
    /* Total bytes needed to send*/
    senditer->transferbytes = 0;
    int n;
    for(n = 0 ; n < 6; n++)
        expected_freeslots[n] = plan->ngarbage[n];
    /* First find some data to send. This gives us space in the particle table.*/
    int task;
    for(task = 0; task < plan->NTask; task++) {
        const int sendtask = (senditer->StartTask + task) % plan->NTask;
        senditer->EndTask = sendtask;
        if(senditer->transferbytes + plan->toGo[sendtask].totalbytes > maxsendexch || sendtask == plan->ThisTask)
            break;
        // message(1, "toget %ld tot %ld recv %d\n", plan->toGet[recvtask].totalbytes, senditer->togetbytes, recvtask);
        senditer->transferbytes += plan->toGo[sendtask].totalbytes;
        for(n = 0; n < 6; n++)
            expected_freeslots[n] += plan->toGo[sendtask].slots[n];
    }
    message(1, "Using %ld bytes to send from %d to %d\n", senditer->transferbytes, senditer->StartTask, senditer->EndTask);
    if(senditer->StartTask == senditer->EndTask) {
        senditer->transferbytes = maxsendexch;
    }
    return;
}

/*Find how many tasks we can transfer in current exchange iteration.
 If the current task does not fit in the buffer, work out how many */
static void
domain_find_recv_iter(ExchangePlan * plan, struct ExchangeIter * recviter, size_t freepart, size_t * expected_freeslots, const size_t maxrecvexch)
{
    /* Last loop was the final part of a task, need to reset and do this again*/
    if(recviter->EndPart > 0 && recviter->EndPart == plan->toGo[recviter->StartTask].base) {
        recviter->EndPart = 0;
        recviter->StartPart = 0;
        recviter->EndTask ++;
    }
    /* Last loop was a subtask*/
    if(recviter->EndPart > 0) {
        recviter->StartPart = recviter->EndPart;
        return;
    }
    recviter->StartTask = recviter->EndTask;
    recviter->transferbytes = 0;
    int task;
    for(task = 0; task < plan->NTask; task++) {
        const int recvtask = (recviter->StartTask - task + plan->NTask) % plan->NTask;
        recviter->EndTask = recvtask;
        /*This checks we have enough space in the particle table for each slot*/
        int n;
        for(n = 0; n < 6; n++) {
            expected_freeslots[n] -= plan->toGet[recvtask].slots[n];
            /* If we overflow the slots available in garbage, we use the global particle table.*/
            if(expected_freeslots[n] < 0) {
                freepart += expected_freeslots[n];
                expected_freeslots[n] = 0;
            }
        }
        /* We do not have enough slots here to continue!
         * This logic means that we may not receive all the things sent to us this iteration,
         * as different ranks may have different particle loads.*/
        if(freepart < 0)
            break;
        if(recviter->transferbytes + plan->toGet[recvtask].totalbytes > maxrecvexch || recvtask == plan->ThisTask)
            break;
        recviter->transferbytes += plan->toGet[recvtask].totalbytes;
        // message(1, "toget %ld tot %ld recv %d\n", plan->toGet[recvtask].totalbytes, recviter->togetbytes, recvtask);
    }
    message(1, "Using %ld bytes to recv from %d to %d\n", recviter->transferbytes, recviter->StartTask, recviter->EndTask);
    if(recviter->StartTask == recviter->EndTask) {
        recviter->transferbytes = maxrecvexch;
    }

    return;
}

/* Post a single receive for the current iteration task when there is not enough buffer space for one task. */
static struct CommBuffer
domain_post_single_recv(ExchangePlan * plan, struct ExchangeIter *thisiter, int tag, MPI_Comm Comm)
{
    /* First post receives*/
    struct CommBuffer recvs;
    alloc_commbuffer(&recvs, 1, 0);
    /* Note that IRecv will happily accept less bytes than are requested.*/
    recvs.databuf = mymalloc("recvbuffer", thisiter->transferbytes * sizeof(char));
    MPI_Irecv(recvs.databuf, thisiter->transferbytes, MPI_BYTE, thisiter->StartTask, tag, Comm, recvs.rdata_all);
    recvs.rqst_task[0] = thisiter->StartTask;
    recvs.displs[0] = 0;
    recvs.nrequest_all=1;
    return recvs;
}

/* Post the receives for the current iteration, and fill up a CommBuffer with Irecvs.
 * Note that receives are posted in reverse order! First the task before this one, then all tasks before that, backwards.
 * The sends are posted forwards, so that all send/recv pairs are ordered.*/
static struct CommBuffer
domain_post_recvs(ExchangePlan * plan, struct ExchangeIter *recviter, int tag, MPI_Comm Comm)
{
    /* First post receives*/
    struct CommBuffer recvs;
    alloc_commbuffer(&recvs, plan->NTask, 0);
    recvs.databuf = mymalloc("recvbuffer",recviter->transferbytes * sizeof(char));

    size_t displs = 0;
    int task;
    for(task=0; task < plan->NTask; task++) {
        const int recvtask = (recviter->StartTask - task + plan->NTask) % plan->NTask;
        if(recvtask == recviter->EndTask)
            break;
        /* Skip zero-size receives*/
        if(plan->toGet[recvtask].totalbytes == 0)
            continue;
        MPI_Irecv(recvs.databuf + displs, plan->toGet[recvtask].totalbytes, MPI_BYTE, recvtask, tag, Comm, &recvs.rdata_all[task]);
        recvs.rqst_task[task] = recvtask;
        recvs.displs[task] = displs;
        displs += plan->toGet[recvtask].totalbytes;
        // message(1, "exch toget %ld tot %ld recv %d\n", plan->toGet[recvtask].totalbytes, thisiter.togetbytes, recvtask);
        recvs.nrequest_all ++;
    }
    if(displs != recviter->transferbytes)
        endrun(3, "Posted receives for %lu bytes expected %lu bytes.\n", displs, recviter->transferbytes);
    return recvs;
}

/* Pack a single send buffer when we don't have space for all the particles on one task.*/
static struct CommBuffer
domain_pack_single_send(ExchangePlan * plan, struct ExchangeIter *senditer, struct part_manager_type * pman, struct slots_manager_type * sman, double *tpack, int tag, MPI_Comm Comm)
{
    struct CommBuffer sends = {0};
    alloc_commbuffer(&sends, 1, 0);
    sends.databuf = mymalloc("sendbuffer", senditer->transferbytes * sizeof(char));
    /* Move the data into the buffer*/
    /* The openmp parallel is done inside exchange_pack_buffer so that we can issue MPI_Isend as soon as possible*/
    double tstart = second();
    size_t packed_bytes = exchange_pack_buffer(sends.databuf, senditer->StartTask, senditer->StartPart, plan, pman, sman, senditer->transferbytes, &senditer->EndPart);
    double tend = second();
    *tpack += timediff(tstart, tend);
    MPI_Isend(sends.databuf, packed_bytes, MPI_BYTE, senditer->StartTask, tag, Comm, sends.rdata_all);
    sends.rqst_task[0] = senditer->StartTask;
    sends.displs[0] = 0;
    sends.nrequest_all = 1;
    return sends;
}

/* Pack the send buffers for each task and issue ISend requests for them*/
static struct CommBuffer
domain_pack_sends(ExchangePlan * plan, struct ExchangeIter *senditer, struct part_manager_type * pman, struct slots_manager_type * sman, double *tpack, int tag, MPI_Comm Comm)
{
    struct CommBuffer sends = {0};
    alloc_commbuffer(&sends, plan->NTask, 0);
    sends.databuf = mymalloc("sendbuffer",senditer->transferbytes * sizeof(char));
    size_t displs = 0;
    int task;
    for(task=0; task < plan->NTask; task++) {
        /* Move the data into the buffer*/
        const int sendtask = (senditer->StartTask + task) % plan->NTask;
        if(sendtask == senditer->EndTask)
            break;
        /* Skip zero-size sends*/
        if(plan->toGo[sendtask].totalbytes == 0)
            continue;
        /* The openmp parallel is done inside exchange_pack_buffer so that we can issue MPI_Isend as soon as possible*/
        double tstart = second();
        exchange_pack_buffer(sends.databuf + displs, sendtask, 0, plan, pman, sman, senditer->transferbytes, &senditer->EndPart);
        if(senditer->EndPart > 0)
            endrun(4, "Expected %ld particles but only packed %lu\n", plan->toGo[sendtask].base, senditer->EndPart);
        double tend = second();
        *tpack += timediff(tstart, tend);
        MPI_Isend(sends.databuf + displs, plan->toGo[sendtask].totalbytes, MPI_BYTE, sendtask, tag, Comm, &sends.rdata_all[task]);
        sends.rqst_task[task] = sendtask;
        sends.displs[task] = displs;
        displs += plan->toGo[sendtask].totalbytes;
        sends.nrequest_all ++;
    }
    if(displs != senditer->transferbytes)
        endrun(3, "Packed %lu bytes for sending but expected %lu bytes.\n", displs, senditer->transferbytes);
    return sends;
}

/* Wait for and unpack the received data into a buffer. We busy-loop until they are done.*/
static void
domain_wait_unpack_recv(ExchangePlan * plan, struct part_manager_type * pman, struct slots_manager_type * sman, struct CommBuffer * recvs, double *tunpack)
{
    /* Now wait for and unpack the receives as they arrive.*/
    int * completed = ta_malloc("completes", int, recvs->nrequest_all);
    memset(completed, 0, recvs->nrequest_all * sizeof(int) );
    int totcomplete = 0, task;
    // message(3, "reqs: %d\n", recvs.nrequest_all);
    /* Test each request in turn until it completes*/
    do {
        for(task = 0; task < recvs->nrequest_all; task++) {
            /* If we already completed, no need to test again*/
            if(completed[task])
                continue;
            /* Check for a completed request: note that cleanup is performed if the request is complete.*/
            MPI_Test(&recvs->rdata_all[task], completed+task, MPI_STATUS_IGNORE);
            // message(3, "complete : %d task %d\n", completed[task], recvs->rqst_task[task]);

            /* Try the next one*/
            if (!completed[task])
                continue;
            totcomplete++;
            double tstart2 = second();
            exchange_unpack_buffer(recvs->databuf+recvs->displs[task], recvs->rqst_task[task], plan, pman, sman);
            double tend2 = second();
            *tunpack += timediff(tstart2, tend2);
        }
    } while(totcomplete < recvs->nrequest_all);
    myfree(completed);
}

static int domain_exchange_once(ExchangePlan * plan, struct part_manager_type * pman, struct slots_manager_type * sman, int tag, const size_t maxexch, MPI_Comm Comm)
{
    double tpack = 0, tunpack = 0, twait = 0, twait2 = 0;
    /* determine for each rank how many particles have to be shifted to other ranks */
    struct ExchangeIter senditer = {0}, recviter = {0};
    domain_init_exchange_iter(&senditer, 1, plan);
    domain_init_exchange_iter(&recviter, -1, plan);
    do {
        size_t expected_freeslots[6];
        domain_find_send_iter(plan, &senditer,  expected_freeslots, maxexch/2);
        domain_find_recv_iter(plan, &recviter, pman->MaxPart - pman->NumPart, expected_freeslots, maxexch/2);
        /* First post receives*/
        struct CommBuffer recvs;
        /* Receiving less than one task!*/
        if(recviter.StartTask == recviter.EndTask)
            recvs = domain_post_single_recv(plan, &recviter, tag, Comm);
        else
            recvs = domain_post_recvs(plan, &recviter, tag, Comm);
        /* Now post sends: note that the sends are done in reverse order to the receives.
         * This ensures that partial sends and receives can complete early.*/
        struct CommBuffer sends;
        if(senditer.StartTask == senditer.EndTask)
            sends = domain_pack_single_send(plan, &senditer, pman, sman, &tpack, tag, Comm);
        else
            sends = domain_pack_sends(plan, &senditer, pman, sman, &tpack, tag, Comm);
        /* Now wait for and unpack the receives as they arrive.*/
        double tstart = second();
        domain_wait_unpack_recv(plan, pman, sman, &recvs, &tunpack);
        double tend = second();
        twait2 += timediff(tstart, tend);
        /* Now wait for the sends before we free the buffers*/
        tstart = second();
        MPI_Waitall(sends.nrequest_all, sends.rdata_all, MPI_STATUSES_IGNORE);
        tend = second();
        twait += timediff(tstart, tend);
        /* Free the buffers*/
        free_commbuffer(&sends);
        free_commbuffer(&recvs);
    } while(recviter.EndTask != plan->ThisTask || senditer.EndTask != plan->ThisTask );

    double timeall = walltime_measure(WALLTIME_IGNORE);
    walltime_add("/Domain/exchange/misc", timeall - (tpack + twait + twait2));
    walltime_add("/Domain/exchange/pack", tpack);
    walltime_add("/Domain/exchange/unpack", tunpack);
    walltime_add("/Domain/exchange/wait2", twait2- tunpack);
    walltime_add("/Domain/exchange/wait1", twait);
    return 0;
}

/* This function builds the list of particles to be exchanged.
 * All particles are processed every time, space is not considered.
 * The exchange list needs to be rebuilt every time gc is run. */
static void
domain_build_exchange_list(ExchangeLayoutFunc layoutfunc, const void * layout_userdata, PreExchangeList * preplan, struct part_manager_type * pman, struct slots_manager_type * sman, MPI_Comm Comm)
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
                threx_local[nexthr_local] = i;
                nexthr_local++;
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
    preplan->ngarbage = ngarbage;
    /*Merge step for the queue.*/
    preplan->nexchange = gadget_compact_thread_arrays(&preplan->ExchangeList, &gthread);
    /*Shrink memory*/
    preplan->ExchangeList = (int *) myrealloc(preplan->ExchangeList, sizeof(int) * preplan->nexchange);
}

/*This function populates the various lists for the domain plan.
 *These are: toGo and toGet arrays (counts).
 * target_list arrays (positions of particles to send).
 * garbage_list array (positions of garbage particles by type).*/
static int
domain_build_plan(ExchangeLayoutFunc layoutfunc, const void * layout_userdata, ExchangePlan * plan, PreExchangeList * preplan, struct part_manager_type * pman, struct slots_manager_type * sman, MPI_Comm Comm)
{
    int ptype;
    size_t n;

    memset(plan->toGo, 0, sizeof(plan->toGo[0]) * plan->NTask);

    int * layouts = (int *) mymalloc2("layoutcache",sizeof(int) * preplan->nexchange);
    int * tmp_garbage_list = mymalloc2("tmp_garbage",sizeof(int)* 6 * preplan->ngarbage);
    ExchangePlanEntry * toGoThread = (ExchangePlanEntry *) mymalloc2("toGoThread",sizeof(ExchangePlanEntry) * plan->NTask* omp_get_max_threads());
    memset(toGoThread, 0, sizeof(toGoThread[0]) * plan->NTask * omp_get_max_threads());
    memset(plan->ngarbage, 0, 6*sizeof(plan->ngarbage[0]));

    /* Compute exchange particle counts for each process*/
    #pragma omp parallel for
    for(n = 0; n < preplan->nexchange; n++)
    {
        const int tid = omp_get_thread_num();
        const int i = preplan->ExchangeList[n];
        const int ptype = pman->Base[i].Type;
        /* Add this to the proto-garbage list*/
        if(pman->Base[i].IsGarbage) {
            int gslot = atomic_fetch_and_add_64(&plan->ngarbage[ptype], 1);
            tmp_garbage_list[ptype * preplan->ngarbage + gslot] = i;
            layouts[n] = plan->ThisTask;
            continue;
        }
        const int target = layoutfunc(i, layout_userdata);
        if(target >= plan->NTask || target < 0)
            endrun(4, "layoutfunc for %d returned unreasonable %d for %d tasks\n", i, target, plan->NTask);
        toGoThread[tid * plan->NTask + target ].base++;
        toGoThread[tid * plan->NTask + target].slots[ptype]++;
        /* Compute total size being sent on this process*/
        toGoThread[tid * plan->NTask + target].totalbytes += sizeof(struct particle_data);
        if(sman->info[ptype].enabled){
            toGoThread[tid * plan->NTask + target].totalbytes += sman->info[ptype].elsize;
        }
        layouts[n] = target;
    }

    /*Do the sum*/
    for(n = 0; n < omp_get_max_threads(); n++)
    {
        int target;
        #pragma omp parallel for
        for(target = 0; target < plan->NTask; target++) {
            plan->toGo[target].base += toGoThread[n * plan->NTask + target].base;
            int i;
            for(i = 0; i < 6; i++)
                plan->toGo[target].slots[i] += toGoThread[n * plan->NTask + target].slots[i];
            plan->toGo[target].totalbytes += toGoThread[n * plan->NTask + target].totalbytes;
        }
    }
    myfree(toGoThread);

    /* Get the send counts from every processor*/
    MPI_Alltoall(plan->toGo, 1, MPI_TYPE_PLAN_ENTRY, plan->toGet, 1, MPI_TYPE_PLAN_ENTRY, Comm);
    memcpy(&plan->toGoSum, &plan->toGo[0], sizeof(plan->toGoSum));
    memcpy(&plan->toGetSum, &plan->toGet[0], sizeof(plan->toGetSum));

    int rank;
    int64_t maxbasetoget=-1, maxbasetogo=-1;
    for(rank = 1; rank < plan->NTask; rank ++) {
        /* Direct assignment breaks compilers like icc */
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

    int64_t finalNumPart = pman->NumPart;
    /* finalNumPart calculation is not completely optimal but reflects the current algorithm.
     * The achievable balance could be a little better as we can make extra slots without extra particles,
     * but then the extra particles would not be in type order.*/
    int64_t slots_needed[6] = {0};
    int need_slot_reserve = 0;
    for(n = 0; n < 6; n++) {
        finalNumPart += plan->toGetSum.slots[n]- plan->toGoSum.slots[n]- plan->ngarbage[n];
        if(sman->info[n].enabled) {
            slots_needed[n] = sman->info[n].size + plan->toGetSum.slots[n]- plan->toGoSum.slots[n]- plan->ngarbage[n];
            if(slots_needed[n] >= sman->info[n].maxsize) {
                need_slot_reserve = 1;
            }
        }
    }
    /* Make sure we have reserved enough slots to fit the new particles into.*/
    if(need_slot_reserve) {
        slots_reserve(1, slots_needed, sman);
    }

    /* Garbage lists with enough space for those from this exchange and those already present.*/
    for(n = 0; n < 6; n++) {
        if(plan->toGoSum.slots[n] + plan->ngarbage[n] == 0) {
            plan->garbage_list[n] = NULL;
            continue;
        }
        plan->garbage_list[n] = (int *) mymalloc("garbage",sizeof(int) * (plan->toGoSum.slots[n] + plan->ngarbage[n]));
        /* Copy over the existing garbage*/
        memcpy(plan->garbage_list[n], &tmp_garbage_list[n * preplan->ngarbage], sizeof(plan->garbage_list[n][0])* plan->ngarbage[n]);
    }
    myfree(tmp_garbage_list);

    /* Transpose to per-target lists*/
    plan->target_list = (int **) ta_malloc("target_list",int*, plan->NTask);
    int target;
    for(target = 0; target < plan->NTask; target++) {
        if(plan->toGo[target].base == 0) {
            plan->target_list[target] = NULL;
            continue;
        }
        plan->target_list[target] = mymalloc("exchangelist",plan->toGo[target].base * sizeof(int));
    }
    size_t * counts = ta_malloc("counts", size_t, plan->NTask);
    memset(counts, 0, sizeof(counts[0]) * plan->NTask);
    for(n = 0; n < preplan->nexchange; n++) {
        const int target = layouts[n];
        /* This is garbage*/
        if(target == plan->ThisTask)
            continue;
        plan->target_list[target][counts[target]] = preplan->ExchangeList[n];
        counts[target]++;
    }
    for(target = 0; target < plan->NTask; target++) {
        if(counts[target] != plan->toGo[target].base)
            endrun(1, "Expected %lu in target list for task %d from plan but got %lu\n", counts[target], target, plan->toGo[target].base);
    }

    myfree(counts);
    myfree(layouts);

    int failure = 0;
    if(finalNumPart > pman->MaxPart) {
        message(1, "Exchange will not succeed, need %ld particle slots but only have %ld. Current NumPart %ld\n", finalNumPart, pman->MaxPart, pman->NumPart);
        failure = 1;
    }

    /* Make failure collective*/
    failure = MPI_Allreduce(MPI_IN_PLACE, &failure, 1, MPI_INT, MPI_SUM, Comm);

    return failure;
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

