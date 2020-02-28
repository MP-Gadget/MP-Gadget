#include <mpi.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <omp.h>

#include "utils.h"

#include "treewalk.h"
#include "partmanager.h"
#include "domain.h"
#include "forcetree.h"

#include <signal.h>
#define BREAKPOINT raise(SIGTRAP)

#define FACT1 0.366025403785	/* FACT1 = 0.5 * (sqrt(3)-1) */

static int *Ngblist;
static int *Exportflag;    /*!< Buffer used for flagging whether a particle needs to be exported to another process */
static int *Exportnodecount;
static int *Exportindex;
static int *Send_offset, *Send_count, *Recv_count, *Recv_offset;

/*!< Memory factor to leave for (N imported particles) > (N exported particles). */
static int ImportBufferBoost;

static struct data_nodelist
{
    int NodeList[NODELISTLENGTH];
}
*DataNodeList;

/*!< the particles to be exported are grouped
by task-number. This table allows the
results to be disentangled again and to be
assigned to the correct particle */
struct data_index
{
    int Task;
    int Index;
    int IndexGet;
};

static struct data_index *DataIndexTable;	/*!< the particles to be exported are grouped
					   by task-number. This table allows the
					   results to be disentangled again and to be
					   assigned to the correct particle */

/*Initialise global treewalk parameters*/
void set_treewalk_params(ParameterSet * ps)
{
    int ThisTask;
    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
    if(ThisTask == 0)
        ImportBufferBoost = param_get_int(ps, "ImportBufferBoost");
    MPI_Bcast(&ImportBufferBoost, 1, MPI_INT, 0, MPI_COMM_WORLD);
}

static void ev_init_thread(TreeWalk * tw, LocalTreeWalk * lv);
static void ev_begin(TreeWalk * tw, int * active_set, const int size);
static void ev_finish(TreeWalk * tw);
static int ev_primary(TreeWalk * tw);
static void ev_get_remote(TreeWalk * tw);
static void ev_secondary(TreeWalk * tw);
static void ev_reduce_result(TreeWalk * tw);
static int ev_ndone(TreeWalk * tw);

static void
treewalk_build_queue(TreeWalk * tw, int * active_set, const int size, int may_have_garbage);

static int
ngb_treefind_threads(TreeWalkQueryBase * I,
        TreeWalkResultBase * O,
        TreeWalkNgbIterBase * iter,
        int startnode,
        LocalTreeWalk * lv);


/*! This function is used as a comparison kernel in a sort routine. It is
 *  used to group particles in the communication buffer that are going to
 *  be sent to the same CPU.
 */
static int data_index_compare(const void *a, const void *b)
{
    if(((struct data_index *) a)->Task < (((struct data_index *) b)->Task))
        return -1;

    if(((struct data_index *) a)->Task > (((struct data_index *) b)->Task))
        return +1;

    if(((struct data_index *) a)->Index < (((struct data_index *) b)->Index))
        return -1;

    if(((struct data_index *) a)->Index > (((struct data_index *) b)->Index))
        return +1;

    if(((struct data_index *) a)->IndexGet < (((struct data_index *) b)->IndexGet))
        return -1;

    if(((struct data_index *) a)->IndexGet > (((struct data_index *) b)->IndexGet))
        return +1;

    return 0;
}


/*
 * for debugging
 */
#define WATCH { \
        printf("tw->WorkSet[0] = %d (%d) %s:%d\n", tw->WorkSet ? tw->WorkSet[0] : 0, tw->WorkSetSize, __FILE__, __LINE__); \
    }
static TreeWalk * GDB_current_ev = NULL;

static void
ev_init_thread(TreeWalk * const tw, LocalTreeWalk * lv)
{
    const int thread_id = omp_get_thread_num();
    const int NTask = tw->NTask;
    int j;
    lv->tw = tw;
    lv->exportflag = Exportflag + thread_id * NTask;
    lv->exportnodecount = Exportnodecount + thread_id * NTask;
    lv->exportindex = Exportindex + thread_id * NTask;
    lv->Ninteractions = 0;
    lv->Nnodesinlist = 0;
    lv->Nlist = 0;
    lv->ngblist = Ngblist + thread_id * PartManager->NumPart;
    for(j = 0; j < NTask; j++)
        lv->exportflag[j] = -1;
}

static void
ev_alloc_threadlocals(const int NTaskTimesThreads)
{
    Exportflag = (int *) ta_malloc2("Exportthreads", int, 3*NTaskTimesThreads);
    Exportindex = Exportflag + NTaskTimesThreads;
    Exportnodecount = Exportflag + 2*NTaskTimesThreads;
}

static void
ev_free_threadlocals()
{
    ta_free(Exportflag);
}

static void
ev_begin(TreeWalk * tw, int * active_set, const int size)
{
    const int NumThreads = omp_get_max_threads();
    MPI_Comm_size(MPI_COMM_WORLD, &tw->NTask);
    tw->NThread = NumThreads;
    /* The last argument is may_have_garbage: in practice the only
     * trivial haswork is the gravtree, which has no (active) garbage because
     * the active list was just rebuilt. If we ever add a trivial haswork after
     * sfr/bh we should change this*/
    treewalk_build_queue(tw, active_set, size, 0);

    Ngblist = (int*) mymalloc("Ngblist", PartManager->NumPart * NumThreads * sizeof(int));

    report_memory_usage(tw->ev_label);

    /*The amount of memory eventually allocated per tree buffer*/
    size_t bytesperbuffer = sizeof(struct data_index) + sizeof(struct data_nodelist) + tw->query_type_elsize;
    /*This memory scales like the number of imports. In principle this could be much larger than Nexport
     * if the tree is very imbalanced and many processors all need to export to this one. In practice I have
     * not seen this happen, but provide a parameter to boost the memory for Nimport just in case.*/
    bytesperbuffer += ImportBufferBoost * (tw->query_type_elsize + tw->result_type_elsize);
    /*Use all free bytes for the tree buffer, as in exchange. Leave some free memory for array overhead.*/
    size_t freebytes = mymalloc_freebytes();
    if(freebytes <= 4096 * 11 * bytesperbuffer) {
        endrun(1231245, "Not enough memory for exporting any particles: needed %d bytes have %d. \n", bytesperbuffer, freebytes-4096*10);
    }
    freebytes -= 4096 * 10 * bytesperbuffer;
    /* if freebytes is greater than 2GB some MPIs have issues */
    if(freebytes > 1024 * 1024 * 2030) freebytes =  1024 * 1024 * 2030;

    tw->BunchSize = (int64_t) floor(((double)freebytes)/ bytesperbuffer);
    DataIndexTable =
        (struct data_index *) mymalloc("DataIndexTable", tw->BunchSize * sizeof(struct data_index));
    DataNodeList =
        (struct data_nodelist *) mymalloc("DataNodeList", tw->BunchSize * sizeof(struct data_nodelist));

#ifdef DEBUG
    memset(DataNodeList, -1, sizeof(struct data_nodelist) * tw->BunchSize);
#endif
    tw->currentIndex = ta_malloc("currentIndexPerThread", int,  NumThreads);
    tw->currentEnd = ta_malloc("currentEndPerThread", int, NumThreads);

    int i;
    for(i = 0; i < NumThreads; i ++) {
        tw->currentIndex[i] = ((size_t) i) * tw->WorkSetSize / NumThreads;
        tw->currentEnd[i] = ((size_t) i + 1) * tw->WorkSetSize / NumThreads;
    }
}

static void ev_finish(TreeWalk * tw)
{
    ta_free(tw->currentEnd);
    ta_free(tw->currentIndex);
    myfree(DataNodeList);
    myfree(DataIndexTable);
    myfree(Ngblist);
    if(!tw->work_set_stolen_from_active)
        myfree(tw->WorkSet);

}

int data_index_compare(const void *a, const void *b);

static void
treewalk_init_query(TreeWalk * tw, TreeWalkQueryBase * query, int i, int * NodeList)
{
    query->ID = P[i].ID;

    int d;
    for(d = 0; d < 3; d ++) {
        query->Pos[d] = P[i].Pos[d];
    }

    if(NodeList) {
        memcpy(query->NodeList, NodeList, sizeof(int) * NODELISTLENGTH);
    } else {
        query->NodeList[0] = tw->tree->firstnode; /* root node */
        query->NodeList[1] = -1; /* terminate immediately */
    }

    tw->fill(i, query, tw);
};

static void
treewalk_init_result(TreeWalk * tw, TreeWalkResultBase * result, TreeWalkQueryBase * query)
{
    memset(result, 0, tw->result_type_elsize);
    result->ID = query->ID;
}

static void
treewalk_reduce_result(TreeWalk * tw, TreeWalkResultBase * result, int i, enum TreeWalkReduceMode mode)
{
    if(tw->reduce != NULL)
        tw->reduce(i, result, mode, tw);
}

static void real_ev(TreeWalk * tw, int * ninter) {
    int tid = omp_get_thread_num();
    LocalTreeWalk lv[1];

    ev_init_thread(tw, lv);
    lv->mode = 0;

    /* Note: exportflag is local to each thread */
    int k;
    /* use old index to recover from a buffer overflow*/;
    TreeWalkQueryBase * input = alloca(tw->query_type_elsize);
    TreeWalkResultBase * output = alloca(tw->result_type_elsize);

    for(k = tw->currentIndex[tid];
        k < tw->currentEnd[tid];
        k++) {
        if(tw->BufferFullFlag) break;

        const int i = tw->WorkSet ? tw->WorkSet[k] : k;
        /* Primary never uses node list */
        treewalk_init_query(tw, input, i, NULL);
        treewalk_init_result(tw, output, input);

        lv->target = i;
        const int rt = tw->visit(input, output, lv);

        if(rt < 0) {
            break; /* export buffer has filled up, redo this particle */
        } else {
            treewalk_reduce_result(tw, output, i, TREEWALK_PRIMARY);
        }
    }
    tw->currentIndex[tid] = k;
    *ninter += lv->Ninteractions;
}

#if 0
static int
cmpint(const void *a, const void *b)
{
    const int * aa = (const int *) a;
    const int * bb = (const int *) b;
    if(aa < bb) return -1;
    if(aa > bb) return 1;
    return 0;

}
#endif

static void
treewalk_build_queue(TreeWalk * tw, int * active_set, const int size, int may_have_garbage) {
    int i;

    if(!tw->haswork && !may_have_garbage)
    {
        tw->WorkSetSize = size;
        tw->WorkSet = active_set;
        tw->work_set_stolen_from_active = 1;
        return;
    }

    /* Since we use a static schedule below we only need size / tw->NThread elements per thread.
     * Add 2 for non-integer parts.*/
    int tsize = size / tw->NThread + 2;
    /*Watch out: tw->WorkSet may change a few lines later due to the realloc*/
    tw->WorkSet = mymalloc("ActiveQueue", tsize * sizeof(int) * tw->NThread);
    tw->work_set_stolen_from_active = 0;

    int nqueue = 0;

    /*We want a lockless algorithm which preserves the ordering of the particle list.*/
    size_t *nqthr = ta_malloc("nqthr", size_t, tw->NThread);
    int **thrqueue = ta_malloc("thrqueue", int *, tw->NThread);

    gadget_setup_thread_arrays(tw->WorkSet, thrqueue, nqthr, tsize, tw->NThread);

    /* We enforce schedule static to ensure that each thread executes on contiguous particles.*/
    #pragma omp parallel for schedule(static)
    for(i=0; i < size; i++)
    {
        const int tid = omp_get_thread_num();
        /*Use raw particle number if active_set is null, otherwise use active_set*/
        const int p_i = active_set ? active_set[i] : i;

        /* Skip the garbage particles */
        if(P[p_i].IsGarbage)
            continue;

        if(tw->haswork && !tw->haswork(p_i, tw))
            continue;
        thrqueue[tid][nqthr[tid]] = p_i;
        nqthr[tid]++;
    }
    /*Merge step for the queue.*/
    nqueue = gadget_compact_thread_arrays(tw->WorkSet, thrqueue, nqthr, tw->NThread);
    ta_free(thrqueue);
    ta_free(nqthr);
    /*Shrink memory*/
    tw->WorkSet = myrealloc(tw->WorkSet, sizeof(int) * nqueue);

#if 0
    /* check the uniqueness of the active_set list. This is very slow. */
    qsort_openmp(tw->WorkSet, nqueue, sizeof(int), cmpint);
    for(i = 0; i < nqueue - 1; i ++) {
        if(tw->WorkSet[i] == tw->WorkSet[i+1]) {
            endrun(8829, "A few particles are twicely active.\n");
        }
    }
#endif
    tw->WorkSetSize = nqueue;
}

/* returns number of exports */
static int ev_primary(TreeWalk * tw)
{
    const int NTask = tw->NTask;
    double tstart, tend;
    tw->BufferFullFlag = 0;
    tw->Nexport = 0;

    int i;
    tstart = second();

    ev_alloc_threadlocals(tw->NTask * tw->NThread);

    int nint = tw->Ninteractions;
#pragma omp parallel reduction(+: nint)
    {
        real_ev(tw, &nint);
    }
    tw->Ninteractions = nint;

    ev_free_threadlocals();

    /* Nexport may go off too much after BunchSize
     * as we don't protect it from over adding in _export_particle
     * */
    if(tw->Nexport > tw->BunchSize)
        tw->Nexport = tw->BunchSize;

    tend = second();
    tw->timecomp1 += timediff(tstart, tend);

    qsort_openmp(DataIndexTable, tw->Nexport, sizeof(struct data_index), data_index_compare);

    /* adjust Nexport to skip the allocated but unused ones due to threads */
    while (tw->Nexport > 0 && DataIndexTable[tw->Nexport - 1].Task == NTask) {
        tw->Nexport --;
    }

    if(tw->BufferFullFlag) {
        message(1, "Tree export buffer full with %d particles. This is not fatal but slows the treewalk. Increase free memory during treewalk if possible.\n", tw->Nexport);
    }

    if(tw->Nexport == 0 && tw->BufferFullFlag) {
        endrun(1231245, "Buffer too small for even one particle. For example, there are too many nodes");
    }

    Send_count = (int *) ta_malloc("Send_count", int, 4*NTask);
    Recv_count = Send_count + NTask;
    Send_offset = Send_count + 2*NTask;
    Recv_offset = Send_count + 3*NTask;
    /*
     * fill the communication layouts,
     * here we reuse the legacy global variable names;
     * really should move them to local variables for the evaluator.
     * */
    memset(Send_count, 0, sizeof(int)*NTask);
    for(i = 0; i < tw->Nexport; i++) {
        Send_count[DataIndexTable[i].Task]++;
    }

    tstart = second();
    MPI_Alltoall(Send_count, 1, MPI_INT, Recv_count, 1, MPI_INT, MPI_COMM_WORLD);
    tend = second();
    tw->timewait1 += timediff(tstart, tend);

    for(i = 0, tw->Nimport = 0, Recv_offset[0] = 0, Send_offset[0] = 0; i < NTask; i++)
    {
        tw->Nimport += Recv_count[i];

        if(i > 0)
        {
            Send_offset[i] = Send_offset[i - 1] + Send_count[i - 1];
            Recv_offset[i] = Recv_offset[i - 1] + Recv_count[i - 1];
        }
    }

    return tw->Nexport;
}

static int ev_ndone(TreeWalk * tw)
{
    int ndone;
    double tstart, tend;
    tstart = second();
    int done = 1;
    int i;
    for(i = 0; i < tw->NThread; i ++) {
        if(tw->currentIndex[i] < tw->currentEnd[i]) {
            done = 0;
            break;
        }
    }
    MPI_Allreduce(&done, &ndone, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    tend = second();
    tw->timewait2 += timediff(tstart, tend);
    return ndone;

}

static void ev_secondary(TreeWalk * tw)
{
    double tstart, tend;

    tstart = second();
    tw->dataresult = mymalloc("EvDataResult", tw->Nimport * tw->result_type_elsize);

    ev_alloc_threadlocals(tw->NTask * tw->NThread);
    int nint = tw->Ninteractions;
    int nnodes = tw->Nnodesinlist;
    int nlist = tw->Nlist;
#pragma omp parallel reduction(+: nint) reduction(+: nnodes) reduction(+: nlist)
    {
        int j;
        LocalTreeWalk lv[1];

        ev_init_thread(tw, lv);
        lv->mode = 1;
#pragma omp for
        for(j = 0; j < tw->Nimport; j++) {
            TreeWalkQueryBase * input = (TreeWalkQueryBase*) (tw->dataget + j * tw->query_type_elsize);
            TreeWalkResultBase * output = (TreeWalkResultBase*)(tw->dataresult + j * tw->result_type_elsize);
            treewalk_init_result(tw, output, input);
            lv->target = -1;
            tw->visit(input, output, lv);
        }
        nint += lv->Ninteractions;
        nnodes += lv->Nnodesinlist;
        nlist += lv->Nlist;
    }
    tw->Ninteractions = nint;
    tw->Nnodesinlist = nnodes;
    tw->Nlist = nlist;

    ev_free_threadlocals();
    tend = second();
    tw->timecomp2 += timediff(tstart, tend);
}

/* export a particle at target and no, thread safely
 *
 * This can also be called from a nonthreaded code
 *
 * */
int treewalk_export_particle(LocalTreeWalk * lv, int no) {
    if(lv->mode != 0) {
        endrun(1, "Trying to export a ghost particle.\n");
    }
    const int target = lv->target;
    int *exportflag = lv->exportflag;
    int *exportnodecount = lv->exportnodecount;
    int *exportindex = lv->exportindex;
    TreeWalk * tw = lv->tw;

    const int task = tw->tree->TopLeaves[no - tw->tree->lastnode].Task;

    if(exportflag[task] != target)
    {
        exportflag[task] = target;
        exportnodecount[task] = NODELISTLENGTH;
    }

    if(exportnodecount[task] == NODELISTLENGTH)
    {
        const int nexp = atomic_fetch_and_add(&tw->Nexport, 1);

        /* out of buffer space. Need to discard work for this particle and interrupt */
        if(nexp >= tw->BunchSize) {
            tw->BufferFullFlag = 1;
            /* This reduces the time until the other threads see the buffer is full and the loop can exit.
             * Since it is a pure optimization, no need for a full atomic.*/
            #pragma omp flush (tw)
            /* Touch up the DataIndexTable, so that exports associated with the current particle
             * won't be exported. This is expensive but rare. */
            int i;
            for(i=0; i < tw->BunchSize; i++) {
                /* target is the current particle, so this reads the buffer looking for
                 * exports associated with the current particle. We cannot just discard
                 * from the end because of threading.*/
                if(DataIndexTable[i].Index == target)
                {
                    /* NTask will be placed to the end by sorting */
                    DataIndexTable[i].Task = tw->NTask;
                    /* put in some junk so that we can detect them */
                    DataNodeList[DataIndexTable[i].IndexGet].NodeList[0] = -2;
                }
            }
            return -1;
        }
        else {
            exportnodecount[task] = 0;
            exportindex[task] = nexp;
            DataIndexTable[nexp].Task = task;
            DataIndexTable[nexp].Index = target;
            DataIndexTable[nexp].IndexGet = nexp;
        }
    }

    /* Set the NodeList entry*/
    DataNodeList[exportindex[task]].NodeList[exportnodecount[task]++] =
            tw->tree->TopLeaves[no - tw->tree->lastnode].treenode;

    if(exportnodecount[task] < NODELISTLENGTH)
            DataNodeList[exportindex[task]].NodeList[exportnodecount[task]] = -1;
    return 0;
}

/* run a treewalk on an active_set.
 *
 * active_set : a list of indices of particles. If active_set is NULL,
 *              all (NumPart) particles are used.
 *
 * */
void
treewalk_run(TreeWalk * tw, int * active_set, int size)
{
    if(!force_tree_allocated(tw->tree)) {
        endrun(0, "Tree has been freed before this treewalk.\n");
    }

    GDB_current_ev = tw;

    ev_begin(tw, active_set, size);

    if(tw->preprocess) {
        int i;
        #pragma omp parallel for
        for(i = 0; i < tw->WorkSetSize; i ++) {
            const int p_i = tw->WorkSet ? tw->WorkSet[i] : i;
            tw->preprocess(p_i, tw);
        }
    }

    if(tw->visit) {
        do
        {
            ev_primary(tw); /* do local particles and prepare export list */
            /* exchange particle data */
            ev_get_remote(tw);
            /* now do the particles that were sent to us */
            ev_secondary(tw);

            /* import the result to local particles */
            ev_reduce_result(tw);

            tw->Niterations ++;
            tw->Nexport_sum += tw->Nexport;
            ta_free(Send_count);
        } while(ev_ndone(tw) < tw->NTask);
    }

#ifdef DEBUG
    /*int64_t totNodesinlist, totlist;
    MPI_Reduce(&tw->Nnodesinlist,  &totNodesinlist, 1, MPI_INT64, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&tw->Nlist,  &totlist, 1, MPI_INT64, MPI_SUM, 0, MPI_COMM_WORLD);
    message(0, "Nodes in nodelist: %g (avg). %ld nodes, %ld lists\n", ((double) totNodesinlist)/totlist, totlist, totNodesinlist);*/
#endif
    double tstart, tend;

    tstart = second();

    if(tw->postprocess) {
        int i;
        #pragma omp parallel for
        for(i = 0; i < tw->WorkSetSize; i ++) {
            const int p_i = tw->WorkSet ? tw->WorkSet[i] : i;
            tw->postprocess(p_i, tw);
        }
    }
    tend = second();
    tw->timecomp3 = timediff(tstart, tend);
    ev_finish(tw);
}

static void
ev_communicate(void * sendbuf, void * recvbuf, size_t elsize, int import) {
    /* if import is 1, import the results from neigbhours */
    MPI_Datatype type;
    MPI_Type_contiguous(elsize, MPI_BYTE, &type);
    MPI_Type_commit(&type);

    if(import) {
        MPI_Alltoallv_sparse(
                sendbuf, Recv_count, Recv_offset, type,
                recvbuf, Send_count, Send_offset, type, MPI_COMM_WORLD);
    } else {
        MPI_Alltoallv_sparse(
                sendbuf, Send_count, Send_offset, type,
                recvbuf, Recv_count, Recv_offset, type, MPI_COMM_WORLD);
    }
    MPI_Type_free(&type);
}

/* returns the remote particles */
static void ev_get_remote(TreeWalk * tw)
{
    int j;
    double tstart, tend;

    void * recvbuf = mymalloc("EvDataGet", tw->Nimport * tw->query_type_elsize);
    char * sendbuf = mymalloc("EvDataIn", tw->Nexport * tw->query_type_elsize);

#ifdef DEBUG
    memset(sendbuf, -1, tw->Nexport * tw->query_type_elsize);
#endif

    tstart = second();
    /* prepare particle data for export */
    //
#pragma omp parallel for
    for(j = 0; j < tw->Nexport; j++)
    {
        int place = DataIndexTable[j].Index;
        TreeWalkQueryBase * input = (TreeWalkQueryBase*) (sendbuf + j * tw->query_type_elsize);
        int * nodelist = DataNodeList[DataIndexTable[j].IndexGet].NodeList;
        treewalk_init_query(tw, input, place, nodelist);
    }
    tend = second();
    tw->timecomp1 += timediff(tstart, tend);

    tstart = second();
    ev_communicate(sendbuf, recvbuf, tw->query_type_elsize, 0);
    tend = second();
    tw->timecommsumm1 += timediff(tstart, tend);
    myfree(sendbuf);
    tw->dataget = recvbuf;
}

static int data_index_compare_by_index(const void *a, const void *b)
{
    if(((struct data_index *) a)->Index < (((struct data_index *) b)->Index))
        return -1;

    if(((struct data_index *) a)->Index > (((struct data_index *) b)->Index))
        return +1;

    if(((struct data_index *) a)->IndexGet < (((struct data_index *) b)->IndexGet))
        return -1;

    if(((struct data_index *) a)->IndexGet > (((struct data_index *) b)->IndexGet))
        return +1;

    return 0;
}

static void ev_reduce_result(TreeWalk * tw)
{

    int j;
    double tstart, tend;

    const int Nexport = tw->Nexport;
    void * sendbuf = tw->dataresult;
    char * recvbuf = (char*) mymalloc("EvDataOut",
                Nexport * tw->result_type_elsize);

    tstart = second();
    ev_communicate(sendbuf, recvbuf, tw->result_type_elsize, 1);
    tend = second();
    tw->timecommsumm2 += timediff(tstart, tend);

    tstart = second();

    for(j = 0; j < Nexport; j++) {
        DataIndexTable[j].IndexGet = j;
    }

    /* mysort is a lie! */
    qsort_openmp(DataIndexTable, Nexport, sizeof(struct data_index), data_index_compare_by_index);

    int * UniqueOff = mymalloc("UniqueIndex", sizeof(int) * (Nexport + 1));
    UniqueOff[0] = 0;
    int Nunique = 0;

    for(j = 1; j < Nexport; j++) {
        if(DataIndexTable[j].Index != DataIndexTable[j-1].Index)
            UniqueOff[++Nunique] = j;
    }
    if(Nexport > 0)
        UniqueOff[++Nunique] = Nexport;

    if(tw->reduce != NULL) {
#pragma omp parallel for private(j) if(Nunique > 16)
        for(j = 0; j < Nunique; j++)
        {
            int k;
            int place = DataIndexTable[UniqueOff[j]].Index;
            int start = UniqueOff[j];
            int end = UniqueOff[j + 1];
            for(k = start; k < end; k++) {
                int get = DataIndexTable[k].IndexGet;
                TreeWalkResultBase * output = (TreeWalkResultBase*) (recvbuf + tw->result_type_elsize * get);
                treewalk_reduce_result(tw, output, place, TREEWALK_GHOSTS);
            }
        }
    }
    myfree(UniqueOff);
    tend = second();
    tw->timecomp1 += timediff(tstart, tend);
    myfree(recvbuf);
    myfree(tw->dataresult);
    myfree(tw->dataget);
}

#if 0
/*The below code is left in because it is a partial implementation of a useful optimisation:
 * the ability to restart the treewalk from a node other than the root node*/
struct ev_task {
    int top_node;
    int place;
} ;


static int ev_task_cmp_by_top_node(const void * p1, const void * p2) {
    const struct ev_task * t1 = p1, * t2 = p2;
    if(t1->top_node > t2->top_node) return 1;
    if(t1->top_node < t2->top_node) return -1;
    return 0;
}

static void fill_task_queue (TreeWalk * tw, struct ev_task * tq, int * pq, int length) {
    int i;
#pragma omp parallel for if(length > 1024)
    for(i = 0; i < length; i++) {
        int no = -1;
        /*
        if(0) {
            no = force_get_father(pq[i], tw->tree);
            while(no != -1) {
                if(tw->tree->Nodes[no].f.TopLevel) {
                    break;
                }
                no = tw->tree->Nodes[no].father;
            }
        }
       */
        tq[i].top_node = no;
        tq[i].place = pq[i];
    }
    // qsort_openmp(tq, length, sizeof(struct ev_task), ev_task_cmp_by_top_node);
}
#endif

/**********
 *
 * This particular TreeWalkVisitFunction that uses the nbgiter memeber of
 * The TreeWalk object to iterate over the neighbours of a Query.
 *
 * All Pairwise interactions are implemented this way.
 *
 * Note: Short range gravity is not based on pair enumeration.
 * We may want to port it over and see if gravtree.c receives any speed up.
 *
 * Required fields in TreeWalk: ngbiter, ngbiter_type_elsize.
 *
 * Before the iteration starts, ngbiter is called with iter->base.other == -1.
 * The callback function shall initialize the interator with Hsml, mask, and symmetric.
 *
 *****/
int treewalk_visit_ngbiter(TreeWalkQueryBase * I,
            TreeWalkResultBase * O,
            LocalTreeWalk * lv)
{

    TreeWalkNgbIterBase * iter = alloca(lv->tw->ngbiter_type_elsize);

    /* Kick-start the iteration with other == -1 */
    iter->other = -1;
    lv->tw->ngbiter(I, O, iter, lv);
    const double BoxSize = lv->tw->tree->BoxSize;

    int ninteractions = 0;
    int inode = 0;

    for(inode = 0; inode < NODELISTLENGTH && I->NodeList[inode] >= 0; inode++)
    {
        int numcand = ngb_treefind_threads(I, O, iter, I->NodeList[inode], lv);
        /* Export buffer is full end prematurally */
        if(numcand < 0) return numcand;

        /* If we are here, export is succesful. Work on the this particle -- first
         * filter out all of the candidates that are actually outside. */
        int numngb;

        for(numngb = 0; numngb < numcand; numngb ++) {
            int other = lv->ngblist[numngb];

            /* Skip garbage*/
            if(P[other].IsGarbage)
                continue;

            /* must be the correct type */
            if(!((1<<P[other].Type) & iter->mask))
                continue;

            /* must be the correct time bin */
            if(lv->tw->type == TREEWALK_SPLIT && !(BINMASK(P[other].TimeBin) & lv->tw->bgmask))
                continue;

            double dist;

            if(iter->symmetric == NGB_TREEFIND_SYMMETRIC) {
                dist = DMAX(P[other].Hsml, iter->Hsml);
            } else {
                dist = iter->Hsml;
            }

            double r2 = 0;
            int d;
            double h2 = dist * dist;
            for(d = 0; d < 3; d ++) {
                /* the distance vector points to 'other' */
                iter->dist[d] = NEAREST(I->Pos[d] - P[other].Pos[d], BoxSize);
                r2 += iter->dist[d] * iter->dist[d];
                if(r2 > h2) break;
            }
            if(r2 > h2) continue;

            /* update the iter and call the iteration function*/
            iter->r2 = r2;
            iter->r = sqrt(r2);
            iter->other = other;

            lv->tw->ngbiter(I, O, iter, lv);
        }

        ninteractions += numngb;
    }

    lv->Ninteractions += ninteractions;
    if(lv->mode == 1) {
        lv->Nnodesinlist += inode;
        lv->Nlist += 1;
    }
    return 0;
}

/**
 * Cull a node.
 *
 * Returns 1 if the node shall be opened;
 * Returns 0 if the node has no business with this query.
 */
static int
cull_node(const TreeWalkQueryBase * const I, const TreeWalkNgbIterBase * const iter, const struct NODE * const current, const double BoxSize)
{
    double dist;
    if(iter->symmetric == NGB_TREEFIND_SYMMETRIC) {
        dist = DMAX(current->mom.hmax, iter->Hsml) + 0.5 * current->len;
    } else {
        dist = iter->Hsml + 0.5 * current->len;
    }

    double r2 = 0;
    double dx = 0;
    /* do each direction */
    int d;
    for(d = 0; d < 3; d ++) {
        dx = NEAREST(current->center[d] - I->Pos[d], BoxSize);
        if(dx > dist) return 0;
        if(dx < -dist) return 0;
        r2 += dx * dx;
    }
    /* now test against the minimal sphere enclosing everything */
    dist += FACT1 * current->len;

    if(r2 > dist * dist) {
        return 0;
    }
    return 1;
}
/*****
 * This is the internal code that looks for particles in the ngb tree from
 * searchcenter upto hsml. if iter->symmetric is NGB_TREE_FIND_SYMMETRIC, then upto
 * max(P[other].Hsml, iter->Hsml).
 *
 * Particle that intersects with other domains are marked for export.
 * The hosting nodes (leaves of the global tree) are exported as well.
 *
 * For all 'other' particle within the neighbourhood and are local on this processor,
 * this function calls the ngbiter member of the TreeWalk object.
 * iter->base.other, iter->base.dist iter->base.r2, iter->base.r, are properly initialized.
 *
 * */
static int
ngb_treefind_threads(TreeWalkQueryBase * I,
        TreeWalkResultBase * O,
        TreeWalkNgbIterBase * iter,
        int startnode,
        LocalTreeWalk * lv)
{
    int no;
    int numcand = 0;

    const ForceTree * tree = lv->tw->tree;
    const double BoxSize = tree->BoxSize;

    no = startnode;

    while(no >= 0)
    {
        if(node_is_particle(no, tree)) {
            int fat = force_get_father(no, tree);
            endrun(12312, "Particles should be added before getting here! no = %d, father = %d (ptype = %d)\n", no, fat, tree->Nodes[fat].f.ChildType);
        }
        if(node_is_pseudo_particle(no, tree)) {
            int fat = force_get_father(no, tree);
            endrun(12312, "Pseudo-Particles should be added before getting here! no = %d, father = %d (ptype = %d)\n", no, fat, tree->Nodes[fat].f.ChildType);
        }

        struct NODE *current = &tree->Nodes[no];

        /* When walking exported particles we start from the encompassing top-level node,
         * so if we get back to a top-level node again we are done.*/
        if(lv->mode == 1) {
            /* The first node is always top-level*/
            if(current->f.TopLevel && no != startnode) {
                /* we reached a top-level node again, which means that we are done with the branch */
                break;
            }
        }

        /* Cull the node */
        if(0 == cull_node(I, iter, current, BoxSize)) {
            /* in case the node can be discarded */
            no = current->sibling;
            continue;
        }

        /* Node contains relevant particles, add them.*/
        if(current->f.ChildType == PARTICLE_NODE_TYPE) {
            int i;
            int * suns = current->s.suns;
            for (i = 0; i < current->s.noccupied; i++) {
                lv->ngblist[numcand++] = suns[i];
            }
            /* Move sideways*/
            no = current->sibling;
            continue;
        }
        else if(current->f.ChildType == PSEUDO_NODE_TYPE) {
            /* pseudo particle */
            if(lv->mode == 1) {
                endrun(12312, "Touching outside of my domain from a node list of a ghost. This shall not happen.");
            } else {
                /* Export the pseudo particle*/
                if(-1 == treewalk_export_particle(lv, current->nextnode))
                    return -1;
                /* Move sideways*/
                no = current->sibling;
                continue;
            }
        }
        /* ok, we need to open the node */
        no = current->nextnode;
    }

    return numcand;
}

