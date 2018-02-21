#include <mpi.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>

#include "utils.h"

#include "allvars.h"
#include "partmanager.h"
#include "treewalk.h"
#include "drift.h"
#include "domain.h"
#include "forcetree.h"
#include "timestep.h"

#define FACT1 0.366025403785	/* FACT1 = 0.5 * (sqrt(3)-1) */

struct ev_task {
    int top_node;
    int place;
} ;

static int *Ngblist;
static int *Exportflag;    /*!< Buffer used for flagging whether a particle needs to be exported to another process */
static int *Exportnodecount;
static int *Exportindex;
static int *Send_offset, *Send_count, *Recv_count, *Recv_offset;

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

static void ev_init_thread(TreeWalk * tw, LocalTreeWalk * lv);
static void fill_task_queue (TreeWalk * tw, struct ev_task * tq, int * pq, int length);
static void ev_begin(TreeWalk * tw, int * active_set, int size);
static void ev_finish(TreeWalk * tw);
static int ev_primary(TreeWalk * tw);
static void ev_get_remote(TreeWalk * tw);
static void ev_secondary(TreeWalk * tw);
static void ev_reduce_result(TreeWalk * tw);
static int ev_ndone(TreeWalk * tw);

static void
treewalk_build_queue(TreeWalk * tw, int * active_set, int size);

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
        printf("tw->PrimaryTasks[0] = %d %d (%d) %s:%d\n", tw->PrimaryTasks[0].top_node, tw->PrimaryTasks[0].place, tw->WorkSetSize, __FILE__, __LINE__); \
    }
static TreeWalk * GDB_current_ev = NULL;

static void
ev_init_thread(TreeWalk * tw, LocalTreeWalk * lv)
{
    int thread_id = omp_get_thread_num();
    int j;
    lv->tw = tw;
    lv->exportflag = Exportflag + thread_id * NTask;
    lv->exportnodecount = Exportnodecount + thread_id * NTask;
    lv->exportindex = Exportindex + thread_id * NTask;
    lv->Ninteractions = 0;
    lv->Nnodesinlist = 0;
    lv->ngblist = Ngblist + thread_id * PartManager->NumPart;
    for(j = 0; j < NTask; j++)
        lv->exportflag[j] = -1;
}

static void
ev_alloc_threadlocals()
{
    int NTaskTimesThreads = All.NumThreads * NTask;

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
ev_begin(TreeWalk * tw, int * active_set, int size)
{
    Ngblist = (int*) mymalloc("Ngblist", PartManager->NumPart * All.NumThreads * sizeof(int));
    tw->BunchSize =
        (int) ((All.BufferSize * 1024 * 1024) / (sizeof(struct data_index) + 
                    sizeof(struct data_nodelist) + tw->query_type_elsize + tw->result_type_elsize));
    DataIndexTable =
        (struct data_index *) mymalloc("DataIndexTable", tw->BunchSize * sizeof(struct data_index));
    DataNodeList =
        (struct data_nodelist *) mymalloc("DataNodeList", tw->BunchSize * sizeof(struct data_nodelist));

    memset(DataNodeList, -1, sizeof(struct data_nodelist) * tw->BunchSize);

    tw->WorkSetSize = 0;

    tw->WorkSet = mymalloc("ActiveQueue", PartManager->NumPart * sizeof(int));

    treewalk_build_queue(tw, active_set, size);

    tw->PrimaryTasks = (struct ev_task *) mymalloc("PrimaryTasks", sizeof(struct ev_task) * tw->WorkSetSize);

    fill_task_queue(tw, tw->PrimaryTasks, tw->WorkSet, tw->WorkSetSize);
    tw->currentIndex = mymalloc("currentIndexPerThread", sizeof(int) * All.NumThreads);
    tw->currentEnd = mymalloc("currentEndPerThread", sizeof(int) * All.NumThreads);

    int i;
    for(i = 0; i < All.NumThreads; i ++) {
        tw->currentIndex[i] = ((size_t) i) * tw->WorkSetSize / All.NumThreads;
        tw->currentEnd[i] = ((size_t) i + 1) * tw->WorkSetSize / All.NumThreads;
    }
}

static void ev_finish(TreeWalk * tw)
{
    myfree(tw->currentEnd);
    myfree(tw->currentIndex);
    myfree(tw->PrimaryTasks);
    myfree(tw->WorkSet);
    myfree(DataNodeList);
    myfree(DataIndexTable);
    myfree(Ngblist);
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
        query->NodeList[0] = RootNode; /* root node */
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

static void real_ev(TreeWalk * tw) {
    int tid = omp_get_thread_num();
    int i;
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

        i = tw->PrimaryTasks[k].place;

        if(P[i].Evaluated) {
            BREAKPOINT;
        }
        if(!tw->haswork(i, tw)) {
            BREAKPOINT;
        }
        int rt;
        /* Primary never uses node list */
        treewalk_init_query(tw, input, i, NULL);
        treewalk_init_result(tw, output, input);

        lv->target = i;
        rt = tw->visit(input, output, lv);

        if(rt < 0) {
            P[i].Evaluated = 0;
            break; /* export buffer has filled up, redo this particle */
        } else {
            P[i].Evaluated = 1;
            treewalk_reduce_result(tw, output, i, TREEWALK_PRIMARY);
        }
    }
    tw->currentIndex[tid] = k;
#pragma omp atomic
    tw->Ninteractions += lv->Ninteractions;
#pragma omp atomic
    tw->Nnodesinlist += lv->Nnodesinlist;
}

#ifdef DEBUG
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
treewalk_build_queue(TreeWalk * tw, int * active_set, int size) {
    int * queue = tw->WorkSet;
    int k = 0;
    if(active_set == NULL) {
        int i;
        #pragma omp parallel for
        for(i = 0; i < PartManager->NumPart; i++) {
            /* Skip the garbage particles */
            if(P[i].IsGarbage) continue;
            if(!tw->haswork(i, tw))
                continue;
            const int lock = atomic_fetch_and_add(&k, 1);
            queue[lock] = i;
        }
    } else {
        int i;
        #pragma omp parallel for
        for(i=0; i < NumActiveParticle; i++)
        {
            const int p_i = ActiveParticle[i];

            /* Skip the garbage particles */
            if(P[p_i].IsGarbage) continue;

            if(!tw->haswork(p_i, tw))
               continue;
            const int lock = atomic_fetch_and_add(&k, 1);
            queue[lock] = p_i;
        }
#ifdef DEBUG
        /* check the uniqueness of ActiveParticle list. */
        /* FIXME: the sort may affect performance of treewalk */
        qsort_openmp(queue, k, sizeof(int), cmpint);
        for(i = 0; i < k - 1; i ++) {
            if(queue[i] == queue[i+1]) {
                endrun(8829, "A few particles are twicely active.");
            }
        }
#endif
    }
    tw->WorkSetSize = k;
}

/* returns number of exports */
static int ev_primary(TreeWalk * tw)
{
    double tstart, tend;
    tw->BufferFullFlag = 0;
    tw->Nexport = 0;

    int i;
    tstart = second();

 #pragma omp parallel for if(tw->BunchSize > 1024) 
    for(i = 0; i < tw->BunchSize; i ++) {
        DataIndexTable[i].Task = NTask;
        /*entries with NTask is not filled with particles, and will be
         * sorted to the end */
    }

    ev_alloc_threadlocals();
#pragma omp parallel 
    {
        real_ev(tw);
    }
    ev_free_threadlocals();

    /* Nexport may go off too much after BunchSize 
     * as we don't protect it from over adding in _export_particle
     * */
    if(tw->Nexport > tw->BunchSize)
        tw->Nexport = tw->BunchSize;

    tend = second();
    tw->timecomp1 += timediff(tstart, tend);


    /* touching up the export list, remove incomplete particles */
#pragma omp parallel for if (tw->Nexport > 1024) 
    for(i = 0; i < tw->Nexport; i ++) {
        /* if the NodeList of the particle is incomplete due
         * to abandoned work, also do not export it */
        int place = DataIndexTable[i].Index;
        if(! P[place].Evaluated) {
            /* NTask will be placed to the end by sorting */
            DataIndexTable[i].Task = NTask; 
            /* put in some junk so that we can detect them */
            DataNodeList[DataIndexTable[i].IndexGet].NodeList[0] = -2;
        }
    }

    qsort_openmp(DataIndexTable, tw->Nexport, sizeof(struct data_index), data_index_compare);

    /* adjust Nexport to skip the allocated but unused ones due to threads */
    while (tw->Nexport > 0 && DataIndexTable[tw->Nexport - 1].Task == NTask) {
        tw->Nexport --;
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
    for(i = 0; i < NTask; i++)
        Send_count[i] = 0;
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
    for(i = 0; i < All.NumThreads; i ++) {
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

    ev_alloc_threadlocals();
#pragma omp parallel 
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
            if(!tw->UseNodeList) {
                if(input->NodeList[0] != RootNode) abort(); /* root node */
                if(input->NodeList[1] != -1) abort(); /* terminate immediately */
            }
            lv->target = -1;
            tw->visit(input, output, lv);
        }
#pragma omp atomic
        tw->Ninteractions += lv->Ninteractions;
#pragma omp atomic
        tw->Nnodesinlist += lv->Nnodesinlist;
    }
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
    int target = lv->target;
    int *exportflag = lv->exportflag;
    int *exportnodecount = lv->exportnodecount;
    int *exportindex = lv->exportindex; 
    TreeWalk * tw = lv->tw;
    int task;

    task = TopLeaves[no - (RootNode + MaxNodes)].Task;

    if(exportflag[task] != target)
    {
        exportflag[task] = target;
        exportnodecount[task] = NODELISTLENGTH;
    }

    if(exportnodecount[task] == NODELISTLENGTH)
    {
        int nexp;

        if(tw->Nexport < tw->BunchSize) {
            nexp = atomic_fetch_and_add(&tw->Nexport, 1);
        } else {
            nexp = tw->BunchSize;
        }

        if(nexp >= tw->BunchSize) {
            /* out of buffer space. Need to discard work for this particle and interrupt */
            tw->BufferFullFlag = 1;
#pragma omp flush
            return -1;
        }
        exportnodecount[task] = 0;
        exportindex[task] = nexp;
        DataIndexTable[nexp].Task = task;
        DataIndexTable[nexp].Index = target;
        DataIndexTable[nexp].IndexGet = nexp;
    }

    if(tw->UseNodeList) 
    {
        DataNodeList[exportindex[task]].NodeList[exportnodecount[task]++] =
            TopLeaves[no - (RootNode + MaxNodes)].treenode;

        if(exportnodecount[task] < NODELISTLENGTH)
            DataNodeList[exportindex[task]].NodeList[exportnodecount[task]] = -1;
    }
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
    if(!force_tree_allocated()) {
        endrun(0, "Tree has been freed before this treewalk.");
    }

    GDB_current_ev = tw;

    ev_begin(tw, active_set, size);

    if(tw->preprocess) {
        int i;
        #pragma omp parallel for if(tw->WorkSetSize > 64)
        for(i = 0; i < tw->WorkSetSize; i ++) {
            tw->preprocess(tw->WorkSet[i], tw);
        }
    }

    if(tw->visit) {
        do
        {
            ev_primary(tw); /* do local particles and prepare export list */
            /* exchange particle data */
            ev_get_remote(tw);
            report_memory_usage(tw->ev_label);
            /* now do the particles that were sent to us */
            ev_secondary(tw);

            /* import the result to local particles */
            ev_reduce_result(tw);

            tw->Niterations ++;
            tw->Nexport_sum += tw->Nexport;
            ta_free(Send_count);
        } while(ev_ndone(tw) < NTask);
    }

    double tstart, tend;

    tstart = second();

    if(tw->postprocess) {
        int i;
        #pragma omp parallel for if(tw->WorkSetSize > 64)
        for(i = 0; i < tw->WorkSetSize; i ++) {
            tw->postprocess(tw->WorkSet[i], tw);
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

    memset(sendbuf, -1, tw->Nexport * tw->query_type_elsize);

    tstart = second();
    /* prepare particle data for export */
    //
#pragma omp parallel for if (tw->Nexport > 128) 
    for(j = 0; j < tw->Nexport; j++)
    {
        int place = DataIndexTable[j].Index;
        TreeWalkQueryBase * input = (TreeWalkQueryBase*) (sendbuf + j * tw->query_type_elsize);
        int * nodelist = NULL;
        if(tw->UseNodeList) {
            nodelist = DataNodeList[DataIndexTable[j].IndexGet].NodeList;
        }
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

    void * sendbuf = tw->dataresult;
    char * recvbuf = (char*) mymalloc("EvDataOut", 
                tw->Nexport * tw->result_type_elsize);

    tstart = second();
    ev_communicate(sendbuf, recvbuf, tw->result_type_elsize, 1);
    tend = second();
    tw->timecommsumm2 += timediff(tstart, tend);

    tstart = second();

    for(j = 0; j < tw->Nexport; j++) {
        DataIndexTable[j].IndexGet = j;
    }

    /* mysort is a lie! */
    qsort_openmp(DataIndexTable, tw->Nexport, sizeof(struct data_index), data_index_compare_by_index);
    
    int * UniqueOff = mymalloc("UniqueIndex", sizeof(int) * (tw->Nexport + 1));
    UniqueOff[0] = 0;
    int Nunique = 0;

    for(j = 1; j < tw->Nexport; j++) {
        if(DataIndexTable[j].Index != DataIndexTable[j-1].Index)
            UniqueOff[++Nunique] = j;
    }
    if(tw->Nexport > 0)
        UniqueOff[++Nunique] = tw->Nexport;

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
static int ev_task_cmp_by_top_node(const void * p1, const void * p2) {
    const struct ev_task * t1 = p1, * t2 = p2;
    if(t1->top_node > t2->top_node) return 1;
    if(t1->top_node < t2->top_node) return -1;
    return 0;
}
#endif

static void fill_task_queue (TreeWalk * tw, struct ev_task * tq, int * pq, int length) {
    int i;
#pragma omp parallel for if(length > 1024)
    for(i = 0; i < length; i++) {
        int no = -1;
        /*
        if(0) {
            no = Father[pq[i]];
            while(no != -1) {
                if(Nodes[no].f.TopLevel) {
                    break;
                }
                no = Nodes[no].father;
            }
        }
       */
        tq[i].top_node = no;
        tq[i].place = pq[i];
        P[pq[i]].Evaluated = 0;
    }
    // qsort_openmp(tq, length, sizeof(struct ev_task), ev_task_cmp_by_top_node);
}

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

    int ninteractions = 0;
    int inode = 0;

    for(inode = 0; inode < NODELISTLENGTH && I->NodeList[inode] >= 0; inode++)
    {
        int startnode = Nodes[I->NodeList[inode]].u.d.nextnode;  /* open it */

        int numcand = ngb_treefind_threads(I, O, iter, startnode, lv);
        /* Export buffer is full end prematurally */
        if(numcand < 0) return numcand;

        /* If we are here, export is succesful. Work on the this particle -- first
         * filter out all of the candidates that are actually outside. */
        int numngb;

        for(numngb = 0; numngb < numcand; numngb ++) {
            int other = lv->ngblist[numngb];

            /* skip garbage */
            if(P[other].IsGarbage) continue;

            /* must be the correct type */
            if(!((1<<P[other].Type) & iter->mask))
                continue;

            /* must be the correct time bin */
            if(lv->tw->type == TREEWALK_SPLIT && !(BINMASK(P[other].TimeBin) & lv->tw->bgmask))
                continue;

            if(lv->tw->type != TREEWALK_SPLIT) /* FIXME: get rid of this entirely */
                drift_particle(other, All.Ti_Current);

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
                iter->dist[d] = NEAREST(I->Pos[d] - P[other].Pos[d]);
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
    lv->Nnodesinlist += inode;
    return 0;
}

/**
 * Cull a node.
 *
 * Returns 1 if the node shall be opened;
 * Returns 0 if the node has no business with this query.
 */
static int
cull_node(TreeWalkQueryBase * I, TreeWalkNgbIterBase * iter, int no)
{
    struct NODE * current = &Nodes[no];

    double dist;
    if(iter->symmetric == NGB_TREEFIND_SYMMETRIC) {
        dist = DMAX(Nodes[no].u.d.hmax, iter->Hsml) + 0.5 * current->len;
    } else {
        dist = iter->Hsml + 0.5 * current->len;
    }

    double r2 = 0;
    double dx = 0;
    /* do each direction */
    int d;
    for(d = 0; d < 3; d ++) {
        dx = NEAREST(current->center[d] - I->Pos[d]);
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
 * The hosting nodes are exported as well, if tw->UseNodeList is True.
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
    struct NODE *current;

    int numcand = 0;

    no = startnode;

    while(no >= 0)
    {
        if(node_is_particle(no))  /* single particle */ {
            lv->ngblist[numcand++] = no;
            no = Nextnode[no];
            continue;
        }
        if(node_is_pseudo_particle(no)) {
            /* pseudo particle */
            if(lv->mode == 1) {
                if(!lv->tw->UseNodeList) {
                    no = Nextnode[no - MaxNodes];
                    continue;
                } else {
                    endrun(12312, "Touching outside of my domain from a node list of a ghost. This shall not happen.");
                }
            } else {
                if(-1 == treewalk_export_particle(lv, no))
                    return -1;
            }

            no = Nextnode[no - MaxNodes];
            continue;
        }

        current = &Nodes[no];

        if(lv->mode == 1) {
            if (lv->tw->UseNodeList) {
                if(current->f.TopLevel) {
                    /* we reached a top-level node again, which means that we are done with the branch */
                    break;
                }
            }
        }

        /* Cull the node */
        if(0 == cull_node(I, iter, no)) {
            /* in case the node can be discarded */
            no = current->u.d.sibling;
            continue;
        }

        /* ok, we need to open the node */
        no = current->u.d.nextnode;
        continue;
    }

    return numcand;
}

