#include <mpi.h>
#include <string.h>
#include <stdlib.h>
#include "allvars.h"
#include "proto.h"
#include "treewalk.h"

#include "openmpsort.h"
#include "mymalloc.h"
#include "domain.h"
#include "forcetree.h"
#include "endrun.h"

#define TAG_EVALUATE_A (9999)
#define TAG_EVALUATE_B (10000)

static int *Exportflag;	        /*!< Buffer used for flagging whether a particle needs to be exported to another process */
static int *Exportnodecount;
static int *Exportindex;
static int *Send_offset, *Send_count, *Recv_count, *Recv_offset, *Sendcount;

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

static void ev_init_thread(TreeWalk * ev, LocalTreeWalk * lv);
static void fill_task_queue (TreeWalk * ev, struct ev_task * tq, int * pq, int length);
static void ev_begin(TreeWalk * ev);
static void ev_finish(TreeWalk * ev);
static int ev_primary(TreeWalk * ev); 
static void ev_get_remote(TreeWalk * ev, int tag);
static void ev_secondary(TreeWalk * ev);
static void ev_reduce_result(TreeWalk * ev, int tag);
static int ev_ndone(TreeWalk * ev);


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
        printf("ev->PrimaryTasks[0] = %d %d (%d) %s:%d\n", ev->PrimaryTasks[0].top_node, ev->PrimaryTasks[0].place, ev->PQueueEnd, __FILE__, __LINE__); \
    }
static TreeWalk * GDB_current_ev = NULL;


/*This routine allocates buffers to store the number of particles that shall be exchanged between MPI tasks.*/
void TreeWalk_allocate_memory(void)
{
    int NTaskTimesThreads;

    NTaskTimesThreads = All.NumThreads * NTask;

    Exportflag = (int *) malloc(NTaskTimesThreads * sizeof(int));
    Exportindex = (int *) malloc(NTaskTimesThreads * sizeof(int));
    Exportnodecount = (int *) malloc(NTaskTimesThreads * sizeof(int));

    Send_count = (int *) malloc(sizeof(int) * NTask);
    Send_offset = (int *) malloc(sizeof(int) * NTask);
    Recv_count = (int *) malloc(sizeof(int) * NTask);
    Recv_offset = (int *) malloc(sizeof(int) * NTask);
}



static void
ev_init_thread(TreeWalk * ev, LocalTreeWalk * lv)
{
    int thread_id = omp_get_thread_num();
    int j;
    lv->ev = ev;
    lv->ngblist = thread_id * NumPart + ev->ngblist;
    lv->exportflag = Exportflag + thread_id * NTask;
    lv->exportnodecount = Exportnodecount + thread_id * NTask;
    lv->exportindex = Exportindex + thread_id * NTask;
    lv->Ninteractions = 0;
    lv->Nnodesinlist = 0;
    for(j = 0; j < NTask; j++)
        lv->exportflag[j] = -1;
}

static void ev_begin(TreeWalk * ev)
{
    ev->BunchSize =
        (int) ((All.BufferSize * 1024 * 1024) / (sizeof(struct data_index) + 
                    sizeof(struct data_nodelist) + ev->ev_datain_elsize + ev->ev_dataout_elsize));
    DataIndexTable =
        (struct data_index *) mymalloc("DataIndexTable", ev->BunchSize * sizeof(struct data_index));
    DataNodeList =
        (struct data_nodelist *) mymalloc("DataNodeList", ev->BunchSize * sizeof(struct data_nodelist));

    memset(DataNodeList, -1, sizeof(struct data_nodelist) * ev->BunchSize);

    ev->PQueueEnd = 0;

    ev->PQueue = ev_get_queue(ev, &ev->PQueueEnd);
    ev->PrimaryTasks = (struct ev_task *) mymalloc("PrimaryTasks", sizeof(struct ev_task) * ev->PQueueEnd);

    fill_task_queue(ev, ev->PrimaryTasks, ev->PQueue, ev->PQueueEnd);
    ev->currentIndex = mymalloc("currentIndexPerThread", sizeof(int) * All.NumThreads);
    ev->currentEnd = mymalloc("currentEndPerThread", sizeof(int) * All.NumThreads);
    ev->ngblist = mymalloc("Ngblist", sizeof(int) * All.NumThreads * NumPart);

    int i;
    for(i = 0; i < All.NumThreads; i ++) {
        ev->currentIndex[i] = ((size_t) i) * ev->PQueueEnd / All.NumThreads;
        ev->currentEnd[i] = ((size_t) i + 1) * ev->PQueueEnd / All.NumThreads;
    }
}

static void ev_finish(TreeWalk * ev)
{
    myfree(ev->ngblist);
    myfree(ev->currentEnd);
    myfree(ev->currentIndex);
    myfree(ev->PrimaryTasks);
    myfree(ev->PQueue);
    myfree(DataNodeList);
    myfree(DataIndexTable);
}

int data_index_compare(const void *a, const void *b);

static void real_ev(TreeWalk * ev) {
    int tid = omp_get_thread_num();
    int i;
    LocalTreeWalk lv[1];

    ev_init_thread(ev, lv);
    lv->mode = 0;

    /* Note: exportflag is local to each thread */
    int k;
            /* use old index to recover from a buffer overflow*/;
    void * input = alloca(ev->ev_datain_elsize);
    void * output = alloca(ev->ev_dataout_elsize);

    for(k = ev->currentIndex[tid];
        k < ev->currentEnd[tid]; 
        k++) {
        if(ev->BufferFullFlag) break;

        i = ev->PrimaryTasks[k].place;

        if(P[i].Evaluated) {
            BREAKPOINT; 
        }
        if(!ev->ev_isactive(i)) {
            BREAKPOINT;
        }
        int rt;
        ev->ev_copy(i, input);
        ((int*) input)[0] = All.MaxPart; /* root node */
        ((int*) input)[1] = -1; /* terminate immediately */
        
        memset(output, 0, ev->ev_dataout_elsize);
        rt = ev->ev_evaluate(i, input, output, lv);
        if(rt < 0) {
            P[i].Evaluated = 0;
            break;		/* export buffer has filled up, redo this particle */
        } else {
            P[i].Evaluated = 1;
            if(ev->ev_reduce != NULL)
                ev->ev_reduce(i, output, 0);
        }
    }
    ev->currentIndex[tid] = k;
#pragma omp atomic
    ev->Ninteractions += lv->Ninteractions;
#pragma omp atomic
    ev->Nnodesinlist += lv->Nnodesinlist;
}
static int cmpint(const void * c1, const void * c2) {
    const int* i1=c1;
    const int* i2=c2;
    return i1 - i2;
}
int * ev_get_queue(TreeWalk * ev, int * len) {
    int i;
    int * queue = mymalloc("ActiveQueue", NumPart * sizeof(int));
    int k = 0;
    if(ev->UseAllParticles) {
        for(i = 0; i < NumPart; i++) {
            if(!ev->ev_isactive(i)) continue;
            queue[k++] = i;
        }
    } else {
        for(i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i])
        {
            if(!ev->ev_isactive(i)) continue;
            queue[k++] = i;
        }
        /* check the uniqueness of ActiveParticle list. */
        qsort(queue, k, sizeof(int), cmpint);
        for(i = 0; i < k - 1; i ++) {
            if(queue[i] == queue[i+1]) {
                endrun(8829, "There are duplicated active particles.");
            }
        }
    }
    *len = k;
    return queue;
}

/* returns number of exports */
static int ev_primary(TreeWalk * ev)
{
    double tstart, tend;
    ev->BufferFullFlag = 0;
    ev->Nexport = 0;

    int i;
    tstart = second();

 #pragma omp parallel for if(ev->BunchSize > 1024) 
    for(i = 0; i < ev->BunchSize; i ++) {
        DataIndexTable[i].Task = NTask;
        /*entries with NTask is not filled with particles, and will be
         * sorted to the end */
    }

#pragma omp parallel 
    {
        real_ev(ev);
    }

    /* Nexport may go off too much after BunchSize 
     * as we don't protect it from over adding in _export_particle
     * */
    if(ev->Nexport > ev->BunchSize)
        ev->Nexport = ev->BunchSize;

    tend = second();
    ev->timecomp1 += timediff(tstart, tend);


    /* touching up the export list, remove incomplete particles */
#pragma omp parallel for if (ev->Nexport > 1024) 
    for(i = 0; i < ev->Nexport; i ++) {
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

    qsort_openmp(DataIndexTable, ev->Nexport, sizeof(struct data_index), data_index_compare);

    /* adjust Nexport to skip the allocated but unused ones due to threads */
    while (ev->Nexport > 0 && DataIndexTable[ev->Nexport - 1].Task == NTask) {
        ev->Nexport --;
    }

    if(ev->Nexport == 0 && ev->BufferFullFlag) {
        endrun(1231245, "Buffer too small for even one particle. For example, there are too many nodes");
    }

    /* 
     * fill the communication layouts, 
     * here we reuse the legacy global variable names;
     * really should move them to local variables for the evaluator.
     * */
    for(i = 0; i < NTask; i++)
        Send_count[i] = 0;
    for(i = 0; i < ev->Nexport; i++) {
        Send_count[DataIndexTable[i].Task]++;
    }

    tstart = second();
    MPI_Alltoall(Send_count, 1, MPI_INT, Recv_count, 1, MPI_INT, MPI_COMM_WORLD);
    tend = second();
    ev->timewait1 += timediff(tstart, tend);

    for(i = 0, ev->Nimport = 0, Recv_offset[0] = 0, Send_offset[0] = 0; i < NTask; i++)
    {
        ev->Nimport += Recv_count[i];

        if(i > 0)
        {
            Send_offset[i] = Send_offset[i - 1] + Send_count[i - 1];
            Recv_offset[i] = Recv_offset[i - 1] + Recv_count[i - 1];
        }
    }

    return ev->Nexport;
}

static int ev_ndone(TreeWalk * ev)
{
    int ndone;
    double tstart, tend;
    tstart = second();
    int done = 1;
    int i;
    for(i = 0; i < All.NumThreads; i ++) {
        if(ev->currentIndex[i] < ev->currentEnd[i]) {
            done = 0;
            break;
        }
    }
    MPI_Allreduce(&done, &ndone, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    tend = second();
    ev->timewait2 += timediff(tstart, tend);
    return ndone;

}

static void ev_secondary(TreeWalk * ev)
{
    double tstart, tend;

    tstart = second();
    ev->dataresult = mymalloc("EvDataResult", ev->Nimport * ev->ev_dataout_elsize);

#pragma omp parallel 
    {
        int j;
        LocalTreeWalk lv[1];

        ev_init_thread(ev, lv);
        lv->mode = 1;
#pragma omp for
        for(j = 0; j < ev->Nimport; j++) {
            void * input = ev->dataget + j * ev->ev_datain_elsize;
            void * output = ev->dataresult + j * ev->ev_dataout_elsize;
            memset(output, 0, ev->ev_dataout_elsize);
            if(!ev->UseNodeList) {
                ((int*) input)[0] = All.MaxPart; /* root node */
                ((int*) input)[1] = -1; /* terminate immediately */
            }
            ev->ev_evaluate(j, input, output, lv);
        }
#pragma omp atomic
        ev->Ninteractions += lv->Ninteractions;
#pragma omp atomic
        ev->Nnodesinlist += lv->Nnodesinlist;
    }
    tend = second();
    ev->timecomp2 += timediff(tstart, tend);
}

/* export a particle at target and no, thread safely
 * 
 * This can also be called from a nonthreaded code
 *
 * */
int ev_export_particle(LocalTreeWalk * lv, int target, int no) {
    int *exportflag = lv->exportflag;
    int *exportnodecount = lv->exportnodecount;
    int *exportindex = lv->exportindex; 
    TreeWalk * ev = lv->ev;
    int task;

    if(exportflag[task = DomainTask[no - (All.MaxPart + MaxNodes)]] != target)
    {
        exportflag[task] = target;
        exportnodecount[task] = NODELISTLENGTH;
    }

    if(exportnodecount[task] == NODELISTLENGTH)
    {
        int nexp;

        if(ev->Nexport < ev->BunchSize) {
            nexp = atomic_fetch_and_add(&ev->Nexport, 1);
        } else {
            nexp = ev->BunchSize;
        }

        if(nexp >= ev->BunchSize) {
            /* out if buffer space. Need to discard work for this particle and interrupt */
            ev->BufferFullFlag = 1;
#pragma omp flush
            return -1;
        }
        exportnodecount[task] = 0;
        exportindex[task] = nexp;
        DataIndexTable[nexp].Task = task;
        DataIndexTable[nexp].Index = target;
        DataIndexTable[nexp].IndexGet = nexp;
    }

    if(ev->UseNodeList) 
    {
        DataNodeList[exportindex[task]].NodeList[exportnodecount[task]++] =
            DomainNodeIndex[no - (All.MaxPart + MaxNodes)];

        if(exportnodecount[task] < NODELISTLENGTH)
            DataNodeList[exportindex[task]].NodeList[exportnodecount[task]] = -1;
    }
    return 0;
}

void ev_run(TreeWalk * ev) {
    /* run the evaluator */
    GDB_current_ev = ev;
    ev_begin(ev);
    do
    {
        ev_primary(ev); /* do local particles and prepare export list */
        /* exchange particle data */
        ev_get_remote(ev, TAG_EVALUATE_A);
        report_memory_usage(ev->ev_label);
        /* now do the particles that were sent to us */
        ev_secondary(ev);
        /* import the result to local particles */
        ev_reduce_result(ev, TAG_EVALUATE_B);
        ev->Niterations ++;
        ev->Nexport_sum += ev->Nexport;
    } while(ev_ndone(ev) < NTask);
    ev_finish(ev);
}

static void ev_im_or_ex(void * sendbuf, void * recvbuf, size_t elsize, int tag, int import) {
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
static void ev_get_remote(TreeWalk * ev, int tag)
{
    int j;
    double tstart, tend;

    void * recvbuf = mymalloc("EvDataGet", ev->Nimport * ev->ev_datain_elsize);
    char * sendbuf = mymalloc("EvDataIn", ev->Nexport * ev->ev_datain_elsize);

    memset(sendbuf, -1, ev->Nexport * ev->ev_datain_elsize);

    tstart = second();
    /* prepare particle data for export */
    //
#pragma omp parallel for if (ev->Nexport > 128) 
    for(j = 0; j < ev->Nexport; j++)
    {
        int place = DataIndexTable[j].Index;
        /* the convention is to have nodelist at the beginning */
        if(ev->UseNodeList) {
            int * nl = DataNodeList[DataIndexTable[j].IndexGet].NodeList;
            memcpy(sendbuf + j * ev->ev_datain_elsize, nl, sizeof(int) * NODELISTLENGTH);
        }
        ev->ev_copy(place, sendbuf + j * ev->ev_datain_elsize);
    }
    tend = second();
    ev->timecomp1 += timediff(tstart, tend);

    tstart = second();
    ev_im_or_ex(sendbuf, recvbuf, ev->ev_datain_elsize, tag, 0);
    tend = second();
    ev->timecommsumm1 += timediff(tstart, tend);
    myfree(sendbuf);
    ev->dataget = recvbuf;
}

int data_index_compare_by_index(const void *a, const void *b)
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

static void ev_reduce_result(TreeWalk * ev, int tag)
{

    int j;
    double tstart, tend;

    void * sendbuf = ev->dataresult;
    char * recvbuf = (char*) mymalloc("EvDataOut", 
                ev->Nexport * ev->ev_dataout_elsize);

    tstart = second();
    ev_im_or_ex(sendbuf, recvbuf, ev->ev_dataout_elsize, tag, 1);
    tend = second();
    ev->timecommsumm2 += timediff(tstart, tend);

    tstart = second();

    for(j = 0; j < ev->Nexport; j++) {
        DataIndexTable[j].IndexGet = j;
    }

    /* mysort is a lie! */
    qsort_openmp(DataIndexTable, ev->Nexport, sizeof(struct data_index), data_index_compare_by_index);
    
    int * UniqueOff = mymalloc("UniqueIndex", sizeof(int) * (ev->Nexport + 1));
    UniqueOff[0] = 0;
    int Nunique = 0;

    for(j = 1; j < ev->Nexport; j++) {
        if(DataIndexTable[j].Index != DataIndexTable[j-1].Index)
            UniqueOff[++Nunique] = j;
    }
    if(ev->Nexport > 0)
        UniqueOff[++Nunique] = ev->Nexport;

    if(ev->ev_reduce != NULL) {
#pragma omp parallel for private(j) if(Nunique > 16)
        for(j = 0; j < Nunique; j++)
        {
            int k;
            int place = DataIndexTable[UniqueOff[j]].Index;
            int start = UniqueOff[j];
            int end = UniqueOff[j + 1];
            for(k = start; k < end; k++) {
                int get = DataIndexTable[k].IndexGet;
                ev->ev_reduce(place, recvbuf + ev->ev_dataout_elsize * get, 1);
            }
        }
    }
    myfree(UniqueOff);
    tend = second();
    ev->timecomp1 += timediff(tstart, tend);
    myfree(recvbuf);
    myfree(ev->dataresult);
    myfree(ev->dataget);
}

#if 0
static int ev_task_cmp_by_top_node(const void * p1, const void * p2) {
    const struct ev_task * t1 = p1, * t2 = p2;
    if(t1->top_node > t2->top_node) return 1;
    if(t1->top_node < t2->top_node) return -1;
    return 0;
}
#endif

static void fill_task_queue (TreeWalk * ev, struct ev_task * tq, int * pq, int length) {
    int i;
#pragma omp parallel for if(length > 1024)
    for(i = 0; i < length; i++) {
        int no = -1;
        /*
        if(0) {
            no = Father[pq[i]];
            while(no != -1) {
                if(Nodes[no].u.d.bitflags & (1 << BITFLAG_TOPLEVEL)) {
                    break;
                }
                no = Nodes[no].u.d.father;
            }
        }
       */
        tq[i].top_node = no;
        tq[i].place = pq[i];
        P[pq[i]].Evaluated = 0;
    }
    // qsort(tq, length, sizeof(struct ev_task), ev_task_cmp_by_top_node);
}


