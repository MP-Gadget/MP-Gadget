#include <string.h>
#include <stdlib.h>
#include "allvars.h"
#include "proto.h"
#include "evaluator.h"
#include "tags.h"

static void evaluate_init_thread(Evaluator * ev, LocalEvaluator * lv);
static void fill_task_queue (Evaluator * ev, struct ev_task * tq, int * pq, int length);

static int atomic_fetch_and_add(int * ptr, int value) {
    int k;
#if _OPENMP >= 201107
#pragma omp atomic capture
    {
      k = (*ptr);
      (*ptr)+=value;
    }
#else
#ifdef OPENMP_USE_SPINLOCK
    k = __sync_fetch_and_add(ptr, value);
#else /* non spinlock*/
#pragma omp critical
    {
      k = (*ptr);
      (*ptr)+=value;
    }
#endif
#endif
    return k;
}
static int atomic_add_and_fetch(int * ptr, int value) {
    int k;
#if _OPENMP >= 201107
#pragma omp atomic capture
    { 
      (*ptr)+=value;
      k = (*ptr);
    }
#else
#ifdef OPENMP_USE_SPINLOCK
    k = __sync_add_and_fetch(ptr, value);
#else /* non spinlock */
#pragma omp critical
    { 
      (*ptr)+=value;
      k = (*ptr);
    }
#endif
#endif
    return k;
}
void evaluate_init_thread(Evaluator * ev, LocalEvaluator * lv) {
    int thread_id = omp_get_thread_num();
    int j;
    lv->ev = ev;
    lv->exportflag = Exportflag + thread_id * NTask;
    lv->exportnodecount = Exportnodecount + thread_id * NTask;
    lv->exportindex = Exportindex + thread_id * NTask;
    lv->Ninteractions = 0;
    lv->Nnodesinlist = 0;
    for(j = 0; j < NTask; j++)
        lv->exportflag[j] = -1;
}

void evaluate_begin(Evaluator * ev) {
    All.BunchSize =
        (int) ((All.BufferSize * 1024 * 1024) / (sizeof(struct data_index) + 
                    sizeof(struct data_nodelist) +
                    ev->ev_datain_elsize + ev->ev_dataout_elsize,
                    sizemax(ev->ev_datain_elsize,
                        ev->ev_dataout_elsize)));
    DataIndexTable =
        (struct data_index *) mymalloc("DataIndexTable", All.BunchSize * sizeof(struct data_index));
    DataNodeList =
        (struct data_nodelist *) mymalloc("DataNodeList", All.BunchSize * sizeof(struct data_nodelist));

    ev->PQueueEnd = 0;

    ev->PQueue = evaluate_get_queue(ev, &ev->PQueueEnd);

    ev->PrimaryTasks = (struct ev_task *) mymalloc("PrimaryTasks", sizeof(struct ev_task) * ev->PQueueEnd);

    int i = 0;
#pragma omp parallel for if(ev->PQueueEnd > 1024)
    for(i = 0; i < ev->PQueueEnd; i ++) {
        int p = ev->PQueue[i];
        P[p].Evaluated = 0;
    }

    fill_task_queue(ev, ev->PrimaryTasks, ev->PQueue, ev->PQueueEnd);
    ev->currentIndex = mymalloc("currentIndexPerThread", sizeof(int) * All.NumThreads);
    ev->currentEnd = mymalloc("currentEndPerThread", sizeof(int) * All.NumThreads);

    for(i = 0; i < All.NumThreads; i ++) {
        ev->currentIndex[i] = ((size_t) i) * ev->PQueueEnd / All.NumThreads;
        ev->currentEnd[i] = ((size_t) i + 1) * ev->PQueueEnd / All.NumThreads;
    }
}

void evaluate_finish(Evaluator * ev) {
    myfree(ev->currentEnd);
    myfree(ev->currentIndex);
    myfree(ev->PrimaryTasks);
    myfree(ev->PQueue);
    myfree(DataNodeList);
    myfree(DataIndexTable);
}

int data_index_compare(const void *a, const void *b);

static void real_ev(Evaluator * ev) {
    int tid = omp_get_thread_num();
    int i;
    LocalEvaluator lv ;
    void * extradata = NULL;
    if(ev->ev_alloc) extradata = ev->ev_alloc();

    evaluate_init_thread(ev, &lv);

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
        rt = ev->ev_evaluate(i, 0, input, output, &lv, extradata);
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
    ev->Ninteractions += lv.Ninteractions;
#pragma omp atomic
    ev->Nnodesinlist += lv.Nnodesinlist;
}
int * evaluate_get_queue(Evaluator * ev, int * len) {
    int i;
    int * queue = mymalloc("ActiveQueue", NumPart * sizeof(int));
    int k = 0;
    for(i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i])
    {
        if(!ev->ev_isactive(i)) continue;
        queue[k++] = i;
    }
    *len = k;
    return queue;
}

/* returns number of exports */
int evaluate_primary(Evaluator * ev) {
    double tstart, tend;
    ev->BufferFullFlag = 0;
    ev->Nexport = 0;

    int i;
    tstart = second();

 #pragma omp parallel for if(All.BunchSize > 1024) 
    for(i = 0; i < All.BunchSize; i ++) {
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
    if(ev->Nexport > All.BunchSize)
        ev->Nexport = All.BunchSize;

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

    qsort(DataIndexTable, ev->Nexport, sizeof(struct data_index), data_index_compare);

    /* adjust Nexport to skip the allocated but unused ones due to threads */
    while (ev->Nexport > 0 && DataIndexTable[ev->Nexport - 1].Task == NTask) {
        ev->Nexport --;
    }

    if(ev->Nexport == 0 && ev->BufferFullFlag) {
        /* buffer too small for even one particle (many nodes there can be)*/
        endrun(1231245);
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

int evaluate_ndone(Evaluator * ev) {
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

void evaluate_secondary(Evaluator * ev) {
    double tstart, tend;

    tstart = second();
    ev->dataresult = mymalloc("EvDataResult", ev->Nimport * ev->ev_dataout_elsize);

#pragma omp parallel 
    {
        int j;
        int thread_id = omp_get_thread_num();
        LocalEvaluator lv;
        void  * extradata = NULL;
        if(ev->ev_alloc)
            extradata = ev->ev_alloc();
        evaluate_init_thread(ev, &lv);
#pragma omp for
        for(j = 0; j < ev->Nimport; j++) {
            void * input = ev->dataget + j * ev->ev_datain_elsize;
            void * output = ev->dataresult + j * ev->ev_dataout_elsize;
            memset(output, 0, ev->ev_dataout_elsize);
            if(!ev->UseNodeList) {
                ((int*) input)[0] = All.MaxPart; /* root node */
                ((int*) input)[1] = -1; /* terminate immediately */
            }
            ev->ev_evaluate(j, 1, input, output, &lv, extradata);
        }
#pragma omp atomic
        ev->Ninteractions += lv.Ninteractions;
#pragma omp atomic
        ev->Nnodesinlist += lv.Nnodesinlist;
    }
    tend = second();
    ev->timecomp2 += timediff(tstart, tend);
}

/* export a particle at target and no, thread safely
 * 
 * This can also be called from a nonthreaded code
 *
 * */
int evaluate_export_particle(LocalEvaluator * lv, int target, int no) {
    int *exportflag = lv->exportflag;
    int *exportnodecount = lv->exportnodecount;
    int *exportindex = lv->exportindex; 
    Evaluator * ev = lv->ev;
    int task;

    if(exportflag[task = DomainTask[no - (All.MaxPart + MaxNodes)]] != target)
    {
        exportflag[task] = target;
        exportnodecount[task] = NODELISTLENGTH;
    }

    if(exportnodecount[task] == NODELISTLENGTH)
    {
        int nexp;

        if(ev->Nexport < All.BunchSize) {
            nexp = atomic_fetch_and_add(&ev->Nexport, 1);
        } else {
            nexp = All.BunchSize;
        }

        if(nexp >= All.BunchSize) {
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

void evaluate_run(Evaluator * ev) {
    /* run the evaluator */
    evaluate_begin(ev);
    do
    {
        evaluate_primary(ev); /* do local particles and prepare export list */
        /* exchange particle data */
        evaluate_get_remote(ev, TAG_EVALUATE_A);
        /* now do the particles that were sent to us */
        evaluate_secondary(ev);
        /* import the result to local particles */
        evaluate_reduce_result(ev, TAG_EVALUATE_B);
    } while(evaluate_ndone(ev) < NTask);
    evaluate_finish(ev);
}

static void evaluate_im_or_ex(void * sendbuf, void * recvbuf, size_t elsize, int tag, int import) {
    /* if import is 1, import the results from neigbhours */
    int ngrp;
    char * sp = sendbuf;
    char * rp = recvbuf;
     
    MPI_Status status;
    
    for(ngrp = 1; ngrp < (1 << PTask); ngrp++)
    {
        int sendTask = ThisTask;
        int recvTask = ThisTask ^ ngrp;

        if(recvTask < NTask)
        {
            if(Send_count[recvTask] > 0 || Recv_count[recvTask] > 0)
            {
                int so = import?Recv_offset[recvTask]:Send_offset[recvTask];
                int ro = import?Send_offset[recvTask]:Recv_offset[recvTask];
                int sc = import?Recv_count[recvTask]:Send_count[recvTask];
                int rc = import?Send_count[recvTask]:Recv_count[recvTask];

                /* get the particles */
                MPI_Sendrecv(sp + elsize * so,
                        sc * elsize, MPI_BYTE,
                        recvTask, tag,
                        rp + elsize * ro,
                        rc * elsize, MPI_BYTE,
                        recvTask, tag, MPI_COMM_WORLD, &status);
            }
        }
    }
}

/* returns the remote particles */
void evaluate_get_remote(Evaluator * ev, int tag) {
    int j;
    double tstart, tend;

    void * recvbuf = mymalloc("EvDataGet", ev->Nimport * ev->ev_datain_elsize);
    char * sendbuf = mymalloc("EvDataIn", ev->Nexport * ev->ev_datain_elsize);

    tstart = second();
    /* prepare particle data for export */
    //
#pragma omp parallel for if (ev->Nexport > 128) 
    for(j = 0; j < ev->Nexport; j++)
    {
        int place = DataIndexTable[j].Index;
        /* the convention is to have nodelist at the beginning */
        if(ev->UseNodeList) {
            memcpy(sendbuf + j * ev->ev_datain_elsize, DataNodeList[DataIndexTable[j].IndexGet].NodeList, 
                    sizeof(int) * NODELISTLENGTH);
        }
        ev->ev_copy(place, sendbuf + j * ev->ev_datain_elsize);
    }
    tend = second();
    ev->timecomp1 += timediff(tstart, tend);

    tstart = second();
    evaluate_im_or_ex(sendbuf, recvbuf, ev->ev_datain_elsize, tag, 0);
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
void evaluate_reduce_result(Evaluator * ev, int tag) {

    int j;
    double tstart, tend;

    void * sendbuf = ev->dataresult;
    char * recvbuf = (char*) mymalloc("EvDataOut", 
                ev->Nexport * ev->ev_dataout_elsize);

    tstart = second();
    evaluate_im_or_ex(sendbuf, recvbuf, ev->ev_dataout_elsize, tag, 1);
    tend = second();
    ev->timecommsumm2 += timediff(tstart, tend);

    tstart = second();

    for(j = 0; j < ev->Nexport; j++) {
        DataIndexTable[j].IndexGet = j;
    }

    /* mysort is a lie! */
    qsort(DataIndexTable, ev->Nexport, sizeof(struct data_index), data_index_compare_by_index);
    
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

static int ev_task_cmp_by_top_node(const void * p1, const void * p2) {
    const struct ev_task * t1 = p1, * t2 = p2;
    if(t1->top_node > t2->top_node) return 1;
    if(t1->top_node < t2->top_node) return -1;
    return 0;
}


static void fill_task_queue (Evaluator * ev, struct ev_task * tq, int * pq, int length) {
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
    }
    // qsort(tq, length, sizeof(struct ev_task), ev_task_cmp_by_top_node);
}
