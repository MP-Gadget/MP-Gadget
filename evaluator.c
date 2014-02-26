#include "allvars.h"
#include "proto.h"
#include "evaluator.h"

static void evaluate_init_exporter(Evaluator * ev, Exporter * exporter);

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
void evaluate_init_exporter(Evaluator * ev, Exporter * exporter) {
    int thread_id = omp_get_thread_num();
    int j;
    exporter->ev = ev;
    exporter->exportflag = Exportflag + thread_id * NTask;
    exporter->exportnodecount = Exportnodecount + thread_id * NTask;
    exporter->exportindex = Exportindex + thread_id * NTask;
    for(j = 0; j < NTask; j++)
        exporter->exportflag[j] = -1;
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

    ev->PQueueRunning = (int*) mymalloc("PQueueRunning", sizeof(int) * NumPart);
    int i = 0;
    int p;
    for(p = FirstActiveParticle; p>= 0; p = NextActiveParticle[p]) {
        if(!ev->ev_isactive(p)) continue;
        ev->PQueueRunning[i] = p;
        P[p].Evaluated = 0;
        i++;
    }
    ev->QueueEnd = i;
    ev->currentIndex = 0;
    ev->done = 0;
}

void evaluate_finish(Evaluator * ev) {
    if(!ev->done) {
        /* this shall not happen */
        endrun(301811);
    }
    myfree(ev->PQueueRunning);
    myfree(DataNodeList);
    myfree(DataIndexTable);
}

int data_index_compare(const void *a, const void *b);

static void real_ev(Evaluator * ev) {
    int i;

    Exporter exporter;
    int * ngblist = ev->ev_alloc();
    evaluate_init_exporter(ev, &exporter);
    int abandoned = -1;
    /* Note: exportflag is local to each thread */
    while(1)
    {
        if(ev->BufferFullFlag) break;

        int k = atomic_fetch_and_add(&(ev->currentIndex), 1);

        if(k >= ev->QueueEnd) {
            break;
        }
        i = ev->PQueueRunning[k];

        if(P[i].Evaluated) {
            BREAKPOINT; 
        }
        if(!ev->ev_isactive(i)) {
            BREAKPOINT;
        }
        int rt;
        rt = ev->ev_evaluate(i, 0, &exporter, ngblist);

        if(rt < 0) {
            abandoned = i;
            P[i].Evaluated = 0;
            break;		/* export buffer has filled up */
        } else {
            P[i].Evaluated = 1;
        }
    }
    /* this barrier is important! to make sure no body is 
     * reading from PQueueRunning */
#pragma omp barrier
#pragma omp single
    {
        if(ev->currentIndex >= ev->QueueEnd) {
            ev->currentIndex = ev->QueueEnd;
        }
    }
#pragma omp barrier
    if(abandoned >= 0) {
        int k = atomic_add_and_fetch(&ev->currentIndex, -1);
        if(k >= ev->QueueEnd) {
            BREAKPOINT;
        }
        ev->PQueueRunning[k] = abandoned;
    }
}
int * evaluate_get_queue(Evaluator * ev, int * len) {
    int * queue = mymalloc("ActiveQueue", NumPart * sizeof(int));
    int Nactive = 0;
    int i;
    for(i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i])
    {
        if(!ev->ev_isactive(i)) continue;
        queue[Nactive++] = i;
    }
    *len = Nactive;
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
    tend = second();
    ev->timecomp1 += timediff(tstart, tend);

    if(ev->BufferFullFlag)
        ev->Nexport = All.BunchSize;
#pragma omp parallel for if (ev->Nexport > 1024) 
    for(i = 0; i < ev->Nexport; i ++) {
        /* if the NodeList of the particle is incomplete due
         * to abandoned work, also do not export it */
        int place = DataIndexTable[i].Index;
        if(! P[place].Evaluated) {
            /*
            int good = 0;
            int j;
            for(j = ev->currentIndex; j < ev->QueueEnd; j++) {
                if(ev->PQueueRunning[j] == place) good = 1;
            }
            if(!good) BREAKPOINT;
            */
            DataIndexTable[i].Task = NTask; 
            /* put in some junk so that we can detect them */
            DataNodeList[DataIndexTable[i].IndexGet].NodeList[0] = -2;
        }
    }

    qsort(DataIndexTable, ev->Nexport, sizeof(struct data_index), data_index_compare);

    int oldNexport = ev->Nexport;
    /* adjust Nexport to skip the allocated but unused ones due to threads */
    while (ev->Nexport > 0 && DataIndexTable[ev->Nexport - 1].Task == NTask) {
        ev->Nexport --;
    }
    if (ev->currentIndex >= ev->QueueEnd) ev->done = 1;

    if(ev->Nexport == 0 && ev->BufferFullFlag) {
        /* buffer too small !*/
        endrun(1231245);
    }

    for(i = 0; i < NTask; i++)
        Send_count[i] = 0;
    for(i = 0; i < ev->Nexport; i++) {
        if(DataIndexTable[i].Task >= NTask ||
                DataIndexTable[i].Task < 0) {
            BREAKPOINT;
        }
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
    MPI_Allreduce(&ev->done, &ndone, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    tend = second();
    ev->timewait2 += timediff(tstart, tend);
    return ndone;

}

void evaluate_secondary(Evaluator * ev) {
    double tstart, tend;

    tstart = second();

#pragma omp parallel 
    {
        int j, *ngblist;
        int thread_id = omp_get_thread_num();
        Exporter dummy;
        ngblist = ev->ev_alloc();

#pragma omp for
        for(j = 0; j < ev->Nimport; j++) {
            ev->ev_evaluate(j, 1, &dummy, ngblist);
        }
    }
    tend = second();
    ev->timecomp2 += timediff(tstart, tend);
}

/* export a particle at target and no, thread safely
 * 
 * This can also be called from a nonthreaded code
 *
 * */
int exporter_export_particle(Exporter * exporter, int target, int no) {
    int *exportflag = exporter->exportflag;
    int *exportnodecount = exporter->exportnodecount;
    int *exportindex = exporter->exportindex; 
    Evaluator * ev = exporter->ev;
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
void * evaluate_get_remote(Evaluator * ev, int tag) {
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
    return recvbuf;
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
void evaluate_reduce_result(Evaluator * ev,
        void * sendbuf, int tag) {

    int j;
    double tstart, tend;

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
    
    int * UniqueOff = mymalloc("UniqueIndex", sizeof(int) * ev->Nexport);
    UniqueOff[0] = 0;
    int Nunique = 0;

    for(j = 1; j < ev->Nexport; j++) {
        if(DataIndexTable[j].Index != DataIndexTable[j-1].Index)
            UniqueOff[++Nunique] = j;
    }
    if(ev->Nexport > 0)
        UniqueOff[++Nunique] = ev->Nexport;

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
    myfree(UniqueOff);
    tend = second();
    ev->timecomp1 += timediff(tstart, tend);
    myfree(recvbuf);
}
