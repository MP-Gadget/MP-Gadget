#include "allvars.h"
#include "proto.h"
#include "evaluator.h"

int Nexport, Nimport;
static int BufferFullFlag;

static int * NextEvParticle;
static void evaluate_init_exporter(Exporter * exporter);

static int atomic_fetch_and_add(int * ptr, int value) {
    int k;
#if _OPENMP >= 201107
#pragma omp atomic capture
    k = *ptr ++;
#else
    k = __sync_fetch_and_add(ptr, value);
#endif
    return k;
}
static int atomic_add_and_fetch(int * ptr, int value) {
    int k;
#if _OPENMP >= 201107
#pragma omp atomic capture
    k = ++ *ptr;
#else
    k = __sync_add_and_fetch(ptr, value);
#endif
    return k;
}
void evaluate_init_exporter(Exporter * exporter) {
    int thread_id = omp_get_thread_num();
    int j;
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

    ev->ParticleQueue = (int*) mymalloc("PQueue", sizeof(int) * NumPart);
    int i = 0;
    int p;
    for(p = FirstActiveParticle; p>= 0; p = NextActiveParticle[p]) {
       ev->ParticleQueue[i] = p;
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
    myfree(ev->ParticleQueue);
    myfree(DataNodeList);
    myfree(DataIndexTable);
}

int data_index_compare(const void *a, const void *b);

/* returns number of exports */
int evaluate_primary(Evaluator * ev) {
    double tstart, tend;
    BufferFullFlag = 0;
    Nexport = 0;

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

    int i, j;

    Exporter exporter;
    int * ngblist = ev->ev_alloc();
    evaluate_init_exporter(&exporter);

    /* Note: exportflag is local to each thread */
    while(1)
    {
        if(BufferFullFlag) break;

        int k = atomic_fetch_and_add(&ev->currentIndex, 1);

        if(k >= ev->QueueEnd) {
            break;
        }
        i = ev->ParticleQueue[k];

        if(ev->ev_isactive(i)) 
        {
            if(ev->ev_evaluate(i, 0, &exporter, ngblist) < 0) {
                /* add the particle back to the top of the queue*/
                int k = atomic_add_and_fetch(&ev->currentIndex, -1);
                ev->ParticleQueue[k] = i;
                if (ThisTask == 0)
                    printf("Task %d rejecting %d\n", ThisTask, i);
                P[i].Evaluated = 0;
                break;		/* export buffer has filled up */
            } else {
                P[i].Evaluated = 1;
            }
        }
    }
    }

    tend = second();
    ev->timecomp1 += timediff(tstart, tend);

#pragma omp parallel for if (Nexport > 128)
    for(i = 0; i < Nexport; i ++) {
        /* if the NodeList of the particle is incomplete due
         * to abandoned work, also do not export it */
        if(! P[DataIndexTable[i].Index].Evaluated) {
            DataIndexTable[i].Task = NTask; 
        }
    }
#ifdef MYSORT
        mysort_dataindex(DataIndexTable, Nexport, sizeof(struct data_index), data_index_compare);
#else
        qsort(DataIndexTable, Nexport, sizeof(struct data_index), data_index_compare);
#endif


    int oldNexport = Nexport;
    /* adjust Nexport to skip the allocated but unused ones due to threads */
    while (Nexport > 0 && DataIndexTable[Nexport - 1].Task == NTask) {
        Nexport --;
    }
    if (ev->currentIndex >= ev->QueueEnd) ev->done = 1;

    if(Nexport == 0 && BufferFullFlag) {
        /* buffer too small !*/
        endrun(1231245);
    }

    for(i = 0; i < NTask; i++)
        Send_count[i] = 0;
    for(i = 0; i < Nexport; i++)
        Send_count[DataIndexTable[i].Task]++;

    tstart = second();
    MPI_Alltoall(Send_count, 1, MPI_INT, Recv_count, 1, MPI_INT, MPI_COMM_WORLD);
    tend = second();
    ev->timewait1 += timediff(tstart, tend);

    for(i = 0, Nimport = 0, Recv_offset[0] = 0, Send_offset[0] = 0; i < NTask; i++)
    {
        Nimport += Recv_count[i];

        if(i > 0)
        {
            Send_offset[i] = Send_offset[i - 1] + Send_count[i - 1];
            Recv_offset[i] = Recv_offset[i - 1] + Recv_count[i - 1];
        }
    }

    return Nexport;
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
        for(j = 0; j < Nimport; j++) {
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
 * forceusenodelist == 1 mimics the behavior in forcetree.c
 *
 * */
int exporter_export_particle(Exporter * exporter, int target, int no, int forceusenodelist) {
    int *exportflag = exporter->exportflag;
    int *exportnodecount = exporter->exportnodecount;
    int *exportindex = exporter->exportindex; 
    int task;

    if(exportflag[task = DomainTask[no - (All.MaxPart + MaxNodes)]] != target)
    {
        exportflag[task] = target;
        exportnodecount[task] = NODELISTLENGTH;
    }

    if(exportnodecount[task] == NODELISTLENGTH)
    {
        int nexp;

        nexp = atomic_fetch_and_add(&Nexport, 1);

        if(nexp >= All.BunchSize) {
            Nexport = All.BunchSize;
            /* out if buffer space. Need to discard work for this particle and interrupt */
            BufferFullFlag = 1;
            return -1;
        }
        exportnodecount[task] = 0;
        exportindex[task] = nexp;
        DataIndexTable[nexp].Task = task;
        DataIndexTable[nexp].Index = target;
        DataIndexTable[nexp].IndexGet = nexp;
    }

    if(forceusenodelist) 
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

void evaluate_get_remote(Evaluator * ev, void * recvbuf, int tag) {
    int j;
    double tstart, tend;

    char * sendbuf = mymalloc("EvDataIn", Nexport * ev->ev_datain_elsize);

    tstart = second();
    /* prepare particle data for export */
#pragma omp parallel for if (Nexport > 128)
    for(j = 0; j < Nexport; j++)
    {
        int place = DataIndexTable[j].Index;
        ev->ev_copy(place, sendbuf + j * ev->ev_datain_elsize,
            DataNodeList[DataIndexTable[j].IndexGet].NodeList);
    }
    tend = second();
    ev->timecomp1 += timediff(tstart, tend);

    tstart = second();
    evaluate_im_or_ex(sendbuf, recvbuf, ev->ev_datain_elsize, tag, 0);
    tend = second();
    ev->timecommsumm1 += timediff(tstart, tend);
    myfree(sendbuf);
}

int data_index_compare_by_index(const void *a, const void *b)
{
    if(((struct data_index *) a)->Index < (((struct data_index *) b)->Index))
        return -1;

    if(((struct data_index *) a)->Index > (((struct data_index *) b)->Index))
        return +1;

    return 0;
}
void evaluate_reduce_result(Evaluator * ev,
        void * sendbuf, int tag) {

    int j;
    double tstart, tend;

    char * recvbuf = (char*) mymalloc("EvDataOut", 
                Nexport * ev->ev_dataout_elsize);

    tstart = second();
    evaluate_im_or_ex(sendbuf, recvbuf, ev->ev_dataout_elsize, tag, 1);
    tend = second();
    ev->timecommsumm2 += timediff(tstart, tend);

    tstart = second();

    for(j = 0; j < Nexport; j++) {
        DataIndexTable[j].IndexGet = j;
    }

#ifdef MYSORT
    mysort_dataindex(DataIndexTable, Nexport, sizeof(struct data_index), data_index_compare_by_index);
#else
    qsort(DataIndexTable, Nexport, sizeof(struct data_index), data_index_compare_by_index);
#endif
    
    int * UniqueOff = mymalloc("UniqueIndex", sizeof(int) * Nexport);
    UniqueOff[0] = 0;
    int Nunique= 1;

    for(j = 1; j < Nexport; j++) {
        if(DataIndexTable[j].Index != DataIndexTable[j-1].Index)
            UniqueOff[Nunique++] = j;
    }
    UniqueOff[Nunique] = Nexport;

#pragma omp parallel for if (Nunique > 16)
    for(j = 0; j < Nunique; j++)
    {
        int k;
        for(k = UniqueOff[j]; k < UniqueOff[j+1]; k++) {
            int place = DataIndexTable[k].Index;
            int get = DataIndexTable[k].IndexGet;
            ev->ev_reduce(place, recvbuf + ev->ev_dataout_elsize * get);
        }
    }
    myfree(UniqueOff);
    tend = second();
    ev->timecomp1 += timediff(tstart, tend);
    myfree(recvbuf);
}
