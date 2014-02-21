#include "allvars.h"
#include "proto.h"
#include "evaluator.h"

extern int NextParticle;
int Nexport, Nimport;
static int BufferFullFlag;

static int * NextEvParticle;
static void evaluate_init_exporter(Exporter * exporter);

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
    NextEvParticle = (int*) mymalloc("NextEv", sizeof(int) * NumPart);
    NextParticle = FirstActiveParticle;
    memcpy(NextEvParticle, NextActiveParticle, sizeof(int) * NumPart);
    ev->done = 0;
}
void evaluate_finish(Evaluator * ev) {
    if(!ev->done) {
        /* this shall not happen */
        endrun(301811);
    }
    myfree(NextEvParticle);
}

int data_index_compare(const void *a, const void *b);
/* returns number of exports */
int evaluate_primary(Evaluator * ev) {
    /* will start the evaluation from NextParticle*/
    BufferFullFlag = 0;
    Nexport = 0;
    int i;
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
#pragma omp critical (lock_nexport) 
        {
            i = BufferFullFlag?-1:NextParticle;
            /* move next particle pointer if a particle is obtained */
            NextParticle = (i < 0)?NextParticle:NextEvParticle[i];
        }   
        if(i < 0) break;

        if(ev->ev_isactive(i)) 
        {
            if(ev->ev_evaluate(i, 0, &exporter, ngblist) < 0) {
#pragma omp critical (lock_nexport) 
                {
                /* add the particle back to the top of the queue*/
                NextEvParticle[i] = NextParticle; 
                NextParticle = i;
                }
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
    if (NextParticle == -1) ev->done = 1;

    if(Nexport == 0 && BufferFullFlag) {
        /* buffer too small !*/
        endrun(1231245);
    }
    return Nexport;
}

void evaluate_secondary(Evaluator * ev) {
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
#pragma omp critical (lock_nexport) 
        {
            nexp = Nexport;
            Nexport = (nexp >= All.BunchSize)?nexp:(nexp + 1);
        }
        if(nexp >= All.BunchSize) {
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
void evaluate_export(void * sendbuf, void * recvbuf, size_t elsize, int tag) {
    evaluate_im_or_ex(sendbuf, recvbuf, elsize, tag, 0);
}
void evaluate_import(void * sendbuf, void * recvbuf, size_t elsize, int tag) {
    evaluate_im_or_ex(sendbuf, recvbuf, elsize, tag, 1);
}
