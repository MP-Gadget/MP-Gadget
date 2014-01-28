#include "allvars.h"
#include "evaluator.h"

extern int NextParticle;
int Nexport, Nimport;
static int BufferFullFlag;

static void evaluate_fix_export_buffer(int save_NextParticle);
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

/* returns number of exports */
int evaluate_primary(Evaluator * ev) {

    BufferFullFlag = 0;
    Nexport = 0;

    int save_NextParticle = NextParticle;

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
            NextParticle = (i < 0)?NextParticle:NextActiveParticle[i];
        }   
        if(i < 0) break;

        ProcessedFlag[i] = 0;
        if(ev->ev_isactive(i))
        {
            if(ev->ev_evaluate(i, 0, &exporter, ngblist) < 0)
                break;		/* export buffer has filled up */
        }

        ProcessedFlag[i] = 1;	/* particle successfully finished */
    }
    }
    evaluate_fix_export_buffer(save_NextParticle);
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
void exporter_export_particle(Exporter * exporter, int target, int no, int forceusenodelist) {
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

#ifdef DONOTUSENODELIST
    if(! forceusenodelist) {
        DataNodeList[exportindex[task]].NodeList[exportnodecount[task]++] =
            DomainNodeIndex[no - (All.MaxPart + MaxNodes)];

        if(exportnodecount[task] < NODELISTLENGTH)
            DataNodeList[exportindex[task]].NodeList[exportnodecount[task]] = -1;
    }
#endif
}

static void evaluate_fix_export_buffer(int save_NextParticle) {
    /* after threaded evaluation, it is possible 
     * particles that has been taken is not processed by primary
     * because of a buffer full fault.
     * this code will rewind NextParticle to the last fully processed
     * particle.
     *
     * In that case some particles may be evaluated twice;
     * Let's hope the code works fine with double evaluations:
     *   if none of the evaluation functions modify the particle that
     *   shall be exported, this should be alright. (likely true)
     * */
    int j, k;
    if(BufferFullFlag)
    {
        int last_nextparticle = NextParticle;

        NextParticle = save_NextParticle;

        while(NextParticle >= 0)
        {
            if(NextParticle == last_nextparticle)
                break;

            if(ProcessedFlag[NextParticle] != 1)
                break;

            ProcessedFlag[NextParticle] = 2;

            NextParticle = NextActiveParticle[NextParticle];
        }

        if(NextParticle == save_NextParticle)
        {
            /* in this case, the buffer is too small to process even a single particle */
            endrun(12998);
        }

        int new_export = 0;

        for(j = 0, k = 0; j < Nexport; j++)
            if(ProcessedFlag[DataIndexTable[j].Index] != 2)
            {
                if(k < j + 1)
                    k = j + 1;

                for(; k < Nexport; k++)
                    if(ProcessedFlag[DataIndexTable[k].Index] == 2)
                    {
                        int old_index = DataIndexTable[j].Index;

                        DataIndexTable[j] = DataIndexTable[k];
                        DataNodeList[j] = DataNodeList[k];
                        DataIndexTable[j].IndexGet = j;
                        new_export++;

                        DataIndexTable[k].Index = old_index;
                        k++;
                        break;
                    }
            }
            else
                new_export++;
        printf("old Nexport = %d, new Nexport = %d\n", Nexport, new_export);
        Nexport = new_export;
    }
}
