#include "allvars.h"
#include "evaluator.h"

extern int NextParticle;
extern int Nexport, Nimport;
extern int BufferFullFlag;

static void evaluate_fix_export_buffer(int save_NextParticle);

void evaluate_primary(Evaluator * ev) {

    int save_NextParticle = NextParticle;

#pragma omp parallel
    {
    int thread_id = omp_get_thread_num();

    int i, j;

    int *exportflag, *exportnodecount, *exportindex, *ngblist;

    ngblist = Ngblist + thread_id * NumPart;
    exportflag = Exportflag + thread_id * NTask;
    exportnodecount = Exportnodecount + thread_id * NTask;
    exportindex = Exportindex + thread_id * NTask;

    /* Note: exportflag is local to each thread */
    for(j = 0; j < NTask; j++)
        exportflag[j] = -1;

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
            if(ev->ev_evaluate(i, 0, exportflag, exportnodecount, exportindex, ngblist) < 0)
                break;		/* export buffer has filled up */
        }

        ProcessedFlag[i] = 1;	/* particle successfully finished */
    }
    }
    evaluate_fix_export_buffer(save_NextParticle);
}

void evaluate_secondary(Evaluator * ev) {
#pragma omp parallel
    {
        int j, dummy, *ngblist;
        int thread_id = omp_get_thread_num();
        ngblist = Ngblist + thread_id * NumPart;

#pragma omp for
        for(j = 0; j < Nimport; j++) {
            ev->ev_evaluate(j, 1, &dummy, &dummy, &dummy, ngblist);
        }
    }

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
