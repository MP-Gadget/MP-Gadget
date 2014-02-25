#ifndef _EVALUATOR_H_
#define _EVALUATOR_H_

typedef struct _Exportor {
    int *exportflag;
    int *exportnodecount;
    int *exportindex;
} Exporter;

typedef int (*ev_evaluate_func) (int target, int mode, Exporter * exportor, void * extradata);

typedef int (*ev_isactive_func) (int i);
typedef void * (*ev_alloc_func) ();

typedef void (*ev_copy_func)(int j, void * data_in, int * nodelist);
typedef void (*ev_reduce_func)(int j, void * data_result);

typedef struct _Evaluator {
    ev_evaluate_func ev_evaluate;
    ev_isactive_func ev_isactive;
    ev_alloc_func ev_alloc;
    ev_copy_func ev_copy;
    ev_reduce_func ev_reduce;

    size_t ev_datain_elsize;
    size_t ev_dataout_elsize;

    double timewait1;
    double timewait2;
    double timecomp1;
    double timecomp2;
    double timecommsumm1;
    double timecommsumm2;
    int done;
    int currentIndex;
    int * ParticleQueue;
    int QueueEnd;
} Evaluator;

void evaluate_begin(Evaluator * ev);
void evaluate_finish(Evaluator * ev);
int evaluate_primary(Evaluator * ev); 
void evaluate_secondary(Evaluator * ev);
void evaluate_init_exporter(Exporter * exporter);

/*returns -1 if the buffer is full */
int exporter_export_particle(Exporter * exporter, int target, int no, int forceusenodelist);

void evaluate_reduce_result(Evaluator * ev, void * sendbuf, int tag);

void evaluate_get_remote(Evaluator * ev, void * recvbuf, int tag);
int evaluate_ndone(Evaluator * ev);
#endif
