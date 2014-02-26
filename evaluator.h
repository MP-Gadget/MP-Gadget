#ifndef _EVALUATOR_H_
#define _EVALUATOR_H_

struct _Evaluator;
typedef struct _Exportor {
    struct _Evaluator * ev;
    int *exportflag;
    int *exportnodecount;
    int *exportindex;
} Exporter;

typedef int (*ev_evaluate_func) (int target, int mode, Exporter * exportor, void * extradata);

typedef int (*ev_isactive_func) (int i);
typedef void * (*ev_alloc_func) ();

typedef void (*ev_copy_func)(int j, void * data_in);
/* mode == 0 is to set the initial local value
 * mode == 1 is to reduce the remote results */
typedef void (*ev_reduce_func)(int j, void * data_result, int mode);

typedef struct _Evaluator {
    ev_evaluate_func ev_evaluate;
    ev_isactive_func ev_isactive;
    ev_alloc_func ev_alloc;
    ev_copy_func ev_copy;
    ev_reduce_func ev_reduce;

    int UseNodeList;
    size_t ev_datain_elsize;
    size_t ev_dataout_elsize;
    
    double timewait1;
    double timewait2;
    double timecomp1;
    double timecomp2;
    double timecommsumm1;
    double timecommsumm2;

    int Nexport;
    int Nimport;
    int BufferFullFlag;

    int done;
    int currentIndex;
    int * PQueueRunning;
    int QueueEnd;
} Evaluator;

void evaluate_begin(Evaluator * ev);
void evaluate_finish(Evaluator * ev);
int evaluate_primary(Evaluator * ev); 
void * evaluate_get_remote(Evaluator * ev, int tag);
void evaluate_secondary(Evaluator * ev);
void evaluate_reduce_result(Evaluator * ev, void * sendbuf, int tag);
int * evaluate_get_queue(Evaluator * ev, int * len);

/*returns -1 if the buffer is full */
int exporter_export_particle(Exporter * exporter, int target, int no);

int evaluate_ndone(Evaluator * ev);
#endif
