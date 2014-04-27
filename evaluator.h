#ifndef _EVALUATOR_H_
#define _EVALUATOR_H_

struct _Evaluator;
typedef struct _LocalEvaluator {
    struct _Evaluator * ev;
    int *exportflag;
    int *exportnodecount;
    int *exportindex;
    int64_t Ninteractions;
    int64_t Nnodesinlist;
} LocalEvaluator;

typedef int (*ev_evaluate_func) (const int target, const int mode, 
        void * input, void * output, LocalEvaluator * lv, void * extradata);

typedef int (*ev_isactive_func) (const int i);
typedef void * (*ev_alloc_func) ();

typedef void (*ev_copy_func)(const int j, void * data_in);
/* mode == 0 is to set the initial local value
 * mode == 1 is to reduce the remote results */
typedef void (*ev_reduce_func)(const int j, void * data_result, const int mode);

struct ev_task {
    int top_node;
    int place; 
} ;


typedef struct _Evaluator {
    ev_evaluate_func ev_evaluate;
    ev_isactive_func ev_isactive;
    ev_alloc_func ev_alloc;
    ev_copy_func ev_copy;
    ev_reduce_func ev_reduce;

    char * dataget;
    char * dataresult;

    int UseNodeList;
    size_t ev_datain_elsize;
    size_t ev_dataout_elsize;
    
    double timewait1;
    double timewait2;
    double timecomp1;
    double timecomp2;
    double timecommsumm1;
    double timecommsumm2;
    int64_t Ninteractions;
    int64_t Nnodesinlist;

    int Nexport;
    int Nimport;
    int BufferFullFlag;

    struct ev_task * PrimaryTasks;
    int * PQueue;
    int PQueueEnd;

    /* per worker thread*/
    int *currentIndex;
    int *currentEnd;
} Evaluator;

void evaluate_run(Evaluator * ev);
void evaluate_begin(Evaluator * ev);
void evaluate_finish(Evaluator * ev);
int evaluate_primary(Evaluator * ev); 
void evaluate_get_remote(Evaluator * ev, int tag);
void evaluate_secondary(Evaluator * ev);
void evaluate_reduce_result(Evaluator * ev, int tag);
int * evaluate_get_queue(Evaluator * ev, int * len);

/*returns -1 if the buffer is full */
int evaluate_export_particle(LocalEvaluator * lv, int target, int no);

int evaluate_ndone(Evaluator * ev);
#define EV_REDUCE(A, B) (A) = (mode==0)?(B):((A) + (B))
#endif
