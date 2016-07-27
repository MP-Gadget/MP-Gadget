#ifndef _EVALUATOR_H_
#define _EVALUATOR_H_

extern int *Send_offset, *Send_count, *Recv_count, *Recv_offset;
void TreeWalk_allocate_memory(void);

struct _TreeWalk;
typedef struct _LocalTreeWalk {
    struct _TreeWalk * ev;
    int *exportflag;
    int *exportnodecount;
    int *exportindex;
    int64_t Ninteractions;
    int64_t Nnodesinlist;
    int * ngblist;
} LocalTreeWalk;

typedef int (*ev_ev_func) (const int target, const int mode, 
        void * input, void * output, LocalTreeWalk * lv);

typedef int (*ev_isactive_func) (const int i);

typedef void (*ev_copy_func)(const int j, void * data_in);
/* mode == 0 is to set the initial local value
 * mode == 1 is to reduce the remote results */
typedef void (*ev_reduce_func)(const int j, void * data_result, const int mode);

struct ev_task {
    int top_node;
    int place; 
} ;


typedef struct _TreeWalk {
    ev_ev_func ev_evaluate;
    ev_isactive_func ev_isactive;
    ev_copy_func ev_copy;
    ev_reduce_func ev_reduce;
    char * ev_label;  /* name of the evaluator (used in printing messages) */

    int * ngblist;

    char * dataget;
    char * dataresult;

    int UseNodeList;
    int UseAllParticles; /* if 1 use all particles 
                             if 0 use active particles */
    size_t ev_datain_elsize;
    size_t ev_dataout_elsize;
    
    /* performance metrics */
    double timewait1;
    double timewait2;
    double timecomp1;
    double timecomp2;
    double timecommsumm1;
    double timecommsumm2;
    int64_t Ninteractions;
    int64_t Nnodesinlist;
    int64_t Nexport_sum;
    int64_t Niterations;

    /* internal flags*/
    int Nexport;
    int Nimport;
    int BufferFullFlag;

    struct ev_task * PrimaryTasks;
    int * PQueue;
    int PQueueEnd;

    /* per worker thread*/
    int *currentIndex;
    int *currentEnd;
} TreeWalk;

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

void ev_run(TreeWalk * ev);
void ev_begin(TreeWalk * ev);
void ev_finish(TreeWalk * ev);
int ev_primary(TreeWalk * ev); 
void ev_get_remote(TreeWalk * ev, int tag);
void ev_secondary(TreeWalk * ev);
void ev_reduce_result(TreeWalk * ev, int tag);
int * ev_get_queue(TreeWalk * ev, int * len);

/*returns -1 if the buffer is full */
int ev_export_particle(LocalTreeWalk * lv, int target, int no);

int ev_ndone(TreeWalk * ev);
#define EV_REDUCE(A, B) (A) = (mode==0)?(B):((A) + (B))
#endif
