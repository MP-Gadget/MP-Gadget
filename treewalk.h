#ifndef _EVALUATOR_H_
#define _EVALUATOR_H_

enum NgbTreeFindSymmetric {
    NGB_TREEFIND_SYMMETRIC,
    NGB_TREEFIND_ASYMMETRIC,
};

enum TreeWalkReduceMode {
    TREEWALK_PRIMARY,
    TREEWALK_GHOSTS,
};
void TreeWalk_allocate_memory(void);
struct _TreeWalk;

typedef struct {
    MyIDType ID;
    int NodeList[NODELISTLENGTH];
    double Pos[3];
} TreeWalkQueryBase;

typedef struct {
    MyIDType ID;
} TreeWalkResultBase;

typedef struct {
    enum NgbTreeFindSymmetric symmetric;
    int mask;
    double Hsml;
    double dist[3];
    double r2;
    double r;
    int other;
} TreeWalkNgbIterBase;

typedef struct {
    struct _TreeWalk * tw;

    int mode; /* 0 for Primary, 1 for Secondary */
    int target; /* defined only for primary (mode == 0) */

    int *exportflag;
    int *exportnodecount;
    int *exportindex;
    int * ngblist;
    int64_t Ninteractions;
    int64_t Nnodesinlist;
} LocalTreeWalk;

typedef int (*TreeWalkVisitFunction) (TreeWalkQueryBase * input, TreeWalkResultBase * output, LocalTreeWalk * lv);

typedef int (*TreeWalkNgbIterFunction) (TreeWalkQueryBase * input, TreeWalkResultBase * output, TreeWalkNgbIterBase * iter, LocalTreeWalk * lv);

typedef int (*TreeWalkIsActiveFunction) (const int i);
typedef int (*TreeWalkProcessFunction) (const int i);

typedef void (*TreeWalkFillQueryFunction)(const int j, TreeWalkQueryBase * query);
typedef void (*TreeWalkReduceResultFunction)(const int j, TreeWalkResultBase * result, const enum TreeWalkReduceMode mode);

typedef struct _TreeWalk {
    /* name of the evaluator (used in printing messages) */
    char * ev_label;

    TreeWalkVisitFunction visit;
    TreeWalkIsActiveFunction isactive;
    TreeWalkFillQueryFunction fill;
    TreeWalkReduceResultFunction reduce;
    TreeWalkNgbIterFunction ngbiter;
    TreeWalkProcessFunction postprocess;

    char * dataget;
    char * dataresult;

    int UseNodeList;
    int UseAllParticles; /* if 1 use all particles
                             if 0 use active particles */
    size_t query_type_elsize;
    size_t result_type_elsize;
    size_t ngbiter_type_elsize;

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
    int BunchSize;

    struct ev_task * PrimaryTasks;
    int * PQueue;
    int PQueueSize;

    /* per worker thread*/
    int *currentIndex;
    int *currentEnd;
} TreeWalk;

void treewalk_run(TreeWalk * tw);
int * treewalk_get_queue(TreeWalk * tw, int * len);
int treewalk_visit_ngbiter(TreeWalkQueryBase * I,
            TreeWalkResultBase * O,
            LocalTreeWalk * lv);

/*returns -1 if the buffer is full */
int treewalk_export_particle(LocalTreeWalk * lv, int no);
#define TREEWALK_REDUCE(A, B) (A) = (mode==TREEWALK_PRIMARY)?(B):((A) + (B))
#endif
