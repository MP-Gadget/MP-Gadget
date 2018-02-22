#ifndef _EVALUATOR_H_
#define _EVALUATOR_H_

#include <stdint.h>
#include "allvars.h"

#define  NODELISTLENGTH      8

enum NgbTreeFindSymmetric {
    NGB_TREEFIND_SYMMETRIC,
    NGB_TREEFIND_ASYMMETRIC,
};

enum TreeWalkReduceMode {
    TREEWALK_PRIMARY,
    TREEWALK_GHOSTS,
};

typedef struct TreeWalk TreeWalk;

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
    TreeWalk * tw;

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

typedef int (*TreeWalkHasWorkFunction) (const int i, TreeWalk * tw);
typedef int (*TreeWalkProcessFunction) (const int i, TreeWalk * tw);

typedef void (*TreeWalkFillQueryFunction)(const int j, TreeWalkQueryBase * query, TreeWalk * tw);
typedef void (*TreeWalkReduceResultFunction)(const int j, TreeWalkResultBase * result, const enum TreeWalkReduceMode mode, TreeWalk * tw);

enum TreeWalkType {
    TREEWALK_ACTIVE = 0,
    TREEWALK_ALL,
    TREEWALK_SPLIT,
};

struct TreeWalk {
    void * priv;

    /* name of the evaluator (used in printing messages) */
    char * ev_label;

    enum TreeWalkType type;

    size_t query_type_elsize;
    size_t result_type_elsize;
    size_t ngbiter_type_elsize;

    binmask_t bgmask; /* if set, the bins to compute force from; used if TreeWalkType is SPLIT */

    TreeWalkVisitFunction visit;                /* Function to be called between a tree node and a particle */
    TreeWalkHasWorkFunction haswork; /* Is the particle part of this interaction? */
    TreeWalkFillQueryFunction fill;       /* Copy the useful attributes of a particle to a query */
    TreeWalkReduceResultFunction reduce;  /* Reduce a partial result to the local particle storage */
    TreeWalkNgbIterFunction ngbiter;     /* called for each pair of particles if visit is set to ngbiter */
    TreeWalkProcessFunction postprocess; /* postprocess finalizes quantities for each particle, e.g. divide the normalization */
    TreeWalkProcessFunction preprocess; /* Preprocess initializes quantities for each particle */

    int UseNodeList;      /* Send tree branches or use the entire tree for ghost particles */

    char * dataget;
    char * dataresult;

    /* performance metrics */
    double timewait1;
    double timewait2;
    double timecomp1;
    double timecomp2;
    double timecomp3;
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
    int * WorkSet;
    int WorkSetSize;

    /* per worker thread*/
    int *currentIndex;
    int *currentEnd;

};

void treewalk_run(TreeWalk * tw, int * active_set, int size);

int treewalk_visit_ngbiter(TreeWalkQueryBase * I,
            TreeWalkResultBase * O,
            LocalTreeWalk * lv);

/*returns -1 if the buffer is full */
int treewalk_export_particle(LocalTreeWalk * lv, int no);
#define TREEWALK_REDUCE(A, B) (A) = (mode==TREEWALK_PRIMARY)?(B):((A) + (B))
#endif
