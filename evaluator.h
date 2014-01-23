typedef struct _EvaluatorData {
    int *exportflag;
    int *exportnodecount;
    int *exportindex;
    int *ngblist;
} EvaluatorData;

typedef int (*ev_evaluate_t) (int target, int mode, EvaluatorData * evdata, void * extradata);

typedef int (*ev_isactive_t) (int i);
typedef void * (*ev_alloc_t) ();

typedef struct _Evaluator {
    ev_evaluate_t ev_evaluate;
    ev_isactive_t ev_isactive;
    ev_alloc_t ev_alloc;
} Evaluator;

void evaluate_primary(Evaluator * ev);
void evaluate_secondary(Evaluator * ev);
