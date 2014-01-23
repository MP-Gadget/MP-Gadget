typedef struct _EvaluatorData {
    int *exportflag;
    int *exportnodecount;
    int *exportindex;
    int *ngblist;
} EvaluatorData;

typedef int (*ev_evaluate_t) (int target, int mode, EvaluatorData * data);

typedef int (*ev_isactive_t) (int i);

typedef struct _Evaluator {
    ev_evaluate_t ev_evaluate;
    ev_isactive_t ev_isactive;
} Evaluator;

void evaluate_primary(Evaluator * ev);
void evaluate_secondary(Evaluator * ev);
