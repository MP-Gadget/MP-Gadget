#ifndef _EVALUATOR_H_
#define _EVALUATOR_H_

typedef struct _Exportor {
    int *exportflag;
    int *exportnodecount;
    int *exportindex;
    int BufferFullFlag;
    int Nexport;
} Exporter;

typedef int (*ev_evaluate_t) (int target, int mode, Exporter * exportor, void * extradata);

typedef int (*ev_isactive_t) (int i);
typedef void * (*ev_alloc_t) ();

typedef struct _Evaluator {
    ev_evaluate_t ev_evaluate;
    ev_isactive_t ev_isactive;
    ev_alloc_t ev_alloc;
} Evaluator;

int evaluate_primary(Evaluator * ev); 
void evaluate_secondary(Evaluator * ev);
void evaluate_init_exporter(Exporter * exporter);

void exporter_export_particle(Exporter * exporter, int target, int no, int forceusenodelist);
#endif
