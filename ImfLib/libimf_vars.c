#include "libimf_vars.h"

char *IMF_Spec_Labels[] = {"Power Law",
                           "Whatever"};;
IMF_SPEC IMF_Spec;

IMF_Type *IMFs, *IMFp, IMFu;

int IMFs_dim;

int Nof_TimeDep_IMF;

gsl_integration_workspace *limf_w = 0x0;

EXTERNALIMF *externalIMFs_byMass;
EXTERNALIMF *externalIMFs_byNum;
EXTERNALIMF *externalIMFs_byEgy;
char **externalIMFs_names;
int NexternalIMFs = 0;
