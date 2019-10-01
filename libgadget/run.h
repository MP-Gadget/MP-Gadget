#ifndef RUN_H
#define RUN_H

#include "domain.h"

void begrun(int RestartSnapNum, DomainDecomp * ddecomp);

void run(DomainDecomp * ddecomp);
void runtests(int RestartSnapNum, DomainDecomp * ddecomp);
void runfof(int RestartSnapNum, DomainDecomp * ddecomp);

#endif
