#ifndef RUN_H
#define RUN_H

#include "domain.h"

void begrun(int RestartSnapNum, Domain * domain);

void run(Domain * domain);
void runtests(Domain * domain);
void runfof(int RestartSnapNum, Domain * domain);

#endif
