#include "timebinmgr.h"
#include <math.h>

static double logTimeInit;
static double logDTime = 0.0;

void init_integer_timeline(double TimeInit, double TimeMax)
{
  logTimeInit = log(TimeInit);
  double logTimeMax = log(TimeMax);
  logDTime = (logTimeMax - logTimeInit) / TIMEBASE;
}

double loga_from_ti(int ti)
{
    return logTimeInit + ti * logDTime;
}

int ti_from_loga(double loga)
{
    return (loga - logTimeInit)/logDTime;
}
