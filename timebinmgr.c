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

double dloga_from_dti(int ti)
{
    return ti * logDTime;
}

int dti_from_dloga(double loga)
{
    return loga /logDTime;
}

int ti_from_loga(double loga)
{
    return (loga - logTimeInit)/logDTime;
}

double get_dloga_for_bin(int timebin)
{
    return (timebin ? (1 << timebin) : 0 ) * logDTime;
}

int enforce_power_of_two(int dti)
{
    /* make dti a power 2 subdivision */
    int ti_min = TIMEBASE;
    while(ti_min > dti)
        ti_min >>= 1;
    return ti_min;
}
