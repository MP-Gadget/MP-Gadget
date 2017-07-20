#include "timebinmgr.h"
#include <math.h>
#include "allvars.h"

/*Each integer time stores in the first 10 bits the snapshot number.
 * Then the rest of the bits are the standard integer timeline,
 * which should be a power-of-two hierarchy.*/

unsigned int out_from_ti(unsigned int ti)
{
    return ti >> TIMEBINS;
}

static double logDTime_from_ti(unsigned int ti)
{
    unsigned int lastsnap = ti >> TIMEBINS;
    /*Use logDTime from the last valid interval*/
    if(lastsnap >= All.OutputListLength-1)
        lastsnap = All.OutputListLength - 2;
    double lastoutput = All.OutputListTimes[lastsnap];
    return (All.OutputListTimes[lastsnap+1] - lastoutput)/TIMEBASE;
}

double loga_from_ti(unsigned int ti)
{
    unsigned int lastsnap = ti >> TIMEBINS;
    if(lastsnap >= All.OutputListLength)
        lastsnap = All.OutputListLength - 1;
    double lastoutput = All.OutputListTimes[lastsnap];
    double logDTime = logDTime_from_ti(ti);
    return lastoutput + (ti & (TIMEBASE-1)) * logDTime;
}

unsigned int ti_from_loga(double loga)
{
    int i;
    int ti;
    for(i=1; i<All.OutputListLength-1; i++)
    {
        if(All.OutputListTimes[i] > loga)
            break;
    }
    /*If loop didn't trigger, i == All.OutputListLength-1*/
    double logDTime = (All.OutputListTimes[i] - All.OutputListTimes[i-1])/TIMEBASE;
    ti = (i-1) << TIMEBINS;
    /* Note this means if we overrun the end of the timeline,
     * we still get something reasonable*/
    ti += (loga - All.OutputListTimes[i-1])/logDTime;
    return ti;
}

double dloga_from_dti(unsigned int dti)
{
    double loga = loga_from_ti(All.Ti_Current);
    double logap = loga_from_ti(All.Ti_Current+dti);
    return logap - loga;
}

unsigned int dti_from_dloga(double loga)
{
    int ti = ti_from_loga(loga_from_ti(All.Ti_Current));
    int tip = ti_from_loga(loga+loga_from_ti(All.Ti_Current));
    return tip - ti;
}

double get_dloga_for_bin(int timebin)
{
    double logDTime = logDTime_from_ti(All.Ti_Current);
    return (timebin ? (1u << timebin) : 0 ) * logDTime;
}

unsigned int enforce_power_of_two(unsigned int dti)
{
    /* make dti a power 2 subdivision */
    unsigned int ti_min = TIMEBASE;
    while(ti_min > dti)
        ti_min >>= 1;
    return ti_min;
}

/*! this function returns the next output time that is equal or larger to
 *  ti_curr
 */
unsigned int find_next_outputtime(unsigned int ti_curr)
{
    unsigned int lastout = out_from_ti(ti_curr);
    unsigned int nextout = (lastout+1) << TIMEBINS;
    return nextout;
}
