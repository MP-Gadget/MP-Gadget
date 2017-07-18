#ifndef TIMEBINMGR_H
#define TIMEBINMGR_H
/* This file manages the integer timeline, 
 * and converts from integers ti to double loga.*/

/*!< The simulated timespan is mapped onto the integer interval [0,TIMEBASE],
 *   where TIMEBASE needs to be a power of 2. Note that (1<<28) corresponds
 *   to 2^29
 */
#define TIMEBINS 29
#define TIMEBASE (1<<TIMEBINS)

void init_integer_timeline(double TimeInit, double TimeMax);

double loga_from_ti(int ti);

int ti_from_loga(double loga);

#endif
