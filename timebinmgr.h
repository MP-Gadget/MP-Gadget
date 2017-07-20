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

/*Initialise the conversion factors from loga to integers*/
void init_integer_timeline(double TimeInit, double TimeMax);

/*Convert an integer to and from loga*/
double loga_from_ti(int ti);
int ti_from_loga(double loga);

/*Convert changes in loga to and from ti*/
int dti_from_dloga(double loga);
double dloga_from_dti(int ti);

/*Get dloga from a timebin*/
double get_dloga_for_bin(int timebin);

/*Enforce that an integer time is a power of two*/
int enforce_power_of_two(int ti);
#endif
