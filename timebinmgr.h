#ifndef TIMEBINMGR_H
#define TIMEBINMGR_H
/* This file manages the integer timeline, 
 * and converts from integers ti to double loga.*/

/*!< The simulated timespan is mapped onto the integer interval [0,TIMEBASE],
 *   where TIMEBASE needs to be a power of 2. Note that (1<<28) corresponds
 *   to 2^29.
 *   We allow some bits at the top of the integer timeline for snapshot outputs
 */
#define MAXSNAPSHOTS (1<<9)
#define TIMEBINS 20
#define TIMEBASE (1<<TIMEBINS)

/*Convert an integer to and from loga*/
double loga_from_ti(int ti);
int ti_from_loga(double loga);

/*Convert changes in loga to and from ti*/
int dti_from_dloga(double loga);
double dloga_from_dti(int ti);

/*Get dloga from a timebin*/
double get_dloga_for_bin(int timebin);

/* Enforce that an integer timestep is a power
 * of two subdivision of TIMEBASE.
 * Note TIMEBASE is the maximum value returned.*/
int enforce_power_of_two(int ti);

/*! this function returns the next output time after ti_curr.*/
int find_next_outputtime(int ti_curr);

/*Get whatever is the last output number from ti*/
int out_from_ti(int ti);
#endif
