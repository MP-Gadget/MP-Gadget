#ifndef TIMEBINMGR_H
#define TIMEBINMGR_H
/* This file manages the integer timeline, 
 * and converts from integers ti to double loga.*/

/*!< The simulated timespan is mapped onto the integer interval [0,TIMEBASE],
 *   where TIMEBASE needs to be a power of 2. Note that (1<<28) corresponds
 *   to 2^29.
 *   We allow some bits at the top of the integer timeline for snapshot outputs.
 *   Note that because each snapshot uses TIMEBASE on the integer timeline, the conversion
 *   factor between loga and ti is not constant across snapshots.
 */
#define TIMEBINS 20
#define TIMEBASE (1u<<TIMEBINS)
#define MAXSNAPSHOTS (1u<<(31-TIMEBINS))

/*Convert an integer to and from loga*/
double loga_from_ti(unsigned int ti);
unsigned int ti_from_loga(double loga);

/*Convert changes in loga to and from ti*/
unsigned int dti_from_dloga(double loga);
double dloga_from_dti(unsigned int ti);

/*Get dloga from a timebin*/
double get_dloga_for_bin(int timebin);

/* Enforce that an integer timestep is a power
 * of two subdivision of TIMEBASE, rounding down
 * to the first power of two less than the ti passed in.
 * Note TIMEBASE is the maximum value returned.*/
unsigned int round_down_power_of_two(unsigned int ti);

/*! this function returns the next output time after ti_curr.*/
unsigned int find_next_outputtime(unsigned int ti_curr);

/*Get whatever is the last output number from ti*/
unsigned int out_from_ti(unsigned int ti);
#endif
