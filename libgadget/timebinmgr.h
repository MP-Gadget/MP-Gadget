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
#define TIMEBINS 30
#define TIMEBASE (1<<TIMEBINS)
/* Now sync point is a 32 bit number*/
#define MAXSNAPSHOTS (1Lu<<30)

#include "types.h"
#include "utils/paramset.h"
#include "cosmology.h"

typedef struct SyncPoint SyncPoint;

struct SyncPoint
{
    double a;
    double loga;
    int write_snapshot;
    int write_fof;
    int calc_uvbg;  //! Calculate the UV background
};


/*Convert an integer to and from loga*/
double loga_from_ti(inttime_t ti);
inttime_t ti_from_loga(double loga);

/*Convert changes in loga to and from ti. Largest possible value returned is TIMEBASE.*/
dti_t dti_from_dloga(double loga, const inttime_t Ti_Current);
/* Find dloga from inttime start to inttime end*/
double dloga_from_dti(const inttime_t start, const inttime_t end);

/* Move forward or backwards on an integer timeline.
 * If dti reaches TIMEBASE, increment to the next syncpoint and reset dti to 0.
 * If dti reaches 0, decrement to the next syncpoint and reset dti to TIMEBASE.*/
inttime_t add_dti_and_inttime(inttime_t start, dti_t diff);
/* Compare two integer times. Returns +1 if a > b, 0 if a == b, -1 if a < b, like a sorting function*/
static inline int compare_two_inttime(inttime_t a, inttime_t b)
{
    if(a.lastsnap > b.lastsnap)
        return 1;
    if(a.lastsnap < b.lastsnap)
        return -1;
    if(a.dti > b.dti)
        return 1;
    if(a.dti < b.dti)
        return -1;
    return 0;
}

/*Get dloga from a timebin*/
double get_dloga_for_bin(int timebin, const inttime_t Ti_Current);

/*Get the dti from the timebin*/
static inline dti_t dti_from_timebin(int bin) {
    /*Casts to work around bug in intel compiler 18.0*/
    return bin > 0 ? (1u << (unsigned) bin) : 0;
}

/* Enforce that an integer timestep is a power
 * of two subdivision of TIMEBASE, rounding down
 * to the first power of two less than the ti passed in.
 * Note TIMEBASE is the maximum value returned.*/
dti_t round_down_power_of_two(dti_t ti);

/*! this function returns the next output time after ti_curr.*/
inttime_t find_next_outputtime(inttime_t ti_curr);

/*Get whatever is the last output number from ti*/
inttime_t out_from_ti(inttime_t ti);

int OutputListAction(ParameterSet * ps, const char * name, void * data);
void set_sync_params_test(int OutputListLength, double * OutputListTimes);
void set_sync_params(ParameterSet * ps);
void setup_sync_points(Cosmology * CP, double TimeIC, double TimeMax, double no_snapshot_until_time, int SnapshotWithFOF);

SyncPoint *
find_next_sync_point(inttime_t ti);

SyncPoint *
find_current_sync_point(inttime_t ti);

SyncPoint *
make_unplanned_sync_point(inttime_t ti);

#endif
