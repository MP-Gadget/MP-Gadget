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
#define TIMEBINS 46
#define TIMEBASE (1Lu<<TIMEBINS)
#define MAXSNAPSHOTS (1Lu<<(62-TIMEBINS))

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
    int write_plane;  //! Write a plane
    int plane_snapnum;  //! The snapshot number for the plane
    inttime_t ti;
};

/*Convert an integer to and from loga*/
double loga_from_ti(inttime_t ti);
inttime_t ti_from_loga(double loga);

/*Convert changes in loga to and from ti*/
inttime_t dti_from_dloga(double loga, const inttime_t Ti_Current);
double dloga_from_dti(inttime_t dti, const inttime_t Ti_Current);

/*Get dloga from a timebin*/
double get_dloga_for_bin(int timebin, const inttime_t Ti_Current);

/*Get the dti from the timebin*/
static inline inttime_t dti_from_timebin(int bin) {
    /*Casts to work around bug in intel compiler 18.0*/
    return bin > 0 ? (1Lu << (uint64_t) bin) : 0;
}

/* Enforce that an integer timestep is a power
 * of two subdivision of TIMEBASE, rounding down
 * to the first power of two less than the ti passed in.
 * Note TIMEBASE is the maximum value returned.*/
inttime_t round_down_power_of_two(inttime_t ti);

/*! this function returns the next output time after ti_curr.*/
inttime_t find_next_outputtime(inttime_t ti_curr);

/*Get whatever is the last output number from ti*/
inttime_t out_from_ti(inttime_t ti);

/*! This function parses a string containing a comma-separated list of variables,
 *  each of which is interpreted as a double.
 *  The purpose is to read an array of output times into the code.
 *  So specifying the output list now looks like:
 *  OutputList  0.1,0.3,0.5,1.0
 *
 *  We sort the input after reading it, so that the initial list need not be sorted.
 *  This function could be repurposed for reading generic arrays in future.
 */
int BuildOutputList(ParameterSet* ps, const char* name, double * outputlist, int64_t * outputlistlength, int64_t maxlength);

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
