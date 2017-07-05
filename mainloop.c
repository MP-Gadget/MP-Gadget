
#include "allvars.h"
#include "timestep.h"

typedef struct GadgetMainLoopPrivate GadgetMainLoopPrivate;
typedef struct GadgetMainLoop GadgetMainLoop;

typedef int (*MonitorFunc)(GadgetMainLoop * mainloop, int ti, void * userdata);
typedef int (*BinTaintedEventFunc)(GadgetMainLoop * mainloop, int bin, void * userdata);

struct GadgetMainLoop {
    GadgetMainLoopPrivate * priv;
    int NTimeBin;
    int TiBase;
    MonitorFunc monitor;
    void * monitor_data;
    BinTaintedEventFunc bin_tainted_event_func;
    void * bin_tainted_event_data;

    /* These variables will change during evolution of the loop. */
    int NumPart;
    int * NextActive;
    int BinStart[32];
    int BinLength[32];
    int BinTiKick[32];
    int BinTiDrift[32];
    double loga0;
    double loga1;
};

struct GadgetMainLoopPrivate {
    int foo;
};

#define PRIV(loop) (((GadgetMainLoop*)loop)->priv)

static int
is_step_edge(GadgetMainLoop * loop, int ti, int bin)
{
    int t = loop->NTimeBin - bin;
    return ((ti % (1 << t)) == 0);
}

static int
is_step_center(GadgetMainLoop * loop, int ti, int bin)
{
    return is_step_edge(loop, ti, bin + 1) && ~ is_step_edge(loop, ti, bin);
}

/* prepend to the list, returns new list head. */
static int
prepend(GadgetMainLoop * loop, int binstart, int i, int * binlength)
{
    loop->NextActive[i] = binstart;
    *binlength ++;
    return i;
}

static int
ti_to_bin(GadgetMainLoop * loop, int dti)
{
   int bin = loop->NTimeBin;

   while(dti) {
       bin--;
       dti >>= 1;
   }

   return bin;
}


GadgetMainLoop *
gadget_main_loop_new(int NTimeBin, struct global_data_all_processes A)
{
    if(NTimeBin > 32) { /* not supported */ abort(); }

    GadgetMainLoop * loop = malloc(sizeof(GadgetMainLoop));
    loop->priv = malloc(sizeof(GadgetMainLoopPrivate));
    loop->NTimeBin = NTimeBin;
    loop->NextActive = malloc(sizeof(int) * A.MaxPart);
    loop->TiBase = 1 << NTimeBin;
    return loop;
}

void
gadget_main_loop_free(GadgetMainLoop * loop)
{

    free(loop->NextActive);
    free(loop->priv);
    free(loop);
}

void
gadget_main_loop_reset_bins(GadgetMainLoop * loop, int NumPart)
{
    int i;
    for(i = 0; i < NumPart; i ++) {
        loop->NextActive[i] = i + 1;
    }
    for(i = 0; i < loop->NTimeBin; i ++) {
        loop->BinStart[i] = -1;
        loop->BinLength[i] = 0;
        loop->BinTiKick[i] = 0;
        loop->BinTiDrift[i] = 0;
    }
    loop->BinStart[0] = 0;
    loop->BinLength[0] = NumPart;
    loop->NumPart = NumPart;

    for(i = 0; i < NumPart; i ++) {
        int bin = P[i].TimeBin;
        loop->BinStart[bin] = prepend(loop, loop->BinStart[bin], i, &loop->BinLength[bin]);
    }
}

static int
get_next(GadgetMainLoop * loop, int bin, int current)
{
    if(current == -1) return loop->BinStart[bin];
    else return loop->NextActive[current];
}

static int
pop_first(GadgetMainLoop * loop, int bin)
{

    int i = loop->BinStart[bin];
    if (i != -1) {
        loop->BinStart[bin] = loop->NextActive[i];
        loop->NextActive[i] = -1;
    }
    return i;
}

void
gadget_main_loop_set_time_range(GadgetMainLoop * loop, double loga0, double dloga)
{
    loop->loga0 = loga0;
    loop->loga1 = loga0 + dloga;
}



/* This is the main loop that runs the PM steps. */
void
gadget_main_loop_run(GadgetMainLoop * loop)
{
    walltime_measure("/Misc");

    domain_balance();

    pm_force();

    do {

        double loga = log(All.Time);

        double dloga = 1.;
        dloga = fmin(dloga, find_dloga_displacement_constraint());
        dloga = fmin(dloga, find_dloga_output_constraint()); /* round to the next output time */

        gadget_main_loop_set_time_range(loop, loga, dloga);

        int i;

        /* put all particle to bin 0, will update in the first ti interaction */
        for(i = 0; i < NumPart; i ++) {
            P[i].TimeBin = 0;
        }

        gadget_main_loop_reset_bins(loop, NumPart);

        pm_kick(loga, loga + dloga * 0.5);

        gadget_main_loop_run_shortrange();

        /* all particle position changed, redo domain decosmposition */
        domain_balance();

        pm_force();

        pm_kick(loga + dloga * 0.5, loga + dloga);

        fof_seed();

        checkout_output();

    } while(1);
}

static int
find_minimal_step_level(GadgetMainLoop * loop)
{
    int i;
    for(i = loop->NTimeBin - 1; i --; i >= 0) {
        if (loop->BinLength[i]) return i;
    }
    /* no particles are in any bins. XXX: assert loop->NumPart == 0*/
    return 0;
}

static void
compute_step_sizes(GadgetMainLoop * loop, int minbin)
{
    /* this will rebuild the timebin starting from minbin
     * no particle is moved to bins slower than minbin;
     * to ensure momentum conservtion (I think). */

    int i;
    int bin = minbin;
    int BinStart[32];
    int BinLength[32];
    for(bin = minbin; bin < loop->NTimeBin; bin ++) {
        BinStart[bin] = -1;
        BinLength[bin] = 0;
    }

    for(bin = minbin; bin < loop->NTimeBin; bin ++) {
        while(1) {
            i = pop_first(loop, bin);

            if(i == -1) break;

            /* FIXME: Use the symmetirzed version from HOLD,
             * which requires a tree.
             * */
            double dloga = get_timestep(i);
            int dti = dloga / (loop->loga1 - loop->loga0) * loop->TiBase;
            int newbin = ti_to_bin(loop, dti);
            /* never go beyond the bin being rebuilt */
            if (newbin < minbin) newbin = minbin;

            BinStart[newbin] = prepend(loop, BinStart[newbin], i, &BinLength[newbin]);

            /* remembers the timebin - useful for pruning tree and for rebuilding
             * after domain exchange*/
            P[i].TimeBin = newbin;
        }
    }

    for(bin = minbin; bin < loop->NTimeBin; bin ++) {
        loop->BinStart[bin] = BinStart[bin];
        loop->BinLength[bin] = BinLength[bin];
    }
}

static void
compute_accelerations(GadgetMainLoop * loop, int fg, binmask_t bg)
{
    liface_set_activelist(loop, BINMASK(fg));
    /* probably import to run hydro before grav such that prediction
     * if done is using the correct acceleration */
    hydro_force(bg);

    grav_short_pair(bg);

    cooling_and_starformation();

    blackhole();
}

static void
liface_build_activelist(GadgetMainLoop * loop, binmask_t binmask)
{
    /* this calls density and rebuild_activelist from timestep.c*/
    set_timebin_active(binmask);
    rebuild_activelist();
}

static void
short_range_kick(GadgetMainLoop * loop, int fg, binmask_t bg, int dti)
{
    double dloga = dti * (loop->loga1 - loop->loga0) / loop->TiBase;
    double loga0 = loop->BinTiKick[fg] * (loop->loga1 - loop->loga0) / loop->TiBase;

    double hydrK = hydr_get_kick_factor(loga0, dloga);
    double gravK = grav_get_kick_factor(loga0, dloga);
    double entrK = entr_get_kick_factor(loga0, dloga);

    compute_accelerations(loop, fg, bg);

    int i = -1;

    while(1) {
        i = get_next(loop, fg, i);

        if(i == -1) break;

        hydr_kick(i, hydrK);
        grav_kick(i, gravK);
        entr_kick(i, entrK);
    }

    if(BINMASK(fg) & bg) {
        /* the force on fg bin by any bins faster than fg has been added;
         * update the book keeping var.
         * */
        loop->BinTiKick[fg] += dti;
    }
}

static void
drift(GadgetMainLoop * loop, int fg, int dti)
{
    double dloga = dti * (loop->loga1 - loop->loga0) / loop->TiBase;
    double loga0 = loop->BinTiDrift[fg] * (loop->loga1 - loop->loga0) / loop->TiBase;

    double D = gadget_get_drift_factor(loga0, dloga);

    while(1) {
        int i = get_next(loop, fg, i);

        if(i == -1) break;

        hydr_drift(i, D);
        grav_drift(i, D);
        entr_drift(i, D);
    }

    gadget_tree_free(fg);

    /* maintains the domain invariance, without rebuilding the domain */
    gadget_domain_exchange(fg);

    /* currently domain_exchange breaks the Bins, so we rebuild them all with
     * The TimeBin value of P */
    gadget_main_loop_reset_bins(loop, NumPart);

    gadget_tree_build(fg);
}


static void
kdk(GadgetMainLoop * loop, int slow, int fast)
{
    int dti = loop->TiBase >> slow;
    int hdti = loop->TiBase >> fast;


    binmask_t binmask = BINMASK(slow) | BINMASK(fast);

    liface_set_activelist(loop, binmask);
    density();

    short_range_kick(loop, fast, BINMASK(slow), hdti);
    short_range_kick(loop, slow, BINMASK(fast) | BINMASK(slow), hdti);

    drift(loop, slow, dti);

    liface_set_activelist(loop, binmask);
    density();

    short_range_kick(loop, fast, BINMASK(slow), hdti);
    short_range_kick(loop, slow, BINMASK(fast) | BINMASK(slow), hdti);
}

static void
sync_empty_timebins(GadgetMainLoop * loop, int binmin)
{
    int dti = loop->TiBase >> binmin;
    int bin;
    for(bin = binmin; bin < loop->NTimeBin; bin ++) {
        loop->BinTiDrift[bin] += dti;
        loop->BinTiKick[bin] += dti;
    }
}

void
gadget_main_loop_run_shortrange(GadgetMainLoop * loop, MonitorFunc monitor)
{
    int ti;
    int TiBase;

    ti = 0; 

    while(ti <= loop->TiBase) {

        int bin;

        int step_level = find_minimal_step_level(loop);

        int monitor_state = 0;

        if (monitor) {
            monitor_state = monitor(loop, ti, loop->monitor_data);
        }

        for(bin = step_level - 1 ; bin >= 0; bin--) {
            if (is_step_center(loop, ti, bin)) {
                kdk(loop, bin, bin + 1);
            }
        }

        for(bin = 0; bin <= step_level; bin ++) {
            if (is_step_edge(loop, ti, bin)) break;
        }

        /* rebuild time bins for any >= bin*/
        compute_step_sizes(loop, bin);

        step_level = find_minimal_step_level(loop);

        sync_empty_timebins(loop, step_level);

        ti += (loop->TiBase >> step_level);
    }

}
