
typedef struct GadgetMainLoopPrivate GadgetMainLoopPrivate;
typedef struct GadgetMainLoop GadgetMainLoop;

typedef int (*MonitorFunc)(GadgetMainLoop * mainloop, int ti, void * userdata);
typedef int (*BinTaintedEventFunc)(GadgetMainLoop * mainloop, int bin, void * userdata);

typedef int binmask_t;

#define TIMEBINMASK(i) (1 << i)

struct GadgetMainLoop {
    GadgetMainLoopPrivate * priv;
    int NTimeBin;
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
};

struct GadgetMainLoopPrivate {
    int foo;
};

#define PRIV(loop) (((GadgetMainLoop*)loop)->priv)

static void
is_step_edge(GadgetMainLoop * loop, int ti, int bin)
{
    t = loop->NTimeBin - bin;
    return ti % (1 << t) == 0;
}

static void
is_step_center(GadgetMainLoop * loop, int ti, int bin)
{
    return is_step_edge(loop, ti, bin + 1) && ~ is_step_edge(loop, ti, bin);
}

GadgetMainLoop *
gadget_main_loop_new(int NTimeBin, struct global_data_all_processes A)
{
    if(NTimeBin > 32) { /* not supported */ abort(); }

    GadgetMainLoop * loop = malloc(sizeof(GadgetMainLoop));
    loop->priv = malloc(sizeof(GadgetMainLoopPrivate));
    loop->NTimeBin = NTimeBin;
    loop->NextActive = malloc(sizeof(int) * A.MaxPart);

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

int
gadget_main_loop_get_next(GadgetMainLoop * loop, int bin, int current)
{
    if(current == -1) return loop->BinStart[bin];
    else return loop->NextActive[current];
}

int
gadget_main_loop_pop_first(GadgetMainLoop * loop, int bin)
{

    int i = loop->BinStart[bin];
    if (i != -1) {
        loop->BinStart[bin] = loop->NextActive[i];
        loop->NextParticle[i] = -1;
    }
    return i;
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

        dloga = 1.;
        dloga = min(dloga, find_dloga_displacement_constraint());
        dloga = min(dloga, find_dloga_output_constraint()); /* round to the next output time */

        gadget_main_loop_set_time_range(loop, loga, dloga);

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
    for(i = loop->NTimeBin - 1; i --; i >= 0) {
        if (loop->BinLength[i]) return i;
    }
    /* no particles are in any bins. XXX: assert loop->NumPart == 0*/
    return 0;
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
    for(bin = minbin; bin < loop->NTimBin; bin ++) {
        BinStart[bin] = -1;
        BinLength[bin] = 0;
    }

    for(bin = minbin; bin < loop->NTimeBin; bin ++) {
        while(1) {
            i = gadget_main_loop_pop_first(loop, bin);

            if(i == -1) break;

            /* FIXME: Use the symmetirzed version from HOLD,
             * which requires a tree.
             * */
            double dloga = get_timestep(i);
            int dti = dloga / loop->dloga * loop->TiBase;
            int newbin = ti_to_bin(loop, dti);
            /* never go beyond the bin being rebuilt */
            if (newbin < minbin) newbin = minbin;

            BinStart[newbin] = prepend(loop, BinStart[newbin], i, &BinLength[newbin]);

            /* remembers the timebin - useful for pruning tree and for rebuilding
             * after domain exchange*/
            P[i].TimeBin = newbin;
        }
    }

    for(bin = minbin; bin < loop->NTimBin; bin ++) {
        loop->BinStart[bin] = BinStart[bin];
        loop->BinLength[bin] = BinLength[bin];
    }
}

static void
compute_accelerations(int fg, bitmask_t bg)
{
    hydro_run(fg, bg);

    gravtree_run(fg, bg);

    sfrcool_run(fg, bg);

    blackholes_run(fg, bg);
}

static void
kdk(GadgetMainLoop * loop, int slow, int fast)
{
    int dti = loop->TiBase >> slow;
    int hdti = loop->TiBase >> fast;

    density_run(TIMEBINMASK(slow) | TIMEBINMASK(fast));

    short_range_kick(fast, TIMEBINMASK(slow), hdti);
    short_range_kick(slow, TIMEBINMASK(fast) | TIMEBINMASK(slow), hdti);

    drift(slow, dti);

    density_run(TIMEBINMASK(slow) | TIMEBINMASK(fast));

    short_range_kick(fast, TIMEBINMASK(slow), hdti);
    short_range_kick(slow, TIMEBINMASK(fast) | TIMEBINMASK(slow), hdti);
}

static void
short_range_kick(GadgetMainLoop * loop, int fg, binmask_t bg, int dti)
{
    double dloga = dti * loop->dloga / loop->TiBase;
    double loga0 = loop->BinTiKick[fg] * loop->dloga / loop->TiBase;

    hydrK = hydr_get_kick_factor(loga0, dloga);
    gravK = grav_get_kick_factor(loga0, dloga);
    entrK = entr_get_kick_factor(loga0, dloga);

    compute_accelerations(fg, bg)

    int i = -1;

    while(1) {
        i = gadget_main_loop_get_next(loop, fg, i);

        if(i == -1) break;

        hydr_kick(i, hydrK);
        grav_kick(i, gravK);
        entr_kick(i, entrK);
    }

    if(TIMEBINMASK(fg) & bg) {
        /* the force on fg bin by any bins faster than fg has been added;
         * update the book keeping var.
         * */
        loop->BinTiKick[fg] += dti;
    }
}

static void
drift(GadgetMainLoop * loop, int fg)
{
    double dloga = dti * loop->dloga / loop->TiBase;
    double loga0 = loop->BinTiDrift[fg] * loop->dloga / loop->TiBase;

    D = gadget_get_drift_factor(loga0, dloga);

    while(1) {
        i = gadget_main_loop_get_next(loop, fg, i);

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
sync_empty_timebins(GadgetMainLoop * loop, int binmin)
{
    int dti = loop->TiBase >> binmin;
    int bin;
    for(bin = minbin; bin < loop->NTimeBin; bin ++) {
        loop->BinTiDrift[bin] += dti;
        loop->BinTiKick[bin] += dti;
    }
}

void
gadget_main_loop_set_time_range(GadgetMainLoop * loop, double loga0, double dloga)
{
    loop->loga0 = loga0;
    loop->loga1 = loga0 + dloga;
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
            monitor_state = monitor(loop, ti);
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

        step_level = find_minimal_step_level();

        sync_empty_timebins(loop, step_level)

        ti += (loop->TiBase >> step_level);
    }

}
