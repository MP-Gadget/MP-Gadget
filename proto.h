#ifndef PROTO_H
#define PROTO_H

void report_VmRSS(void);

#ifdef MPISENDRECV_CHECKSUM
int MPI_Check_Sendrecv(void *sendbuf, int sendcount, MPI_Datatype sendtype,
                       int dest, int sendtag, void *recvbufreal, int recvcount,
                       MPI_Datatype recvtype, int source, int recvtag, MPI_Comm comm, MPI_Status * status);
#define MPI_Sendrecv MPI_Check_Sendrecv
#endif

#ifdef MPISENDRECV_SIZELIMIT
int MPI_Sizelimited_Sendrecv(void *sendbuf, int sendcount, MPI_Datatype sendtype,
			     int dest, int sendtag, void *recvbuf, int recvcount,
			     MPI_Datatype recvtype, int source, int recvtag, MPI_Comm comm, MPI_Status * status);
#define MPI_Sendrecv MPI_Sizelimited_Sendrecv
#endif

int drift_particle_full(int i, int time1, int blocking);
void drift_particle(int i, int time1);
void lock_particle_if_not(int i, MyIDType id);
void unlock_particle_if_not(int i, MyIDType id);

int ShouldWeDoDynamicUpdate(void);

void write_cpu_log(void);

void do_the_kick(int i, int tstart, int tend, int tcurrent);


void x86_fix(void) ;

void blackhole_accretion(void);
void blackhole_make_one(int index);

int  blackhole_compare_key(const void *a, const void *b);

double get_random_number(MyIDType id);
void set_random_numbers(void);

int data_index_compare(const void *a, const void *b);

void mysort_dataindex(void *b, size_t n, size_t s, int (*cmp) (const void *, const void *));

size_t my_fwrite(void *ptr, size_t size, size_t nmemb, FILE * stream);
size_t my_fread(void *ptr, size_t size, size_t nmemb, FILE * stream);

double density_decide_hsearch(int targettype, double h);

size_t sizemax(size_t a, size_t b);

void reconstruct_timebins(void);

void catch_abort(int sig);
void catch_fatal(int sig);
void terminate_processes(void);
void enable_core_dumps_and_fpu_exceptions(void);
void write_pid_file(void);

void move_particles(int time1);

void find_next_sync_point_and_drift(void);
void find_dt_displacement_constraint(double hfac);

void set_units_sfr(void);

void allocate_memory(void);
void begrun(void);
void check_omega(void);
void close_outputfiles(void);
void compute_accelerations(int mode);
void compute_global_quantities_of_system(void);
void construct_timetree(void);
void cooling_and_starformation(void);
void cooling_only(void);
void density(void);
void do_box_wrapping(void);
void domain_Decomposition(void);
void energy_statistics(void);

void every_timestep_stuff(void);

int find_next_outputtime(int time);
void free_memory(void);
void set_global_time(double newtime);
void advance_and_find_timesteps(void);
int get_timestep(int p, double *a, int flag);
int get_timestep_bin(int ti_step);

double get_starformation_rate(int i);
void gravity_tree(void);
void hydro_force(void);
void init(void);
void init_clouds(void);
void integrate_sfr(void);
void open_outputfiles(void);
void peano_hilbert_order(void);
int read_outputlist(char *fname);
void read_parameter_file(char *fname);
void reorder_gas(void);
void reorder_particles(void);
void restart(int mod);
void run(void);
void savepositions(int num, int reason);
double second(void);
void set_softenings(void);
void set_units(void);
void setup_smoothinglengths(void);
void fof_fof(int num);

void sumup_large_ints(int n, int *src, int64_t *res);
void sumup_longs(int n, int64_t *src, int64_t *res);
int64_t count_to_offset(int64_t countLocal);
int64_t count_sum(int64_t countLocal);

int MPI_Alltoallv_smart(void *sendbuf, int *sendcnts, int *sdispls,
        MPI_Datatype sendtype, void *recvbuf, int *recvcnts,
        int *rdispls, MPI_Datatype recvtype, MPI_Comm comm);

int MPI_Alltoallv_sparse(void *sendbuf, int *sendcnts, int *sdispls,
        MPI_Datatype sendtype, void *recvbuf, int *recvcnts,
        int *rdispls, MPI_Datatype recvtype, MPI_Comm comm);

static inline int atomic_fetch_and_add(int * ptr, int value) {
    int k;
#if _OPENMP >= 201107
#pragma omp atomic capture
    {
      k = (*ptr);
      (*ptr)+=value;
    }
#else
#ifdef OPENMP_USE_SPINLOCK
    k = __sync_fetch_and_add(ptr, value);
#else /* non spinlock*/
#pragma omp critical
    {
      k = (*ptr);
      (*ptr)+=value;
    }
#endif
#endif
    return k;
}
static inline int atomic_add_and_fetch(int * ptr, int value) {
    int k;
#if _OPENMP >= 201107
#pragma omp atomic capture
    {
      (*ptr)+=value;
      k = (*ptr);
    }
#else
#ifdef OPENMP_USE_SPINLOCK
    k = __sync_add_and_fetch(ptr, value);
#else /* non spinlock */
#pragma omp critical
    {
      (*ptr)+=value;
      k = (*ptr);
    }
#endif
#endif
    return k;
}

double timediff(double t0, double t1);

double get_hydrokick_factor(int time0, int time1);
double get_gravkick_factor(int time0, int time1);
double drift_integ(double a, void *param);
double gravkick_integ(double a, void *param);
double hydrokick_integ(double a, void *param);
void init_drift_table(void);
double get_drift_factor(int time0, int time1);
double measure_time(void);

void long_range_init(void);
void long_range_force(void);

void readjust_timebase(double TimeMax_old, double TimeMax_new);

#ifdef LIGHTCONE
void lightcone_init();
void lightcone_cross(int p, double oldpos[3]);
void lightcone_set_time(double a);
#endif
#endif //PROTO_H
