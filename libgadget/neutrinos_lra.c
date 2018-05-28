#include <math.h>
#include <bigfile-mpi.h>

#include "neutrinos_lra.h"

#include "utils/endrun.h"
#include "utils/mymalloc.h"
#include "petaio.h"

/*Structure which holds the neutrino state*/
_delta_tot_table delta_tot_table;
static _transfer_init_table t_init_data;
static _transfer_init_table * t_init = &t_init_data;

/* Constructor for delta_tot. Does some sanity checks and initialises delta_nu_last.*/
void delta_tot_resume(_delta_tot_table * const d_tot, const int nk_in, const double wavenum[])
{
    int ik;
    if(nk_in > d_tot->nk_allocated){
           endrun(2011,"input power of %d is longer than memory of %d\n",nk_in,d_tot->nk_allocated);
    }
    if(d_tot->nk != nk_in)
        endrun(201, "Number of neutrino bins %d != stored value %d\n",nk_in, d_tot->nk);

    /*Set the wave number*/
    for(ik=0;ik<d_tot->nk;ik++){
        d_tot->wavenum[ik] = wavenum[ik];
    }
    /*Initialise delta_nu_last*/
    get_delta_nu_combined(d_tot, exp(d_tot->scalefact[d_tot->ia-1]), wavenum, d_tot->delta_nu_last);
    d_tot->delta_tot_init_done=1;
    return;
}

/* Constructor. transfer_init_tabulate must be called before this function.
 * Initialises delta_tot (including from a file) and delta_nu_init from the transfer functions.
 * read_all_nu_state must be called before this if you want reloading from a snapshot to work
 * Note delta_cdm_curr includes baryons, and is only used if not resuming.*/
void delta_tot_first_init(_delta_tot_table * const d_tot, const int nk_in, const double wavenum[], const double delta_cdm_curr[], const double TimeIC)
{
    int ik;
    if(nk_in > d_tot->nk_allocated){
           endrun(2011,"input power of %d is longer than memory of %d\n",nk_in,d_tot->nk_allocated);
    }
    d_tot->nk=nk_in;
    const double OmegaNua3=get_omega_nu_nopart(d_tot->omnu, d_tot->TimeTransfer)*pow(d_tot->TimeTransfer,3);
    const double OmegaNu1 = get_omega_nu(d_tot->omnu, 1);
    gsl_interp_accel *acc = gsl_interp_accel_alloc();
    gsl_interp * spline=gsl_interp_alloc(gsl_interp_cspline,t_init->NPowerTable);
    gsl_interp_init(spline,t_init->logk,t_init->T_nu,t_init->NPowerTable);
    /*Check we have a long enough power table*/
    if(log(wavenum[d_tot->nk-1]) > t_init->logk[t_init->NPowerTable-1])
        endrun(2,"Want k = %g but maximum in CAMB table is %g\n",wavenum[d_tot->nk-1], exp(t_init->logk[t_init->NPowerTable-1]));
    for(ik=0;ik<d_tot->nk;ik++) {
            /* T_nu contains T_nu / T_cdm.*/
            double T_nubyT_nonu = gsl_interp_eval(spline,t_init->logk,t_init->T_nu,log(wavenum[ik]),acc);
            /*Initialise delta_nu_init to use the first timestep's delta_cdm_curr
             * so that it includes potential Rayleigh scattering. */
            d_tot->delta_nu_init[ik] = delta_cdm_curr[ik]*T_nubyT_nonu;
            const double partnu = particle_nu_fraction(&d_tot->omnu->hybnu, TimeIC, 0);
            /*Initialise the first delta_tot*/
            d_tot->delta_tot[ik][0] = get_delta_tot(d_tot->delta_nu_init[ik], delta_cdm_curr[ik], OmegaNua3, d_tot->Omeganonu, OmegaNu1, partnu);
            d_tot->wavenum[ik] = wavenum[ik];
    }
    gsl_interp_accel_free(acc);
    gsl_interp_free(spline);

    /*If we are not restarting, make sure we set the scale factor*/
    d_tot->scalefact[0]=log(TimeIC);
    d_tot->ia=1;
    /*Initialise delta_nu_last*/
    get_delta_nu_combined(d_tot, exp(d_tot->scalefact[0]), wavenum, d_tot->delta_nu_last);
    myfree(t_init->logk);
    d_tot->delta_tot_init_done=1;
    return;
}

void petaio_save_neutrinos(BigFile * bf, int ThisTask)
{
#pragma omp master
    {
    double * scalefact = delta_tot_table.scalefact;
    size_t nk = delta_tot_table.nk, ia = delta_tot_table.ia;
    size_t ik, i;
    double * delta_tot = mymalloc("tmp_delta",nk * ia * sizeof(double));
    /*Save a flat memory block*/
    for(ik=0;ik< nk;ik++)
        for(i=0;i< ia;i++)
            delta_tot[ik*ia+i] = delta_tot_table.delta_tot[ik][i];

    BigBlock bn = {0};
    if(0 != big_file_mpi_create_block(bf, &bn, "Neutrino", NULL, 0, 0, 0, MPI_COMM_WORLD)) {
        endrun(0, "Failed to create block at %s:%s\n", "Neutrino",
                big_file_get_error_message());
    }
    if ( (0 != big_block_set_attr(&bn, "Nscale", &ia, "u8", 1)) ||
       (0 != big_block_set_attr(&bn, "scalefact", scalefact, "f8", ia)) ||
        (0 != big_block_set_attr(&bn, "Nkval", &nk, "u8", 1)) ) {
        endrun(0, "Failed to write neutrino attributes %s\n",
                    big_file_get_error_message());
    }
    if(0 != big_block_mpi_close(&bn, MPI_COMM_WORLD)) {
        endrun(0, "Failed to close block %s\n",
                    big_file_get_error_message());
    }
    BigArray deltas = {0};
    size_t dims[2] = {0 , ia};
    /*The neutrino state is shared between all processors,
     *so only write on master task*/
    if(ThisTask == 0) {
        dims[0] = nk;
    }
    ptrdiff_t strides[2] = {sizeof(double) * ia, sizeof(double)};
    big_array_init(&deltas, delta_tot, "=f8", 2, dims, strides);
    petaio_save_block(bf, "Neutrino/Deltas", &deltas);
    myfree(delta_tot);
    /*Now write the initial neutrino power*/
    BigArray delta_nu = {0};
    dims[1] = 1;
    strides[0] = sizeof(double);
    big_array_init(&delta_nu, delta_tot_table.delta_nu_init, "=f8", 2, dims, strides);
    petaio_save_block(bf, "Neutrino/DeltaNuInit", &delta_nu);
    /*Now write the k values*/
    BigArray kvalue = {0};
    big_array_init(&kvalue, delta_tot_table.wavenum, "=f8", 2, dims, strides);
    petaio_save_block(bf, "Neutrino/kvalue", &kvalue);
    }
}

void petaio_read_icnutransfer(BigFile * bf, int ThisTask)
{
#pragma omp master
    {
    t_init->NPowerTable = 2;
    BigBlock bn = {{0}};
    /* Read the size of the ICTransfer block.
     * If we can't read it, just set it to zero*/
    if(0 == big_file_mpi_open_block(bf, &bn, "ICTransfers", MPI_COMM_WORLD)) {
        if(0 != big_block_get_attr(&bn, "Nentry", &t_init->NPowerTable, "u8", 1))
            endrun(0, "Failed to read attr: %s\n", big_file_get_error_message());
        if(0 != big_block_mpi_close(&bn, MPI_COMM_WORLD))
            endrun(0, "Failed to close block %s\n",big_file_get_error_message());
    }
    message(1,"Found transfer function, using %d rows.\n", t_init->NPowerTable);
    t_init->logk = (double *) mymalloc2("Transfer_functions", 2*t_init->NPowerTable* sizeof(double));
    t_init->T_nu=t_init->logk+t_init->NPowerTable;

    /*Defaults: zero*/
    t_init->logk[0] = -100;
    t_init->logk[t_init->NPowerTable-1] = 100;

    t_init->T_nu[0] = 0;
    t_init->T_nu[t_init->NPowerTable-1] = 0;

    /*Now read the arrays*/
    BigArray Tnu = {0};
    BigArray logk = {0};

    size_t dims[2] = {0, 1};
    ptrdiff_t strides[2] = {sizeof(double), sizeof(double)};
    /*The neutrino state is shared between all processors,
     *so only read on master task and broadcast*/
    if(ThisTask == 0) {
        dims[0] = t_init->NPowerTable;
    }
    big_array_init(&Tnu, t_init->T_nu, "=f8", 2, dims, strides);
    big_array_init(&logk, t_init->logk, "=f8", 2, dims, strides);
    /*This is delta_nu / delta_tot: note technically the most massive eigenstate only.
     * But if there are significant differences the mass is so low this is basically zero.*/
    petaio_read_block(bf, "ICTransfers/DELTA_NU", &Tnu, 0);
    petaio_read_block(bf, "ICTransfers/logk", &logk, 0);
    /*Also want d_{cdm+bar} / d_tot so we can get d_nu/(d_cdm+d_b)*/
    double * T_cb = (double *) mymalloc("tmp1", t_init->NPowerTable* sizeof(double));
    BigArray Tcb = {0};
    big_array_init(&Tcb, T_cb, "=f8", 2, dims, strides);
    petaio_read_block(bf, "ICTransfers/DELTA_CB", &Tcb, 0);
    int i;
    for(i = 0; i < t_init->NPowerTable; i++)
        t_init->T_nu[i] /= T_cb[i];
    myfree(T_cb);
    /*Broadcast the arrays.*/
    MPI_Bcast(t_init->logk,2*t_init->NPowerTable,MPI_DOUBLE,0,MPI_COMM_WORLD);
    }
}

/*Read the neutrino data from the snapshot*/
void petaio_read_neutrinos(BigFile * bf, int ThisTask)
{
#pragma omp master
    {
    size_t nk, ia, ik, i;
    BigBlock bn = {0};
    if(0 != big_file_mpi_open_block(bf, &bn, "Neutrino", MPI_COMM_WORLD)) {
        endrun(0, "Failed to open block at %s:%s\n", "Neutrino",
                    big_file_get_error_message());
    }
    if(
    (0 != big_block_get_attr(&bn, "Nscale", &ia, "u8", 1)) ||
    (0 != big_block_get_attr(&bn, "Nkval", &nk, "u8", 1))) {
        endrun(0, "Failed to read attr: %s\n",
                    big_file_get_error_message());
    }
    double *delta_tot = (double *) mymalloc("tmp_nusave",ia*nk*sizeof(double));
    /*Allocate list of scale factors, and space for delta_tot, in one operation.*/
    if(0 != big_block_get_attr(&bn, "scalefact", delta_tot_table.scalefact, "f8", ia))
        endrun(0, "Failed to read attr: %s\n", big_file_get_error_message());
    if(0 != big_block_mpi_close(&bn, MPI_COMM_WORLD)) {
        endrun(0, "Failed to close block %s\n",
                    big_file_get_error_message());
    }
    BigArray deltas = {0};
    size_t dims[2] = {0, ia};
    ptrdiff_t strides[2] = {sizeof(double)*ia, sizeof(double)};
    /*The neutrino state is shared between all processors,
     *so only read on master task and broadcast*/
    if(ThisTask == 0) {
        dims[0] = nk;
    }
    big_array_init(&deltas, delta_tot, "=f8", 2, dims, strides);
    petaio_read_block(bf, "Neutrino/Deltas", &deltas, 1);

    /*Save a flat memory block*/
    for(ik=0;ik<nk;ik++)
        for(i=0;i<ia;i++)
            delta_tot_table.delta_tot[ik][i] = delta_tot[ik*ia+i];
    delta_tot_table.nk = nk;
    delta_tot_table.ia = ia;
    myfree(delta_tot);
    /* Read the initial delta_nu. This is basically zero anyway,
     * so for backwards compatibility do not require it*/
    BigArray delta_nu = {0};
    dims[1] = 1;
    strides[0] = sizeof(double);
    memset(delta_tot_table.delta_nu_init, 0, delta_tot_table.nk);
    big_array_init(&delta_nu, delta_tot_table.delta_nu_init, "=f8", 2, dims, strides);
    petaio_read_block(bf, "Neutrino/DeltaNuInit", &delta_nu, 0);
    /* Read the k values*/
    BigArray kvalue = {0};
    memset(delta_tot_table.wavenum, 0, delta_tot_table.nk);
    big_array_init(&kvalue, delta_tot_table.wavenum, "=f8", 2, dims, strides);
    petaio_read_block(bf, "Neutrino/kvalue", &kvalue, 0);

    /*Broadcast the arrays.*/
    MPI_Bcast(&(delta_tot_table.ia), 1,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Bcast(&(delta_tot_table.nk), 1,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Bcast(delta_tot_table.delta_nu_init,delta_tot_table.nk,MPI_DOUBLE,0,MPI_COMM_WORLD);

    if(delta_tot_table.ia > 0) {
        /*Broadcast data for scalefact and delta_tot, Delta_tot is allocated as the same block of memory as scalefact.
          Not all this memory will actually have been used, but it is easiest to bcast all of it.*/
        MPI_Bcast(delta_tot_table.scalefact,delta_tot_table.namax*(delta_tot_table.nk+1),MPI_DOUBLE,0,MPI_COMM_WORLD);
    }
    }
}
