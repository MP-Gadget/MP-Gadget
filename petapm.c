#ifdef PETA_PM
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
/* do NOT use complex.h it breaks the code */
#include <pfft.h>

#include "allvars.h"
#include "proto.h"

#ifndef PETA_PM_ORDER
#define PETA_PM_ORDER 1
#warning Using low resolution force differentiation kernel. Consider using -DPETA_PM_ORDER=3
#endif

static size_t HighMark_petapm = 0;
static struct Region {
    /* represents a region in the FFT Mesh */
    ptrdiff_t offset[3];
    ptrdiff_t size[3];
    ptrdiff_t strides[3];
    size_t totalsize;
    double * buffer; 
    /* below are used mostly for investigation */
    double center[3];
    double len;
    double hmax;
    int numpart;
    int no; /* node number for debugging */
} real_space_region, fourier_space_region;

static struct Region * regions = NULL;
static int Nregions = 0;
static void region_init_strides(struct Region * region);

/* a layout is the communication object, represent 
 * pencil / cells exchanged  */

struct Layout {
    int NpExport;
    int NpImport;
    int * NpSend;   
    int * NpRecv;
    int * DpSend;
    int * DpRecv;
    struct Pencil * PencilSend;
    struct Pencil * PencilRecv;

    int NcExport;
    int NcImport;
    int * NcSend;
    int * NcRecv;
    int * DcSend;
    int * DcRecv;

    double * BufSend;
    double * BufRecv;
    int * ibuffer;
};

static void layout_prepare (struct Layout * L);
static void layout_build_and_exchange_pencils(struct Layout * L);
static void layout_finish(struct Layout * L);
static void layout_build_and_exchange_cells_to_pfft(struct Layout * L);
static void layout_build_and_exchange_cells_to_local(struct Layout * L);

/* cell_iterator nees to be thread safe !*/
typedef void (* cell_iterator)(double * cell_value, double * comm_buffer);
static void layout_iterate_cells(struct Layout * L, cell_iterator iter);

struct Pencil { /* a pencil starting at offset, with lenght len */
    int offset[3];
    int len;
    int first; 
    int meshbuf_first; /* first pixel in meshbuf */
    int task;
};
static int pencil_cmp_target(const struct Pencil * p1, const struct Pencil * p2);
static int pos_get_target(const int pos[2]);

static int64_t reduce_int64(int64_t input);
/* for debuggin */
static void verify_density_field();

static double * real;
static double * meshbuf;
static size_t meshbufsize;
static pfft_complex * complx;
static pfft_complex * pot_k;
static int fftsize;
static void pm_alloc();
static void pm_free();
static pfft_plan plan_forw, plan_back;
MPI_Comm comm_cart_2d;
static int * ParticleRegion;
static int ThisTask2d[2];
static int NTask2d[2];
static int * (Mesh2Task[2]); /* conversion from real space mesh to task2d,  */
static int * Mesh2K; /* convertion fourier mesh to integer frequency (or K)  */
static MPI_Datatype MPI_PENCIL;

static double pot_factor;

void petapm_init_periodic(void) {

    /* define the global long / short range force cut */

    All.Asmth[0] = ASMTH * All.BoxSize / PMGRID;
    All.Rcut[0] = RCUT * All.Asmth[0];
    /* fac is - 4pi G     (L / 2pi) **2 / L ** 3 
     *        Gravity       k2            DFT  
     * */

    pot_factor = - All.G / (M_PI * All.BoxSize);	/* to get potential */


    pfft_init();

    ptrdiff_t n[3] = {PMGRID, PMGRID, PMGRID};
    ptrdiff_t np[2];

    /* The following memory will never be freed */
    Mesh2Task[0] = malloc(sizeof(int) * PMGRID);
    Mesh2Task[1] = malloc(sizeof(int) * PMGRID);
    Mesh2K = malloc(sizeof(int) * PMGRID);

    /* initialize the MPI Datatype of pencil */
    MPI_Type_contiguous(sizeof(struct Pencil), MPI_BYTE, &MPI_PENCIL);
    MPI_Type_commit(&MPI_PENCIL);

    /* try to find a square 2d decomposition */
    int i;
    int k;
    for(i = sqrt(NTask) + 1; i >= 0; i --) {
        if(NTask % i == 0) break;
    }
    np[0] = i;
    np[1] = NTask / i;

    if(ThisTask == 0) {
        printf("Using 2D Task mesh %td x %td \n", np[0], np[1]);
    }
    if( pfft_create_procmesh_2d(MPI_COMM_WORLD, np[0], np[1], &comm_cart_2d) ){
        fprintf(stderr, "Error: This test file only works with %td processes.\n", np[0]*np[1]);
        abort();
    }

    int periods_unused[2];
    MPI_Cart_get(comm_cart_2d, 2, NTask2d, periods_unused, ThisTask2d);

    if(NTask2d[0] != np[0]) abort();
    if(NTask2d[1] != np[1]) abort();

    fftsize = 2 * pfft_local_size_dft_r2c_3d(n, comm_cart_2d, 
           PFFT_TRANSPOSED_OUT, 
           real_space_region.size, real_space_region.offset, 
           fourier_space_region.size, fourier_space_region.offset);

    /*
     * In fourier space, the transposed array is ordered in
     * are in (y, z, x). The strides and sizes returned
     * from local size is in (Nx, Ny, Nz), hence we roll them once
     * so that the strides will give correct linear indexing for 
     * integer coordinates given in order of (x, y, z).
     * */

#define ROLL(a, N, j) { \
    typeof(a[0]) tmp[N]; \
    ptrdiff_t k; \
    for(k = 0; k < N; k ++) tmp[k] = a[k]; \
    for(k = 0; k < N; k ++) a[k] = tmp[(k + j)% N]; \
    }

    ROLL(fourier_space_region.offset, 3, 1);
    ROLL(fourier_space_region.size, 3, 1);

#undef ROLL

    /* calculate the strides */
    region_init_strides(&real_space_region);
    region_init_strides(&fourier_space_region); 

    /* planning the fft; need temporary arrays */

    pm_alloc();

    plan_forw = pfft_plan_dft_r2c_3d(
        n, real, pot_k, comm_cart_2d, PFFT_FORWARD, 
        PFFT_TRANSPOSED_OUT | PFFT_ESTIMATE | PFFT_DESTROY_INPUT);    
    plan_back = pfft_plan_dft_c2r_3d(
        n, complx, real, comm_cart_2d, PFFT_BACKWARD, 
        PFFT_TRANSPOSED_IN | PFFT_ESTIMATE | PFFT_DESTROY_INPUT);    

    pm_free();

    /* now lets fill up the mesh2task arrays */

    printf("ThisTask = %d (%td %td %td) - (%td %td %td)\n", ThisTask, 
            real_space_region.offset[0], 
            real_space_region.offset[1], 
            real_space_region.offset[2],
            real_space_region.size[0], 
            real_space_region.size[1], 
            real_space_region.size[2]);
    for(k = 0; k < 2; k ++) {
        int * tmp = (int*) alloca(sizeof(int) * PMGRID);
        for(i = 0; i < PMGRID; i ++) {
            tmp[i] = 0;
        }
        for(i = 0; i < real_space_region.size[k]; i ++) {
            tmp[i + real_space_region.offset[k]] = ThisTask2d[k];
        }
        /* which column / row hosts this tile? */
        /* FIXME: this is very inefficient */
        MPI_Allreduce(tmp, Mesh2Task[k], PMGRID, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        /*
        if(ThisTask == 0) {
            for(i = 0; i < PMGRID; i ++) {
                printf("Mesh2Task[%d][%d] == %d\n", k, i, Mesh2Task[k][i]);
            }
        }
        */
    }
    /* as well as the mesh2k array, we save the 2PI factor as done in
     * pm_periodic */
    for(i = 0; i < PMGRID; i ++) {
        if(i <= PMGRID / 2)
            Mesh2K[i] = i;
        else
            Mesh2K[i] = i - PMGRID;
    }

}

static int pm_mark_region_for_node(int startno, int rid);

static void convert_node_to_region(int no, int r);
void petapm_prepare() {
    /* 
     *
     * walks down the tree, identify nodes that contains local mass and
     * are sufficiently large in volume.
     *
     * for each nodes, a mesh region is created.
     * the particles in a node are linked to their hosting region 
     * (each particle belongs
     * to exactly one region even though it may be covered by two) 
     *
     * */
    int no;
    /* In worst case, each topleave becomes a region: thus
     * NTopleaves is sufficient */
    regions = malloc(sizeof(struct Region) * NTopleaves);

    int r = 0;

    no = All.MaxPart; /* start with the root */
    while(no >= 0) {
        if(!(Nodes[no].u.d.bitflags & (1 << BITFLAG_DEPENDS_ON_LOCAL_MASS))) {
            /* node doesn't contain particles on this process, do not open */
            no = Nodes[no].u.d.sibling;
            continue;
        }
        if(Nodes[no].len + 2 * Extnodes[no].hmax <= All.BoxSize / PMGRID * 24
            /* node is large */
       ||  (
            !(Nodes[no].u.d.bitflags & (1 << BITFLAG_INTERNAL_TOPLEVEL))
            && (Nodes[no].u.d.bitflags & (1 << BITFLAG_TOPLEVEL)))
            /* node is a top leaf */
                ) {
            convert_node_to_region(no, r);
            r ++;
            /* do not open */
            no = Nodes[no].u.d.sibling;
            continue;
        } 
        /* open */
        no = Nodes[no].u.d.nextnode;
    }

    Nregions = r;
    int maxNregions;
    MPI_Reduce(&Nregions, &maxNregions, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    if(ThisTask == 0) {
        printf("max number of regions is %d\n", maxNregions);
    }

    /* now lets mark particles to their hosting region */
    int numpart = 0;
#pragma omp parallel for reduction(+: numpart)
    for(r = 0; r < Nregions; r++) {
        regions[r].numpart = pm_mark_region_for_node(regions[r].no, r);
        numpart += regions[r].numpart;
    }
    /* All particles shall have been processed just once. Otherwise we die */
    if(numpart != NumPart) {
        abort();
    }
    walltime_measure("/PMgrav/Regions");
}

static void convert_node_to_region(int no, int r) {
    int k;
    double cellsize = All.BoxSize / PMGRID;
#if 0
    printf("task = %d no = %d len = %g hmax = %g center = %g %g %g\n",
            ThisTask, no, Nodes[no].len, Extnodes[no].hmax, 
            Nodes[no].center[0],
            Nodes[no].center[1],
            Nodes[no].center[2]);
#endif
    for(k = 0; k < 3; k ++) {
        regions[r].offset[k] = (Nodes[no].center[k] - Nodes[no].len * 0.5  - Extnodes[no].hmax) / cellsize;
        int end = (int) ((Nodes[no].center[k] + Nodes[no].len * 0.5  + Extnodes[no].hmax) / cellsize) + 1;
        regions[r].size[k] = end - regions[r].offset[k] + 1;
        regions[r].center[k] = Nodes[no].center[k];
    }

    /* setup the internal data structure of the region */
    region_init_strides(&regions[r]);

    regions[r].len  = Nodes[no].len;
    regions[r].hmax = Extnodes[no].hmax;
    regions[r].no = no;
}

void petapm_finish() {
    free(regions);
    regions = NULL;
}
    
static void pm_move_to_pfft();
static void pm_move_to_local();

/* 
 * read out field to particle i, with value no need to be thread safe 
 * (particle i is never done by same thread)
 * */
typedef void (* pm_iterator)(int i, double * mesh, double weight);
static void pm_iterate(pm_iterator iterator);
/* apply transfer function to value, kpos array is in x, y, z order */
typedef void (*transfer_function) (int64_t k2, int kpos[3], pfft_complex * value);
static void pm_apply_transfer_function(struct Region * fourier_space_region, 
        pfft_complex * src, 
        pfft_complex * dst, transfer_function H);

static void potential_transfer(int64_t k2, int kpos[3], pfft_complex * value);
static void force_x_transfer(int64_t k2, int kpos[3], pfft_complex * value);
static void force_y_transfer(int64_t k2, int kpos[3], pfft_complex * value);
static void force_z_transfer(int64_t k2, int kpos[3], pfft_complex * value);
static void put_particle_to_mesh(int i, double * mesh, double weight);
static void readout_potential(int i, double * mesh, double weight);
static void readout_force_x(int i, double * mesh, double weight);
static void readout_force_y(int i, double * mesh, double weight);
static void readout_force_z(int i, double * mesh, double weight);

void petapm_force() {
    int current_region = 0;
    int ndone = 0;
    int iterations;
    int i;
    struct Layout layout;

    pm_alloc();

    /* this takes care of the padding */
    memset(real, 0, sizeof(double) * fftsize);
    memset(meshbuf, 0, meshbufsize * sizeof(double));

    walltime_measure("/PMgrav/Misc");
    pm_iterate(put_particle_to_mesh);
    walltime_measure("/PMgrav/cic");

    layout_prepare(&layout);

    layout_build_and_exchange_cells_to_pfft(&layout);
    walltime_measure("/PMgrav/comm");
#if 1
    verify_density_field();
#endif
    walltime_measure("/PMgrav/Misc");

    /* call pfft pot_k is CFT of rho */

    /* this is because 
     *
     * CFT = DFT * dx **3
     * CFT[rho] = DFT [rho * dx **3] = DFT[CIC]
     * */
    pfft_execute_dft_r2c(plan_forw, real, pot_k);
    walltime_measure("/PMgrav/r2c");

    /* potential */

    /* apply the greens functionb turn pot_k into potential in fourier space */
    pm_apply_transfer_function(&fourier_space_region, pot_k, complx, potential_transfer);
    //
//    memcpy(complx, pot_k, sizeof(double) * fftsize);
    /* backup k space potential to pot_k */
    memcpy(pot_k, complx, sizeof(double) * fftsize);
    walltime_measure("/PMgrav/calc");

    pfft_execute_dft_c2r(plan_back, complx, real);
    walltime_measure("/PMgrav/c2r");
    /* read out the potential */
    layout_build_and_exchange_cells_to_local(&layout);
    walltime_measure("/PMgrav/comm");
    
    pm_iterate(readout_potential);
    walltime_measure("/PMgrav/readout");

    /* forces */

    pm_apply_transfer_function(&fourier_space_region, pot_k, complx, force_x_transfer);
    walltime_measure("/PMgrav/calc");
    pfft_execute_dft_c2r(plan_back, complx, real);
    walltime_measure("/PMgrav/c2r");
    layout_build_and_exchange_cells_to_local(&layout);
    walltime_measure("/PMgrav/comm");
    pm_iterate(readout_force_x);
    walltime_measure("/PMgrav/readout");

    pm_apply_transfer_function(&fourier_space_region, pot_k, complx, force_y_transfer);
    walltime_measure("/PMgrav/calc");
    pfft_execute_dft_c2r(plan_back, complx, real);
    walltime_measure("/PMgrav/c2r");
    layout_build_and_exchange_cells_to_local(&layout);
    walltime_measure("/PMgrav/comm");
    pm_iterate(readout_force_y);
    walltime_measure("/PMgrav/readout");

    pm_apply_transfer_function(&fourier_space_region, pot_k, complx, force_z_transfer);
    walltime_measure("/PMgrav/calc");
    pfft_execute_dft_c2r(plan_back, complx, real);
    walltime_measure("/PMgrav/c2r");
    layout_build_and_exchange_cells_to_local(&layout);
    walltime_measure("/PMgrav/comm");
    pm_iterate(readout_force_z);
    walltime_measure("/PMgrav/readout");

    layout_finish(&layout);
    pm_free();
    walltime_measure("/PMgrav/Misc");
}

/* build a communication layout */

static void layout_build_pencils(struct Layout * L);
static void layout_exchange_pencils(struct Layout * L);
static void layout_prepare (struct Layout * L) {
    int r;
    int i;

    L->ibuffer = mymalloc("PMlayout", sizeof(int) * NTask * 8);

    memset(L->ibuffer, 0, sizeof(int) * NTask * 8);
    L->NpSend = &L->ibuffer[NTask * 0];
    L->NpRecv = &L->ibuffer[NTask * 1];
    L->NcSend = &L->ibuffer[NTask * 2];
    L->NcRecv = &L->ibuffer[NTask * 3];
    L->DcSend = &L->ibuffer[NTask * 4];
    L->DcRecv = &L->ibuffer[NTask * 5];
    L->DpSend = &L->ibuffer[NTask * 6];
    L->DpRecv = &L->ibuffer[NTask * 7];

    L->NpExport = 0;
    L->NcExport = 0;
    L->NpImport = 0;
    L->NcImport = 0;

    int NpAlloc = 0;
    /* count pencils until buffer would run out */
    for (r = 0; r < Nregions; r ++) {
        NpAlloc += regions[r].size[0] * regions[r].size[1];
    }

    L->PencilSend = mymalloc("PencilSend", NpAlloc * sizeof(struct Pencil));

    layout_build_pencils(L);

    /* sort the pencils by the target rank for ease of next step */
    qsort(L->PencilSend, NpAlloc, sizeof(struct Pencil), pencil_cmp_target);
    /* zero length pixels are moved to the tail */

    /* now shrink NpExport*/
    L->NpExport = NpAlloc;
    while(L->PencilSend[L->NpExport - 1].len == 0) {
        L->NpExport --;
    }

    /* count total number of cells to be exported */
    int NcExport = 0;
#pragma omp parallel for reduction(+: NcExport)
    for(i = 0; i < L->NpExport; i++) {
        NcExport += L->PencilSend[i].len;
    }
    L->NcExport = NcExport;

#pragma omp parallel for
    for(i = 0; i < L->NpExport; i ++) {
        int task = pos_get_target(L->PencilSend[i].offset);
#pragma omp atomic
        L->NpSend[task] ++;
#pragma omp atomic
        L->NcSend[task] += L->PencilSend[i].len;
    }

    MPI_Alltoall(L->NpSend, 1, MPI_INT, L->NpRecv, 1, MPI_INT, MPI_COMM_WORLD);
    MPI_Alltoall(L->NcSend, 1, MPI_INT, L->NcRecv, 1, MPI_INT, MPI_COMM_WORLD);

    /* build the displacement array; why doesn't MPI build these automatically? */
    L->DpSend[0] = 0; L->DpRecv[0] = 0;
    L->DcSend[0] = 0; L->DcRecv[0] = 0;
    for(i = 1; i < NTask; i ++) {
        L->DpSend[i] = L->NpSend[i - 1] + L->DpSend[i - 1];
        L->DpRecv[i] = L->NpRecv[i - 1] + L->DpRecv[i - 1];
        L->DcSend[i] = L->NcSend[i - 1] + L->DcSend[i - 1];
        L->DcRecv[i] = L->NcRecv[i - 1] + L->DcRecv[i - 1];
    }
    L->NpImport = L->DpRecv[NTask -1] + L->NpRecv[NTask -1];
    L->NcImport = L->DcRecv[NTask -1] + L->NcRecv[NTask -1];

    /* some checks */
    if(L->DpSend[NTask - 1] + L->NpSend[NTask -1] != L->NpExport) abort();
    if(L->DcSend[NTask - 1] + L->NcSend[NTask -1] != L->NcExport) abort();

    int64_t totNpAlloc = reduce_int64(NpAlloc);
    int64_t totNpExport = reduce_int64(L->NpExport);
    int64_t totNcExport = reduce_int64(L->NcExport);
    int64_t totNpImport = reduce_int64(L->NpImport);
    int64_t totNcImport = reduce_int64(L->NcImport);

    if(totNpExport != totNpImport) {
        abort();
    }
    if(totNcExport != totNcImport) {
        abort();
    }

    /* exchange the pencils */
    if(ThisTask == 0) {
        printf("PetaPM:  %010ld/%010ld Pencils and %010ld Cells\n", totNpExport, totNpAlloc, totNcExport);
    }
    L->PencilRecv = mymalloc("PencilRecv", L->NpImport * sizeof(struct Pencil));
    layout_exchange_pencils(L);
}

static void layout_build_pencils(struct Layout * L) {
    /* now build pencils to be exported */
    int p0 = 0;
    int r;
    for (r = 0; r < Nregions; r++) {
        int ix;
#pragma omp parallel for private(ix)
        for(ix = 0; ix < regions[r].size[0]; ix++) {
            int iy;
            for(iy = 0; iy < regions[r].size[1]; iy++) {
                int poffset = ix * regions[r].size[1] + iy;
                struct Pencil * p = &L->PencilSend[p0 + poffset];

                p->offset[0] = ix + regions[r].offset[0];
                p->offset[1] = iy + regions[r].offset[1];
                p->offset[2] = regions[r].offset[2];
                p->len = regions[r].size[2];
                p->meshbuf_first = (regions[r].buffer - meshbuf) +
                    regions[r].strides[0] * ix +
                    regions[r].strides[1] * iy;
                /* now lets compress the pencil */
                while((p->len > 0) && (meshbuf[p->meshbuf_first + p->len - 1] == 0.0)) {
                    p->len --;
                }
                while((p->len > 0) && (meshbuf[p->meshbuf_first] == 0.0)) {
                    p->len --;
                    p->meshbuf_first++;
                    p->offset[2] ++;
                }

                p->task = pos_get_target(p->offset);
            }
        }
        p0 += regions[r].size[0] * regions[r].size[1];
    }

}

static void layout_exchange_pencils(struct Layout * L) {
    int r;
    int i;
    int offset;

    /* build the first pointers to refer to the correct relative buffer locations */
    /* note that the buffer hasn't bee assembled yet */
    offset = 0;
    for(i = 0; i < NTask; i ++) {
        int j;
        struct Pencil * p = &L->PencilSend[offset];
        p->first = 0;
        for(j = 1; j < L->NpSend[i]; j++) {
            p[j].first = p[j - 1].first + p[j - 1].len;
        }
        offset += L->NpSend[i];
    }

    MPI_Alltoallv(
            L->PencilSend, L->NpSend, L->DpSend, MPI_PENCIL,
            L->PencilRecv, L->NpRecv, L->DpRecv, MPI_PENCIL, 
            MPI_COMM_WORLD);

    /* set first to point to absolute position in the full import cell buffer */
    offset = 0;
    for(i = 0; i < NTask; i ++) {
        struct Pencil * p = &L->PencilRecv[offset];
        int j;
        for(j = 0; j < L->NpRecv[i]; j++) {
            p[j].first += L->DcRecv[i];
        }
        offset += L->NpRecv[i];
    }
    
    /* set first to point to absolute position in the full export cell buffer */
    offset = 0;
    for(i = 0; i < NTask; i ++) {
        struct Pencil * p = &L->PencilSend[offset];
        int j;
        for(j = 0; j < L->NpSend[i]; j++) {
            p[j].first += L->DcSend[i];
        }
        offset += L->NpSend[i];
    }
}

static void layout_finish(struct Layout * L) {
    myfree(L->PencilRecv);
    myfree(L->PencilSend);
    myfree(L->ibuffer);
}

/* exchange cells to their pfft host, then reduce the cells to the pfft
 * array */
static void to_pfft(double * cell, double * buf) {
#pragma omp atomic
            cell[0] += buf[0];
}
static void layout_build_and_exchange_cells_to_pfft(struct Layout * L) {
    L->BufSend = mymalloc("PMBufSend", L->NcExport * sizeof(double));
    L->BufRecv = mymalloc("PMBufRecv", L->NcImport * sizeof(double));

    int i;
    int offset;

    /* collect all cells into the send buffer */
    offset = 0;
    for(i = 0; i < L->NpExport; i ++) {
        struct Pencil * p = &L->PencilSend[i];
        memcpy(L->BufSend + offset, &meshbuf[p->meshbuf_first], 
                sizeof(double) * p->len);
        offset += p->len;
    }

    /* receive cells */
    MPI_Alltoallv(
            L->BufSend, L->NcSend, L->DcSend, MPI_DOUBLE,
            L->BufRecv, L->NcRecv, L->DcRecv, MPI_DOUBLE, 
            MPI_COMM_WORLD);

#if 0
    double massExport = 0;
    for(i = 0; i < L->NcExport; i ++) {
        massExport += L->BufSend[i];
    }

    double massImport = 0;
    for(i = 0; i < L->NcImport; i ++) {
        massImport += L->BufRecv[i];
    }
    double totmassExport;
    double totmassImport;
    MPI_Allreduce(&massExport, &totmassExport, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&massImport, &totmassImport, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    if(ThisTask == 0) {
        printf("totmassExport = %g totmassImport = %g\n", totmassExport, totmassImport);
    }
#endif

    layout_iterate_cells(L, to_pfft);
    myfree(L->BufRecv);
    myfree(L->BufSend);
}

/* readout cells on their pfft host, then exchange the cells to the domain 
 * host */
static void to_region(double * cell, double * region) {
    *region = *cell;
}

static void layout_build_and_exchange_cells_to_local(struct Layout * L) {
    L->BufSend = mymalloc("PMBufSend", L->NcExport * sizeof(double));
    L->BufRecv = mymalloc("PMBufRecv", L->NcImport * sizeof(double));
    int i;
    int offset;

    layout_iterate_cells(L, to_region);

    /* exchange cells */
    /* notice the order is reversed from to_pfft */
    MPI_Alltoallv(
            L->BufRecv, L->NcRecv, L->DcRecv, MPI_DOUBLE, 
            L->BufSend, L->NcSend, L->DcSend, MPI_DOUBLE,
            MPI_COMM_WORLD);

    /* distribute BufSend to meshbuf */
    offset = 0;
    for(i = 0; i < L->NpExport; i ++) {
        struct Pencil * p = &L->PencilSend[i];
        memcpy(&meshbuf[p->meshbuf_first], 
                L->BufSend + offset, 
                sizeof(double) * p->len);
        offset += p->len;
    }
    myfree(L->BufRecv);
    myfree(L->BufSend);
}

/* iterate over the pairs of real field cells and RecvBuf cells 
 *
 * !!! iter has to be thread safe. !!!
 * */
static void layout_iterate_cells(struct Layout * L, cell_iterator iter) {
    int i;
#pragma omp parallel for
    for(i = 0; i < L->NpImport; i ++) {
        struct Pencil * p = &L->PencilRecv[i];
        int k;
        ptrdiff_t linear0 = 0;
        for(k = 0; k < 2; k ++) {
            int ix = p->offset[k];
            while(ix < 0) ix += PMGRID;
            while(ix >= PMGRID) ix -= PMGRID;
            ix -= real_space_region.offset[k];
            if(ix >= real_space_region.size[k]) {
                /* seroius problem assmpution about pfft layout was wrong*/
                abort();
            }
            linear0 += ix * real_space_region.strides[k];
        }
        int j;
        for(j = 0; j < p->len; j ++) {
            int iz = p->offset[2] + j;
            while(iz < 0) iz += PMGRID;
            while(iz >= PMGRID) iz -= PMGRID;
            if(iz >= real_space_region.size[2]) {
                /* seroius problem assmpution about pfft layout was wrong*/
                abort();
            }
            ptrdiff_t linear = iz * real_space_region.strides[2] + linear0;
            /* 
             * operate on the pencil, either modifying real or BufRecv 
             * */
            iter(&real[linear], &L->BufRecv[p->first + j]);
        }
    }
}
static void pm_alloc() {
    real = (double * ) mymalloc("PMreal", fftsize * sizeof(double));
    complx = (pfft_complex *) mymalloc("PMcomplex", fftsize * sizeof(double));
    pot_k = (pfft_complex * ) mymalloc("PMpot_k", fftsize * sizeof(double));
    if(regions) {
        int i;
        size_t size = 0;
        for(i = 0 ; i < Nregions; i ++) {
            size += regions[i].totalsize;
        }
        meshbufsize = size;
        meshbuf = (double *) mymalloc("PMmesh", size * sizeof(double));
        report_memory_usage(&HighMark_petapm, "PetaPM");
        size = 0;
        for(i = 0 ; i < Nregions; i ++) {
            regions[i].buffer = meshbuf + size;
            size += regions[i].totalsize;
        }
    }
}


static void pm_iterate_one(int i, pm_iterator iterator) {
    struct Region * region = &regions[P[i].RegionInd];
    int k;
    double cellsize = All.BoxSize / PMGRID;
    int iCell[3];  /* integer coordinate on the regional mesh */
    double Res[3]; /* residual*/
    for(k = 0; k < 3; k++) {
        double tmp = P[i].Pos[k] / cellsize;
        iCell[k] = tmp;
        Res[k] = tmp - iCell[k];
        iCell[k] -= region->offset[k];
        if(iCell[k] >= region->size[k] - 1) {
            /* seriously?! particles are supposed to be contained in cells */
            abort(); 
        }
        if(iCell[k] < 0) {
            abort();
        }
    }

    int connection = 0;
    double mass = P[i].Mass;
    for(connection = 0; connection < 8; connection++) {
        double weight = 1.0;
        ptrdiff_t linear = 0;
        for(k = 0; k < 3; k++) {
            int offset = (connection >> k) & 1;
            int tmp = iCell[k] + offset;
            linear += tmp * region->strides[k];
            weight *= offset?
                /* offset == 1*/ (Res[k])    :
                /* offset == 0*/ (1 - Res[k]);
        }
        if(linear >= region->totalsize) abort();
        iterator(i, &region->buffer[linear], weight);
    }
}

/* 
 * iterate over all particle / mesh pairs, call iterator 
 * function . iterator function shall be aware of thread safety.
 * no threads run on same particle same time but may 
 * access one mesh points same time.
 * */
static void pm_iterate(pm_iterator iterator) {
    int i;
#pragma omp parallel for 
    for(i = 0; i < NumPart; i ++) {
        pm_iterate_one(i, iterator); 
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

static int pm_mark_region_for_node(int startno, int rid) {
    int numpart = 0;
    int p;
    int endno = Nodes[startno].u.d.sibling;
    int no = Nodes[startno].u.d.nextnode;
    while(no >= 0)
    {
        if(no < All.MaxPart)	/* single particle */
        {
            p = no;
            no = Nextnode[no];
            drift_particle(p, All.Ti_Current);
            P[p].RegionInd = rid;
            numpart ++;
        }
        else
        {
            if(no >= All.MaxPart + MaxNodes)	/* pseudo particle */
            {
                /* skip pseudo particles */
                no = Nextnode[no - MaxNodes];
                continue;
            }

            if(no == endno)
                /* we arrived to the sibling which means that we are done with the node */
            {
                break;
            }
            force_drift_node(no, All.Ti_Current);

            no = Nodes[no].u.d.nextnode;	/* ok, we need to open the node */
        }
    }
    return numpart;
}

static void region_init_strides(struct Region * region) {
    int k;
    size_t rt = 1;
    for(k = 2; k >= 0; k --) {
        region->strides[k] = rt;
        rt = region->size[k] * rt;
    }
    region->totalsize = rt;
    region->buffer = NULL;
}

static int pos_get_target(const int pos[2]) {
    int k;
    int task2d[2];
    int rank;
    for(k = 0; k < 2; k ++) {
        int ix = pos[k];
        while(ix < 0) ix += PMGRID;
        while(ix >= PMGRID) ix -= PMGRID;
        task2d[k] = Mesh2Task[k][ix];
    }
    MPI_Cart_rank(comm_cart_2d, task2d, &rank);
    return rank;
}
static int pencil_cmp_target(const struct Pencil * p1, const struct Pencil * p2) {
    /* move zero length pixels to the end */
    if(p2->len == 0) return -1;
    if(p1->len == 0) return 1;
    int t1 = pos_get_target(p1->offset); 
    int t2 = pos_get_target(p2->offset); 
    return ((t2 < t1) - (t1 < t2)) * 2 +
        ((p2->first < p1->first) - (p1->first < p2->first));
}

static void pm_free() {
    if(regions) {
        myfree(meshbuf);
    }
    myfree(pot_k);
    myfree(complx);
    myfree(real);
}
static void verify_density_field() {
    int i;
    /* verify the density field */
    double mass_Part = 0;
#pragma omp parallel for reduction(+: mass_Part)
    for(i = 0; i < NumPart; i ++) {
        mass_Part += P[i].Mass;
    }
    double totmass_Part = 0;
    MPI_Allreduce(&mass_Part, &totmass_Part, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    double mass_Region = 0;
#pragma omp parallel for reduction(+: mass_Region)
    for(i = 0; i < meshbufsize; i ++) {
        mass_Region += meshbuf[i];    
    }
    double totmass_Region = 0;
    MPI_Allreduce(&mass_Region, &totmass_Region, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    double mass_CIC = 0;
#pragma omp parallel for reduction(+: mass_CIC)
    for(i = 0; i < real_space_region.totalsize; i ++) {
        mass_CIC += real[i];
    }
    double totmass_CIC = 0;
    MPI_Allreduce(&mass_CIC, &totmass_CIC, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    if(ThisTask == 0) {
        printf("total Region mass = %g CIC mass = %g Particle mass = %g\n", totmass_Region, totmass_CIC, totmass_Part);
    }
}

static void pm_apply_transfer_function(struct Region * region, 
        pfft_complex * src, 
        pfft_complex * dst, transfer_function H
        ){
    ptrdiff_t ip = 0;

#pragma omp parallel for
    for(ip = 0; ip < region->totalsize; ip ++) {
        ptrdiff_t tmp = ip;
        int pos[3];
        int kpos[3];
        int64_t k2 = 0.0;
        int k;
        for(k = 0; k < 3; k ++) {
            pos[k] = tmp / region->strides[k];
            tmp -= pos[k] * region->strides[k];
            /* lets get the abs pos on the grid*/
            pos[k] += region->offset[k];
            /* check */
            if(pos[k] >= PMGRID) abort();
            kpos[k] = Mesh2K[pos[k]];
            /* Watch out the cast */
            k2 += ((int64_t)kpos[k]) * kpos[k];
        }
        /* swap 0 and 1 because fourier space was transposed */
        /* kpos is y, z, x */
        pos[0] = kpos[2];
        pos[1] = kpos[0];
        pos[2] = kpos[1];
        dst[ip][0] = src[ip][0];
        dst[ip][1] = src[ip][1];
        H(k2, pos, &dst[ip]);
    }

}

/********************
 * transfer functions for 
 *
 * potential from mass in cell
 *
 * and 
 *
 * force from potential
 *
 *********************/

/* unnormalized sinc function sin(x) / x */
static double sinc_unnormed(double x) {
    if(x < 1e-5 && x > -1e-5) {
        double x2 = x * x;
        return 1.0 - x2 / 6. + x2  * x2 / 120.;
    } else {
        return sin(x) / x;
    }
}

static void potential_transfer(int64_t k2, int kpos[3], pfft_complex *value) {
    if(k2 == 0) {
        /* remote zero mode corresponding to the mean */
        value[0][0] = 0.0;
        value[0][1] = 0.0;
        return;
    } 
    double asmth2 = (2 * M_PI) * All.Asmth[0] / All.BoxSize;
    asmth2 *= asmth2;
    double f = 1.0;
    double smth = exp(-k2 * asmth2) / k2;
    /* the CIC deconvolution kernel is
     *
     * sinc_unnormed(k_x L / 2 PMGRID) ** 2
     *
     * k_x = kpos * 2pi / L
     *
     * */
    int k;
    for(k = 0; k < 3; k ++) {
        double tmp = (kpos[k] * M_PI) / PMGRID;
        tmp = sinc_unnormed(tmp);
        f *= 1. / (tmp * tmp);
    }
    /* 
     * first decovolution is CIC in par->mesh
     * second decovolution is correcting readout 
     * I don't understand the second yet!
     * */
    double fac = pot_factor * smth * f * f;
    value[0][0] *= fac;
    value[0][1] *= fac;
}

/* the transfer functions for force in fourier space applied to potential */
/* super lanzcos in CH6 P 122 Digital Filters by Richard W. Hamming */
static double super_lanzcos_diff_kernel_3(double w) {
/* order N = 3*/
    return 1. / 594 * 
       (126 * sin(w) + 193 * sin(2 * w) + 142 * sin (3 * w) - 86 * sin(4 * w));
}
static double super_lanzcos_diff_kernel_2(double w) {
/* order N = 2*/
    return 1 / 126. * (58 * sin(w) + 67 * sin (2 * w) - 22 * sin(3 * w));
}
static double super_lanzcos_diff_kernel_1(double w) {
/* order N = 1 */
/* 
 * This is the same as GADGET-2 but in fourier space: 
 * see gadget-2 paper and Hamming's book.
 * c1 = 2 / 3, c2 = 1 / 12
 * */
    return 1 / 6.0 * (8 * sin (w) - sin (2 * w));
}
static double diff_kernel(double w) {
#if PETA_PM_ORDER == 1
        return super_lanzcos_diff_kernel_1(w);
#endif
#if PETA_PM_ORDER == 2
        return super_lanzcos_diff_kernel_2(w);
#endif
#if PETA_PM_ORDER == 3
        return super_lanzcos_diff_kernel_3(w);
#endif
#if PETA_PM_ORDER > 3 
#error PETA_PM_ORDER too high.
#endif
}
static void force_transfer(int k, pfft_complex * value) {
    double tmp0;
    double tmp1;
    /* 
     * negative sign is from force_x = - Del_x pot 
     *
     * filter is   i K(w)
     * */
    double fac = -1 * diff_kernel (k * (2 * M_PI / PMGRID)) * (PMGRID / All.BoxSize);
    tmp0 = - value[0][1] * fac;
    tmp1 = value[0][0] * fac;
    value[0][0] = tmp0;
    value[0][1] = tmp1;
}
static void force_x_transfer(int64_t k2, int kpos[3], pfft_complex * value) {
    return force_transfer(kpos[0], value);
}
static void force_y_transfer(int64_t k2, int kpos[3], pfft_complex * value) {
    return force_transfer(kpos[1], value);
}
static void force_z_transfer(int64_t k2, int kpos[3], pfft_complex * value) {
    return force_transfer(kpos[2], value);
}

/**************
 * functions iterating over particle / mesh pairs
 ***************/
static void put_particle_to_mesh(int i, double * mesh, double weight) {
#pragma omp atomic
    mesh[0] += weight * P[i].Mass;
}
static void readout_potential(int i, double * mesh, double weight) {
    P[i].PM_Potential += weight * mesh[0];
}
static void readout_force_x(int i, double * mesh, double weight) {
    P[i].GravPM[0] += weight * mesh[0];
}
static void readout_force_y(int i, double * mesh, double weight) {
    P[i].GravPM[1] += weight * mesh[0];
}
static void readout_force_z(int i, double * mesh, double weight) {
    P[i].GravPM[2] += weight * mesh[0];
}

static int64_t reduce_int64(int64_t input) {
    int64_t result = 0;
    MPI_Allreduce(&input, &result, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
    return result;
}
#endif

/** Some FFT notes
 *
 *
 * CFT = dx * iDFT (thus CFT has no 2pi factors and iCFT has, 
 *           same as wikipedia.)
 *
 * iCFT(CFT) = (2pi) ** 3, thus iCFT also has no 2pi factors.
 * **************************8*/
