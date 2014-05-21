#ifdef PETA_PM
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pfft.h>

#include "allvars.h"
#include "proto.h"
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
} real_space_region, fourier_space_region;

static struct Region * regions = NULL;
static int Nregions = 0;
static void region_init(struct Region * region);


struct Pencil { /* a pencil starting at offset, with lenght len */
    int offset[3];
    int len;
    ptrdiff_t first; /* first pixel in meshbuf */
    int task;
};
static int pencil_cmp_target(const struct Pencil * p1, const struct Pencil * p2);
static int pencil_get_target(const struct Pencil * pencil);

/* for debuggin */
static void verify_density_field();

static double * real;
static double * meshbuf;
static size_t meshbufsize;
static pfft_complex * complx;
static pfft_complex * rho_k;
static int fftsize;
static int maxfftsize;
static void pm_alloc();
static void pm_free();
static pfft_plan plan_forw, plan_back;
MPI_Comm comm_cart_2d;
static int * ParticleRegion;
static int ThisTask2d[2];
static int NTask2d[2];
static int * (Mesh2Task[2]); /* convertion mesh to task2d */
static MPI_Datatype MPI_PENCIL;

void petapm_init_periodic(void) {
    All.Asmth[0] = ASMTH * All.BoxSize / PMGRID;
    All.Rcut[0] = RCUT * All.Asmth[0];
    pfft_init();
    ptrdiff_t n[3] = {PMGRID, PMGRID, PMGRID};
    ptrdiff_t np[2];
    Mesh2Task[0] = malloc(sizeof(int) * PMGRID);
    Mesh2Task[1] = malloc(sizeof(int) * PMGRID);

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

    region_init(&real_space_region);
    region_init(&fourier_space_region); 

    MPI_Allreduce(&fftsize, &maxfftsize, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

    /* do the planning */
    pm_alloc();
    plan_forw = pfft_plan_dft_r2c_3d(
        n, real, rho_k, comm_cart_2d, PFFT_FORWARD, 
        PFFT_TRANSPOSED_OUT | PFFT_MEASURE | PFFT_DESTROY_INPUT);    
    plan_back = pfft_plan_dft_c2r_3d(
        n, complx, real, comm_cart_2d, PFFT_FORWARD, 
        PFFT_TRANSPOSED_OUT | PFFT_MEASURE | PFFT_DESTROY_INPUT);    

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
}

static int pm_mark_region_for_node(int startno, int rid);

void petapm_prepare() {
    /* build a list of regions and record which region a particle belongs to */
    int i;
    int m;
    int no;
    double cellsize = All.BoxSize / PMGRID;
    regions = malloc(sizeof(struct Region) * NTopleaves);

    int r = 0;
    int k;

    for(m = 0; m < MULTIPLEDOMAINS; m++) {
        for(i = DomainStartList[ThisTask * MULTIPLEDOMAINS + m]; i <= DomainEndList[ThisTask * MULTIPLEDOMAINS + m]; i++) {
            no = DomainNodeIndex[i];
#if 0
            printf("task = %d no = %d len = %g hmax = %g center = %g %g %g\n",
                    ThisTask, no, Nodes[no].len, Extnodes[no].hmax, 
                    Nodes[no].center[0],
                    Nodes[no].center[1],
                    Nodes[no].center[2]);
#endif
            for(k = 0; k < 3; k ++) {
                regions[r].offset[k] = (Nodes[no].center[k] - Nodes[no].len * 0.5  - Extnodes[no].hmax) / cellsize;
                int end = ceil(Nodes[no].center[k] + Nodes[no].len * 0.5  + Extnodes[no].hmax) / cellsize;
                regions[r].size[k] = end - regions[r].offset[k] + 1;
                regions[r].center[k] = Nodes[no].center[k];
            }

            /* setup the internal data structure of the region */
            region_init(&regions[r]);

            regions[r].len  = Nodes[no].len;
            regions[r].hmax = Extnodes[no].hmax;
            /* now lets mark particles to their hosting region */
            regions[r].numpart = pm_mark_region_for_node(no, r);
            r++;
        }
    }
    Nregions = r;
    int numpart = 0;
    for(r = 0; r < Nregions; r++) {
        numpart += regions[r].numpart;
    }
    /* all particles shall have been processed just once. */
    if(numpart != NumPart) {
        abort();
    }
}

void petapm_finish() {
    free(regions);
    regions = NULL;
}
    
static void pm_put_particle_to_mesh(int i);

void petapm_force() {

    pm_alloc();
    memset(real, 0, sizeof(double) * fftsize);
    int iterations;
    int i;

    /* CIC */
    if(ThisTask == 0) 
        printf("starting CIC\n");

#pragma omp parallel for 
    for(i = 0; i < NumPart; i ++) {
        pm_put_particle_to_mesh(i); 
    }

    int ndone = 0;
    int current_region = 0;
    iterations = 0;
    while(ndone < NTask) {
        int done = pm_regions_to_pfft(&current_region);
        MPI_Allreduce(&done, &ndone, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        if(ThisTask == 0)
            printf("regions to pfft iteration = %d\n", iterations);
        iterations++;
    }
    verify_density_field();

    /* potential */
    /* call pfft */

    /* read out the potential */
    ndone = 0;
    while(ndone < NTask) {
//        int done = pm_pfft_to_regions();
        int done = 1;
        MPI_Allreduce(&done, &ndone, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    }

    /* forces */
    /* call pfft */
    /* read out the forces */
    ndone = 0;
    while(ndone < NTask) {
 //       int done = pm_pfft_to_regions();
        int done = 1;
        MPI_Allreduce(&done, &ndone, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    }

    pm_free();
}
static int pm_regions_to_pfft(int * current_region) {

    ptrdiff_t offset; /* used for various purposes */
    int i;
    if(*current_region == Nregions)
        return 1;

    int * Np_send = alloca(sizeof(int) * NTask);
    int * Np_recv = alloca(sizeof(int) * NTask);
    int * Nc_send = alloca(sizeof(int) * NTask);
    int * Nc_recv = alloca(sizeof(int) * NTask);
    int * sdisp = alloca(sizeof(int) * NTask);
    int * rdisp = alloca(sizeof(int) * NTask);

    memset(Np_send, 0, sizeof(int) * NTask);
    memset(Nc_send, 0, sizeof(int) * NTask);

    int NpExport = 0;
    int NcExport = 0;
    int NcImport = 0;
    int NpImport = 0;

    All.BunchSize = (int)(All.BufferSize * 1024 * 1024) / sizeof(double);

    /* count pencils until buffer would run out */
    int r;
    int end_region; /* the last region in this bunch*/

    for (r = *current_region; r < Nregions; r ++) {
        if(NcExport + regions[r].totalsize > All.BunchSize 
            /* at least send one region */
                && r > *current_region) {
            break;
        }
        NcExport += regions[r].totalsize;
        NpExport += regions[r].size[0] * regions[r].size[1];
    }
    end_region = r;

    struct Pencil * PencilSend = (struct Pencil * ) 
        mymalloc("PencilSend", NpExport * sizeof(struct Pencil));

    /* now build pencils */
    int p0 = 0;
    for (r = *current_region; r < end_region; r++) {
        int ix;
#pragma omp parallel for private(ix)
        for(ix = 0; ix < regions[r].size[0]; ix++) {
            int iy;
            for(iy = 0; iy < regions[r].size[1]; iy++) {
                int poffset = ix * regions[r].size[1] + iy;
                int p = p0 + poffset;
                PencilSend[p].offset[0] = ix + regions[r].offset[0];
                PencilSend[p].offset[1] = iy + regions[r].offset[1];
                PencilSend[p].offset[2] = regions[r].offset[2];
                PencilSend[p].len = regions[r].size[2];
                PencilSend[p].first = (regions[r].buffer - meshbuf) +
                    regions[r].strides[0] * ix +
                    regions[r].strides[1] * iy;
                PencilSend[p].task = pencil_get_target(&PencilSend[p]);
            }
        }
        p0 += regions[r].size[0] * regions[r].size[1];
    }

    /* sort the pencils by the target rank for ease of next step */
    qsort(PencilSend, NpExport, sizeof(struct Pencil), pencil_cmp_target);

    /* collect the cells to be sent to the commbuffer */ 
    double * BufSend = mymalloc("PMBufSend", NcExport * sizeof(double));
    offset = 0;
    for(i = 0; i < NpExport; i ++) {
        memcpy(BufSend + offset, &meshbuf[PencilSend[i].first], sizeof(double) * PencilSend[i].len);
        offset += PencilSend[i].len;
    }

    /* count Np_send and Nc_send, detach the `first' element to relative to the
     * send-recv chunk */
    int oldtarget = -1;
    for(i = 0; i < NpExport; i ++) {
        int target = pencil_get_target(&PencilSend[i]);
        if (target != oldtarget) {
            /* be aware PencilSend is sorted by target! */
            offset = 0;
            oldtarget= target;
        }
        PencilSend[i].first = offset;
        offset += PencilSend[i].len;
        Np_send[target] ++;
        Nc_send[target] += PencilSend[i].len;
    }

    /* Alltoall to get Np_recv, Nc_recv */
    MPI_Alltoall(Np_send, 1, MPI_INT, Np_recv, 1, MPI_INT, MPI_COMM_WORLD);
    MPI_Alltoall(Nc_send, 1, MPI_INT, Nc_recv, 1, MPI_INT, MPI_COMM_WORLD);

    /* Now we are ready to exchange the pencils */
    /* first count total recv */

    for(i = 0; i < NTask; i ++) {
        NpImport += Np_recv[i];
        NcImport += Nc_recv[i];
    }
    struct Pencil * PencilRecv = (struct Pencil * ) 
        mymalloc("PencilRecv", NpImport * sizeof(struct Pencil));
    double * BufRecv = mymalloc("PMBufRecv", NcImport * sizeof(double));

    /* receive pencils */
    sdisp[0] = 0; rdisp[0] = 0;
    for(i = 1; i < NTask; i ++) {
        sdisp[i] = Np_send[i - 1] + sdisp[i - 1];
        rdisp[i] = Np_recv[i - 1] + rdisp[i - 1];
    }
    MPI_Alltoallv(
            PencilSend, Np_send, sdisp, MPI_PENCIL,
            PencilRecv, Np_recv, rdisp, MPI_PENCIL, 
            MPI_COMM_WORLD);

    /* receive cells */
    sdisp[0] = 0; rdisp[0] = 0;
    for(i = 1; i < NTask; i ++) {
        sdisp[i] = Nc_send[i - 1] + sdisp[i - 1];
        rdisp[i] = Nc_recv[i - 1] + rdisp[i - 1];
    }
    MPI_Alltoallv(
            BufSend, Nc_send, sdisp, MPI_DOUBLE,
            BufRecv, Nc_recv, rdisp, MPI_DOUBLE, 
            MPI_COMM_WORLD);


    /* fix the first pointers */
    offset = 0;
    for(i = 0; i < NTask; i ++) {
        int p;
        for(p = offset; p < offset + Np_recv[i]; p++) {
            PencilRecv[p].first += rdisp[i];
        }
        offset += Np_recv[i];
    }

#if 0 
    double massExport = 0;
    for(i = 0; i < NcExport; i ++) {
        massExport += BufSend[i];
    }

    double massImport = 0;
    for(i = 0; i < NcImport; i ++) {
        massImport += BufRecv[i];
    }
    double totmassExport;
    double totmassImport;
    MPI_Allreduce(&massExport, &totmassExport, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&massImport, &totmassImport, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    if(ThisTask == 0) {
        printf("totmassExport = %g totmassImport = %g\n", totmassExport, totmassImport);
    }
#endif
    /* unpack penciles */
#pragma omp parallel for
    for(i = 0; i < NpImport; i ++) {
        int k;
        ptrdiff_t linear0 = 0;
        for(k = 0; k < 2; k ++) {
            int ix = PencilRecv[i].offset[k];
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
        for(j = 0; j < PencilRecv[i].len; j ++) {
            int iz = PencilRecv[i].offset[2] + j;
            while(iz < 0) iz += PMGRID;
            while(iz >= PMGRID) iz -= PMGRID;
            if(iz >= real_space_region.size[2]) {
                /* seroius problem assmpution about pfft layout was wrong*/
                abort();
            }
            ptrdiff_t linear = iz * real_space_region.strides[2] + linear0;
            /* most import line here, add the pencil to the real space field */
#pragma omp atomic
            real[linear] += BufRecv[PencilRecv[i].first + j];
        }
    }
    myfree(BufRecv);
    myfree(PencilRecv);
    myfree(BufSend);
    myfree(PencilSend);

    /* next call will start processing from the end of this call */
    *current_region = end_region;
    return 0;
}
static void pm_alloc() {
    real = (double * ) mymalloc("PMreal", fftsize * sizeof(double));
    complx = (pfft_complex *) mymalloc("PMcomplex", fftsize * sizeof(double));
    rho_k = (pfft_complex * ) mymalloc("PMrho_k", fftsize * sizeof(double));
    if(regions) {
        int i;
        size_t size = 0;
        for(i = 0 ; i < Nregions; i ++) {
            size += regions[i].totalsize;
        }
        meshbufsize = size;
        meshbuf = (double *) mymalloc("PMmesh", size * sizeof(double));
        memset(meshbuf, 0, size * sizeof(double));
        report_memory_usage(&HighMark_petapm, "PetaPM");
        size = 0;
        for(i = 0 ; i < Nregions; i ++) {
            regions[i].buffer = meshbuf + size;
            size += regions[i].totalsize;
        }
    }
}

static void pm_put_particle_to_mesh(int i) {
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
        if(iCell[k] > region->size[k] - 1) {
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
#pragma omp atomic
        region->buffer[linear] += weight * mass;
    }
}
static int pm_mark_region_for_node(int startno, int rid) {
    int numpart = 0;
    int p;
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

            if(Nodes[no].u.d.bitflags & (1 << BITFLAG_TOPLEVEL))	
                /* we reached a top-level node again, which means that we are done with the branch */
            {
                break;
            }
            force_drift_node(no, All.Ti_Current);

            no = Nodes[no].u.d.nextnode;	/* ok, we need to open the node */
        }
    }
    return numpart;
}

static void region_init(struct Region * region) {
    int k;
    size_t rt = 1;
    for(k = 2; k >= 0; k --) {
        region->strides[k] = rt;
        rt = region->size[k] * rt;
    }
    region->totalsize = rt;
    region->buffer = NULL;
}

static int pencil_get_target(const struct Pencil * pencil) {
    int k;
    int task2d[2];
    int rank;
    for(k = 0; k < 2; k ++) {
        int ix = pencil->offset[k];
        while(ix < 0) ix += PMGRID;
        while(ix >= PMGRID) ix -= PMGRID;
        task2d[k] = Mesh2Task[k][ix];
    }
    MPI_Cart_rank(comm_cart_2d, task2d, &rank);
    return rank;
}
static int pencil_cmp_target(const struct Pencil * p1, const struct Pencil * p2) {
    int t1 = pencil_get_target(p1); 
    int t2 = pencil_get_target(p2); 
    return ((t2 < t1) - (t1 < t2)) * 2 +
        ((p2->first < p1->first) - (p1->first < p2->first));
}

static void pm_free() {
    if(regions) {
        myfree(meshbuf);
    }
    myfree(rho_k);
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

    printf("on me Region mass = %g CIC mass = %g Particle mass = %g\n", mass_Region, mass_CIC, mass_Part);
    if(ThisTask == 0) {
        printf("total Region mass = %g CIC mass = %g Particle mass = %g\n", totmass_Region, totmass_CIC, totmass_Part);
    }
}
#endif
