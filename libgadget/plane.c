#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <limits.h>
#include <math.h>
#include <omp.h>
#include <gsl/gsl_interp.h>
#include "lenstools.h"
#include "utils/string.h"
#include "utils/mymalloc.h"
#include "cosmology.h"
#include "plane.h"
#include "physconst.h"
#include "partmanager.h"
#include "petapm.h"
#include "gravity.h"
#include "neutrinos_lra.h"
#include "timefac.h"
#include "utils/endrun.h"
#include "utils/system.h"
#include "timebinmgr.h"

static struct plane_params
{
    int64_t NormalsLength;
    int Normals[3];

    int64_t CutPointsLength;
    double CutPoints[1024];

    int Resolution;
    double Thickness; // in kpc/h
    int MassiveNuCorrection;
    int DoubleOut;
} PlaneParams;

typedef struct {
    int pos[3];
    double mass;
} PlaneCellContribution;

typedef struct {
    PetaPM pm;
    double *real;
    double mean_mass_cell;
    double inv_fft_norm;
    int nmesh;
    int owns_pm;
} PlanePMGrid;

static Cosmology * PlanePMCP = NULL;
static double PlanePMTime = 0;
static double PlanePMTimeIC = 0;

static double
plane_wrap_position(double x, const double L)
{
    while(x < 0) x += L;
    while(x >= L) x -= L;
    return x;
}

static double
plane_particle_omega_source(const Cosmology * CP, const double atime)
{
    double omega_source = CP->Omega0;
    if(CP->MassiveNuLinRespOn)
        omega_source -= pow(atime, 3) * get_omega_nu_nopart(&CP->ONu, atime);
    if(omega_source <= 0)
        endrun(1, "Non-positive particle matter density for potential plane: OmegaSource = %g\n", omega_source);
    return omega_source;
}

static int64_t
plane_count_active_particles(const Cosmology * CP, const double atime)
{
    int64_t local_count = 0;
    int64_t total_count = 0;
#pragma omp parallel for reduction(+:local_count)
    for(int64_t p = 0; p < PartManager->NumPart; p++)
        if(lenstools_particle_is_active(CP, atime, p))
            local_count++;
    MPI_Allreduce(&local_count, &total_count, 1, MPI_INT64, MPI_SUM, MPI_COMM_WORLD);
    return total_count;
}

static int
plane_pm_cell_owner(PetaPM * pm, const int ix, const int iy)
{
    int task2d[2] = {pm->Mesh2Task[0][ix], pm->Mesh2Task[1][iy]};
    int rank;
    MPI_Cart_rank(pm->priv->comm_cart_2d, task2d, &rank);
    return rank;
}

static void
plane_pm_particle_cic(const int64_t p, const int nmesh, const double cellsize,
        int base[3], double res[3])
{
    for(int d = 0; d < 3; d++) {
        const double pos = plane_wrap_position(P[p].Pos[d] - PartManager->CurrentParticleOffset[d], PartManager->BoxSize);
        const double meshpos = pos / cellsize;
        base[d] = (int) floor(meshpos);
        if(base[d] >= nmesh)
            base[d] = 0;
        res[d] = meshpos - floor(meshpos);
    }
}

static double *
plane_pm_deposit_particles(PetaPM * pm, const Cosmology * CP, const double atime, double * total_mass_out)
{
    int NTask;
    MPI_Comm_size(pm->comm, &NTask);

    double *real = (double *) mymalloc("PlanePMRealIn", pm->priv->fftsize * sizeof(double));
    memset(real, 0, pm->priv->fftsize * sizeof(double));

    int *send_count = (int *) mymalloc("PlaneSendCount", NTask * sizeof(int));
    int *recv_count = (int *) mymalloc("PlaneRecvCount", NTask * sizeof(int));
    int *send_disp = (int *) mymalloc("PlaneSendDisp", NTask * sizeof(int));
    int *recv_disp = (int *) mymalloc("PlaneRecvDisp", NTask * sizeof(int));
    int *cursor = (int *) mymalloc("PlaneCursor", NTask * sizeof(int));
    memset(send_count, 0, NTask * sizeof(int));

    const int nmesh = pm->Nmesh;
    const double cellsize = pm->BoxSize / nmesh;
    double local_mass = 0;

    for(int64_t p = 0; p < PartManager->NumPart; p++) {
        if(!lenstools_particle_is_active(CP, atime, p))
            continue;
        local_mass += P[p].Mass;

        int base[3];
        double res[3];
        plane_pm_particle_cic(p, nmesh, cellsize, base, res);

        for(int connection = 0; connection < 8; connection++) {
            double weight = 1.0;
            int cell[3];
            for(int d = 0; d < 3; d++) {
                const int offset = (connection >> d) & 1;
                cell[d] = base[d] + offset;
                if(cell[d] >= nmesh)
                    cell[d] -= nmesh;
                weight *= offset ? res[d] : (1 - res[d]);
            }
            if(weight == 0)
                continue;
            const int target = plane_pm_cell_owner(pm, cell[0], cell[1]);
            if(send_count[target] == INT_MAX)
                endrun(1, "Too many plane PM cell contributions for one MPI task.\n");
            send_count[target]++;
        }
    }

    MPI_Allreduce(&local_mass, total_mass_out, 1, MPI_DOUBLE, MPI_SUM, pm->comm);
    if(*total_mass_out <= 0)
        endrun(1, "Cannot build a potential plane from zero active particle mass.\n");

    MPI_Alltoall(send_count, 1, MPI_INT, recv_count, 1, MPI_INT, pm->comm);

    int total_send = 0;
    int total_recv = 0;
    for(int i = 0; i < NTask; i++) {
        send_disp[i] = total_send;
        recv_disp[i] = total_recv;
        total_send += send_count[i];
        total_recv += recv_count[i];
        cursor[i] = send_disp[i];
    }

    PlaneCellContribution *sendbuf = (PlaneCellContribution *) mymalloc("PlaneSendCells", total_send * sizeof(PlaneCellContribution));
    PlaneCellContribution *recvbuf = (PlaneCellContribution *) mymalloc("PlaneRecvCells", total_recv * sizeof(PlaneCellContribution));

    for(int64_t p = 0; p < PartManager->NumPart; p++) {
        if(!lenstools_particle_is_active(CP, atime, p))
            continue;

        int base[3];
        double res[3];
        plane_pm_particle_cic(p, nmesh, cellsize, base, res);

        for(int connection = 0; connection < 8; connection++) {
            double weight = 1.0;
            int cell[3];
            for(int d = 0; d < 3; d++) {
                const int offset = (connection >> d) & 1;
                cell[d] = base[d] + offset;
                if(cell[d] >= nmesh)
                    cell[d] -= nmesh;
                weight *= offset ? res[d] : (1 - res[d]);
            }
            if(weight == 0)
                continue;

            const int target = plane_pm_cell_owner(pm, cell[0], cell[1]);
            PlaneCellContribution * contrib = &sendbuf[cursor[target]++];
            contrib->pos[0] = cell[0];
            contrib->pos[1] = cell[1];
            contrib->pos[2] = cell[2];
            contrib->mass = P[p].Mass * weight;
        }
    }

    MPI_Datatype MPI_PLANE_CELL;
    MPI_Type_contiguous(sizeof(PlaneCellContribution), MPI_BYTE, &MPI_PLANE_CELL);
    MPI_Type_commit(&MPI_PLANE_CELL);
    MPI_Alltoallv(sendbuf, send_count, send_disp, MPI_PLANE_CELL,
                  recvbuf, recv_count, recv_disp, MPI_PLANE_CELL, pm->comm);
    MPI_Type_free(&MPI_PLANE_CELL);

    PetaPMRegion * region = &pm->real_space_region;
    for(int i = 0; i < total_recv; i++) {
        const int lx = recvbuf[i].pos[0] - region->offset[0];
        const int ly = recvbuf[i].pos[1] - region->offset[1];
        const int lz = recvbuf[i].pos[2] - region->offset[2];
        if(lx < 0 || lx >= region->size[0] ||
           ly < 0 || ly >= region->size[1] ||
           lz < 0 || lz >= region->size[2]) {
            endrun(1, "Received plane PM cell outside local PFFT region: %d %d %d; region off %td %td %td size %td %td %td\n",
                recvbuf[i].pos[0], recvbuf[i].pos[1], recvbuf[i].pos[2],
                region->offset[0], region->offset[1], region->offset[2],
                region->size[0], region->size[1], region->size[2]);
        }
        const ptrdiff_t linear = lx * region->strides[0] + ly * region->strides[1] + lz * region->strides[2];
        real[linear] += recvbuf[i].mass;
    }

    myfree(recvbuf);
    myfree(sendbuf);
    myfree(cursor);
    myfree(recv_disp);
    myfree(send_disp);
    myfree(recv_count);
    myfree(send_count);

    return real;
}

static void
plane_pm_apply_transfer(PetaPM * pm, pfft_complex * src, pfft_complex * dst, petapm_transfer_func H)
{
    PetaPMRegion * region = &pm->fourier_space_region;

#pragma omp parallel for
    for(size_t ip = 0; ip < region->totalsize; ip++) {
        ptrdiff_t tmp = ip;
        int pos[3];
        int kpos[3];
        int64_t k2 = 0;
        for(int k = 0; k < 3; k++) {
            pos[k] = tmp / region->strides[k];
            tmp -= pos[k] * region->strides[k];
            pos[k] += region->offset[k];
            if(pos[k] >= pm->Nmesh)
                endrun(1, "Plane PM Fourier position did not make sense.\n");
            kpos[k] = petapm_mesh_to_k(pm, pos[k]);
            k2 += ((int64_t) kpos[k]) * kpos[k];
        }

        pos[0] = kpos[2];
        pos[1] = kpos[0];
        pos[2] = kpos[1];
        dst[ip][0] = src[ip][0];
        dst[ip][1] = src[ip][1];
        if(H)
            H(pm, k2, pos, &dst[ip]);
    }
}

static void
plane_compute_neutrino_power(PetaPM * pm)
{
    Power * ps = pm->ps;
    powerspectrum_sum(ps);
    for(int i = 0; i < ps->nonzero; i++)
        ps->Power[i] = sqrt(ps->Power[i]);

    delta_nu_from_power(ps, PlanePMCP, PlanePMTime, PlanePMTimeIC);

    ps->nu_spline = gsl_interp_alloc(gsl_interp_linear, ps->nonzero);
    ps->nu_acc = gsl_interp_accel_alloc();
    gsl_interp_init(ps->nu_spline, ps->logknu, ps->delta_nu_ratio, ps->nonzero);
    powerspectrum_zero(ps);
}

static void
plane_neutrino_correction_transfer(PetaPM * pm, int64_t k2, int kpos[3], pfft_complex * value)
{
    if(k2 == 0 || !PlanePMCP || !PlanePMCP->MassiveNuLinRespOn) {
        value[0][0] = 0.0;
        value[0][1] = 0.0;
        return;
    }

    Power * ps = pm->ps;
    double logk = log(sqrt(k2) * 2 * M_PI / ps->BoxSize_in_MPC);
    if(logk < ps->logknu[0] && logk > ps->logknu[0] - log(2))
        logk = ps->logknu[0];
    else if(logk > ps->logknu[ps->nonzero - 1])
        logk = ps->logknu[ps->nonzero - 1];

    const double nufac_minus_one = ps->nu_prefac * gsl_interp_eval(ps->nu_spline, ps->logknu,
            ps->delta_nu_ratio, logk, ps->nu_acc);
    value[0][0] *= nufac_minus_one;
    value[0][1] *= nufac_minus_one;
}

static void
plane_pm_grid_init_neutrino_correction(PlanePMGrid * grid, PetaPM * pm,
        Cosmology * CP, const double atime, const double TimeIC, const double UnitLength_in_cm)
{
    memset(grid, 0, sizeof(*grid));
    grid->pm = *pm;
    grid->nmesh = pm->Nmesh;
    grid->owns_pm = 0;
    grid->inv_fft_norm = 1.0 / ((double)grid->nmesh * grid->nmesh * grid->nmesh);

    double total_mass = 0;
    double * real_mass = plane_pm_deposit_particles(&grid->pm, CP, atime, &total_mass);
    grid->mean_mass_cell = total_mass / ((double)grid->nmesh * grid->nmesh * grid->nmesh);

    pfft_complex * density_k = (pfft_complex *) mymalloc2("PlaneDensityK", grid->pm.priv->fftsize * sizeof(double));
    pfft_complex * corrected_k = (pfft_complex *) mymalloc2("PlaneNuCorrectionK", grid->pm.priv->fftsize * sizeof(double));

    pfft_execute_dft_r2c(grid->pm.priv->plan_forw, real_mass, density_k);
    myfree(real_mass);

    PlanePMCP = CP;
    PlanePMTime = atime;
    PlanePMTimeIC = TimeIC;

    powerspectrum_alloc(grid->pm.ps, grid->pm.Nmesh, omp_get_max_threads(), 1, grid->pm.BoxSize * UnitLength_in_cm);
    plane_pm_apply_transfer(&grid->pm, density_k, corrected_k, measure_power_spectrum);
    plane_compute_neutrino_power(&grid->pm);
    plane_pm_apply_transfer(&grid->pm, density_k, corrected_k, plane_neutrino_correction_transfer);

    gsl_interp_free(grid->pm.ps->nu_spline);
    gsl_interp_accel_free(grid->pm.ps->nu_acc);
    powerspectrum_free(grid->pm.ps);

    grid->real = (double *) mymalloc("PlaneNuCorrectionReal", grid->pm.priv->fftsize * sizeof(double));
    pfft_execute_dft_c2r(grid->pm.priv->plan_back, corrected_k, grid->real);

    myfree(corrected_k);
    myfree(density_k);
}

static void
plane_pm_grid_free(PlanePMGrid * grid)
{
    myfree(grid->real);
    if(grid->owns_pm)
        petapm_destroy(&grid->pm);
}

static double
plane_interval_overlap(const double a0, const double a1, const double b0, const double b1)
{
    const double lo = a0 > b0 ? a0 : b0;
    const double hi = a1 < b1 ? a1 : b1;
    return hi > lo ? hi - lo : 0.0;
}

static double
plane_periodic_slab_overlap(const double cell_start, const double cellsize,
        const double center, const double thickness, const double L)
{
    if(thickness >= L)
        return cellsize;

    const double c = plane_wrap_position(center, L);
    const double slab_start = c - 0.5 * thickness;
    const double slab_end = slab_start + thickness;
    const double cell_end = cell_start + cellsize;
    double overlap = 0.0;

    for(int shift = -1; shift <= 1; shift++) {
        const double offset = shift * L;
        overlap += plane_interval_overlap(cell_start, cell_end, slab_start + offset, slab_end + offset);
    }
    return overlap;
}

static void
cutPlanePMNeutrinoCorrection(const PlanePMGrid * grid, double comoving_distance, double Lbox,
        const Cosmology * CP, const double atime, const int normal, const double center,
        const double thickness, double *lensing_potential)
{
    const int nmesh = grid->nmesh;
    const double cellsize = Lbox / nmesh;
    const double smooth = 1.0;
    double *density = allocate_2d_array_as_1d(nmesh, nmesh);

    const int plane_directions[2] = { (normal + 1) % 3, (normal + 2) % 3 };
    const PetaPMRegion * region = &grid->pm.real_space_region;

#pragma omp parallel for
    for(ptrdiff_t ix = 0; ix < region->size[0]; ix++) {
        for(ptrdiff_t iy = 0; iy < region->size[1]; iy++) {
            for(ptrdiff_t iz = 0; iz < region->size[2]; iz++) {
                int global[3] = {
                    (int)(region->offset[0] + ix),
                    (int)(region->offset[1] + iy),
                    (int)(region->offset[2] + iz)
                };
                if(global[0] >= nmesh || global[1] >= nmesh || global[2] >= nmesh)
                    continue;

                const double cell_start = global[normal] * cellsize;
                const double overlap = plane_periodic_slab_overlap(cell_start, cellsize, center, thickness, Lbox);
                if(overlap <= 0)
                    continue;

                const ptrdiff_t linear = ix * region->strides[0] + iy * region->strides[1] + iz * region->strides[2];
                const double delta_nu_scaled = grid->real[linear] * grid->inv_fft_norm / grid->mean_mass_cell;
                const int pix0 = global[plane_directions[0]];
                const int pix1 = global[plane_directions[1]];
                const double contribution = delta_nu_scaled * overlap / thickness;
#pragma omp atomic update
                ACCESS_2D(density, pix0, pix1, nmesh) += contribution;
            }
        }
    }

    calculate_lensing_potential(density, nmesh, cellsize, cellsize,
            comoving_distance, smooth, lensing_potential);

    const double omega_source = plane_particle_omega_source(CP, atime);
    const double H0 = 100 * CP->HubbleParam * 3.2407793e-20;
    const double cosmo_normalization = 1.5 * H0 * H0 * omega_source / (LIGHTCGS * LIGHTCGS);
    const double density_normalization = thickness * comoving_distance * pow(CM_PER_KPC / CP->HubbleParam, 2) / atime;

    for(int i = 0; i < nmesh; i++) {
        for(int j = 0; j < nmesh; j++) {
            ACCESS_2D(lensing_potential, i, j, nmesh) *= cosmo_normalization * density_normalization;
        }
    }

    myfree(density);
}

static void
plane_add_periodic_bilinear(double * dst, const int dst_n, const double * src, const int src_n)
{
#pragma omp parallel for
    for(int i = 0; i < dst_n; i++) {
        const double x = ((i + 0.5) * src_n / dst_n) - 0.5;
        int i0 = (int) floor(x);
        const double tx = x - i0;
        while(i0 < 0) i0 += src_n;
        while(i0 >= src_n) i0 -= src_n;
        const int i1 = (i0 + 1) % src_n;

        for(int j = 0; j < dst_n; j++) {
            const double y = ((j + 0.5) * src_n / dst_n) - 0.5;
            int j0 = (int) floor(y);
            const double ty = y - j0;
            while(j0 < 0) j0 += src_n;
            while(j0 >= src_n) j0 -= src_n;
            const int j1 = (j0 + 1) % src_n;

            const double v00 = ACCESS_2D(src, i0, j0, src_n);
            const double v10 = ACCESS_2D(src, i1, j0, src_n);
            const double v01 = ACCESS_2D(src, i0, j1, src_n);
            const double v11 = ACCESS_2D(src, i1, j1, src_n);
            ACCESS_2D(dst, i, j, dst_n) +=
                (1 - tx) * (1 - ty) * v00 +
                tx * (1 - ty) * v10 +
                (1 - tx) * ty * v01 +
                tx * ty * v11;
        }
    }
}

char *
plane_get_output_fname(const int snapnum, const char * OutputDir, const int cut, const int normal)
{
    // Format the filename to include '!' to overwrite existing files
    char * fname = fastpm_strdup_printf("!%s/snap%d_potentialPlane%d_normal%d.fits", OutputDir, snapnum, cut, normal);
    return fname;
}

/* This is basically BuildOutputList but with an integer*/
int set_plane_normals(ParameterSet* ps)
{
    char * Normals = param_get_string(ps, "PlaneNormals");
    if(!Normals){
        PlaneParams.NormalsLength = 0;
        return 0;
    }
    char * strtmp = fastpm_strdup(Normals);
    char * token;
    int count;
    // print string
    //message(0,"Normals = %s\n", Normals);

    /*First parse the string to get the number of outputs*/
    for(count=0, token=strtok(strtmp,","); token; count++, token=strtok(NULL, ","))
    {}

    myfree(strtmp);

    /*Allocate enough memory*/
    PlaneParams.NormalsLength = count;
    size_t maxcount = sizeof(PlaneParams.Normals) / sizeof(PlaneParams.Normals[0]);

    if((size_t) PlaneParams.NormalsLength > maxcount) {
        message(1, "Too many entries (%ld) in the Normals, can take no more than %lu.\n", PlaneParams.NormalsLength, maxcount);
        return 1;
    }
    /*Now read in the values*/
    for(count=0,token=strtok(Normals,","); count < PlaneParams.NormalsLength && token; count++, token=strtok(NULL,","))
    {
        /* Skip a leading quote if one exists.
         * Extra characters are ignored by atof, so
         * no need to skip matching char.*/
        if(token[0] == '"')
            token+=1;

        int n = atoi(token);

        if(n != 0 && n != 1 && n != 2) {
            endrun(1, "Requesting a normal direction beyond 0, 1 and 2: %d\n", n);
        }
        PlaneParams.Normals[count] = n;
/*         message(1, "Output at: %g\n", Sync.OutputListTimes[count]); */
    }
    return 0;
}

/*Set the plane parameters*/
void
set_plane_params(ParameterSet * ps)
{
    int ThisTask;
    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
    if(ThisTask == 0) {
        // plane resolution
        PlaneParams.Resolution = param_get_int(ps, "PlaneResolution");

        // plane thickness
        PlaneParams.Thickness = param_get_double(ps, "PlaneThickness");

        PlaneParams.MassiveNuCorrection = param_get_int(ps, "PlaneMassiveNuCorrection");
        if(PlaneParams.MassiveNuCorrection != 0 && PlaneParams.MassiveNuCorrection != 1)
            endrun(1, "PlaneMassiveNuCorrection must be 0 or 1, got %d\n", PlaneParams.MassiveNuCorrection);

        PlaneParams.DoubleOut = param_get_int(ps, "PlaneDoubleOut");
        if(PlaneParams.DoubleOut != 0 && PlaneParams.DoubleOut != 1)
            endrun(1, "PlaneDoubleOut must be 0 or 1, got %d\n", PlaneParams.DoubleOut);

        // Plane normals
        set_plane_normals(ps);

        // Plane cut points
        if (!param_get_string(ps, "PlaneCutPoints")) {
            message(0, "No cut points provided, a set of default values will be set: (1/2 + i) * plane thickness (< box size, i = 0, 1, 2...)\n");
            // set the length to 0
            PlaneParams.CutPointsLength = 0;
        }
        else
            BuildOutputList(ps, "PlaneCutPoints", PlaneParams.CutPoints, &PlaneParams.CutPointsLength, 1024);
    }
    MPI_Bcast(&PlaneParams, sizeof(struct plane_params), MPI_BYTE, 0, MPI_COMM_WORLD);
}

void write_plane(int snapnum, const double atime, const double TimeIC, Cosmology * CP, PetaPM * pm, const char * OutputDir, const double UnitVelocity_in_cm_per_s, const double UnitLength_in_cm) {

    double BoxSize = PartManager->BoxSize;

    int64_t num_particles_tot = plane_count_active_particles(CP, atime);

    // plane parameters
    int plane_resolution = PlaneParams.Resolution;
    double thickness = PlaneParams.Thickness; // in kpc/h
    if (thickness <= 0.0) {
        message(0, "No positive thickness provided, the side length of the box, %g, will be used.\n", BoxSize);
        thickness = BoxSize;
    }

    // set a set of cut points if NULL
    if (PlaneParams.CutPointsLength == 0) {
        PlaneParams.CutPointsLength = (int64_t) (BoxSize / thickness);
        for (int i = 0; i < PlaneParams.CutPointsLength; i++) {
            PlaneParams.CutPoints[i] = (.5 + i) * thickness;
        }
        // print cut points
        message(0, "Cut points set automatically:\n");
        for (int i = 0; i < PlaneParams.CutPointsLength; i++) {
            message(0,"CutPoints[%d] = %g\n", i, PlaneParams.CutPoints[i]);
        }
    }


    int ThisTask;
    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);

    double redshift = 1./atime - 1.;
    message(0, "Computing and writing potential planes.\n");

    double *plane_result = allocate_2d_array_as_1d(plane_resolution, plane_resolution);
    double *summed_plane_result = allocate_2d_array_as_1d(plane_resolution, plane_resolution);

    double comoving_distance = compute_comoving_distance(CP, atime, 1., UnitVelocity_in_cm_per_s);

    // print comoving distance
    message(0, "Comoving distance: %g\n", comoving_distance);

    PlanePMGrid plane_pm_grid;
    memset(&plane_pm_grid, 0, sizeof(plane_pm_grid));
    const int use_pm_neutrino_correction = PlaneParams.MassiveNuCorrection && CP->MassiveNuLinRespOn;
    if(use_pm_neutrino_correction) {
        if(plane_resolution < 2)
            endrun(1, "PlaneResolution must be at least 2 for PlaneMassiveNuCorrection.\n");
        if(!pm)
            endrun(1, "PlaneMassiveNuCorrection requires the existing PM mesh for massive-neutrino corrections.\n");
        message(0, "Using existing PM mesh massive-neutrino correction with Nmesh = %d and PlaneResolution = %d.\n", pm->Nmesh, plane_resolution);
        plane_pm_grid_init_neutrino_correction(&plane_pm_grid, pm, CP, atime, TimeIC, UnitLength_in_cm);
    }
    else if(PlaneParams.MassiveNuCorrection) {
        message(0, "PlaneMassiveNuCorrection requested, but massive-neutrino linear response is disabled; no PM correction will be added.\n");
    }
    else if(CP->MassiveNuLinRespOn) {
        message(0, "WARNING: PlaneMassiveNuCorrection is disabled, so potential planes omit linear-response massive neutrino perturbations.\n");
    }

    /* loop over cut points and normal directions to generate lensing potential planes */
    for (int i = 0; i < PlaneParams.CutPointsLength; i++) {
        for (int j = 0; j < PlaneParams.NormalsLength; j++) {
            message(0, "Computing for cut point %g and normal %d\n", PlaneParams.CutPoints[i], PlaneParams.Normals[j]);

            double left_corner[3] = {0, 0, 0};
            int64_t num_particles_plane = 0, num_particles_plane_tot = 0;

            memset(plane_result, 0, plane_resolution * plane_resolution * sizeof(double));

            /* The particle plane always uses the 2D high-resolution path.
             * PlaneMassiveNuCorrection only controls the optional PM neutrino correction below. */
            num_particles_plane = cutPlaneGaussianGrid(num_particles_tot,  comoving_distance, BoxSize, CP, atime, PlaneParams.Normals[j], PlaneParams.CutPoints[i], thickness, left_corner, plane_resolution, plane_result);

            if(use_pm_neutrino_correction) {
                double * pm_correction = allocate_2d_array_as_1d(plane_pm_grid.nmesh, plane_pm_grid.nmesh);
                cutPlanePMNeutrinoCorrection(&plane_pm_grid, comoving_distance, BoxSize, CP, atime, PlaneParams.Normals[j], PlaneParams.CutPoints[i], thickness, pm_correction);
                plane_add_periodic_bilinear(plane_result, plane_resolution, pm_correction, plane_pm_grid.nmesh);
                myfree(pm_correction);
            }

            /*sum up planes from all tasks*/
            MPI_Reduce(plane_result, summed_plane_result, plane_resolution * plane_resolution, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            MPI_Reduce(&num_particles_plane, &num_particles_plane_tot, 1, MPI_INT64, MPI_SUM, 0, MPI_COMM_WORLD);

            /*saving planes*/
            if (ThisTask == 0) {
#ifdef USE_CFITSIO
                char * file_path = plane_get_output_fname(snapnum, OutputDir, i, PlaneParams.Normals[j]);
                savePotentialPlane(summed_plane_result, plane_resolution, plane_resolution, file_path, BoxSize, CP, redshift, comoving_distance, num_particles_plane_tot, UnitLength_in_cm, PlaneParams.DoubleOut);
                message(0, "Plane saved for cut %d and normal %d to %s\n", i, PlaneParams.Normals[j], file_path + 1); // skip the '!' in the filename
                myfree(file_path);
#endif
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }
    }
    if(use_pm_neutrino_correction)
        plane_pm_grid_free(&plane_pm_grid);
    myfree(summed_plane_result);
    myfree(plane_result);


    if (ThisTask == 0) {
        double comoving_distance_Mpc  = comoving_distance * UnitLength_in_cm / CM_PER_MPC;
        char * buf = fastpm_strdup_printf("%s/info.txt", OutputDir);
        FILE * fd = fopen(buf, "a");
        fprintf(fd, "s=%d,d=%lf Mpc/h,z=%lf\n", snapnum, comoving_distance_Mpc, redshift);
        fclose(fd);
        myfree(buf);
    }
}
