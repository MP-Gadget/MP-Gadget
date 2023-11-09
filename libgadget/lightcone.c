#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_integration.h>
/*For mkdir*/
#include <sys/stat.h>
#include <sys/types.h>

#include "utils.h"

#include "timefac.h"
#include "partmanager.h"
#include "cosmology.h"
#include "physconst.h"

#define NENTRY 4096
static double tab_loga[NENTRY];
static double dloga;
static double tab_Dc[NENTRY];
/*
 * light cone on the fly:
 *
 * assuming the origin is at (0, 0, 0)
 *
 * */

/*
 * replicas to consider, function of redshift;
 *
 * */
static int Nreplica;
static int BoxBoost = 20;
static double Reps[8192][3];
static double HorizonDistance2;
static double HorizonDistance;
static double HorizonDistancePrev;
static double HorizonDistance2Prev;
static double HorizonDistanceRef;
static double zmin = 0.1;
static double zmax = 80.0;
static double ReferenceRedshift = 2.0; /* write all particles below this redshift; write a fraction above this. */
static double SampleFraction; /* current fraction of particle gets written */
static FILE * fd_lightcone;

static double lightcone_get_horizon(double a);
static void lightcone_cross(int p, double ddrift, const RandTable * const rnd);
static void lightcone_set_time(double a, const double BoxSize);
/*
M, L = self.M, self.L
  logx = numpy.linspace(log10amin, 0, Np)
  def kernel(log10a):
    a = numpy.exp(log10a)
    return 1 / self.Ea(a) * a ** -1 # dz = - 1 / a dlog10a
  y = numpy.array( [romberg(kernel, log10a, 0, vec_func=True, divmax=10) for log10a in logx])
*/
static double kernel(double loga, void * params) {
    double a = exp(loga);
      Cosmology * CP = (Cosmology *) params;
    return 1 / hubble_function(CP, a) * CP->Hubble / a;
}

static void lightcone_init_entry(Cosmology * CP, int i, const double UnitLength_in_cm) {
    tab_loga[i] = - dloga * (NENTRY - i - 1);

    gsl_integration_workspace * w = gsl_integration_workspace_alloc (1000);

    double result, error;

    gsl_function F;
    F.function = &kernel;
    F.params = CP;
    gsl_integration_qags (&F, tab_loga[i], 0, 0, 1e-7, 1000,
            w, &result, &error);

    /* result is in DH, hubble distance */
    /* convert to cm / h */
    result *= LIGHTCGS / HUBBLE;
    /* convert to Kpc/h or internal units */
    result /= UnitLength_in_cm;

    gsl_integration_workspace_free (w);
    tab_Dc[i] = result;
//    double a = exp(tab_loga[i]);
//    double z = 1 / a - 1;
//    printf("a = %g z = %g Dc = %g\n", a, z, result);
}

void lightcone_init(Cosmology * CP, double timeBegin, const double UnitLength_in_cm, const char * OutputDir)
{
    int i;
    dloga = (0.0 - log(timeBegin)) / (NENTRY - 1);
    for(i = 0; i < NENTRY; i ++) {
        lightcone_init_entry(CP, i, UnitLength_in_cm);
    };
    char buf[1024];
    int chunk = 100;
    int ThisTask;
    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);

    sprintf(buf, "%s/lightcone/", OutputDir);
    mkdir(buf, 02755);
    sprintf(buf, "%s/lightcone/%03d/", OutputDir, (int)(ThisTask / chunk));
    mkdir(buf, 02755);
    sprintf(buf, "%s/lightcone/%03d/lightcone-%05d.raw", OutputDir, (int)(ThisTask / chunk), ThisTask);

    fd_lightcone = fopen(buf, "a+");
    if(fd_lightcone == NULL) {
        endrun(1, "failed to open %s\n", buf);
    }
    HorizonDistanceRef = lightcone_get_horizon(1 / (1 + ReferenceRedshift));
    printf("lightcone reference redshift = %g distance = %g\n",
            ReferenceRedshift, HorizonDistanceRef);
}

/* returns the horizon distance */
static double lightcone_get_horizon(double a) {
    double loga = log(a);
    int bin = (log(a) -tab_loga[0]) / dloga;
    if (bin < 0) {
        return tab_Dc[0];
    }
    if (bin >= NENTRY - 1) {
        return tab_Dc[NENTRY - 1];
    }
    double u1 = loga - tab_loga[bin];
    double u2 = tab_loga[bin + 1] - loga;
    u1 /= (tab_loga[bin + 1] - tab_loga[bin]);
    u2 /= (tab_loga[bin + 1] - tab_loga[bin]);
    return tab_Dc[bin] * u2 + tab_Dc[bin + 1] * u1;
}

/* fill in the table of box offsets for current time */
static void update_replicas(double a, double BoxSize) {
    int Nmax = BoxBoost * BoxBoost * BoxBoost;
    int i;
    int rx, ry, rz;
    rx = ry = rz = 0;
    Nreplica = 0;

    for(i = 0; i < Nmax; i ++) {
        double dx = BoxSize * rx;
        double dy = BoxSize * ry;
        double dz = BoxSize * rz;
        double d1, d2;
        d1 = dx * dx + dy * dy + dz * dz;
        dx += BoxSize;
        dy += BoxSize;
        dz += BoxSize;
        d2 = dx * dx + dy * dy + dz * dz;
        if(d1 <= HorizonDistance2 && d2 >= HorizonDistance2) {
            Reps[Nreplica][0] = rx * BoxSize;
            Reps[Nreplica][1] = ry * BoxSize;
            Reps[Nreplica][2] = rz * BoxSize;
            Nreplica ++;
            if(Nreplica > 1000) {
                endrun(951234, "too many replica");
            }
        }
        rz ++;
        if(rz == BoxBoost) {
            rz = 0;
            ry ++;
        }
        if(ry == BoxBoost) {
            ry = 0;
            rx ++;
        }
    }
}

/* Compute a list of particles which crossed
 * the lightcone boundaries on this timestep and
 * write them to the lightcone file*/
void lightcone_compute(double a, double BoxSize, Cosmology * CP, inttime_t ti_curr, inttime_t ti_next, const RandTable * const rnd)
{
    int i;
    lightcone_set_time(a, BoxSize);
    const double ddrift = get_exact_drift_factor(CP, ti_curr, ti_next);
    #pragma omp parallel for
    for(i = 0; i < PartManager->NumPart; i++)
    {
        lightcone_cross(i, ddrift, rnd);
    }
}

void lightcone_set_time(double a, const double BoxSize) {
    double z = 1 / a - 1;
    if(z > zmin && z < zmax) {
        HorizonDistancePrev = HorizonDistance;
        HorizonDistance2Prev = HorizonDistance2;
        HorizonDistance = lightcone_get_horizon(a);
        HorizonDistance2 = HorizonDistance * HorizonDistance;
        update_replicas(a, BoxSize);
        fflush(fd_lightcone);
        if (z < ReferenceRedshift) {
            SampleFraction = 1.0;
        } else {
            /* write a smaller fraction of the points at high redshift
             */
            /* This is the angular resolution rule */
            SampleFraction = HorizonDistanceRef / HorizonDistance;
            SampleFraction *= SampleFraction;
            SampleFraction *= SampleFraction;
            /* This is the luminosity resolution rule */
#if 0
            SampleFraction = HorizonDistanceRef / HorizonDistance;
            SampleFraction *= (1 + ReferenceRedshift) / (1 + z);
            SampleFraction *= SampleFraction;

#endif
        }
        message(0,"RefRedeshit=%g, SampleFraction=%g HorizonDistance=%g\n", ReferenceRedshift, SampleFraction, HorizonDistance);
    } else {
        SampleFraction = 0;
    }
}

/* check crossing of the horizon, write the particle */
static void lightcone_cross(int p, double ddrift, const RandTable * const rnd) {
    if(SampleFraction <= 0.0) return;
    int i;
    int k;
    /* DM only */
    if(P[p].Type != 1) return;

    for(i = 0; i < Nreplica; i++) {
        double r = get_random_number(P[p].ID + i, rnd);
        if(r > SampleFraction) continue;

        double pnew[3];
        double pold[3];
        double p3[4];
        double dnew = 0, dold = 0;
        for(k = 0; k < 3; k ++) {
            pold[k] = P[p].Pos[k] + Reps[i][k] - PartManager->CurrentParticleOffset[k];
            pnew[k] = P[p].Pos[k] + P[i].Vel[k] * ddrift - PartManager->CurrentParticleOffset[k];
            dnew += pnew[k] * pnew[k];
            dold += pold[k] * pold[k];
        }
        if(
            (dold <= HorizonDistance2Prev && dnew >= HorizonDistance2)
         ) {
            double u1, u2;
            if(dold != dnew) {
                double cnew, cold;
                dnew = sqrt(dnew);
                dold = sqrt(dold);
                cnew = dnew - HorizonDistance;
                cold = dold - HorizonDistancePrev;
                u1 = -cold / (cnew - cold);
                u2 = cnew / (cnew - cold);
            } else {
                /* really should write all particles along the line:
                 * this partilce is moving along the horizon! */
                u1 = u2 = 0.5;
            }

            /* write particle position */
            for(k = 0; k < 3; k ++) {
                p3[k] = pold[k] * u2 + pnew[k] * u1;
            }
            p3[3] = SampleFraction;
            fwrite(p3, sizeof(double), 4, fd_lightcone);
        }
    }
}
