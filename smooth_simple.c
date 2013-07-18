#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <gsl/gsl_math.h>

#include "allvars.h"
#include "proto.h"

#if (defined(DIVBCLEANING_DEDNER) || defined(SMOOTH_ROTB) || defined(BSMOOTH) || defined(SCAL_PRO_CLEAN) || defined(VECT_POTENTIAL))

/*! Structure for communication during the density computation. Holds data that is sent to other processors.
*/
static struct smoothdata_in
{
    MyDouble Pos[3];
    MyFloat Hsml;
    int NodeList[NODELISTLENGTH];
}
*SmoothDataIn, *SmoothDataGet;


static struct smoothdata_out
{
#ifdef SMOOTH_PHI
    MyFloat SmoothPhi;
#endif
#ifdef VECT_POTENTIAL
    MyFloat SmoothA[3];
#endif
#if defined(DIVBCLEANING_DEDNER) || defined(SCAL_PRO_CLEAN)
    MyFloat SmoothDivB;
#endif
#ifdef SMOOTH_ROTB
    MyFloat SmoothRotB[3];
#endif
#if defined(BSMOOTH)
    MyFloat BSmooth[3];
#endif
#if defined(BLACK_HOLES)
    MyLongDouble SmoothedEntr;
#endif
    MyFloat DensityNorm;
}
*SmoothDataResult, *SmoothDataOut;

void smoothed_values(void)
{
    int ngrp, sendTask, recvTask, place, nexport, nimport;
    int i, j, ndone, ndone_flag, dummy;
    double timeall = 0, timecomp1 = 0, timecomp2 = 0, timecommsumm1 = 0, timecommsumm2 = 0, timewait1 =
        0, timewait2 = 0;
    double timecomp, timecomm, timewait;
    double tstart, tend, t0, t1;

#if defined(BSMOOTH)
    int Smooth_Flag = 0;
    double dB[3];
#endif

    /* Display information message that this step is executed on Task 0 ... */
    if(ThisTask == 0)
    {
        printf("Updating SPH interpolants for:"
#if defined(SMOOTH_PHI)
                " (Phi - Dedner)"
#endif /* SMOOTH_PHI */
#if defined(VECT_POTENTIAL)
                " (Vect A)"
#endif /* VECT_POTENTIAL */
#if defined(DIVBCLEANING_DEDNER) || defined(SCAL_PRO_CLEAN)
                " (DivB) "
#endif
#ifdef SMOOTH_ROTB
                " (rotB)"
#endif /* SMOOTH_ROTB */
#if defined(BSMOOTH)
                " (B)"
#endif /* BSMOOTH */
                "\n");
#ifdef BSMOOTH
        printf("Flag_FullStep = %d, Main TimestepCounts = %d\n", Flag_FullStep, All.MainTimestepCounts);
#endif
    }
#if defined(BSMOOTH)
    if(Flag_FullStep == 1)
    {
        if((All.MainTimestepCounts % All.BSmoothInt == 0) && (All.BSmoothInt >= 0))
        {
            Smooth_Flag = 1;
            if(ThisTask == 0)
                printf("Smoothing B %d, %f\n", All.BSmoothInt, All.BSmoothFrac);
        }
        All.MainTimestepCounts++;
    }
#endif

    Ngblist = (int *) mymalloc("Ngblist", NumPart * sizeof(int));

    All.BunchSize =
        (int) ((All.BufferSize * 1024 * 1024) / (sizeof(struct data_index) + sizeof(struct data_nodelist) +
                    sizeof(struct smoothdata_in) + sizeof(struct smoothdata_out) +
                    sizemax(sizeof(struct smoothdata_in),
                        sizeof(struct smoothdata_out))));

    DataIndexTable =
        (struct data_index *) mymalloc("DataIndexTable", All.BunchSize * sizeof(struct data_index));
    DataNodeList =
        (struct data_nodelist *) mymalloc("DataNodeList", All.BunchSize * sizeof(struct data_nodelist));

    CPU_Step[CPU_SMTHMISC] += measure_time();
    t0 = second();

    i = FirstActiveParticle;	/* begin with this index */

    do
    {
        for(j = 0; j < NTask; j++)
        {
            Send_count[j] = 0;
            Exportflag[j] = -1;
        }

        /* do local particles and prepare export list */
        tstart = second();
        for(nexport = 0; i >= 0; i = NextActiveParticle[i])
        {
            if(density_isactive(i))
                {
                    if(smoothed_evaluate(i, 0, &nexport, Send_count) < 0)
                        break;
                }
        }
        tend = second();
        timecomp1 += timediff(tstart, tend);

#ifdef MYSORT
        mysort_dataindex(DataIndexTable, nexport, sizeof(struct data_index), data_index_compare);
#else
        qsort(DataIndexTable, nexport, sizeof(struct data_index), data_index_compare);
#endif

        tstart = second();

        MPI_Alltoall(Send_count, 1, MPI_INT, Recv_count, 1, MPI_INT, MPI_COMM_WORLD);

        tend = second();
        timewait1 += timediff(tstart, tend);

        for(j = 0, nimport = 0, Recv_offset[0] = 0, Send_offset[0] = 0; j < NTask; j++)
        {
            nimport += Recv_count[j];

            if(j > 0)
            {
                Send_offset[j] = Send_offset[j - 1] + Send_count[j - 1];
                Recv_offset[j] = Recv_offset[j - 1] + Recv_count[j - 1];
            }
        }

        SmoothDataGet =
            (struct smoothdata_in *) mymalloc("SmoothDataGet", nimport * sizeof(struct smoothdata_in));
        SmoothDataIn =
            (struct smoothdata_in *) mymalloc("SmoothDataIn", nexport * sizeof(struct smoothdata_in));

        /* prepare particle data for export */
        for(j = 0; j < nexport; j++)
        {
            place = DataIndexTable[j].Index;

            SmoothDataIn[j].Pos[0] = P[place].Pos[0];
            SmoothDataIn[j].Pos[1] = P[place].Pos[1];
            SmoothDataIn[j].Pos[2] = P[place].Pos[2];
            SmoothDataIn[j].Hsml = PPP[place].Hsml;
            memcpy(SmoothDataIn[j].NodeList,
                    DataNodeList[DataIndexTable[j].IndexGet].NodeList, NODELISTLENGTH * sizeof(int));
        }

        /* exchange particle data */
        tstart = second();
        for(ngrp = 1; ngrp < (1 << PTask); ngrp++)
        {
            sendTask = ThisTask;
            recvTask = ThisTask ^ ngrp;

            if(recvTask < NTask)
            {
                if(Send_count[recvTask] > 0 || Recv_count[recvTask] > 0)
                {
                    /* get the particles */
                    MPI_Sendrecv(&SmoothDataIn[Send_offset[recvTask]],
                            Send_count[recvTask] * sizeof(struct smoothdata_in), MPI_BYTE,
                            recvTask, TAG_SMOOTH_A,
                            &SmoothDataGet[Recv_offset[recvTask]],
                            Recv_count[recvTask] * sizeof(struct smoothdata_in), MPI_BYTE,
                            recvTask, TAG_SMOOTH_A, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
            }
        }
        tend = second();
        timecommsumm1 += timediff(tstart, tend);

        myfree(SmoothDataIn);
        SmoothDataResult =
            (struct smoothdata_out *) mymalloc("SmoothDataResult", nimport * sizeof(struct smoothdata_out));
        SmoothDataOut =
            (struct smoothdata_out *) mymalloc("SmoothDataOut", nexport * sizeof(struct smoothdata_out));


        /* now do the particles that were sent to us */

        tstart = second();
        for(j = 0; j < nimport; j++)
            smoothed_evaluate(j, 1, &dummy, &dummy);
        tend = second();
        timecomp2 += timediff(tstart, tend);

        if(i < 0)
            ndone_flag = 1;
        else
            ndone_flag = 0;

        tstart = second();
        MPI_Allreduce(&ndone_flag, &ndone, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        tend = second();
        timewait2 += timediff(tstart, tend);


        /* get the result */
        tstart = second();
        for(ngrp = 1; ngrp < (1 << PTask); ngrp++)
        {
            sendTask = ThisTask;
            recvTask = ThisTask ^ ngrp;
            if(recvTask < NTask)
            {
                if(Send_count[recvTask] > 0 || Recv_count[recvTask] > 0)
                {
                    /* send the results */
                    MPI_Sendrecv(&SmoothDataResult[Recv_offset[recvTask]],
                            Recv_count[recvTask] * sizeof(struct smoothdata_out),
                            MPI_BYTE, recvTask, TAG_SMOOTH_B,
                            &SmoothDataOut[Send_offset[recvTask]],
                            Send_count[recvTask] * sizeof(struct smoothdata_out),
                            MPI_BYTE, recvTask, TAG_SMOOTH_B, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
            }
        }
        tend = second();
        timecommsumm2 += timediff(tstart, tend);


        /* add the result to the local particles */
        tstart = second();
        for(j = 0; j < nexport; j++)
        {
            place = DataIndexTable[j].Index;

            if(P[place].Type == 0)
            {
#if defined(SMOOTH_PHI)
                SphP[place].SmoothPhi += SmoothDataOut[j].SmoothPhi;
#endif /* SMOOTH_PHI */
#if defined(VECT_POTENTIAL)
                SphP[place].SmoothA[0] += SmoothDataOut[j].SmoothA[0];
                SphP[place].SmoothA[1] += SmoothDataOut[j].SmoothA[1];
                SphP[place].SmoothA[2] += SmoothDataOut[j].SmoothA[2];
#endif /* VECT_POTENTIAL */
#if defined(DIVBCLEANING_DEDNER) || defined(SCAL_PRO_CLEAN)
                SphP[place].SmoothDivB += SmoothDataOut[j].SmoothDivB;
#endif
#ifdef SMOOTH_ROTB
                SphP[place].SmoothedRotB[0] += SmoothDataOut[j].SmoothRotB[0];
                SphP[place].SmoothedRotB[1] += SmoothDataOut[j].SmoothRotB[1];
                SphP[place].SmoothedRotB[2] += SmoothDataOut[j].SmoothRotB[2];
#endif /* SMOOTH_ROTB */

#if defined(BSMOOTH)
                SphP[place].BSmooth[0] += SmoothDataOut[j].BSmooth[0];
                SphP[place].BSmooth[1] += SmoothDataOut[j].BSmooth[1];
                SphP[place].BSmooth[2] += SmoothDataOut[j].BSmooth[2];
#endif /* BSMOOTH */

                SphP[place].DensityNorm += SmoothDataOut[j].DensityNorm;
            }

        }
        tend = second();
        timecomp1 += timediff(tstart, tend);


        myfree(SmoothDataOut);
        myfree(SmoothDataResult);
        myfree(SmoothDataGet);
    }
    while(ndone < NTask);

    myfree(DataNodeList);
    myfree(DataIndexTable);
    myfree(Ngblist);



    /* do final operations on results */
    tstart = second();
    for(i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i])
    {
        if(density_isactive(i))
            {
                if(P[i].Type == 0)
                {
#if defined(SMOOTH_PHI)
                    SphP[i].SmoothPhi /= SphP[i].DensityNorm;
#endif /* SMOOTH_PHI */
#if defined(VECT_POTENTIAL)
                    SphP[i].SmoothA[0] /= SphP[i].DensityNorm;
                    SphP[i].SmoothA[1] /= SphP[i].DensityNorm;
                    SphP[i].SmoothA[2] /= SphP[i].DensityNorm;
                    SphP[i].APred[0] = SphP[i].SmoothA[0];
                    SphP[i].APred[1] = SphP[i].SmoothA[1];
                    SphP[i].APred[2] = SphP[i].SmoothA[2];
#endif
#if defined(DIVBCLEANING_DEDNER) || defined(SCAL_PRO_CLEAN)
                    SphP[i].SmoothDivB /= SphP[i].DensityNorm;
#endif
#ifdef SMOOTH_ROTB
                    SphP[i].SmoothedRotB[0] /= SphP[i].DensityNorm;
                    SphP[i].SmoothedRotB[1] /= SphP[i].DensityNorm;
                    SphP[i].SmoothedRotB[2] /= SphP[i].DensityNorm;
#endif /* SMOOTH_ROTB */

#if defined(BSMOOTH)
                    SphP[i].BSmooth[0] /= SphP[i].DensityNorm;
                    SphP[i].BSmooth[1] /= SphP[i].DensityNorm;
                    SphP[i].BSmooth[2] /= SphP[i].DensityNorm;

                    if(Smooth_Flag == 1)
                    {
                        dB[0] = All.BSmoothFrac * (SphP[i].BSmooth[0] - SphP[i].BPred[0]);
                        dB[1] = All.BSmoothFrac * (SphP[i].BSmooth[1] - SphP[i].BPred[1]);
                        dB[2] = All.BSmoothFrac * (SphP[i].BSmooth[2] - SphP[i].BPred[2]);
                        SphP[i].BPred[0] += dB[0];
                        SphP[i].BPred[1] += dB[1];
                        SphP[i].BPred[2] += dB[2];
#ifndef EULERPOTENTIALS
                        SphP[i].B[0] += dB[0];
                        SphP[i].B[1] += dB[1];
                        SphP[i].B[2] += dB[2];
#endif
                    }
#endif /* BSMOOTH */

                }
            }
    }
    tend = second();
    timecomp1 += timediff(tstart, tend);

    /* collect some timing information */

    t1 = WallclockTime = second();
    timeall += timediff(t0, t1);

    timecomp = timecomp1 + timecomp2;
    timewait = timewait1 + timewait2;
    timecomm = timecommsumm1 + timecommsumm2;

    CPU_Step[CPU_SMTHCOMPUTE] += timecomp;
    CPU_Step[CPU_SMTHWAIT] += timewait;
    CPU_Step[CPU_SMTHCOMM] += timecomm;
    CPU_Step[CPU_SMTHMISC] += timeall - (timecomp + timewait + timecomm);

}



/*! This function represents the core of the SPH density computation. The
 *  target particle may either be local, or reside in the communication
 *  buffer.
 */
int smoothed_evaluate(int target, int mode, int *nexport, int *nsend_local)
{
    int j, n, listindex = 0;
    int startnode, numngb_inbox;
    double h, h2, hinv, hinv3, hinv4;
    double wk, dwk;
    double dx, dy, dz, r, r2, u, mass_j;
    MyFloat *pos;
    double DensityNorm = 0;

#ifdef SMOOTH_PHI
    double SmoothPhi = 0.0;
#endif /* SMOOTH_PHI */
#if defined(VECT_POTENTIAL)
    double smoothA[3];

    smoothA[0] = smoothA[1] = smoothA[2] = 0.0;
#endif
#if defined(DIVBCLEANING_DEDNER) || defined(SCAL_PRO_CLEAN)
    double SmoothDivB = 0.0;
#endif
#ifdef SMOOTH_ROTB
    double smoothrotb[3];
#endif /* SMOOTH_ROTB */

#if defined(BSMOOTH)
    double BSmooth[3];

    BSmooth[0] = BSmooth[1] = BSmooth[2] = 0;
#endif /* BSMOOTH */

    /*  MyIDType myID; */

    int Type;

#ifdef SMOOTH_ROTB
    smoothrotb[0] = smoothrotb[1] = smoothrotb[2] = 0;
#endif /* SMOOTH_ROTB */

#if defined(BSMOOTH)
    BSmooth[0] = BSmooth[1] = BSmooth[2] = 0;
#endif /* BSMOOTH */

    if(mode == 0)
    {
        pos = P[target].Pos;
        h = PPP[target].Hsml;
    }
    else
    {
        pos = SmoothDataGet[target].Pos;
        h = SmoothDataGet[target].Hsml;
    }


    h2 = h * h;
    hinv = 1.0 / h;
#ifndef  TWODIMS
    hinv3 = hinv * hinv * hinv;
#else
    hinv3 = hinv * hinv / boxSize_Z;
#endif
    hinv4 = hinv3 * hinv;

    if(mode == 0)
    {
        startnode = All.MaxPart;	/* root node */
    }
    else
    {
        startnode = SmoothDataGet[target].NodeList[0];
        startnode = Nodes[startnode].u.d.nextnode;	/* open it */
    }

    while(startnode >= 0)
    {
        while(startnode >= 0)
        {
            numngb_inbox = ngb_treefind_variable(&pos[0], h, target, &startnode, mode, nexport, nsend_local);

            if(numngb_inbox < 0)
                return -1;

            for(n = 0; n < numngb_inbox; n++)
            {
                j = Ngblist[n];

                dx = pos[0] - P[j].Pos[0];
                dy = pos[1] - P[j].Pos[1];
                dz = pos[2] - P[j].Pos[2];

#ifdef PERIODIC			/*  now find the closest image in the given box size  */
                if(dx > boxHalf_X)
                    dx -= boxSize_X;
                if(dx < -boxHalf_X)
                    dx += boxSize_X;
                if(dy > boxHalf_Y)
                    dy -= boxSize_Y;
                if(dy < -boxHalf_Y)
                    dy += boxSize_Y;
                if(dz > boxHalf_Z)
                    dz -= boxSize_Z;
                if(dz < -boxHalf_Z)
                    dz += boxSize_Z;
#endif
                r2 = dx * dx + dy * dy + dz * dz;

                if(r2 < h2)
                {
                    r = sqrt(r2);

                    u = r * hinv;

                    if(u < 0.5)
                    {
                        wk = hinv3 * (KERNEL_COEFF_1 + KERNEL_COEFF_2 * (u - 1) * u * u);
                        dwk = hinv4 * u * (KERNEL_COEFF_3 * u - KERNEL_COEFF_4);
                    }
                    else
                    {
                        wk = hinv3 * KERNEL_COEFF_5 * (1.0 - u) * (1.0 - u) * (1.0 - u);
                        dwk = hinv4 * KERNEL_COEFF_6 * (1.0 - u) * (1.0 - u);
                    }

                    mass_j = P[j].Mass;
                    wk /= SphP[j].d.Density;

#if defined(SMOOTH_PHI)
                    SmoothPhi += mass_j * wk * SphP[j].PhiPred;
#endif /* SMOOTH_PHI */
#if defined(VECT_POTENTIAL)
                    smoothA[0] += mass_j * wk * SphP[j].APred[0];
                    smoothA[1] += mass_j * wk * SphP[j].APred[1];
                    smoothA[2] += mass_j * wk * SphP[j].APred[2];
#endif
#if defined(DIVBCLEANING_DEDNER) || defined(SCAL_PRO_CLEAN)
                    SmoothDivB += mass_j * wk * SphP[j].divB;
#endif
#ifdef SMOOTH_ROTB
                    smoothrotb[0] += mass_j * wk * SphP[j].RotB[0];
                    smoothrotb[1] += mass_j * wk * SphP[j].RotB[1];
                    smoothrotb[2] += mass_j * wk * SphP[j].RotB[2];
#endif /* SMOOTH_ROTB */

#if defined(BSMOOTH)
                    BSmooth[0] += mass_j * wk * SphP[j].BPred[0];
                    BSmooth[1] += mass_j * wk * SphP[j].BPred[1];
                    BSmooth[2] += mass_j * wk * SphP[j].BPred[2];
#endif /* BSMOOTH */
                    DensityNorm += mass_j * wk;
                }
            }
        }
        if(mode == 1)
        {
            listindex++;
            if(listindex < NODELISTLENGTH)
            {
                startnode = SmoothDataGet[target].NodeList[listindex];
                if(startnode >= 0)
                    startnode = Nodes[startnode].u.d.nextnode;	/* open it */
            }
        }
    }


    if(mode == 0)
    {
#if defined(SMOOTH_PHI)
            SphP[target].SmoothPhi = SmoothPhi;
#endif /* SMOOTH_PHI */
#if defined(VECT_POTENTIAL)
            SphP[target].SmoothA[0] = smoothA[0];
            SphP[target].SmoothA[1] = smoothA[1];
            SphP[target].SmoothA[2] = smoothA[2];
#endif
#if defined(DIVBCLEANING_DEDNER) || defined(SCAL_PRO_CLEAN)
            SphP[target].SmoothDivB = SmoothDivB;
#endif
#ifdef SMOOTH_ROTB
            SphP[target].SmoothedRotB[0] = smoothrotb[0];
            SphP[target].SmoothedRotB[1] = smoothrotb[1];
            SphP[target].SmoothedRotB[2] = smoothrotb[2];
#endif /* SMOOTH_ROTB */

#if defined(BSMOOTH)
            SphP[target].BSmooth[0] = BSmooth[0];
            SphP[target].BSmooth[1] = BSmooth[1];
            SphP[target].BSmooth[2] = BSmooth[2];
#endif /* BSMOOTH */

            SphP[target].DensityNorm = DensityNorm;

    }
    else
    {
#if defined(SMOOTH_PHI)
        SmoothDataResult[target].SmoothPhi = SmoothPhi;
#endif /* SMOOTH_PHI */
#if defined(VECT_POTENTIAL)
        SmoothDataResult[target].SmoothA[0] = smoothA[0];
        SmoothDataResult[target].SmoothA[1] = smoothA[1];
        SmoothDataResult[target].SmoothA[2] = smoothA[2];
#endif
#if defined(DIVBCLEANING_DEDNER) || defined(SCAL_PRO_CLEAN)
        SmoothDataResult[target].SmoothDivB = SmoothDivB;
#endif
#ifdef SMOOTH_ROTB
        SmoothDataResult[target].SmoothRotB[0] = smoothrotb[0];
        SmoothDataResult[target].SmoothRotB[1] = smoothrotb[1];
        SmoothDataResult[target].SmoothRotB[2] = smoothrotb[2];
#endif /* SMOOTH_ROTB */

#if defined(BSMOOTH)
        SmoothDataResult[target].BSmooth[0] = BSmooth[0];
        SmoothDataResult[target].BSmooth[1] = BSmooth[1];
        SmoothDataResult[target].BSmooth[2] = BSmooth[2];
#endif /* BSMOOTH */

        SmoothDataResult[target].DensityNorm = DensityNorm;

    }

    return 0;
}



#endif
