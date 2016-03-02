#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <math.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <gsl/gsl_math.h>
#include <inttypes.h>

#include "allvars.h"
#include "proto.h"
#include "domain.h"
#include "mpsort/mpsort.h"

/*! \file fof.c
 *  \brief parallel FoF group finder
 */

#ifdef FOF
#include "fof.h"

void fof_save_particles(int num);

int Ngroups, TotNgroups;
int64_t TotNids;

struct group_properties *Group;



static struct fofdata_in
{
    int NodeList[NODELISTLENGTH];
    MyDouble Pos[3];
    MyFloat Hsml;
    MyIDType MinID;
    MyIDType MinIDTask;
}
*FoFDataIn, *FoFDataGet;

struct fofdata_out
{
    MyFloat Distance;
    MyIDType MinID;
    MyIDType MinIDTask;
};


static struct fof_particle_list
{
    MyIDType MinID;
    MyIDType MinIDTask;
    int Pindex;
}
*FOF_PList;

static struct fof_group_list
{
    MyIDType MinID;
    MyIDType MinIDTask;
    int LocCount;
    int ExtCount;
#ifdef DENSITY_SPLIT_BY_TYPE
    int LocDMCount;
    int ExtDMCount;
#endif
    int GrNr;
}
*FOF_GList;

static struct id_list
{
    MyIDType ID;
    unsigned int GrNr;
}
*ID_list;


static double LinkL;
static int NgroupsExt, Nids;


static MyIDType *Head, *Len, *Next, *Tail, *MinID, *MinIDTask;
static char *NonlocalFlag;


static float *fof_nearest_distance;
static float *fof_nearest_hsml;


void fof_fof(int num)
{
    int i, ndm, start, lenloc, largestgroup, n;
    double mass, masstot, rhodm, t0, t1;
    struct unbind_data *d;
    int64_t ndmtot;


    if(ThisTask == 0)
    {
        printf("\nBegin to compute FoF group catalogues...  (presently allocated=%g MB)\n",
                AllocatedBytes / (1024.0 * 1024.0));
        fflush(stdout);
    }

    walltime_measure("/Misc");

    domain_Decomposition();

    for(i = 0, ndm = 0, mass = 0; i < NumPart; i++)
#ifdef DENSITY_SPLIT_BY_TYPE
    {
        if(((1 << P[i].Type) & (FOF_PRIMARY_LINK_TYPES)))
            ndm++;
        if(((1 << P[i].Type) & (DENSITY_SPLIT_BY_TYPE)))
            mass += P[i].Mass;
    }
#else
    if(((1 << P[i].Type) & (FOF_PRIMARY_LINK_TYPES)))
    {
        ndm++;
        mass += P[i].Mass;
    }
#endif

    sumup_large_ints(1, &ndm, &ndmtot);
    MPI_Allreduce(&mass, &masstot, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

#ifdef DENSITY_SPLIT_BY_TYPE
    rhodm = All.Omega0 * 3 * All.Hubble * All.Hubble / (8 * M_PI * All.G);
#else
    rhodm = (All.Omega0 - All.OmegaBaryon) * 3 * All.Hubble * All.Hubble / (8 * M_PI * All.G);
#endif

    LinkL = LINKLENGTH * pow(masstot / ndmtot / rhodm, 1.0 / 3);

    if(ThisTask == 0)
    {
        printf("\nComoving linking length: %g    ", LinkL);
        printf("(presently allocated=%g MB)\n", AllocatedBytes / (1024.0 * 1024.0));
        fflush(stdout);
    }

    FOF_PList =
        (struct fof_particle_list *) mymalloc("FOF_PList", NumPart *
                sizemax(sizeof(struct fof_particle_list), 3 * sizeof(MyIDType)));

    MinID = (MyIDType *) FOF_PList;
    MinIDTask = MinID + NumPart;
    Head = MinIDTask + NumPart;
    Len = (MyIDType *) mymalloc("Len", NumPart * sizeof(MyIDType));
    Next = (MyIDType *) mymalloc("Next", NumPart * sizeof(MyIDType));
    Tail = (MyIDType *) mymalloc("Tail", NumPart * sizeof(MyIDType));

    if(ThisTask == 0)
        printf("Tree construction.\n");

    force_treeallocate((int) (All.TreeAllocFactor * All.MaxPart) + NTopnodes, All.MaxPart);


    /* build index list of particles of selected primary species */
    d = (struct unbind_data *) mymalloc("d", NumPart * sizeof(struct unbind_data));
    for(i = 0, n = 0; i < NumPart; i++)
        if(((1 << P[i].Type) & (FOF_PRIMARY_LINK_TYPES)))
            d[n++].index = i;

    force_treebuild(n, d);

    walltime_measure("/FOF/Build");
    myfree(d);


    for(i = 0; i < NumPart; i++)
    {
        Head[i] = Tail[i] = i;
        Len[i] = 1;
        Next[i] = -1;
        MinID[i] = P[i].ID;
        MinIDTask[i] = ThisTask;
    }


    t0 = second();

    fof_find_groups();

    t1 = second();
    if(ThisTask == 0)
        printf("group finding took = %g sec\n", timediff(t0, t1));
    walltime_measure("/FOF/FindGroups");

    t0 = second();

    fof_find_nearest_dmparticle();
    walltime_measure("/FOF/NearestDM");

    t1 = second();
    if(ThisTask == 0)
        printf("attaching gas and star particles to nearest dm particles took = %g sec\n", timediff(t0, t1));


    t0 = second();

    for(i = 0; i < NumPart; i++)
    {
        Next[i] = MinID[Head[i]];
        Tail[i] = MinIDTask[Head[i]];

        if(Tail[i] >= NTask)	/* it appears that the Intel C 9.1 on Itanium2 produces incorrect code if
                                   this if-statemet is omitted. Apparently, the compiler then joins the two loops,
                                   but this is here not permitted because storage for FOF_PList actually overlaps
                                   (on purpose) with MinID/MinIDTask/Head */
        {
            printf("oh no: ThisTask=%d i=%d Head[i]=%d  NumPart=%d MinIDTask[Head[i]]=%d\n",
                    ThisTask, i, (int) Head[i], NumPart, (int) MinIDTask[Head[i]]);
            fflush(stdout);
            endrun(8812);
        }
    }

    for(i = 0; i < NumPart; i++)
    {
        FOF_PList[i].MinID = Next[i];
        FOF_PList[i].MinIDTask = Tail[i];
        FOF_PList[i].Pindex = i;
    }

    force_treefree();

    myfree(Tail);
    myfree(Next);
    myfree(Len);

    FOF_GList = (struct fof_group_list *) mymalloc("FOF_GList", sizeof(struct fof_group_list) * NumPart);

    fof_compile_catalogue();

    t1 = second();
    if(ThisTask == 0)
        printf("compiling local group data and catalogue took = %g sec\n", timediff(t0, t1));


    MPI_Allreduce(&Ngroups, &TotNgroups, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    sumup_large_ints(1, &Nids, &TotNids);

    if(TotNgroups > 0)
    {
        int largestloc = 0;

        for(i = 0; i < NgroupsExt; i++)
            if(FOF_GList[i].LocCount + FOF_GList[i].ExtCount > largestloc)
                largestloc = FOF_GList[i].LocCount + FOF_GList[i].ExtCount;
        MPI_Allreduce(&largestloc, &largestgroup, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    }
    else
        largestgroup = 0;

    if(ThisTask == 0)
    {
        printf("\nTotal number of groups with at least %d particles: %d\n", FOF_GROUP_MIN_LEN, TotNgroups);
        if(TotNgroups > 0)
        {
            printf("Largest group has %d particles.\n", largestgroup);
            printf("Total number of particles in groups: %d%09d\n\n",
                    (int) (TotNids / 1000000000), (int) (TotNids % 1000000000));
        }
    }
    walltime_measure("/FOF/Compile");

    t0 = second();

    Group =
        (struct group_properties *) mymalloc("Group", sizeof(struct group_properties) *
                IMAX(NgroupsExt, TotNgroups / NTask + 1));

    if(ThisTask == 0)
    {
        printf("group properties are now allocated.. (presently allocated=%g MB)\n",
                AllocatedBytes / (1024.0 * 1024.0));
        fflush(stdout);
    }

    for(i = 0, start = 0; i < NgroupsExt; i++)
    {
        while(FOF_PList[start].MinID < FOF_GList[i].MinID)
        {
            start++;
            if(start > NumPart)
                endrun(78);
        }

        if(FOF_PList[start].MinID != FOF_GList[i].MinID)
            endrun(123);

        for(lenloc = 0; start + lenloc < NumPart;)
            if(FOF_PList[start + lenloc].MinID == FOF_GList[i].MinID)
                lenloc++;
            else
                break;

        Group[i].MinID = FOF_GList[i].MinID;
        Group[i].MinIDTask = FOF_GList[i].MinIDTask;

        fof_compute_group_properties(i, start, lenloc);

        start += lenloc;
    }

    fof_exchange_group_data();

    fof_finish_group_properties();

    walltime_measure("/FOF/Prop");
    t1 = second();
    if(ThisTask == 0)
        printf("computation of group properties took = %g sec\n", timediff(t0, t1));

#ifdef BLACK_HOLES
    if(num < 0)
        fof_make_black_holes();
#endif

    walltime_measure("/FOF/Misc");

    if(num >= 0)
    {
        fof_save_groups(num);
    }

    myfree(Group);

    myfree(FOF_GList);
    myfree(FOF_PList);

    if(ThisTask == 0)
    {
        printf("Finished computing FoF groups.  (presently allocated=%g MB)\n\n",
                AllocatedBytes / (1024.0 * 1024.0));
        fflush(stdout);
    }


    domain_Decomposition();

    walltime_measure("/FOF/MISC");
    force_treebuild_simple();
}



void fof_find_groups(void)
{
    int i, j, ndone_flag, link_count, dummy, nprocessed;
    int ndone, ngrp, recvTask, place, nexport, nimport, link_across;
    int npart, marked;
    int64_t totmarked, totnpart;
    int64_t link_across_tot, ntot;
    MyIDType *MinIDOld;
    char *FoFDataOut, *FoFDataResult, *MarkedFlag, *ChangedFlag;
    double t0, t1;

    if(ThisTask == 0)
    {
        printf("\nStart linking particles (presently allocated=%g MB)\n", AllocatedBytes / (1024.0 * 1024.0));
        fflush(stdout);
    }


    /* allocate buffers to arrange communication */

    Ngblist = (int *) mymalloc("Ngblist", NumPart * sizeof(int));

    All.BunchSize =
        (int) ((All.BufferSize * 1024 * 1024) / (sizeof(struct data_index) + sizeof(struct data_nodelist) +
                    2 * sizeof(struct fofdata_in)));
    DataIndexTable =
        (struct data_index *) mymalloc("DataIndexTable", All.BunchSize * sizeof(struct data_index));
    DataNodeList =
        (struct data_nodelist *) mymalloc("DataNodeList", All.BunchSize * sizeof(struct data_nodelist));

    NonlocalFlag = (char *) mymalloc("NonlocalFlag", NumPart * sizeof(char));
    MarkedFlag = (char *) mymalloc("MarkedFlag", NumPart * sizeof(char));
    ChangedFlag = (char *) mymalloc("ChangedFlag", NumPart * sizeof(char));
    MinIDOld = (MyIDType *) mymalloc("MinIDOld", NumPart * sizeof(MyIDType));

    t0 = second();

    /* first, link only among local particles */
    for(i = 0, marked = 0, npart = 0; i < NumPart; i++)
    {
        if(((1 << P[i].Type) & (FOF_PRIMARY_LINK_TYPES)))
        {
            fof_find_dmparticles_evaluate(i, -1, &dummy, &dummy);

            npart++;

            if(NonlocalFlag[i])
                marked++;
        }
    }


    sumup_large_ints(1, &marked, &totmarked);
    sumup_large_ints(1, &npart, &totnpart);

    t1 = second();


    if(ThisTask == 0)
    {
        printf
            ("links on local processor done (took %g sec).\nMarked=%d%09d out of the %d%09d primaries which are linked\n",
             timediff(t0, t1),
             (int) (totmarked / 1000000000), (int) (totmarked % 1000000000),
             (int) (totnpart / 1000000000), (int) (totnpart % 1000000000));

        printf("\nlinking across processors (presently allocated=%g MB) \n",
                AllocatedBytes / (1024.0 * 1024.0));
        fflush(stdout);
    }

    for(i = 0; i < NumPart; i++)
    {
        MinIDOld[i] = MinID[Head[i]];
        MarkedFlag[i] = 1;
    }

    do
    {
        t0 = second();

        for(i = 0; i < NumPart; i++)
        {
            ChangedFlag[i] = MarkedFlag[i];
            MarkedFlag[i] = 0;
        }

        i = 0;			/* begin with this index */
        link_across = 0;
        nprocessed = 0;

        do
        {
            for(j = 0; j < NTask; j++)
            {
                Send_count[j] = 0;
                Exportflag[j] = -1;
            }

            /* do local particles and prepare export list */
            for(nexport = 0; i < NumPart; i++)
            {
                if(((1 << P[i].Type) & (FOF_PRIMARY_LINK_TYPES)))
                {
                    if(NonlocalFlag[i] && ChangedFlag[i])
                    {
                        if(fof_find_dmparticles_evaluate(i, 0, &nexport, Send_count) < 0)
                            break;

                        nprocessed++;
                    }
                }
            }

#ifdef MYSORT
            mysort_dataindex(DataIndexTable, nexport, sizeof(struct data_index), data_index_compare);
#else
            qsort(DataIndexTable, nexport, sizeof(struct data_index), data_index_compare);
#endif

            MPI_Alltoall(Send_count, 1, MPI_INT, Recv_count, 1, MPI_INT, MPI_COMM_WORLD);

            for(j = 0, nimport = 0, Recv_offset[0] = 0, Send_offset[0] = 0; j < NTask; j++)
            {
                nimport += Recv_count[j];

                if(j > 0)
                {
                    Send_offset[j] = Send_offset[j - 1] + Send_count[j - 1];
                    Recv_offset[j] = Recv_offset[j - 1] + Recv_count[j - 1];
                }
            }

            FoFDataGet = (struct fofdata_in *) mymalloc("FoFDataGet", nimport * sizeof(struct fofdata_in));
            FoFDataIn = (struct fofdata_in *) mymalloc("FoFDataIn", nexport * sizeof(struct fofdata_in));


            /* prepare particle data for export */
            for(j = 0; j < nexport; j++)
            {
                place = DataIndexTable[j].Index;

                FoFDataIn[j].Pos[0] = P[place].Pos[0];
                FoFDataIn[j].Pos[1] = P[place].Pos[1];
                FoFDataIn[j].Pos[2] = P[place].Pos[2];
                FoFDataIn[j].MinID = MinID[Head[place]];
                FoFDataIn[j].MinIDTask = MinIDTask[Head[place]];

                memcpy(FoFDataIn[j].NodeList,
                        DataNodeList[DataIndexTable[j].IndexGet].NodeList, NODELISTLENGTH * sizeof(int));
            }

            /* exchange particle data */
            for(ngrp = 1; ngrp < (1 << PTask); ngrp++)
            {
                recvTask = ThisTask ^ ngrp;

                if(recvTask < NTask)
                {
                    if(Send_count[recvTask] > 0 || Recv_count[recvTask] > 0)
                    {
                        /* get the particles */
                        MPI_Sendrecv(&FoFDataIn[Send_offset[recvTask]],
                                Send_count[recvTask] * sizeof(struct fofdata_in), MPI_BYTE,
                                recvTask, TAG_DENS_A,
                                &FoFDataGet[Recv_offset[recvTask]],
                                Recv_count[recvTask] * sizeof(struct fofdata_in), MPI_BYTE,
                                recvTask, TAG_DENS_A, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    }
                }
            }

            myfree(FoFDataIn);
            FoFDataResult = (char *) mymalloc("FoFDataResult", nimport * sizeof(char));
            FoFDataOut = (char *) mymalloc("FoFDataOut", nexport * sizeof(char));

            /* now do the particles that were sent to us */

            for(j = 0; j < nimport; j++)
            {
                link_count = fof_find_dmparticles_evaluate(j, 1, &dummy, &dummy);
                link_across += link_count;
                if(link_count)
                    FoFDataResult[j] = 1;
                else
                    FoFDataResult[j] = 0;
            }

            /* exchange data */
            for(ngrp = 1; ngrp < (1 << PTask); ngrp++)
            {
                recvTask = ThisTask ^ ngrp;

                if(recvTask < NTask)
                {
                    if(Send_count[recvTask] > 0 || Recv_count[recvTask] > 0)
                    {
                        /* get the particles */
                        MPI_Sendrecv(&FoFDataResult[Recv_offset[recvTask]],
                                Recv_count[recvTask] * sizeof(char),
                                MPI_BYTE, recvTask, TAG_DENS_B,
                                &FoFDataOut[Send_offset[recvTask]],
                                Send_count[recvTask] * sizeof(char),
                                MPI_BYTE, recvTask, TAG_DENS_B, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    }
                }
            }

            /* need to mark the particle if it induced a link */
            for(j = 0; j < nexport; j++)
            {
                place = DataIndexTable[j].Index;
                if(FoFDataOut[j])
                    MarkedFlag[place] = 1;
            }

            myfree(FoFDataOut);
            myfree(FoFDataResult);
            myfree(FoFDataGet);

            if(i >= NumPart)
                ndone_flag = 1;
            else
                ndone_flag = 0;

            MPI_Allreduce(&ndone_flag, &ndone, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        }
        while(ndone < NTask);


        sumup_large_ints(1, &link_across, &link_across_tot);
        sumup_large_ints(1, &nprocessed, &ntot);

        t1 = second();

        if(ThisTask == 0)
        {
            printf("have done %d%09d cross links (processed %d%09d, took %g sec)\n",
                    (int) (link_across_tot / 1000000000), (int) (link_across_tot % 1000000000),
                    (int) (ntot / 1000000000), (int) (ntot % 1000000000), timediff(t0, t1));
            fflush(stdout);
        }


        /* let's check out which particles have changed their MinID */
        for(i = 0; i < NumPart; i++)
            if(NonlocalFlag[i])
            {
                if(MinID[Head[i]] != MinIDOld[i])
                    MarkedFlag[i] = 1;

                MinIDOld[i] = MinID[Head[i]];
            }

    }
    while(link_across_tot > 0);

    myfree(MinIDOld);
    myfree(ChangedFlag);
    myfree(MarkedFlag);
    myfree(NonlocalFlag);

    myfree(DataNodeList);
    myfree(DataIndexTable);
    myfree(Ngblist);

    if(ThisTask == 0)
    {
        printf("Local groups found.\n\n");
        fflush(stdout);
    }
}


int fof_find_dmparticles_evaluate(int target, int mode, int *nexport, int *nsend_local)
{
    int j, n, links, p, s, ss, listindex = 0;
    int startnode, numngb_inbox;
    MyDouble *pos;

    links = 0;

    if(mode == 0 || mode == -1)
        pos = P[target].Pos;
    else
        pos = FoFDataGet[target].Pos;

    if(mode == 0 || mode == -1)
    {
        startnode = All.MaxPart;	/* root node */
    }
    else
    {
        startnode = FoFDataGet[target].NodeList[0];
        startnode = Nodes[startnode].u.d.nextnode;	/* open it */
    }

    while(startnode >= 0)
    {
        while(startnode >= 0)
        {
            if(mode == -1)
                *nexport = 0;

            numngb_inbox = ngb_treefind_fof_primary(pos, LinkL, target, &startnode, mode, nexport, nsend_local);

            if(numngb_inbox < 0)
                return -1;

            if(mode == -1)
            {
                if(*nexport == 0)
                    NonlocalFlag[target] = 0;
                else
                    NonlocalFlag[target] = 1;
            }

            for(n = 0; n < numngb_inbox; n++)
            {
                j = Ngblist[n];

                if(mode == 0 || mode == -1)
                {
                    if(Head[target] != Head[j])	/* only if not yet linked */
                    {

                        if(mode == 0)
                            endrun(87654);

                        if(Len[Head[target]] > Len[Head[j]])	/* p group is longer */
                        {
                            p = target;
                            s = j;
                        }
                        else
                        {
                            p = j;
                            s = target;
                        }
                        Next[Tail[Head[p]]] = Head[s];

                        Tail[Head[p]] = Tail[Head[s]];

                        Len[Head[p]] += Len[Head[s]];

                        ss = Head[s];
                        do
                            Head[ss] = Head[p];
                        while((ss = Next[ss]) >= 0);

                        if(MinID[Head[s]] < MinID[Head[p]])
                        {
                            MinID[Head[p]] = MinID[Head[s]];
                            MinIDTask[Head[p]] = MinIDTask[Head[s]];
                        }
                    }
                }
                else		/* mode is 1 */
                {
                    if(MinID[Head[j]] > FoFDataGet[target].MinID)
                    {
                        MinID[Head[j]] = FoFDataGet[target].MinID;
                        MinIDTask[Head[j]] = FoFDataGet[target].MinIDTask;
                        links++;
                    }
                }
            }
        }

        if(mode == 1)
        {
            listindex++;
            if(listindex < NODELISTLENGTH)
            {
                startnode = FoFDataGet[target].NodeList[listindex];
                if(startnode >= 0)
                    startnode = Nodes[startnode].u.d.nextnode;	/* open it */
            }
        }
    }

    return links;
}



void fof_compile_catalogue(void)
{
    int i, j, start, nimport, ngrp, recvTask;
    struct fof_group_list *get_FOF_GList;

    /* sort according to MinID */
    qsort(FOF_PList, NumPart, sizeof(struct fof_particle_list), fof_compare_FOF_PList_MinID);

    for(i = 0; i < NumPart; i++)
    {
        FOF_GList[i].MinID = FOF_PList[i].MinID;
        FOF_GList[i].MinIDTask = FOF_PList[i].MinIDTask;
        if(FOF_GList[i].MinIDTask == ThisTask)
        {
            FOF_GList[i].LocCount = 1;
            FOF_GList[i].ExtCount = 0;
#ifdef DENSITY_SPLIT_BY_TYPE
            if(((1 << P[FOF_PList[i].Pindex].Type) & (FOF_PRIMARY_LINK_TYPES)))
                FOF_GList[i].LocDMCount = 1;
            else
                FOF_GList[i].LocDMCount = 0;
            FOF_GList[i].ExtDMCount = 0;
#endif
        }
        else
        {
            FOF_GList[i].LocCount = 0;
            FOF_GList[i].ExtCount = 1;
#ifdef DENSITY_SPLIT_BY_TYPE
            FOF_GList[i].LocDMCount = 0;
            if(((1 << P[FOF_PList[i].Pindex].Type) & (FOF_PRIMARY_LINK_TYPES)))
                FOF_GList[i].ExtDMCount = 1;
            else
                FOF_GList[i].ExtDMCount = 0;
#endif
        }
    }

    /* eliminate duplicates in FOF_GList with respect to MinID */

    if(NumPart)
        NgroupsExt = 1;
    else
        NgroupsExt = 0;

    for(i = 1, start = 0; i < NumPart; i++)
    {
        if(FOF_GList[i].MinID == FOF_GList[start].MinID)
        {
            FOF_GList[start].LocCount += FOF_GList[i].LocCount;
            FOF_GList[start].ExtCount += FOF_GList[i].ExtCount;
#ifdef DENSITY_SPLIT_BY_TYPE
            FOF_GList[start].LocDMCount += FOF_GList[i].LocDMCount;
            FOF_GList[start].ExtDMCount += FOF_GList[i].ExtDMCount;
#endif
        }
        else
        {
            start = NgroupsExt;
            FOF_GList[start] = FOF_GList[i];
            NgroupsExt++;
        }
    }


    /* sort the remaining ones according to task */
    qsort(FOF_GList, NgroupsExt, sizeof(struct fof_group_list), fof_compare_FOF_GList_MinIDTask);

    /* count how many we have of each task */
    for(i = 0; i < NTask; i++)
        Send_count[i] = 0;
    for(i = 0; i < NgroupsExt; i++)
        Send_count[FOF_GList[i].MinIDTask]++;

    MPI_Alltoall(Send_count, 1, MPI_INT, Recv_count, 1, MPI_INT, MPI_COMM_WORLD);

    for(j = 0, nimport = 0, Recv_offset[0] = 0, Send_offset[0] = 0; j < NTask; j++)
    {
        if(j == ThisTask)		/* we will not exchange the ones that are local */
            Recv_count[j] = 0;
        nimport += Recv_count[j];

        if(j > 0)
        {
            Send_offset[j] = Send_offset[j - 1] + Send_count[j - 1];
            Recv_offset[j] = Recv_offset[j - 1] + Recv_count[j - 1];
        }
    }

    get_FOF_GList =
        (struct fof_group_list *) mymalloc("get_FOF_GList", nimport * sizeof(struct fof_group_list));

    for(ngrp = 1; ngrp < (1 << PTask); ngrp++)
    {
        recvTask = ThisTask ^ ngrp;

        if(recvTask < NTask)
        {
            if(Send_count[recvTask] > 0 || Recv_count[recvTask] > 0)
            {
                /* get the group info */
                MPI_Sendrecv(&FOF_GList[Send_offset[recvTask]],
                        Send_count[recvTask] * sizeof(struct fof_group_list), MPI_BYTE,
                        recvTask, TAG_DENS_A,
                        &get_FOF_GList[Recv_offset[recvTask]],
                        Recv_count[recvTask] * sizeof(struct fof_group_list), MPI_BYTE,
                        recvTask, TAG_DENS_A, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
    }

    for(i = 0; i < nimport; i++)
        get_FOF_GList[i].MinIDTask = i;


    /* sort the groups according to MinID */
    qsort(FOF_GList, NgroupsExt, sizeof(struct fof_group_list), fof_compare_FOF_GList_MinID);
    qsort(get_FOF_GList, nimport, sizeof(struct fof_group_list), fof_compare_FOF_GList_MinID);

    /* merge the imported ones with the local ones */
    for(i = 0, start = 0; i < nimport; i++)
    {
        while(FOF_GList[start].MinID < get_FOF_GList[i].MinID)
        {
            start++;
            if(start >= NgroupsExt)
                endrun(7973);
        }

        if(get_FOF_GList[i].LocCount != 0)
            endrun(123);

        if(FOF_GList[start].MinIDTask != ThisTask)
            endrun(124);

        FOF_GList[start].ExtCount += get_FOF_GList[i].ExtCount;
#ifdef DENSITY_SPLIT_BY_TYPE
        FOF_GList[start].ExtDMCount += get_FOF_GList[i].ExtDMCount;
#endif
    }

    /* copy the size information back into the list, to inform the others */
    for(i = 0, start = 0; i < nimport; i++)
    {
        while(FOF_GList[start].MinID < get_FOF_GList[i].MinID)
        {
            start++;
            if(start >= NgroupsExt)
                endrun(797831);
        }

        get_FOF_GList[i].ExtCount = FOF_GList[start].ExtCount;
        get_FOF_GList[i].LocCount = FOF_GList[start].LocCount;
#ifdef DENSITY_SPLIT_BY_TYPE
        get_FOF_GList[i].ExtDMCount = FOF_GList[start].ExtDMCount;
        get_FOF_GList[i].LocDMCount = FOF_GList[start].LocDMCount;
#endif
    }

    /* sort the imported/exported list according to MinIDTask */
    qsort(get_FOF_GList, nimport, sizeof(struct fof_group_list), fof_compare_FOF_GList_MinIDTask);
    qsort(FOF_GList, NgroupsExt, sizeof(struct fof_group_list), fof_compare_FOF_GList_MinIDTask);


    for(i = 0; i < nimport; i++)
        get_FOF_GList[i].MinIDTask = ThisTask;

    for(ngrp = 1; ngrp < (1 << PTask); ngrp++)
    {
        recvTask = ThisTask ^ ngrp;

        if(recvTask < NTask)
        {
            if(Send_count[recvTask] > 0 || Recv_count[recvTask] > 0)
            {
                /* get the group info */
                MPI_Sendrecv(&get_FOF_GList[Recv_offset[recvTask]],
                        Recv_count[recvTask] * sizeof(struct fof_group_list), MPI_BYTE,
                        recvTask, TAG_DENS_A,
                        &FOF_GList[Send_offset[recvTask]],
                        Send_count[recvTask] * sizeof(struct fof_group_list), MPI_BYTE,
                        recvTask, TAG_DENS_A, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
    }

    myfree(get_FOF_GList);

    /* eliminate all groups that are too small, and count local groups */
    for(i = 0, Ngroups = 0, Nids = 0; i < NgroupsExt; i++)
    {
#ifdef DENSITY_SPLIT_BY_TYPE
        if(FOF_GList[i].LocDMCount + FOF_GList[i].ExtDMCount < FOF_GROUP_MIN_LEN)
#else
            if(FOF_GList[i].LocCount + FOF_GList[i].ExtCount < FOF_GROUP_MIN_LEN)
#endif
            {
                FOF_GList[i] = FOF_GList[NgroupsExt - 1];
                NgroupsExt--;
                i--;
            }
            else
            {
                if(FOF_GList[i].MinIDTask == ThisTask)
                {
                    Ngroups++;
                    Nids += FOF_GList[i].LocCount + FOF_GList[i].ExtCount;
                }
            }
    }

    /* sort the group list according to MinID */
    qsort(FOF_GList, NgroupsExt, sizeof(struct fof_group_list), fof_compare_FOF_GList_MinID);
}



void fof_compute_group_properties(int gr, int start, int len)
{
    int j, k, index;
    double xyz[3];

    Group[gr].Len = 0;
    Group[gr].Mass = 0;
#ifdef SFR
    Group[gr].Sfr = 0;
#endif
#ifdef BLACK_HOLES
    Group[gr].BH_Mass = 0;
    Group[gr].BH_Mdot = 0;
    Group[gr].index_maxdens = Group[gr].task_maxdens = -1;
    Group[gr].MaxDens = 0;
#endif

    for(k = 0; k < 3; k++)
    {
        Group[gr].CM[k] = 0;
        Group[gr].Vel[k] = 0;
        Group[gr].FirstPos[k] = P[FOF_PList[start].Pindex].Pos[k];
    }

    for(k = 0; k < 6; k++)
    {
        Group[gr].LenType[k] = 0;
        Group[gr].MassType[k] = 0;
    }

    for(k = 0; k < len; k++)
    {
        index = FOF_PList[start + k].Pindex;

        Group[gr].Len++;
        Group[gr].Mass += P[index].Mass;
        Group[gr].LenType[P[index].Type]++;
        Group[gr].MassType[P[index].Type] += P[index].Mass;


#ifdef SFR
        if(P[index].Type == 0) {
            Group[gr].Sfr += get_starformation_rate(index);
        }
#endif
#ifdef BLACK_HOLES
        if(P[index].Type == 5)
        {
            Group[gr].BH_Mdot += BHP(index).Mdot;
            Group[gr].BH_Mass += BHP(index).Mass;
        }
        if(P[index].Type == 0)
        {
#ifdef WINDS
            /* make bh in non wind gas on bh wind*/
            if(SPHP(index).DelayTime <= 0)
#endif
                if(SPHP(index).Density > Group[gr].MaxDens)
                {
                    Group[gr].MaxDens = SPHP(index).Density;
                    Group[gr].index_maxdens = index;
                    Group[gr].task_maxdens = ThisTask;
                }
        }
#endif

        for(j = 0; j < 3; j++)
        {
            xyz[j] = P[index].Pos[j];
#ifdef PERIODIC
            xyz[j] = fof_periodic(xyz[j] - Group[gr].FirstPos[j]);
#endif
            Group[gr].CM[j] += P[index].Mass * xyz[j];
            Group[gr].Vel[j] += P[index].Mass * P[index].Vel[j];
        }
    }
}


void fof_exchange_group_data(void)
{
    struct group_properties *get_Group;
    int i, j, ngrp, recvTask, nimport, start;
    double xyz[3];

    /* sort the groups according to task */
    qsort(Group, NgroupsExt, sizeof(struct group_properties), fof_compare_Group_MinIDTask);

    /* count how many we have of each task */
    for(i = 0; i < NTask; i++)
        Send_count[i] = 0;
    for(i = 0; i < NgroupsExt; i++)
        Send_count[FOF_GList[i].MinIDTask]++;

    MPI_Alltoall(Send_count, 1, MPI_INT, Recv_count, 1, MPI_INT, MPI_COMM_WORLD);

    for(j = 0, nimport = 0, Recv_offset[0] = 0, Send_offset[0] = 0; j < NTask; j++)
    {
        if(j == ThisTask)		/* we will not exchange the ones that are local */
            Recv_count[j] = 0;
        nimport += Recv_count[j];

        if(j > 0)
        {
            Send_offset[j] = Send_offset[j - 1] + Send_count[j - 1];
            Recv_offset[j] = Recv_offset[j - 1] + Recv_count[j - 1];
        }
    }

    get_Group = (struct group_properties *) mymalloc("get_Group", sizeof(struct group_properties) * nimport);

    for(ngrp = 1; ngrp < (1 << PTask); ngrp++)
    {
        recvTask = ThisTask ^ ngrp;

        if(recvTask < NTask)
        {
            if(Send_count[recvTask] > 0 || Recv_count[recvTask] > 0)
            {
                /* get the group data */
                MPI_Sendrecv(&Group[Send_offset[recvTask]],
                        Send_count[recvTask] * sizeof(struct group_properties), MPI_BYTE,
                        recvTask, TAG_DENS_A,
                        &get_Group[Recv_offset[recvTask]],
                        Recv_count[recvTask] * sizeof(struct group_properties), MPI_BYTE,
                        recvTask, TAG_DENS_A, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
    }

    /* sort the groups again according to MinID */
    qsort(Group, NgroupsExt, sizeof(struct group_properties), fof_compare_Group_MinID);
    qsort(get_Group, nimport, sizeof(struct group_properties), fof_compare_Group_MinID);

    /* now add in the partial imported group data to the main ones */
    for(i = 0, start = 0; i < nimport; i++)
    {
        while(Group[start].MinID < get_Group[i].MinID)
        {
            start++;
            if(start >= NgroupsExt)
                endrun(797890);
        }

        Group[start].Len += get_Group[i].Len;
        Group[start].Mass += get_Group[i].Mass;

        for(j = 0; j < 6; j++)
        {
            Group[start].LenType[j] += get_Group[i].LenType[j];
            Group[start].MassType[j] += get_Group[i].MassType[j];
        }

#ifdef SFR
        Group[start].Sfr += get_Group[i].Sfr;
#endif
#ifdef BLACK_HOLES
        Group[start].BH_Mdot += get_Group[i].BH_Mdot;
        Group[start].BH_Mass += get_Group[i].BH_Mass;
        if(get_Group[i].MaxDens > Group[start].MaxDens)
        {
            Group[start].MaxDens = get_Group[i].MaxDens;
            Group[start].index_maxdens = get_Group[i].index_maxdens;
            Group[start].task_maxdens = get_Group[i].task_maxdens;
        }
#endif

        for(j = 0; j < 3; j++)
        {
            xyz[j] = get_Group[i].CM[j] / get_Group[i].Mass + get_Group[i].FirstPos[j];
#ifdef PERIODIC
            xyz[j] = fof_periodic(xyz[j] - Group[start].FirstPos[j]);
#endif
            Group[start].CM[j] += get_Group[i].Mass * xyz[j];
            Group[start].Vel[j] += get_Group[i].Vel[j];
        }
    }

    myfree(get_Group);
}

void fof_finish_group_properties(void)
{
    double cm[3];
    int i, j, ngr;

    for(i = 0; i < NgroupsExt; i++)
    {
        if(Group[i].MinIDTask == ThisTask)
        {
            for(j = 0; j < 3; j++)
            {
                Group[i].Vel[j] /= Group[i].Mass;

                cm[j] = Group[i].CM[j] / Group[i].Mass;
#ifdef PERIODIC
                cm[j] = fof_periodic_wrap(cm[j] + Group[i].FirstPos[j]);
#endif
                Group[i].CM[j] = cm[j];
            }
        }
    }

    /* eliminate the non-local groups */
    for(i = 0, ngr = NgroupsExt; i < ngr; i++)
    {
        if(Group[i].MinIDTask != ThisTask)
        {
            Group[i] = Group[ngr - 1];
            i--;
            ngr--;
        }
    }

    if(ngr != Ngroups)
        endrun(876889);

    qsort(Group, Ngroups, sizeof(struct group_properties), fof_compare_Group_MinID);
}



static void fof_radix_Group_GrNr(const void * a, void * radix, void * arg);
static void fof_radix_FOF_GList_LocCountTaskDiffMinID(const void * a, void * radix, void * arg);
static void fof_radix_FOF_GList_ExtCountMinID(const void * a, void * radix, void * arg);

void fof_save_groups(int num)
{
    int i, j, start, lenloc, ngr, totlen;
    int64_t totNids;
    double t0, t1;

    if(ThisTask == 0)
    {
        printf("start global sorting of group catalogues\n");
        fflush(stdout);
    }

    t0 = second();

    /* assign group numbers (at this point, both Group and FOF_GList are sorted by MinID) */
    for(i = 0; i < NgroupsExt; i++)
    {
        FOF_GList[i].LocCount += FOF_GList[i].ExtCount;	/* total length */
        FOF_GList[i].ExtCount = ThisTask;	/* original task */
#ifdef DENSITY_SPLIT_BY_TYPE
        FOF_GList[i].LocDMCount += FOF_GList[i].ExtDMCount;	/* total length */
        FOF_GList[i].ExtDMCount = ThisTask;	/* not longer needed/used (hopefully) */
#endif
    }

    mpsort_mpi(FOF_GList, NgroupsExt, sizeof(struct fof_group_list),
            fof_radix_FOF_GList_LocCountTaskDiffMinID, 24, NULL, MPI_COMM_WORLD);

    for(i = 0, ngr = 0; i < NgroupsExt; i++)
    {
        if(FOF_GList[i].ExtCount == FOF_GList[i].MinIDTask)
            ngr++;

        FOF_GList[i].GrNr = ngr;
    }

    MPI_Allgather(&ngr, 1, MPI_INT, Send_count, 1, MPI_INT, MPI_COMM_WORLD);
    for(j = 1, Send_offset[0] = 0; j < NTask; j++)
        Send_offset[j] = Send_offset[j - 1] + Send_count[j - 1];

    for(i = 0; i < NgroupsExt; i++)
        FOF_GList[i].GrNr += Send_offset[ThisTask];


    MPI_Allreduce(&ngr, &i, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    if(i != TotNgroups)
    {
        printf("i=%d\n", i);
        endrun(123123);
    }

    /* bring the group list back into the original order */
      mpsort_mpi(FOF_GList, NgroupsExt, sizeof(struct fof_group_list), 
            fof_radix_FOF_GList_ExtCountMinID, 16, NULL, MPI_COMM_WORLD);

    /* Assign the group numbers to the group properties array */
    for(i = 0, start = 0; i < Ngroups; i++)
    {
        while(FOF_GList[start].MinID < Group[i].MinID)
        {
            start++;
            if(start >= NgroupsExt)
                endrun(7297890);
        }
        Group[i].GrNr = FOF_GList[start].GrNr;
    }

    /* sort the groups according to group-number */
    mpsort_mpi(Group, Ngroups, sizeof(struct group_properties), 
            fof_radix_Group_GrNr, 8, NULL, MPI_COMM_WORLD);

    /* fill in the offset-values */
    for(i = 0, totlen = 0; i < Ngroups; i++)
    {
        if(i > 0)
            Group[i].Offset = Group[i - 1].Offset + Group[i - 1].Len;
        else
            Group[i].Offset = 0;
        totlen += Group[i].Len;
    }

    MPI_Allgather(&totlen, 1, MPI_INT, Send_count, 1, MPI_INT, MPI_COMM_WORLD);
    ptrdiff_t *uoffset = mymalloc("uoffset", NTask * sizeof(ptrdiff_t));

    for(j = 1, uoffset[0] = 0; j < NTask; j++)
        uoffset[j] = uoffset[j - 1] + Send_count[j - 1];

    for(i = 0; i < Ngroups; i++)
        Group[i].Offset += uoffset[ThisTask];

    myfree(uoffset);

    /* prepare list of ids with assigned group numbers */

    ID_list = mymalloc("ID_list", sizeof(struct id_list) * NumPart);

    for(i = 0; i < NumPart; i++)
        P[i].GrNr = -1;	/* will mark particles that are not in any group */

    for(i = 0, start = 0, Nids = 0; i < NgroupsExt; i++)
    {
        while(FOF_PList[start].MinID < FOF_GList[i].MinID)
        {
            start++;
            if(start > NumPart)
                endrun(78);
        }

        if(FOF_PList[start].MinID != FOF_GList[i].MinID)
            endrun(1313);

        for(lenloc = 0; start + lenloc < NumPart;)
            if(FOF_PList[start + lenloc].MinID == FOF_GList[i].MinID)
            {
                ID_list[Nids].GrNr = FOF_GList[i].GrNr;
                ID_list[Nids].ID = P[FOF_PList[start + lenloc].Pindex].ID;
                P[FOF_PList[start + lenloc].Pindex].GrNr = FOF_GList[i].GrNr;
                Nids++;
                lenloc++;
            }
            else
                break;

        start += lenloc;
    }

    sumup_large_ints(1, &Nids, &totNids);

    MPI_Allgather(&Nids, 1, MPI_INT, Send_count, 1, MPI_INT, MPI_COMM_WORLD);
    for(j = 1, Send_offset[0] = 0; j < NTask; j++)
        Send_offset[j] = Send_offset[j - 1] + Send_count[j - 1];


    if(totNids != TotNids)
    {
        printf("Task=%d Nids=%d totNids=%d TotNids=%d\n", ThisTask, Nids, (int) totNids, (int) TotNids);
        endrun(12);
    }
    
    myfree(ID_list);

    fof_save_particles(num);

    t1 = second();

    if(ThisTask == 0)
    {
        printf("Group catalogues saved. took = %g sec\n", timediff(t0, t1));
        fflush(stdout);
    }
}

static void fof_nearest_copy(int place, struct fofdata_in * I) {
    int k;
    for (k = 0; k < 3; k ++) {
        I->Pos[k] = P[place].Pos[k];
    }
    I->Hsml = fof_nearest_hsml[place];
}
static void * fof_nearest_ngblist() {
    int threadid = omp_get_thread_num();
    return Ngblist + threadid * NumPart;
}
static int fof_nearest_isactive(int n) {
    return (((1 << P[n].Type) & (FOF_SECONDARY_LINK_TYPES)));
}
static void fof_nearest_reduce(int place, struct fofdata_out * O, int mode) {
    if(O->Distance < fof_nearest_distance[place])
    {
        fof_nearest_distance[place] = O->Distance;
        MinID[place] = O->MinID;
        MinIDTask[place] = O->MinIDTask;
    }
}
static int fof_nearest_evaluate(int target, int mode, 
        struct fofdata_in * I, struct fofdata_out * O,
        LocalEvaluator * lv, int *ngblist);
void fof_find_nearest_dmparticle(void)
{
    int i, n, iter;
    int64_t ntot;
    Evaluator ev = {0};
    ev.ev_label = "FOF_FIND_NEAREST";
    ev.ev_evaluate = (ev_ev_func) fof_nearest_evaluate;
    ev.ev_isactive = fof_nearest_isactive;
    ev.ev_alloc = fof_nearest_ngblist;
    ev.ev_copy = (ev_copy_func) fof_nearest_copy;
    ev.ev_reduce = (ev_reduce_func) fof_nearest_reduce;
    ev.UseNodeList = 1;
    ev.UseAllParticles = 1;
    ev.ev_datain_elsize = sizeof(struct fofdata_in);
    ev.ev_dataout_elsize = sizeof(struct fofdata_out);

    if(ThisTask == 0)
    {
        printf("Start finding nearest dm-particle (presently allocated=%g MB)\n",
                AllocatedBytes / (1024.0 * 1024.0));
        fflush(stdout);
    }

    fof_nearest_distance = (float *) mymalloc("fof_nearest_distance", sizeof(float) * NumPart);
    fof_nearest_hsml = (float *) mymalloc("fof_nearest_hsml", sizeof(float) * NumPart);

    for(n = 0; n < NumPart; n++)
    {
        if(((1 << P[n].Type) & (FOF_SECONDARY_LINK_TYPES)))
        {
            fof_nearest_distance[n] = 1.0e30;
            if(P[n].Type == 0) {
                /* use gas sml as a hint (faster convergence than 0.1 LinkL at high-z */
                fof_nearest_hsml[n] = 0.5 * P[n].Hsml;
            } else {
                fof_nearest_hsml[n] = 0.1 * LinkL;
            }
        }
    }

    /* allocate buffers to arrange communication */

    Ngblist = (int *) mymalloc("Ngblist", NumPart * sizeof(int) * All.NumThreads);

    iter = 0;
    /* we will repeat the whole thing for those particles where we didn't find enough neighbours */
    if(ThisTask == 0)
    {
        printf("fof-nearest iteration started\n");
        fflush(stdout);
    }

    do 
    {
        ev_begin(&ev);

        do
        {
            ev_primary(&ev);
            ev_get_remote(&ev, TAG_DENS_A);
            ev_secondary(&ev);
            ev_reduce_result(&ev, TAG_DENS_B);
        }
        while(ev_ndone(&ev) < NTask);
        ev_finish(&ev);

        int Nactive;
        int * queue = ev_get_queue(&ev, &Nactive);

        /* do final operations on results */
        int npleft = 0;
        int count = 0;
        int64_t counttot = 0;
/* CRAY cc doesn't do this one right */
//#pragma omp parallel for reduction(+: npleft)
        for(i = 0; i < Nactive; i++)
        {
            int p = queue[i];
            count ++;
            if(fof_nearest_distance[p] > 1.0e29)
            {
                if(fof_nearest_hsml[p] < 4 * LinkL)  /* we only search out to a maximum distance */
                {
                    /* need to redo this particle */
                    npleft++;
                    fof_nearest_hsml[p] *= 2.0;
/*
                    if(iter >= MAXITER - 10)
                    {
                        printf("i=%d task=%d ID=%llu Hsml=%g  pos=(%g|%g|%g)\n",
                                p, ThisTask, P[p].ID, fof_nearest_hsml[p],
                                P[p].Pos[0], P[p].Pos[1], P[p].Pos[2]);
                        fflush(stdout);
                    }
*/
                } else {
                    fof_nearest_distance[p] = 0;  /* we not continue to search for this particle */
                }
            }
        }
        sumup_large_ints(1, &npleft, &ntot);
        sumup_large_ints(1, &count, &counttot);
        if(ThisTask == 0)
        {
            printf("fof-nearest iteration %d: need to repeat for %010ld /%010ld particles.\n", iter, ntot, counttot);
            fflush(stdout);
        }
        if(ntot < 0) abort();
        if(ntot > 0)
        {
            iter++;
            if(iter > MAXITER)
            {
                printf("failed to converge in fof-nearest\n");
                fflush(stdout);
                endrun(1159);
            }
        }
        myfree(queue);
    }
    while(ntot > 0);

    myfree(Ngblist);

    myfree(fof_nearest_hsml);
    myfree(fof_nearest_distance);

    if(ThisTask == 0)
    {
        printf("done finding nearest dm-particle\n");
        fflush(stdout);
    }
}

static int fof_nearest_evaluate(int target, int mode, 
        struct fofdata_in * I, struct fofdata_out * O,
        LocalEvaluator * lv, int *ngblist)
{
    int j, n, index, listindex = 0;
    int startnode, numngb_inbox;
    double h, r2max;

    startnode = I->NodeList[0];
    listindex ++;
    startnode = Nodes[startnode].u.d.nextnode;	/* open it */

    index = -1;
    h = I->Hsml;
    r2max = 1.0e30;

    while(startnode >= 0)
    {
        while(startnode >= 0)
        {
            numngb_inbox = ngb_treefind_threads(I->Pos, h, target, &startnode, mode, 
                    lv, ngblist, NGB_TREEFIND_ASYMMETRIC, FOF_PRIMARY_LINK_TYPES);

            if(numngb_inbox < 0)
                return -1;

            for(n = 0; n < numngb_inbox; n++)
            {
                j = ngblist[n];
                double dx, dy, dz, r2;
                dx = I->Pos[0] - P[j].Pos[0];
                dy = I->Pos[1] - P[j].Pos[1];
                dz = I->Pos[2] - P[j].Pos[2];
#ifdef PERIODIC			/*  now find the closest image in the given box size  */
                dx = NEAREST_X(dx);
                dy = NEAREST_Y(dy);
                dz = NEAREST_Z(dz);
#endif
                r2 = dx * dx + dy * dy + dz * dz;
                if(r2 < r2max && r2 < h * h)
                {
                    index = j;
                    r2max = r2;
                }
            }
        }

        if(listindex < NODELISTLENGTH)
        {
            startnode = I->NodeList[listindex];
            if(startnode >= 0) {
                startnode = Nodes[startnode].u.d.nextnode;	/* open it */
                listindex ++;
            }
        }
    }


    if(index >= 0)
    {
        O->Distance = sqrt(r2max);
        O->MinID = MinID[Head[index]];
        O->MinIDTask = MinIDTask[Head[index]];
    }
    else {
        O->Distance = 2.0e30;
    }
    return 0;
}




#ifdef BLACK_HOLES

void fof_make_black_holes(void)
{
    int i, j, n, ntot;
    int nexport, nimport, recvTask, level;
    int *import_indices, *export_indices;
    double massDMpart;

    if(All.MassTable[1] > 0)
        massDMpart = All.MassTable[1];
    else {
        endrun(991234569); /* deprecate massDMpart in paramfile*/
    }

    for(n = 0; n < NTask; n++)
        Send_count[n] = 0;

    for(i = 0; i < Ngroups; i++)
    {
        if(Group[i].LenType[1] * massDMpart >=
                (All.Omega0 - All.OmegaBaryon) / All.Omega0 * All.MinFoFMassForNewSeed)
            if(Group[i].LenType[5] == 0)
            {
                if(Group[i].index_maxdens >= 0)
                    Send_count[Group[i].task_maxdens]++;
            }
    }

    MPI_Alltoall(Send_count, 1, MPI_INT, Recv_count, 1, MPI_INT, MPI_COMM_WORLD);

    for(j = 0, nimport = nexport = 0, Recv_offset[0] = 0, Send_offset[0] = 0; j < NTask; j++)
    {
        nexport += Send_count[j];
        nimport += Recv_count[j];

        if(j > 0)
        {
            Send_offset[j] = Send_offset[j - 1] + Send_count[j - 1];
            Recv_offset[j] = Recv_offset[j - 1] + Recv_count[j - 1];
        }
    }

    import_indices = mymalloc("import_indices", nimport * sizeof(int));
    export_indices = mymalloc("export_indices", nexport * sizeof(int));

    for(n = 0; n < NTask; n++)
        Send_count[n] = 0;

    for(i = 0; i < Ngroups; i++)
    {
        if(Group[i].LenType[1] * massDMpart >=
                (All.Omega0 - All.OmegaBaryon) / All.Omega0 * All.MinFoFMassForNewSeed)
            if(Group[i].LenType[5] == 0)
            {
                if(Group[i].index_maxdens >= 0)
                    export_indices[Send_offset[Group[i].task_maxdens] +
                        Send_count[Group[i].task_maxdens]++] = Group[i].index_maxdens;
            }
    }

    memcpy(&import_indices[Recv_offset[ThisTask]], &export_indices[Send_offset[ThisTask]],
            Send_count[ThisTask] * sizeof(int));

    for(level = 1; level < (1 << PTask); level++)
    {
        recvTask = ThisTask ^ level;

        if(recvTask < NTask) {
            if(Send_count[recvTask] > 0 || Recv_count[recvTask] > 0)  {
                MPI_Sendrecv(&export_indices[Send_offset[recvTask]],
                        Send_count[recvTask] * sizeof(int),
                        MPI_BYTE, recvTask, TAG_FOF_E,
                        &import_indices[Recv_offset[recvTask]],
                        Recv_count[recvTask] * sizeof(int),
                        MPI_BYTE, recvTask, TAG_FOF_E, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
    }

    MPI_Allreduce(&nimport, &ntot, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    if(ThisTask == 0)
    {
        printf("\nMaking %d new black hole particles\n\n", ntot);
        fflush(stdout);
    }

    for(n = 0; n < nimport; n++)
    {
        blackhole_make_one(import_indices[n]);
    }

    myfree(export_indices);
    myfree(import_indices);
}



#endif


double fof_periodic(double x)
{
    if(x >= 0.5 * All.BoxSize)
        x -= All.BoxSize;
    if(x < -0.5 * All.BoxSize)
        x += All.BoxSize;
    return x;
}


double fof_periodic_wrap(double x)
{
    while(x >= All.BoxSize)
        x -= All.BoxSize;
    while(x < 0)
        x += All.BoxSize;
    return x;
}

void fof_radix_FOF_PList_MinID(const void * a, void * radix, void * arg) {
    uint64_t * u = (uint64_t *) radix;
    struct fof_particle_list * f = (struct fof_particle_list *) a;
    u[0] = f->MinID;
}

int fof_compare_FOF_PList_MinID(const void *a, const void *b)
{
    if(((struct fof_particle_list *) a)->MinID < ((struct fof_particle_list *) b)->MinID)
        return -1;

    if(((struct fof_particle_list *) a)->MinID > ((struct fof_particle_list *) b)->MinID)
        return +1;

    return 0;
}

void fof_radix_FOF_GList_MinID(const void * a, void * radix, void * arg) {
    uint64_t * u = (uint64_t *) radix;
    struct fof_group_list * f = (struct fof_group_list *) a;
    u[0] = f->MinID;
}
int fof_compare_FOF_GList_MinID(const void *a, const void *b)

{
    if(((struct fof_group_list *) a)->MinID < ((struct fof_group_list *) b)->MinID)
        return -1;

    if(((struct fof_group_list *) a)->MinID > ((struct fof_group_list *) b)->MinID)
        return +1;

    return 0;
}

void fof_radix_FOF_GList_MinIDTask(const void * a, void * radix, void * arg) {
    uint64_t * u = (uint64_t *) radix;
    struct fof_group_list * f = (struct fof_group_list *) a;
    u[0] = f->MinIDTask;
}
int fof_compare_FOF_GList_MinIDTask(const void *a, const void *b)
{
    if(((struct fof_group_list *) a)->MinIDTask < ((struct fof_group_list *) b)->MinIDTask)
        return -1;

    if(((struct fof_group_list *) a)->MinIDTask > ((struct fof_group_list *) b)->MinIDTask)
        return +1;

    return 0;
}

static void fof_radix_FOF_GList_LocCountTaskDiffMinID(const void * a, void * radix, void * arg) {
    uint64_t * u = (uint64_t *) radix;
    struct fof_group_list * f = (struct fof_group_list *) a;
    u[0] = labs(f->ExtCount - f->MinIDTask);
    u[1] = f->MinID;
    u[2] = UINT64_MAX - f->LocCount;
}

int fof_compare_FOF_GList_LocCountTaskDiffMinID(const void *a, const void *b)
{
    if(((struct fof_group_list *) a)->LocCount > ((struct fof_group_list *) b)->LocCount)
        return -1;

    if(((struct fof_group_list *) a)->LocCount < ((struct fof_group_list *) b)->LocCount)
        return +1;

    if(((struct fof_group_list *) a)->MinID < ((struct fof_group_list *) b)->MinID)
        return -1;

    if(((struct fof_group_list *) a)->MinID > ((struct fof_group_list *) b)->MinID)
        return +1;

    if(labs(((struct fof_group_list *) a)->ExtCount - ((struct fof_group_list *) a)->MinIDTask) <
            labs(((struct fof_group_list *) b)->ExtCount - ((struct fof_group_list *) b)->MinIDTask))
        return -1;

    if(labs(((struct fof_group_list *) a)->ExtCount - ((struct fof_group_list *) a)->MinIDTask) >
            labs(((struct fof_group_list *) b)->ExtCount - ((struct fof_group_list *) b)->MinIDTask))
        return +1;

    return 0;
}

static void fof_radix_FOF_GList_ExtCountMinID(const void * a, void * radix, void * arg) {
    uint64_t * u = (uint64_t *) radix;
    struct fof_group_list * f = (struct fof_group_list *) a;
    u[0] = f->MinID;
    u[1] = f->ExtCount;
}

int fof_compare_FOF_GList_ExtCountMinID(const void *a, const void *b)
{
    if(((struct fof_group_list *) a)->ExtCount < ((struct fof_group_list *) b)->ExtCount)
        return -1;

    if(((struct fof_group_list *) a)->ExtCount > ((struct fof_group_list *) b)->ExtCount)
        return +1;

    if(((struct fof_group_list *) a)->MinID < ((struct fof_group_list *) b)->MinID)
        return -1;

    if(((struct fof_group_list *) a)->MinID > ((struct fof_group_list *) b)->MinID)
        return +1;

    return 0;
}

void fof_radix_Group_MinID(const void * a, void * radix, void * arg) {
    uint64_t * u = (uint64_t *) radix;
    struct group_properties * f = (struct group_properties*) a;
    u[0] = f->MinID;
}


int fof_compare_Group_MinID(const void *a, const void *b)
{
    if(((struct group_properties *) a)->MinID < ((struct group_properties *) b)->MinID)
        return -1;

    if(((struct group_properties *) a)->MinID > ((struct group_properties *) b)->MinID)
        return +1;

    return 0;
}

static void fof_radix_Group_GrNr(const void * a, void * radix, void * arg) {
    uint64_t * u = (uint64_t *) radix;
    struct group_properties * f = (struct group_properties*) a;
    u[0] = f->GrNr;
}


int fof_compare_Group_GrNr(const void *a, const void *b)
{
    if(((struct group_properties *) a)->GrNr < ((struct group_properties *) b)->GrNr)
        return -1;

    if(((struct group_properties *) a)->GrNr > ((struct group_properties *) b)->GrNr)
        return +1;

    return 0;
}

void fof_radix_Group_MinIDTask(const void * a, void * radix, void * arg) {
    uint64_t * u = (uint64_t *) radix;
    struct group_properties * f = (struct group_properties *) a;
    u[0] = f->MinIDTask;
}

int fof_compare_Group_MinIDTask(const void *a, const void *b)
{
    if(((struct group_properties *) a)->MinIDTask < ((struct group_properties *) b)->MinIDTask)
        return -1;

    if(((struct group_properties *) a)->MinIDTask > ((struct group_properties *) b)->MinIDTask)
        return +1;

    return 0;
}

void fof_radix_Group_MinIDTask_MinID(const void * a, void * radix, void * arg) {
    uint64_t * u = (uint64_t *) radix;
    struct group_properties * f = (struct group_properties *) a;
    u[0] = f->MinID;
    u[1] = f->MinIDTask;
}

int fof_compare_Group_MinIDTask_MinID(const void *a, const void *b)
{
    if(((struct group_properties *) a)->MinIDTask < ((struct group_properties *) b)->MinIDTask)
        return -1;

    if(((struct group_properties *) a)->MinIDTask > ((struct group_properties *) b)->MinIDTask)
        return +1;

    if(((struct group_properties *) a)->MinID < ((struct group_properties *) b)->MinID)
        return -1;

    if(((struct group_properties *) a)->MinID > ((struct group_properties *) b)->MinID)
        return +1;

    return 0;
}

void fof_radix_Group_Len(const void * a, void * radix, void * arg) {
    uint64_t * u = (uint64_t *) radix;
    struct group_properties * f = (struct group_properties *) a;
    u[0] = UINT64_MAX - f->Len;
}

int fof_compare_Group_Len(const void *a, const void *b)
{
    if(((struct group_properties *) a)->Len > ((struct group_properties *) b)->Len)
        return -1;

    if(((struct group_properties *) a)->Len < ((struct group_properties *) b)->Len)
        return +1;

    return 0;
}


/* unused */
int fof_compare_ID_list_GrNrID(const void *a, const void *b)
{
    if(((struct id_list *) a)->GrNr < ((struct id_list *) b)->GrNr)
        return -1;

    if(((struct id_list *) a)->GrNr > ((struct id_list *) b)->GrNr)
        return +1;

    if(((struct id_list *) a)->ID < ((struct id_list *) b)->ID)
        return -1;

    if(((struct id_list *) a)->ID > ((struct id_list *) b)->ID)
        return +1;

    return 0;
}

#endif /* of FOF */
