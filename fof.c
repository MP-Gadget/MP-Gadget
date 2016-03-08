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
#include "forcetree.h"
#include "domain.h"
#include "mpsort.h"
#include "mymalloc.h"

/*! \file fof.c
 *  \brief parallel FoF group finder
 */

#ifdef FOF
#include "fof.h"

static double fof_periodic(double x)
{
    if(x >= 0.5 * All.BoxSize)
        x -= All.BoxSize;
    if(x < -0.5 * All.BoxSize)
        x += All.BoxSize;
    return x;
}


static double fof_periodic_wrap(double x)
{
    while(x >= All.BoxSize)
        x -= All.BoxSize;
    while(x < 0)
        x += All.BoxSize;
    return x;
}

static void fof_label_secondary(void);
static int fof_compare_HaloLabel_MinID(const void *a, const void *b);
static int fof_compare_Group_MinIDTask(const void *a, const void *b);
static int fof_compare_Group_MinID(const void *a, const void *b);

static void fof_finish_group_properties(void);
static void fof_compile_catalogue(void);
static void fof_assign_grnr();

void fof_label_primary(void);
extern void fof_save_particles(int num);
extern void fof_save_groups(int num);

static void fof_make_black_holes(void);

uint64_t Ngroups, TotNgroups;
int64_t TotNids;

struct group_properties *Group;

struct fofdata_in
{
    int NodeList[NODELISTLENGTH];
    MyDouble Pos[3];
    MyFloat Hsml;
    MyIDType MinID;
    MyIDType MinIDTask;
};

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
*HaloLabel;

static double LinkL;
static int NgroupsExt;

static float *fof_secondary_distance;
static float *fof_secondary_hsml;


void fof_fof(int num)
{
    int i, ndm, start, lenloc, n;
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

    for(i = 0, ndm = 0, mass = 0; i < NumPart; i++) {
        if(((1 << P[i].Type) & (FOF_PRIMARY_LINK_TYPES)))
        {
            ndm++;
            mass += P[i].Mass;
        }
    }
    sumup_large_ints(1, &ndm, &ndmtot);
    MPI_Allreduce(&mass, &masstot, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    rhodm = (All.Omega0 - All.OmegaBaryon) * 3 * All.Hubble * All.Hubble / (8 * M_PI * All.G);

    LinkL = All.FOFHaloLinkingLength * pow(masstot / ndmtot / rhodm, 1.0 / 3);

    if(ThisTask == 0)
    {
        printf("\nComoving linking length: %g    ", LinkL);
        printf("(presently allocated=%g MB)\n", AllocatedBytes / (1024.0 * 1024.0));
        fflush(stdout);
    }

    HaloLabel = (struct fof_particle_list *) mymalloc("HaloLabel", NumPart * sizeof(struct fof_particle_list));

    /* HaloLabel stores the MinID and MinIDTask of particles, this pair serves as a halo label. */
    for(i = 0; i < NumPart; i++) {
        HaloLabel[i].Pindex = i;
    }

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


    /* Fill FOFP_List of primary */
    t0 = second();
    fof_label_primary();
    t1 = second();

    if(ThisTask == 0)
        printf("group finding took = %g sec\n", timediff(t0, t1));
    walltime_measure("/FOF/Primary");

    /* Fill FOFP_List of secondary */
    t0 = second();
    fof_label_secondary();
    walltime_measure("/FOF/Secondary");
    t1 = second();

    if(ThisTask == 0)
        printf("attaching gas and star particles to nearest dm particles took = %g sec\n", timediff(t0, t1));


    force_treefree();

    t0 = second();
    fof_compile_catalogue();
    t1 = second();
    if(ThisTask == 0)
        printf("compiling local group data and catalogue took = %g sec\n", timediff(t0, t1));

    walltime_measure("/FOF/Compile");

    t0 = second();

    if(ThisTask == 0)
    {
        printf("group properties are now allocated.. (presently allocated=%g MB)\n",
                AllocatedBytes / (1024.0 * 1024.0));
        fflush(stdout);
    }

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

    myfree(HaloLabel);

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

static struct LinkList {
    MyIDType MinIDOld;
    int head;
    int len;
    int next;
    char marked;
} * LinkList;
#define HEAD(i) LinkList[i].head
#define NEXT(i) LinkList[i].next
#define LEN(i) LinkList[HEAD(i)].len

static void fof_primary_copy(int place, struct fofdata_in * I) {
    I->Pos[0] = P[place].Pos[0];
    I->Pos[1] = P[place].Pos[1];
    I->Pos[2] = P[place].Pos[2];
    int head = HEAD(place);
    I->MinID = HaloLabel[head].MinID;
    I->MinIDTask = HaloLabel[head].MinIDTask;
}

static int fof_primary_isactive(int n) {
    return (((1 << P[n].Type) & (FOF_PRIMARY_LINK_TYPES))) && LinkList[n].marked;
}

static int fof_primary_evaluate(int target, int mode, 
        struct fofdata_in * I, struct fofdata_out * O,
        LocalEvaluator * lv);

void fof_label_primary(void)
{
    int i;
    int64_t link_across;
    int64_t link_across_tot;
    double t0, t1;

    if(ThisTask == 0)
    {
        printf("\nStart linking particles (presently allocated=%g MB)\n", AllocatedBytes / (1024.0 * 1024.0));
        fflush(stdout);
    }

    Evaluator ev = {0};
    ev.ev_label = "FOF_FIND_GROUPS";
    ev.ev_evaluate = (ev_ev_func) fof_primary_evaluate;
    ev.ev_isactive = fof_primary_isactive;
    ev.ev_copy = (ev_copy_func) fof_primary_copy;
    ev.ev_reduce = NULL;
    ev.UseNodeList = 1;
    ev.UseAllParticles = 1;
    ev.ev_datain_elsize = sizeof(struct fofdata_in);
    ev.ev_dataout_elsize = 1;

    LinkList = (struct LinkList *) mymalloc("FOF_Links", NumPart * sizeof(struct LinkList));
    /* allocate buffers to arrange communication */

    t0 = second();

    for(i = 0; i < NumPart; i++)
    {
        HEAD(i) = i;
        LEN(i) = 1;
        NEXT(i) = -1;
        LinkList[i].MinIDOld = P[i].ID;
        LinkList[i].marked = 1;

        HaloLabel[i].MinID = P[i].ID;
        HaloLabel[i].MinIDTask = ThisTask;
    }

    do
    {
        t0 = second();

        ev_run(&ev);

        t1 = second();

        /* let's check out which particles have changed their MinID,
         * mark them for next round. */
        link_across = 0;
#pragma omp parallel for
        for(i = 0; i < NumPart; i++) {
            MyIDType newMinID = HaloLabel[HEAD(i)].MinID;
            if(newMinID != LinkList[i].MinIDOld) {
                LinkList[i].marked = 1;
#pragma omp atomic
                link_across += 1;
            } else {
                LinkList[i].marked = 0;
            }
            LinkList[i].MinIDOld = newMinID;
        }
        MPI_Allreduce(&link_across, &link_across_tot, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
        if(ThisTask == 0)
        {
            printf("Linked %ld particles %g seconds\n", link_across_tot, t1 - t0);
        }
    }
    while(link_across_tot > 0);

    /* Update MinID of all linked (primary-linked) particles */
    for(i = 0; i < NumPart; i++)
    {
        HaloLabel[i].MinID = HaloLabel[HEAD(i)].MinID;
        HaloLabel[i].MinIDTask = HaloLabel[HEAD(i)].MinIDTask;
    }

    if(ThisTask == 0)
    {
        printf("Local groups found.\n\n");
        fflush(stdout);
    }
    myfree(LinkList);
}

static void fofp_merge(int target, int j)
{
    /* must be in a critical section! */
    if(HEAD(target) == HEAD(j))	
        return;
    int p, s;
    int ss, oldnext, last;

    /* only if not yet linked */
    if(LEN(target) > LEN(j))	/* p group is longer */
    {
        p = target; s = j;
    } else {
        p = j; s = target;
    }

    ss = HEAD(s);
    oldnext = NEXT(p);
    NEXT(p) = ss;
    last = -1;
    do {
        HEAD(ss) = HEAD(p);
        last = ss;
    } while((ss = NEXT(ss)) >= 0);

    NEXT(last) = oldnext;

    if(HaloLabel[HEAD(s)].MinID < HaloLabel[HEAD(p)].MinID)
    {
        HaloLabel[HEAD(p)].MinID = HaloLabel[HEAD(s)].MinID;
        HaloLabel[HEAD(p)].MinIDTask = HaloLabel[HEAD(s)].MinIDTask;
    }
}

static int fof_primary_evaluate(int target, int mode, 
        struct fofdata_in * I, struct fofdata_out * O,
        LocalEvaluator * lv) {
    int listindex = 0;
    int startnode, numngb_inbox;
    

    startnode = I->NodeList[0];
    listindex ++;
    startnode = Nodes[startnode].u.d.nextnode;	/* open it */

    while(startnode >= 0)
    {
        while(startnode >= 0)
        {
            numngb_inbox = ngb_treefind_threads(I->Pos, LinkL, target, &startnode, mode, 
                    lv, NGB_TREEFIND_ASYMMETRIC, FOF_PRIMARY_LINK_TYPES);

            if(numngb_inbox < 0)
                return -1;
            int n;
            for(n = 0; n < numngb_inbox; n++)
            {
                int j = lv->ngblist[n];
                if(mode == 0) {
                    /* Local FOF */
                    if(HEAD(target) != HEAD(j)) {
#pragma omp critical
                        fofp_merge(target, j);
                    }
                } else		/* mode is 1, target is a ghost */
                {
#pragma omp critical
                    if(HaloLabel[HEAD(j)].MinID > I->MinID)
                    {
                        HaloLabel[HEAD(j)].MinID = I->MinID;
                        HaloLabel[HEAD(j)].MinIDTask = I->MinIDTask;
                    }
                }
            }
        }

        if(listindex < NODELISTLENGTH)
        {
            startnode = I->NodeList[listindex];
            if(startnode >= 0) {
                startnode = Nodes[startnode].u.d.nextnode;	/* open it */
                listindex++;
            }
        }
    }

    return 0;
}

static void add_group_to_group(struct group_properties * gdst, struct group_properties * gsrc) {
    int j;
    double xyz[3];
    gdst->Length += gsrc->Length;
    gdst->Mass += gsrc->Mass;

    for(j = 0; j < 6; j++)
    {
        gdst->LenType[j] += gsrc->LenType[j];
        gdst->MassType[j] += gsrc->MassType[j];
    }

#ifdef SFR
    gdst->Sfr += gsrc->Sfr;
#endif
#ifdef BLACK_HOLES
    gdst->BH_Mdot += gsrc->BH_Mdot;
    gdst->BH_Mass += gsrc->BH_Mass;
    if(gsrc->MaxDens > gdst->MaxDens)
    {
        gdst->MaxDens = gsrc->MaxDens;
        gdst->index_maxdens = gsrc->index_maxdens;
        gdst->task_maxdens = gsrc->task_maxdens;
    }
#endif

    for(j = 0; j < 3; j++)
    {
        xyz[j] = gsrc->CM[j] / gsrc->Mass + gsrc->FirstPos[j];

        xyz[j] = fof_periodic(xyz[j] - gdst->FirstPos[j]);

        gdst->CM[j] += gsrc->Mass * xyz[j];
        gdst->Vel[j] += gsrc->Vel[j];
    }

}
static void add_particle_to_group(struct group_properties * gdst, int i) {

    /* My local number of particles contributing to the full catalogue. */
    int j, k;
    double xyz[3];

    int index = HaloLabel[i].Pindex;

    if(gdst->Length == 0) {
        gdst->Mass = 0;
    #ifdef SFR
        gdst->Sfr = 0;
    #endif
    #ifdef BLACK_HOLES
        gdst->BH_Mass = 0;
        gdst->BH_Mdot = 0;
        gdst->index_maxdens = gdst->task_maxdens = -1;
        gdst->MaxDens = 0;
    #endif

        for(k = 0; k < 3; k++)
        {
            gdst->CM[k] = 0;
            gdst->Vel[k] = 0;
            gdst->FirstPos[k] = P[index].Pos[k];
        }

        for(k = 0; k < 6; k++)
        {
            gdst->LenType[k] = 0;
            gdst->MassType[k] = 0;
        }
    }

    gdst->Length ++;
    gdst->Mass += P[index].Mass;
    gdst->LenType[P[index].Type]++;
    gdst->MassType[P[index].Type] += P[index].Mass;


#ifdef SFR
    if(P[index].Type == 0) {
        gdst->Sfr += get_starformation_rate(index);
    }
#endif
#ifdef BLACK_HOLES
    if(P[index].Type == 5)
    {
        gdst->BH_Mdot += BHP(index).Mdot;
        gdst->BH_Mass += BHP(index).Mass;
    }
    if(P[index].Type == 0)
    {
#ifdef WINDS
        /* make bh in non wind gas on bh wind*/
        if(SPHP(index).DelayTime <= 0)
#endif
            if(SPHP(index).Density > gdst->MaxDens)
            {
                gdst->MaxDens = SPHP(index).Density;
                gdst->index_maxdens = index;
                gdst->task_maxdens = ThisTask;
            }
    }
#endif

    for(j = 0; j < 3; j++)
    {
        xyz[j] = P[index].Pos[j];

        xyz[j] = fof_periodic(xyz[j] - gdst->FirstPos[j]);

        gdst->CM[j] += P[index].Mass * xyz[j];
        gdst->Vel[j] += P[index].Mass * P[index].Vel[j];
    }
}

static void fof_compile_catalogue(void)
{
    int i, j, start;
    struct group_properties * GhostGroup;

    /* sort according to MinID */
    qsort(HaloLabel, NumPart, sizeof(struct fof_particle_list), fof_compare_HaloLabel_MinID);

    NgroupsExt = 0;
    for(i = 0; i < NumPart; i ++) {
        if(i == 0 || HaloLabel[i].MinID != HaloLabel[i - 1].MinID) NgroupsExt ++;
    }

    printf("NgroupsExt = %d NumPart = %d\n", NgroupsExt, NumPart);

    Group = (struct group_properties *) mymalloc("Group", sizeof(struct group_properties) * NgroupsExt);

    memset(Group, 0, sizeof(Group[0]) * NgroupsExt);

    int next = 0;
    int item = -1;

    for(i = 0; i < NumPart; i++)
    {
        if(i == 0 || HaloLabel[i].MinID != HaloLabel[i - 1].MinID) {
            item = next;
            Group[item].MinID = HaloLabel[i].MinID;
            Group[item].MinIDTask = HaloLabel[i].MinIDTask;
            next ++;
        }
        add_particle_to_group(&Group[item], i);
    }
    if(next != NgroupsExt) abort();

    /* local groups will be moved to the beginning, we skip them with offset */
    qsort(Group, NgroupsExt, sizeof(struct group_properties), fof_compare_Group_MinIDTask);

    int * Send_count = alloca(sizeof(int) * NTask);
    int * Recv_count = alloca(sizeof(int) * NTask);

    /* count how many we have of each task */
    memset(Send_count, 0, sizeof(int) * NTask);
    for(i = 0; i < NgroupsExt; i++)
        Send_count[Group[i].MinIDTask]++;

    /* Skip local groups, receive ghosts from others */
    int Nlocal = Send_count[ThisTask];
    Send_count[ThisTask] = 0;
     
    MPI_Alltoall(Send_count, 1, MPI_INT, Recv_count, 1, MPI_INT, MPI_COMM_WORLD);

    int nimport = 0;
    for(i = 0; i < NTask; i ++) {
        nimport += Recv_count[i];
    }
    GhostGroup = (struct group_properties *) 
            mymalloc("GhostGroup", nimport * sizeof(struct group_properties));

    MPI_Datatype MPI_TYPE;
    MPI_Type_contiguous(sizeof(Group[0]), MPI_BYTE, &MPI_TYPE);
    MPI_Type_commit(&MPI_TYPE);

    MPI_Alltoallv_smart(&Group[Nlocal], Send_count, NULL, MPI_TYPE, 
                        GhostGroup, Recv_count, NULL, MPI_TYPE, MPI_COMM_WORLD);
        
    /* record the original ordering of GhostGroup with a MinIDTask sort */
    for(i = 0; i < nimport; i++)
        GhostGroup[i].MinIDTask = i;

    /* sort the groups according to MinID */
    qsort(Group, Nlocal, sizeof(struct group_properties), fof_compare_Group_MinID);
    qsort(GhostGroup, nimport, sizeof(struct group_properties), fof_compare_Group_MinID);

    /* merge the imported ones with the local ones */
    for(i = 0, start = 0; i < nimport; i++)
    {
        while(Group[start].MinID < GhostGroup[i].MinID)
        {
            start++;
            if(start >= Nlocal)
                endrun(7973);
        }
        add_group_to_group(&Group[start], &GhostGroup[i]); 
    }

    /* copy the group attributes back into the list, to inform the others */
    for(i = 0, start = 0; i < nimport; i++)
    {
        while(Group[start].MinID < GhostGroup[i].MinID)
            start++;

        int oldMinIDTask = GhostGroup[i].MinIDTask;
        GhostGroup[i] = Group[start];
        GhostGroup[i].MinIDTask = oldMinIDTask;
    }

    /* reset the ordering of imported list, such that it can be properly returned */
    qsort(GhostGroup, nimport, sizeof(struct group_properties), fof_compare_Group_MinIDTask);

    for(i = 0; i < nimport; i++)
        GhostGroup[i].MinIDTask = ThisTask;

    MPI_Alltoallv_smart(GhostGroup, Recv_count, NULL, MPI_TYPE, 
                        &Group[Nlocal], Send_count, NULL, MPI_TYPE, MPI_COMM_WORLD);

    myfree(GhostGroup);

    /* At this point, each Group entry has the reduced attribute of the full group */
    /* Now it's time to apply FOFHaloMinLength cut */

    /* sort the group list according to MinID 
     * FIXME: this is likely not needed assign_grnr will redo the sorting */
    qsort(Group, NgroupsExt, sizeof(struct group_properties), fof_compare_Group_MinID);

    /* eliminate all groups that are too small */
    for(i = 0; i < NgroupsExt; i++)
    {
        if(Group[i].Length < All.FOFHaloMinLength)
        {
            Group[i] = Group[NgroupsExt - 1];
            NgroupsExt--;
            i--;
        }
    }

    /* count Groups and number of particles hosted by me */
    Ngroups = 0;
    int64_t Nids = 0;
    for(i = 0; i < NgroupsExt; i ++) {
        if(Group[i].MinIDTask == ThisTask)
        {
            Ngroups++;
            Nids += Group[i].Length;
        }
    }

    /* Assign grnr for groups and particles */
    fof_assign_grnr();

    MPI_Allreduce(&Ngroups, &TotNgroups, 1, MPI_UINT64, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&Nids, &TotNids, 1, MPI_INT64, MPI_SUM, MPI_COMM_WORLD);

    /* report some statictics */
    int largestgroup;
    if(TotNgroups > 0)
    {
        int largestloc = 0;

        for(i = 0; i < NgroupsExt; i++)
            if(Group[i].Length > largestloc)
                largestloc = Group[i].Length;
        MPI_Allreduce(&largestloc, &largestgroup, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    }
    else
        largestgroup = 0;

    if(ThisTask == 0)
    {
        printf("\nTotal number of groups with at least %d particles: %ld\n", All.FOFHaloMinLength, TotNgroups);
        if(TotNgroups > 0)
        {
            printf("Largest group has %d particles.\n", largestgroup);
            printf("Total number of particles in groups: %d%09d\n\n",
                    (int) (TotNids / 1000000000), (int) (TotNids % 1000000000));
        }
    }

    MPI_Type_free(&MPI_TYPE);

}

static void fof_finish_group_properties(void)
{
    double cm[3];
    int i, j, ngr;

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


    for(i = 0; i < ngr; i++)
    {
        for(j = 0; j < 3; j++)
        {
            Group[i].Vel[j] /= Group[i].Mass;

            cm[j] = Group[i].CM[j] / Group[i].Mass;

            cm[j] = fof_periodic_wrap(cm[j] + Group[i].FirstPos[j]);

            Group[i].CM[j] = cm[j];
        }
    }

    if(ngr != Ngroups)
        endrun(876889);

    qsort(Group, Ngroups, sizeof(struct group_properties), fof_compare_Group_MinID);
}



static void fof_radix_Group_TotalCountTaskDiffMinID(const void * a, void * radix, void * arg);
static void fof_radix_Group_OriginalTaskMinID(const void * a, void * radix, void * arg);

static void fof_assign_grnr() 
{
    int i, j;
    int ngr;

    for(i = 0; i < NgroupsExt; i++)
    {
        Group[i].OriginalTask = ThisTask;	/* original task */
    }

    mpsort_mpi(Group, NgroupsExt, sizeof(struct group_properties),
            fof_radix_Group_TotalCountTaskDiffMinID, 24, NULL, MPI_COMM_WORLD);

    /* assign group numbers 
     * at this point, both Group are is sorted by length,
     * and the every time OriginalTask == MinIDTask, a list of ghost groups is stored.
     * they shall get the same GrNr.
     * */
    ngr = 0;
    for(i = 0; i < NgroupsExt; i++)
    {
        if(Group[i].OriginalTask == Group[i].MinIDTask) {
            ngr++;
        }
        Group[i].GrNr = ngr;
    }

    int64_t * ngra;
    ngra = alloca(sizeof(ngra[0]) * NTask);
    MPI_Allgather(&ngr, 1, MPI_INT64, ngra, 1, MPI_INT64, MPI_COMM_WORLD);

    /* shift to the global grnr. */
    int64_t groffset = 0;
    for(j = 0; j < ThisTask; j++)
        groffset += ngra[j];
    for(i = 0; i < NgroupsExt; i++)
        Group[i].GrNr += groffset;

    /* bring the group list back into the original order */
    mpsort_mpi(Group, NgroupsExt, sizeof(struct group_properties), 
            fof_radix_Group_OriginalTaskMinID, 16, NULL, MPI_COMM_WORLD);

    for(i = 0; i < NumPart; i++)
        P[i].GrNr = -1;	/* will mark particles that are not in any group */

    int start;
    for(i = 0, start = 0; i < NgroupsExt; i++)
    {
        while(HaloLabel[start].MinID < Group[i].MinID)
        {
            start++;
            if(start > NumPart)
                endrun(78);
        }

        if(HaloLabel[start].MinID != Group[i].MinID)
            endrun(1313);
        int lenloc; 
        /* FIXME: This is twisted Volker-C rewrite it in plain C */
        for(lenloc = 0; start + lenloc < NumPart;)
            if(HaloLabel[start + lenloc].MinID == Group[i].MinID)
            {
                P[HaloLabel[start + lenloc].Pindex].GrNr = Group[i].GrNr;
                lenloc++;
            }
            else
                break;

        start += lenloc;
    }
}


void fof_save_groups(int num)
{
    double t0, t1;
    if(ThisTask == 0)
    {
        printf("start global sorting of group catalogues\n");
        fflush(stdout);
    }

    t0 = second();

    fof_save_particles(num);

    t1 = second();

    if(ThisTask == 0)
    {
        printf("Group catalogues saved. took = %g sec\n", timediff(t0, t1));
        fflush(stdout);
    }
}

static void fof_secondary_copy(int place, struct fofdata_in * I) {
    int k;
    for (k = 0; k < 3; k ++) {
        I->Pos[k] = P[place].Pos[k];
    }
    I->Hsml = fof_secondary_hsml[place];
}
static int fof_secondary_isactive(int n) {
    return (((1 << P[n].Type) & (FOF_SECONDARY_LINK_TYPES)));
}
static void fof_secondary_reduce(int place, struct fofdata_out * O, int mode) {
    if(O->Distance < fof_secondary_distance[place])
    {
        fof_secondary_distance[place] = O->Distance;
        HaloLabel[place].MinID = O->MinID;
        HaloLabel[place].MinIDTask = O->MinIDTask;
    }
}
static int fof_secondary_evaluate(int target, int mode, 
        struct fofdata_in * I, struct fofdata_out * O,
        LocalEvaluator * lv);

static void fof_label_secondary(void)
{
    int i, n, iter;
    int64_t ntot;
    Evaluator ev = {0};
    ev.ev_label = "FOF_FIND_NEAREST";
    ev.ev_evaluate = (ev_ev_func) fof_secondary_evaluate;
    ev.ev_isactive = fof_secondary_isactive;
    ev.ev_copy = (ev_copy_func) fof_secondary_copy;
    ev.ev_reduce = (ev_reduce_func) fof_secondary_reduce;
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

    fof_secondary_distance = (float *) mymalloc("fof_secondary_distance", sizeof(float) * NumPart);
    fof_secondary_hsml = (float *) mymalloc("fof_secondary_hsml", sizeof(float) * NumPart);

    for(n = 0; n < NumPart; n++)
    {
        if(((1 << P[n].Type) & (FOF_SECONDARY_LINK_TYPES)))
        {
            fof_secondary_distance[n] = 1.0e30;
            if(P[n].Type == 0) {
                /* use gas sml as a hint (faster convergence than 0.1 LinkL at high-z */
                fof_secondary_hsml[n] = 0.5 * P[n].Hsml;
            } else {
                fof_secondary_hsml[n] = 0.1 * LinkL;
            }
        }
    }

    /* allocate buffers to arrange communication */

    iter = 0;
    /* we will repeat the whole thing for those particles where we didn't find enough neighbours */
    if(ThisTask == 0)
    {
        printf("fof-nearest iteration started\n");
        fflush(stdout);
    }

    do 
    {
        ev_run(&ev);

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
            if(fof_secondary_distance[p] > 1.0e29)
            {
                if(fof_secondary_hsml[p] < 4 * LinkL)  /* we only search out to a maximum distance */
                {
                    /* need to redo this particle */
                    npleft++;
                    fof_secondary_hsml[p] *= 2.0;
/*
                    if(iter >= MAXITER - 10)
                    {
                        printf("i=%d task=%d ID=%llu Hsml=%g  pos=(%g|%g|%g)\n",
                                p, ThisTask, P[p].ID, fof_secondary_hsml[p],
                                P[p].Pos[0], P[p].Pos[1], P[p].Pos[2]);
                        fflush(stdout);
                    }
*/
                } else {
                    fof_secondary_distance[p] = 0;  /* we not continue to search for this particle */
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

    myfree(fof_secondary_hsml);
    myfree(fof_secondary_distance);

    if(ThisTask == 0)
    {
        printf("done finding nearest dm-particle\n");
        fflush(stdout);
    }
}

static int fof_secondary_evaluate(int target, int mode, 
        struct fofdata_in * I, struct fofdata_out * O,
        LocalEvaluator * lv)
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
                    lv, NGB_TREEFIND_ASYMMETRIC, FOF_PRIMARY_LINK_TYPES);

            if(numngb_inbox < 0)
                return -1;

            for(n = 0; n < numngb_inbox; n++)
            {
                j = lv->ngblist[n];
                double dx, dy, dz, r2;
                dx = I->Pos[0] - P[j].Pos[0];
                dy = I->Pos[1] - P[j].Pos[1];
                dz = I->Pos[2] - P[j].Pos[2];

                dx = NEAREST(dx);
                dy = NEAREST(dy);
                dz = NEAREST(dz);

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
        O->MinID = HaloLabel[index].MinID;
        O->MinIDTask = HaloLabel[index].MinIDTask;
    }
    else {
        O->Distance = 2.0e30;
    }
    return 0;
}




#ifdef BLACK_HOLES

static void fof_make_black_holes(void)
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

static int fof_compare_HaloLabel_MinID(const void *a, const void *b)
{
    if(((struct fof_particle_list *) a)->MinID < ((struct fof_particle_list *) b)->MinID)
        return -1;

    if(((struct fof_particle_list *) a)->MinID > ((struct fof_particle_list *) b)->MinID)
        return +1;

    return 0;
}

static int fof_compare_Group_MinID(const void *a, const void *b)

{
    if(((struct group_properties *) a)->MinID < ((struct group_properties *) b)->MinID)
        return -1;

    if(((struct group_properties *) a)->MinID > ((struct group_properties *) b)->MinID)
        return +1;

    return 0;
}

static int fof_compare_Group_MinIDTask(const void *a, const void *b)
{
    const struct group_properties * p1 = a;
    const struct group_properties * p2 = b;
    int t1 = p1->MinIDTask;
    int t2 = p2->MinIDTask;
    if(t1 == ThisTask) t1 = -1;
    if(t2 == ThisTask) t2 = -1;

    if(t1 < t2) return -1;
    if(t1 > t2) return +1;
    return 0;

}

static void fof_radix_Group_TotalCountTaskDiffMinID(const void * a, void * radix, void * arg) {
    uint64_t * u = (uint64_t *) radix;
    struct group_properties * f = (struct group_properties *) a;
    u[0] = labs(f->OriginalTask - f->MinIDTask);
    u[1] = f->MinID;
    u[2] = UINT64_MAX - (f->Length);
}

static void fof_radix_Group_OriginalTaskMinID(const void * a, void * radix, void * arg) {
    uint64_t * u = (uint64_t *) radix;
    struct group_properties * f = (struct group_properties *) a;
    u[0] = f->MinID;
    u[1] = f->OriginalTask;
}

#endif /* of FOF */
