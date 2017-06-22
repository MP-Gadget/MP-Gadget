#include "garbage.h"
#include "allvars.h"
#include "timestep.h"
#include "system.h"
#include "endrun.h"
#include "forcetree.h"
/*For domain_count_particles*/
#include "exchange.h"

static int
domain_all_garbage_collection();
static int
domain_sph_garbage_collection_reclaim();
static int
domain_bh_garbage_collection();

int domain_fork_particle(int parent) {
    /* this will fork a zero mass particle at the given location of parent.
     *
     * Assumes the particle is protected by locks in threaded env.
     *
     * The Generation of parent is incremented.
     * The child carries the incremented generation number.
     * The ID of the child is modified, with the new generation number set
     * at the highest 8 bits.
     *
     * the new particle's index is returned.
     *
     * Its mass and ptype can be then adjusted. (watchout detached BH /SPH
     * data!)
     * Its PIndex still points to the old Pindex!
     * */

    if(NumPart >= All.MaxPart)
    {
        endrun(8888,
                "On Task=%d with NumPart=%d we try to spawn. Sorry, no space left...(All.MaxPart=%d)\n",
                ThisTask, NumPart, All.MaxPart);
    }
    /*This is all racy if ActiveParticle or P is accessed from another thread*/
    int child = atomic_fetch_and_add(&NumPart, 1);
    /*Update the active particle list*/
    int childactive = atomic_fetch_and_add(&NumActiveParticle, 1);
    ActiveParticle[childactive] = child;

    P[parent].Generation ++;
    uint64_t g = P[parent].Generation;
    /* change the child ID according to the generation. */
    P[child] = P[parent];
    P[child].ID = (P[parent].ID & 0x00ffffffffffffffL) + (g << 56L);

    /* the PIndex still points to the old PIndex */
    P[child].Mass = 0;

    /* FIXME: these are not thread safe !!not !!*/
    timebin_add_particle_to_active(parent, child, P[child].TimeBin);

    /*! When a new additional star particle is created, we can put it into the
     *  tree at the position of the spawning gas particle. This is possible
     *  because the Nextnode[] array essentially describes the full tree walk as a
     *  link list. Multipole moments of tree nodes need not be changed.
     */

    /* we do this only if there is an active force tree 
     * checking Nextnode is not the best way of doing so though.
     * */
    if(force_tree_allocated()) {
        int no;
        no = Nextnode[parent];
        Nextnode[parent] = child;
        Nextnode[child] = no;
        Father[child] = Father[parent];
    }
    return child;
}

/* remove mass = 0 particles, holes in sph chunk and holes in bh buffer;
 * returns 1 if tree / timebin is invalid */
int
domain_garbage_collection(void)
{
    int tree_invalid = 0;

    /* tree is invalidated of the sequence on P is reordered; */
    /* TODO: in principle we can track this change and modify the tree nodes;
     * But doing so requires cleaning up the TimeBin link lists, and the tree
     * link lists first. likely worth it, since GC happens only in domain decompose
     * and snapshot IO, both take far more time than rebuilding the tree. */
    tree_invalid |= domain_sph_garbage_collection_reclaim();
    tree_invalid |= domain_all_garbage_collection();
    tree_invalid |= domain_bh_garbage_collection();

    MPI_Allreduce(MPI_IN_PLACE, &tree_invalid, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    /*This is expensive: maybe try to avoid it*/
    domain_count_particles();

    return tree_invalid;
}

static int
domain_sph_garbage_collection_reclaim()
{
    int tree_invalid = 0;

    int64_t total0, total;
    sumup_large_ints(1, &N_sph_slots, &total0);

#ifdef SFR
    int i;
    for(i = 0; i < N_sph_slots; i++) {
        while(P[i].Type != 0 && i < N_sph_slots) {
            /* remove this particle from SphP, because
             * it is no longer a SPH
             * */
            /* note that when i == N-sph - 1 this doesn't really do any
             * thing. no harm done */
            struct particle_data psave;
            psave = P[i];
            P[i] = P[N_sph_slots - 1];
            SPHP(i) = SPHP(N_sph_slots - 1);
            P[N_sph_slots - 1] = psave;
            tree_invalid = 1;
            N_sph_slots --;
        }
    }
#endif
    sumup_large_ints(1, &N_sph_slots, &total);
    if(total != total0) {
        message(0, "GC: Reclaiming SPH slots from %ld to %ld\n", total0, total);
    }
    return tree_invalid;
}

static int
domain_all_garbage_collection()
{
    int i, tree_invalid = 0; 
    int count_elim, count_gaselim;
    int64_t total0, total;
    int64_t total0_gas, total_gas;

    sumup_large_ints(1, &N_sph_slots, &total0_gas);
    sumup_large_ints(1, &NumPart, &total0);

    count_elim = 0;
    count_gaselim = 0;

    for(i = 0; i < NumPart; i++)
        if(P[i].Mass == 0)
        {
            TimeBinCount[P[i].TimeBin]--;

            if(P[i].Type == 0)
            {
                TimeBinCountSph[P[i].TimeBin]--;

                P[i] = P[N_sph_slots - 1];
                SPHP(i) = SPHP(N_sph_slots - 1);

                P[N_sph_slots - 1] = P[NumPart - 1];

                N_sph_slots--;

                count_gaselim++;
            } else
            {
                P[i] = P[NumPart - 1];
            }

            NumPart--;
            i--;

            count_elim++;
        }

    sumup_large_ints(1, &N_sph_slots, &total_gas);
    sumup_large_ints(1, &NumPart, &total);

    if(total_gas != total0_gas) {
        message(0, "GC : Reducing SPH slots from %ld to %ld\n", total0_gas, total_gas);
    }

    if(total != total0) {
        message(0, "GC : Reducing Particle slots from %ld to %ld\n", total0, total);
        tree_invalid = 1;
    }
    return tree_invalid;
}

static int bh_cmp_reverse_link(const void * b1in, const void * b2in) {
    const struct bh_particle_data * b1 = (struct bh_particle_data *) b1in;
    const struct bh_particle_data * b2 = (struct bh_particle_data *) b2in;
    if(b1->ReverseLink == -1 && b2->ReverseLink == -1) {
        return 0;
    }
    if(b1->ReverseLink == -1) return 1;
    if(b2->ReverseLink == -1) return -1;
    return (b1->ReverseLink > b2->ReverseLink) - (b1->ReverseLink < b2->ReverseLink);

}

static int
domain_bh_garbage_collection()
{
    /*
     *  BhP is a lifted structure. 
     *  changing BH slots doesn't affect tree's consistency;
     *  this function always return 0. */

    /* gc the bh */
    int i, j;
    int64_t total = 0;

    int64_t total0 = 0;

    sumup_large_ints(1, &N_bh_slots, &total0);

    /* If there are no blackholes, there cannot be any garbage. bail. */
    if(total0 == 0) return 0;

#pragma omp parallel for
    for(i = 0; i < All.MaxPartBh; i++) {
        BhP[i].ReverseLink = -1;
    }

#pragma omp parallel for
    for(i = 0; i < NumPart; i++) {
        if(P[i].Type == 5) {
            BhP[P[i].PI].ReverseLink = i;
            if(P[i].PI >= N_bh_slots) {
                endrun(1, "bh PI consistency failed2, N_bh_slots = %d, N_bh = %d, PI=%d\n", N_bh_slots, NLocal[5], P[i].PI);
            }
            if(BhP[P[i].PI].ID != P[i].ID) {
                endrun(1, "bh id consistency failed1\n");
            }
        }
    }

    /* put unused guys to the end, and sort the used ones
     * by their location in the P array */
    qsort(BhP, N_bh_slots, sizeof(BhP[0]), bh_cmp_reverse_link);

    while(N_bh_slots > 0 && BhP[N_bh_slots - 1].ReverseLink == -1) {
        N_bh_slots --;
    }
    /* Now update the link in BhP */
    for(i = 0; i < N_bh_slots; i ++) {
        P[BhP[i].ReverseLink].PI = i;
    }

    /* Now invalidate ReverseLink */
    for(i = 0; i < N_bh_slots; i ++) {
        BhP[i].ReverseLink = -1;
    }

    j = 0;
#pragma omp parallel for
    for(i = 0; i < NumPart; i++) {
        if(P[i].Type != 5) continue;
        if(P[i].PI >= N_bh_slots) {
            endrun(1, "bh PI consistency failed2\n");
        }
        if(BhP[P[i].PI].ID != P[i].ID) {
            endrun(1, "bh id consistency failed2\n");
        }
#pragma omp atomic
        j ++;
    }
    if(j != N_bh_slots) {
        endrun(1, "bh count failed2, j=%d, N_bh=%d\n", j, N_bh_slots);
    }

    sumup_large_ints(1, &N_bh_slots, &total);

    if(total != total0) {
        message(0, "GC: Reducing number of BH slots from %ld to %ld\n", total0, total);
    }
    /* bh gc never invalidates the tree */
    return 0;
}



