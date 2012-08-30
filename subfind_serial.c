#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "allvars.h"
#include "proto.h"

#ifdef SUBFIND
#include "subfind.h"
#include "fof.h"

/* this file processes the local groups in serial mode */


#ifndef MAX_NGB_CHECK
#define MAX_NGB_CHECK 2
#endif

static int *Head, *Next, *Tail, *Len;
static struct cand_dat
{
  int head;
  int len;
  int nsub;
  int rank, subnr, parent;
  int bound_length;
}
 *candidates;




int subfind_process_group_serial(int gr, int Offs)
{
  int i, j, k, p, len, subnr, totlen, ss, ngbs, ndiff, N, head = 0, head_attach, count_cand;
  int listofdifferent[2], count, prev;
  int ngb_index, part_index, nsubs, rank;
  double SubMass, SubPos[3], SubVel[3], SubCM[3], SubVelDisp, SubVmax, SubVmaxRad, SubSpin[3],
    SubHalfMass, SubMassTab[6], SubLum[6];
  MyIDType SubMostBoundID;
  static struct unbind_data *ud;

  while(P[Offs].GrNr != Group[gr].GrNr)
    {
      Offs++;
      if(Offs >= NumPart)
	{
	  printf("don't find a particle for groupnr=%d\n", Group[gr].GrNr);
	  endrun(312);
	}
    }

  N = Group[gr].Len;
  GrNr = Group[gr].GrNr;

  for(i = 0; i < N; i++)
    {
      if(P[Offs + i].GrNr != Group[gr].GrNr)
	{
	  printf
	    ("task=%d, gr=%d: don't have the number of particles for GrNr=%d group-len=%d found=%d before=%d\n",
	     ThisTask, gr, Group[gr].GrNr, N, P[Offs + i].GrNr, P[Offs - 1].GrNr);
	  endrun(312);
	}
    }


  candidates = mymalloc("candidates", N * sizeof(struct cand_dat));

  Head = mymalloc("Head", N * sizeof(int));
  Next = mymalloc("Next", N * sizeof(int));
  Tail = mymalloc("Tail", N * sizeof(int));
  Len = mymalloc("Len", N * sizeof(int));
  ud = mymalloc("ud", N * sizeof(struct unbind_data));

  Head -= Offs;
  Next -= Offs;
  Tail -= Offs;
  Len -= Offs;

  for(i = 0; i < N; i++)
    {
      ud[i].index = Offs + i;
    }

  subfind_loctree_findExtent(N, ud);

  subfind_loctree_treebuild(N, ud);	/* build tree for all particles of this group */

  for(i = Offs; i < Offs + N; i++)
    Head[i] = Next[i] = Tail[i] = -1;

  /* note: particles are already ordered in the order of decreasing density */

  for(i = 0, count_cand = 0; i < N; i++)
    {
      part_index = Offs + i;

      subfind_locngb_treefind(P[part_index].Pos, All.DesLinkNgb, P[part_index].DM_Hsml);

      /* note: returned neighbours are already sorted by distance */

      for(k = 0, ndiff = 0, ngbs = 0; k < All.DesLinkNgb && ngbs < MAX_NGB_CHECK && ndiff < 2; k++)
	{
	  ngb_index = R2list[k].index;

	  if(ngb_index != part_index)	/* to exclude the particle itself */
	    {
	      /* we only look at neighbours that are denser */
	      if(P[ngb_index].u.DM_Density > P[part_index].u.DM_Density)
		{
		  ngbs++;

		  if(Head[ngb_index] >= 0)	/* neighbor is attached to a group */
		    {
		      if(ndiff == 1)
			if(listofdifferent[0] == Head[ngb_index])
			  continue;

		      /* a new group has been found */
		      listofdifferent[ndiff++] = Head[ngb_index];
		    }
		  else
		    {
		      printf("this may not occur.\n");
		      printf
			("ThisTask=%d gr=%d k=%d i=%d part_index=%d ngb_index = %d  head[ngb_index]=%d P[part_index].DM_Density=%g %g GrNrs= %d %d \n",
			 ThisTask, gr, k, i, part_index, ngb_index, Head[ngb_index],
			 P[part_index].u.DM_Density, P[ngb_index].u.DM_Density, P[part_index].GrNr,
			 P[ngb_index].GrNr);
		      endrun(2);
		    }
		}
	    }
	}

      switch (ndiff)		/* treat the different possible cases */
	{
	case 0:		/* this appears to be a lonely maximum -> new group */
	  head = part_index;
	  Head[part_index] = Tail[part_index] = part_index;
	  Len[part_index] = 1;
	  Next[part_index] = -1;
	  break;

	case 1:		/* the particle is attached to exactly one group */
	  head = listofdifferent[0];
	  Head[part_index] = head;
	  Next[Tail[head]] = part_index;
	  Tail[head] = part_index;
	  Len[head]++;
	  Next[part_index] = -1;
	  break;

	case 2:		/* the particle merges two groups together */

	  head = listofdifferent[0];
	  head_attach = listofdifferent[1];

	  if(Len[head_attach] > Len[head])	/* other group is longer, swap them */
	    {
	      head = listofdifferent[1];
	      head_attach = listofdifferent[0];
	    }

	  /* only in case the attached group is long enough we bother to register is 
	     as a subhalo candidate */

	  if(Len[head_attach] >= All.DesLinkNgb)
	    {
	      candidates[count_cand].len = Len[head_attach];
	      candidates[count_cand].head = Head[head_attach];
	      count_cand++;
	    }

	  /* now join the two groups */
	  Next[Tail[head]] = head_attach;
	  Tail[head] = Tail[head_attach];
	  Len[head] += Len[head_attach];

	  ss = head_attach;
	  do
	    {
	      Head[ss] = head;
	    }
	  while((ss = Next[ss]) >= 0);

	  /* finally, attach the particle */
	  Head[part_index] = head;
	  Next[Tail[head]] = part_index;
	  Tail[head] = part_index;
	  Len[head]++;
	  Next[part_index] = -1;
	  break;

	default:
	  printf("can't be! (a)\n");
	  endrun(1);
	  break;
	}
    }

  /* add the full thing as a subhalo candidate */
  for(i = 0, prev = -1; i < N; i++)
    {
      if(Head[Offs + i] == Offs + i)
	if(Next[Tail[Offs + i]] == -1)
	  {
	    if(prev < 0)
	      head = Offs + i;
	    if(prev >= 0)
	      Next[prev] = Offs + i;

	    prev = Tail[Offs + i];
	  }
    }

  candidates[count_cand].len = N;
  candidates[count_cand].head = head;
  count_cand++;

  /* go through them once and assign the rank */
  for(i = 0, p = head, rank = 0; i < N; i++)
    {
      Len[p] = rank++;
      p = Next[p];
    }

  /* for each candidate, we now pull out the rank of its head */
  for(k = 0; k < count_cand; k++)
    candidates[k].rank = Len[candidates[k].head];

  for(i = Offs; i < Offs + N; i++)
    Tail[i] = -1;

  for(k = 0, nsubs = 0; k < count_cand; k++)
    {
      for(i = 0, p = candidates[k].head, len = 0; i < candidates[k].len; i++, p = Next[p])
	if(Tail[p] < 0)
	  ud[len++].index = p;

      if(len >= All.DesLinkNgb)
	len = subfind_unbind(ud, len);

      if(len >= All.DesLinkNgb)
	{
	  /* ok, we found a substructure */

	  for(i = 0; i < len; i++)
	    Tail[ud[i].index] = nsubs;	/* we use this to flag the substructures */

	  candidates[k].nsub = nsubs;
	  candidates[k].bound_length = len;
	  nsubs++;
	}
      else
	{
	  candidates[k].nsub = -1;
	  candidates[k].bound_length = 0;
	}
    }

#ifdef VERBOSE
  printf("\nGroupLen=%d  (gr=%d)\n", N, gr);
  printf("Number of substructures: %d\n", nsubs);
#endif

  Group[gr].Nsubs = nsubs;
  Group[gr].Pos[0] = Group[gr].CM[0];
  Group[gr].Pos[1] = Group[gr].CM[1];
  Group[gr].Pos[2] = Group[gr].CM[2];

  qsort(candidates, count_cand, sizeof(struct cand_dat), subfind_compare_serial_candidates_boundlength);

  /* now we determine the parent subhalo for each candidate */
  for(k = 0; k < count_cand; k++)
    {
      candidates[k].subnr = k;
      candidates[k].parent = 0;
    }

  qsort(candidates, count_cand, sizeof(struct cand_dat), subfind_compare_serial_candidates_rank);


  for(k = 0; k < count_cand; k++)
    {
      for(j = k + 1; j < count_cand; j++)
	{
	  if(candidates[j].rank > candidates[k].rank + candidates[k].len)
	    break;

	  if(candidates[k].rank + candidates[k].len >= candidates[j].rank + candidates[j].len)
	    {
	      if(candidates[k].bound_length >= All.DesLinkNgb)
		candidates[j].parent = candidates[k].subnr;
	    }
	  else
	    {
	      printf("k=%d|%d has rank=%d and len=%d.  j=%d has rank=%d and len=%d bound=%d\n",
		     k, count_cand, (int) candidates[k].rank, candidates[k].len,
		     (int) candidates[k].bound_length, candidates[j].rank,
		     (int) candidates[j].len, candidates[j].bound_length);
	      endrun(121235513);
	    }
	}
    }

  qsort(candidates, count_cand, sizeof(struct cand_dat), subfind_compare_serial_candidates_subnr);

  /* now determine the properties */

  for(k = 0, subnr = 0, totlen = 0; k < nsubs; k++)
    {
      len = candidates[k].bound_length;

#ifdef VERBOSE
      printf("subnr=%d  SubLen=%d\n", subnr, len);
#endif

      totlen += len;

      for(i = 0, p = candidates[k].head, count = 0; i < candidates[k].len; i++)
	{
	  if(Tail[p] == candidates[k].nsub)
	    ud[count++].index = p;

	  p = Next[p];
	}

      if(count != len)
	endrun(12);

      subfind_determine_sub_halo_properties(ud, len, &SubMass,
					    &SubPos[0], &SubVel[0], &SubCM[0], &SubVelDisp, &SubVmax,
					    &SubVmaxRad, &SubSpin[0], &SubMostBoundID, &SubHalfMass,
					    &SubMassTab[0]);

      if(Nsubgroups >= MaxNsubgroups)
	endrun(899);

      if(subnr == 0)
	{
	  for(j = 0; j < 3; j++)
	    Group[gr].Pos[j] = SubPos[j];
	}

      SubGroup[Nsubgroups].Len = len;
      if(subnr == 0)
	SubGroup[Nsubgroups].Offset = Group[gr].Offset;
      else
	SubGroup[Nsubgroups].Offset = SubGroup[Nsubgroups - 1].Offset + SubGroup[Nsubgroups - 1].Len;
      SubGroup[Nsubgroups].GrNr = GrNr - 1;
      SubGroup[Nsubgroups].SubNr = subnr;
      SubGroup[Nsubgroups].SubParent = candidates[k].parent;
      SubGroup[Nsubgroups].Mass = SubMass;
      SubGroup[Nsubgroups].SubMostBoundID = SubMostBoundID;
      SubGroup[Nsubgroups].SubVelDisp = SubVelDisp;
      SubGroup[Nsubgroups].SubVmax = SubVmax;
      SubGroup[Nsubgroups].SubVmaxRad = SubVmaxRad;
      SubGroup[Nsubgroups].SubHalfMass = SubHalfMass;

      for(j = 0; j < 3; j++)
	{
	  SubGroup[Nsubgroups].Pos[j] = SubPos[j];
	  SubGroup[Nsubgroups].CM[j] = SubCM[j];
	  SubGroup[Nsubgroups].Vel[j] = SubVel[j];
	  SubGroup[Nsubgroups].Spin[j] = SubSpin[j];
	}

#ifdef SAVE_MASS_TAB
      for(j = 0; j < 6; j++)
	SubGroup[Nsubgroups].MassTab[j] = SubMassTab[j];
#endif

      Nsubgroups++;

      /* Let's now assign the subgroup number */

      for(i = 0; i < len; i++)
	P[ud[i].index].SubNr = subnr;

      subnr++;
    }

#ifdef VERBOSE
  printf("Fuzz=%d\n", N - totlen);
#endif

  myfree(ud);
  myfree(Len + Offs);
  myfree(Tail + Offs);
  myfree(Next + Offs);
  myfree(Head + Offs);

  myfree(candidates);

  return Offs;
}




int subfind_unbind(struct unbind_data *ud, int len)
{
  double *bnd_energy, energy_limit, weakly_bound_limit = 0;
  int i, j, p, minindex, unbound, phaseflag;
  double ddxx, s[3], dx[3], v[3], dv[3], pos[3];
  double vel_to_phys, H_of_a, atime, pot, minpot = 0;
  double boxsize, boxhalf;
  double TotMass;

  boxsize = All.BoxSize;
  boxhalf = 0.5 * All.BoxSize;

  if(All.ComovingIntegrationOn)
    {
      vel_to_phys = 1.0 / All.Time;
      H_of_a = hubble_function(All.Time);
      atime = All.Time;
    }
  else
    {
      vel_to_phys = atime = 1;
      H_of_a = 0;
    }

  bnd_energy = (double *) mymalloc("bnd_energy", len * sizeof(double));

  phaseflag = 0;		/* this means we will recompute the potential for all particles */

  do
    {
      subfind_loctree_treebuild(len, ud);

      /* let's compute the potential  */

      if(phaseflag == 0)	/* redo it for all the particles */
	{
	  for(i = 0, minindex = -1, minpot = 1.0e30; i < len; i++)
	    {
	      p = ud[i].index;

	      pot = subfind_loctree_treeevaluate_potential(p);
	      /* note: add self-energy */
	      P[p].u.DM_Potential = pot + P[p].Mass / All.SofteningTable[P[p].Type];
	      P[p].u.DM_Potential *= All.G / atime;

	      if(All.TotN_gas > 0 && (FOF_PRIMARY_LINK_TYPES & 1) == 0 && All.OmegaBaryon > 0)
		P[p].u.DM_Potential *= All.Omega0 / (All.Omega0 - All.OmegaBaryon);

	      if(P[p].u.DM_Potential < minpot || minindex == -1)
		{
		  minpot = P[p].u.DM_Potential;
		  minindex = p;
		}
	    }

	  for(j = 0; j < 3; j++)
	    pos[j] = P[minindex].Pos[j];	/* position of minimum potential */
	}
      else
	{
	  /* we only repeat for those close to the unbinding threshold */
	  for(i = 0; i < len; i++)
	    {
	      p = ud[i].index;

	      if(P[p].v.DM_BindingEnergy >= weakly_bound_limit)
		{
		  pot = subfind_loctree_treeevaluate_potential(p);
		  /* note: add self-energy */
		  P[p].u.DM_Potential = pot + P[p].Mass / All.SofteningTable[P[p].Type];
		  P[p].u.DM_Potential *= All.G / atime;

		  if(All.TotN_gas > 0 && (FOF_PRIMARY_LINK_TYPES & 1) == 0 && All.OmegaBaryon > 0)
		    P[p].u.DM_Potential *= All.Omega0 / (All.Omega0 - All.OmegaBaryon);
		}
	    }
	}

      /* let's get bulk velocity and the center-of-mass */

      v[0] = v[1] = v[2] = 0;
      s[0] = s[1] = s[2] = 0;

      for(i = 0, TotMass = 0; i < len; i++)
	{
	  p = ud[i].index;

	  for(j = 0; j < 3; j++)
	    {
#ifdef PERIODIC
	      ddxx = NEAREST(P[p].Pos[j] - pos[j]);
#else
	      ddxx = P[p].Pos[j] - pos[j];
#endif
	      s[j] += P[p].Mass * ddxx;
	      v[j] += P[p].Mass * P[p].Vel[j];
	    }
	  TotMass += P[p].Mass;
	}

      for(j = 0; j < 3; j++)
	{
	  v[j] /= TotMass;
	  s[j] /= TotMass;	/* center-of-mass */

	  s[j] += pos[j];

#ifdef PERIODIC
	  while(s[j] < 0)
	    s[j] += boxsize;
	  while(s[j] >= boxsize)
	    s[j] -= boxsize;
#endif
	}

      for(i = 0; i < len; i++)
	{
	  p = ud[i].index;

	  for(j = 0; j < 3; j++)
	    {
	      dv[j] = vel_to_phys * (P[p].Vel[j] - v[j]);
#ifdef PERIODIC
	      dx[j] = atime * NEAREST(P[p].Pos[j] - s[j]);
#else
	      dx[j] = atime * (P[p].Pos[j] - s[j]);
#endif
	      dv[j] += H_of_a * dx[j];
	    }

	  P[p].v.DM_BindingEnergy =
	    P[p].u.DM_Potential + 0.5 * (dv[0] * dv[0] + dv[1] * dv[1] + dv[2] * dv[2]);

#ifdef DENSITY_SPLIT_BY_TYPE
	  if(P[p].Type == 0)
	    P[p].v.DM_BindingEnergy += P[p].w.int_energy;
#endif
	  bnd_energy[i] = P[p].v.DM_BindingEnergy;
	}

      qsort(bnd_energy, len, sizeof(double), subfind_compare_binding_energy);	/* largest comes first! */

      energy_limit = bnd_energy[(int) (0.25 * len)];

      for(i = 0, unbound = 0; i < len - 1; i++)
	{
	  if(bnd_energy[i] > 0)
	    unbound++;
	  else
	    unbound--;

	  if(unbound <= 0)
	    break;
	}
      weakly_bound_limit = bnd_energy[i];

      /* now omit unbound particles,  but at most 1/4 of the original size */

      for(i = 0, unbound = 0; i < len; i++)
	{
	  p = ud[i].index;
	  if(P[p].v.DM_BindingEnergy > 0 && P[p].v.DM_BindingEnergy > energy_limit)
	    {
	      unbound++;
	      ud[i] = ud[len - 1];
	      i--;
	      len--;
	    }
	}

      if(len < All.DesLinkNgb)
	break;

      if(phaseflag == 0)
	{
	  if(unbound > 0)
	    phaseflag = 1;
	}
      else
	{
	  if(unbound == 0)
	    {
	      phaseflag = 0;	/* this will make us repeat everything once more for all particles */
	      unbound = 1;
	    }
	}
    }
  while(unbound > 0);

  myfree(bnd_energy);

  return (len);
}



int subfind_compare_grp_particles(const void *a, const void *b)
{
  if(((struct particle_data *) a)->GrNr < ((struct particle_data *) b)->GrNr)
    return -1;

  if(((struct particle_data *) a)->GrNr > ((struct particle_data *) b)->GrNr)
    return +1;

  if(((struct particle_data *) a)->SubNr < ((struct particle_data *) b)->SubNr)
    return -1;

  if(((struct particle_data *) a)->SubNr > ((struct particle_data *) b)->SubNr)
    return +1;

  if(((struct particle_data *) a)->v.DM_BindingEnergy < ((struct particle_data *) b)->v.DM_BindingEnergy)
    return -1;

  if(((struct particle_data *) a)->v.DM_BindingEnergy > ((struct particle_data *) b)->v.DM_BindingEnergy)
    return +1;

  return 0;
}

void subfind_determine_sub_halo_properties(struct unbind_data *d, int num, double *totmass,
					   double *pos, double *vel, double *cm, double *veldisp,
					   double *vmax, double *vmaxrad, double *spin,
					   MyIDType * mostboundid, double *halfmassrad, double *mass_tab)
{
  int i, j, p;
  double s[3], v[3], max, vel_to_phys, H_of_a, atime, minpot;
  double lx, ly, lz, dv[3], dx[3], disp;
  double boxhalf, boxsize, ddxx;
  sort_r2list *rr_list = 0;
  int minindex;
  double mass, maxrad;

  boxsize = All.BoxSize;
  boxhalf = 0.5 * boxsize;

  if(All.ComovingIntegrationOn)
    {
      vel_to_phys = 1.0 / All.Time;
      H_of_a = hubble_function(All.Time);
      atime = All.Time;
    }
  else
    {
      vel_to_phys = atime = 1;
      H_of_a = 0;
    }

  for(i = 0, minindex = -1, minpot = 1.0e30; i < num; i++)
    {
      p = d[i].index;
      if(P[p].u.DM_Potential < minpot || minindex == -1)
	{
	  minpot = P[p].u.DM_Potential;
	  minindex = p;
	}
    }

  for(j = 0; j < 3; j++)
    pos[j] = P[minindex].Pos[j];


  /* pos[] now holds the position of minimum potential */
  /* we take it that as the center */


  for(i = 0, minindex = -1, minpot = 1.0e30; i < num; i++)
    {
      p = d[i].index;
      if(P[p].v.DM_BindingEnergy < minpot || minindex == -1)
	{
	  minpot = P[p].v.DM_BindingEnergy;
	  minindex = p;
	}
    }

  *mostboundid = P[minindex].ID;


  /* let's get bulk velocity and the center-of-mass */

  for(j = 0; j < 3; j++)
    s[j] = v[j] = 0;

  for(j = 0; j < 6; j++)
    mass_tab[j] = 0;

  for(i = 0, mass = 0; i < num; i++)
    {
      p = d[i].index;
      for(j = 0; j < 3; j++)
	{
#ifdef PERIODIC
	  ddxx = NEAREST(P[p].Pos[j] - pos[j]);
#else
	  ddxx = P[p].Pos[j] - pos[j];
#endif
	  s[j] += P[p].Mass * ddxx;
	  v[j] += P[p].Mass * P[p].Vel[j];
	}
      mass += P[p].Mass;

      mass_tab[P[p].Type] += P[p].Mass;
    }

  *totmass = mass;

  for(j = 0; j < 3; j++)
    {
      s[j] /= mass;		/* center of mass */
      v[j] /= mass;

      vel[j] = vel_to_phys * v[j];
    }

  for(j = 0; j < 3; j++)
    {
      s[j] += pos[j];

#ifdef PERIODIC
      while(s[j] < 0)
	s[j] += boxsize;
      while(s[j] >= boxsize)
	s[j] -= boxsize;
#endif

      cm[j] = s[j];
    }


  disp = lx = ly = lz = 0;

  rr_list = mymalloc("rr_list", sizeof(sort_r2list) * num);

  for(i = 0; i < num; i++)
    {
      p = d[i].index;
      rr_list[i].r = 0;
      rr_list[i].mass = P[p].Mass;

      for(j = 0; j < 3; j++)
	{
#ifdef PERIODIC
	  ddxx = NEAREST(P[p].Pos[j] - s[j]);
#else
	  ddxx = P[p].Pos[j] - s[j];
#endif
	  dx[j] = atime * ddxx;
	  dv[j] = vel_to_phys * (P[p].Vel[j] - v[j]);
	  dv[j] += H_of_a * dx[j];

	  disp += P[p].Mass * dv[j] * dv[j];
	  /* for rotation curve computation, take minimum of potential as center */
#ifdef PERIODIC
	  ddxx = NEAREST(P[p].Pos[j] - pos[j]);
#else
	  ddxx = P[p].Pos[j] - pos[j];
#endif
	  ddxx = atime * ddxx;
	  rr_list[i].r += ddxx * ddxx;
	}

      lx += P[p].Mass * (dx[1] * dv[2] - dx[2] * dv[1]);
      ly += P[p].Mass * (dx[2] * dv[0] - dx[0] * dv[2]);
      lz += P[p].Mass * (dx[0] * dv[1] - dx[1] * dv[0]);

      rr_list[i].r = sqrt(rr_list[i].r);
    }

  *veldisp = sqrt(disp / (3 * mass));	/* convert to 1d velocity dispersion */

  spin[0] = lx / mass;
  spin[1] = ly / mass;
  spin[2] = lz / mass;

  qsort(rr_list, num, sizeof(sort_r2list), subfind_compare_dist_rotcurve);

  *halfmassrad = rr_list[num / 2].r;

  /* compute cumulative mass */
  for(i = 1; i < num; i++)
    rr_list[i].mass = rr_list[i - 1].mass + rr_list[i].mass;

  for(i = num - 1, max = 0, maxrad = 0; i > 5; i--)
    if(rr_list[i].mass / rr_list[i].r > max)
      {
	max = rr_list[i].mass / rr_list[i].r;
	maxrad = rr_list[i].r;
      }

  *vmax = sqrt(All.G * max);
  *vmaxrad = maxrad;

#ifdef KD_FRICTION_DYNAMIC
  for(i = 0; i < num; i++)
    {
      p = d[i].index;
      P[p].BH_sigma = *vmax / sqrt(3);
      P[p].BH_bmax = *halfmassrad;         
    }
#endif

  myfree(rr_list);
}

int subfind_compare_serial_candidates_boundlength(const void *a, const void *b)
{
  if(((struct cand_dat *) a)->bound_length > ((struct cand_dat *) b)->bound_length)
    return -1;

  if(((struct cand_dat *) a)->bound_length < ((struct cand_dat *) b)->bound_length)
    return +1;

  if(((struct cand_dat *) a)->rank < ((struct cand_dat *) b)->rank)
    return -1;

  if(((struct cand_dat *) a)->rank > ((struct cand_dat *) b)->rank)
    return +1;

  return 0;
}

int subfind_compare_serial_candidates_rank(const void *a, const void *b)
{
  if(((struct cand_dat *) a)->rank < ((struct cand_dat *) b)->rank)
    return -1;

  if(((struct cand_dat *) a)->rank > ((struct cand_dat *) b)->rank)
    return +1;

  if(((struct cand_dat *) a)->len > ((struct cand_dat *) b)->len)
    return -1;

  if(((struct cand_dat *) a)->len < ((struct cand_dat *) b)->len)
    return +1;

  return 0;
}

int subfind_compare_serial_candidates_subnr(const void *a, const void *b)
{
  if(((struct cand_dat *) a)->subnr < ((struct cand_dat *) b)->subnr)
    return -1;

  if(((struct cand_dat *) a)->subnr > ((struct cand_dat *) b)->subnr)
    return +1;

  return 0;
}


#endif
