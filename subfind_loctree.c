#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#ifdef SUBFIND

#include "allvars.h"
#include "proto.h"
#include "domain.h"
#include "subfind.h"


static int last;		/* auxialiary variable used to set-up non-recursive walk */

static double RootLen, RootCenter[3];

void subfind_loctree_findExtent(int npart, struct unbind_data *mp)
{
  int i, j, k;
  double len, xmin[3], xmax[3];

  /* determine extension */
  for(i = 0; i < 3; i++)
    {
      xmin[i] = MAX_REAL_NUMBER;
      xmax[i] = -MAX_REAL_NUMBER;
    }

  for(k = 0; k < npart; k++)
    {
      if(mp)
	i = mp[k].index;
      else
	endrun(1121);

      for(j = 0; j < 3; j++)
	{
	  if(xmin[j] > P[i].Pos[j])
	    xmin[j] = P[i].Pos[j];

	  if(xmax[j] < P[i].Pos[j])
	    xmax[j] = P[i].Pos[j];
	}
    }

  len = 0;
  for(j = 0; j < 3; j++)
    if(xmax[j] - xmin[j] > len)
      len = xmax[j] - xmin[j];

  len *= 1.001;

  for(j = 0; j < 3; j++)
    RootCenter[j] = 0.5 * (xmin[j] + xmax[j]);

  RootLen = len;
}

void subfind_loctree_copyExtent(void)
{
  int j;

  for(j = 0; j < 3; j++)
    RootCenter[j] = DomainCenter[j];

  RootLen = DomainLen;
}

int subfind_loctree_treebuild(int npart, struct unbind_data *mp)
{
  int i, k, subnode = 0, parent = -1, numnodes;
  int nfree, th, nn;
  double lenhalf;
  struct NODE *nfreep;

  /* select first node */
  nfree = All.MaxPart;
  nfreep = &Nodes[nfree];


  /* create an empty  root node  */
  nfreep->len = RootLen;
  for(i = 0; i < 3; i++)
    nfreep->center[i] = RootCenter[i];

  for(i = 0; i < 8; i++)
    nfreep->u.suns[i] = -1;

  numnodes = 1;
  nfreep++;
  nfree++;

  /* insert all particles */

  for(k = 0; k < npart; k++)
    {
      if(mp)
	i = mp[k].index;
      else
	endrun(211);

      th = All.MaxPart;

      while(1)
	{
	  if(th >= All.MaxPart)	/* we are dealing with an internal node */
	    {
	      subnode = 0;
	      if(P[i].Pos[0] > Nodes[th].center[0])
		subnode += 1;
	      if(P[i].Pos[1] > Nodes[th].center[1])
		subnode += 2;
	      if(P[i].Pos[2] > Nodes[th].center[2])
		subnode += 4;

	      nn = Nodes[th].u.suns[subnode];

	      if(nn >= 0)	/* ok, something is in the daughter slot already, need to continue */
		{
		  parent = th;	/* note: subnode can still be used in the next step of the walk */
		  th = nn;
		}
	      else
		{
		  /* here we have found an empty slot where we can 
		   * attach the new particle as a leaf 
		   */
		  Nodes[th].u.suns[subnode] = i;
		  break;	/* done for this particle */
		}
	    }
	  else
	    {
	      /* we try to insert into a leaf with a single particle
	       * need to generate a new internal node at this point 
	       */
	      Nodes[parent].u.suns[subnode] = nfree;

	      nfreep->len = 0.5 * Nodes[parent].len;
	      lenhalf = 0.25 * Nodes[parent].len;

	      if(subnode & 1)
		nfreep->center[0] = Nodes[parent].center[0] + lenhalf;
	      else
		nfreep->center[0] = Nodes[parent].center[0] - lenhalf;

	      if(subnode & 2)
		nfreep->center[1] = Nodes[parent].center[1] + lenhalf;
	      else
		nfreep->center[1] = Nodes[parent].center[1] - lenhalf;

	      if(subnode & 4)
		nfreep->center[2] = Nodes[parent].center[2] + lenhalf;
	      else
		nfreep->center[2] = Nodes[parent].center[2] - lenhalf;

	      nfreep->u.suns[0] = -1;
	      nfreep->u.suns[1] = -1;
	      nfreep->u.suns[2] = -1;
	      nfreep->u.suns[3] = -1;
	      nfreep->u.suns[4] = -1;
	      nfreep->u.suns[5] = -1;
	      nfreep->u.suns[6] = -1;
	      nfreep->u.suns[7] = -1;

	      subnode = 0;
	      if(P[th].Pos[0] > nfreep->center[0])
		subnode += 1;
	      if(P[th].Pos[1] > nfreep->center[1])
		subnode += 2;
	      if(P[th].Pos[2] > nfreep->center[2])
		subnode += 4;

	      if(nfreep->len < 1.0e-3 * All.SofteningTable[P[i].Type])
		{
		  /* seems like we're dealing with particles   
		   * at identical locations. randomize 
		   * subnode index (well below gravitational softening scale). 
		   */
		  subnode = (int) (8.0 * get_random_number(nfree));
		  if(subnode >= 8)
		    subnode = 7;
		}

	      nfreep->u.suns[subnode] = th;

	      th = nfree;	/* resume trying to insert the new particle at 
				   the newly created internal node */

	      numnodes++;
	      nfree++;
	      nfreep++;

	      if((numnodes) >= MaxNodes)
		{
		  printf("maximum number %d of tree-nodes reached.\n", MaxNodes);
		  printf("for particle %d  %g %g %g\n", i, P[i].Pos[0], P[i].Pos[1], P[i].Pos[2]);
		  endrun(1);
		}
	    }
	}
    }

  /* now compute the multipole moments recursively */
  last = -1;
  subfind_loctree_update_node_recursive(All.MaxPart, -1, -1);

  if(last >= All.MaxPart)
    Nodes[last].u.d.nextnode = -1;
  else
    Nextnode[last] = -1;

  /*
     fprintf(Logfile,"Have put %d particles into tree.\n", num);
   */

  return numnodes;
}

/* this routine computes the multipole moments for a given internal node and
 * all its subnodes using a recursive computation.  Note that the moments of
 * the daughter nodes are already stored in single precision. For very large
 * particle numbers, loss of precision may results for certain particle
 * distributions
 */
void subfind_loctree_update_node_recursive(int no, int sib, int father)
{
  int j, jj, p, pp = 0, nextsib, suns[8];
  double mass;
  double s[3];

  if(no >= All.MaxPart)
    {
      for(j = 0; j < 8; j++)
	suns[j] = Nodes[no].u.suns[j];	/* this "backup" is necessary because the nextnode entry will
					   overwrite one element (union!) */
      if(last >= 0)
	{
	  if(last >= All.MaxPart)
	    Nodes[last].u.d.nextnode = no;
	  else
	    Nextnode[last] = no;
	}

      last = no;

      mass = 0;
      s[0] = 0;
      s[1] = 0;
      s[2] = 0;

      for(j = 0; j < 8; j++)
	{
	  if((p = suns[j]) >= 0)
	    {
	      /* check if we have a sibling on the same level */
	      for(jj = j + 1; jj < 8; jj++)
		if((pp = suns[jj]) >= 0)
		  break;

	      if(jj < 8)	/* yes, we do */
		nextsib = pp;
	      else
		nextsib = sib;

	      subfind_loctree_update_node_recursive(p, nextsib, no);

	      if(p >= All.MaxPart)	/* an internal node or pseudo particle */
		{
		  mass += Nodes[p].u.d.mass;	/* we assume a fixed particle mass */
		  s[0] += Nodes[p].u.d.mass * Nodes[p].u.d.s[0];
		  s[1] += Nodes[p].u.d.mass * Nodes[p].u.d.s[1];
		  s[2] += Nodes[p].u.d.mass * Nodes[p].u.d.s[2];
		}
	      else		/* a particle */
		{
		  mass += P[p].Mass;
		  s[0] += P[p].Mass * P[p].Pos[0];
		  s[1] += P[p].Mass * P[p].Pos[1];
		  s[2] += P[p].Mass * P[p].Pos[2];
		}
	    }
	}

      if(mass > 0)
	{
	  s[0] /= mass;
	  s[1] /= mass;
	  s[2] /= mass;
	}
      else
	{
	  s[0] = Nodes[no].center[0];
	  s[1] = Nodes[no].center[1];
	  s[2] = Nodes[no].center[2];
	}

      Nodes[no].u.d.s[0] = s[0];
      Nodes[no].u.d.s[1] = s[1];
      Nodes[no].u.d.s[2] = s[2];
      Nodes[no].u.d.mass = mass;

      Nodes[no].u.d.sibling = sib;
      Nodes[no].u.d.father = father;
    }
  else				/* single particle or pseudo particle */
    {
      if(last >= 0)
	{
	  if(last >= All.MaxPart)
	    Nodes[last].u.d.nextnode = no;
	  else
	    Nextnode[last] = no;
	}

      last = no;

      if(no < All.MaxPart)	/* only set it for single particles */
	Father[no] = father;
    }
}





double subfind_loctree_treeevaluate_potential(int target)
{
  struct NODE *nop = 0;
  int no;
  double r2, dx, dy, dz, mass, r, u, h, h_inv, wp;
  double pot, pos_x, pos_y, pos_z;

  pos_x = P[target].Pos[0];
  pos_y = P[target].Pos[1];
  pos_z = P[target].Pos[2];

  h = All.ForceSoftening[P[target].Type];
  h_inv = 1.0 / h;

  pot = 0;

  no = All.MaxPart;

  while(no >= 0)
    {
      if(no < All.MaxPart)	/* single particle */
	{
	  dx = P[no].Pos[0] - pos_x;
	  dy = P[no].Pos[1] - pos_y;
	  dz = P[no].Pos[2] - pos_z;
	  mass = P[no].Mass;
	}
      else
	{
	  nop = &Nodes[no];
	  dx = nop->u.d.s[0] - pos_x;
	  dy = nop->u.d.s[1] - pos_y;
	  dz = nop->u.d.s[2] - pos_z;
	  mass = nop->u.d.mass;
	}

#ifdef PERIODIC
      dx = NEAREST(dx);
      dy = NEAREST(dy);
      dz = NEAREST(dz);
#endif
      r2 = dx * dx + dy * dy + dz * dz;

      if(no < All.MaxPart)
	{
	  no = Nextnode[no];
	}
      else			/* we have an internal node. Need to check opening criterion */
	{
	  if(nop->len * nop->len > r2 * All.ErrTolThetaSubfind * All.ErrTolThetaSubfind)
	    {
	      /* open cell */
	      if(mass)
		{
		  no = nop->u.d.nextnode;
		  continue;
		}
	    }

	  no = nop->u.d.sibling;	/* node can be used */
	}

      r = sqrt(r2);

      if(r >= h)
	pot -= mass / r;
      else
	{
	  u = r * h_inv;

	  if(u < 0.5)
	    wp = -2.8 + u * u * (5.333333333333 + u * u * (6.4 * u - 9.6));
	  else
	    wp =
	      -3.2 + 0.066666666667 / u + u * u * (10.666666666667 +
						   u * (-16.0 + u * (9.6 - 2.133333333333 * u)));

	  pot += mass * h_inv * wp;
	}
    }

  return pot;
}






int subfind_locngb_compare_key(const void *a, const void *b)
{
  if(((struct r2data *) a)->r2 < (((struct r2data *) b)->r2))
    return -1;

  if(((struct r2data *) a)->r2 > (((struct r2data *) b)->r2))
    return +1;

  return 0;
}


double subfind_locngb_treefind(MyDouble xyz[3], int desngb, double hguess)
{
  int numngb;
  double part_dens;
  double h2max;

  if(hguess == 0)
    {
      part_dens = All.Omega0 * 3 * All.Hubble * All.Hubble / (8 * M_PI * All.G) / P[0].Mass;
      hguess = pow(3 * desngb / (4 * M_PI) / part_dens, 1.0 / 3);
    }

  do
    {
      /*
         fprintf(Logfile,"hguess= %g\n", hguess);
       */
      numngb = subfind_locngb_treefind_variable(xyz, hguess);

      if(numngb < desngb)
	{
	  hguess *= 1.26;
	  continue;
	}

      if(numngb >= desngb)
	{
	  qsort(R2list, numngb, sizeof(struct r2data), subfind_locngb_compare_key);
	  h2max = R2list[desngb - 1].r2;
	  break;
	}

      hguess *= 1.26;
    }
  while(1);

  return sqrt(h2max);
}


int subfind_locngb_treefind_variable(MyDouble searchcenter[3], double hguess)
{
  int numngb, no, p;
  double dx, dy, dz, r2, h2;
  struct NODE *this;

  h2 = hguess * hguess;

  numngb = 0;
  no = All.MaxPart;

  while(no >= 0)
    {
      if(no < All.MaxPart)	/* single particle */
	{
	  p = no;
	  no = Nextnode[no];

#ifdef PERIODIC
	  dx = NEAREST(P[p].Pos[0] - searchcenter[0]);
	  dy = NEAREST(P[p].Pos[1] - searchcenter[1]);
	  dz = NEAREST(P[p].Pos[2] - searchcenter[2]);
#else
	  dx = P[p].Pos[0] - searchcenter[0];
	  dy = P[p].Pos[1] - searchcenter[1];
	  dz = P[p].Pos[2] - searchcenter[2];
#endif
	  if(dx < -hguess)
	    continue;
	  if(dx > hguess)
	    continue;

	  if(dy < -hguess)
	    continue;
	  if(dy > hguess)
	    continue;

	  if(dz < -hguess)
	    continue;
	  if(dz > hguess)
	    continue;

	  r2 = dx * dx + dy * dy + dz * dz;

	  if(r2 < h2)
	    {
	      R2list[numngb].r2 = r2;
	      R2list[numngb].index = p;
	      numngb++;
	    }
	}
      else
	{
	  this = &Nodes[no];

	  no = Nodes[no].u.d.sibling;	/* in case the node can be discarded */

#ifdef PERIODIC
	  if((NEAREST(this->center[0] - searchcenter[0]) + 0.5 * this->len) < -hguess)
	    continue;
	  if((NEAREST(this->center[0] - searchcenter[0]) - 0.5 * this->len) > hguess)
	    continue;
	  if((NEAREST(this->center[1] - searchcenter[1]) + 0.5 * this->len) < -hguess)
	    continue;
	  if((NEAREST(this->center[1] - searchcenter[1]) - 0.5 * this->len) > hguess)
	    continue;
	  if((NEAREST(this->center[2] - searchcenter[2]) + 0.5 * this->len) < -hguess)
	    continue;
	  if((NEAREST(this->center[2] - searchcenter[2]) - 0.5 * this->len) > hguess)
	    continue;
#else
	  if(((this->center[0] - searchcenter[0]) + 0.5 * this->len) < -hguess)
	    continue;
	  if(((this->center[0] - searchcenter[0]) - 0.5 * this->len) > hguess)
	    continue;
	  if(((this->center[1] - searchcenter[1]) + 0.5 * this->len) < -hguess)
	    continue;
	  if(((this->center[1] - searchcenter[1]) - 0.5 * this->len) > hguess)
	    continue;
	  if(((this->center[2] - searchcenter[2]) + 0.5 * this->len) < -hguess)
	    continue;
	  if(((this->center[2] - searchcenter[2]) - 0.5 * this->len) > hguess)
	    continue;
#endif

	  no = this->u.d.nextnode;	/* ok, we need to open the node */
	}
    }

  return numngb;
}








/* this function allocates memory used for storage of the tree
 * and auxiliary arrays for tree-walk and link-lists.
 */
size_t subfind_loctree_treeallocate(int maxnodes, int maxpart)	/* usually maxnodes=0.7*maxpart is sufficient */
{
  size_t bytes, allbytes = 0;

  MaxNodes = maxnodes;

  Nodes_base = mymalloc("Nodes_base", bytes = (MaxNodes + 1) * sizeof(struct NODE));
  allbytes += bytes;

  Nextnode = mymalloc("Nextnode", bytes = maxpart * sizeof(int));
  allbytes += bytes;

  Father = mymalloc("Father", bytes = maxpart * sizeof(int));
  allbytes += bytes;

  R2list = mymalloc("R2list", bytes = maxpart * sizeof(struct r2data));
  allbytes += bytes;

  Nodes = Nodes_base - All.MaxPart;

  return allbytes;
}


/* free the allocated memory
 */
void subfind_loctree_treefree(void)
{
  myfree(R2list);
  myfree(Father);
  myfree(Nextnode);
  myfree(Nodes_base);
}


#endif
