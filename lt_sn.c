#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>

#include "allvars.h"
#include "proto.h"

#include "lt_error_codes.h"

#ifdef LT_STELLAREVOLUTION

#include "lt_sn.h"


static int ndone, ntotdone;


//! 
/*! 

  \brief This routine evolves all active stars and star-forming gas particles
  \param mode instructs evolve_SN whether to evolve all the particles or the active only
  
  \return the number of evolved particles

  This routine evolves all active stars and star-forming gas
  particles in order to advance them on the chemical timeline and
  to spread the released metals and energy released on their neighbourhood

*/

unsigned int evolve_SN(void)
{
  unsigned int I;		/*!< points to the right MetP memeber */
  int starsnum, allstarsun, gasnum;
  int i, j, k, source;		/*!< general counters */
  unsigned int place;
  int ndone_flag, done_count;	/*!< to account for the bunch size */
  int nimport, nexport;
  int sendTask, recvTask, level, ngrp;	/*!< used in the communication structure */
  MPI_Status status;
  double Metals[LT_NMet], energy, LMMass;	/*!< stores the result from stellarevolution() */
  double tstart, tend;		/*!< used to account for used cpu-time */
  int IMFi;			/*!< points to the right IMF          */
  int Yset, YZbin;		/*!< points to the right yields set   */
  double Z;


  /* : ---------------------------------------------- */
  /* : preliminary operations                         */
  /* : ---------------------------------------------- */


  /* if this option is set, the release of metals is
     halted below
     All.Below_this_redshift_stop_cooling */
#ifdef LT_STOP_MET_BELOW_Z
  if((1.0 / All.Time - 1.0) < All.Below_this_redshift_stop_cooling)
    return 0;
#endif


  /* find the total number of active stars and sf gas
     particles */
  tstart = second();

  count_evolving_stars(&starsnum, &gasnum);

  tend = second();
  SN_Find = timediff(tstart, tend);
  MPI_Reduce(&SN_Find, &sumSN_Find, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  //  All.CPU_SN_find += sumSN_Find / NTask;

  sumup_large_ints(1, &starsnum, &tot_starsnum);
  sumup_large_ints(1, &N_stars, &tot_allstarsnum);
  sumup_large_ints(1, &gasnum, &tot_gasnum);


/*   MPI_Allreduce(&starsnum, &tot_starsnum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD); */
/*   MPI_Allreduce(&gasnum, &tot_gasnum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD); */


  if(tot_starsnum + tot_gasnum == 0)	/* if no particles are active, exit */
    {
#if defined(LT_SEv_INFO) && defined(LT_SEvDbg)
      if(SEvInfo_GRAIN == 0)	/* it it is time, checksum the metals */
	get_metals_sumcheck(9);
#endif
      myfree(NextChemActiveParticle);
      return 0;
    }

#ifdef LT_SEvDbg
  if(ThisTask == 0)
    {
      weightlist = (double *) mymalloc("dbg_weightlist", NTask * sizeof(double));
      ngblist = (double *) mymalloc("dbg_ngblist", NTask * sizeof(double));
    }
#endif

  if(ThisTask == 0)
    printf("calculating stellar evolution for %llu stars (over %llu) and %llu gas partcles\n", tot_starsnum,
	   tot_allstarsnum, tot_gasnum);
  fflush(stdout);

  for(i = 0; i < INUM; i++)
    infos[i] = 0;

#ifdef LT_SEv_INFO_DETAILS
  if(ThisTask == 0)
    Details = (struct details *) mymalloc("Details", tot_starsnum * 3 * sizeof(struct details));
  else
    Details = (struct details *) mymalloc("Details", starsnum * 3 * sizeof(struct details));
  DetailsPos = 0;
#endif


  /* : ----------------------------------------------- */
  /* : initialize info vars                            */
  /* : ----------------------------------------------- */
#ifdef LT_SEv_INFO

  memset(Zmass, 0, SFs_dim * SPECIES * 2 * LT_NMet * sizeof(double));
  memset(tot_Zmass, 0, SFs_dim * SPECIES * 2 * LT_NMet * sizeof(double));
  memset(SNdata, 0, SFs_dim * SPECIES * 2 * SN_INUM * sizeof(double));
  memset(tot_SNdata, 0, SFs_dim * SPECIES * 2 * SN_INUM * sizeof(double));

#if defined(LT_EJECTA_IN_HOTPHASE)
  for(i = 0; i < 2; i++)
    {
      SpreadEgy[i] = tot_SpreadEgy[i] = 0;
      SpreadMinMaxEgy[i][0] = SpreadMinMaxEgy[i][1] = 0;
      tot_SpreadMinMaxEgy[i][0] = tot_SpreadMinMaxEgy[i][1] = 0;
    }
  for(i = 0; i < 3; i++)
    {
      AgbFrac[i] = 0;
      SpecEgyChange[i] = 0;
      CFracChange[i] = 0;
    }
  SpreadMinMaxEgy[0][0] = MIN_INIT_VALUE;
  AgbFrac[1] = MIN_INIT_VALUE;
  SpecEgyChange[1] = MIN_INIT_VALUE;
  CFracChange[2] = MIN_INIT_VALUE;
  CFracChange[3] = 0;
#endif

  if(++SEvInfo_grain > SEvInfo_GRAIN)
    {
      for(i = 0; i < AL_INUM_sum; i++)
	ALdata_sum[i] = 0;
      for(i = 0; i < AL_INUM_max; i++)
	ALdata_max[i] = 0;
      for(i = 0; i < AL_INUM_min; i++)
	ALdata_min[i] = MIN_INIT_VALUE;

      for(i = 0; i < S_INUM_sum; i++)
	Stat_sum[i] = 0;
      for(i = 0; i < S_INUM_max; i++)
	Stat_max[i] = 0;
      for(i = 0; i < S_INUM_min; i++)
	Stat_min[i] = MIN_INIT_VALUE;

#ifdef LT_SEv_INFO_DETAILS_onSPREAD
      for(i = 0; i < SP_INUM; i++)
	SP[i] = tot_SP[i] = 0;
      SP[SP_W_min] = SP[SP_N_min] = MIN_INIT_VALUE;
#endif
    }
#endif

  /* : ----------------------------------------------- */
  /* : allocate memory to exchange data                */
  /* : ----------------------------------------------- */

  /* allocate the space for the list of neighbours */
  Ngblist = (int *) mymalloc("Ngblist", NumPart * sizeof(int));
  /* find available room in memory                 */
  All.BunchSize =
    (int) ((All.BufferSize * 1024 * 1024) /
	   (sizeof(struct data_index) + sizeof(struct data_nodelist) + sizeof(struct metaldata_index)));

  DataIndexTable =
    (struct data_index *) mymalloc("DataIndexTable", All.BunchSize * sizeof(struct data_index));
  DataNodeList =
    (struct data_nodelist *) mymalloc("DataNodeList", All.BunchSize * sizeof(struct data_nodelist));

  MetalDataIn =
    (struct metaldata_index *) mymalloc("MetalDataIn", All.BunchSize * sizeof(struct metaldata_index));

  /* now, all the evolving stars have the right spreading lenght and
   *  have stored the total weight of neighbours.
   *
   * the nex step is to communicate the stellar evolution details and
   *  the total weight in order to spread over the neighbours
   */

  i = 0;

  /* MAIN (outer) cycle                 */
  /* .................................. */

  i = FirstChemActiveParticle;

  do
    {
      for(j = 0; j < NTask; j++)
	{
	  Send_count[j] = 0;
	  Recv_count[j] = 0;
	  Exportflag[j] = -1;
	}

      /* do local particles and prepare export list */
      tstart = second();
      for(nexport = 0; i >= 0; i = NextChemActiveParticle[i])
	{
	  if(P[i].Type & EVOLVE)
	    {
	      P[i].Type &= ~EVOLVE;
	      /* here we account for the stellar evolution and do
	         operations for local particles 
	       */
	      if((LMMass = perform_stellarevolution_operations(i, &nexport, &Metals[0], &energy)) < 0)
		{
		  P[i].Type |= EVOLVE;
		  break;
		}
	      else
		{
#ifdef LT_SEvDbg
		  if(FirstID > 0 && P[i].ID == FirstID)
		    {
		      do_spread_dbg = 1;
		      printf
			("[SEvDBG] @ %g %d spread: spreading particle over %f neighbours withinh %f radius, having %g weight:: ",
			 All.Time, All.NumCurrentTiStep, PPP[i].n.NumNgb, PPP[i].Hsml,
			 MetP[P[i].MetID].weight);
		      for(j = 0; j < LT_NMet; j++)
			printf("%g ", Metals[j]);
		      printf("\n");
		      fflush(stdout);
		    }
#endif
		  if(P[i].Type == 4)
		    MetP[P[i].MetID].weight = 0;
		}
	    }
	}
#ifdef LT_SEvDbg
      MPI_Allgather(&do_spread_dbg, 1, MPI_INT, &do_spread_dbg_list[0], 1, MPI_INT, MPI_COMM_WORLD);
      for(j = 0; j < NTask; j++)
	if((do_spread_dbg = do_spread_dbg_list[j]))
	  break;
#endif
      tend = second();
      //timecomp1 += timediff(tstart, tend);

#ifdef MYSORT
      mysort_dataindex(DataIndexTable, nexport, sizeof(struct data_index), data_index_compare);
#else
      qsort(DataIndexTable, nexport, sizeof(struct data_index), data_index_compare);
#endif
      qsort(MetalDataIn, nexport, sizeof(struct metaldata_index), metaldata_index_compare);

      tstart = second();

      MPI_Alltoall(Send_count, 1, MPI_INT, Recv_count, 1, MPI_INT, MPI_COMM_WORLD);

      tend = second();
      //timewait1 += timediff(tstart, tend);

      for(j = 0, nimport = 0, Recv_offset[0] = 0, Send_offset[0] = 0; j < NTask; j++)
	{
	  nimport += Recv_count[j];

	  if(j > 0)
	    {
	      Send_offset[j] = Send_offset[j - 1] + Send_count[j - 1];
	      Recv_offset[j] = Recv_offset[j - 1] + Recv_count[j - 1];
	    }
	}

      /* preparing particle data for export has been done during the local evaluation */

      /* exchange particle data */

      MetalDataGet = mymalloc("MetalDataGet", nimport * sizeof(struct metaldata_index));
#ifdef LT_SEvDBG_global
      MetalDataSpreadReport = mymalloc("MetalDataSpreadReport", nimport * sizeof(struct spreaddata_index));
#endif

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
		  MPI_Sendrecv(&MetalDataIn[Send_offset[recvTask]],
			       Send_count[recvTask] * sizeof(struct metaldata_index), MPI_BYTE,
			       recvTask, TAG_SMOOTH_A,
			       &MetalDataGet[Recv_offset[recvTask]],
			       Recv_count[recvTask] * sizeof(struct metaldata_index), MPI_BYTE,
			       recvTask, TAG_SMOOTH_A, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}
	    }
	}
      tend = second();
      //timecommsumm1 += timediff(tstart, tend);

      /* now do the particles that were sent to us */

      tstart = second();
      for(j = 0; j < nimport; j++)
	spread_evaluate(j, BUFFER, MetalDataGet[j].Metals, MetalDataGet[j].LMMass, MetalDataGet[j].energy,
			&ngrp, &ngrp);
      tend = second();
      /*timecomp2 += timediff(tstart, tend); */

      if(i < 0)			/* flag the actual active particle */
	ndone_flag = 1;
      else
	ndone_flag = 0;

      tstart = second();
      MPI_Allreduce(&ndone_flag, &ndone, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
      tend = second();
      /* timewait2 += timediff(tstart, tend); */

      myfree(MetalDataGet);

#ifdef LT_SEvDBG
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
		  MPI_Sendrecv(&MetalDataIn[Send_offset[recvTask]],
			       Send_count[recvTask] * sizeof(struct metaldata_index), MPI_BYTE,
			       recvTask, TAG_SMOOTH_A,
			       &MetalDataGet[Recv_offset[recvTask]],
			       Recv_count[recvTask] * sizeof(struct metaldata_index), MPI_BYTE,
			       recvTask, TAG_SMOOTH_A, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}
	    }
	}
      tend = second();

#endif
    }
  while(ndone < NTask);

  myfree(MetalDataIn);
  myfree(DataNodeList);
  myfree(DataIndexTable);
  myfree(Ngblist);

  double Factor;

  for(i = 0; i < NumPart; i++)
    if(P[i].Type == 0 && SphP[i].MassRes > 0)
      {
	P[i].Mass += SphP[i].MassRes;
	SphP[i].MassRes = 0;
#ifdef LT_POPIII_FLAGS
	if(SphP[i].PIIIflag & STILL_POPIII)
	  {
	    Factor = get_metalmass(SphP[i].Metals);
	    if(fabs(Factor - SphP[i].prec_metal_mass) / Factor > 1e-4)
	      fprintf(stderr, "  ** %8.6e %8.6e %8.6e ", All.Time,
		      fabs(Factor - SphP[i].prec_metal_mass) / Factor, Factor);
	    if(SFi != All.PopIII_IMF_idx)
	      {
		SphP[i].PIIIflag &= ~((int) STILL_POPIII);
		SphP[i].prec_metal_mass = get_metallicity(i, -1);
	      }
	  }
#endif
      }

  /* ******************************************** ****** *** **  * */

#ifdef LT_SEvDbg
  MPI_Barrier(MPI_COMM_WORLD);
  if(do_spread_dbg)
    {
      MPI_Gather(&weight_sum, 1, MPI_DOUBLE, &weightlist[0], 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
      MPI_Gather(&ngb_sum, 1, MPI_DOUBLE, &ngblist[0], 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
      if(ThisTask == 0)
	{
	  for(k = 1; k < NTask; k++)
	    {
	      weight_sum += weightlist[k];
	      ngb_sum += ngblist[k];
	    }
	  printf("[SEvDBG] @ %g %d spread sum weight: %f :: and NumNgb : %f ::", All.Time,
		 All.NumCurrentTiStep, weight_sum, ngb_sum);
	  for(k = 0; k < NTask; k++)
	    if(weightlist[k] + ngblist[k] > 0)
	      printf(" w/f from %d are %g/%g ", k, weightlist[k], ngblist[k]);
	  printf("\n");
	  fflush(stdout);
	}
      do_spread_dbg = 0;
      weight_sum = ngb_sum = 0;
      for(j = 0; j < NTask; j++)
	do_spread_dbg_list[j] = 0;
    }
#endif

#ifdef LT_SEv_INFO

  tstart = second();

#ifdef LT_SEv_INFO_DETAILS_onSPREAD
  if(SEvInfo_grain > SEvInfo_GRAIN)
    {
      for(k = 0; k < N_gas; k++)
	if(P[k].Type == 0 && SphP[k].weight_spread > 0)
	  {
	    SP[SP_Sum] += 1;
	    SP[SP_W] += (Z = SphP[k].weight_spread / SphP[k].NeighStars);
	    SP[SP_N] += (double) SphP[k].NeighStars;

	    if(Z < SP[SP_W_min])
	      SP[SP_W_min] = Z;
	    if(Z > SP[SP_W_max])
	      SP[SP_W_max] = Z;

	    if((double) SphP[k].NeighStars < SP[SP_N_min])
	      SP[SP_N_min] = (double) SphP[k].NeighStars;
	    if((double) SphP[k].NeighStars > SP[SP_N_max])
	      SP[SP_N_max] = (double) SphP[k].NeighStars;

	    SphP[k].weight_spread = SphP[k].NeighStars = 0;
	  }

      MPI_Reduce(&SP[SP_Sum], &tot_SP[SP_Sum], 3, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
      tot_SP[SP_W] /= tot_SP[SP_Sum];
      tot_SP[SP_N] /= tot_SP[SP_Sum];
      MPI_Reduce(&SP[SP_W_min], &tot_SP[SP_W_min], 2, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
      MPI_Reduce(&SP[SP_W_max], &tot_SP[SP_W_max], 2, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    }
#endif

  MPI_Reduce(SNdata, tot_SNdata, SPECIES * 2 * SN_INUM * SFs_dim, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  MPI_Reduce(Zmass, tot_Zmass, SPECIES * 2 * LT_NMet * SFs_dim, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

#if defined(LT_EJECTA_IN_HOTPHASE)
  MPI_Reduce(&SpreadEgy[0], &tot_SpreadEgy[0], 2, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&SpreadMinMaxEgy[0][0], &tot_SpreadMinMaxEgy[0][0], 1, MPI_2DOUBLE_PRECISION,
	     MPI_MINLOC, 0, MPI_COMM_WORLD);
  MPI_Reduce(&SpreadMinMaxEgy[1][0], &tot_SpreadMinMaxEgy[1][0], 1, MPI_2DOUBLE_PRECISION,
	     MPI_MAXLOC, 0, MPI_COMM_WORLD);
  MPI_Reduce(&AgbFrac[0], &tot_AgbFrac[0], 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&AgbFrac[1], &tot_AgbFrac[1], 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
  MPI_Reduce(&AgbFrac[2], &tot_AgbFrac[2], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

  MPI_Reduce(&SpecEgyChange[0], &tot_SpecEgyChange[0], 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&SpecEgyChange[1], &tot_SpecEgyChange[1], 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
  MPI_Reduce(&SpecEgyChange[2], &tot_SpecEgyChange[2], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

  MPI_Reduce(&CFracChange[0], &tot_CFracChange[0], 2, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&CFracChange[2], &tot_CFracChange[2], 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
  MPI_Reduce(&CFracChange[3], &tot_CFracChange[3], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

#endif

  if(ThisTask == 0)
    {

#if defined(LT_EJECTA_IN_HOTPHASE)
      if(tot_SpreadEgy[0] > 0)
	{
	  fprintf(FdExtEgy, "> %g %g min %g %g max %g %g - %g %g %g"
		  "- %g %g %g - %g %g %g \n", All.Time,
		  tot_SpreadEgy[1] / tot_SpreadEgy[0],
		  tot_SpreadMinMaxEgy[0][0], tot_SpreadMinMaxEgy[0][1],
		  tot_SpreadMinMaxEgy[1][0], tot_SpreadMinMaxEgy[1][1],
		  tot_AgbFrac[0] / tot_SpreadEgy[0],
		  tot_AgbFrac[1], tot_AgbFrac[2],
		  tot_SpecEgyChange[0] / tot_SpreadEgy[0],
		  tot_SpecEgyChange[1], tot_SpecEgyChange[2],
		  tot_CFracChange[1] / tot_CFracChange[0], tot_CFracChange[2], tot_CFracChange[3]);
	  fflush(FdExtEgy);
	}
#endif

      for(k = 0; k < SFs_dim; k++)
	{
	  if(UseSnIa)
	    {
	      if(GET3d(SN_INUM, tot_SNdata, k, SPEC_snIa, SN_num))
		{
		  fprintf(FdSn, "%3s %1d %9.7g %9i %9.7g %9.7g ", "Ia", k,
			  All.Time, tot_starsnum,
			  GET3d(SN_INUM, tot_SNdata, k, SPEC_snIa, SN_num),
			  GET3d(SN_INUM, tot_SNdata, k, SPEC_snIa, SN_egy));
		  for(j = 0; j < LT_NMet; j++)
		    fprintf(FdSn, "%9.7g ", GET3d(LT_NMet, tot_Zmass, k, SPEC_snIa, j));
		  fprintf(FdSn, "\n");
		  fflush(FdSn);
		}

	      /* write losses */
	      if(GET3d(SN_INUM, tot_SNdata, k, (SPEC_snIa + 1), SN_num))
		{
		  fprintf(FdSnLost, "%3s %1d %9.7g %9.7g %9.7g ", "Ia", k,
			  All.Time,
			  GET3d(SN_INUM, tot_SNdata, k, (SPEC_snIa + 1), SN_num),
			  GET3d(SN_INUM, tot_SNdata, k, (SPEC_snIa + 1), SN_egy));

		  for(j = 0; j < LT_NMet; j++)
		    fprintf(FdSnLost, "%9.7g ", GET3d(LT_NMet, tot_Zmass, k, (SPEC_snIa + 1), j));
		  fprintf(FdSnLost, "\n");
		  fflush(FdSnLost);
		}
	    }

	  if(UseAGB)
	    {
	      if(GET3d(SN_INUM, tot_SNdata, k, SPEC_agb, SN_num))
		{
		  fprintf(FdSn, "%3s %1d %9.7g %9i %9.7g 0 ", "AGB", k, All.Time,
			  tot_starsnum, GET3d(SN_INUM, tot_SNdata, k, SPEC_agb, SN_num));
		  for(j = 0; j < LT_NMet; j++)
		    fprintf(FdSn, "%9.7g ", GET3d(LT_NMet, tot_Zmass, k, SPEC_agb, j));
		  fprintf(FdSn, "\n");
		  fflush(FdSn);
		}

	      /* write losses */
	      if(GET3d(SN_INUM, tot_SNdata, k, (SPEC_agb + 1), SN_num))
		{
		  fprintf(FdSnLost, "%3s %1d %9.7g %9.7g ", "AGB", k, All.Time,
			  GET3d(SN_INUM, tot_SNdata, k, (SPEC_agb + 1), SN_num));
		  for(j = 0; j < LT_NMet; j++)
		    fprintf(FdSnLost, "%9.7g ", GET3d(LT_NMet, tot_Zmass, k, (SPEC_agb + 1), j));
		  fprintf(FdSnLost, "\n");
		  fflush(FdSnLost);
		}
	    }

	  if(UseSnII)
	    {
	      if(GET3d(SN_INUM, tot_SNdata, k, SPEC_snII, SN_num) > 0)
		{
		  fprintf(FdSn, "%3s %1d %9.7g %9i %9.7g %9.7g ", "II", k,
			  All.Time, tot_starsnum,
			  GET3d(SN_INUM, tot_SNdata, k, SPEC_snII, SN_num),
			  GET3d(SN_INUM, tot_SNdata, k, SPEC_snII, SN_egy));
		  for(j = 0; j < LT_NMet; j++)
		    fprintf(FdSn, "%9.7g ", GET3d(LT_NMet, tot_Zmass, k, SPEC_snII, j));
		  fprintf(FdSn, "\n");
		  fflush(FdSn);
		}

	      /* write losses */
	      if(GET3d(SN_INUM, tot_SNdata, k, (SPEC_snII + 1), SN_num) > 0)
		{
		  fprintf(FdSnLost, "%3s %1d %9.7g %9.7g %9.7g ", "II", k,
			  All.Time,
			  GET3d(SN_INUM, tot_SNdata, k, (SPEC_snII + 1), SN_num),
			  GET3d(SN_INUM, tot_SNdata, k, (SPEC_snII + 1), SN_egy));
		  for(j = 0; j < LT_NMet; j++)
		    fprintf(FdSnLost, "%9.7g ", GET3d(LT_NMet, tot_Zmass, k, (SPEC_snII + 1), j));
		  fprintf(FdSnLost, "\n");
		  fflush(FdSnLost);
		}

	      if(GET3d(SN_INUM, tot_SNdata, k, SPEC_IRA, SN_num) > 0)
		{
		  fprintf(FdSn, "%3s %1d %9.7g %9i %9.7g %9.7g ", "IRA", k,
			  All.Time, tot_starsnum,
			  GET3d(SN_INUM, tot_SNdata, k, SPEC_IRA, SN_num),
			  GET3d(SN_INUM, tot_SNdata, k, SPEC_IRA, SN_egy));
		  for(j = 0; j < LT_NMet; j++)
		    fprintf(FdSn, "%9.7g ", GET3d(LT_NMet, tot_Zmass, k, SPEC_IRA, j));
		  fprintf(FdSn, "\n");
		  fflush(FdSn);
		}

	      /* write losses */
	      if(GET3d(SN_INUM, tot_SNdata, k, (SPEC_IRA + 1), SN_num) > 0)
		{
		  fprintf(FdSnLost, "%3s %1d %9.7g %9.7g %9.7g ", "IRA", k,
			  All.Time,
			  GET3d(SN_INUM, tot_SNdata, k, (SPEC_IRA + 1), SN_num),
			  GET3d(SN_INUM, tot_SNdata, k, (SPEC_IRA + 1), SN_egy));
		  for(j = 0; j < LT_NMet; j++)
		    fprintf(FdSnLost, "%9.7g ", GET3d(LT_NMet, tot_Zmass, k, (SPEC_IRA + 1), j));
		  fprintf(FdSnLost, "\n");
		  fflush(FdSnLost);
		}
	    }
	}

#ifdef LT_SEv_INFO_DETAILS_onSPREAD
      if(tot_SP[SP_Sum] > 0)
	fprintf(FdSPinfo, "%8.6lg "
		"%8.6lg %8.6lg %8.6lg "
		"%8.6lg %8.6lg %8.6lg %8.6lg\n",
		All.Time,
		tot_SP[SP_Sum], tot_SP[SP_W], tot_SP[SP_N],
		tot_SP[SP_W_min], tot_SP[SP_W_max], tot_SP[SP_N_min], tot_SP[SP_N_max]);
      fflush(FdSPinfo);
#endif
    }


  if(SEvInfo_grain > SEvInfo_GRAIN)
    {
      SEvInfo_grain = 0;
      write_metallicity_stat();
#ifdef LT_SMOOTH_SIZE
      AvgSmoothN = 0;

      AvgSmoothSize = 0;
      MinSmoothSize = 1e6;
      MaxSmoothSize = 0;

      AvgSmoothNgb = 0;
      MinSmoothNgb = 1e6;
      MaxSmoothNgb = 0;
#endif
    }

  tend = second();
  infos[SN_info] += timediff(tstart, tend);
#endif

#ifdef LT_SEv_INFO_DETAILS

  for(j = 0; j < NTask; j++)
    /* consider to use Gatherv.. */
    {
      if(ThisTask == j && ThisTask > 0)
	{
	  MPI_Send(&DetailsPos, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
	  MPI_Send(&Details[0], DetailsPos * sizeof(struct details), MPI_BYTE, 0, 1, MPI_COMM_WORLD);
	}
      else if(ThisTask == 0)
	{
	  if(j > 0)
	    {
	      MPI_Recv(&starsnum, 1, MPI_INT, j, 0, MPI_COMM_WORLD, &status);
	      MPI_Recv(&Details[DetailsPos], starsnum * sizeof(struct details), MPI_BYTE, j, 1,
		       MPI_COMM_WORLD, &status);
	      DetailsPos += starsnum;
	    }
	}
      MPI_Barrier(MPI_COMM_WORLD);
    }

  if(ThisTask == 0)
    fwrite(&Details[0], sizeof(struct details), DetailsPos, FdSnDetails);

  myfree(Details);

#endif

#ifdef LT_SEvDbg
  if(ThisTask == 0)
    {
      myfree(ngblist);
      myfree(weightlist);
    }
#endif
  myfree(NextChemActiveParticle);


  MPI_Reduce(&infos[0], &sum_infos[0], INUM, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  if(ThisTask == 0)
    {
      //      All.CPU_SN_info += sum_infos[SN_info] / NTask;
      //      All.CPU_SN_Comm += sum_infos[SN_Comm] / NTask;
      //      All.CPU_SN_Calc += sum_infos[SN_Calc] / NTask;
      //      All.CPU_SN_NeighFind += sum_infos[SN_NeighFind] / NTask;
      //      All.CPU_SN_NeighCheck += sum_infos[SN_NeighCheck] / NTask;
      //      All.CPU_SN_Imbalance += sum_infos[SN_Imbalance] / NTask;
      //      All.CPU_SN_Spread += sum_infos[SN_Spread] / NTask;
    }

  search_for_metalspread = 0;

  return tot_starsnum;
}

/*
   =======================================================
     END of   M a i n   R o u t i n e
     .........................................
   =======================================================
*/


/* :::.............................................................. */
/* ***************************************************************** */


/* NOTE: not used now */
int iterate(int n, int ntrue, MyFloat grad, double *deltax)
{
  MyFloat old, Lold, L, L3, rho;

  int i = 0;

  Lold = *deltax;		/* at the begin *deltax contains the guess for L */
  rho = n / (*deltax * *deltax * *deltax);
  L = *deltax * pow(ntrue / n, 0.33333333);
  L3 = L * L * L;
  *deltax = L - *deltax;
  old = *deltax / 2;

  while(fabs(*deltax - old) / (*deltax) > 0.001)
    {
      i++;
      old = *deltax;
      L3 = L * L * L;
      *deltax = (ntrue - rho * L3) / (grad * L3);
      L = Lold + *deltax;
    }
  return i;
}


/* / ---------------------------------------------- \
 * |                                                |
 * | SECTION IIa                                    |
 * |                                                |
 * | ........................                       |
 * |                                                |
 * | Neighbours search                              |
 * | Stellar Evolution Driver                       |
 * | Spreading                                      |
 * |                                                |
 * |    > stellarevolution                          |
 * |    > spread                                    |  
 * |                                                |
 * \ ---------------------------------------------- / */



/*! This routine is a comparison kernel used in a sort routine to group
 *  particles that are exported to the same processor.
 */


int metaldata_index_compare(const void *a, const void *b)
{
  if(((struct metaldata_index *) a)->Task < (((struct metaldata_index *) b)->Task))
    return -1;

  if(((struct metaldata_index *) a)->Task > (((struct metaldata_index *) b)->Task))
    return +1;

  if(((struct metaldata_index *) a)->Index < (((struct metaldata_index *) b)->Index))
    return -1;

  if(((struct metaldata_index *) a)->Index > (((struct metaldata_index *) b)->Index))
    return +1;

  if(((struct metaldata_index *) a)->IndexGet < (((struct metaldata_index *) b)->IndexGet))
    return -1;

  if(((struct metaldata_index *) a)->IndexGet > (((struct metaldata_index *) b)->IndexGet))
    return +1;

  return 0;
}


/* 
   * ..................... *
   :                       :
   :   Evolving Stars      :
   * ..................... *
*/


double perform_stellarevolution_operations(int i, int *nexport, double *Metals, double *energy)
{
  int j, IMFi;
  int Yset, YZbin;
  double Z;
  double tstart, tend;
  double LMMass;
  float metals[LT_NMet];
  int ret;

#ifdef LT_TRACK_CONTRIBUTES
  NULL_CONTRIB(&contrib);
  NULL_EXPCONTRIB(&IIcontrib);
  NULL_EXPCONTRIB(&Iacontrib);
  NULL_EXPCONTRIB(&AGBcontrib);
#endif

#ifdef LT_SEv_INFO

  spreading_on = 1;

  /* collect some information */
  if(SEvInfo_grain > SEvInfo_GRAIN && (P[i].Type & 4))
    {
      tstart = second();

      /* ..about the minimum, maximum and mean spreading lenght */
      if(PPP[i].Hsml < ALdata_min[MIN_sl])
	ALdata_min[MIN_sl] = PPP[i].Hsml;
      if(PPP[i].Hsml > ALdata_max[MAX_sl])
	ALdata_max[MAX_sl] = PPP[i].Hsml;
      ALdata_sum[MEAN_sl] += PPP[i].Hsml;

      /* ..about the minimum, maximum and mean number of neighbours and the associate spreading legnth */
      if((PPP[i].n.NumNgb >= All.NeighInfNum) && (PPP[i].n.NumNgb < (int) ALdata_min[MIN_ngb]))
	ALdata_min[MIN_ngb] = PPP[i].n.NumNgb;
      if(PPP[i].n.NumNgb < All.LeftNumNgbSN)
	ALdata_sum[NUM_uspread]++;
      if(PPP[i].n.NumNgb > (int) ALdata_max[MAX_ngb])
	ALdata_max[MAX_ngb] = PPP[i].n.NumNgb;
      ALdata_sum[MEAN_ngb] += PPP[i].n.NumNgb;

      tend = second();
      infos[SN_info] += timediff(tstart, tend);
    }
#endif


  for(j = 0; j < LT_NMet; j++)
    Metals[j] = 0;

  *energy = 0;
  LMMass = 0;

  get_SF_index(i, &SFi, &IMFi);
  SFp = (SF_Type *) & SFs[SFi];
  IMFp = (IMF_Type *) & IMFs[IMFi];

  tstart = second();
#ifndef LT_LOCAL_IRA
  if(P[i].Type & 4)
#endif
    {
      //printf("calling stellarevolution for %llu\n", (unsigned long long)P[i].ID); /* if this is an active star, calculate its evolution and update fields */
      LMMass = stellarevolution(i, &Metals[0], energy);
    }

#ifndef LT_LOCAL_IRA
  else if(SFp->nonZeroIRA > 0)
    {				/* if this is a star-forming gas particle accountfor the IRA part, if any */

      Yset = IMFp->YSet;
      Z = get_metallicity(i, -1);
      for(YZbin = IIZbins_dim[Yset] - 1; Z < IIZbins[Yset][YZbin] && YZbin > 0; YZbin--)
	;
      for(j = 0; j < LT_NMet; j++)
	{
	  Metals[j] = SphP[i].mstar * SnII_ShortLiv_Yields[Yset][j][YZbin];
#ifdef LT_SEv_INFO
	  if(spreading_on && j < LT_NMet)
	    ADD3d(LT_NMet, Zmass, SFi, SPEC_IRA, j, Metals[j])
	    else
	    ADD3d(LT_NMet, Zmass, SFi, (SPEC_IRA + 1), j, Metals[j]);
#endif
#ifdef LT_SEv_INFO_DETAILS
	  DetailsWo[j] += Metals[j];
#endif
	}

#ifdef LT_SEv_INFO
      if(spreading_on)
	{
	  ADD3d(SN_INUM, SNdata, SFi, SPEC_IRA, SN_num, SphP[i].mstar);
	  ADD3d(SN_INUM, SNdata, SFi, SPEC_IRA, SN_egy, SFp->IRA_erg_per_g * SphP[i].mstar);
	}
      else
	{
	  ADD3d(SN_INUM, SNdata, SFi, (SPEC_IRA + 1), SN_num, SphP[i].mstar);
	  ADD3d(SN_INUM, SNdata, SFi, (SPEC_IRA + 1), SN_egy, SFp->IRA_erg_per_g * SphP[i].mstar);
	}
#endif
#ifndef LT_SNegy_IN_HOTPHASE
      *energy = 0;		/* already used in effective model */
#endif
#if defined(LT_HOT_EJECTA) || defined(LT_SNegy_IN_HOTPHASE)
      LMMass = 0;
#endif

#ifdef LT_TRACK_CONTRIBUTES
      for(j = 0; j < LT_NMetP; j++)
	{
	  IIcontrib[j] = 1;
	  Iacontrib[j] = AGBcontrib[j] = 0;
	}
#endif
    }
#endif

#ifdef LT_TRACK_CONTRIBUTES
  pack_contrib(&contrib, IMFi, IIcontrib, Iacontrib, AGBcontrib);
#endif

  tend = second();
  infos[SN_Calc] += timediff(tstart, tend);

  tstart = second();

  for(j = 0; j < LT_NMet; j++)
    metals[j] = (float) Metals[j];
  ret = spread_evaluate(i, LOCAL, &metals[0], LMMass, *energy, nexport, Send_count);

  tend = second();
  infos[SN_Spread] += timediff(tstart, tend);


  if(ret >= 0)
    return LMMass;
  else
    return -1;
}


/* This routine return the stellar evolution products for a given star
 * it also upgrade the mass of the star and the last chemical times to the
 * current values.
 */
double stellarevolution(int i, double *metals, double *energy)
{
  int I, IMFi, Yset;

/* #if !(defined (UM_CHEMISTRY) && defined (UM_METAL_COOLING)) */
  double mymetals[LT_NMet], sum_mymetals[LT_NMet];
/* #else */
/*   double mymetals[LT_NMet+1], sum_mymetals[LT_NMet+1]; */
/* #endif */
  double myenergy;
  double mass;
  double starlifetime, mylifetime, delta_evolve_time, prec_evolve_time;
  double inf_mass, sup_mass;
  double numsn;
  int current_chem_bin, chem_step, bin;
  double NextChemTime, LMmass = 0;
  int j,ti_min;

#ifdef LT_SEvDbg
  char string[5000];
#endif

  I = P[i].MetID;
  /* find the IMF associated to this particle */
  IMFi = SFs[SFi].IMFi;
  IMFp = (IMF_Type *) & IMFs[IMFi];
  /* find the yield set associated to this IMF */
  Yset = IMFs[IMFi].YSet;

  for(j = 0; j < LT_NMet; j++)
    mymetals[j] = sum_mymetals[j] = 0;
  *energy = 0;

  myenergy = 0;
  /* calculate the lifetime of the star */
  if(P[i].StellarAge < (float) All.Time)
    starlifetime = get_age(P[i].StellarAge) - All.Time_Age;
  else
    /* due to float-double conversion possible round off */
    starlifetime = 0;

  if(UseSnII && (MetP[I].LastChemTime < All.mean_lifetime))
    {

      mylifetime = starlifetime;

      if((mylifetime >= SFs[SFi].ShortLiv_TimeTh) &&
	 (mylifetime - MetP[I].LastChemTime >= All.MinChemTimeStep))
	{
	  prec_evolve_time = MetP[I].LastChemTime;
	  if(mylifetime > All.mean_lifetime)
	    mylifetime = All.mean_lifetime;

	  inf_mass = dying_mass(mylifetime);
	  sup_mass = dying_mass(prec_evolve_time);

	  mass = get_SnII_product(i, Yset, &mymetals[0], &myenergy, inf_mass, sup_mass, &numsn);

	  if(mass >= 0)
	    {
	      P[i].Mass -= mass;


#ifdef LT_SEv_INFO_DETAILS
	      Details[DetailsPos].ID = P[i].ID;
	      Details[DetailsPos].type = (char) 2;
	      Details[DetailsPos].Data[0] = All.Time;
	      Details[DetailsPos].Data[1] = prec_evolve_time;
	      Details[DetailsPos].Data[2] = mylifetime;
	      for(j = 0; j < LT_NMet; j++)
		Details[DetailsPos].Data[DetailsZ + j] = mymetals[j];
	      DetailsPos++;
#endif
	      for(j = 0; j < LT_NMet; j++)
		sum_mymetals[j] += mymetals[j];
	      *energy += myenergy;

	    }

	  MetP[I].LastChemTime = mylifetime;
	}

#ifdef LT_TRACK_CONTRIBUTES
      for(j = 0; j < LT_NMetP; j++)
	IIcontrib[j] = mymetals[j];
#endif

#ifdef LT_SEvDbg
      if(FirstID != 0 && P[i].ID == FirstID)
	{
	  sprintf(string, "\n[SEvDBG] II %g %g %g %g %g %g %g ", MetP[I].iMass, get_metallicity(i, -1),
		  prec_evolve_time, mylifetime, mass, myenergy, numsn);
	  for(j = 0; j < LT_NMet; j++)
	    sprintf(&string[strlen(string)], "%g ", mymetals[j]);
	  printf("%s\n", string);
	  fflush(stdout);
	}
#endif
    }



  if(UseSnIa || UseAGB)
    {
      if(starlifetime > All.mean_lifetime)
	{
	  prec_evolve_time = MetP[I].LastChemTime;
	  delta_evolve_time = starlifetime - prec_evolve_time;

	  if(delta_evolve_time >= All.MinChemTimeStep)
	    {
	      inf_mass = dying_mass(prec_evolve_time + delta_evolve_time);
	      sup_mass = dying_mass(prec_evolve_time);

	      if(UseSnIa)
		{
		  mass =
		    get_SnIa_product(i, Yset, &mymetals[0], &myenergy, prec_evolve_time, delta_evolve_time);

		  if(mass >= 0)
		    {
		      P[i].Mass -= mass;

		      *energy += myenergy;
		      for(j = 0; j < LT_NMet; j++)
			{
			  sum_mymetals[j] += mymetals[j];
			}

#ifdef LT_SEv_INFO_DETAILS
		      Details[DetailsPos].ID = P[i].ID;
		      Details[DetailsPos].type = (char) 1;
		      Details[DetailsPos].Data[0] = All.Time;
		      Details[DetailsPos].Data[1] = prec_evolve_time;
		      Details[DetailsPos].Data[2] = prec_evolve_time + delta_evolve_time;
		      for(j = 0; j < LT_NMet; j++)
			Details[DetailsPos].Data[DetailsZ + j] = mymetals[j];
		      DetailsPos++;
#endif
		    }

#ifdef LT_TRACK_CONTRIBUTES
		  for(j = 0; j < LT_NMetP; j++)
		    Iacontrib[j] = (float) mymetals[j];
#endif


#ifdef LT_SEvDbg
		  if(FirstID > 0 && P[i].ID == FirstID)
		    {
		      sprintf(string, "[SEvDBG] I %g %g %g %g %g %g ", MetP[I].iMass, get_metallicity(i, -1),
			      prec_evolve_time, starlifetime, mass, myenergy);
		      for(j = 0; j < LT_NMet; j++)
			sprintf(&string[strlen(string)], "%g ", mymetals[j]);
		      printf("%s\n", string);
		      fflush(stdout);
		    }
#endif

		}

	      if(UseAGB)
		{
		  mass = get_AGB_product(i, Yset, &mymetals[0], inf_mass, sup_mass, &numsn);

		  if(mass >= 0)
		    {
		      P[i].Mass -= mass;


		      for(j = 0; j < LT_NMet; j++)
			{
			  sum_mymetals[j] += mymetals[j];
			  /* keep track of ejecta that are not injected explosively */
			  LMmass += mymetals[j];
			}

#ifdef LT_SEv_INFO_DETAILS
		      Details[DetailsPos].ID = P[i].ID;
		      Details[DetailsPos].type = (char) 3;
		      Details[DetailsPos].Data[0] = All.Time;
		      Details[DetailsPos].Data[1] = prec_evolve_time;
		      Details[DetailsPos].Data[2] = starlifetime;
		      for(j = 0; j < LT_NMet; j++)
			Details[DetailsPos].Data[DetailsZ + j] = mymetals[j];
		      DetailsPos++;
#endif

		    }

#ifdef LT_TRACK_CONTRIBUTES
		  for(j = 0; j < LT_NMetP; j++)
		    AGBcontrib[j] = (float) mymetals[j];
#endif

#ifdef LT_SEvDbg
		  if(FirstID != 0 && P[i].ID == FirstID)
		    {
		      sprintf(string, "[SEvDBG] AGB %g %g %g %g %g %g 0 ", MetP[I].iMass,
			      get_metallicity(i, -1), prec_evolve_time, starlifetime, mass, numsn);
		      for(j = 0; j < LT_NMet; j++)
			sprintf(&string[strlen(string)], "%g ", mymetals[j]);
		      printf("%s\n", string);
		      fflush(stdout);
		    }
#endif
		  MetP[I].LastChemTime = starlifetime;
		}
	    }

	}
    }

  NextChemTime = get_NextChemTime(starlifetime, SFi, 0x0);
  j = get_chemstep_bin(All.Time, All.Time_Age - NextChemTime, &chem_step, i);

/*   double Time_Age_new, Time_new; */
/*   Time_new = All.TimeBegin * exp( (All.Ti_Current + (1 << j)) * All.Timebase_interval); */
/*   Time_Age_new = get_age(Time_new); */
/*   printf(">>> %llu @ %d @ %g has chemtimestep=%g, j=%d => new a: %u %u %g %g dt: %g frac: %g\n",  */
/* 	 (unsigned long long int)P[i].ID, All.NumCurrentTiStep, starlifetime,All.Time_Age - NextChemTime, j,  */
/* 	 All.Ti_Current, 1<<j, All.Time, Time_new, */
/* 	 All.Time_Age - Time_Age_new,  */
/* 	 (NextChemTime - Time_Age_new)/(All.Time_Age - NextChemTime)); */
  /*       if(TimeBinActive[j] == 0) */
  /*         { */
  /*           while(TimeBinActive[j] == 0 && j > 0) */
  /*             j--; */
  /*           chem_step = j ? (1 << j) : 0; */
  /*         } */
  if(All.Ti_Current >= TIMEBASE)
    chem_step = j = 0;
  
  if((TIMEBASE - All.Ti_Current) < chem_step)
    {
      chem_step = TIMEBASE - All.Ti_Current;
      ti_min = TIMEBASE;
      while(ti_min > chem_step)
        ti_min >>= 1;
      chem_step = ti_min;
      j = get_timestep_bin(chem_step);
    }
  
  if(j != MetP[I].ChemTimeBin)
    {
      TimeBinCountStars[MetP[I].ChemTimeBin]--;
      TimeBinCountStars[j]++;
      MetP[I].ChemTimeBin = j;
    }


#ifdef LT_TRACK_CONTRIBUTES
  for(j = 0; j < LT_NMetP; j++)
    if(sum_mymetals[j] > 0)
      {
	IIcontrib[j] = (float) ((double) IIcontrib[j] / sum_mymetals[j]);
	AGBcontrib[j] = (float) ((double) AGBcontrib[j] / sum_mymetals[j]);
	Iacontrib[j] = (float) ((double) Iacontrib[j] / sum_mymetals[j]);
      }
    else
      {
	IIcontrib[j] = 0;
	AGBcontrib[j] = 0;
	Iacontrib[j] = 0;
      }
#endif

  if(P[i].Mass <= 0)
    {
      /*.. shouldn't occour (oh, really??) */
      printf("  @@@@@@@@@ %i %i %u %i %g %g %g\n", ThisTask, i, P[i].ID, All.NumCurrentTiStep, P[i].Mass,
	     P[i].StellarAge, starlifetime);
      fflush(stdout);
    }

  for(j = 0; j < LT_NMet; j++)
    {
      metals[j] = (float) sum_mymetals[j];
#ifdef LT_SEv_INFO_DETAILS
      if(j < LT_NMetP)
	DetailsWo[j] += sum_mymetals[j];
#endif
    }

  return (float) LMmass;
}



/* 
   * ..................... *
   :                       :
   :   Spreading           :
   * ..................... *
*/


int spread_evaluate(int target, int mode, float *metals, float LMmass, double energy, int *nexport,
		    int *nsend_local)
{
  int I, j, k, n, startnode, numngb, listindex = 0, Type;
  int *ngblist, numngb_inbox;
  MyFloat Pos[3], L;
  double linv, linv3, weight, myweight, wfac;
  double dx, dy, dz, dist, L2, u;
  double add_mass;
  double egyfrac;

#ifdef LT_EJECTA_IN_HOTPHASE
  double a3inv;
  double dt, current_egy, current_egyhot, hotmass;

#ifdef LT_SNegy_IN_HOTPHASE
  double x_hotejecta, sn_spec_egy;
#endif
#if defined(LT_HOT_EJECTA) || defined(LT_SNegy_IN_HOTPHASE)
  double v;
#endif
#endif

#ifdef LT_SEvDbg
  MyIDType ID = 0;
  int being_exported = 0;
#endif

#ifdef PERIODIC
  MyFloat xtmp;
#endif

#ifdef LT_TRACK_CONTRIBUTES
  float contrib_metals[LT_NMetP];
#endif

#if defined(LT_SEv_INFO) && defined(LT_EJECTA_IN_HOTPHASE)
  double tot_spreadegy = 0, spreadegy_ratio, tot_agbspreadegy = 0;
  double agb_frac;
  double egy_ratio = 0, x_ratio = 0;
#endif

#if defined(LT_ZAGE) || defined(LT_POPIII_FLAGS)
  double metal_add_mass;
#endif
#ifdef LT_ZAGE_LLV
  double llvmetal_add_mass;
#endif
#ifdef LT_POPIII_FLAGS
  double mymetalmass;
  double prec_metal_mass;
#endif

  int myIMFi, mySFi;

/* #if defined (UM_CHEMISTRY) && defined (UM_METAL_COOLING) */
/*   float myFillEl_mu; */
/* #endif */

#ifdef LT_EJECTA_IN_HOTPHASE
  a3inv = 1.0 / (All.Time * All.Time * All.Time);
#endif

  startnode = All.MaxPart;

#ifdef LT_ZAGE
  metal_add_mass = 0;
#endif
#ifdef LT_ZAGE_LLV
  llvmetal_add_mass = metals[Iron];
#endif
  for(k = 0, add_mass = 0; k < LT_NMet; k++)
    {
      add_mass += metals[k];
#ifdef LT_ZAGE
      if(k != Hyd && k != Hel)
	metal_add_mass += metals[k];
#endif
    }
  if(add_mass == 0)
    /* may happen if a star is evolved when too less
     * time has elapsed since the last evolution;
     * altghough not dangerous, this calls for a more 
     * clever timing! */
    return(0);

  if(mode)
    {
      Type = MetalDataGet[target].Type;
      for(j = 0; j < 3; j++)
	Pos[j] = MetalDataGet[target].Pos[j];
      L = MetalDataGet[target].L;
      weight = MetalDataGet[target].weight;
#ifdef LT_SEvDbg
      if(FirstID > 0 && MetalDataGet[target].ID == FirstID)
	ID = 1;
#endif
#ifdef LT_TRACK_CONTRIBUTES
      contrib = MetalDataGet[target].contrib;
#endif
      mySFi = MetalDataGet[target].SFi;
      SFp = (SF_Type *) & SFs[SFi];
      myIMFi = SFs[mySFi].IMFi;
      IMFp = (IMF_Type *) & IMFs[myIMFi];
    }
  else
    {
      Type = P[target].Type & 4;
      L = PPP[target].Hsml;
      if(Type)
	{
	  I = P[target].MetID;
	  weight = MetP[I].weight;
	}
      else
	weight = SphP[target].d.Density;

      for(j = 0; j < 3; j++)
	Pos[j] = P[target].Pos[j];

#ifdef LT_SEvDbg
      if(FirstID > 0 && P[target].ID == FirstID)
	ID = 1;
#endif
      mySFi = (int) (SFp - SFs);
      myIMFi = SFs[mySFi].IMFi;
    }

#ifdef LT_EJECTA_IN_HOTPHASE
  agb_frac = LMmass / add_mass;
#endif

#ifdef LT_SNegy_IN_HOTPHASE
  sn_spec_egy = energy / add_mass;
  energy = 0;
#endif

  linv = 1.0 / (L);
  linv3 = linv * linv * linv;
  L2 = L * L;


  if(mode == 0)
    {
      startnode = All.MaxPart;	/* root node */
    }
  else
    {
      startnode = MetalDataGet[target].NodeList[0];
      startnode = Nodes[startnode].u.d.nextnode;	/* open it */
    }

  while(startnode >= 0)
    {
      while(startnode >= 0)
	{
	  numngb_inbox = ngb_treefind_variable(&Pos[0], L, target, &startnode, mode, nexport, nsend_local);
	  if(numngb_inbox < 0)
	    return -1;

	  if(mode == LOCAL)
	    {
	      n = *nexport - 1;
	      while((n >= 0) && (DataIndexTable[n].Index == target))
		{
		  memcpy(MetalDataIn[n].Pos, Pos, sizeof(MyFloat) * 3);
		  memcpy(MetalDataIn[n].Metals, metals, sizeof(float) * LT_NMet);
		  MetalDataIn[n].L = L;
		  MetalDataIn[n].weight = weight;
		  MetalDataIn[n].energy = energy;
		  MetalDataIn[n].SFi = mySFi;
#if defined(LT_EJECTA_IN_HOTPHASE) || defined(LT_HOT_EJECTA) || defined(LT_SNegy_IN_HOTPHASE)
		  MetalDataIn[n].LMMass = LMmass;
#endif
#if defined(LT_SEvDbg)
		  MetalDataIn[n].ID = P[target].ID;
#endif

		  memcpy(MetalDataIn[n].NodeList,
			 DataNodeList[DataIndexTable[n].IndexGet].NodeList, NODELISTLENGTH * sizeof(int));
		  MetalDataIn[n].Type = Type;
		  MetalDataIn[n].Task = DataIndexTable[n].Task;
		  MetalDataIn[n].Index = target;
		  MetalDataIn[n].IndexGet = n;
#ifdef LT_TRACK_CONTRIBUTES
		  MetalDataIn[n].contrib = contrib;                      /* contrib is globa ìl in this scope; it has been set in perform_stellarevolution_operations() */
#endif
		  n--;
#ifdef LT_SEvDbg
		  if(ID)
		    {
		      printf("[SEvDBG] @ %g %d particle is being exported to task %d\n",
			     All.Time, All.NumCurrentTiStep, MetalDataIn[n + 1].Task);
		      fflush(stdout);
		      being_exported++;
		    }
#endif
		}
#ifdef LT_SEvDbg
	      if(ID)
		printf("[SEvDBG] @ %g %d particle is %s being exported to other tasks\n", All.Time,
		       All.NumCurrentTiStep, (being_exported) ? "" : "not");
	      fflush(stdout);
#endif
	    }

	  for(n = 0; n < numngb_inbox; n++)
	    {
	      j = Ngblist[n];

	      dx = NEAREST_X(Pos[0] - P[j].Pos[0]);
	      dy = NEAREST_Y(Pos[1] - P[j].Pos[1]);
	      dz = NEAREST_Z(Pos[2] - P[j].Pos[2]);

	      if((dist = dx * dx + dy * dy + dz * dz) <= L2)
		{
		  /* > ============================================== < */
		  if(Type)	/* >  calculate weights                             < */
		    /* >  for a star                                    < */
		    {
#ifndef LT_USE_TOP_HAT_WEIGHT
#ifndef LT_USE_SOLIDANGLE_WEIGHT
		      dist = sqrt(dist);
		      u = dist * linv;

		      if(u < 0.5)
			myweight = linv3 * (KERNEL_COEFF_1 + KERNEL_COEFF_2 * (u - 1.0) * u * u);
		      else
			myweight = linv3 * KERNEL_COEFF_5 * (1.0 - u) * (1.0 - u) * (1.0 - u);
#else
		      myweight = L * L / (dist * dist);
#endif
#else
		      myweight = 1;
#endif
		    }
		  else		/* >  for a gas particle                            <  */
		    /* >  this means that LT_LOCAL_IRA is not defined   <  */
		    {
		      dist = sqrt(dist);
		      u = dist * linv;

		      if(u < 0.5)
			myweight = linv3 * (KERNEL_COEFF_1 + KERNEL_COEFF_2 * (u - 1.0) * u * u);
		      else
			myweight = linv3 * KERNEL_COEFF_5 * (1.0 - u) * (1.0 - u) * (1.0 - u);
		    }

#if defined(LT_SEvDbg)
		  if(ID)
#if defined(LT_NOFIXEDMASSINSTARKERNEL)
		    ngb_sum += 1;
#else
		    ngb_sum += NORM_COEFF * myweight / linv3;
#endif
#endif

		  if(Type)
		    {
#if !defined(LT_USE_TOP_HAT_WEIGHT_STRICT)
		      myweight *= P[j].Mass;
#if !defined(LT_DONTUSE_DENSITY_in_WEIGHT)
		      myweight /= SphP[j].d.Density;
#endif
#endif
		    }
		  else
		    myweight *= P[j].Mass;

		  if((wfac = myweight / weight) > (1 + 1e-3))	/* > ============================================== < */
		    /* >  relative weight                               < */
		    {
		      printf("Something strange arose when calculating weight in metal distribution:\n"
			     "weightsum: %.8g <  wfac : %.8g\n  *  spreading length : %.8g\n"
			     "Task : %i, neighbour : %i, neigh id: %u, neigh type: %i, neigh dist: %g, "
			     "particle : %i, part ID: %u, mode: %i, numngb: %i\n",
			     weight, wfac, L, ThisTask,
			     j, P[j].ID, P[j].Type, u, target, (mode == LOCAL) ? P[target].ID : -1, mode,
			     numngb_inbox);

		      fflush(stdout);
		      endrun(101010);
		    }

		  if(wfac < 0)	/* > ============================================== < */
		    /* >  should not happen                             < */
		    {
		      if(wfac > -5e-3)
			{
			  printf
			    ("warning: particle has got a negative weight factor! possibly due to a round-off error.\n"
			     "         [%d][%d][%d] %g %g %g %g\n", ThisTask, mode, j, u, myweight, weight,
			     wfac);
			  wfac = 0;
			}
		      else
			{
			  printf
			    ("warning: particle has got a negative weight factor! too large for being a round-off error.\n"
			     "         [%d][%d][%d] %g %g %g %g\n", ThisTask, mode, j, u, myweight, weight,
			     wfac);
			  endrun(919193);
			}
		    }

#ifdef LT_SEv_INFO_DETAILS_onSPREAD
		  SphP[j].weight_spread += (float) wfac;
		  SphP[j].NeighStars++;
#endif

#ifdef LT_TRACK_CONTRIBUTES	/* > ============================================== < */
		  /* >  update tracks                                 < */
		  for(k = 0; k < LT_NMetP; k++)
		    contrib_metals[k] = metals[k] * wfac;

		  update_contrib(&SphP[j].contrib, &SphP[j].Metals[0], &contrib, &contrib_metals[0]);
#endif

#if defined(LT_SEvDbg)
		  if(ID)
		    weight_sum += wfac;
#endif

		  for(k = 0; k < LT_NMetP; k++)	/* > ============================================== < */
		    /* >  check consistency                             < */
		    {
		      /* Hydrogen is the last element, we don't store it in the Metals array */
		      if((SphP[j].Metals[k] += metals[k] * wfac) < 0)
			{
			  printf(" \n ooops... it's a shame! %i %i %i %i %i %i %g %g %g \n",
				 ThisTask, mode, j, target, k, Type, metals[k], wfac, weight);
			  endrun(333333);
			}
#ifdef LT_SEv_INFO_DETAILS	/* >  collect deltails                              < */
		      DetailsW[k] += metals[k] * wfac;
#endif
		    }


		  double zage_term;

#ifdef LT_ZAGE			/* > ============================================== < */
		  /* >  update ZAGE                                   < */
		  zage_term =
		    (cosmic_time - All.Time_Age) * metal_add_mass * wfac / P[j].Mass;
#ifdef LT_LOGZAGE
		  zage_term = log10(zage_term);
#endif
		  SphP[j].ZAge += zage_term;
		  SphP[j].ZAgeW += metal_add_mass * wfac / P[j].Mass;
#endif
#ifdef LT_ZAGE_LLV		/* > ============================================== < */
		  /* >  update ZAGE for Fe                            < */
		  zage_term =
		    (cosmic_time - All.Time_Age) * llvmetal_add_mass * wfac / P[j].Mass;
#ifdef LT_LOGZAGE
		  zage_term = log10(zage_term);
#endif
		  SphP[j].ZAge_llv += zage_term;
		  SphP[j].ZAgeW_llv += llvmetal_add_mass * wfac / P[j].Mass;

/*                   SphP[j].ZAge += (cosmic_time - All.Time_Age) * get_metallicity(j, -1); */
/*                   SphP[j].ZAgeW += get_metallicity(j, -1); */

#endif

		  /* > ============================================== < */
		  /* >  FEEDBACK                                      < */
		  /* ===============================================================
		   *
		   * a feedback form
		   * ejecta from sn (not from agb!) are put in the hot phase of the
		   * gas; thie means:
		   *  (1) the current intrinsic energy is update supposing that
		   *      the added mass has the same erg/g than the hot phase
		   *  (2) the entropy or the entropy change rate is update accordingly
		   *
		   * also, you can choose to put the ejecta into the hot phase with
		   * some their own specific energy, either using the specific energy
		   * of the supernovae (1^51 erg / ejecta mass for each SN) or a
		   * specific energy that you specify in paramfile. These two options
		   * correspond to switching on either LT_SNegy_IN_HOTPHASE or
		   * LT_HOT_EJECTA.
		   *
		   * =============================================================== */

#ifdef LT_EJECTA_IN_HOTPHASE
		  /* calculate the current specific energy */
		  if(P[j].Ti_endstep == All.Ti_Current)
		    {
		      /* if this is an active particle, calculate from the end-step quantities */
		      dt = (P[j].Ti_endstep - P[j].Ti_begstep) * All.Timebase_interval;
		      current_egy = DMAX(All.MinEgySpec, (SphP[j].Entropy + SphP[j].e.DtEntropy * dt) /
					 GAMMA_MINUS1 * pow(SphP[j].d.Density * a3inv, GAMMA_MINUS1));
		    }
		  else
		    {
		      dt = 0;
		      current_egy =
			SphP[j].Entropy / GAMMA_MINUS1 * pow(SphP[j].d.Density * a3inv, GAMMA_MINUS1);
		    }

		  /* calculate the current specific energy of the hot phase */
		  current_egyhot = (current_egy - All.EgySpecCold * SphP[j].x) / (1 - SphP[j].x);
		  hotmass = P[j].Mass * (1 - SphP[j].x);

		  /* update the mass and the cold fraction */
		  if(SphP[j].x > 0)
		    {
		      x_ratio = SphP[j].x;
#ifndef LT_LOCAL_IRA
		      if(dist == 0 && !Type && mode == 0)
			/* the particle itself; should be sufficient dist == 0 */
			{
			  SphP[j].x = (SphP[j].x * P[j].Mass - add_mass) /
			    (P[j].Mass - (1 - wfac) * add_mass);
			  SphP[j].MassRes -= (1 - wfac) * add_mass;
			}
		      else
#endif
			{
			  SphP[j].x *= P[j].Mass / (P[j].Mass + add_mass * wfac);
			  SphP[j].MassRes += add_mass * wfac;
			}

		      x_ratio /= SphP[j].x;
		    }
		  else
		    {
		      x_ratio = 0;
#ifndef LT_LOCAL_IRA
		      if(dist == 0 && !Type && mode == 0)
			/* the particle itself; should be sufficient dist == 0 */
			SphP[j].MassRes -= (1 - wfac) * add_mass;
		      else
#endif
			SphP[j].MassRes += add_mass * wfac;
		    }

#if defined(LT_HOT_EJECTA) || defined(LT_SNegy_IN_HOTPHASE)
		  v = (add_mass - LMmass) * wfac / (hotmass + add_mass * wfac);
#ifdef LT_HOT_EJECTA
		  current_egyhot = current_egyhot * (1 - v) + All.EgySpecEjecta * v;
#endif
#ifdef LT_SNegy_IN_HOTPHASE
		  current_egyhot = current_egyhot * (1 - v) + sn_spec_egy * v;
#endif
#endif
		  egy_ratio = current_egy;
		  current_egy = current_egyhot * (1 - SphP[j].x) + All.EgySpecCold * SphP[j].x;
		  egy_ratio /= current_egy;
#if defined(LT_SEv_INFO)
		  tot_spreadegy += current_egyhot * wfac;
#endif



		  if(dt > 0 && SphP[j].e.DtEntropy != 0)
		    {
		      SphP[j].e.DtEntropy =
			(current_egy * GAMMA_MINUS1 / pow(SphP[j].d.Density * a3inv, GAMMA_MINUS1) -
			 SphP[j].Entropy) / dt;
		      if(SphP[j].e.DtEntropy < -0.5 * SphP[j].Entropy / dt)
			SphP[j].e.DtEntropy = -0.5 * SphP[j].Entropy / dt;
		    }
		  else
		    SphP[j].Entropy =
		      current_egy * GAMMA_MINUS1 / pow(SphP[j].d.Density * a3inv, GAMMA_MINUS1);

		  /* ===============================================================
		   *
		   * end of feedback
		   * =============================================================== */

#else /*  LT_EJECTA_IN_HOTPHASE */
		  /* > ============================================== < */
		  /* >  NORMAL FEEDBACK                               < */

		  SphP[j].EgyRes += energy * wfac;

		  /* here we update the mass of the receveing particle if LT_EJECTA_IN_HOTPHASE
		     has not been used */

#ifndef LT_LOCAL_IRA
		  if(dist == 0 && !Type && mode == 0)
		    /* the particle itself; should be sufficient dist == 0 */
		    SphP[j].MassRes -= (1 - wfac) * add_mass;
		  else
#endif
		    SphP[j].MassRes += add_mass * wfac;
#endif /* > ============================================== < */
		  /* >  end of feedback                               < */


		  /* > ============================================== < */
		  /* >  collect INFOS                                 < */
#ifdef LT_SEv_INFO
#if defined(LT_EJECTA_IN_HOTPHASE) && !defined(LT_SNegy_IN_HOTPHASE)
		  if(energy > 0)
		    {
		      tot_spreadegy *= add_mass / energy;
		      tot_agbspreadegy = tot_spreadegy * agb_frac;

		      SpreadEgy[0] += 1;
		      SpreadEgy[1] += tot_spreadegy;

		      if(SpreadMinMaxEgy[0][0] > tot_spreadegy)
			{
			  SpreadMinMaxEgy[0][0] = tot_spreadegy;
			  SpreadMinMaxEgy[0][1] = tot_agbspreadegy;
			}
		      if(SpreadMinMaxEgy[1][0] < tot_spreadegy)
			{
			  SpreadMinMaxEgy[1][0] = tot_spreadegy;
			  SpreadMinMaxEgy[1][1] = tot_agbspreadegy;
			}

		      AgbFrac[0] += tot_agbspreadegy;
		      if(AgbFrac[1] > tot_agbspreadegy)
			AgbFrac[1] = tot_agbspreadegy;
		      if(AgbFrac[2] < tot_agbspreadegy)
			AgbFrac[2] = tot_agbspreadegy;

		      SpecEgyChange[0] += egy_ratio;
		      if(SpecEgyChange[1] > egy_ratio)
			SpecEgyChange[1] = egy_ratio;
		      if(SpecEgyChange[2] < egy_ratio)
			SpecEgyChange[2] = egy_ratio;

		      if(x_ratio > 0)
			{
			  CFracChange[0] += 1;
			  CFracChange[1] += x_ratio;
			  if(CFracChange[2] > x_ratio)
			    CFracChange[2] = x_ratio;
			  if(CFracChange[3] < x_ratio)
			    CFracChange[3] = x_ratio;
			}
		    }
#endif

		  if(SEvInfo_grain > SEvInfo_GRAIN)
		    {
		      egyfrac = energy * wfac /
			(SphP[j].Entropy / GAMMA_MINUS1 *
			 pow(SphP[j].d.Density * All.Time * All.Time * All.Time, GAMMA_MINUS1));

		      if(egyfrac > 0 && egyfrac < Stat_min[MIN_egyf])
			Stat_min[MIN_egyf] = egyfrac;
		      else if(egyfrac > Stat_max[MAX_egyf])
			Stat_max[MAX_egyf] = egyfrac;
		      if(Stat_min[MIN_egyf] == MIN_INIT_VALUE)
			Stat_min[MIN_egyf] = 0;
		      Stat_sum[MEAN_egyf] += egyfrac;
		      Stat_sum[NUM_egyf]++;
		    }

#endif

		}
	    }
	}			/* closes inner while */
      if(mode == 1)
	{
	  listindex++;
	  if(listindex < NODELISTLENGTH)
	    {
	      startnode = MetalDataGet[target].NodeList[listindex];
	      if(startnode >= 0)
		startnode = Nodes[startnode].u.d.nextnode;	/* open it */
	    }
	}
    }				/* closes outer while */


#ifdef LT_SEvDbg
  if(ID)
    printf("[SEvDBG] @ %g %d particle has got a weight=%g and ngb=%g from task %d\n",
	   All.Time, All.NumCurrentTiStep, weight_sum, ngb_sum, ThisTask);
  fflush(stdout);
#endif
  return numngb_inbox;
}



/* / ---------------------------------------------- \
 * |                                                |
 * |   here we include lt_sn_calc.c                 |
 * |   it contains all the routines that actually   |
 * |   calculate the produced elements              |
 * |                                                |
 * |   don't include it in the OBJS section of the  |
 * |   Makefile                                     |
 * |                                                |
 * | SECTION III                                    |
 * |                                                |
 * | ........................                       |
 * |                                                |
 * | General Routines                               |
 * |                                                |
 * |    > get_SnIa_product                          |
 * |    > nRSnIa                                    |
 * |    > get_AGB_product                           |  
 * |    > get_SnII_product                          |
 * |    > nRSnII                                    |
 * |    > mRSnII                                    |
 * |    > zmRSnII                                   |
 * |    > ztRSnII                                   |
 * |    > ejectaSnII                                |
 * |                                                |
 * \ ---------------------------------------------- / */


#include "lt_sn_calc.c"

/* / ---------------------------------------------- \
 * |                                                |
 * | SECTION IV                                     |
 * |                                                |
 * | ........................                       |
 * |                                                |
 * | General Routines                               |
 * |                                                |
 * |    > initialize_star_lifetimes                 |
 * |    > dm_dt                                     |
 * |    > lifetime                                  |  
 * |    > dying_mass                                |
 * |    > sec_dist                                  |
 * |    > get_metallicity_stat                      |
 * |    > write_metallicity_stat                    |
 * |    > get_metals_sumcheck                       |
 * |                                                |
 * \ ---------------------------------------------- / */

void initialize_star_lifetimes(void)
     /*- initialize stellar lifetimes for given range of star masses -*/
{
  int i;

  All.mean_lifetime = lifetime(All.Mup);
  All.sup_lifetime = lifetime(All.MBms);
  All.inf_lifetime = All.sup_lifetime;
  if(ThisTask == 0)
    {
      printf("\nstellar mean lifetime:   mean (%5.4g Msun) = %g Gyrs\n\n", All.Mup, All.mean_lifetime);
      fflush(stdout);
    }

  for(i = 0; i < IMFs_dim; i++)
    {
      IMFs[i].inf_lifetime = lifetime(IMFs[i].MU);
      if(IMFs[i].Mm > All.MBms)
	IMFs[i].sup_lifetime = lifetime(IMFs[i].Mm);
      else
	IMFs[i].sup_lifetime = All.sup_lifetime;

      if(All.inf_lifetime > IMFs[i].inf_lifetime)
	All.inf_lifetime = IMFs[i].inf_lifetime;

      if(ThisTask == 0)
	{
	  printf("\n"
		 "   IMF %3d      inf (%5.4g Msun) = %g Gyrs\n"
		 "                sup (%5.4g Msun) = %g Gyrs\n",
		 i, IMFs[i].MU, IMFs[i].inf_lifetime, IMFs[i].Mm, IMFs[i].sup_lifetime);
	  fflush(stdout);
	}
    }

  if(ThisTask == 0)
    {
      printf("\nstellar mean lifetime:   mean (%5.4g Msun) = %g Gyrs\n\n", All.Mup, All.mean_lifetime);
      fflush(stdout);
    }
}


double INLINE_FUNC dm_dt(double m, double t)
{
  /* t is in Gyr */
  /* the last factor 1/agefact normalize in 1/yr: otherwise
     the results would be 1/Gyr */
  /*
     if(t > 0.0302233)
     return -0.37037 * m / (t - 0.012);
     else
     return -0.54054 * m / (t - 0.003);
   */

#ifdef LT_PM_LIFETIMES
  /* padovani & matteucci 1993 */
  if(t > 0.039765318659064693)
    return -m / t * (1.338 - 0.1116 * (9 + log10(t)));
  else
    return -0.45045045045 * m / (t - 0.003);
#endif

#ifdef LT_MM_LIFETIMES
  /* maeder & meynet 1989 */
  if(m <= 1.3)
    return -m / t / 0.6545;
  if(m > 1.3 && m <= 3)
    return -m / t / 3.7;
  if(m > 3 && m <= 7)
    return -m / t / 2.51;
  if(m > 7 && m <= 15)
    return -m / t / 1.78;
  if(m > 15 && m <= 53.054)
    return -m / t / 0.86;
  if(m > 53.054)
    return -0.54054054054 * m / (t - 0.003);
#endif

}

double INLINE_FUNC lifetime(double mass)
{
  /* calculates lifetime for a given mass  */
  /*                                       */
  /* padovani & matteucci (1993) approach  */
  /* move to gibson (1997) one             */
  /*                                       */
  /* mass is intended in solar units, life */
  /* time in Gyr                           */

  if(mass > 100)
    return 0;
  else if(mass < 0.6)
    return 160;			/* should be INF */


  /* padovani & matteucci 1993 */
#ifdef LT_PM_LIFETIMES
  if(mass <= 6.6)
    return pow(10, ((1.338 - sqrt(1.790 - 0.2232 * (7.764 - log10(mass)))) / 0.1116) - 9);
  else
    return 1.2 * pow(mass, -1.85) + 0.003;
#endif

  /*
     if(mass <= 8)
     return 5 * pow(mass, -2.7) + 0.012;
     else
     return 1.2 * pow(mass, -1.85) + 0.003;
   */

#ifdef LT_MM_LIFETIMES
  /* maeder & meynet 1989 */
  if(mass <= 1.3)
    return pow(10, -0.6545 * log10(mass) + 1);
  if(mass > 1.3 && mass <= 3)
    return pow(10, -3.7 * log10(mass) + 1.35);
  if(mass > 3 && mass <= 7)
    return pow(10, -2.51 * log10(mass) + 0.77);
  if(mass > 7 && mass <= 15)
    return pow(10, -1.78 * log10(mass) + 0.17);
  if(mass > 15 && mass <= 53.054)
    return pow(10, -0.86 * log10(mass) - 0.94);
  if(mass > 53.054)
    return 1.2 * pow(mass, -1.85) + 0.003;
#endif
}

double INLINE_FUNC dying_mass(double time)
{
  /* calculates mass dying at some time */
  /*                                    */
  /* time is time_in_Gyr                */

  if((time < All.inf_lifetime) || (time > All.sup_lifetime))
    return 0;

  /*
     if(time > 0.0302233)
     return pow((time - 0.012) / 5, -0.37037);
     else
     return pow((time - 0.003) / 1.2, -0.54054);
   */

#ifdef LT_PM_LIFETIMES
  /* padovani & matteucci 1993 */
  if(time > 0.039765318659064693)
    return pow(10, 7.764 - (1.79 - pow(1.338 - 0.1116 * (9 + log10(time)), 2)) / 0.2232);
  else
    return pow((time - 0.003) / 1.2, -1.0 / 1.85);
#endif

#ifdef LT_MM_LIFETIMES
  /* maeder & meynet 1989 */
  if(time >= 8.4221714076)
    return pow(10, (1 - log10(time)) / 0.6545);
  if(time < 8.4221714076 && time >= 0.38428316376)
    return pow(10, (1.35 - log10(time)) / 3.7);
  if(time < 0.38428316376 && time >= 0.044545508363)
    return pow(10, (0.77 - log10(time)) / 2.51);
  if(time < 0.044545508363 && time >= 0.01192772338)
    return pow(10, (0.17 - log10(time)) / 1.78);
  if(time < 0.01192772338 && time >= 0.0037734864318)
    return pow(10, -(0.94 + log10(time)) / 0.86);
  if(time < 0.0037734864318)
    return pow((time - 0.003) / 1.2, -0.54054);
#endif
}


double INLINE_FUNC sec_dist(double gamma, double mu)
{
  /* calculates secondary distribution function */
  /* for Sn type Ia                             */
  /* as far, we take gamma=2, so this function  */
  /* isn't used                                 */

  return pow(2, 1 + gamma) * (1 + gamma) * pow(mu, gamma);
}



#ifdef LT_SEvDbg
void get_metals_sumcheck(int mode)
{
  FILE *outstream;

#define star (LT_NMetP-1)
#define sum_gas (2 * star)
#define sum_star sum_gas + 1
#define sum sum_star + 1

  int i, j;
  double metals[2 * (LT_NMetP - 1) + 3], tot_metals[2 * (LT_NMetP - 1) + 3];

  for(i = 0; i < 2 * (LT_NMetP - 1) + 3; i++)
    metals[i] = tot_metals[i] = 0;

  for(i = 0; i < NumPart; i++)
    {
      if(P[i].Type == 0)
	{
	  for(j = 1; j < LT_NMetP; j++)
	    {
	      metals[j - 1] += SphP[i].Metals[j];
	      metals[sum_gas] += SphP[i].Metals[j];
	      metals[sum] += SphP[i].Metals[j];
	    }
	}
      else if(P[i].Type == 4)
	{
	  for(j = 1; j < LT_NMetP; j++)
	    {
	      metals[star + j - 1] += MetP[P[i].MetID].Metals[j];
	      metals[sum_star] += MetP[P[i].MetID].Metals[j];
	      metals[sum] += MetP[P[i].MetID].Metals[j];
	    }
	}
    }

  MPI_Reduce(&metals[0], &tot_metals[0], 2 * (LT_NMetP - 1) + 3, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  if(ThisTask == 0)
    {
      if(mode & 1)
	{
	  outstream = FdMetSumCheck;
	  mode &= ~1;
	}
      else
	outstream = stdout;

      fprintf(outstream, "%c %7.6e ", (mode) ? ((mode & 2) ? '>' : ((mode & 4) ? '<' : '@')) : ' ', All.Time);
      for(i = 0; i < 2 * (LT_NMetP - 1) + 3; i++)
	fprintf(outstream, "%9.7e ", tot_metals[i] / All.HubbleParam);
      fprintf(outstream, "\n");
    }

}
#endif



/* .........................................
   * ------------------------------------- *
   |                                       |
   |   Chemical Stepping                   |
   * ------------------------------------- *
*/


double INLINE_FUNC get_NextChemTime(double lifetime, int mySFi, int *bin)
     /*!<
        returns the next look-back Time (in Gyr) for the 
        chemical evolution. *bin will contain the ordinal
        number of the array's timestep in which the
        star currently lives.

        NOTE:: mySFi MUST be set to the appropriate value
      */
{
  int i;
  double diff;

  for(i = 1; i < Nsteps[mySFi]; i++)
    if(lifetime < SNtimesteps[mySFi][0][i + 1])
      break;

  if(bin != 0x0)
    *bin = i;

  if(i == Nsteps[mySFi])
    {
      if(bin != 0x0)
	*bin = Nsteps[mySFi] - 1;
      return FOREVER;
    }


  if((diff =
      (SNtimesteps[mySFi][0][i + 1] - lifetime) / (SNtimesteps[mySFi][0][i + 1] -
						   SNtimesteps[mySFi][0][i])) <= 0.4)
    /* if less than the 40% of the current chem timestep is left, use all the next timestep */
    diff = All.Time_Age - (SNtimesteps[mySFi][0][i + 2] - lifetime);
  else
    diff = All.Time_Age - (SNtimesteps[mySFi][0][i + 1] - lifetime);

  if(diff < 0)
    /* would mean at negative redshift, then set the last time right before z=0 */
    diff = 1e-4;

  return diff;
}


double INLINE_FUNC get_da_dota(double y, void *param)
     /* gives da/(da/dt) */
{
  double f;

  //#ifdef LT_USE_SYSTEM_HUBBLE_FUNCTION
  return 1 / (y * sqrt(All.Omega0 * y * y * y + All.OmegaLambda +
		       (1 - All.Omega0 - All.OmegaLambda) * y * y));
  //#else
  //return 1.0 / (y * hubble_function(y) / All.Hubble);
  //#endif

}

double INLINE_FUNC get_age(double a)
     /* gives the look-back time corresponding to the expansion factor a */
{
  static double sec_in_Gyr = (86400.0 * 365.0 * 1e9);
  double result, error;

  F.function = &get_da_dota;
  F.params = NULL;

  if((gsl_status =
      gsl_integration_qag(&F, 1, 1 / a, 1e-4, 1e-5, gsl_ws_dim, qag_INT_KEY, w, &result, &error)))
    {
      printf(">>>>> [%3d] qag integration error %d in get_age\n", ThisTask, gsl_status);
      endrun(LT_ERROR_INTEGRATION_ERROR);
    }

  return 1.0 / (HUBBLE * All.HubbleParam) * result / sec_in_Gyr;
}


/* :::........................................................................ */
/* *************************************************************************** */

#ifdef DOUBLEPRECISION
double INLINE_FUNC myfloor(double v)
{
  return floor(v);
}
#else
float INLINE_FUNC myfloor(double v)
{
  return floorf((float) v);
}
#endif



/* / ---------------------------------------------- \
 * |                                                |
 * | SECTION II                                     |
 * |                                                |
 * | ........................                       |
 * |                                                |
 * | Searching for Stars and Main Cycle             |
 * |                                                |
 * |    > count_evolving_stars                      |
 * |    > evolve_SN                                 |
 * |                                                |
 * \ ---------------------------------------------- / */


/* ...........................................................
   * ............................... *
   :                                 :
   :   Searching for Evolving Stars  :
   * ............................... *
*/


int is_chemically_active(int i)
{
#ifdef LT_EVOLVE_EVERYTIMESTEP
  double lifetime;
#endif

  if(P[i].Type > 0 && P[i].Type != 4)
    return 0;

#ifndef LT_LOCAL_IRA
  if(P[i].Type == 0)
    {
      if(SphP[i].mstar > 0)
	return 1;
      else
	return 0;
    }
#endif

  if(P[i].Type == 4)
    {
#ifdef LT_EVOLVE_EVERYTIMESTEP
      lifetime = get_age(P[i].StellarAge) - All.Time_Age;

      if((lifetime - MetP[I].LastChemTime) > All.MichChemTimeStep)
	flag = 1;
#endif

      if(TimeBinActive[MetP[P[i].MetID].ChemTimeBin] && P[i].StellarAge < (MyFloat) All.Time)
	/* if( (MetP[P[i].MetID].NextChemTime >= All.Time_Age) ) */
	/*  || ((All.Time_Age - MetP[I].NextChemTime) / (lifetime - MetP[I].LastChemTime) < 0.05) ) */
	/* condition 1 : 
	   the look-back time NextChemTime is larger than the present look-backtime
	   condition 2 :
	   the look-back time NextChemTime is lower than the present look-backtime, but
	   the difference between the twos is less than 0.05 of the requested chemical timestep
	 */
	return 1;
    }

  return 0;
}


void count_evolving_stars(int *num_of_stars, int *num_of_gas)
{
  int flag;
  int starsnum, gasnum, prev;
  long int i, num_tobe_done;

  NextChemActiveParticle = (int *) mymalloc("NextChemActiveParticle", NumPart * sizeof(int));

  FirstChemActiveParticle = -1;
  prev = -1;

  for(i = starsnum = gasnum = 0; i < NumPart; i++)
    {
      flag = is_chemically_active(i);

      if(flag)
	{
	  if(prev == -1)
	    FirstChemActiveParticle = i;
	  else
	    NextChemActiveParticle[prev] = i;
	  prev = i;

	  if(P[i].Type == 0)
	    gasnum++;
	  else
	    {
	      starsnum++;
	      /*
#ifdef LT_BH_GUESSHSML
	      if(MetP[P[i].MetID].weight == 0)
		{
		  printf
		    ("\n\t task %d has found a discrepancy between density and chemical queues for particle %d, ID = %llu  %g %g\n",
		     ThisTask, i, (long long unsigned) P[i].ID, P[i].StellarAge, All.Time);
		  endrun(912345);
		}
#endif
	      */
	    }

	  P[i].Type |= EVOLVE;
	}
    }
  if (prev >= 0) 
    NextChemActiveParticle[prev] = -1;

  *num_of_gas = gasnum;
  *num_of_stars = starsnum;

  return;
}


/* :::........................................................................ */
/* *************************************************************************** */


/* / ---------------------------------------------- \
 * |                                                |
 * | SECTION  V                                     |
 * |                                                |
 * | ........................                       |
 * |                                                |
 * | Initialization Routines                        |
 * |                                                |
 * |    > calculate_effective_yields                |
 * |    > calculate_FactorSN                        |
 * |    > init_SN                                   |  
 * |    > build_SN_Stepping                         |
 * |    > get_Egy_and_Beta                          |
 * |    > calculate_ShortLiving_related             |
 * |    > setup_SF_related                          |
 * |    > TestStellarEvolution                      |
 * |                                                |
 * \ ---------------------------------------------- / */



#include "lt_sn_init.c"


void TestStellarEvolution()
{
#define Mass_N 30
#define Z_N 20
  int i, j, k;
  double delta;
  double top_masses[Mass_N], bot_masses[Mass_N];
  double mymetals[LT_NMet], myenergy, numsn, Zstar;
  FILE *outfile;
  float save_mass;

  printf("Start testing..\n");

  top_masses[0] = 50;
  bot_masses[0] = 35;
  top_masses[Mass_N - 1] = 1.0;
  bot_masses[Mass_N - 1] = 0.8;

  for(i = 1; i < Mass_N - 1; i++)
    {
      top_masses[i] = top_masses[0] - (top_masses[0] - top_masses[Mass_N - 1]) / (Mass_N - 1) * i;
      bot_masses[i] = bot_masses[0] - (bot_masses[0] - bot_masses[Mass_N - 1]) / (Mass_N - 1) * i;
    }

  P[0].Type = 4;
  save_mass = P[0].Mass;
  P[0].Mass = 1.0;
  MetP[0].PID = 0;
  MetP[0].iMass = 1;
  MetP[0].Metals[0] = (1 - HYDROGEN_MASSFRAC);

  if((outfile = fopen("SE.dbg", "w")) == 0x0)
    {
      printf("it has been impossible to open file SE.dbg..\n");
      fflush(stdout);
      return;
    }

  delta = log10(0.2 / 2e-5) / (Z_N - 1);

  for(k = 0; k < Z_N; k++)
    {
      fflush(stdout);
      Zstar = 2e-5 * pow(10, delta * k);
      MetP[0].Metals[1] = Zstar / (1 + Zstar) * 0.76;


      for(i = 0; i <= Mass_N - 1; i++)
	{
	  if(UseSnII)
	    {
	      numsn = 0;
	      get_SnII_product(0, 0, &mymetals[0], &myenergy, bot_masses[i], top_masses[i], &numsn);
	      fprintf(outfile, "II 1 %10.8e %10.8e %10.8e %10.8e %10.8e :: %10.8e ",
		      Zstar, lifetime(top_masses[i]), lifetime(bot_masses[i]), bot_masses[i], top_masses[i],
		      numsn);
	      for(j = 0; j < LT_NMet; j++)
		fprintf(outfile, "%10.8e ", mymetals[j]);
	      fprintf(outfile, "\n");
	    }


	  if(UseSnIa)
	    {
	      get_SnIa_product(0, 0, &mymetals[0], &myenergy, lifetime(top_masses[i]),
			       lifetime(bot_masses[i]) - lifetime(top_masses[i]));
	      fprintf(outfile, "Ia 1 %10.8e %10.8e %10.8e %10.8e %10.8e :: 0 ", Zstar,
		      lifetime(top_masses[i]), lifetime(bot_masses[i]), bot_masses[i], top_masses[i]);
	      for(j = 0; j < LT_NMet; j++)
		fprintf(outfile, "%10.8e ", mymetals[j]);
	      fprintf(outfile, "\n");
	    }

	  if(UseAGB)
	    {
	      numsn = 0;
	      get_AGB_product(0, 0, &mymetals[0], bot_masses[i], top_masses[i], &numsn);
	      fprintf(outfile, "AGB 1 %10.8e %10.8e %10.8e %10.8e %10.8e :: %10.8e ",
		      Zstar, lifetime(top_masses[i]), lifetime(bot_masses[i]), bot_masses[i], top_masses[i],
		      numsn);
	      for(j = 0; j < LT_NMet; j++)
		fprintf(outfile, "%10.8e ", mymetals[j]);
	      fprintf(outfile, "\n");
	    }
	}
      fprintf(outfile, "#\n#\n");
    }
  fclose(outfile);

  P[0].Type = 0;
  P[0].Mass = save_mass;
  MetP[0].PID = 0;
  MetP[0].iMass = 0;
  MetP[0].Metals[0] = 0;

  return;
}



#ifdef LT_SEv_INFO

void write_metallicity_stat(void)
{
  int i, count, count_b;

#ifdef LT_SMOOTH_SIZE
  int tot_AvgSmoothN;
  double tot_AvgSmoothSize, tot_MinSmoothSize, tot_MaxSmoothSize;
  int tot_AvgSmoothNgb, tot_MinSmoothNgb, tot_MaxSmoothNgb;
#endif


#ifdef LT_SEvDbg
  get_metals_sumcheck(1);
#endif

  MPI_Reduce(&ALdata_min[0], &tot_ALdata_min[0], AL_INUM_min, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
  MPI_Reduce(&ALdata_max[0], &tot_ALdata_max[0], AL_INUM_max, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Reduce(&ALdata_sum[0], &tot_ALdata_sum[0], AL_INUM_sum, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  if(tot_starsnum > 1)
    {
      if(tot_starsnum > 3)
	{
	  tot_ALdata_sum[MEAN_sl] -= tot_ALdata_max[MAX_sl] + tot_ALdata_min[MIN_sl];
	  tot_ALdata_sum[MEAN_ngb] -= tot_ALdata_max[MAX_ngb] + tot_ALdata_min[MIN_ngb];
	  count = tot_starsnum - 2;
	}
      else
	count = tot_starsnum;
      tot_ALdata_sum[MEAN_sl] /= count;
      tot_ALdata_sum[MEAN_ngb] /= count;
    }

  get_metallicity_stat();

  MPI_Reduce(&Stat_sum[0], &tot_Stat_sum[0], S_INUM_sum, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&Stat_min[0], &tot_Stat_min[0], S_INUM_min, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
  MPI_Reduce(&Stat_max[0], &tot_Stat_max[0], S_INUM_max, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

#ifdef LT_SMOOTH_SIZE
  MPI_Reduce(&AvgSmoothN, &tot_AvgSmoothN, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

  MPI_Reduce(&AvgSmoothSize, &tot_AvgSmoothSize, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&MinSmoothSize, &tot_MinSmoothSize, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
  MPI_Reduce(&MaxSmoothSize, &tot_MaxSmoothSize, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

  MPI_Reduce(&AvgSmoothNgb, &tot_AvgSmoothNgb, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&MinSmoothNgb, &tot_MinSmoothNgb, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
  MPI_Reduce(&MaxSmoothNgb, &tot_MaxSmoothNgb, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
#endif

  if(ThisTask == 0)
    {
      /* metallicities */
      if(tot_Stat_sum[NUM_star] > 3)
	{
	  count = tot_Stat_sum[NUM_star];
	  count_b = tot_Stat_sum[NUM_star] - 2;
	  tot_Stat_sum[MEAN_Zstar] -= tot_Stat_max[MAX_Zstar] + tot_Stat_min[MIN_Zstar];
	}
      else if(tot_Stat_sum[NUM_star] == 0)
	count = count_b = 1;

      for(i = 0; i < LT_NMet; i++)
	{
	  tot_Stat_sum[MEAN_Zs + i] /= All.TotN_gas;
	  tot_Stat_sum[MEAN_Zsstar + i] /= count;
	}
      if(All.TotN_gas > 3)
	{
	  tot_Stat_sum[MEAN_Z] -= tot_Stat_max[MAX_Z] + tot_Stat_min[MIN_Z];
	  tot_Stat_sum[MEAN_Z] /= All.TotN_gas - 2;	/* assume that we have more than 3 gas particles! */
	}
      else
	tot_Stat_sum[MEAN_Z] /= All.TotN_gas;
      tot_Stat_sum[MEAN_Zstar] /= count_b;

      /* energy ratios */
      if(tot_Stat_sum[NUM_egyf] > 3)
	{
	  count = tot_Stat_sum[NUM_egyf];
	  count_b = tot_Stat_sum[NUM_egyf] - 2;
	  tot_Stat_sum[MEAN_egyf] -= tot_Stat_max[MAX_egyf] + tot_Stat_min[MIN_egyf];
	}
      else if(tot_Stat_sum[NUM_egyf] == 0)
	count = count_b = 1;
      tot_Stat_sum[MEAN_egyf] /= count_b;

      fprintf(FdMetals, "%9.7e ", All.Time);
      fprintf(FdMetals, "%9.7e %9.7e %9.7e %9.7e %9.7e %9.7e ",
	      tot_Stat_min[MIN_Z], tot_Stat_max[MAX_Z], tot_Stat_sum[MEAN_Z],
	      tot_Stat_min[MIN_Zstar], tot_Stat_max[MAX_Zstar], tot_Stat_sum[MEAN_Zstar]);
      for(i = 0; i < LT_NMet; i++)
	fprintf(FdMetals, "%9.7e %9.7e ", tot_Stat_sum[MEAN_Zs + i], tot_Stat_sum[MEAN_Zsstar + i]);
      fprintf(FdMetals, "%9.7e %9.7e %9.7e ", tot_Stat_min[MIN_egyf], tot_Stat_max[MAX_egyf],
	      tot_Stat_sum[MEAN_egyf]);
      fprintf(FdMetals, "%9.7e %9.7e %9.7e %9.7e %9.7e %9.7e ", tot_ALdata_min[MIN_sl],
	      tot_ALdata_max[MAX_sl], tot_ALdata_sum[MEAN_sl], tot_ALdata_min[MIN_ngb],
	      tot_ALdata_max[MAX_ngb], tot_ALdata_sum[MEAN_ngb]);

#ifdef LT_SMOOTH_SIZE
      fprintf(FdMetals, "%9.7e %9.7e %9.7e %9.7e %d %d ",
	      tot_AvgSmoothSize / tot_AvgSmoothN, tot_MinSmoothSize, tot_MaxSmoothSize,
	      (double) tot_AvgSmoothNgb / tot_AvgSmoothN, tot_MinSmoothNgb, tot_MaxSmoothNgb);
#endif
      fprintf(FdMetals, "\n");
      fflush(FdMetals);
    }
  MPI_Barrier(MPI_COMM_WORLD);
}

void get_metallicity_stat(void)
{
  int i, j;

  double Z, zmass, R;


  for(i = 0; i < NumPart; i++)
    {
      if(P[i].Type == 0)
	{
	  /* find total metal mass (exclude Helium) */
	  zmass = 0;
	  for(j = 0; j < LT_NMetP; j++)
	    {
	      if(j != Hel)
		zmass += SphP[i].Metals[j];
	    }
	  /* metallicities are calculated as metal_mass / hydrogen_mass */
	  R = 1.0 / (P[i].Mass - zmass - SphP[i].Metals[Hel]);
	  for(j = 0; j < LT_NMetP; j++)
	    Stat_sum[MEAN_Zs + j] += SphP[i].Metals[j] * R;

	  Stat_sum[MEAN_Z] += (Z = zmass * R);
	  if(Z < Stat_min[MIN_Z])
	    Stat_min[MIN_Z] = Z;
	  else if(Z > Stat_max[MAX_Z])
	    Stat_max[MAX_Z] = Z;
	}
      else if(P[i].Type == 4)
	{
	  Stat_sum[NUM_star]++;
	  /* find total metal mass (exclude Helium) */
	  zmass = 0;
	  for(j = 0; j < LT_NMetP; j++)
	    {
	      if(j != Hel)
		zmass += MetP[P[i].MetID].Metals[j];
	    }
	  /* metallicities are calculated as metal_mass / hydrogen_mass */
	  R = 1.0 / (MetP[P[i].MetID].iMass - zmass - MetP[P[i].MetID].Metals[Hel]);
	  for(j = 0; j < LT_NMetP; j++)
	    Stat_sum[MEAN_Zsstar + j] += MetP[P[i].MetID].Metals[j] * R;

	  Stat_sum[MEAN_Zstar] += (Z = zmass * R);
	  if(Z < Stat_min[MIN_Zstar])
	    Stat_min[MIN_Zstar] = Z;
	  else if(Z > Stat_max[MAX_Zstar])
	    Stat_max[MAX_Zstar] = Z;
	}
    }

  return;
}
#endif



void fsolver_error_handler(const char *reason, const char *file, int line, int err)
{
  if(err == GSL_EINVAL)
    {
      gsl_status = err;
      return;
    }
  return;
}

#include "lt_test_suite.c"

#endif

/*
 *
 * SCRATCH.NOTES AREA
 *

 *
 * END of SCRATCH.NOTES AREA
 *
 */

/*
 * here below and example of how to convert a time interval in code discretized time steps
 */

/* int INLINE_FUNC convert_time_to_timesteps(double start_a, double start_time, double delta_time) */
/* { */
/*   double delta_a; */

/*   /\* get delta_expansion_factor when moving from start_time to start_time + delta_time   * */
/*    * start_a is the expansion factor corresponding to start_time                         *\/ */
/*   delta_a = gsl_spline_eval(spline, cosmic_time - start_time + delta_time * 1.01, accel) - start_a; */

/*   /\* converts in code steps *\/ */
/*   return (int) (log(delta_a / start_a + 1) / All.Timebase_interval); */
/* } */
