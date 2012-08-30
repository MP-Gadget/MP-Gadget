
#include "f2c.h"

#ifdef CHEMCOOL

#ifndef ABHE
#define ABHE 0.079
#endif

#ifndef CHEMCOOL_H
#define CHEMCOOL_H

#define UNUSED 0

#if CHEMISTRYNETWORK == 1  /* Primordial (no metals) */
#define TRAC_NUM 6
#define NSPEC 7
#define IH2   0
#define IHP   1
#define IDP   2
#define IHD   3
#define IHEP  4
#define IHEPP 5
#define ITMP  6
#define IC    UNUSED
#define ISi   UNUSED
#define IO    UNUSED
#define ICO   UNUSED
#define IC2   UNUSED
#define IOH   UNUSED
#define IH2O  UNUSED
#define IO2   UNUSED
#define IHCOP UNUSED
#define ICH   UNUSED
#define ICH2  UNUSED
#define ISIPP UNUSED
#define ICH3P UNUSED
#define IMGP  UNUSED
#define ITD   UNUSED
#define ICH3  UNUSED
#define ICH4  UNUSED
#define ICO2  UNUSED
#define ICOS  UNUSED
#define IH2OS UNUSED
#define IO2S  UNUSED
#endif

#if CHEMISTRYNETWORK == 2  /* Low metallicity network, no CO, OH, H2O */
#define TRAC_NUM 10
#define NSPEC 11
#define IH2   0
#define IHP   1
#define IC    2
#define ISi   3
#define ISIPP 4
#define IO    5
#define IDP   6
#define IHD   7
#define IHEP  8
#define IHEPP 9
#define ITMP  10
#define ICO   UNUSED
#define IC2   UNUSED
#define IOH   UNUSED
#define IH2O  UNUSED
#define IO2   UNUSED
#define IHCOP UNUSED
#define ICH   UNUSED
#define ICH2  UNUSED
#define ICH3P UNUSED
#define IMGP  UNUSED
#define ITD   UNUSED
#define ICH3  UNUSED
#define ICH4  UNUSED
#define ICO2  UNUSED
#define ICOS  UNUSED
#define IH2OS UNUSED
#define IO2S  UNUSED
#endif

#if CHEMISTRYNETWORK == 3  /* Full low metallicity network */
#define TRAC_NUM 19
#define NSPEC 20
#define IH2   0
#define IHP   1
#define IC    2
#define ISi   3
#define IO    4
#define IDP   5
#define IHD   6
#define IHEP  7
#define IHEPP 8
#define ICO   9
#define IC2   10
#define IOH   11
#define IH2O  12
#define IO2   13
#define IHCOP 14
#define ICH   15
#define ICH2  16
#define ISIPP 17
#define ICH3P 18
#define ITMP  19
#define IMGP  UNUSED
#define ITD   UNUSED
#define ICH3  UNUSED
#define ICH4  UNUSED
#define ICO2  UNUSED
#define ICOS  UNUSED
#define IH2OS UNUSED
#define IO2S  UNUSED
#endif

#if CHEMISTRYNETWORK == 4  /* GMC formation - simple H2, H+ network */
#define TRAC_NUM 2
#define NSPEC 3
#define IH2   0
#define IHP   1
#define ITMP  2
#define IDP   UNUSED
#define IHD   UNUSED
#define IHEP  UNUSED
#define IHEPP UNUSED
#define IC    UNUSED
#define ISi   UNUSED
#define IO    UNUSED
#define ICO   UNUSED
#define IC2   UNUSED
#define IOH   UNUSED
#define IH2O  UNUSED
#define IO2   UNUSED
#define IHCOP UNUSED
#define ICH   UNUSED
#define ICH2  UNUSED
#define ISIPP UNUSED
#define ICH3P UNUSED
#define IMGP  UNUSED
#define ITD   UNUSED
#define ICH3  UNUSED
#define ICH4  UNUSED
#define ICO2  UNUSED
#define ICOS  UNUSED
#define IH2OS UNUSED
#define IO2S  UNUSED
#endif

#if CHEMISTRYNETWORK == 5  /* As 4, but with approx. CO formation from NL97 */
#define TRAC_NUM 3
#define NSPEC 4
#define IH2   0
#define IHP   1
#define ICO   2
#define ITMP  3
#define IDP   UNUSED
#define IHD   UNUSED
#define IHEP  UNUSED
#define IHEPP UNUSED
#define IC    UNUSED
#define ISi   UNUSED
#define IO    UNUSED
#define IC2   UNUSED
#define IOH   UNUSED
#define IH2O  UNUSED
#define IO2   UNUSED
#define IHCOP UNUSED
#define ICH   UNUSED
#define ICH2  UNUSED
#define ISIPP UNUSED
#define ICH3P UNUSED
#define IMGP  UNUSED
#define ITD   UNUSED
#define ICH3  UNUSED
#define ICH4  UNUSED
#define ICO2  UNUSED
#define ICOS  UNUSED
#define IH2OS UNUSED
#define IO2S  UNUSED
#endif

#if CHEMISTRYNETWORK == 6  /* As 5, but with CO depletion/desorption */
#define TRAC_NUM 4
#define NSPEC 5
#define IH2   0
#define IHP   1
#define ICO   2
#define ICOS  3
#define ITMP  4
#define IDP   UNUSED
#define IHD   UNUSED
#define IHEP  UNUSED
#define IHEPP UNUSED
#define IC    UNUSED
#define ISi   UNUSED
#define IO    UNUSED
#define IC2   UNUSED
#define IOH   UNUSED
#define IH2O  UNUSED
#define IO2   UNUSED
#define IHCOP UNUSED
#define ICH   UNUSED
#define ICH2  UNUSED
#define ISIPP UNUSED
#define ICH3P UNUSED
#define IMGP  UNUSED
#define ITD   UNUSED
#define ICH3  UNUSED
#define ICH4  UNUSED
#define ICO2  UNUSED
#define IH2OS UNUSED
#define IO2S  UNUSED
#endif

#if CHEMISTRYNETWORK == 7  /* Full GMC network */
#define TRAC_NUM 14
#define NSPEC 15
#define IH2   0
#define IHP   1
#define IC    2
#define IO    3
#define IHEP  4
#define ICO   5
#define IC2   6
#define IOH   7
#define IH2O  8
#define IO2   9
#define IHCOP 10
#define ICH   11
#define ICH2  12
#define ICH3P 13
#define ITMP  14
#define ISi   UNUSED
#define IDP   UNUSED
#define IHD   UNUSED
#define ISIPP UNUSED
#define IMGP  UNUSED
#define IHEPP UNUSED
#define ITD   UNUSED
#define ICH3  UNUSED
#define ICH4  UNUSED
#define ICO2  UNUSED
#define ICOS  UNUSED
#define IH2OS UNUSED
#define IO2S  UNUSED
#endif

#if CHEMISTRYNETWORK == 8  /* Highly simplified primordial network, for high n sims. */
#define TRAC_NUM 3
#define NSPEC    4
#define IH2   0
#define IHD   1
#define IHP   2
#define ITMP  3
#define IC    UNUSED
#define IO    UNUSED
#define IHEP  UNUSED
#define ICO   UNUSED
#define IC2   UNUSED
#define IOH   UNUSED
#define IH2O  UNUSED
#define IO2   UNUSED
#define IHCOP UNUSED
#define ICH   UNUSED
#define ICH2  UNUSED
#define ICH3P UNUSED
#define ISi   UNUSED
#define IDP   UNUSED
#define ISIPP UNUSED
#define IMGP  UNUSED
#define IHEPP UNUSED
#define ITD   UNUSED
#define ICH3  UNUSED
#define ICH4  UNUSED
#define ICO2  UNUSED
#define ICOS  UNUSED
#define IH2OS UNUSED
#define IO2S  UNUSED
#endif

#if CHEMISTRYNETWORK == 9  /* Reduced network from Omukai et al 2005 */
#define TRAC_NUM 15
#define NSPEC    16
#define IH2   0
#define IHP   1
#define IHD   2
#define IDP   3
#define IC    4
#define IO    5
#define IOH   6
#define IH2O  7
#define ICO   8
#define IO2   9
#define ICH   10
#define ICH2  11
#define ICH3  12
#define ICH4  13
#define ICO2  14
#define ITMP  15
#define IHEP  UNUSED
#define IC2   UNUSED
#define IHCOP UNUSED
#define ICH3P UNUSED
#define ISi   UNUSED
#define ISIPP UNUSED
#define IMGP  UNUSED
#define IHEPP UNUSED
#define ITD   UNUSED
#define ICOS  UNUSED
#define IH2OS UNUSED
#define IO2S  UNUSED
#endif

#if CHEMISTRYNETWORK == 10  /* Simplified low metallicity network, for high n sims. */
#define TRAC_NUM 8
#define NSPEC    9
#define IH2   0
#define IHD   1
#define IHP   2
#define IOH   3
#define IH2O  4
#define IO2   5
#define ICO   6
#define ITD   7
#define ITMP  8
#define IC    UNUSED
#define IO    UNUSED
#define IHEP  UNUSED
#define IC2   UNUSED
#define IHCOP UNUSED
#define ICH   UNUSED
#define ICH2  UNUSED
#define ICH3P UNUSED
#define ISi   UNUSED
#define IDP   UNUSED
#define ISIPP UNUSED
#define IMGP  UNUSED
#define IHEPP UNUSED
#define ICH3  UNUSED
#define ICH4  UNUSED
#define ICO2  UNUSED
#define ICOS  UNUSED
#define IH2OS UNUSED
#define IO2S  UNUSED
#endif

#if CHEMISTRYNETWORK == 11  /* Full GMC network + solid phase */
#define TRAC_NUM 17
#define NSPEC 18
#define IH2   0
#define IHP   1
#define IC    2
#define IO    3
#define IHEP  4
#define ICO   5
#define IC2   6
#define IOH   7
#define IH2O  8
#define IO2   9
#define IHCOP 10
#define ICH   11
#define ICH2  12
#define ICH3P 13
#define ICOS  14
#define IH2OS 15
#define IO2S  16
#define ITMP  17
#define ISi   UNUSED
#define IDP   UNUSED
#define IHD   UNUSED
#define ISIPP UNUSED
#define IMGP  UNUSED
#define IHEPP UNUSED
#define ITD   UNUSED
#define ICH3  UNUSED
#define ICH4  UNUSED
#define ICO2  UNUSED
#endif

#define  ABUND_NUM 16

#define NSTEP 300000

#define DTCOOL_SCALE_FACTOR 0.1
#define ENTROPY_TIMESTEP_FACTOR 0.2

#ifdef FF_COLLAPSE_TEST
#define  NRPAR 6
#else
#ifdef RAYTRACE
#ifdef CO_SHIELDING
#define  NRPAR 21
#else /* CO_SHIELDING */
#define  NRPAR 15
#endif /* CO_SHIELDING */
#else /* RAYTRACE */
#define  NRPAR 7
#endif /* RAYTRACE */
#endif /* FF_COLLAPSE_TEST */
#define  NIPAR 2
#define  RTOL      1d-4
#define  ATOL_H2   1d-7
#define  ATOL_HP   1d-12
#define  ATOL_C    1d-16
#define  ATOL_SI   1d-16
#define  ATOL_O    1d-16
#define  ATOL_DP   1d-14
#define  ATOL_HD   1d-12
#define  ATOL_HEP  1d-14
#define  ATOL_HEPP 1d-14
#define  ATOL_CO   1d-14
#define  ATOL_C2   1d-14
#define  ATOL_OH   1d-14
#define  ATOL_H2O  1d-14
#define  ATOL_O2   1d-14
#define  ATOL_HCOP 1d-18
#define  ATOL_CH   1d-18
#define  ATOL_CH2  1d-18
#define  ATOL_SIPP 1d-14
#define  ATOL_CH3P 1d-18
#define  ATOL_MGP  1d-17
#define  ATOL_CH3  1d-18
#define  ATOL_CH4  1d-18
#define  ATOL_CO2  1d-18
#define  ATOL_COS  1d-14
#define  ATOL_H2OS 1d-14
#define  ATOL_O2S  1d-14
#define  ATOL_TMP  0d0
#define  ATOL_TD   1d-2

#define  EPS_MAX   1d-4

#define  NMD 10000
#define  NRATES 28
#if CHEMISTRYNETWORK == 1
#define  NRATES_CHEM 20
#endif
#if CHEMISTRYNETWORK == 2
#define  NRATES_CHEM 23
#endif
#if CHEMISTRYNETWORK == 3 
#define  NRATES_CHEM 23
#endif
#if CHEMISTRYNETWORK == 7 || CHEMISTRYNETWORK == 11
#define  NRATES_CHEM 22
#endif
#if CHEMISTRYNETWORK == 4 || CHEMISTRYNETWORK == 5 || CHEMISTRYNETWORK == 6
#define  NRATES_CHEM 6
#endif
#if CHEMISTRYNETWORK == 8 || CHEMISTRYNETWORK == 10
#define  NRATES_CHEM 5
#endif
#if CHEMISTRYNETWORK == 9
#define  NRATES_CHEM 5
#endif
#define  NRATES_TOT NRATES+NRATES_CHEM
#define  NH2DATA 41

#define  COOL_EPS 1d-6
#define  NCLTAB 67
#define  NCHTAB 146
#define  NCRTAB 12
#define  NCRPHOT 11
#define  NPHTAB 47
#define  NPR 47
#define  NCONST 80
#define  ELECTRON_VOLT 1.60219e-12
#define  BOLTZMANN   1.38066e-16
#define  PROTONMASS  1.6726e-24

#ifndef PI
#define  PI          3.14159265358979323846
#endif

#ifndef GRAVITY
#define  GRAVITY           6.672e-8   
#endif


#ifdef RAYTRACE
#define NCOL 128
#endif

#endif /* CHEMCOOL_H */
#endif /* CHEMCOOL */
