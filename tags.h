
#define TAG_N             10      /*!< Various tags used for labelling MPI messages */ 
#define TAG_HEADER        11
#define TAG_PDATA         12
#define TAG_SPHDATA       13
#define TAG_KEY           14
#define TAG_DMOM          15
#define TAG_NODELEN       16
#define TAG_HMAX          17
#define TAG_GRAV_A        18
#define TAG_GRAV_B        19
#define TAG_DIRECT_A      20
#define TAG_DIRECT_B      21
#define TAG_HYDRO_A       22 
#define TAG_HYDRO_B       23
#define TAG_NFORTHISTASK  24
#define TAG_PERIODIC_A    25
#define TAG_PERIODIC_B    26
#define TAG_PERIODIC_C    27
#define TAG_PERIODIC_D    28
#define TAG_NONPERIOD_A   29 
#define TAG_NONPERIOD_B   30
#define TAG_NONPERIOD_C   31
#define TAG_NONPERIOD_D   32
#define TAG_POTENTIAL_A   33
#define TAG_POTENTIAL_B   34
#define TAG_DENS_A        35
#define TAG_DENS_B        36
#define TAG_LOCALN        37
#define TAG_BH_A          38
#define TAG_BH_B          39
#define TAG_SMOOTH_A      40
#define TAG_SMOOTH_B      41
#define TAG_ENRICH_A      42
#define TAG_CONDUCT_A     43
#define TAG_CONDUCT_B     44
#define TAG_FOF_A         45
#define TAG_FOF_B         46
#define TAG_FOF_C         47
#define TAG_FOF_D         48
#define TAG_FOF_E         49
#define TAG_FOF_F         50
#define TAG_FOF_G         51
#define TAG_HOTNGB_A      52
#define TAG_HOTNGB_B      53
#define TAG_SWAP          54
#define TAG_PM_FOLD       55

#ifdef BG_SFR
#define TAG_STARDATA      60
#endif

#define TAG_PDATA_SPH     70
#define TAG_KEY_SPH       71
#define TAG_BHDATA        75

#ifdef RADTRANSFER
#define TAG_RT_A          72
#define TAG_RT_B          73
#endif

#ifdef LT_STELLAREVOLUTION
#define TAG_PDATA_STARS   100
#define TAG_KEY_STARS     (TAG_PDATA_STARS + 1)
#define TAG_METDATA       (TAG_KEY_STARS + 1)
#define TAG_SE            (TAG_METDATA + 1)
#define TAG_TRCK          (TAG_SE + 1)
#define TAG_CCRIT         (TAG_TRCK + 1)


#endif
