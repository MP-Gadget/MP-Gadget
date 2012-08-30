#include <math.h>



#define N_data_dumps   50
#define N_pts_bound    7
#define N_nu_base      200
#define N_nu_bound     12
#define N_T            1000
#define N_nu           N_nu_base + N_nu_bound*N_pts_bound /*  284 */


/* ----- Set various cutoff values in eV. */

#define  e24     13.6
#define  e25     54.4
#define  e26     24.6
#define  e27     0.755
#define  e28a    2.65
#define  e28b    11.27
#define  e28c    21.0
#define  e29a    15.42
#define  e29b    16.5
#define  e29c    17.7
#define  e30a    30.0
#define  e30b    70.0


/* ----- Declare some control parameters. */

#define H_flag                1/*   .TRUE. */
#define He_flag               1/*   .TRUE. */
#define H2_flag               1/*   .TRUE. */
#if defined (UM_CHEMISTRY) && defined (UM_HD_COOLING)
#define D_flag                1/*   .TRUE. */
#endif
#define species_flag          1/*   .TRUE. */
#define opt_thin_flag         1/*   .TRUE. */
#define rad_flag              1/*   .TRUE. */
#define energy_flag           1/*   .TRUE. */
#define H2_shield_flag        1/*   .TRUE. */


#define  k1_flag    H_flag
#define  k2_flag    H_flag
#define  k3_flag    He_flag
#define  k4_flag    He_flag
#define  k5_flag    He_flag
#define  k6_flag    He_flag
#define  k7_flag    H_flag
#define  k8_flag    H_flag * H2_flag
#define  k9_flag    H_flag * H2_flag
#define  k10_flag    H_flag * H2_flag
#define  k11_flag    H_flag * H2_flag
#define  k12_flag    H_flag * H2_flag
#define  k13_flag    H_flag * H2_flag
#define  k14_flag    H_flag * H2_flag
#define  k15_flag    H_flag * H2_flag
#define  k16_flag    H_flag * H2_flag
#define  k17_flag    H_flag * H2_flag
#define  k18_flag    H_flag * H2_flag
#define  k19_flag    H_flag * H2_flag
#define  k20_flag    H_flag * H2_flag
#define  k21_flag    H_flag * H2_flag
#define  k24_flag    H_flag * rad_flag
#define  k25_flag    He_flag * rad_flag
#define  k26_flag    He_flag * rad_flag
#define  k27_flag    H_flag * H2_flag * rad_flag
#define  k28_flag    H_flag * H2_flag * rad_flag
#define  k29_flag    H_flag * H2_flag * rad_flag
#define  k30_flag    H_flag * H2_flag * rad_flag
#define  k31_flag    H_flag * H2_flag * rad_flag


