#ifndef METAL_TABLES_H
#define METAL_TABLES_H

/* Metals followed:
 * H, He, C, N, O, Ne, Mg, Si, Fe (11, following 1703.02970)
 */
#define NSPECIES 9
/* Metallicity values (in terms of metal yield, not solar metallicity)
 * for the stellar lifetime table. Columns of lifetime.*/
#define LIFE_NMET 5
#define LIFE_NMASS 30
static const double lifetime_metallicity[LIFE_NMET] = { 0.0004 , 0.004 , 0.008, 0.02, 0.05 };
/* Mass values in solar masses for the stellar lifetime table. Rows of lifetime*/
static const double lifetime_masses[LIFE_NMASS] = {0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5,
    1.6, 1.7, 1.8, 1.9, 2.0, 2.5, 3 , 4 , 5 , 6 , 7 , 9 , 12 , 15 , 20 , 30 , 40 , 60 , 100, 120};
/* Stellar lifetimes as a function of mass and metallicity in years.
 * Table 14 of Portinari et al, astro-ph/9711337 */
static const double lifetime[LIFE_NMASS*LIFE_NMET] = {
4.28e+10,   5.35E+10,   6.47E+10,   7.92E+10,   7.18E+10,
2.37E+10,   2.95E+10,   3.54E+10,   4.45E+10,   4.00E+10,
1.41E+10,   1.73E+10,   2.09E+10,   2.61E+10,   2.33E+10,
8.97E+09,   1.09E+10,   1.30E+10,   1.59E+10,   1.42E+10,
6.03E+09,   7.13E+09,   8.46E+09,   1.03E+10,   8.88E+09,
4.23E+09,   4.93E+09,   5.72E+09,   6.89E+09,   5.95E+09,
3.08E+09,   3.52E+09,   4.12E+09,   4.73E+09,   4.39E+09,
2.34E+09,   2.64E+09,   2.92E+09,   3.59E+09,   3.37E+09,
1.92E+09,   2.39E+09,   2.36E+09,   2.87E+09,   3.10E+09,
1.66E+09,   1.95E+09,   2.18E+09,   2.64E+09,   2.51E+09,
1.39E+09,   1.63E+09,   1.82E+09,   2.18E+09,   2.06E+09,
1.18E+09,   1.28E+09,   1.58E+09,   1.84E+09,   1.76E+09,
1.11E+09,   1.25E+09,   1.41E+09,   1.59E+09,   1.51E+09,
9.66E+08,   1.23E+09,   1.25E+09,   1.38E+09,   1.34E+09,
8.33E+08,   1.08E+09,   1.23E+09,   1.21E+09,   1.24E+09,
4.64E+08,   5.98E+08,   6.86E+08,   7.64E+08,   6.58E+08,
3.03E+08,   3.67E+08,   4.12E+08,   4.56E+08,   3.81E+08,
1.61E+08,   1.82E+08,   1.93E+08,   2.03E+08,   1.64E+08,
1.01E+08,   1.11E+08,   1.15E+08,   1.15E+08,   8.91E+07,
7.15E+07,   7.62E+07,   7.71E+07,   7.45E+07,   5.67E+07,
5.33E+07,   5.61E+07,   5.59E+07,   5.31E+07,   3.97E+07,
3.42E+07,   3.51E+07,   3.44E+07,   3.17E+07,   2.33E+07,
2.13E+07,   2.14E+07,   2.10E+07,   1.89E+07,   1.39E+07,
1.54E+07,   1.52E+07,   1.49E+07,   1.33E+07,   9.95E+06,
1.06E+07,   1.05E+07,   1.01E+07,   9.15E+06,   6.99E+06,
6.90E+06,   6.85E+06,   6.65E+06,   6.13E+06,   5.15E+06,
5.45E+06,   5.44E+06,   5.30E+06,   5.12E+06,   4.34E+06,
4.20E+06,   4.19E+06,   4.15E+06,   4.12E+06,   3.62E+06,
3.32E+06,   3.38E+06,   3.44E+06,   3.39E+06,   3.11E+06,
3.11E+06,   3.23E+06,   3.32E+06,   3.23E+06,   3.11E+06};

/* Sn1a yields from the W7 model of Nomoto et al 1997 https://arxiv.org/abs/astro-ph/9706025
 * I extracted this from the latex source of their table 1 by hand.
 * total_metals is just the sum of all metal masses in the table.
 */
static const double sn1a_total_metals = 1.3743416565891;
static const double sn1a_yields[NSPECIES] = {0, 0, 4.83E-02, 1.16E-06 , 1.43E-01 , 4.51E-03, 8.57E-03, 1.53E-01, 7.43e-01};

/* AGB yields from Karakas 2010, 0912.2142 Tables A2 - A5. These have been parsed by the script in tools/extract_yields.py
 * Massive stars are from Doherty 2014, https://doi.org/10.1093/mnras/stt1877 and https://doi.org/10.1093/mnras/stu571
 * Some of the metallicities in Karakas are listed at M = 2 and some at M = 2.1. I have altered them all to be at M = 2,
 * a change which is within the uncertainty of the calculation.
 */
#define AGB_NMET 4
#define AGB_NMASS 18
static const double agb_masses[AGB_NMASS] = { 1.00,1.25,1.50,1.75,1.90,2.00,2.25,2.50,3.00,3.50,4.00,4.50,5.00,5.50,6.00,6.50,7.00,7.50 };
static const double agb_metallicities[AGB_NMET] = { 0.0001,0.0040,0.0080,0.0200 };
static const double agb_total_mass[AGB_NMET*AGB_NMASS] = {
0.280,0.390,0.423,0.436,
0.582,0.608,0.650,0.676,
0.839,0.872,0.867,0.900,
1.086,1.120,1.114,1.135,
1.219,1.260,1.260,1.270,
1.315,1.450,1.456,1.360,
1.537,1.586,1.598,1.590,
1.768,1.829,1.837,1.837,
2.187,2.269,2.306,2.318,
2.646,2.686,2.734,2.782,
3.126,3.148,3.164,3.208,
3.603,3.628,3.639,3.648,
4.071,4.095,4.114,4.121,
4.534,4.568,4.593,4.600,
4.994,5.023,5.052,5.071,
5.401,5.494,5.548,5.537,
5.827,5.936,6.001,6.033,
6.269,6.342,6.442,6.489,

};

static const double agb_total_metals[AGB_NMET*AGB_NMASS] = {
2.674e-04,6.293e-06,1.198e-05,2.701e-05,
3.907e-03,1.942e-04,3.162e-05,7.465e-05,
1.247e-02,1.939e-03,3.072e-04,1.282e-04,
2.578e-02,5.893e-03,3.192e-03,1.818e-04,
3.371e-02,9.377e-03,5.453e-03,2.050e-04,
4.116e-02,1.256e-02,1.081e-02,2.204e-04,
5.712e-02,2.538e-02,1.576e-02,1.269e-03,
5.633e-02,3.826e-02,2.290e-02,6.299e-03,
2.561e-02,4.226e-02,4.220e-02,1.771e-02,
2.685e-02,2.557e-02,2.842e-02,2.396e-02,
3.358e-02,3.340e-02,2.094e-02,1.405e-02,
2.737e-02,5.052e-02,4.058e-02,1.406e-02,
2.825e-02,6.915e-02,5.270e-02,1.450e-02,
3.203e-02,4.969e-02,4.459e-02,2.389e-02,
4.010e-02,5.086e-02,4.063e-02,2.019e-02,
1.001e-01,2.534e-02,3.349e-02,1.729e-02,
4.911e-02,1.478e-02,2.021e-02,1.449e-02,
3.322e-02,1.289e-02,1.541e-02,1.375e-02,

};

static const double agb_yield[NSPECIES][AGB_NMET*AGB_NMASS] = {
{-5.642623e-03,-8.130491e-03,-8.729696e-03,-8.104831e-03,
-2.988347e-02,-1.487437e-02,-1.440054e-02,-1.229414e-02,
-5.875391e-02,-2.164262e-02,-1.743931e-02,-1.283199e-02,
-1.129371e-01,-3.801405e-02,-2.593309e-02,-1.180619e-02,
-1.408402e-01,-4.909682e-02,-3.086823e-02,-1.085097e-02,
-1.705172e-01,-5.411971e-02,-4.785442e-02,-1.018792e-02,
-2.153547e-01,-1.002597e-01,-6.493235e-02,-1.505363e-02,
-2.019812e-01,-1.499805e-01,-9.563482e-02,-3.003204e-02,
-1.022217e-01,-1.602389e-01,-1.657324e-01,-6.952620e-02,
-1.438086e-01,-9.483194e-02,-1.112996e-01,-9.376347e-02,
-2.370245e-01,-1.629348e-01,-9.016657e-02,-6.305790e-02,
-2.922697e-01,-2.664463e-01,-1.985617e-01,-1.085691e-01,
-3.727276e-01,-4.288444e-01,-3.247516e-01,-1.821017e-01,
-4.527774e-01,-4.300790e-01,-3.500514e-01,-2.753654e-01,
-6.039465e-01,-5.543039e-01,-4.454107e-01,-3.515351e-01,
-9.320000e-01,-5.492000e-01,-4.908000e-01,-4.197447e-01,
-8.000000e-01,-5.843000e-01,-5.298000e-01,-4.832000e-01,
-7.539000e-01,-6.247000e-01,-5.790000e-01,-5.584000e-01,

},
{
5.375872e-03,8.124574e-03,8.717207e-03,8.073388e-03,
2.597561e-02,1.467854e-02,1.436804e-02,1.221924e-02,
4.628178e-02,1.970336e-02,1.713157e-02,1.270509e-02,
8.715840e-02,3.212019e-02,2.274111e-02,1.162459e-02,
1.071259e-01,3.971975e-02,2.541648e-02,1.064693e-02,
1.293526e-01,4.155880e-02,3.704193e-02,9.967277e-03,
1.582319e-01,7.488358e-02,4.917347e-02,1.378389e-02,
1.456492e-01,1.117203e-01,7.273614e-02,2.373195e-02,
7.660912e-02,1.179791e-01,1.235350e-01,5.181634e-02,
1.169609e-01,6.926319e-02,8.287817e-02,6.980080e-02,
2.034503e-01,1.295337e-01,6.922756e-02,4.900520e-02,
2.648871e-01,2.159211e-01,1.579783e-01,9.450633e-02,
3.444618e-01,3.596774e-01,2.720431e-01,1.675983e-01,
4.207259e-01,3.803780e-01,3.054525e-01,2.514721e-01,
5.638373e-01,5.033906e-01,4.047518e-01,3.313334e-01,
8.319003e-01,5.239039e-01,4.573048e-01,4.024458e-01,
7.508000e-01,5.695037e-01,5.096048e-01,4.687159e-01,
7.217002e-01,6.119060e-01,5.635056e-01,5.446183e-01,

},
{
2.505355e-04,-3.863257e-05,-6.939759e-05,-1.502565e-04,
3.724916e-03,7.324660e-05,-1.880346e-04,-4.238875e-04,
1.160401e-02,1.670982e-03,-8.481364e-05,-7.459906e-04,
2.313006e-02,5.328459e-03,2.486207e-03,-1.061258e-03,
2.967089e-02,8.558645e-03,4.562841e-03,-1.194751e-03,
3.568269e-02,1.150873e-02,9.461715e-03,-1.286097e-03,
4.841891e-02,2.288348e-02,1.389286e-02,-5.679179e-04,
4.919233e-02,3.386392e-02,2.024639e-02,3.908196e-03,
2.392389e-02,3.768617e-02,3.681748e-02,1.381412e-02,
7.459310e-03,2.312945e-02,2.506377e-02,1.883355e-02,
5.448533e-03,8.984008e-03,1.812641e-02,9.359143e-03,
3.791906e-03,3.863892e-03,2.782905e-03,8.760184e-03,
3.130832e-03,3.361759e-03,-2.609563e-04,5.204891e-03,
2.909447e-03,3.262132e-03,9.055629e-04,-5.865926e-03,
1.488005e-03,1.933324e-03,-9.956316e-04,-9.304656e-03,
3.753500e-03,8.188000e-04,-1.667700e-03,-1.071584e-02,
2.456000e-03,-5.340000e-04,-3.253300e-03,-1.330940e-02,
2.450100e-03,-7.882000e-04,-3.988600e-03,-1.417580e-02,

},
{
4.019394e-06,4.960512e-05,8.478541e-05,1.791282e-04,
1.978621e-05,1.194106e-04,2.273566e-04,5.024799e-04,
3.987334e-05,2.037581e-04,3.918285e-04,8.813317e-04,
5.356670e-05,3.059018e-04,5.702331e-04,1.247711e-03,
6.427879e-05,3.572941e-04,6.325749e-04,1.405151e-03,
7.084068e-05,3.643631e-04,7.750321e-04,1.514490e-03,
7.290039e-05,4.499870e-04,8.502585e-04,1.928996e-03,
4.855711e-05,5.828001e-04,1.088917e-03,2.270405e-03,
7.569660e-05,7.138748e-04,1.444833e-03,3.088211e-03,
1.803907e-02,1.052384e-03,1.810410e-03,4.029449e-03,
2.632309e-02,2.296469e-02,2.083586e-03,5.070370e-03,
2.239781e-02,4.498992e-02,3.659366e-02,6.411231e-03,
2.402889e-02,6.473691e-02,5.362213e-02,1.185130e-02,
2.793715e-02,4.762633e-02,4.596258e-02,3.396277e-02,
3.743055e-02,5.291434e-02,4.764608e-02,3.619429e-02,
9.042615e-02,2.897641e-02,4.087194e-02,3.656377e-02,
4.483642e-02,1.983596e-02,3.027111e-02,3.665618e-02,
2.939749e-02,1.776606e-02,2.587046e-02,3.849444e-02,

},
{
1.144322e-05,-4.698614e-06,-3.418436e-06,-1.890063e-06,
9.542812e-05,-9.794108e-07,-7.722605e-06,-4.305696e-06,
2.757290e-04,2.895315e-05,-7.110182e-06,-7.257497e-06,
4.271349e-04,1.115789e-04,3.172567e-05,-4.887602e-06,
4.791704e-04,1.395040e-04,6.018509e-05,-5.719930e-06,
5.163211e-04,1.617246e-04,4.385436e-05,-8.307815e-06,
6.312759e-04,2.004913e-04,3.723495e-05,-1.664491e-04,
6.913206e-04,9.687062e-05,-2.083897e-04,-3.786100e-04,
4.169877e-04,7.961497e-06,-5.806949e-04,-1.085916e-03,
4.511217e-04,9.774490e-05,-5.366908e-04,-1.662603e-03,
6.495725e-04,1.965128e-05,-4.174997e-04,-1.649598e-03,
4.780282e-04,-8.681399e-04,-1.131905e-03,-2.320373e-03,
4.151088e-04,-2.757884e-03,-3.824554e-03,-3.691087e-03,
3.734910e-04,-3.197394e-03,-4.478804e-03,-6.020938e-03,
4.220641e-04,-5.819265e-03,-7.806752e-03,-8.059783e-03,
1.645854e-03,-5.172320e-03,-7.201990e-03,-9.673627e-03,
8.050500e-04,-4.928750e-03,-7.596540e-03,-9.842700e-03,
9.265416e-04,-4.485890e-03,-7.109070e-03,-1.148170e-02,

},
{
1.316208e-06,-1.705147e-07,-1.627603e-07,-1.199229e-07,
6.577726e-05,2.099968e-06,-4.171525e-07,-2.359548e-07,
5.429187e-04,3.384444e-05,5.660989e-06,-1.502591e-06,
2.097578e-03,1.422673e-04,9.815844e-05,-4.044923e-06,
3.360130e-03,3.125627e-04,1.882832e-04,-6.296348e-06,
4.657263e-03,5.134346e-04,5.131787e-04,-7.713325e-06,
7.456716e-03,1.784828e-03,9.471586e-04,5.918453e-05,
5.567156e-03,3.564760e-03,1.709659e-03,4.709854e-04,
9.493749e-04,3.534962e-03,4.278601e-03,1.808903e-03,
6.661942e-04,1.020317e-03,1.828225e-03,2.603001e-03,
8.596598e-04,9.965575e-04,9.145597e-04,1.127866e-03,
5.040222e-04,1.772577e-03,1.689808e-03,9.890546e-04,
4.378024e-04,2.345827e-03,2.196763e-03,8.754801e-04,
4.574929e-04,1.186304e-03,1.394905e-03,1.277145e-03,
3.805046e-04,9.145097e-04,9.954287e-04,8.641534e-04,
1.361119e-03,3.639210e-04,9.023300e-04,6.037974e-04,
2.729696e-04,1.928320e-04,4.175280e-04,5.228700e-04,
1.251973e-04,1.767410e-04,2.580760e-04,3.510900e-04,

},
{
2.101775e-08,-1.950411e-09,-3.446985e-10,6.639312e-10,
3.107607e-07,-2.095385e-08,-1.542503e-09,4.432513e-08,
4.065824e-06,-1.894387e-08,-2.833258e-08,1.637090e-10,
2.954731e-05,3.739224e-07,1.113294e-07,2.633897e-09,
6.334146e-05,1.472212e-06,5.624825e-07,3.274181e-10,
1.203592e-04,3.339264e-06,3.038516e-06,-5.085894e-09,
3.571086e-04,2.362346e-05,7.462444e-06,1.385124e-07,
7.121892e-04,7.700664e-05,2.238590e-05,2.543791e-06,
2.239338e-04,2.494427e-04,1.430589e-04,2.413336e-05,
2.109235e-04,2.288232e-04,1.995587e-04,8.106456e-05,
2.664080e-04,3.813975e-04,1.792058e-04,7.764138e-05,
1.768835e-04,6.691558e-04,5.495575e-04,1.336896e-04,
2.143041e-04,1.301971e-03,8.204378e-04,1.475886e-04,
3.261622e-04,7.120281e-04,6.836627e-04,3.754471e-04,
3.339121e-04,8.232322e-04,6.777801e-04,3.226652e-04,
2.483586e-03,2.752000e-04,4.487000e-04,3.238369e-04,
6.028700e-04,1.545000e-04,2.621000e-04,2.200800e-04,
2.685900e-04,1.737000e-04,2.654000e-04,2.922000e-04,

},
{
7.956011e-09,1.624699e-09,-3.578862e-10,-9.868018e-10,
2.078031e-08,1.618218e-09,-9.231372e-11,4.744834e-08,
4.353817e-08,1.280068e-08,1.992976e-08,9.986252e-10,
1.172747e-07,4.653293e-08,3.718014e-08,1.000444e-09,
2.037812e-07,9.340783e-08,6.963364e-08,6.803020e-10,
3.920963e-07,1.545645e-07,1.859590e-07,1.018634e-10,
1.946635e-06,5.372162e-07,3.371579e-07,1.544104e-07,
8.281033e-06,1.278180e-06,7.175022e-07,2.689630e-07,
3.722101e-06,4.903038e-06,3.306106e-06,1.144679e-06,
4.131819e-06,1.028857e-05,7.060952e-06,2.775138e-06,
6.149365e-06,1.767164e-05,1.181986e-05,3.273374e-06,
4.090958e-06,3.144513e-05,2.921447e-05,6.945937e-06,
5.094176e-06,5.902792e-05,5.114446e-05,8.771024e-06,
8.535403e-06,3.797211e-05,4.357890e-05,2.379176e-05,
1.079105e-05,4.533040e-05,4.955597e-05,2.406067e-05,
1.161400e-04,2.314300e-05,4.429700e-05,2.561744e-05,
5.279800e-05,1.513600e-05,2.990200e-05,2.193500e-05,
1.921710e-05,1.588500e-05,2.282800e-05,2.742800e-05,

},
{
-5.202560e-09,3.400146e-09,4.412755e-10,1.088324e-09,
-1.735274e-08,5.487403e-10,-2.997922e-10,8.541770e-08,
-4.139017e-08,1.020765e-09,3.078192e-08,4.040885e-09,
-9.139746e-08,2.070381e-09,6.519134e-09,6.548362e-10,
-1.010660e-07,2.194828e-10,6.677392e-09,3.150944e-09,
-1.175865e-07,-5.594377e-09,1.132296e-08,3.597052e-10,
-2.769324e-07,-2.000941e-07,1.475980e-08,2.383207e-07,
-4.956826e-07,-8.677151e-07,-1.305380e-07,1.867202e-07,
-2.704734e-07,-4.930788e-06,-2.348083e-06,2.837674e-07,
-2.824531e-07,-5.852666e-06,-5.762917e-06,-1.056045e-06,
-3.587168e-07,-8.559529e-06,-6.419793e-06,-2.071806e-06,
-3.041530e-07,-1.474106e-05,-1.746426e-05,-4.965648e-06,
-3.304431e-07,-2.338617e-05,-2.567751e-05,-6.411890e-06,
-4.124452e-07,-1.563560e-05,-2.152593e-05,-1.550979e-05,
-4.961879e-07,-1.650463e-05,-2.098344e-05,-1.442869e-05,
-1.384616e-06,-6.884100e-06,-1.426100e-05,-1.341671e-05,
-6.475570e-07,-3.975300e-06,-8.627700e-06,-8.529000e-06,
-2.965600e-07,-3.577700e-06,-7.476700e-06,-1.068100e-05,
}
};

/* Supernova II yields are from Kobayashi 2006. There is a mass gap from 8 - 13 Msun, between AGB and SNII,
 * for which we extrapolate Kobayashi 2006 yields to lower masses.*/
#define SNII_NMET 4
#define SNII_NMASS 7
static const double snii_masses[SNII_NMASS] = { 13.00,15.00,18.00,20.00,25.00,30.00,40.00 };
static const double snii_metallicities[SNII_NMET] = { 0.0000,0.0010,0.0040,0.0200 };
static const double snii_total_mass[SNII_NMET*SNII_NMASS] = {
11.430,11.350,11.390,11.400,
13.520,13.470,13.500,13.500,
16.350,16.300,16.390,16.420,
18.340,18.150,18.240,18.450,
23.080,23.090,23.320,23.310,
27.930,27.940,27.440,27.900,
37.110,36.830,37.190,37.790,

};

static const double snii_total_metals[SNII_NMET*SNII_NMASS] = {
1.539e+01,1.556e+01,1.532e+01,1.500e+01,
1.802e+01,1.724e+01,1.672e+01,1.624e+01,
2.215e+01,2.067e+01,1.965e+01,1.988e+01,
2.529e+01,2.506e+01,2.306e+01,2.202e+01,
3.133e+01,3.206e+01,2.941e+01,2.762e+01,
3.878e+01,3.868e+01,3.706e+01,3.206e+01,
5.409e+01,5.186e+01,4.745e+01,3.540e+01,

};

static const double snii_yield[NSPECIES][SNII_NMET*SNII_NMASS] = {
{6.600000e+00,6.440000e+00,6.370000e+00,6.160000e+00,
7.580000e+00,7.450000e+00,7.110000e+00,6.790000e+00,
8.430000e+00,8.460000e+00,7.470000e+00,7.530000e+00,
8.770000e+00,8.430000e+00,8.950000e+00,7.930000e+00,
1.060000e+01,9.800000e+00,1.020000e+01,8.410000e+00,
1.170000e+01,1.110000e+01,1.010000e+01,8.750000e+00,
1.400000e+01,1.290000e+01,1.030000e+01,3.550000e+00,

},
{
4.010041e+00,3.860143e+00,4.040170e+00,4.300196e+00,
4.400041e+00,5.160153e+00,4.950159e+00,5.250218e+00,
5.420033e+00,6.540157e+00,6.060224e+00,6.110230e+00,
5.940048e+00,5.940160e+00,7.030175e+00,6.760238e+00,
8.030211e+00,6.970126e+00,8.480185e+00,7.240221e+00,
9.520206e+00,8.380144e+00,7.920184e+00,8.360212e+00,
1.190003e+01,1.090012e+01,8.120180e+00,4.710051e+00,

},
{
7.410008e-02,1.071670e-01,8.798800e-02,1.080000e-01,
1.720001e-01,8.505380e-02,8.830900e-02,6.625000e-02,
2.190000e-01,1.300720e-01,1.653000e-01,1.373800e-01,
2.110000e-01,1.280196e-01,9.769200e-02,2.464500e-01,
2.940000e-01,2.150981e-01,1.323830e-01,2.186000e-01,
3.380000e-01,1.210820e-01,1.823390e-01,2.519200e-01,
4.290000e-01,7.398200e-02,4.583680e-01,5.964310e-01,

},
{
1.830064e-03,9.077570e-03,9.086840e-03,4.804090e-02,
1.860069e-03,3.580859e-03,1.290870e-02,6.155970e-02,
1.890240e-04,4.470921e-03,1.262000e-01,6.611530e-02,
5.421130e-05,1.290137e-02,1.842780e-02,7.212400e-02,
5.911180e-04,9.207240e-03,3.159530e-02,1.306000e-01,
1.656800e-06,6.190379e-03,2.010498e-02,1.020066e-01,
1.218000e-06,8.692450e-03,2.600501e-02,5.810572e-02,

},
{
4.500017e-01,5.058796e-01,3.870375e-01,2.223680e-01,
7.730065e-01,2.943916e-01,2.930546e-01,1.653520e-01,
1.380005e+00,4.223302e-01,5.741100e-01,7.825760e-01,
2.110000e+00,2.180030e+00,9.953840e-01,1.056171e+00,
2.790002e+00,3.820098e+00,2.200960e+00,2.435640e+00,
4.810000e+00,5.330076e+00,4.790164e+00,3.227870e+00,
8.380000e+00,8.370055e+00,7.960996e+00,7.343272e+00,

},
{
1.530074e-02,6.751500e-02,1.332350e-01,3.944500e-02,
3.270537e-01,1.903415e-01,1.258970e-01,3.575000e-02,
4.941169e-01,1.775626e-01,2.051700e-01,1.558320e-01,
9.121122e-01,6.283170e-01,2.794180e-01,4.048500e-01,
5.330335e-01,1.221979e+00,8.249530e-01,8.713900e-01,
8.511408e-01,1.452171e+00,9.439010e-01,9.585700e-01,
3.070175e-01,2.879870e-01,1.884040e+00,2.225870e+00,

},
{
8.642770e-02,6.583400e-02,4.642000e-02,2.994000e-02,
6.889700e-02,6.572000e-02,7.848000e-02,4.110000e-02,
1.584600e-01,6.117300e-02,8.396000e-02,1.159800e-01,
1.503540e-01,2.468400e-01,1.005800e-01,9.487000e-02,
1.200906e-01,1.827300e-01,2.457500e-01,2.766000e-01,
2.273760e-01,2.938200e-01,2.321700e-01,2.472000e-01,
4.785540e-01,7.073200e-01,4.043000e-01,4.562000e-01,

},
{
8.257000e-02,9.317000e-02,6.229700e-02,7.784000e-02,
7.358800e-02,4.370700e-02,1.054100e-01,8.875000e-02,
1.167870e-01,1.541350e-01,1.006900e-01,1.147800e-01,
9.969200e-02,1.298860e-01,1.268800e-01,6.768000e-02,
3.513464e-01,1.207150e-01,1.225100e-01,1.412500e-01,
2.488430e-01,1.667390e-01,4.031900e-01,2.579800e-01,
1.036660e+00,8.971400e-01,5.340500e-01,2.607500e-01,

},
{
7.172600e-02,7.559680e-02,7.493770e-02,8.746100e-02,
7.238000e-02,7.327010e-02,7.540890e-02,8.976000e-02,
7.227800e-02,7.444550e-02,9.409600e-02,9.294600e-02,
7.228700e-02,7.404090e-02,7.768170e-02,9.375600e-02,
7.377700e-02,7.395780e-02,7.734500e-02,9.664700e-02,
7.457300e-02,7.518580e-02,8.269400e-02,1.038800e-01,
8.000101e-02,8.257700e-02,8.525500e-02,8.967500e-02,
}
};


#endif
