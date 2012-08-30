#ifdef CHEMCOOL
c
c CO self-shielding, shielding by H2. Common block is initialized in
c cheminmo.F
c
      integer num_CO_self_shield
      parameter (num_CO_self_shield = 51)

      REAL CO_shield_columns(num_CO_self_shield)
      REAL CO_shield_factors(num_CO_self_shield)

      DATA CO_shield_columns/1.200e+01, 1.222e+01, 1.248e+01, 1.278e+01,
     $                       1.312e+01, 1.350e+01, 1.393e+01, 1.439e+01,
     $                       1.490e+01, 1.522e+01, 1.541e+01, 1.565e+01,
     $                       1.578e+01, 1.595e+01, 1.613e+01, 1.622e+01,
     $                       1.636e+01, 1.649e+01, 1.663e+01, 1.677e+01,
     $                       1.690e+01, 1.704e+01, 1.718e+01, 1.731e+01,
     $                       1.745e+01, 1.763e+01, 1.781e+01, 1.798e+01,
     $                       1.816e+01, 1.834e+01, 1.852e+01, 1.862e+01,
     $                       1.875e+01, 1.889e+01, 1.903e+01, 1.911e+01,
     $                       1.923e+01, 1.934e+01, 1.940e+01, 1.951e+01,
     $                       1.961e+01, 1.972e+01, 1.977e+01, 1.987e+01,
     $                       1.992e+01, 1.997e+01, 2.001e+01, 2.006e+01,
     $                       2.011e+01, 2.015e+01, 2.020e+01/

      DATA CO_shield_factors/9.990e-01, 9.981e-01, 9.961e-01, 9.912e-01,
     $                       9.815e-01, 9.601e-01, 9.113e-01, 8.094e-01,
     $                       6.284e-01, 4.808e-01, 3.889e-01, 2.827e-01,
     $                       2.293e-01, 1.695e-01, 1.224e-01, 1.017e-01,
     $                       7.764e-02, 5.931e-02, 4.546e-02, 3.506e-02,
     $                       2.728e-02, 2.143e-02, 1.700e-02, 1.360e-02,
     $                       1.094e-02, 8.273e-03, 6.283e-03, 4.773e-03,
     $                       3.611e-03, 2.704e-03, 1.986e-03, 1.657e-03,
     $                       1.258e-03, 9.332e-04, 6.745e-04, 5.596e-04,
     $                       4.123e-04, 2.982e-04, 2.490e-04, 1.827e-04,
     $                       1.324e-04, 9.473e-05, 7.891e-05, 5.668e-05,
     $                       4.732e-05, 3.967e-05, 3.327e-05, 2.788e-05,
     $                       2.331e-05, 1.944e-05, 1.619e-05/

      integer num_CO_H2_shield
      parameter (num_CO_H2_shield = 43)

      REAL H2_shield_columns(num_CO_H2_shield)
      REAL H2_shield_factors(num_CO_H2_shield)

      DATA H2_shield_columns/1.300e+01,
     $                       1.343e+01, 1.458e+01, 1.582e+01, 1.695e+01,
     $                       1.797e+01, 1.800e+01, 1.831e+01, 1.848e+01,
     $                       1.861e+01, 1.870e+01, 1.878e+01, 1.885e+01,
     $                       1.891e+01, 1.897e+01, 1.901e+01, 1.931e+01,
     $                       1.948e+01, 1.961e+01, 1.971e+01, 1.978e+01,
     $                       1.985e+01, 1.991e+01, 1.997e+01, 2.001e+01,
     $                       2.031e+01, 2.048e+01, 2.061e+01, 2.071e+01,
     $                       2.078e+01, 2.085e+01, 2.091e+01, 2.097e+01,
     $                       2.101e+01, 2.131e+01, 2.148e+01, 2.161e+01,
     $                       2.171e+01, 2.178e+01, 2.185e+01, 2.191e+01,
     $                       2.197e+01, 2.201e+01/

      DATA H2_shield_factors/1.000e+00,
     $                       9.999e-01, 9.893e-01, 9.678e-01, 9.465e-01,
     $                       9.137e-01, 9.121e-01, 8.966e-01, 8.862e-01,
     $                       8.781e-01, 8.716e-01, 8.660e-01, 8.612e-01,
     $                       8.569e-01, 8.524e-01, 8.497e-01, 8.262e-01,
     $                       8.118e-01, 8.011e-01, 7.921e-01, 7.841e-01,
     $                       7.769e-01, 7.702e-01, 7.626e-01, 7.579e-01,
     $                       7.094e-01, 6.712e-01, 6.378e-01, 6.074e-01,
     $                       5.791e-01, 5.524e-01, 5.271e-01, 4.977e-01,
     $                       4.793e-01, 2.837e-01, 1.526e-01, 7.774e-02,
     $                       3.952e-02, 2.093e-02, 1.199e-02, 7.666e-03,
     $                       5.333e-03, 4.666e-03/

      integer imax_COss, imax_COH2
      parameter (imax_COss = 161)
      parameter (imax_COH2 = 181)

      REAL NCO_shield_min, NCO_shield_max
      parameter (NCO_shield_min = 1d12)
      parameter (NCO_shield_max = 1d20)			      

      REAL NH2_shield_min, NH2_shield_max
      parameter (NH2_shield_min = 1d13)
      parameter (NH2_shield_max = 1d22)

      REAL dNshield
      parameter (dNshield = 0.05d0)
  
      integer ioff_h2, ioff_co
c Some compilers don't like intrinsic functions in parameter statements, 
c so precompute these
c ioff_h2 = (dlog10(NH2_shield_min) / dNshield) - 1
      parameter (ioff_h2 = 259)
c
c ioff_co = (dlog10(NCO_shield_min) / dNshield) - 1
c
      parameter(ioff_co = 239)

      REAL CO_self_shielding(imax_COss), CO_H2_shielding(imax_COH2)
      common /co_shield/ CO_self_shielding, CO_H2_shielding

#endif /* CHEMCOOL */
