#ifdef CHEMCOOL
      integer isize, isize2
      parameter (isize = 250000)
      parameter (isize2 = 50000)
      REAL temp_table(isize), gamma_table(isize)
      REAL eh2_table(isize2)

      common /gamma_data/temp_table, gamma_table,
     $        eh2_table
#endif
