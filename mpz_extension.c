#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>


#ifdef VORONOI

#include "allvars.h"
#include "proto.h"
#include "voronoi.h"
#include <gmp.h>



#if USEDBITS > 31

void MY_mpz_set_si(mpz_t dest, signed long long int val)
{
  mp_size_t size;
  unsigned long long int vl;

  vl = (unsigned long long int) (val >= 0 ? val : -val);

  if(vl > GMP_NUMB_MAX)
    {
      mpz_realloc2(dest, 2 * GMP_NUMB_BITS);
      dest->_mp_d[1] = (vl >> GMP_NUMB_BITS);
      size = 2;
    }
  else
    {
      size = vl != 0;
    }

  dest->_mp_d[0] = (vl & GMP_NUMB_MASK);
  dest->_mp_size = val >= 0 ? size : -size;
}


void MY_mpz_mul_si(mpz_t prod, mpz_t mult, signed long long int val)
{
  mpz_t tmp;

  mpz_init(tmp);

  MY_mpz_set_si(tmp, val);

  mpz_mul(prod, mult, tmp);

  mpz_clear(tmp);
}

void MY_mpz_sub_ui(mpz_t prod, mpz_t mult, unsigned long long int val)
{
  mpz_t tmp;

  mpz_init(tmp);

  MY_mpz_set_si(tmp, val);

  mpz_sub(prod, mult, tmp);

  mpz_clear(tmp);
}


#endif

#endif
