
/* I (Volker Springel) have written this class `dd' based in part on 
 * code contained in the 
 *  ------------------------------------------------------------------------
 *  | QUAD-DOUBLE/DOUBLE-DOUBLE COMPUTATION PACKAGE                        |
 *  |                                                                      |
 *  | Yozo Hida        U.C. Berkeley               yozo@cs.berkeley.edu    |
 *  | Xiaoye S. Li     Lawrence Berkeley Natl Lab  xiaoye@nersc.gov        |
 *  | David H. Bailey  Lawrence Berkeley Natl Lab  dhbailey@lbl.gov        |
 *  |                                                                      |
 *  | Revised  2005-03-12  Copyright (c) 2005                              |
 *  ------------------------------------------------------------------------
 *  (available at http://crd.lbl.gov/~dhbailey/mpdist/)
 */

#ifndef _QD_DD_H
#define _QD_DD_H

class dd
{
public:
  double hi, lo;

  double quick_two_sum(double a, double b, double &err)
  {
    double s = a + b;

    err = b - (s - a);
    return s;
  }

  double two_sum(double a, double b, double &err)
  {
    double s = a + b;
    double bb = s - a;

    err = (a - (s - bb)) + (b - bb);
    return s;
  }


  /* Self-Addition with a double */
  dd & operator+=(double a)
  {
    double s1, s2;

    s1 = two_sum(hi, a, s2);
    s2 += lo;
    hi = quick_two_sum(s1, s2, lo);
    return *this;
  };

  /* Self-Addition with a doubledouble */
  dd & operator+=(const dd & a)
  {
    double s1, s2, t1, t2;

    s1 = two_sum(hi, a.hi, s2);
    t1 = two_sum(lo, a.lo, t2);
    s2 += t1;
    s1 = quick_two_sum(s1, s2, s2);
    s2 += t2;
    hi = quick_two_sum(s1, s2, lo);
    return *this;
  };

  /* Assignment */
  dd & operator=(double a)
    {
      hi = a;
      lo = 0.0;
      return *this;
    };

  /* Cast */
  operator  double () const 
   {
    return hi;
   };

};

#endif /* _QD_DD_H */
