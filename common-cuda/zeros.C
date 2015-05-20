
/* $Id$ */

#include <cstring>
#include "fort.h"
#include "complex.h"

template<typename T> void template_zeros(const int n, T *a)
{
  memset(a, 0, n*sizeof(T));
}

extern "C" {
  // Fortran version: DZeros
  void FORT(dzeros)(const int &n, double *x)
  { template_zeros<double>(n, x); }
  
  // Fortran version: IZeros
  void FORT(izeros)(const int &n, int *x)
  { template_zeros<int>(n, x); }
  
  // Fotran version: DCZeros
  void FORT(dczeros)(const int &n, Complex *x)
  { template_zeros<Complex>(n, x); }
}





