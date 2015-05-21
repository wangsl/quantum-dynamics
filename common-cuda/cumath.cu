
#include "cumath.h"

__device__ double atomicAdd(double *address, double val)
{
  unsigned long long int* address_as_ull = (unsigned long long int*)address;
  unsigned long long int old = *address_as_ull, assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
                    __double_as_longlong(val + __longlong_as_double(assumed)));
  } while (assumed != old);
  return __longlong_as_double(old);
}

__device__ double atomicAdd(double &address, const double &val)
{ return atomicAdd(&address, val); }

__device__ Complex atomicAdd(Complex &sum, const Complex &val)
{ 
  double re = atomicAdd(sum.re, val.re); 
  double im = atomicAdd(sum.im, val.im); 
  return Complex(re, im);
}

__device__ Complex atomicAdd(Complex *sum, const Complex &val)
{ return atomicAdd(*sum, val); }


