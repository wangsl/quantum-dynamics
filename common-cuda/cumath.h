
#ifndef CUMATH_H
#define CUMATH_H

#ifdef __NVCC__

#include "complex.h"

__device__ double atomicAdd(double *address, double val);
__device__ double atomicAdd(double &address, const double &val);

__device__ Complex atomicAdd(Complex *sum, const Complex &val);
__device__ Complex atomicAdd(Complex &sum, const Complex &val);

#endif

#endif /* CUMATH_H */


