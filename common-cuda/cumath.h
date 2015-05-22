
#ifndef CUMATH_H
#define CUMATH_H

#ifdef __NVCC__

#include <cuda_runtime.h>
#include <cufft.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include <sm_30_intrinsics.hpp>

#include "complex.h"

// do not use printf macro in matlab 
#ifdef printf
#undef printf
#endif

inline bool is_pow_2(int x) { return ((x&(x-1)) == 0); }

__device__ inline double atomicAdd(double *address, double val)
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

__device__ inline double atomicAdd(double &address, const double &val)
{ return atomicAdd(&address, val); }

__device__ inline Complex atomicAdd(Complex &sum, const Complex &val)
{ 
  atomicAdd(sum.re, val.re); 
  atomicAdd(sum.im, val.im); 
  return sum;
}

__device__ inline Complex atomicAdd(Complex *sum, const Complex &val)
{ return atomicAdd(*sum, val); }

template<class T1, class T2, class T3> 
__global__ void DotProduct(const T1 *r, const T2 *c, T3 *dot, const int size)
{
  extern __shared__ T3 s[];
  
  // dot should be zero
  s[threadIdx.x] = *dot; 
  const int j = threadIdx.x + blockDim.x*blockIdx.x;
  if(j < size)
    s[threadIdx.x] = r[j]*c[j];
  
  __syncthreads();
  
  // do reduction in shared memory
  for(int i = blockDim.x/2; i > 0; i /= 2) {
    if(threadIdx.x < i)
      s[threadIdx.x] += s[threadIdx.x+i];
    __syncthreads();
  }

  if(threadIdx.x == 0) {
    //printf("s[0] = %.f\n", s[0]);
    atomicAdd(dot, s[0]);
  }
}

template __global__ 
void DotProduct<double, Complex, Complex>(const double *r, const Complex *c, Complex *dot, const int size);

template<class T1, class T2, class T3> 
static __global__ void _vector_multiplication_(T1 *vOut, const T2 *vIn1, const T3 *vIn2, const int n)
{
  const int j = threadIdx.x + blockDim.x*blockIdx.x;
  if(j < n) vOut[j] = vIn1[j]*vIn2[j];
}

template __global__ 
void _vector_multiplication_<Complex, Complex, double>(Complex *vOut, const Complex *vIn1, const double *vIn2, const int n);

#endif
#endif /* CUMATH_H */
