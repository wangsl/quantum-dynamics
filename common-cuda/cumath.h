
#ifndef CUMATH_H
#define CUMATH_H

#ifdef __NVCC__

#include "complex.h"

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

#if 0
// Ref: http://devblogs.nvidia.com/parallelforall/faster-parallel-reductions-kepler/
__device__ inline double __shfl(double var, int srcLane, int width=32) 
{
  int2 a = *reinterpret_cast<int2 *> (&var);
  a.x = __shfl((int) a.x, srcLane, width);
  a.y = __shfl((int) a.y, srcLane, width);
  return *reinterpret_cast<double*>(&a);
}
#endif

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
  
  if(threadIdx.x == 0) 
    atomicAdd(dot, s[0]);
}

template __global__ void DotProduct<double, Complex, Complex>(const double *r, const Complex *c, Complex *dot, const int size);

#endif
#endif /* CUMATH_H */
