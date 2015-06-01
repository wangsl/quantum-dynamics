
#include <iostream>
using namespace std;
#include <cassert>
#include <iomanip>  

#include "gradients.h"
#include "cumath.h"

// Reference
// http://www.trentfguidry.net/post/2010/09/04/Numerical-differentiation-formulas.aspx

using cumath::ij_2_index;
using cumath::index_2_ij;
using cumath::ijk_2_index;

template<class T> __device__ void gradients_3d_11_points(const int n1, const int n2, const int n3,
							 const int n2p, const double dx2,
							 const T *f, T *v, T *g)
{
  const int index = threadIdx.x + blockDim.x*blockIdx.x;
  
  if(index < n1*n3) {
    int i = -1; int k = -1;
    index_2_ij(index, n1, n3, i, k);

    v[index] = f[ijk_2_index(n1, n2, n3, i, n2p, k)];
    
    T g_tmp = 
           2*(f[ijk_2_index(n1, n2, n3, i, n2p+5, k)] - f[ijk_2_index(n1, n2, n3, i, n2p-5, k)]) 
      -   25*(f[ijk_2_index(n1, n2, n3, i, n2p+4, k)] - f[ijk_2_index(n1, n2, n3, i, n2p-4, k)]) 
      +  150*(f[ijk_2_index(n1, n2, n3, i, n2p+3, k)] - f[ijk_2_index(n1, n2, n3, i, n2p-3, k)]) 
      -  600*(f[ijk_2_index(n1, n2, n3, i, n2p+2, k)] - f[ijk_2_index(n1, n2, n3, i, n2p-2, k)]) 
      + 2100*(f[ijk_2_index(n1, n2, n3, i, n2p+1, k)] - f[ijk_2_index(n1, n2, n3, i, n2p-1, k)]);

    g[index] = g_tmp/(2520*dx2);
  }
}

template<class T> __device__ void gradients_3d_9_points(const int n1, const int n2, const int n3,
							const int n2p, const double dx2,
							const T *f, T *v, T *g)
{
  const int index = threadIdx.x + blockDim.x*blockIdx.x;
  
  if(index < n1*n3) {
    int i = -1; int k = -1;
    index_2_ij(index, n1, n3, i, k);

    v[index] = f[ijk_2_index(n1, n2, n3, i, n2p, k)];
    
    T g_tmp = 
      -   3*(f[ijk_2_index(n1, n2, n3, i, n2p+4, k)] - f[ijk_2_index(n1, n2, n3, i, n2p-4, k)]) 
      +  32*(f[ijk_2_index(n1, n2, n3, i, n2p+3, k)] - f[ijk_2_index(n1, n2, n3, i, n2p-3, k)]) 
      - 168*(f[ijk_2_index(n1, n2, n3, i, n2p+2, k)] - f[ijk_2_index(n1, n2, n3, i, n2p-2, k)]) 
      + 672*(f[ijk_2_index(n1, n2, n3, i, n2p+1, k)] - f[ijk_2_index(n1, n2, n3, i, n2p-1, k)]);

    g[index] = g_tmp/(840*dx2);
  }
}

template<class T> __device__ void gradients_3d_7_points(const int n1, const int n2, const int n3,
							const int n2p, const double dx2,
							const T *f, T *v, T *g)
{
  const int index = threadIdx.x + blockDim.x*blockIdx.x;
  
  if(index < n1*n3) {
    int i = -1; int k = -1;
    index_2_ij(index, n1, n3, i, k);
    
    v[index] = f[ijk_2_index(n1, n2, n3, i, n2p, k)];
    
    T g_tmp = 
           (f[ijk_2_index(n1, n2, n3, i, n2p+3, k)] - f[ijk_2_index(n1, n2, n3, i, n2p-3, k)]) 
      -  9*(f[ijk_2_index(n1, n2, n3, i, n2p+2, k)] - f[ijk_2_index(n1, n2, n3, i, n2p-2, k)]) 
      + 45*(f[ijk_2_index(n1, n2, n3, i, n2p+1, k)] - f[ijk_2_index(n1, n2, n3, i, n2p-1, k)]);

    g[index] = g_tmp/(60*dx2);
  }
}

template<class T> __device__ void gradients_3d_5_points(const int n1, const int n2, const int n3,
							const int n2p, const double dx2,
							const T *f, T *v, T *g)
{
  const int index = threadIdx.x + blockDim.x*blockIdx.x;
  
  if(index < n1*n3) {
    int i = -1; int k = -1;
    index_2_ij(index, n1, n3, i, k);
    
    v[index] = f[ijk_2_index(n1, n2, n3, i, n2p, k)];
    
    T g_tmp = 
      -   (f[ijk_2_index(n1, n2, n3, i, n2p+2, k)] - f[ijk_2_index(n1, n2, n3, i, n2p-2, k)]) 
      + 8*(f[ijk_2_index(n1, n2, n3, i, n2p+1, k)] - f[ijk_2_index(n1, n2, n3, i, n2p-1, k)]);

    g[index] = g_tmp/(12*dx2);
  }
}

template<class T> __device__ void gradients_3d_3_points(const int n1, const int n2, const int n3,
							const int n2p, const double dx2,
							const T *f, T *v, T *g)
{
  const int index = threadIdx.x + blockDim.x*blockIdx.x;
  
  if(index < n1*n3) {
    int i = -1; int k = -1;
    index_2_ij(index, n1, n3, i, k);
    
    v[index] = f[ijk_2_index(n1, n2, n3, i, n2p, k)];
    
    T g_tmp = (f[ijk_2_index(n1, n2, n3, i, n2p+1, k)] - f[ijk_2_index(n1, n2, n3, i, n2p-1, k)]);
    g[index] = g_tmp/(2*dx2);
  }
}

template<class T> __global__ void gradients_3d(const int n1, const int n2, const int n3,
					       const int n2p, const double dx2,
					       const T *f, T *v, T *g, const int n_points)
{
  switch (n_points) {
    
  case 11:
    gradients_3d_11_points<T>(n1, n2, n3, n2p, dx2, f, v, g);
    break;
    
  case 9:
    gradients_3d_9_points<T>(n1, n2, n3, n2p, dx2, f, v, g);
    break;
    
  case 7:
    gradients_3d_7_points<T>(n1, n2, n3, n2p, dx2, f, v, g);
    break;
    
  case 5:
    gradients_3d_5_points<T>(n1, n2, n3, n2p, dx2, f, v, g);
    break;
    
  case 3:
    gradients_3d_3_points<T>(n1, n2, n3, n2p, dx2, f, v, g);
    break;
  }
}

// For complex					  
template __global__ void gradients_3d<Complex>(const int n1, const int n2, const int n3,
					       const int n2p, const double dx2,
					       const Complex *f, Complex *v, Complex *g, 
					       const int n_points);
// For double
template __global__ void gradients_3d<double>(const int n1, const int n2, const int n3,
					      const int n2p, const double dx2,
					      const double *f, double *v, double *g, 
					      const int n_points);

