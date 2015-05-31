
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
							 const int n2p, const double dx,
							 const T *f, T *v, T *g)
{
  const int index = threadIdx.x + blockDim.x*blockIdx.x;
  
  if(index < n1*n3) {
    int i = -1; int k = -1;
    index_2_ij(index, n1*n2, 0, i, k);

    v[ij_2_index(n1, n3, i, k)] = f[ijk_2_index(n1, n2, n3, i, n2p, k)];
    
    T g_tmp = 
           2*(f[ijk_2_index(n1, n2, n3, i, n2p+5, k)] - f[ijk_2_index(n1, n2, n3, i, n2p-5, k)]) 
      -   25*(f[ijk_2_index(n1, n2, n3, i, n2p+4, k)] - f[ijk_2_index(n1, n2, n3, i, n2p-4, k)]) 
      +  150*(f[ijk_2_index(n1, n2, n3, i, n2p+3, k)] - f[ijk_2_index(n1, n2, n3, i, n2p-3, k)]) 
      -  600*(f[ijk_2_index(n1, n2, n3, i, n2p+2, k)] - f[ijk_2_index(n1, n2, n3, i, n2p-2, k)]) 
      + 2100*(f[ijk_2_index(n1, n2, n3, i, n2p+1, k)] - f[ijk_2_index(n1, n2, n3, i, n2p-1, k)]);

    g[ij_2_index(n1, n3, i, k)] = g_tmp/(2520*dx);
  }
}

template<class T> __global__ void gradients_3d(const int n1, const int n2, const int n3,
					       const int n2p, const double dx,
					       const T *f, T *v, T *g, const int n_points)
{
  switch (n_points) {
    
  case 11:
    gradients_3d_11_points<T>(n1, n2, n3, n2p, dx, f, v, g);
    break;
  }
}

// For complex					  
template __global__ void gradients_3d<Complex>(const int n1, const int n2, const int n3,
					       const int n2p, const double dx,
					       const Complex *f, Complex *v, Complex *g, 
					       const int n_points);
// For double
template __global__ void gradients_3d<double>(const int n1, const int n2, const int n3,
					      const int n2p, const double dx,
					      const double *f, double *v, double *g, 
					      const int n_points);
