
#include "cumath.h"

#if 0
__device__ void cumath::setup_momentum_for_fft(double *p, const int n, const double xl)
{
  if(n/2*2 != n) return;
  
  const double two_Pi_xl = 2*Pi/xl;
  
  for(int i = threadIdx.x; i < n; i += blockDim.x) {
    if(i <= n/2) 
      p[i] = two_pi_xl*i;
    else if(i > n/2)
      p[i] = two_pi_xl*(-n+i);
  }
}
#endif
