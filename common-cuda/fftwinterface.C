
/* $Id$ */

#include <omp.h>
#include <cstring>
#include "fftwinterface.h"

FFTWInterface::FFTWInterface(double *f_, const int n_, const int flag_) : 
  f(f_), n(n_), m(0), fftw_flag(flag_),
  column_major(0),
  has_forward_plan(0), has_backward_plan(0)
{ }

FFTWInterface::FFTWInterface(double *f_, const int m_, const int n_, 
			     const int flag_, const int column_major_) : 
  f(f_), m(m_), n(n_),
  fftw_flag(flag_),
  column_major(column_major_),
  has_forward_plan(0), has_backward_plan(0)
{ }

FFTWInterface::~FFTWInterface() 
{ 
  if(f) f = 0; 
  
  if(has_forward_plan) {
    fftw_destroy_plan(forward_plan);
    has_forward_plan = 0;
  }
  
  if(has_backward_plan) {
    fftw_destroy_plan(backward_plan);
    has_backward_plan = 0; 
  }
}

void FFTWInterface::create_fftw_plan(int &has_plan, fftw_plan &plan,
				     const int fftw_direction, const int fftw_flag)
{
  if(has_plan) return;
  
  assert(f);
  
  fftw_plan_with_nthreads(omp_get_max_threads());
  
  double *f_copy = 0;

  if(n && !m) {
    // 1d
    f_copy = new double [2*n];
    assert(f_copy);
    memcpy(f_copy, f, 2*n*sizeof(double));
  } else if(n && m) {
    // 2d case
    f_copy = new double [2*m*n];
    assert(f_copy);
    memcpy(f_copy, f, 2*m*n*sizeof(double));
  }
  
  assert(f_copy);
  
  if(n && !m) {
    plan = fftw_plan_dft_1d(n,
			    reinterpret_cast <fftw_complex *> (f),
			    reinterpret_cast <fftw_complex *> (f),
			    fftw_direction, fftw_flag);
    
    memcpy(f, f_copy, 2*n*sizeof(double));
    
  } else if(m && n) {
    int row = column_major ? n : m;
    int col = column_major ? m : n;
    plan = fftw_plan_dft_2d(row, col, 
			    reinterpret_cast <fftw_complex *> (f),
			    reinterpret_cast <fftw_complex *> (f),
			    fftw_direction, fftw_flag);
    
    memcpy(f, f_copy, 2*m*n*sizeof(double));
  }
  
  if(f_copy) { delete [] f_copy; f_copy = 0; }
  
  has_plan = 1;
}

void FFTWInterface::create_forward_plan()
{
  if(has_forward_plan) return;
  create_fftw_plan(has_forward_plan, forward_plan, FFTW_FORWARD, fftw_flag);
}

void FFTWInterface::create_backward_plan()
{
  if(has_backward_plan) return;
  create_fftw_plan(has_backward_plan, backward_plan, FFTW_BACKWARD, fftw_flag);
}

void FFTWInterface::forward_transform()
{
  create_forward_plan();
  fftw_execute(forward_plan);
}

void FFTWInterface::backward_transform()
{
  create_backward_plan();
  fftw_execute(backward_plan);
}

void FFTWInterface::get_momentum_for_fftw(double *p, const int nm, const double xl)
{
  insist(nm/2*2 == nm);
  const double two_Pi_xl = 2.0*Pi/xl;
  const int nm_half = nm/2;
  double p1 = 0;
  for(int i = 0; i <= nm_half; i++) {
    p[i] = p1;
    p1 += two_Pi_xl;
  }
  p1 = two_Pi_xl*(-nm_half+1);
  for(int i = nm_half+1; i < nm; i++) {
    p[i] = p1;
    p1 += two_Pi_xl;
  }
}

void FFTWInterface::get_momentum_for_fftw(RVec &p, const double xl)
{
  get_momentum_for_fftw((double *) p, p.size(), xl);
}

/*** Matlab script to test FFT ***
     
format long

n = 1024;
x = linspace(-10, 10, n);

dx = x(2) - x(1)

V = 1/2*x.*x;

f = 1/pi^(1/4)*exp(-1/2*x.^2);
% f = sqrt(2)/pi^(1/4)*x.*exp(-1/2*x.^2);

sum(conj(f).*f)*dx

sum(conj(f).*V.*f)*dx

f = fft(f);

sum(conj(f).*f)/n*dx

L = n*dx;
N = n;
k = (2*pi/L)*[0:N/2 (-N/2+1):(-1)];

sum(conj(f).*k.^2.*f)/n*dx/2
    
***/

/*** output ***

dx =

   0.019550342130987


ans =

   0.999999999999998


ans =

   0.250000000000000


ans =

   0.999999999999999


ans =

   0.250000000000000

 ***/
