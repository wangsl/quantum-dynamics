     
#include <iostream>
using namespace std;
#include <cassert>
#include <iomanip>  

#include <cuda_runtime.h>
#include <cufft.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include <cublas_v2.h>

#include "complex.h"
#include "matutils.h"

#include "mat.h"
#include "fftwinterface.h"
#include "cumath.h"

inline int number_of_blocks(const int n_threads, const int n)
{ return n/n_threads*n_threads == n ? n/n_threads : n/n_threads+1; }

static __global__ void _psi_times_kinitic_energy_(Complex *psiOut, const Complex *psiIn, 
						  const double xl, const int nx, const double dx,
						  const double yl, const int ny, const double dy)
{
  extern __shared__ double s_data[];
  
  double *Tx = (double *) s_data;
  double *Ty = (double *) &Tx[nx];
  
  cumath::setup_kinetic_energy_for_fft(Tx, nx, xl+dx, 1.0);
  cumath::setup_kinetic_energy_for_fft(Ty, ny, yl+dy, 1.0);
  __syncthreads();
  
  const int index = threadIdx.x + blockDim.x*blockIdx.x;
  if(index < nx*ny) {
    int i = -1; int j = -1;
    cumath::index_2_ij(index, nx, ny, i, j);
    psiOut[index] = psiIn[index]*(Tx[i] + Ty[j]);
  }
}

void setup_kinetic_energy_for_fft(double *kin, const int n, const double xl, const double mass)
{
  insist(n/2*2 == n);
  
  const double two_pi_xl = 2*Pi/xl;
  
  for(int i = 0; i < n; i++) {
    if(i < n/2) {
      kin[i] = cumath::sq(two_pi_xl*i)/(mass+mass);
    } else if(i >= n/2) {
      kin[i] = cumath::sq(two_pi_xl*(-n+i))/(mass+mass);
    }
  }
}

void calculate_kinetic_energy(double &ke,
			      const Complex *phi, 
			      const double xl, const int nx, const double dx,
			      const double yl, const int ny, const double dy)
{
  double *kx = new double [nx];
  insist(kx);
  setup_kinetic_energy_for_fft(kx, nx, xl+dx, 1.0);

  double *ky = new double [ny];
  insist(ky);
  setup_kinetic_energy_for_fft(ky, ny, yl+dy, 1.0);
  
  Complex s(0.0, 0.0);
  for(int k = 0; k < nx*ny; k++) {
    int i = -1; int j = -1;
    cumath::index_2_ij(k, nx, ny, i, j);
    s += abs2(phi[k])*(kx[i] + ky[j]);
  }
  
  ke = s.real()*dx*dy/nx/ny;
  
  if(kx) { delete [] kx; kx = 0; }
  if(ky) { delete [] ky; ky = 0; }
}

void cuda_test()
{
  cout << "Harmonic oscillator test" << endl;
  
  cout << " sizeof(Complex) = " << sizeof(Complex) << endl;
  
  const double pi = Pi;
  const double pi_1_4 = pow(pi, -0.25);

  const int m = 160;
  const int nx = 1024;
  const int ny = 2048;

  Complex *phi = new Complex [nx*ny];
  insist(phi);
  
  const double xl = 54.248;
  const double dx = xl/(nx-1); 

  const double yl = 34.896;
  const double dy = yl/(ny-1);

  cout << " dx: " << dx << " dy: " << dy << endl;
  cout << " " << -0.5*xl << " " << -0.5*xl+(nx-1)*dx << endl;
  
  int k = 0;
  double y = -0.5*yl;
  for(int j = 0; j < ny; j++) {
    const Complex phiy = Complex(pi_1_4*sqrt(2.0)*y*exp(-0.5*y*y), 0.0);
    
    double x = -0.5*xl;
    for(int i = 0; i < nx; i++) {
      const Complex phix = Complex(pi_1_4*exp(-0.5*x*x), 0.0);
      phi[k] = phix*phiy;
      x += dx;
      k++;
    }
    y += dy;
  }
  
  Complex dot = Complex(0.0, 0.0);
  for(int i = 0; i < nx*ny; i++) { dot += phi[i]*conj(phi[i]); }
  cout << dot*dx*dy << endl;

  // CPU version test
  FFTWInterface fftw((double *) phi, nx, ny, FFTW_ESTIMATE, 1);
  fftw.forward_transform();
  double ke = 0.0;
  calculate_kinetic_energy(ke, phi, xl, nx, dx, yl, ny, dy);
  fftw.backward_transform();
  for(int i = 0; i < nx*ny; i++) { phi[i] /= nx*ny; }
  cout << " CPU kinetic energy: " << ke << endl;
  
  Complex *phi_dev = 0;
  checkCudaErrors(cudaMalloc((void **) &phi_dev, nx*ny*m*sizeof(Complex)));

  for(int j = 0; j < m; j++)
    checkCudaErrors(cudaMemcpy(phi_dev+j*nx*ny, phi, nx*ny*sizeof(Complex), cudaMemcpyHostToDevice));
  
  Complex *work_dev = 0;
  checkCudaErrors(cudaMalloc(&work_dev, nx*ny*sizeof(Complex)));
  
  StopWatchInterface *timer = 0;
  sdkCreateTimer(&timer);

  /* CUFFT performs FFTs in row-major or C order.
     For example, if the user requests a 3D transform plan for sizes X, Y, and Z,
     CUFFT transforms along Z, Y, and then X. 
     The user can configure column-major FFTs by simply changing the order of size parameters 
     to the plan creation API functions.
  */
  cufftHandle cufft_plan;
  int dim[] = { ny, nx };
  insist(cufftPlanMany(&cufft_plan, 2, dim, NULL, 1, 0, NULL, 1, 0, CUFFT_Z2Z, m) == CUFFT_SUCCESS);
  
  cublasHandle_t cublas_handle;
  insist(cublasCreate(&cublas_handle) == CUBLAS_STATUS_SUCCESS);
  
  for(int i = 0; i < 1000; i++) {
    
    cout << "\n Loop " << i << endl;
    
    sdkResetTimer(&timer); sdkStartTimer(&timer);
    
    dot.zero();
    for(int k = 0; k < m; k++) {
      cuDoubleComplex *phi_dev_ = (cufftDoubleComplex *) phi_dev + k*nx*ny;
      Complex dot_(0.0, 0.0);
      assert(cublasZdotc(cublas_handle, nx*ny, phi_dev_, 1, phi_dev_, 1, (cufftDoubleComplex *) &dot_) 
	     == CUBLAS_STATUS_SUCCESS);
      dot += dot_;
    }
    cout << " Norm before FFT: " << dot.real()*(dx*dy)/m << endl;
    
    checkCudaErrors(cufftExecZ2Z(cufft_plan, (cufftDoubleComplex *) phi_dev, (cufftDoubleComplex *) phi_dev, CUFFT_FORWARD));

    const int n_threads = 512;
    const int n_blocks = number_of_blocks(n_threads, nx*ny);

    cuDoubleComplex *phi_tmp_dev = (cuDoubleComplex *) work_dev;
    
    double kinetic_energy = 0.0;
    double ke_cpu = 0.0;
    for(int k = 0; k < m; k++) {
      const cuDoubleComplex *phi_dev_ = (cuDoubleComplex *) phi_dev + k*nx*ny;
      
      _psi_times_kinitic_energy_<<<n_blocks, n_threads, (nx+ny)*sizeof(double)>>>
	((Complex *) phi_tmp_dev, (const Complex *) phi_dev_, xl, nx, dx, yl, ny, dy);

      checkCudaErrors(cudaDeviceSynchronize());
      
      Complex dot(0.0, 0.0);
      insist(cublasZdotc(cublas_handle, nx*ny, phi_dev_, 1, phi_tmp_dev, 1, (cuDoubleComplex *) &dot)
	     == CUBLAS_STATUS_SUCCESS);
      kinetic_energy += dot.real();

      memset(phi, 0, nx*ny*sizeof(Complex));
      double ke = 0.0;
      checkCudaErrors(cudaMemcpy(phi, phi_dev, nx*ny*sizeof(Complex), cudaMemcpyDeviceToHost));
      calculate_kinetic_energy(ke, phi, xl, nx, dx, yl, ny, dy);
      ke_cpu += ke;
    }

    cout << " Kinetic energy GPU: " << kinetic_energy*dx*dy/(nx*ny*m) << endl;
    cout << " Kinetic energy CPU: " << ke_cpu/m << endl;
    
    dot.zero();
    for(int k = 0; k < m; k++) {
      cuDoubleComplex *phi_dev_ = (cufftDoubleComplex *) phi_dev + k*nx*ny;
      Complex dot_(0.0, 0.0);
      assert(cublasZdotc(cublas_handle, nx*ny, phi_dev_, 1, phi_dev_, 1, (cufftDoubleComplex *) &dot_) 
	     == CUBLAS_STATUS_SUCCESS);
      dot += dot_;
    }
    cout << " Norm after forward FFT: " << dot.real()*(dx*dy)/(m*nx*ny) << endl;

    checkCudaErrors(cufftExecZ2Z(cufft_plan, (cufftDoubleComplex *) phi_dev, (cufftDoubleComplex *) phi_dev, CUFFT_INVERSE));
    
    const double s = 1.0/(nx*ny);
    insist(cublasZdscal(cublas_handle, nx*ny*m, &s, (cuDoubleComplex *) phi_dev, 1) == CUBLAS_STATUS_SUCCESS);

    dot.zero();
    for(int k = 0; k < m; k++) {
      cuDoubleComplex *phiDev_ = (cufftDoubleComplex *) phi_dev + k*nx*ny;
      Complex dot_(0.0, 0.0);
      assert(cublasZdotc(cublas_handle, nx*ny, phiDev_, 1, phiDev_, 1, (cufftDoubleComplex *) &dot_) 
	     == CUBLAS_STATUS_SUCCESS);
      dot += dot_;
    }
    cout << " Norm after backward FFT: " << dot.real()*(dx*dy)/m << endl;
    
    sdkStopTimer(&timer);
    double reduceTime = sdkGetAverageTimerValue(&timer);
    cout << " GPU time: " << reduceTime*1e-3 << endl;
  }

  if(timer) sdkDeleteTimer(&timer);

  insist(cublasDestroy(cublas_handle) == CUBLAS_STATUS_SUCCESS);

  if(phi_dev) { checkCudaErrors(cudaFree(phi_dev)); phi_dev = 0; }
  if(work_dev) { checkCudaErrors(cudaFree(work_dev)); work_dev = 0; }
  if(phi) { delete [] phi; phi = 0; }
}

/*** 

clc
clear all
format long

xL = 54.248
n = 1024;
x = linspace(-xL/2, xL/2, n);

dx = x(2) - x(1)

V = 1/2*x.*x;

f = 1/pi^(1/4)*exp(-1/2*x.^2);

sum(conj(f).*f)*dx

sum(conj(f).*V.*f)*dx

f = fft(f);

sum(conj(f).*f)/n*dx

L = xL + dx
N = n;
k = (2*pi/L)*[0:N/2 (-N/2+1):(-1)];

sum(conj(f).*k.^2.*f)/n*dx/2

=== Output ===

xL =

  54.247999999999998


dx =

   0.053028347996090


ans =

   1.000000000000001


ans =

   0.250000000000000


ans =

   1.000000000000002


L =

  54.301028347996088


ans =

   0.250000000000000

***/
