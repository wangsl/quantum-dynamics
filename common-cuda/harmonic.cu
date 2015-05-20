     
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

void cuda_test()
{
  cout << "Harmonic oscillator test" << endl;
  
  cout << " sizeof(Complex) = " << sizeof(Complex) << endl;
  
  const double pi = Pi;
  const double pi_1_4 = pow(pi, -0.25);

  const int m = 200;
  const int n1 = 512;
  const int n2 = 512;

  Complex *phi = new Complex [n1*n2];
  assert(phi);

  const double xl = 20.0;
  const double dx = xl/(n1-1);

  int k = 0;
  double x = -0.5*xl;
  for(int i = 0; i < n1; i++) {
    const  Complex phix = Complex(pi_1_4*exp(-0.5*x*x), 0.0);
    double y = -0.5*xl;
    for(int j = 0; j < n2; j++) {
      const Complex phiy = Complex(pi_1_4*exp(-0.5*y*y), 0.0);
      y += dx;
      phi[k] = phix*phiy;
      k++;
    }
    x += dx;
  }

  Complex dot = Complex(0.0, 0.0);
  for(int i = 0; i < n1*n2; i++) {
    dot += phi[i]*conj(phi[i]);
  }
  cout << dot.re*dx*dx << "  " << dot.im*dx*dx << endl << endl;
  
  Complex *phiDev = 0;
  checkCudaErrors(cudaMalloc((void **) &phiDev, n1*n2*m*sizeof(Complex)));
  for(int j = 0; j < m; j++)
    checkCudaErrors(cudaMemcpy(phiDev+j*n1*n2, phi, n1*n2*sizeof(Complex), cudaMemcpyHostToDevice));
  
  if(phi) { delete [] phi; phi = 0; }

  StopWatchInterface *timer = 0;
  sdkCreateTimer(&timer);

  cudaEvent_t start, stop;
  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&stop));
  
  // CUFFT plan
  cufftHandle plan;
  checkCudaErrors(cufftPlan2d(&plan, n1, n2, CUFFT_Z2Z));
  
  cublasHandle_t handle;
  assert(cublasCreate(&handle) == CUBLAS_STATUS_SUCCESS);
  
  for(int i = 0; i < 100; i++) {

    cout << i << " ";
    
    sdkResetTimer(&timer);
    sdkStartTimer(&timer);

    cudaEventRecord(start);
    
    for(int j = 0; j < m; j++) {
      cuDoubleComplex *phiDev_ = (cufftDoubleComplex *) phiDev + j*n1*n2;
      dot.zero();
      assert(cublasZdotc(handle, n1*n2, phiDev_, 1, phiDev_, 1, (cufftDoubleComplex *) &dot) 
	     == CUBLAS_STATUS_SUCCESS);
    }

    cout << dot*(dx*dx) << " ";

    for(int j = 0; j < m; j++) {
      cuDoubleComplex *phiDev_ = (cufftDoubleComplex *) phiDev + j*n1*n2;
      checkCudaErrors(cufftExecZ2Z(plan, phiDev_, phiDev_, CUFFT_FORWARD));
    }
    
    for(int j = 0; j < m; j++) {
      cuDoubleComplex *phiDev_ = (cufftDoubleComplex *) phiDev + j*n1*n2;
      checkCudaErrors(cufftExecZ2Z(plan, phiDev_, phiDev_, CUFFT_INVERSE));
      const double s = 1.0/(n1*n2);
      assert(cublasZdscal(handle, n1*n2, &s, phiDev_, 1) == CUBLAS_STATUS_SUCCESS);
      dot.zero();
      assert(cublasZdotc(handle, n1*n2, phiDev_, 1, phiDev_, 1, (cuDoubleComplex *) &dot) 
	     == CUBLAS_STATUS_SUCCESS);
    }

    cout << dot*(dx*dx) << " ";

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    sdkStopTimer(&timer);
    double reduceTime = sdkGetAverageTimerValue(&timer);

    float milliseconds = 0.0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    cout << "GPU time: " << reduceTime*1e-3 << endl; // << "  " << milliseconds*1.0e-3 << endl;
  }

  sdkDeleteTimer(&timer);
  checkCudaErrors(cudaEventDestroy(start));
  checkCudaErrors(cudaEventDestroy(stop));

  assert(cublasDestroy(handle) ==  CUBLAS_STATUS_SUCCESS);

  if(phiDev) { checkCudaErrors(cudaFree(phiDev)); phiDev = 0; }
  if(phi) { delete [] phi; phi = 0; }
}
