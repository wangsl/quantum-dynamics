
#include "timeEvolCUDA.h"
#include "cumath.h"

#if 0
__constant__ __device__ double dt;

__global__ void setup_exp_ipot_dt_on_device(Complex *exp_pot, const double *pot, int n)
{
  const int j = threadIdx.x + blockDim.x*blockIdx.x;
  if(j < n) 
    exp_pot[j] = exp(Complex(0.0, -dt)*pot[j]);
}
#endif

void TimeEvolutionCUDA::allocate_device_memories()
{ 
  cout << " Allocate device memory" << endl;
  const int &n1 = r1.n;
  const int &n2 = r2.n;
  const int &n_theta = theta.n;

  cout << n1 << " " << n2 << " " << n_theta << endl;

  // time step
  //checkCudaErrors(cudaMemcpyToSymbol(dt, &time.time_step, sizeof(double)));

  if(!pot_dev) {
    checkCudaErrors(cudaMalloc(&pot_dev, n1*n2*n_theta*sizeof(double)));
    insist(pot);
    checkCudaErrors(cudaMemcpy(pot_dev, pot, n1*n2*n_theta*sizeof(double), cudaMemcpyHostToDevice));
  }

  if(!psi_dev) {
    checkCudaErrors(cudaMalloc(&psi_dev, n1*n2*n_theta*sizeof(Complex)));
    insist(psi);
    checkCudaErrors(cudaMemcpy(psi_dev, psi, n1*n2*n_theta*sizeof(Complex), cudaMemcpyHostToDevice));
  }
  
  if(!work_dev) {
    const int max_dim = max(n1*n2 + n_theta, 100);
    checkCudaErrors(cudaMalloc(&work_dev, max_dim*sizeof(Complex)));
  }
  
  if(!w_dev) {
    checkCudaErrors(cudaMalloc(&w_dev, n_theta*sizeof(double)));
    const double *w = theta.w;
    insist(w);
    checkCudaErrors(cudaMemcpy(w_dev, w, n_theta*sizeof(double), cudaMemcpyHostToDevice));
  }
}

void TimeEvolutionCUDA::deallocate_device_memories()
{
  cout << " Deallocate device memory" << endl;

  if(pot_dev) { checkCudaErrors(cudaFree(pot_dev)); pot_dev = 0; }
  if(psi_dev) { checkCudaErrors(cudaFree(psi_dev)); psi_dev = 0; }
  if(work_dev) { checkCudaErrors(cudaFree(work_dev)); work_dev = 0; }
  if(w_dev) { checkCudaErrors(cudaFree(w_dev)); w_dev = 0; }
  if(exp_ipot_dt_dev) { checkCudaErrors(cudaFree(exp_ipot_dt_dev)); exp_ipot_dt_dev = 0; }
}

void TimeEvolutionCUDA::cuda_fft_test()
{ 
  const int &n1 = r1.n;
  const int &n2 = r2.n;
  const int &n_theta = theta.n;
  
  cublasHandle_t cublas_handle;
  insist(cublasCreate(&cublas_handle) == CUBLAS_STATUS_SUCCESS);
  
  cufftHandle cufft_plan;
  insist(cufftPlan2d(&cufft_plan, n1, n2, CUFFT_Z2Z) == CUFFT_SUCCESS);

  insist(psi_dev);
  
  StopWatchInterface *timer = 0;
  sdkCreateTimer(&timer);

  const int &total_steps = time.total_steps;
  
  for(int k = 0; k < total_steps; k++) {

    cout << "\n " << k << " ";

    sdkResetTimer(&timer);
    sdkStartTimer(&timer);
    
    Complex dot(0.0, 0.0);
    assert(cublasZdotc(cublas_handle, n1*n2*n_theta, (cuDoubleComplex *) psi_dev, 1, 
		       (cuDoubleComplex *) psi_dev, 1, (cufftDoubleComplex *) &dot) 
	   == CUBLAS_STATUS_SUCCESS);
    cout << dot << " ";
    
    for(int l = 0; l < n_theta; l++) {
      cuDoubleComplex *psi_ = (cufftDoubleComplex *) psi_dev + l*n1*n2;
      insist(cufftExecZ2Z(cufft_plan, psi_, psi_, CUFFT_FORWARD) == CUFFT_SUCCESS);
    }
    
    for(int l = 0; l < n_theta; l++) {
      cuDoubleComplex *psi_ = (cufftDoubleComplex *) psi_dev + l*n1*n2;
      insist(cufftExecZ2Z(cufft_plan, psi_, psi_, CUFFT_INVERSE) == CUFFT_SUCCESS);
    }
    
    const double s = 1.0/(n1*n2);
    insist(cublasZdscal(cublas_handle, n1*n2*n_theta, &s, (cuDoubleComplex *) psi_dev, 1) 
	   == CUBLAS_STATUS_SUCCESS);

    sdkStopTimer(&timer);
    double reduceTime = sdkGetAverageTimerValue(&timer);
    
    cout << "GPU time: " << reduceTime*1e-3 << endl;
  }
    
  insist(cufftDestroy(cufft_plan) == CUFFT_SUCCESS);
  insist(cublasDestroy(cublas_handle) ==  CUBLAS_STATUS_SUCCESS);
}

void TimeEvolutionCUDA::cuda_fft_test_with_many_plan()
{ 
  const int &n1 = r1.n;
  const int &n2 = r2.n;
  const int &n_theta = theta.n;
  
  cublasHandle_t cublas_handle;
  insist(cublasCreate(&cublas_handle) == CUBLAS_STATUS_SUCCESS);
  
  int dim[] = { n1, n2 };

  cufftHandle cufft_plan;
  insist(cufftPlanMany(&cufft_plan, 2, dim, NULL, 1, n1*n2, NULL, 1, n1*n2,
		       CUFFT_Z2Z, n_theta) == CUFFT_SUCCESS);
  
  insist(psi_dev);
  
  StopWatchInterface *timer = 0;
  sdkCreateTimer(&timer);

  const int &total_steps = time.total_steps;
  
  for(int k = 0; k < total_steps; k++) {

    cout << "\n " << k << " ";

    sdkResetTimer(&timer); sdkStartTimer(&timer);
    
    insist(cufftExecZ2Z(cufft_plan, (cuDoubleComplex *) psi_dev, (cuDoubleComplex *) psi_dev, 
			CUFFT_FORWARD) == CUFFT_SUCCESS);
    
    insist(cufftExecZ2Z(cufft_plan, (cuDoubleComplex *) psi_dev, (cuDoubleComplex *) psi_dev,
			CUFFT_INVERSE) == CUFFT_SUCCESS);

    insist(cudaDeviceSynchronize() == CUFFT_SUCCESS);
   
    const double s = 1.0/(n1*n2);
    insist(cublasZdscal(cublas_handle, n1*n2*n_theta, &s, (cuDoubleComplex *) psi_dev, 1) 
	   == CUBLAS_STATUS_SUCCESS);
    
    sdkStopTimer(&timer); cout << "GPU time: " << sdkGetAverageTimerValue(&timer)*1e-3 << endl;

    
    sdkResetTimer(&timer); sdkStartTimer(&timer);
    cuda_psi_normal_test();
    sdkStopTimer(&timer); cout << "GPU reduction time: " << sdkGetAverageTimerValue(&timer)*1e-3 << endl;

    sdkResetTimer(&timer); sdkStartTimer(&timer);
    cuda_psi_normal_test_with_stream();
    sdkStopTimer(&timer); cout << "GPU reduction with stream time: " << sdkGetAverageTimerValue(&timer)*1e-3 << endl;

  }
    
  insist(cufftDestroy(cufft_plan) == CUFFT_SUCCESS);
  insist(cublasDestroy(cublas_handle) ==  CUBLAS_STATUS_SUCCESS);
}

void TimeEvolutionCUDA::cuda_psi_normal_test()
{
  const int &n1 = r1.n;
  const int &n2 = r2.n;
  const int &n_theta = theta.n;

  cublasHandle_t cublas_handle;
  insist(cublasCreate(&cublas_handle) == CUBLAS_STATUS_SUCCESS);
  insist(cublasSetPointerMode(cublas_handle, CUBLAS_POINTER_MODE_DEVICE) == CUBLAS_STATUS_SUCCESS);

  Complex *mod_dev = (Complex *) work_dev;
  
  StopWatchInterface *timer = 0;
  sdkCreateTimer(&timer);
  
  sdkResetTimer(&timer); sdkStartTimer(&timer);
  for(int k = 0; k < n_theta; k++) {
    cuDoubleComplex *psi_ = (cuDoubleComplex *) psi_dev + k*n1*n2;
    insist(cublasZdotc(cublas_handle, n1*n2, psi_, 1, psi_, 1, (cuDoubleComplex *) &mod_dev[k])
	   == CUBLAS_STATUS_SUCCESS);
  }
  sdkStopTimer(&timer); cout << "Reduction 1 GPU time: " << sdkGetAverageTimerValue(&timer)*1e-3 << endl;

  insist(cublasDestroy(cublas_handle) == CUBLAS_STATUS_SUCCESS);

  Complex dot(0.0, 0.0);
  const int n_threads = 64;
  
  sdkResetTimer(&timer); sdkStartTimer(&timer);
  checkCudaErrors(cudaMemset(mod_dev+n_theta, 0, sizeof(Complex)));
  DotProduct<double, Complex, Complex>					\
    <<<n_theta/n_threads+1, n_threads, n_threads*sizeof(Complex)>>>(w_dev, mod_dev, mod_dev+n_theta, n_theta);
  dot.zero();
  checkCudaErrors(cudaMemcpy(&dot, mod_dev+n_theta, sizeof(Complex), cudaMemcpyDeviceToHost));
  sdkStopTimer(&timer); cout << "Reduction 2 GPU time: " << sdkGetAverageTimerValue(&timer)*1e-3 << endl;
  dot *= r1.dr*r2.dr;
  cout << dot << endl;
}

void TimeEvolutionCUDA::cuda_psi_normal_test_with_stream()
{
  const int &n1 = r1.n;
  const int &n2 = r2.n;
  const int &n_theta = theta.n;

  cublasHandle_t cublas_handle;
  insist(cublasCreate(&cublas_handle) == CUBLAS_STATUS_SUCCESS);
  insist(cublasSetPointerMode(cublas_handle, CUBLAS_POINTER_MODE_DEVICE) == CUBLAS_STATUS_SUCCESS);

  cudaStream_t *streams = (cudaStream_t *) malloc(n_theta*sizeof(cudaStream_t));
  insist(streams);
  for(int k = 0; k < n_theta; k++) 
    checkCudaErrors(cudaStreamCreate(&streams[k]));
		    
  Complex *mod_dev = (Complex *) work_dev;
  
  StopWatchInterface *timer = 0;
  sdkCreateTimer(&timer);
  
  sdkResetTimer(&timer); sdkStartTimer(&timer);
  for(int k = 0; k < n_theta; k++) {
    insist(cublasSetStream(cublas_handle, streams[k]) == CUBLAS_STATUS_SUCCESS);
    
    cuDoubleComplex *psi_ = (cuDoubleComplex *) psi_dev + k*n1*n2;
    insist(cublasZdotc(cublas_handle, n1*n2, psi_, 1, psi_, 1, (cuDoubleComplex *) &mod_dev[k])
	   == CUBLAS_STATUS_SUCCESS);
  }
  sdkStopTimer(&timer); cout << "Reduction stream 1 GPU time: " << sdkGetAverageTimerValue(&timer)*1e-3 << endl;
  
  for(int k = 0; k < n_theta; k++) 
    checkCudaErrors(cudaStreamDestroy(streams[k]));

  insist(cublasDestroy(cublas_handle) == CUBLAS_STATUS_SUCCESS);

  Complex dot(0.0, 0.0);
  const int n_threads = 64;
  
  sdkResetTimer(&timer); sdkStartTimer(&timer);
  checkCudaErrors(cudaMemset(mod_dev+n_theta, 0, sizeof(Complex)));
  DotProduct<double, Complex, Complex>
    <<<n_theta/n_threads+1, n_threads, n_threads*sizeof(Complex)>>>(w_dev, mod_dev, mod_dev+n_theta, n_theta);
  dot.zero();
  checkCudaErrors(cudaMemcpy(&dot, mod_dev+n_theta, sizeof(Complex), cudaMemcpyDeviceToHost));
  sdkStopTimer(&timer); cout << "Reduction stream 2 GPU time: " << sdkGetAverageTimerValue(&timer)*1e-3 << endl;
  dot *= r1.dr*r2.dr;
  cout << dot << endl;
}


