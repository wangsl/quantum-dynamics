
#include "evolutionCUDA.h"
#include "cumath.h"

__constant__ __device__ double dt;

#if 0
template<class T1, class T2, class T3>  
static __global__ void _vector_multiplication_(T1 *vOut, const T2 *vIn1, const T3 *vIn2, const int n)
{
  const int j = threadIdx.x + blockDim.x*blockIdx.x;
  if(j < n) vOut[j] = vIn1[j]*vIn2[j];
}

template __global__ void _vector_multiplication_<Complex, Complex, double>
(Complex *vOut, const Complex *vIn1, const double *vIn2, const int n);
#endif

static __global__ void _evolution_with_potential_dt_(Complex *psi, const double *pot, int n)
{
  const int j = threadIdx.x + blockDim.x*blockIdx.x;
  if(j < n) psi[j] *= exp(Complex(0.0, -dt)*pot[j]);
}

void EvolutionCUDA::allocate_device_memories()
{ 
  cout << " Allocate device memory" << endl;
  const int &n1 = r1.n;
  const int &n2 = r2.n;
  const int &n_theta = theta.n;

  const int n = n1*n2*n_theta;
  
  cout << n1 << " " << n2 << " " << n_theta << " " << n << endl;
  
  // time step
  checkCudaErrors(cudaMemcpyToSymbol(dt, &time.time_step, sizeof(double)));
  
  if(!pot_dev) {
    checkCudaErrors(cudaMalloc(&pot_dev, n*sizeof(double)));
    insist(pot);
    checkCudaErrors(cudaMemcpy(pot_dev, pot, n*sizeof(double), cudaMemcpyHostToDevice));
  }

  if(!psi_dev) {
    checkCudaErrors(cudaMalloc(&psi_dev, n*sizeof(Complex)));
    insist(psi);
    checkCudaErrors(cudaMemcpy(psi_dev, psi, n*sizeof(Complex), cudaMemcpyHostToDevice));
  }
  
  if(!work_dev) {
    const int max_dim = n1*n2 + n_theta + 1024;
    checkCudaErrors(cudaMalloc(&work_dev, max_dim*sizeof(Complex)));
  }
  
  if(!w_dev) {
    checkCudaErrors(cudaMalloc(&w_dev, n_theta*sizeof(double)));
    const double *w = theta.w;
    insist(w);
    checkCudaErrors(cudaMemcpy(w_dev, w, n_theta*sizeof(double), cudaMemcpyHostToDevice));
  }
}

void EvolutionCUDA::deallocate_device_memories()
{
  cout << " Deallocate device memory" << endl;
  
  if(pot_dev) { checkCudaErrors(cudaFree(pot_dev)); pot_dev = 0; }
  if(psi_dev) { checkCudaErrors(cudaFree(psi_dev)); psi_dev = 0; }
  if(work_dev) { checkCudaErrors(cudaFree(work_dev)); work_dev = 0; }
  if(w_dev) { checkCudaErrors(cudaFree(w_dev)); w_dev = 0; }
  if(exp_ipot_dt_dev) { checkCudaErrors(cudaFree(exp_ipot_dt_dev)); exp_ipot_dt_dev = 0; }
}

double EvolutionCUDA::module_for_psi() const
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
  Complex *dot_dev = mod_dev + n_theta;
  
  StopWatchInterface *timer = 0;
  sdkCreateTimer(&timer);
  
  sdkResetTimer(&timer); sdkStartTimer(&timer);

  for(int k = 0; k < n_theta; k++) {
    insist(cublasSetStream(cublas_handle, streams[k]) == CUBLAS_STATUS_SUCCESS);
    
    cuDoubleComplex *psi_ = (cuDoubleComplex *) psi_dev + k*n1*n2;
    insist(cublasZdotc(cublas_handle, n1*n2, psi_, 1, psi_, 1, (cuDoubleComplex *) &mod_dev[k])
	   == CUBLAS_STATUS_SUCCESS);
  }
  
  for(int k = 0; k < n_theta; k++) 
    checkCudaErrors(cudaStreamDestroy(streams[k]));
  
  insist(cublasDestroy(cublas_handle) == CUBLAS_STATUS_SUCCESS);
  sdkStopTimer(&timer); cout << " module GPU time: " << sdkGetAverageTimerValue(&timer)*1e-3 << endl;
  
  Complex dot(0.0, 0.0);
  const int n_threads = 512;
  const int n_blocks = n_theta/n_threads*n_threads == n_theta ? n_theta/n_threads : n_theta/n_threads+1;
  
  checkCudaErrors(cudaMemset(dot_dev, 0, sizeof(Complex)));
  DotProduct<double, Complex, Complex>
    <<<n_blocks, n_threads, n_threads*sizeof(Complex)>>>(w_dev, mod_dev, dot_dev, n_theta);
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaMemcpy(&dot, dot_dev, sizeof(Complex), cudaMemcpyDeviceToHost));
  //sdkStopTimer(&timer); cout << " module GPU time: " << sdkGetAverageTimerValue(&timer)*1e-3 << endl;
  
  dot *= r1.dr*r2.dr;
  cout << dot << endl;

  return dot.real();
}

double EvolutionCUDA::potential_energy()
{
  const int &n1 = r1.n;
  const int &n2 = r2.n;
  const int &n_theta = theta.n;
  const int n = n1*n2*n_theta;

  cublasHandle_t cublas_handle;
  insist(cublasCreate(&cublas_handle) == CUBLAS_STATUS_SUCCESS);
  insist(cublasSetPointerMode(cublas_handle, CUBLAS_POINTER_MODE_DEVICE) == CUBLAS_STATUS_SUCCESS);

  insist(work_dev);

  StopWatchInterface *timer = 0;
  sdkCreateTimer(&timer);

  cuDoubleComplex *psi_tmp_dev = (cuDoubleComplex *) work_dev;
  cuDoubleComplex *dot_dev = psi_tmp_dev + n1*n2;
  Complex *potential_energy_dev = (Complex *) (dot_dev + n_theta);

  sdkResetTimer(&timer); sdkStartTimer(&timer);

  for(int k = 0; k < n_theta; k++) {
    const cuDoubleComplex *psi_in_dev = (cuDoubleComplex *) psi_dev + k*n1*n2;

    const int n_threads = 512;
    const int n_blocks = (n1*n2)/n_threads*n_threads == n1*n2 ? (n1*n2)/n_threads : (n1*n2)/n_threads+1;

    //_psi_times_pot_<<<n_blocks, n_threads>>>((Complex *) psi_tmp_dev, (const Complex *) psi_in_dev, 
    //pot_dev+k*n1*n2, n1*n2);

    _vector_multiplication_<Complex, Complex, double><<<n_blocks, n_threads>>>
      ((Complex *) psi_tmp_dev, (const Complex *) psi_in_dev, pot_dev+k*n1*n2, n1*n2);

    checkCudaErrors(cudaDeviceSynchronize());
    
    insist(cublasZdotc(cublas_handle, n1*n2, psi_in_dev, 1, psi_tmp_dev, 1, &dot_dev[k])
	   == CUBLAS_STATUS_SUCCESS);
  }

  insist(cublasDestroy(cublas_handle) == CUBLAS_STATUS_SUCCESS);
  sdkStopTimer(&timer); cout << " potential energy GPU time: " << sdkGetAverageTimerValue(&timer)*1e-3 << endl;

  insist(w_dev);
  
  const int n_threads = 512;
  const int n_blocks = n_theta/n_threads*n_threads == n_theta ? n_theta/n_threads : n_theta/n_threads+1;
  
  checkCudaErrors(cudaMemset(potential_energy_dev, 0, sizeof(Complex)));
  DotProduct<double, Complex, Complex>
    <<<n_blocks, n_threads, n_threads*sizeof(Complex)>>>(w_dev, (const Complex *) dot_dev, potential_energy_dev, n_theta);
  checkCudaErrors(cudaDeviceSynchronize());

  Complex pot_e(0.0, 0.0);
  checkCudaErrors(cudaMemcpy(&pot_e, potential_energy_dev, sizeof(Complex), cudaMemcpyDeviceToHost));

  //sdkStopTimer(&timer); cout << " potential energy GPU time: " << sdkGetAverageTimerValue(&timer)*1e-3 << endl;
  
  pot_e *= r1.dr*r2.dr;
  return pot_e.real();
}

void EvolutionCUDA::evolution_with_potential_dt()
{
  insist(pot_dev && psi_dev);

  const int &n1 = r1.n;
  const int &n2 = r2.n;
  const int &n_theta = theta.n;
  const int n = n1*n2*n_theta;
  
  const int n_threads = 512;
  const int n_blocks = n/n_threads*n_threads == n ? n/n_threads : n/n_threads+1;
  
  StopWatchInterface *timer = 0;
  sdkCreateTimer(&timer);
  
  sdkResetTimer(&timer); sdkStartTimer(&timer);
  _evolution_with_potential_dt_<<<n_blocks, n_threads>>>(psi_dev, pot_dev, n);
  sdkStopTimer(&timer); cout << " evolution_with_potential_dt GPU time: " << sdkGetAverageTimerValue(&timer)*1e-3 << endl;
}


double EvolutionCUDA::potential_energy2()
{
  const int &n1 = r1.n;
  const int &n2 = r2.n;
  const int &n_theta = theta.n;
  const int n = n1*n2*n_theta;

  cublasHandle_t cublas_handle;
  insist(cublasCreate(&cublas_handle) == CUBLAS_STATUS_SUCCESS);
  insist(cublasSetPointerMode(cublas_handle, CUBLAS_POINTER_MODE_DEVICE) == CUBLAS_STATUS_SUCCESS);

  insist(work_dev);

  StopWatchInterface *timer = 0;
  sdkCreateTimer(&timer);

  cuDoubleComplex *psi_tmp_dev = (cuDoubleComplex *) work_dev;
  cuDoubleComplex *dot_dev = psi_tmp_dev + n1*n2;
  Complex *potential_energy_dev = (Complex *) (dot_dev + n_theta);

  sdkResetTimer(&timer); sdkStartTimer(&timer);

  for(int k = 0; k < n_theta; k++) {
    const cuDoubleComplex *psi_in_dev = (cuDoubleComplex *) psi_dev + k*n1*n2;

    const int n_threads = 1024;
    const int n_blocks = (n1*n2)/n_threads*n_threads == n1*n2 ? (n1*n2)/n_threads : (n1*n2)/n_threads+1;

    //_psi_times_pot_<<<n_blocks, n_threads>>>((Complex *) psi_tmp_dev, (const Complex *) psi_in_dev, 
    //pot_dev+k*n1*n2, n1*n2);
    
    _vector_multiplication_<Complex, Complex, double><<<n_blocks, n_threads>>>
      ((Complex *) psi_tmp_dev, (const Complex *) psi_in_dev, pot_dev+k*n1*n2, n1*n2);
    
    checkCudaErrors(cudaDeviceSynchronize());
    
    insist(cublasZdotc(cublas_handle, n1*n2, psi_in_dev, 1, psi_tmp_dev, 1, &dot_dev[k])
	   == CUBLAS_STATUS_SUCCESS);
  }

  insist(cublasDestroy(cublas_handle) == CUBLAS_STATUS_SUCCESS);

  insist(w_dev);
  
  const int n_threads = 512;
  const int n_blocks = n_theta/n_threads*n_threads == n_theta ? n_theta/n_threads : n_theta/n_threads+1;
  
  checkCudaErrors(cudaMemset(potential_energy_dev, 0, sizeof(Complex)));
  DotProduct<double, Complex, Complex>
    <<<n_blocks, n_threads, n_threads*sizeof(Complex)>>>(w_dev, (const Complex *) dot_dev, potential_energy_dev, n_theta);
  checkCudaErrors(cudaDeviceSynchronize());

  Complex pot_e(0.0, 0.0);
  checkCudaErrors(cudaMemcpy(&pot_e, potential_energy_dev, sizeof(Complex), cudaMemcpyDeviceToHost));

  sdkStopTimer(&timer); cout << " potential energy GPU time: " << sdkGetAverageTimerValue(&timer)*1e-3 << endl;
  
  pot_e *= r1.dr*r2.dr;
  return pot_e.real();
}

double EvolutionCUDA::module_for_psi_withou_streams() const
{
  const int &n1 = r1.n;
  const int &n2 = r2.n;
  const int &n_theta = theta.n;
  
  cublasHandle_t cublas_handle;
  insist(cublasCreate(&cublas_handle) == CUBLAS_STATUS_SUCCESS);
  insist(cublasSetPointerMode(cublas_handle, CUBLAS_POINTER_MODE_DEVICE) == CUBLAS_STATUS_SUCCESS);

  Complex *mod_dev = (Complex *) work_dev;
  Complex *dot_dev = mod_dev + n_theta;
  
  StopWatchInterface *timer = 0;
  sdkCreateTimer(&timer);
  
  sdkResetTimer(&timer); sdkStartTimer(&timer);

  for(int k = 0; k < n_theta; k++) {
    cuDoubleComplex *psi_ = (cuDoubleComplex *) psi_dev + k*n1*n2;
    insist(cublasZdotc(cublas_handle, n1*n2, psi_, 1, psi_, 1, (cuDoubleComplex *) &mod_dev[k])
	   == CUBLAS_STATUS_SUCCESS);
  }
  
  insist(cublasDestroy(cublas_handle) == CUBLAS_STATUS_SUCCESS);
  sdkStopTimer(&timer); cout << " module without streams GPU time: " << sdkGetAverageTimerValue(&timer)*1e-3 << endl;
  
  Complex dot(0.0, 0.0);
  const int n_threads = 512;
  const int n_blocks = n_theta/n_threads*n_threads == n_theta ? n_theta/n_threads : n_theta/n_threads+1;
  
  checkCudaErrors(cudaMemset(dot_dev, 0, sizeof(Complex)));
  DotProduct<double, Complex, Complex>
    <<<n_blocks, n_threads, n_threads*sizeof(Complex)>>>(w_dev, mod_dev, dot_dev, n_theta);
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaMemcpy(&dot, dot_dev, sizeof(Complex), cudaMemcpyDeviceToHost));
  //sdkStopTimer(&timer); cout << " module GPU time: " << sdkGetAverageTimerValue(&timer)*1e-3 << endl;
  
  dot *= r1.dr*r2.dr;
  cout << dot << endl;

  return dot.real();
}
