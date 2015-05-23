
#include "evolutionCUDA.h"
#include "cumath.h"

__constant__ __device__ double dt;

inline int number_of_blocks(const int n_threads, const int n)
{ return n/n_threads*n_threads == n ? n/n_threads : n/n_threads+1; }

static __global__ void _evolution_with_potential_dt_(Complex *psi, const double *pot, int n)
{
  const int j = threadIdx.x + blockDim.x*blockIdx.x;
  if(j < n) psi[j] *= exp(Complex(0.0, -dt)*pot[j]);
}

void gpu_memory_usage()
{
  size_t free_byte ;
  size_t total_byte ;
  checkCudaErrors(cudaMemGetInfo(&free_byte, &total_byte));
  
  cout << " GPU memory usage:" 
       << " used = " << (total_byte-free_byte)/1024.0/1024.0 << "MB,"
       << " free = " << free_byte/1024.0/1024.0 << "MB,"
       << " total = " << total_byte/1024.0/1024.0 << "MB" <<endl;
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

  setup_cublas_handle();
  setup_cufft_plan();
}

void EvolutionCUDA::deallocate_device_memories()
{
  cout << " Deallocate device memory" << endl;
  
  if(pot_dev) { checkCudaErrors(cudaFree(pot_dev)); pot_dev = 0; }
  if(psi_dev) { checkCudaErrors(cudaFree(psi_dev)); psi_dev = 0; }
  if(work_dev) { checkCudaErrors(cudaFree(work_dev)); work_dev = 0; }
  if(w_dev) { checkCudaErrors(cudaFree(w_dev)); w_dev = 0; }
  if(exp_ipot_dt_dev) { checkCudaErrors(cudaFree(exp_ipot_dt_dev)); exp_ipot_dt_dev = 0; }
  if(legendre_dev) { checkCudaErrors(cudaFree(legendre_dev)); legendre_dev = 0; }
  if(weight_legendre_dev) { checkCudaErrors(cudaFree(weight_legendre_dev)); weight_legendre_dev = 0; }
  if(legendre_psi_dev) { checkCudaErrors(cudaFree(legendre_psi_dev)); legendre_psi_dev = 0; }

  destroy_cublas_handle();
  destroy_cufft_plan();
}

void EvolutionCUDA::setup_cublas_handle()
{
  if(has_cublas_handle) return;
  insist(cublasCreate(&cublas_handle) == CUBLAS_STATUS_SUCCESS);
  has_cublas_handle = 1;
}

void EvolutionCUDA::destroy_cublas_handle()
{
  if(!has_cublas_handle) return;
  insist(cublasDestroy(cublas_handle) == CUBLAS_STATUS_SUCCESS);
  has_cublas_handle = 0;
}

void EvolutionCUDA::setup_cufft_plan()
{
  if(has_cufft_plan) return;

  const int &n1 = r1.n;
  const int &n2 = r2.n;
  const int &n_theta = theta.n;
  
  int dim[] = { n1, n2 };

  insist(cufftPlanMany(&cufft_plan, 2, dim, NULL, 1, n1*n2, NULL, 1, n1*n2,
		       CUFFT_Z2Z, n_theta) == CUFFT_SUCCESS);
  has_cufft_plan = 1;
}

void EvolutionCUDA::destroy_cufft_plan()
{
  if(!has_cufft_plan) return;
  insist(cufftDestroy(cufft_plan) == CUFFT_SUCCESS);
  has_cufft_plan = 0;
}

void EvolutionCUDA::setup_legendre()
{
  if(legendre_dev) return;

  const int &n_theta = theta.n;
  const int m = theta.m + 1;
  const RMat &P = theta.legendre;

  Mat<Complex> P_complex(m, n_theta);
  for(int k = 0; k < n_theta; k++) {
    for(int l = 0; l < m; l++) {
      P_complex(l,k) = Complex(P(l,k), 0.0);
    }
  }
  
  checkCudaErrors(cudaMalloc(&legendre_dev, m*n_theta*sizeof(Complex)));

  checkCudaErrors(cudaMemcpy(legendre_dev, (const Complex *) P_complex,
			     m*n_theta*sizeof(Complex), cudaMemcpyHostToDevice));
}

void EvolutionCUDA::setup_weight_legendre()
{ 
  if(weight_legendre_dev) return;

  const int &n_theta = theta.n;
  const int m = theta.m + 1;
  
  Mat<Complex> weight_legendre(n_theta, m);
  
  const double *w = theta.w;
  const RMat &P = theta.legendre;

  Mat<Complex> &wp = weight_legendre;
  
  for(int l = 0; l < m; l++) {
    const double f = l+0.5;
    for(int k = 0; k < n_theta; k++) {
      wp(k,l) = Complex(f*w[k]*P(l,k), 0.0);
    }
  }
  
  checkCudaErrors(cudaMalloc(&weight_legendre_dev, m*n_theta*sizeof(Complex)));
  
  checkCudaErrors(cudaMemcpy(weight_legendre_dev, (const Complex *) weight_legendre,
			     m*n_theta*sizeof(Complex), cudaMemcpyHostToDevice));
  
}

void EvolutionCUDA::evolution_with_potential_dt()
{
  insist(pot_dev && psi_dev);
  
  const int &n1 = r1.n;
  const int &n2 = r2.n;
  const int &n_theta = theta.n;
  const int n = n1*n2*n_theta;
  
  const int n_threads = 1024;
  const int n_blocks = number_of_blocks(n_threads, n);
  
  _evolution_with_potential_dt_<<<n_blocks, n_threads>>>(psi_dev, pot_dev, n);
}

double EvolutionCUDA::potential_energy()
{
  const int &n1 = r1.n;
  const int &n2 = r2.n;
  const int &n_theta = theta.n;

  const double *w = theta.w;
  
  insist(work_dev);
  cuDoubleComplex *psi_tmp_dev = (cuDoubleComplex *) work_dev;
  
  const int n_threads = 256;
  const int n_blocks = number_of_blocks(n_threads, n1*n2);
  
  double sum = 0.0;
  for(int k = 0; k < n_theta; k++) {
    const cuDoubleComplex *psi_in_dev = (cuDoubleComplex *) psi_dev + k*n1*n2;
    
    _vector_multiplication_<Complex, Complex, double><<<n_blocks, n_threads>>>
      ((Complex *) psi_tmp_dev, (const Complex *) psi_in_dev, pot_dev+k*n1*n2, n1*n2);
    
    checkCudaErrors(cudaDeviceSynchronize());
    
    Complex dot(0.0, 0.0);
    insist(cublasZdotc(cublas_handle, n1*n2, psi_in_dev, 1, psi_tmp_dev, 1, (cuDoubleComplex *) &dot)
	   == CUBLAS_STATUS_SUCCESS);
    
    sum += w[k]*dot.real();
  }

  sum *= r1.dr*r2.dr;
  return sum;
}

double EvolutionCUDA::module_for_psi() const
{
  const int &n1 = r1.n;
  const int &n2 = r2.n;
  const int &n_theta = theta.n;
  
  const double *w = theta.w;
  
  double sum= 0.0;
  for(int k = 0; k < n_theta; k++) {
    Complex dot(0.0, 0.0);
    const cuDoubleComplex *psi_ = (cuDoubleComplex *) psi_dev + k*n1*n2;
    insist(cublasZdotc(cublas_handle, n1*n2, psi_, 1, psi_, 1, (cuDoubleComplex *) &dot)
	   == CUBLAS_STATUS_SUCCESS);
    sum += w[k]*dot.real();
  }

  sum *= r1.dr*r2.dr;
  return sum;
}

void EvolutionCUDA::setup_legendre_psi()
{
  if(legendre_psi_dev) return;

  const int &n1 = r1.n;
  const int &n2 = r2.n;
  const int m = theta.m + 1;
  
  checkCudaErrors(cudaMalloc(&legendre_psi_dev, n1*n2*m*sizeof(Complex)));
  checkCudaErrors(cudaMemset(legendre_psi_dev, 0, n1*n2*m*sizeof(Complex)));
}

void EvolutionCUDA::cuda_fft_test()
{ 
  const int &n1 = r1.n;
  const int &n2 = r2.n;
  const int &n_theta = theta.n;
  
  insist(psi_dev);
  
  StopWatchInterface *timer = 0;
  sdkCreateTimer(&timer);

  const int &total_steps = time.total_steps;
  
  for(int k = 0; k < total_steps; k++) {
    
    cout << " " << k << " ";
    
    sdkResetTimer(&timer); sdkStartTimer(&timer);
    
    insist(cufftExecZ2Z(cufft_plan, (cuDoubleComplex *) psi_dev, (cuDoubleComplex *) psi_dev, 
			CUFFT_FORWARD) == CUFFT_SUCCESS);
    
    insist(cufftExecZ2Z(cufft_plan, (cuDoubleComplex *) psi_dev, (cuDoubleComplex *) psi_dev,
			CUFFT_INVERSE) == CUFFT_SUCCESS);
    
    insist(cudaDeviceSynchronize() == CUFFT_SUCCESS);
    
    const double s = 1.0/(n1*n2);
    insist(cublasZdscal(cublas_handle, n1*n2*n_theta, &s, (cuDoubleComplex *) psi_dev, 1) 
	   == CUBLAS_STATUS_SUCCESS);

    legendre_transform_test();
    
    sdkStopTimer(&timer); cout << "GPU time: " << sdkGetAverageTimerValue(&timer)*1e-3 << endl;
  }
}

void EvolutionCUDA::forward_legendre_transform()
{
  //cout << " Forward Legendre transform" << endl;


  setup_legendre_transform();

  const int &n1 = r1.n;
  const int &n2 = r2.n;
  const int &n_theta = theta.n;
  const int m = theta.m + 1;
  
  const Complex one(1.0, 0.0);
  const Complex zero(0.0, 0.0);

  insist(cublasZgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
		     n1*n2, m, n_theta, 
		     (const cuDoubleComplex *) &one,
		     (const cuDoubleComplex *) psi_dev, n1*n2,
		     (const cuDoubleComplex *) weight_legendre_dev, n_theta,
		     (const cuDoubleComplex *) &zero,
		     (cuDoubleComplex *) legendre_psi_dev, n1*n2) == CUBLAS_STATUS_SUCCESS);
}

void EvolutionCUDA::backward_legendre_transform()
{
  //cout << " Backward Legendre transform" << endl;
  
  setup_legendre_transform();

  const int &n1 = r1.n;
  const int &n2 = r2.n;
  const int &n_theta = theta.n;
  const int m = theta.m + 1;
  
  const Complex one(1.0, 0.0);
  const Complex zero(0.0, 0.0);
  
  insist(cublasZgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
		     n1*n2, n_theta, m,
		     (const cuDoubleComplex *) &one,
		     (const cuDoubleComplex *) legendre_psi_dev, n1*n2,
		     (const cuDoubleComplex *) legendre_dev, m,
		     (const cuDoubleComplex *) &zero,
		     (cuDoubleComplex *) psi_dev, n1*n2) == CUBLAS_STATUS_SUCCESS);
}

void EvolutionCUDA::legendre_transform_test()
{
  forward_legendre_transform();
  backward_legendre_transform();
}
