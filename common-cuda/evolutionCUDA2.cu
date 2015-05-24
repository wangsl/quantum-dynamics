
#include "evolutionCUDA.h"
#include "cumath.h"

// #define IDX2F(i,j,ld) ((((j)-1)*(ld))+((i)-1)) 1-based
// #define IDX2C(i,j,ld) (((j)*(ld))+(i)) 0-based

struct RadialCoordinates
{ 
  double dr;
  double mass;
  int n;
};

__constant__ double dt;
__constant__ RadialCoordinates r1_dev;
__constant__ RadialCoordinates r2_dev;

inline int number_of_blocks(const int n_threads, const int n)
{ return n/n_threads*n_threads == n ? n/n_threads : n/n_threads+1; }

inline void copy_radial_coordinates_to_device(const RadialCoordinates &r, const int &n, 
					      const double &dr, const double &mass)	    
{									
  RadialCoordinates r_h;
  r_h.dr = dr;
  r_h.mass = mass;
  r_h.n = n;
  checkCudaErrors(cudaMemcpyToSymbol(r, &r_h, sizeof(RadialCoordinates))); 
}

static __global__ void setup_momentum_for_fft(double *p, const int n, const double xl)
{ cumath::setup_momentum_for_fft(p, n, xl); }

static __device__ void setup_kinetic_energy_for_fft(double *kin, const int n, const double xl, const double mass)
{ cumath::setup_kinetic_energy_for_fft(kin, n, xl, mass); }

static __global__ void _evolution_with_potential_dt_(Complex *psi, const double *pot, int n)
{
  const int j = threadIdx.x + blockDim.x*blockIdx.x;
  if(j < n) psi[j] *= exp(Complex(0.0, -dt)*pot[j]);
}

static __global__ void _psi_times_kinitic_energy_(Complex *psiOut, const Complex *psiIn, 
						  const double *kin1, const int n1,
						  const double *kin2, const int n2)
{
  extern __shared__ double kin_share[];

  double *sKin1 = (double *) kin_share;
  double *sKin2 = (double *) &sKin1[n1];
  
  setup_kinetic_energy_for_fft(sKin1, r1_dev.n, (r1_dev.n*r1_dev.dr), r1_dev.mass);
  setup_kinetic_energy_for_fft(sKin2, r2_dev.n, (r2_dev.n*r2_dev.dr), r2_dev.mass);
  
  __syncthreads();
  
  const int index = threadIdx.x + blockDim.x*blockIdx.x;
  if(index < n1*n2) {
    int j = index/n1;
    int i = index - j*n1;
    psiOut[index] = psiIn[index]*(sKin1[i] + sKin2[j]);
  }
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

  if(!kinetic_1_dev) {
    checkCudaErrors(cudaMalloc(&kinetic_1_dev, n1*sizeof(double)));
    checkCudaErrors(cudaMemcpy(kinetic_1_dev, r1.psq2m,  n1*sizeof(double), cudaMemcpyHostToDevice));
  }
  
  if(!kinetic_2_dev) {
    checkCudaErrors(cudaMalloc(&kinetic_2_dev, n2*sizeof(double)));
    checkCudaErrors(cudaMemcpy(kinetic_2_dev, r2.psq2m,  n2*sizeof(double), cudaMemcpyHostToDevice));
  }
  
  copy_radial_coordinates_to_device(r1_dev, r1.n, r1.dr, r1.mass);
  copy_radial_coordinates_to_device(r2_dev, r2.n, r2.dr, r2.mass);

  setup_cublas_handle();
}

void EvolutionCUDA::deallocate_device_memories()
{
  cout << " Deallocate device memory" << endl;

#define _CUDA_FREE_(x) if(x) { checkCudaErrors(cudaFree(x)); x = 0; }

  _CUDA_FREE_(pot_dev);
  _CUDA_FREE_(psi_dev);
  _CUDA_FREE_(work_dev);
  _CUDA_FREE_(w_dev);
  _CUDA_FREE_(exp_ipot_dt_dev);
  _CUDA_FREE_(legendre_dev);
  _CUDA_FREE_(weight_legendre_dev);
  _CUDA_FREE_(legendre_psi_dev);
  _CUDA_FREE_(kinetic_1_dev);
  _CUDA_FREE_(kinetic_2_dev);
  
#undef _CUDA_FREE_

  destroy_cublas_handle();
  destroy_cufft_plan_for_psi();
  destroy_cufft_plan_for_legendre_psi();
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

void EvolutionCUDA::setup_cufft_plan_for_psi()
{
  if(has_cufft_plan_for_psi) return;

  const int &n1 = r1.n;
  const int &n2 = r2.n;
  const int &n_theta = theta.n;
  
  int dim[] = { n1, n2 };

  insist(cufftPlanMany(&cufft_plan_for_psi, 2, dim, NULL, 1, n1*n2, NULL, 1, n1*n2,
		       CUFFT_Z2Z, n_theta) == CUFFT_SUCCESS);
  
  has_cufft_plan_for_psi = 1;
}

void EvolutionCUDA::destroy_cufft_plan_for_psi()
{
  if(!has_cufft_plan_for_psi) return;
  insist(cufftDestroy(cufft_plan_for_psi) == CUFFT_SUCCESS);
  has_cufft_plan_for_psi = 0;
}

void EvolutionCUDA::setup_cufft_plan_for_legendre_psi()
{
  if(has_cufft_plan_for_legendre_psi) return;
  
  const int &n1 = r1.n;
  const int &n2 = r2.n;
  const int m = theta.m + 1;
  
  int dim[] = { n1, n2 };
  
  insist(cufftPlanMany(&cufft_plan_for_legendre_psi, 2, dim, NULL, 1, n1*n2, NULL, 1, n1*n2,
		       CUFFT_Z2Z, m) == CUFFT_SUCCESS);
  
  has_cufft_plan_for_legendre_psi = 1;
}

void EvolutionCUDA::destroy_cufft_plan_for_legendre_psi()
{
  if(!has_cufft_plan_for_legendre_psi) return;
  insist(cufftDestroy(cufft_plan_for_legendre_psi) == CUFFT_SUCCESS);
  has_cufft_plan_for_legendre_psi = 0;
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
  
  const int n_threads = 1024;
  const int n_blocks = number_of_blocks(n_threads, n1*n2);
  
  double sum = 0.0;
  for(int k = 0; k < n_theta; k++) {
    const cuDoubleComplex *psi_in_dev = (cuDoubleComplex *) psi_dev + k*n1*n2;
    
    cumath::_vector_multiplication_<Complex, Complex, double><<<n_blocks, n_threads>>>
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
  cout << " === EvolutionCUDA test ===" << endl;

  insist(psi_dev);
  
  StopWatchInterface *timer = 0;
  sdkCreateTimer(&timer);
  
  const int &total_steps = time.total_steps;
  
  for(int k = 0; k < total_steps; k++) {
    
    cout << "\n " << k << " ";
    
    sdkResetTimer(&timer); sdkStartTimer(&timer);
    
    cout << " Module: " << module_for_psi() << endl;
    
    evolution_with_potential_dt();
    
    cout << " Potential energy: " << potential_energy() << endl;
    cout << " Kinetic energy for psi: " << kinetic_energy_for_psi() << endl;
    
    forward_legendre_transform();
    backward_legendre_transform();
    
    sdkStopTimer(&timer); cout << " GPU time: " << sdkGetAverageTimerValue(&timer)*1e-3 << endl;
  }

  cout << " === End of EvolutionCUDA test ===\n" << endl;
}

void EvolutionCUDA::forward_legendre_transform()
{
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

void EvolutionCUDA::forward_fft_for_psi()
{ 
  setup_cufft_plan_for_psi();
  
  insist(cufftExecZ2Z(cufft_plan_for_psi, (cuDoubleComplex *) psi_dev, (cuDoubleComplex *) psi_dev, 
		      CUFFT_FORWARD) == CUFFT_SUCCESS);

  checkCudaErrors(cudaDeviceSynchronize());
}

void EvolutionCUDA::backward_fft_for_psi()
{
  setup_cufft_plan_for_psi();
 
  insist(cufftExecZ2Z(cufft_plan_for_psi, (cuDoubleComplex *) psi_dev, (cuDoubleComplex *) psi_dev, 
		      CUFFT_INVERSE) == CUFFT_SUCCESS);
  
  checkCudaErrors(cudaDeviceSynchronize());
}

double EvolutionCUDA::kinetic_energy_for_psi()
{
  const int &n1 = r1.n;
  const int &n2 = r2.n;
  const int &n_theta = theta.n;

  forward_fft_for_psi();

  const double *w = theta.w;
  
  insist(work_dev);
  cuDoubleComplex *psi_tmp_dev = (cuDoubleComplex *) work_dev;
  
  const int n_threads = 512;
  const int n_blocks = number_of_blocks(n_threads, n1*n2);
  
  double sum = 0.0;
  for(int k = 0; k < n_theta; k++) {
    const cuDoubleComplex *psi_in_dev = (cuDoubleComplex *) psi_dev + k*n1*n2;

    _psi_times_kinitic_energy_<<<n_blocks, n_threads, (n1+n2)*sizeof(double)>>>
    ((Complex *) psi_tmp_dev, (const Complex *) psi_in_dev, 
       kinetic_1_dev, n1, kinetic_2_dev, n2);
    
    checkCudaErrors(cudaDeviceSynchronize());
    
    Complex dot(0.0, 0.0);
    insist(cublasZdotc(cublas_handle, n1*n2, psi_in_dev, 1, psi_tmp_dev, 1, (cuDoubleComplex *) &dot)
	   == CUBLAS_STATUS_SUCCESS);
    
    sum += w[k]*dot.real();
  }

  sum *= r1.dr*r2.dr/n1/n2;
  
  backward_fft_for_psi();
  
  const double s = 1.0/(n1*n2);
  insist(cublasZdscal(cublas_handle, n1*n2*n_theta, &s, (cuDoubleComplex *) psi_dev, 1) 
	 == CUBLAS_STATUS_SUCCESS);

  return sum;
}
