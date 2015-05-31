
#include "evolutionCUDA.h"
#include "cumath.h"
#include "gradients.h"

void cuda_test();

// #define IDX2F(i,j,ld) ((((j)-1)*(ld))+((i)-1)) 1-based
// #define IDX2C(i,j,ld) (((j)*(ld))+(i)) 0-based

struct RadialCoordinates
{ 
  double dr;
  double r_left;
  double mass;
  int n;
};

__constant__ RadialCoordinates r1_dev;
__constant__ RadialCoordinates r2_dev;
__constant__ double dump1_dev[1024];
__constant__ double dump2_dev[1024];
__constant__ double energies_dev[1024];
__constant__ double legendre_weight_dev[256];

inline int number_of_blocks(const int n_threads, const int n)
{ return n/n_threads*n_threads == n ? n/n_threads : n/n_threads+1; }

inline void copy_radial_coordinates_to_device(const RadialCoordinates &r, const int n, 
					      const double dr, const double r_left, const double mass)	    
{									
  RadialCoordinates r_h;
  r_h.dr = dr;
  r_h.r_left = r_left;
  r_h.mass = mass;
  r_h.n = n;
  checkCudaErrors(cudaMemcpyToSymbol(r, &r_h, sizeof(RadialCoordinates))); 
}

static __global__ void _evolution_with_potential_(Complex *psi, const double *pot, int n, const double dt)
{
  const int index = threadIdx.x + blockDim.x*blockIdx.x;
  if(index < n) psi[index] *= exp(Complex(0.0, -dt)*pot[index]);
}

static __global__ void _evolution_with_kinetic_(Complex *psi, const int n1, const int n2, const int m, 
						const double dt)
{
  extern __shared__ double s_data[];
  
  double *kin1 = (double *) s_data;
  double *kin2 = (double *) &kin1[n1];
  
  cumath::setup_kinetic_energy_for_fft(kin1, r1_dev.n, r1_dev.n*r1_dev.dr, r1_dev.mass);
  cumath::setup_kinetic_energy_for_fft(kin2, r2_dev.n, r2_dev.n*r2_dev.dr, r2_dev.mass);
  __syncthreads();

  const int index = threadIdx.x + blockDim.x*blockIdx.x;
  if(index < n1*n2*m) {
    int i = -1; int j = -1; int k = -1;
    cumath::index_2_ijk(index, n1, n2, m, i, j, k);
    psi[index] *= exp(Complex(0.0, -dt)*(kin1[i]+kin2[j]));
  }
}

static __global__ void _evolution_with_rotational_(Complex *psi, const int n1, const int n2, const int m,
						   const double dt)
{
  extern __shared__ double s_data[];
  
  double *I1 = (double *) s_data;
  double *I2 = (double *) &I1[n1];

  cumath::setup_moments_of_inertia(I1, r1_dev.n, r1_dev.r_left, r1_dev.dr, r1_dev.mass);
  cumath::setup_moments_of_inertia(I2, r2_dev.n, r2_dev.r_left, r2_dev.dr, r2_dev.mass);
  __syncthreads();
  
  const int index = threadIdx.x + blockDim.x*blockIdx.x;
  if(index < n1*n2*m) {
    int i = -1; int j = -1; int l = -1;
    cumath::index_2_ijk(index, n1, n2, m, i, j, l);
    psi[index] *= exp(-Complex(0.0, 1.0)*dt*l*(l+1)*(I1[i]+I2[j]));
  }
}

static __global__ void _psi_times_kinitic_energy_(Complex *psiOut, const Complex *psiIn, 
						  const int n1, const int n2)
{
  extern __shared__ double s_data[];

  double *kin1 = (double *) s_data;
  double *kin2 = (double *) &kin1[n1];
  
  cumath::setup_kinetic_energy_for_fft(kin1, r1_dev.n, r1_dev.n*r1_dev.dr, r1_dev.mass);
  cumath::setup_kinetic_energy_for_fft(kin2, r2_dev.n, r2_dev.n*r2_dev.dr, r2_dev.mass);
  __syncthreads();

  const int index = threadIdx.x + blockDim.x*blockIdx.x;
  if(index < n1*n2) {
    int i = -1; int j = -1;
    cumath::index_2_ij(index, n1, n2, i, j);
    psiOut[index] = psiIn[index]*(kin1[i] + kin2[j]);
  }
}

static __global__ void _legendre_psi_times_moments_of_inertia_(Complex *psiOut, const Complex *psiIn, 
							      const int n1, const int n2)
{
  extern __shared__ double s_data[];

  double *I1 = (double *) s_data;
  double *I2 = (double *) &I1[n1];
  
  cumath::setup_moments_of_inertia(I1, r1_dev.n, r1_dev.r_left, r1_dev.dr, r1_dev.mass);
  cumath::setup_moments_of_inertia(I2, r2_dev.n, r2_dev.r_left, r2_dev.dr, r2_dev.mass);
  __syncthreads();
  
  const int index = threadIdx.x + blockDim.x*blockIdx.x;
  if(index < n1*n2) {
    int i = -1; int j = -1;
    cumath::index_2_ij(index, n1, n2, i, j);
    psiOut[index] = psiIn[index]*(I1[i] + I2[j]);
  }
}

static __global__ void _dump_wavepacket_(Complex *psi, const int n1, const int n2, const int n_theta)
{
  const int index = threadIdx.x + blockDim.x*blockIdx.x;
  if(index < n1*n2*n_theta) {
    int i = -1; int j = -1; int k = -1;
    cumath::index_2_ijk(index, n1, n2, n_theta, i, j, k);
    psi[index] *= dump1_dev[i]*dump2_dev[j];
  }
}

static __global__ void _psi_time_to_fai_energy_on_surface_(const int n, const int nE,
							   const double t, const double dt,
							   Complex *psi, Complex *fai,
							   Complex *dpsi, Complex *dfai)
{
  extern __shared__ Complex exp_iet_dt[];
  
  for(int i = threadIdx.x; i < nE; i += blockDim.x) 
    exp_iet_dt[i] = exp(Complex(0.0, t)*energies_dev[i])*dt;
  __syncthreads();

  const int index = threadIdx.x + blockDim.x*blockIdx.x;
  
  if(index < n*nE) {
    int i = -1; int iE = -1;
    cumath::index_2_ij(index, n, nE, i, iE);
    fai[index] += exp_iet_dt[iE] * psi[index];
    dfai[index] += exp_iet_dt[iE] * dpsi[index];
  }
}

static __global__ void _fai_on_dividing_surface_with_legendre_weight_(const int n1, const int n_theta,
								      const int nE, 
								      Complex *fai, const int op)
{
  const int index = threadIdx.x + blockDim.x*blockIdx.x;
  if(index < n1*n_theta*nE) {
    int i = -1; int k = -1; int iE = -1;
    cumath::index_2_ijk(index, n1, n_theta, nE, i, k, iE);
    if(op == 1)
      fai[index] *= legendre_weight_dev[k];
    else if(op == -1)
      fai[index] /= legendre_weight_dev[k];
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
  
  cout << " Wavepacket size: " << n1 << " " << n2 << " " << n_theta << " " << n << endl;
  
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

  if(apply_dump()) {
    size_t size = 0;
    
    checkCudaErrors(cudaGetSymbolSize(&size, dump1_dev));
    insist(size/sizeof(double) > n1);
    checkCudaErrors(cudaMemcpyToSymbol(dump1_dev, dump1.dump, n1*sizeof(double)));

    checkCudaErrors(cudaGetSymbolSize(&size, dump2_dev));
    insist(size/sizeof(double) > n2);
    checkCudaErrors(cudaMemcpyToSymbol(dump2_dev, dump2.dump, n2*sizeof(double)));
  }

  if(CRP.calculate_CRP) setup_CRP_data_on_device();

  copy_radial_coordinates_to_device(r1_dev, r1.n, r1.dr, r1.r[0], r1.mass);
  copy_radial_coordinates_to_device(r2_dev, r2.n, r2.dr, r2.r[0], r2.mass);

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
  _CUDA_FREE_(legendre_dev);
  _CUDA_FREE_(weight_legendre_dev);
  _CUDA_FREE_(legendre_psi_dev);
  
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
  
  int dim[] = { n2, n1 };
  
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
  
  /* CUFFT performs FFTs in row-major or C order.
     For example, if the user requests a 3D transform plan for sizes X, Y, and Z,
     CUFFT transforms along Z, Y, and then X. 
     The user can configure column-major FFTs by simply changing the order of size parameters 
     to the plan creation API functions.
  */
  int dim[] = { n2, n1 };
  
  insist(cufftPlanMany(&cufft_plan_for_legendre_psi, 2, dim, 
		       NULL, 1, 0, NULL, 1, 0, CUFFT_Z2Z, m) == CUFFT_SUCCESS);
  
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

void EvolutionCUDA::evolution_with_potential(const double dt)
{
  insist(pot_dev && psi_dev);
  
  const int &n1 = r1.n;
  const int &n2 = r2.n;
  const int &n_theta = theta.n;
  const int n = n1*n2*n_theta;
  
  const int n_threads = 512;
  const int n_blocks = number_of_blocks(n_threads, n);
  
  _evolution_with_potential_<<<n_blocks, n_threads>>>(psi_dev, pot_dev, n, dt);
}

double EvolutionCUDA::potential_energy()
{
  const int &n1 = r1.n;
  const int &n2 = r2.n;
  const int &n_theta = theta.n;

  const double *w = theta.w;
  
  insist(work_dev);
  cuDoubleComplex *psi_tmp_dev = (cuDoubleComplex *) work_dev;
  
  const int n_threads = 512;
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

  time_evolution();
  
  cout << " === End of EvolutionCUDA test ===\n" << endl;
}

void EvolutionCUDA::time_evolution()
{
  insist(psi_dev);
  
  const int &total_steps = time.total_steps;
  int &steps = time.steps;
  const double &dt = time.time_step;
  
  StopWatchInterface *timer = 0;
  sdkCreateTimer(&timer);
  
  for(int k = 0; k < total_steps; k++) {
    
    cout << "\n Step: " << k << endl;

    sdkResetTimer(&timer); sdkStartTimer(&timer);

    if(k == 0 && steps == 0) evolution_with_potential(-dt/2);
    
    evolution_with_potential(dt);
    
    forward_legendre_transform();
    
    evolution_with_rotational(dt/2);
    
    forward_fft_for_legendre_psi();
    
    evolution_with_kinetic(dt);
    
    const double e_kin = kinetic_energy_for_legendre_psi(0);
    
    backward_fft_for_legendre_psi(1);
    
    evolution_with_rotational(dt/2);

    const double e_rot = rotational_energy(0);
    
    backward_legendre_transform();

    const double e_pot = potential_energy();
    const double module = module_for_psi();
    
    cout << " e_kin: " << e_kin << "\n"
	 << " e_rot: " << e_rot << "\n"
	 << " e_pot: " << e_pot << "\n"
	 << " e_tot: " << e_kin + e_rot + e_pot << "\n"
	 << " module: " << module << endl;

    steps++;
    
    dump_wavepacket();
    
    const int calculate_CRP = steps%options.steps_to_copy_psi_from_device_to_host == 0 ? 1 : 0;
    //calculate_reaction_probabilities(calculate_CRP, (k+1)*dt);

    if(options.wave_to_matlab && steps%options.steps_to_copy_psi_from_device_to_host == 0) {
      copy_psi_from_device_to_host();
      wavepacket_to_matlab(options.wave_to_matlab);
    }
    
    sdkStopTimer(&timer); cout << " GPU time: " << sdkGetAverageTimerValue(&timer)*1e-3 << endl;

    cout.flush();
  }
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
}

void EvolutionCUDA::backward_fft_for_psi(const int do_scale)
{
  setup_cufft_plan_for_psi();
  
  insist(cufftExecZ2Z(cufft_plan_for_psi, (cuDoubleComplex *) psi_dev, (cuDoubleComplex *) psi_dev, 
		      CUFFT_INVERSE) == CUFFT_SUCCESS);
  
  if(do_scale) {
    const int &n1 = r1.n;
    const int &n2 = r2.n;
    const int &n_theta = theta.n;
    const double s = 1.0/(n1*n2);
    insist(cublasZdscal(cublas_handle, n1*n2*n_theta, &s, (cuDoubleComplex *) psi_dev, 1) 
	   == CUBLAS_STATUS_SUCCESS);
    
  }
}

double EvolutionCUDA::kinetic_energy_for_psi(const int do_fft)
{
  const int &n1 = r1.n;
  const int &n2 = r2.n;
  const int &n_theta = theta.n;
  
  if(do_fft) forward_fft_for_psi();
  
  const double *w = theta.w;
  
  insist(work_dev);
  cuDoubleComplex *psi_tmp_dev = (cuDoubleComplex *) work_dev;
  
  const int n_threads = 512;
  const int n_blocks = number_of_blocks(n_threads, n1*n2);
  
  double sum = 0.0;
  for(int k = 0; k < n_theta; k++) {
    const cuDoubleComplex *psi_in_dev = (cuDoubleComplex *) psi_dev + k*n1*n2;
    
    _psi_times_kinitic_energy_<<<n_blocks, n_threads, (n1+n2)*sizeof(double)>>>
      ((Complex *) psi_tmp_dev, (const Complex *) psi_in_dev, n1, n2);
    
    checkCudaErrors(cudaDeviceSynchronize());
    
    Complex dot(0.0, 0.0);
    insist(cublasZdotc(cublas_handle, n1*n2, psi_in_dev, 1, psi_tmp_dev, 1, (cuDoubleComplex *) &dot)
	   == CUBLAS_STATUS_SUCCESS);
    
    sum += w[k]*dot.real();
  }
  
  sum *= r1.dr*r2.dr/n1/n2;
  
  if(do_fft) backward_fft_for_psi(1);
  
  return sum;
}

void EvolutionCUDA::forward_fft_for_legendre_psi()
{ 
  setup_cufft_plan_for_legendre_psi();
  
  insist(cufftExecZ2Z(cufft_plan_for_legendre_psi, 
		      (cuDoubleComplex *) legendre_psi_dev, (cuDoubleComplex *) legendre_psi_dev, 
		      CUFFT_FORWARD) == CUFFT_SUCCESS);

  checkCudaErrors(cudaDeviceSynchronize());
}

void EvolutionCUDA::backward_fft_for_legendre_psi(const int do_scale)
{
  setup_cufft_plan_for_legendre_psi();
  
  insist(cufftExecZ2Z(cufft_plan_for_legendre_psi, 
		      (cuDoubleComplex *) legendre_psi_dev, (cuDoubleComplex *) legendre_psi_dev, 
		      CUFFT_INVERSE) == CUFFT_SUCCESS);
  
  checkCudaErrors(cudaDeviceSynchronize());
  
  if(do_scale) {
    const int &n1 = r1.n;
    const int &n2 = r2.n;
    const int m = theta.m + 1;
    
    const double s = 1.0/(n1*n2);
    insist(cublasZdscal(cublas_handle, n1*n2*m, &s, (cuDoubleComplex *) legendre_psi_dev, 1) 
	   == CUBLAS_STATUS_SUCCESS);
  }
}

double EvolutionCUDA::kinetic_energy_for_legendre_psi(const int do_fft)
{
  const int &n1 = r1.n;
  const int &n2 = r2.n;
  const int m = theta.m + 1;
  
  if(do_fft) forward_fft_for_legendre_psi();

  insist(work_dev);
  cuDoubleComplex *psi_tmp_dev = (cuDoubleComplex *) work_dev;
  
  const int n_threads = 512;
  const int n_blocks = number_of_blocks(n_threads, n1*n2);
  
  double sum = 0.0;
  for(int l = 0; l < m; l++) {
    const cuDoubleComplex *legendre_psi_in_dev = (cuDoubleComplex *) legendre_psi_dev + l*n1*n2;
    
    _psi_times_kinitic_energy_<<<n_blocks, n_threads, (n1+n2)*sizeof(double)>>>
      ((Complex *) psi_tmp_dev, (const Complex *) legendre_psi_in_dev, n1, n2);
    
    checkCudaErrors(cudaDeviceSynchronize());
    
    Complex dot(0.0, 0.0);
    insist(cublasZdotc(cublas_handle, n1*n2, legendre_psi_in_dev, 1, psi_tmp_dev, 1, (cuDoubleComplex *) &dot)
	   == CUBLAS_STATUS_SUCCESS);
    
    sum += 2.0/(2*l+1)*dot.real();
  }

  sum *= r1.dr*r2.dr/n1/n2;

  if(do_fft) backward_fft_for_legendre_psi(1);
  
  return sum;
}

double EvolutionCUDA::rotational_energy(const int do_legendre_transform)
{
  const int &n1 = r1.n;
  const int &n2 = r2.n;
  const int m = theta.m + 1;
  
  if(do_legendre_transform) forward_legendre_transform();
  
  insist(work_dev);
  cuDoubleComplex *psi_tmp_dev = (cuDoubleComplex *) work_dev;

  const int n_threads = 512;
  const int n_blocks = number_of_blocks(n_threads, n1*n2);
  
  double sum = 0.0;
  for(int l = 0; l < m; l++) {
    const cuDoubleComplex *legendre_psi_in_dev = (cuDoubleComplex *) legendre_psi_dev + l*n1*n2;
    
    _legendre_psi_times_moments_of_inertia_<<<n_blocks, n_threads, (n1+n2)*sizeof(double)>>>
      ((Complex *) psi_tmp_dev, (const Complex *) legendre_psi_in_dev, n1, n2);
    
    checkCudaErrors(cudaDeviceSynchronize());
    
    Complex dot(0.0, 0.0);
    insist(cublasZdotc(cublas_handle, n1*n2, legendre_psi_in_dev, 1, psi_tmp_dev, 1, (cuDoubleComplex *) &dot)
	   == CUBLAS_STATUS_SUCCESS);
    
    sum += l*(l+1)/(l+0.5)*dot.real();
  }

  sum *= r1.dr*r2.dr;

  if(do_legendre_transform) backward_legendre_transform();
  
  return sum;
}

double EvolutionCUDA::module_for_legendre_psi()
{
  const int &n1 = r1.n;
  const int &n2 = r2.n;
  const int m = theta.m + 1;
  
  forward_legendre_transform();

  const int n_threads = 512;
  const int n_blocks = number_of_blocks(n_threads, n1*n2);
  
  double sum = 0.0;
  for(int l = 0; l < m; l++) {
    const cuDoubleComplex *legendre_psi_in_dev = (cuDoubleComplex *) legendre_psi_dev + l*n1*n2;
    
    Complex dot(0.0, 0.0);
    insist(cublasZdotc(cublas_handle, n1*n2, legendre_psi_in_dev, 1, legendre_psi_in_dev,
		       1, (cuDoubleComplex *) &dot) == CUBLAS_STATUS_SUCCESS);
    
    sum += 2.0/(2*l+1)*dot.real();
  }

  sum *= r1.dr*r2.dr;

  backward_legendre_transform();

  return sum;
}

void EvolutionCUDA::evolution_with_kinetic(const double dt)
{
  const int &n1 = r1.n;
  const int &n2 = r2.n;
  const int m = theta.m + 1;
  const int n = n1*n2*m;
  
  const int n_threads = 512;
  const int n_blocks = number_of_blocks(n_threads, n);
  
  _evolution_with_kinetic_<<<n_blocks, n_threads, (n1+n2)*sizeof(double)>>>
    (legendre_psi_dev, n1, n2, m, dt);
}

void EvolutionCUDA::evolution_with_rotational(const double dt)
{
  const int &n1 = r1.n;
  const int &n2 = r2.n;
  const int m = theta.m + 1;
  const int n = n1*n2*m;
  
  const int n_threads = 512;
  const int n_blocks = number_of_blocks(n_threads, n);
  
  _evolution_with_rotational_<<<n_blocks, n_threads, (n1+n2)*sizeof(double)>>>
    (legendre_psi_dev, n1, n2, m, dt);
}

void EvolutionCUDA::copy_psi_from_device_to_host()
{
  cout << " Copy wavepacket from device to host" << endl;
  
  insist(psi && psi_dev);

  const int &n1 = r1.n;
  const int &n2 = r2.n;
  const int &n_theta = theta.n;
  
  checkCudaErrors(cudaMemcpy(psi, psi_dev, n1*n2*n_theta*sizeof(Complex), cudaMemcpyDeviceToHost));
}

void EvolutionCUDA::dump_wavepacket()
{
  if(!apply_dump()) return;

  cout << " Dump wavepacket" << endl;

  const int &n1 = r1.n;
  const int &n2 = r2.n;
  const int &n_theta = theta.n;

  const int n_threads = 1024;
  const int n_blocks = number_of_blocks(n_threads, n1*n2*n_theta);

  _dump_wavepacket_<<<n_blocks, n_threads>>>(psi_dev, n1, n2, n_theta);
}

void EvolutionCUDA::setup_CRP_data_on_device()
{
  if(psi_on_surface_dev && d_psi_on_surface_dev && 
     fai_on_surface_dev && d_fai_on_surface_dev) return;
  
  const int &n1 = r1.n;
  const int &n_theta = theta.n;
  const int &n_energies = CRP.n_energies;
  
  cout << " Allocate CRP data memory on device" << endl;
  
  size_t size = 0;

  checkCudaErrors(cudaGetSymbolSize(&size, legendre_weight_dev));
  insist(size/sizeof(double) > n_theta);
  checkCudaErrors(cudaMemcpyToSymbol(legendre_weight_dev, theta.w, n_theta*sizeof(double)));

  checkCudaErrors(cudaGetSymbolSize(&size, energies_dev));
  insist(size/sizeof(double) > CRP.n_energies);
  checkCudaErrors(cudaMemcpyToSymbol(energies_dev, (const double *) CRP.energies, 
				     CRP.n_energies*sizeof(double)));
  
  if(!psi_on_surface_dev) {
    checkCudaErrors(cudaMalloc(&psi_on_surface_dev, n1*n_theta*sizeof(Complex)));
    insist(psi_on_surface_dev);
  }
  
  if(!d_psi_on_surface_dev) {
    checkCudaErrors(cudaMalloc(&d_psi_on_surface_dev, n1*n_theta*sizeof(Complex)));
    insist(d_psi_on_surface_dev);
  }
  
  if(!fai_on_surface_dev) {
    checkCudaErrors(cudaMalloc(&fai_on_surface_dev, n1*n_theta*n_energies*sizeof(Complex)));
    insist(fai_on_surface_dev);
    checkCudaErrors(cudaMemset(fai_on_surface_dev, 0, n1*n_theta*n_energies*sizeof(Complex)));
  }
  
  if(!d_fai_on_surface_dev) {
    checkCudaErrors(cudaMalloc(&d_fai_on_surface_dev, n1*n_theta*n_energies*sizeof(Complex)));
    insist(d_fai_on_surface_dev);
    checkCudaErrors(cudaMemset(d_fai_on_surface_dev, 0, n1*n_theta*n_energies*sizeof(Complex)));
  }
}

void EvolutionCUDA::calculate_psi_gradient_on_dividing_surface()
{
  cout << " Calculate Psi gradients on dividing surface" << endl;

  setup_CRP_data_on_device();

  const int &n1 = r1.n;
  const int &n2 = r2.n;
  const int &n_theta = theta.n;
  const double &dr2 = r2.dr;

  const int &n_dividing_surface = CRP.n_dividing_surface;
  
  // not sure why CRP.n_gradient_points always gives segmentation fault error
  // CRP.n_gradient_points;
  const int n_gradient_points = 11;
  insist(n_gradient_points == 11);

  const int n_threads = 256;
  const int n_blocks = number_of_blocks(n_threads, n1*n_theta);
  
  gradients_3d<Complex><<<n_blocks, n_threads>>>(n1, n2, n_theta, n_dividing_surface, dr2, psi_dev,
						 psi_on_surface_dev, d_psi_on_surface_dev,
						 n_gradient_points);
}

void EvolutionCUDA::psi_time_to_fai_energy_on_surface(const double t)
{
  cout << " Psi to Fai on dividing surface" << endl;

  const int &n1 = r1.n;
  const int &n_theta = theta.n;
  const int &n_energies = CRP.n_energies;
  const double &dt = time.time_step;

  const int n_threads = 512;
  const int n_blocks = number_of_blocks(n_threads, n1*n_theta*n_energies);

  _psi_time_to_fai_energy_on_surface_<<<n_blocks, n_threads, n_energies*sizeof(Complex)>>>
    (n1*n_theta, n_energies, t, dt, 
     psi_on_surface_dev, fai_on_surface_dev,
     d_psi_on_surface_dev, d_fai_on_surface_dev);
}

void EvolutionCUDA::_calculate_reaction_probabilities()
{
  const int &n1 = r1.n;
  const int &n_theta = theta.n;
  const double &dr1 = r1.dr;
  const double &mu2 = r2.mass;
  
  const int &n_energies = CRP.n_energies;
  RVec &crp = CRP.CRP;
  const RVec &eta_sq = CRP.eta_sq;
  
  const double dr1_mu2 = dr1/mu2;
  
  fai_on_dividing_surface_times_legendre_weight();  
  
  for(int iE = 0; iE < n_energies; iE++) {
    const Complex *fai_ = fai_on_surface_dev + iE*n1*n_theta;
    const Complex *dfai_ = d_fai_on_surface_dev + iE*n1*n_theta;
    Complex dot(0.0, 0.0);
    insist(cublasZdotc(cublas_handle, n1*n_theta,
		       (cuDoubleComplex *) dfai_, 1, 
		       (cuDoubleComplex *) fai_, 1, 
		       (cuDoubleComplex *) &dot) == CUBLAS_STATUS_SUCCESS);

    crp[iE] = dot.imag()/eta_sq[iE]*dr1_mu2;
  }
  
  fai_on_dividing_surface_divides_legendre_weight();
}

void EvolutionCUDA::calculate_reaction_probabilities(const int cal_CRP, const double time)
{
  cout << " Calculate reaction probabilities" << endl;

  setup_CRP_data_on_device();
  calculate_psi_gradient_on_dividing_surface();
  psi_time_to_fai_energy_on_surface(time);
  
  if(cal_CRP) 
    _calculate_reaction_probabilities();
}

void EvolutionCUDA::fai_on_dividing_surface_with_legendre_weight(const int op)
{
  insist(op == 1 || op == -1);
  
  const int &n1 = r1.n;
  const int &n_theta = theta.n;
  const int &n_energies = CRP.n_energies;
  
  const int n_threads = 512;
  const int n_blocks = number_of_blocks(n_threads, n1*n_theta*n_energies);
  
  _fai_on_dividing_surface_with_legendre_weight_<<<n_blocks, n_threads>>>
    (n1, n_theta, n_energies, fai_on_surface_dev, op);
}
