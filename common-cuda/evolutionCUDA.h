
#ifndef EVOLUTIONCUDA_H
#define EVOLUTIONCUDA_H

#ifdef __NVCC__
#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include <cublas_v2.h>
#include <cufft.h>
#endif

#include "complex.h"
#include "MatlabStructures.h"
#include "matlabArray.h"

class EvolutionCUDA
{
public:
  EvolutionCUDA(const MatlabArray<double> &pot,
		MatlabArray<Complex> &psi,
		const RadialCoordinate &r1,
		const RadialCoordinate &r2,
		const AngleCoordinate &theta,
		EvolutionTime &time,
		const Options &options, 
		const DumpFunction &dump1, 
		const DumpFunction &dump2,
		CummulativeReactionProbabilities &CRP
		);

  ~EvolutionCUDA();

  void test();

private:
  double *pot;
  Complex *psi;
  
  const MatlabArray<double> &m_pot;
  MatlabArray<Complex> &m_psi;
  
  const RadialCoordinate &r1;
  const RadialCoordinate &r2;
  const AngleCoordinate &theta;
  EvolutionTime &time;
  const Options &options;
  const DumpFunction &dump1;
  const DumpFunction &dump2;
  CummulativeReactionProbabilities &CRP;

  // device data
  double *pot_dev;
  Complex *psi_dev;
  double *work_dev;
  double *w_dev;
  Complex *legendre_dev;
  Complex *weight_legendre_dev;
  Complex *legendre_psi_dev;

  Complex *psi_on_surface_dev;
  Complex *d_psi_on_surface_dev;
  Complex *fai_on_surface_dev;
  Complex *d_fai_on_surface_dev;
  
  int has_cublas_handle;

  int has_cufft_plan_for_psi;
  int has_cufft_plan_for_legendre_psi;
  
#ifdef __NVCC__
  cublasHandle_t cublas_handle;
  cufftHandle cufft_plan_for_psi;
  cufftHandle cufft_plan_for_legendre_psi;
#endif

  // device memory management 
  void allocate_device_memories();
  void deallocate_device_memories();
  
  // cublas handle
  void setup_cublas_handle();
  void destroy_cublas_handle();

  // Legendre transform
  void setup_legendre();
  void setup_weight_legendre();
  void setup_legendre_psi();
  void setup_legendre_transform()
  { 
    setup_legendre();
    setup_weight_legendre();
    setup_legendre_psi();
  }

  void forward_legendre_transform();
  void backward_legendre_transform();

  // FFT for psi
  void setup_cufft_plan_for_psi();
  void destroy_cufft_plan_for_psi();

  void forward_fft_for_psi();
  void backward_fft_for_psi(const int do_scale = 0);
  
  // FFT for Legendre psi
  void setup_cufft_plan_for_legendre_psi();
  void destroy_cufft_plan_for_legendre_psi();
  
  void forward_fft_for_legendre_psi();
  void backward_fft_for_legendre_psi(const int do_scale = 0);
  
  // energy
  double module_for_psi() const;
  double module_for_legendre_psi();
  double potential_energy();
  double kinetic_energy_for_psi(const int do_fft = 1);
  double kinetic_energy_for_legendre_psi(const int do_fft = 1);
  double rotational_energy(const int do_legendre_transform = 1);
  
  // time evolution
  void evolution_with_potential(const double dt);
  void evolution_with_rotational(const double dt);
  void evolution_with_kinetic(const double dt);
  void time_evolution();
  void copy_psi_from_device_to_host();

  // dump functions
  int apply_dump() const { return (dump1.dump && dump2.dump) ? 1 : 0; }
  void dump_wavepacket();

  // reaction probabilities
  void setup_CRP_data_on_device();
  void calculate_psi_gradient_on_dividing_surface();
  void psi_time_to_fai_energy_on_surface(const double t);
  void _calculate_reaction_probabilities();
  void calculate_reaction_probabilities(const int cal_CRP, const double time);
  
  void fai_on_dividing_surface_with_legendre_weight(const int op);

  void fai_on_dividing_surface_times_legendre_weight()
  { fai_on_dividing_surface_with_legendre_weight(1); }
  
  void fai_on_dividing_surface_divides_legendre_weight()
  { fai_on_dividing_surface_with_legendre_weight(-1); }
  
  void cuda_fft_test();  
};

void gpu_memory_usage();

#endif /* EVOLUTIONCUDA_H */

