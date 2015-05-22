
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

  // Device data
  double *pot_dev;
  Complex *psi_dev;
  double *work_dev;
  double *w_dev;
  Complex *exp_ipot_dt_dev;

  int has_cublas_handle;
  int has_cufft_plan;

#ifdef __NVCC__
  cublasHandle_t cublas_handle;
  cufftHandle cufft_plan;
#endif
  
  void setup_cublas_handle();
  void destroy_cublas_handle();

  void setup_cufft_plan();
  void destroy_cufft_plan();

  // device functions
  void allocate_device_memories();
  void deallocate_device_memories();

  double module_for_psi() const;
  double potential_energy();
  
  void evolution_with_potential_dt();

  void cuda_fft_test();
};

#endif /* EVOLUTIONCUDA_H */

