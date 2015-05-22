
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

  // device functions
  void allocate_device_memories();
  void deallocate_device_memories();

  double module_for_psi() const;
  double module_for_psi_withou_streams() const;

  double potential_energy();
  double potential_energy2();
  
  void evolution_with_potential_dt();
};

#endif /* EVOLUTIONCUDA_H */

