
#include "evolutionCUDA.h"

EvolutionCUDA::EvolutionCUDA(const MatlabArray<double> &m_pot_,
			     MatlabArray<Complex> &m_psi_,
			     const RadialCoordinate &r1_, 
			     const RadialCoordinate &r2_,
			     const AngleCoordinate &theta_,
			     EvolutionTime &time_,
			     const Options &options_,
			     const DumpFunction &dump1_,
			     const DumpFunction &dump2_,
			     CummulativeReactionProbabilities &CRP_
			     ) :
  m_pot(m_pot_), m_psi(m_psi_), 
  r1(r1_), r2(r2_), theta(theta_), time(time_), options(options_),
  dump1(dump1_), dump2(dump2_),
  CRP(CRP_),
  // device memory
  pot_dev(0), psi_dev(0), work_dev(0), w_dev(0),
  legendre_dev(0), weight_legendre_dev(0), legendre_psi_dev(0),
  psi_on_surface_dev(0), d_psi_on_surface_dev(0),
  fai_on_surface_dev(0), d_fai_on_surface_dev(0),
  has_cublas_handle(0), has_cufft_plan_for_psi(0), 
  has_cufft_plan_for_legendre_psi(0),
  has_copied_gradient_coeffients_to_device(0)
{ 
  pot = m_pot.data;
  insist(pot);
  
  psi = m_psi.data;
  insist(psi);

  allocate_device_memories();
} 

EvolutionCUDA::~EvolutionCUDA()
{
  pot = 0;
  psi = 0;

  deallocate_device_memories();
}

void EvolutionCUDA::test()
{
  cuda_fft_test();
  gpu_memory_usage();
}
