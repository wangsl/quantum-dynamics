
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
  exp_ipot_dt_dev(0)
{ 
  pot = m_pot.data;
  insist(pot);

  psi = m_psi.data;
  insist(psi);

  allocate_device_memories();
} 

EvolutionCUDA::~EvolutionCUDA()
{
  cout << " EvolutionCUDA Destructor" << endl;

  pot = 0;
  psi = 0;

  // device memory
  deallocate_device_memories();
}template <class T, unsigned int blockSize, bool nIsPow2>
__global__ void reduce6(T *g_idata, T *g_odata, unsigned int n);

void EvolutionCUDA::test()
{
  cout << " EvolutionCUDA test" << endl;

  module_for_psi();

  evolution_with_potential_dt();

  cout << " Potential energy: " << potential_energy() << endl;

  module_for_psi_withou_streams();
}
