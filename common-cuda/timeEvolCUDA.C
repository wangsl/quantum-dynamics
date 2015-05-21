
/* $Id$ */

// http://www.walkingrandomly.com/?p=1795

#include "timer.h"
#include "timeEvolCUDA.h"
#include "mat.h"

#define _PrintFunction_ { cout << " " << __func__ << endl; }

//#define _PrintFunction_ { }

extern "C" {
  void FORT(forwardlegendretransform)(const Complex *CPsi, Complex *LPsi, 
				      const int &NR, const int &NTheta, const int &NLeg, 
				      const double *WLegP);
  
  void FORT(backwardlegendretransform)(Complex *CPsi, const Complex *LPsi, 
				       const int &NR, const int &NTheta, const int &NLeg, 
				       const double *LegP);
  void FORT(gradient3d)(const int &N1, const int &N2, const int &N3, 
			const int &N2DivSurf, const double &Dx, const Complex *F, 
			Complex *V, Complex *G, const int &NPoints);
  
  // PsiTimeToPsiEnergyOnSurface
  void FORT(psitimetopsienergyonsurface)(const int &N1, const int &NTheta, const int &NEnergies,
					 const Complex *ExpIETDt, const Complex *PsiT, 
					 const Complex *DPsiT, Complex *faiE, Complex *DFaiE);

  // CalculateCRP
  void FORT(calculatecrp)(const int &N1, const int &NTheta, const int &NE, 
			  const double &Dr1, const double &mu2, 
			  const double *W, const double *EtaSq,
			  const Complex *Fai, const Complex *DFai, double *CRP);
}

TimeEvolutionCUDA::TimeEvolutionCUDA(const MatlabArray<double> &m_pot_,
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
  m_pot(m_pot_),  m_psi(m_psi_), 
  r1(r1_), r2(r2_), theta(theta_), time(time_), options(options_),
  dump1(dump1_), dump2(dump2_),
  CRP(CRP_),
  _legendre_psi(0), 
  exp_ipot_dt(0), exp_irot_dt_2(0), exp_ikin_dt(0),
  weight_legendre(0),
  dump(0),
  exp_ienergy_dt(0), exp_ienergy_t(0),
  psi_surface(0), d_psi_surface(0), fai_surface(0), d_fai_surface(0),
  // device memory
  pot_dev(0), psi_dev(0), work_dev(0), w_dev(0)
{ 
  pot = m_pot.data;
  insist(pot);

  psi = m_psi.data;
  insist(psi);

  allocate_device_memories();
} 

TimeEvolutionCUDA::~TimeEvolutionCUDA()
{
  cout << " TimeEvolutionCUDA Destructor" << endl;

  pot = 0;
  psi = 0;
  
  destroy_fftw_interface_for_psi();
  destroy_fftw_interface_for_legendre_psi();
  
  if(_legendre_psi) { delete [] _legendre_psi; _legendre_psi = 0; }
  if(exp_ipot_dt) { delete [] exp_ipot_dt; exp_ipot_dt = 0; }
  if(exp_irot_dt_2) { delete [] exp_irot_dt_2; exp_irot_dt_2 = 0; }
  if(exp_ikin_dt) { delete [] exp_ikin_dt; exp_ikin_dt = 0; }
  if(dump) { delete [] dump; dump = 0; }
  if(weight_legendre) { delete [] weight_legendre; weight_legendre = 0; }
  if(exp_ienergy_dt) { delete [] exp_ienergy_dt; exp_ienergy_dt = 0; }
  if(exp_ienergy_t) { delete [] exp_ienergy_t; exp_ienergy_t = 0; }
  if(psi_surface) { delete [] psi_surface; psi_surface = 0; }
  if(d_psi_surface) { delete [] d_psi_surface; d_psi_surface = 0; }
  if(fai_surface) { delete [] fai_surface; fai_surface = 0; }
  if(d_fai_surface) { delete [] d_fai_surface; d_fai_surface = 0; }

  // device memory
  deallocate_device_memories();
}

Complex * &TimeEvolutionCUDA::legendre_psi()
{
  if(!_legendre_psi) {
    const int &n1 = r1.n;
    const int &n2 = r2.n;
    const int m = theta.m + 1;
    _legendre_psi = new Complex [n1*n2*m];
    insist(_legendre_psi);
  }
  return _legendre_psi;
}

void TimeEvolutionCUDA::destroy_fftw_interface_for_psi()
{
  Vec<FFTWInterface *> &fftw = fftw_for_psi;
  for(int i = 0; i < fftw.size(); i++) {
    if(fftw[i]) {
      delete fftw[i];
      fftw[i] = 0;
    }
  }
  fftw.resize(0);
}

void TimeEvolutionCUDA::destroy_fftw_interface_for_legendre_psi()
{
  Vec<FFTWInterface *> &fftw = fftw_for_legendre_psi;
  for(int i = 0; i < fftw.size(); i++) {
    if(fftw[i]) {
      delete fftw[i];
      fftw[i] = 0;
    }
  }
  fftw.resize(0);
}

void TimeEvolutionCUDA::setup_fftw_interface_for_psi()
{
  Vec<FFTWInterface *> &fftw = fftw_for_psi;

  const int &n_theta = theta.n;
  if(fftw.size() == n_theta) return;
  
  fftw_init_threads();
  
  fftw.resize(n_theta);
  fftw.zero();
  
  const int &n1 = r1.n;
  const int &n2 = r2.n;
  double *p = (double *) psi;
  for(int k = 0; k < n_theta; k++) {
    fftw[k] = new FFTWInterface(p, n1, n2, FFTW_MEASURE, 1);
    insist(fftw[k]);
    p += 2*n1*n2;
  }
}

void TimeEvolutionCUDA::setup_fftw_interface_for_legendre_psi()
{
  Vec<FFTWInterface *> &fftw = fftw_for_legendre_psi;
  
  const int m = theta.m + 1;
  if(fftw.size() == m) return;
  
  fftw_init_threads();
  
  fftw.resize(m);
  fftw.zero();
  
  const int &n1 = r1.n;
  const int &n2 = r2.n;
  double *p = (double *) legendre_psi();
  for(int k = 0; k < m; k++) {
    fftw[k] = new FFTWInterface(p+k*2*n1*n2, n1, n2, FFTW_MEASURE, 1);
    insist(fftw[k]);
  }
}

void TimeEvolutionCUDA::forward_fft_for_psi()
{
  setup_fftw_interface_for_psi();
  for(int i = 0; i < fftw_for_psi.size(); i++) 
    fftw_for_psi[i]->forward_transform();
}

void TimeEvolutionCUDA::backward_fft_for_psi()
{
  setup_fftw_interface_for_psi();
  for(int i = 0; i < fftw_for_psi.size(); i++) 
    fftw_for_psi[i]->backward_transform();
}

void TimeEvolutionCUDA::forward_fft_for_legendre_psi()
{
  setup_fftw_interface_for_legendre_psi();
  for(int i = 0; i < fftw_for_legendre_psi.size(); i++) 
    fftw_for_legendre_psi[i]->forward_transform();
}

void TimeEvolutionCUDA::backward_fft_for_legendre_psi()
{
  setup_fftw_interface_for_legendre_psi();
  for(int i = 0; i < fftw_for_legendre_psi.size(); i++) 
    fftw_for_legendre_psi[i]->backward_transform();
}

void TimeEvolutionCUDA::forward_legendre_transform()
{
  setup_weight_legendre();
  
  const int &n1 = r1.n;
  const int &n2 = r2.n;
  const int &n_theta = theta.n;
  const int m = theta.m + 1;
  
  FORT(forwardlegendretransform)(psi, legendre_psi(), n1*n2, n_theta, m, 
				 weight_legendre);
}

void TimeEvolutionCUDA::backward_legendre_transform()
{
  const int &n1 = r1.n;
  const int &n2 = r2.n;
  const int &n_theta = theta.n;
  const int m = theta.m + 1;
  
  FORT(backwardlegendretransform)(psi, legendre_psi(), n1*n2, n_theta, m, 
				  theta.legendre);
}

double TimeEvolutionCUDA::module_for_psi() const
{
  const int &n1 = r1.n;
  const int &n2 = r2.n;
  const int &n_theta = theta.n;
  const RVec &w = theta.w;
  
  Mat<Complex> p(n1*n2, n_theta, psi);
  
  double s = 0.0;
#pragma omp parallel for if(p.columns() > 100)	      \
  default(shared) schedule(static, 1)                 \
  reduction(+:s)
  for(int k = 0; k < p.columns(); k++) {
    double sk = 0.0;
    for(int i = 0; i < p.rows(); i++) {
      sk += abs2(p(i,k));
    }
    s += w[k]*sk;
  }
  
  s *= r1.dr*r2.dr;
  return s;
}

void TimeEvolutionCUDA::calculate_energy()
{
  double e_kin = kinetic_energy_for_psi();
  double e_rot = rotational_energy();
  double e_pot = potential_energy();
  
  double e_total = e_kin + e_rot + e_pot;

  cout << " Total energy: " << e_total << endl;
}

double TimeEvolutionCUDA::potential_energy()
{
  const int &n1 = r1.n;
  const int &n2 = r2.n;
  const double &dr1 = r1.dr;
  const double &dr2 = r2.dr;
  const int &n_theta = theta.n;
  const RVec &w = theta.w;
  
  const Mat<Complex> Psi(n1*n2, n_theta, psi);
  const RMat Pot(n1*n2, n_theta, pot);
  
  double s = 0.0;
#pragma omp parallel for if(Psi.columns() > 100)      \
  default(shared) schedule(static, 1)                 \
  reduction(+:s)
  for(int k = 0; k < Psi.columns(); k++) {
    double sk = 0.0;
    for(int i = 0; i < Psi.rows(); i++) {
      sk += abs2(Psi(i,k))*Pot(i,k);
    }
    s += w[k]*sk;
  }
  
  double e_pot = s*r1.dr*r2.dr;

  return e_pot;
}

double TimeEvolutionCUDA::kinetic_energy_for_psi()
{ 
  const int &n1 = r1.n;
  const int &n2 = r2.n;
  const int n_theta = theta.n;
  const double &dr1 = r1.dr;
  const double &dr2 = r2.dr;
  const RVec &w = theta.w;
  
  const RVec &kin1 = r1.psq2m;
  const RVec &kin2 = r2.psq2m;
  const double n1n2 = n1*n2;

  forward_fft_for_psi();
  
  double e = 0.0;
#pragma omp parallel for			\
  default(shared) schedule(static, 1)		\
  reduction(+:e)
  for(int k = 0; k < n_theta; k++) {
    Mat<Complex> Psi(n1, n2, psi+k*n1*n2);
    double ek = 0.0;
    for(int j = 0; j < n2; j++) {
      for(int i = 0; i < n1; i++) {
	ek += abs2(Psi(i,j))*(kin1[i] + kin2[j]);
	Psi(i,j) /= n1n2;
      }
    }
    e += w[k]*ek;
  }
  
  backward_fft_for_psi();
  
  double e_kin = e*dr1*dr2/(n1*n2);

  return e_kin;
}

double TimeEvolutionCUDA::rotational_energy(const int do_legendre_transform)
{ 
  if(do_legendre_transform)
    forward_legendre_transform();
  
  const double &dr1 = r1.dr;
  const double &dr2 = r2.dr;
  const int &n1 = r1.n;
  const int &n2 = r2.n;
  const int m = theta.m + 1;
  
  const RVec &I1 = r1.one2mr2;
  const RVec &I2 = r2.one2mr2;
  
  Complex *p = legendre_psi();
  
  double s = 0.0;
#pragma omp parallel for			\
  default(shared) schedule(static, 1)		\
  reduction(+:s)
  for(int l = 0; l < m; l++) {
    const Mat<Complex> LPsi(n1, n2, p+l*n1*n2);
    double sl = 0.0;
    for(int j = 0; j < n2; j++) {
      for(int i = 0; i < n1; i++) {
	sl += abs2(LPsi(i,j))*(I1[i]+I2[j]);
      }
    }
    s += sl*l*(l+1)/(l+0.5);
  }
  
  if(do_legendre_transform)
    backward_legendre_transform();
  
  double e_rot = s*dr1*dr2;
  
  return e_rot;
} 

double TimeEvolutionCUDA::module_for_legendre_psi()
{ 
  const int &n1 = r1.n;
  const int &n2 = r2.n;
  const int m = theta.m + 1;
  
  const double &dr1 = r1.dr;
  const double &dr2 = r2.dr;
  
  Mat<Complex> LPsi(n1*n2, m, legendre_psi());
  
  double s = 0.0;
#pragma omp parallel for			      \
  default(shared) schedule(static, 1)                 \
  reduction(+:s)
  for(int l = 0; l < LPsi.columns(); l++) {
    double sl = 0.0;
    for(int i = 0; i < LPsi.rows(); i++) {
      sl += abs2(LPsi(i,l));
    }
    s += sl/(l+0.5);
  }
  
  s *= dr1*dr2;
  
  return s;
}

double TimeEvolutionCUDA::kinetic_energy_for_legendre_psi(const int do_fft)
{ 
  const int &n1 = r1.n;
  const int &n2 = r2.n;
  const double &dr1 = r1.dr;
  const double &dr2 = r2.dr;
  
  const RVec &kin1 = r1.psq2m;
  const RVec &kin2 = r2.psq2m;
  
  const int m = theta.m + 1;
  
  const double n1n2 = n1*n2;
  
  if(do_fft)
    forward_fft_for_legendre_psi();
  
  Complex * &p = legendre_psi();
  
  double s = 0.0;
#pragma omp parallel for			\
  default(shared) schedule(static, 1)		\
  reduction(+:s)
  for(int l = 0; l < m; l++) {
    Mat<Complex> LPsi(n1, n2, p+l*n1*n2);
    double sl = 0.0;
    for(int j = 0; j < n2; j++) {
      for(int i = 0; i < n1; i++) {
	sl += abs2(LPsi(i,j))*(kin1[i] + kin2[j]);
	if(do_fft)
	  LPsi(i,j) /= n1n2;
      }
    }
    s += sl/(l+0.5);
  }

  double e_kin = s*dr1*dr2;
  
  if(do_fft) {
    backward_fft_for_legendre_psi();
    e_kin /= n1*n2;
  } else {
    e_kin *= n1*n2;
  }
  
  return e_kin;
}

void TimeEvolutionCUDA::setup_exp_ipot_dt()
{
  if(exp_ipot_dt) return;

  const int &n1 = r1.n;
  const int &n2 = r2.n;
  const int &n_theta = theta.n;
  
  exp_ipot_dt = new Complex [n1*n2*n_theta];
  insist(exp_ipot_dt);
  
  const double &dt = time.time_step;
  const Complex Idt(0.0, -dt);
  
#pragma omp parallel for			\
  default(shared) schedule(static, 1)		
  for(int k = 0; k < n_theta; k++) {
    RMat v(n1, n2, pot+k*n1*n2);
    Mat<Complex> p(n1, n2, exp_ipot_dt+k*n1*n2);
    for(int j = 0; j < n2; j++) {
      for(int i = 0; i < n1; i++) {
	p(i,j) = exp(Idt*v(i,j));
      }
    }
  }
}

void TimeEvolutionCUDA::setup_exp_irot_dt_2()
{ 
  if(exp_irot_dt_2) return;

  const int &n1 = r1.n;
  const int &n2 = r2.n;
  const int m = theta.m + 1;
  
  exp_irot_dt_2 = new Complex [n1*n2*m];
  insist(exp_irot_dt_2);

  const double &dt = time.time_step;
  const Complex Idt2(0.0, -dt/2);

#pragma omp parallel for			\
  default(shared) schedule(static, 1)	
  for(int l = 0; l < m; l++) {
    Mat<Complex> p(n1, n2, exp_irot_dt_2+l*n1*n2);
    for(int j = 0; j < n2; j++) {
      const double &I2 = r2.one2mr2[j];
      for(int i = 0; i < n1; i++) {
	const double &I1 = r1.one2mr2[i];
	p(i,j) = exp(l*(l+1)*(I1+I2)*Idt2);
      }
    }
  }
}

void TimeEvolutionCUDA::setup_exp_ikin_dt()
{ 
  if(exp_ikin_dt) return;

  const int &n1 = r1.n;
  const int &n2 = r2.n;
  
  exp_ikin_dt = new Complex [n1*n2];
  insist(exp_ikin_dt);
  
  const double &dt = time.time_step;
  const Complex Idt(0.0, -dt);
  
  Mat<Complex> p(n1, n2, exp_ikin_dt);
  
#pragma omp parallel for			\
  default(shared) schedule(static, 1)	
  for(int j = 0; j < n2; j++) {
    const double &T2 = r2.psq2m[j];
    for(int i = 0; i < n1; i++) {
      const double &T1 = r1.psq2m[i];
      p(i,j) = exp((T1+T2)*Idt)/n1/n2;
    }
  }
}

void TimeEvolutionCUDA::pre_evolution_with_potential_dt_2()
{ 
  setup_exp_ipot_dt();
  
  const int &n1 = r1.n;
  const int &n2 = r2.n;
  const int &n_theta = theta.n;
  
#pragma omp parallel for			\
  default(shared) schedule(static, 1)	    
  for(int k = 0; k < n_theta; k++) {
    const Mat<Complex> v(n1, n2, exp_ipot_dt+k*n1*n2);
    Mat<Complex> Psi(n1, n2, psi+k*n1*n2);
    for(int j = 0; j < n2; j++) {
      for(int i = 0; i < n1; i++) {
	Psi(i,j) /= sqrt(v(i,j));
      }
    }
  }
}

void TimeEvolutionCUDA::evolution_with_potential_dt()
{ 
  setup_exp_ipot_dt();
  
  const int &n1 = r1.n;
  const int &n2 = r2.n;
  const int &n_theta = theta.n;
  
#pragma omp parallel for			\
  default(shared) schedule(static, 1)	    
  for(int k = 0; k < n_theta; k++) {
    const Mat<Complex> v(n1, n2, exp_ipot_dt+k*n1*n2);
    Mat<Complex> Psi(n1, n2, psi+k*n1*n2);
    for(int j = 0; j < n2; j++) {
      for(int i = 0; i < n1; i++) {
	Psi(i,j) *= v(i,j);
      }
    }
  }
}

void TimeEvolutionCUDA::evolution_with_rotational_dt_2()
{
  setup_exp_irot_dt_2();
  
  const int &n1 = r1.n;
  const int &n2 = r2.n;
  const int m = theta.m + 1;
  
  Complex * &lpsi = legendre_psi();
  
#pragma omp parallel for			\
  default(shared) schedule(static, 1)	
  for(int l = 0; l < m; l++) {
    Mat<Complex> rot(n1, n2, exp_irot_dt_2+l*n1*n2);
    Mat<Complex> LPsi(n1, n2, lpsi+l*n1*n2);
    for(int j = 0; j < n2; j++) {
      for(int i = 0; i < n1; i++) {
	LPsi(i,j) *= rot(i,j);
      }
    }
  }
}

void TimeEvolutionCUDA::evolution_with_kinetic_dt()
{ 
  setup_exp_ikin_dt();

  const int &n1 = r1.n;
  const int &n2 = r2.n;
  const int m = theta.m + 1;
  
  Mat<Complex> kin(n1, n2, exp_ikin_dt);

  Complex * &lpsi = legendre_psi();

#pragma omp parallel for			\
  default(shared) schedule(static, 1)	
  for(int l = 0; l < m; l++) {	       
    Mat<Complex> LPsi(n1, n2, lpsi+l*n1*n2);
    for(int j = 0; j < n2; j++) {
      for(int i = 0; i < n1; i++) {
	LPsi(i,j) *= kin(i,j);
      }
    }
  }
}

void TimeEvolutionCUDA::test()
{
  cout << " TimeEvolutionCUDA::test()" << endl;
  //cuda_psi_normal_test();
  //cuda_fft_test();
  cuda_fft_test_with_many_plan();
}

void TimeEvolutionCUDA::evolution_dt()
{
  evolution_with_potential_dt();

  forward_legendre_transform();

  evolution_with_rotational_dt_2();

  forward_fft_for_legendre_psi();

  evolution_with_kinetic_dt();

  const double e_kin = kinetic_energy_for_legendre_psi(0);

  backward_fft_for_legendre_psi();

  evolution_with_rotational_dt_2();

  const double e_rot = rotational_energy(0);

  backward_legendre_transform();

  const double e_pot = potential_energy();

  cout << " e_kin: " << e_kin << "\n"
       << " e_rot: " << e_rot << "\n"
       << " e_pot: " << e_pot << "\n"
       << " e_tot: " << e_kin + e_rot + e_pot << endl;
}

void TimeEvolutionCUDA::time_evolution()
{
  const int &total_steps = time.total_steps;
  int &steps = time.steps;

  WallTimer wall_timer;
  CPUTimer cpu_timer;
  
  for(int i_step = 0; i_step < total_steps; i_step++) {

    wall_timer.reset();
    cpu_timer.reset();
    
    cout << "\n Step: " << i_step << endl;
    
    if(i_step == 0 && steps == 0)
      pre_evolution_with_potential_dt_2();
    
    evolution_dt();
    
    dump_psi();
    
    cout << " module: " << module_for_psi() << endl;

    if(CRP.calculate_CRP)
      calculate_reaction_probabilities();
      
    if(options.wave_to_matlab)
      wavepacket_to_matlab(options.wave_to_matlab);
    
    cout << " Wall time: " << wall_timer.time() << " s, CPU time: "
	 << cpu_timer.time() << " s" << endl;
    
    steps++;
    cout.flush();
  }
}

void TimeEvolutionCUDA::setup_dump()
{
  if(dump) return;
  if(!apply_dump()) return;
  
  const int &n1 = r1.n;
  const int &n2 = r2.n;
  
  if(!dump) { 
    dump = new double [n1*n2];
    insist(dump);
  }
  
  RMat d(n1, n2, dump);
  const double *d1 = dump1.dump;
  const double *d2 = dump2.dump;
  
#pragma omp parallel for			\
  default(shared) schedule(static, 1)
  for(int j = 0; j < n2; j++) {
    for(int i = 0; i < n1; i++) {
      d(i,j) = d1[i]*d2[j];
    } 
  }
}

void TimeEvolutionCUDA::dump_psi()
{
  if(!apply_dump()) return;
  
  setup_dump();
  
  const int &n1 = r1.n;
  const int &n2 = r2.n;
  const int &n_theta = theta.n;

  const RMat d(n1, n2, dump);

#pragma omp parallel for			\
  default(shared) schedule(static, 1)	    
  for(int k = 0; k < n_theta; k++) {
    Mat<Complex> Psi(n1, n2, psi+k*n1*n2);
    for(int j = 0; j < n2; j++) {
      for(int i = 0; i < n1; i++) {
	Psi(i,j) *= d(i,j);
      }
    }
  }
}

void  TimeEvolutionCUDA::setup_weight_legendre()
{
  if(weight_legendre) return;

  const int &n_theta = theta.n;
  const int m = theta.m + 1;

  weight_legendre = new double[n_theta*m];
  insist(weight_legendre);
  
  const double *w = theta.w;
  const RMat &P = theta.legendre;
  
  RMat wp(n_theta, m, weight_legendre);
  
#pragma omp parallel for			\
  default(shared) schedule(static, 1)
  for(int l = 0; l < m; l++) {
    const double f = l+0.5;
    for(int k = 0; k < n_theta; k++) {
      wp(k,l) = f*w[k]*P(l,k);
    }
  }
}

void TimeEvolutionCUDA::setup_CRP_data()
{
  if(exp_ienergy_dt && exp_ienergy_t) return;
  
  const int &n_dividing_surface = CRP.n_dividing_surface;
  const int &n_energies = CRP.n_energies;
  const int &n_gradient_points = CRP.n_gradient_points;

  const int &n1 = r1.n;
  const int &nTheta = theta.n;

  psi_surface = new Complex [n1*nTheta];
  insist(psi_surface);
  
  d_psi_surface = new Complex [n1*nTheta];
  insist(d_psi_surface);
  
  fai_surface = new Complex [n1*nTheta*n_energies];
  insist(fai_surface);
  memset(fai_surface, 0, n1*nTheta*n_energies*sizeof(Complex));
  
  d_fai_surface = new Complex [n1*nTheta*n_energies];
  insist(d_fai_surface);
  memset(d_fai_surface, 0, n1*nTheta*n_energies*sizeof(Complex));
  
  const RVec &e = CRP.energies;
  const double &dt = time.time_step;
  
  exp_ienergy_dt = new Complex [n_energies];
  insist(exp_ienergy_dt);
  
  exp_ienergy_t = new Complex [n_energies];
  insist(exp_ienergy_t);
  
  const Complex I(0.0, 1.0);
  const Complex dt_complex(dt, 0.0);
  
#pragma omp parallel for			\
  default(shared) schedule(static, 1)
  for(int i = 0; i < n_energies; i++) {
    exp_ienergy_dt[i] = exp(I*e[i]*dt);
    exp_ienergy_t[i] = dt_complex;
  }
}

void TimeEvolutionCUDA::update_exp_ienergy_t()
{
  setup_CRP_data();

  const int &n = CRP.n_energies;
  
#pragma omp parallel for			\
  default(shared) schedule(static, 1)
  for(int i = 0; i < n; i++)
    exp_ienergy_t[i] *= exp_ienergy_dt[i];
}

void TimeEvolutionCUDA::calculate_psi_gradient_on_dividing_surface()
{
  const int &n1 = r1.n;
  const int &n2 = r2.n;
  const int &nTheta = theta.n;
  const double &dr2 = r2.dr;
  
  const int &n_dividing_surface = CRP.n_dividing_surface;
  const int &n_gradient_points = CRP.n_gradient_points;
  
  insist(psi_surface);
  insist(d_psi_surface);
  
  FORT(gradient3d)(n1, n2, nTheta, n_dividing_surface, dr2,
		   psi, psi_surface, d_psi_surface, n_gradient_points);
  
}

void TimeEvolutionCUDA::psi_time_to_fai_energy_on_surface()
{
  const int &n1 = r1.n;
  const int &nTheta = theta.n;
  const int &n_energies = CRP.n_energies;
  
  insist(psi_surface);
  insist(d_psi_surface);
  insist(fai_surface);
  insist(d_fai_surface);
  
  FORT(psitimetopsienergyonsurface)(n1, nTheta, n_energies, exp_ienergy_t, 
				    psi_surface, d_psi_surface, 
				    fai_surface, d_fai_surface);
}

void TimeEvolutionCUDA::_calculate_reaction_probabilities()
{
  const int &n1 = r1.n;
  const int &nTheta = theta.n;
  const RVec &w = theta.w;
  const double &dr1 = r1.dr;
  const double &mu2 = r2.mass;

  const int &n_energies = CRP.n_energies;

  RVec &crp = CRP.CRP;
  const RVec &eta_sq = CRP.eta_sq;

  FORT(calculatecrp)(n1, nTheta, n_energies, dr1, mu2, w, eta_sq, 
		     fai_surface, d_fai_surface, crp);
}

void TimeEvolutionCUDA::calculate_reaction_probabilities()
{
  cout << " Calculate reaction probabilities" << endl;
  setup_CRP_data();
  calculate_psi_gradient_on_dividing_surface();
  update_exp_ienergy_t();
  psi_time_to_fai_energy_on_surface();
  _calculate_reaction_probabilities();
}
