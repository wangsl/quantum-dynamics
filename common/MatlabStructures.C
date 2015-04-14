
/* $Id$ */

#include "MatlabStructures.h"
#include "matutils.h"
#include "fftwinterface.h"

void remove_matlab_script_extension(char *script, const char *extension)
{
  insist(script);
  insist(extension);
  const int len = strlen(script) - strlen(extension);
  if(!strcmp((const char *) script+len, extension)) 
    ((char *) script)[len] = '\0';
}

RadialCoordinate::RadialCoordinate(const mxArray *mx) :
  mx(mx),
  n(*(int *) mxGetData(mx, "n")),
  dr(*(double *) mxGetData(mx, "dr")),
  mass(*(double *) mxGetData(mx, "mass"))
{ 
  r = RVec(n, (double *) mxGetData(mx, "r"));
  
  psq2m.resize(n);
  FFTWInterface::get_momentum_for_fftw(psq2m, n*dr);
  for(int i = 0; i < n; i++) {
    psq2m[i] =  psq2m[i]*psq2m[i]/(2*mass);
  }

  one2mr2.resize(n);
  const double m2 = mass+mass;
  for(int i = 0; i < n; i++) { 
    one2mr2[i] = 1.0/(m2*r[i]*r[i]);
  }
}

AngleCoordinate::AngleCoordinate(const mxArray *mx) :
  mx(mx),
  n(*(int *) mxGetData(mx, "n")),
  m(*(int *) mxGetData(mx, "m"))
{
  x = RVec(n, (double *) mxGetData(mx, "x"));
  w = RVec(n, (double *) mxGetData(mx, "w"));
  
  double *p = (double *) mxGetData(mx, "legendre");
  insist(p);

  legendre = RMat(m+1, n, p);
}
  
EvolutionTime::EvolutionTime(const mxArray *mx) :
  mx(mx),
  total_steps(*(int *) mxGetData(mx, "total_steps")),
  steps(*(int *) mxGetData(mx, "steps")),
  time_step(*(double *) mxGetData(mx, "time_step"))
{ }

Options::Options(const mxArray *mx) :
  mx(mx),
  wave_to_matlab(0),
  test_name(0)
{
  wave_to_matlab = mxGetString(mx, "wave_to_matlab");
  if(wave_to_matlab)
    remove_matlab_script_extension(wave_to_matlab, ".m");
  
  test_name = mxGetString(mx, "test_name");
}

Options::~Options()
{
  if(wave_to_matlab) { delete [] wave_to_matlab; wave_to_matlab = 0; }
  if(test_name) { delete [] test_name; test_name = 0; }
}

DumpFunction::DumpFunction(const mxArray *mx) :
  mx(mx), dump(0)
{
  dump = (double *) mxGetData(mx, "dump");
}

DumpFunction::~DumpFunction()
{
  if(dump) dump = 0;
}

CummulativeReactionProbabilities::CummulativeReactionProbabilities(const mxArray *mx) :
  mx(mx),
  n_dividing_surface(*(int *) mxGetData(mx, "n_dividing_surface")),
  n_gradient_points(*(int *) mxGetData(mx, "n_gradient_points")),
  n_energies(*(int *) mxGetData(mx, "n_energies")),
  calculate_CRP(*(int *) mxGetData(mx, "calculate_CRP"))
{
  energies = RVec(n_energies, (double *) mxGetData(mx, "energies"));
  eta_sq = RVec(n_energies, (double *) mxGetData(mx, "eta_sq"));
  CRP = RVec(n_energies, (double *) mxGetData(mx, "CRP"));
}


