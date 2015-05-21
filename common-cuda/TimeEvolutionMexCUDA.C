

/* $Id$ */

#include <iostream>
#include <cstring>
#include <cmath>
#include <mex.h>
#include "matutils.h"
#include "MatlabStructures.h"
#include "fort.h"
#include "timeEvolCUDA.h"

extern "C" int FORT(myisnan)(const double &x)
{
  return isnan(x);
}

void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[])
{
  const int np = std::cout.precision();
  std::cout.precision(15);
  
  std::cout << " 3D Time evolotion" << std::endl;

  insist(nrhs == 1);

  mxArray *mxPtr = 0;

  mxPtr = mxGetField(prhs[0], 0, "r1");
  insist(mxPtr);
  RadialCoordinate r1(mxPtr);

  mxPtr = mxGetField(prhs[0], 0, "r2");
  insist(mxPtr);
  RadialCoordinate r2(mxPtr);

  mxPtr = mxGetField(prhs[0], 0, "theta");
  insist(mxPtr);
  AngleCoordinate theta(mxPtr);
  
  mxPtr = mxGetField(prhs[0], 0, "pot");
  insist(mxPtr);
  MatlabArray<double> pot(mxPtr);
  
  mxPtr = mxGetField(prhs[0], 0, "psi");
  insist(mxPtr);
  MatlabArray<Complex> psi(mxPtr);
  
  mxPtr = mxGetField(prhs[0], 0, "time");
  insist(mxPtr);
  EvolutionTime time(mxPtr);
  
  mxPtr = mxGetField(prhs[0], 0, "options");
  insist(mxPtr);
  Options options(mxPtr);

  mxPtr = mxGetField(prhs[0], 0, "dump1");
  insist(mxPtr);
  DumpFunction dump1(mxPtr);

  mxPtr = mxGetField(prhs[0], 0, "dump2");
  insist(mxPtr);
  DumpFunction dump2(mxPtr);

  mxPtr = mxGetField(prhs[0], 0, "CRP");
  insist(mxPtr);
  CummulativeReactionProbabilities CRP(mxPtr);
  
  TimeEvolutionCUDA time_evolCUDA(pot, psi, r1, r2, theta, time, options, 
			      dump1, dump2, CRP);
  
  //time_evolCUDA.time_evolution();
  time_evolCUDA.test();

  //void cuda_test();
  //cuda_test();
  
  std::cout.flush();
  std::cout.precision(np);
}
