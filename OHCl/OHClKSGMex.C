
#include <iostream>
using namespace std;
#include <cassert>
#include <cstring>

#include "fort.h"
#include "mex.h"
#include "matutils.h"

extern "C" {
  void FORT(ohclksgpes)(const double &rOH, const double &rOCl, 
			const double &rHCl, double &V);
  // InitKSHParameters
  void FORT(initkshparameters)();
}

void mexFunction(int nlhs, mxArray *plhs[], 
                 int nrhs, const mxArray **prhs)
{
  if(nrhs != 3) 
    mexErrMsgTxt("OHClKSGMex requires 3 input arguments");
  
  if(nlhs > 1)
    mexErrMsgTxt("OHClKSGMex output arguments == 1");
  
  int i = 0;
  const double *rOH = mxGetPr(prhs[i]);
  insist(rOH);
  const int m = mxGetM(prhs[i]);
  const int n = mxGetN(prhs[i]);
  
  if(n > 1) 
    mexErrMsgTxt("OHClKSGMex input one dimensional vector only");
  
  i++;
  const double *rOCl = mxGetPr(prhs[i]);
  insist(rOCl);
  insist(m == mxGetM(prhs[i]) && n == mxGetN(prhs[i]));
  
  i++;
  const double *rHCl = mxGetPr(prhs[i]);
  insist(rHCl);
  insist(m == mxGetM(prhs[i]) && n == mxGetN(prhs[i]));

  // FORT(ohclksgpes) can be run in OpenMP mode only after 
  // declare KSGVar as threadprivate
  if(nlhs == 0 || nlhs == 1) {
    plhs[0] = mxCreateDoubleMatrix(m, n, mxREAL);
    double *pl0 = mxGetPr(plhs[0]);
    
    FORT(initkshparameters)();
    
#pragma omp parallel for			\
  default(shared) schedule(static, 1)
    for(int i = 0; i < m*n; i++) {
      double &V = pl0[i];
      FORT(ohclksgpes)(rOH[i], rOCl[i], rHCl[i], V);
    }
  } 

  cout << flush;

  return;
}
