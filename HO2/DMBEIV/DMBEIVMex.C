
#include <iostream>
using namespace std;
#include <cassert>
#include <cstring>

#include "fort.h"
#include "mex.h"
#include "matutils.h"

extern "C" void FORT(ho2sur)(const double *x, double &v);

void mexFunction(int nlhs, mxArray *plhs[], 
                 int nrhs, const mxArray **prhs)
{
  if(nrhs != 3) 
    mexErrMsgTxt("DMBEIVMex requires 3 input arguments");
  
  if(nlhs > 1)
    mexErrMsgTxt("DMBEIVMex output arguments == 1");
  
  int i = 0;
  const double *rOO = mxGetPr(prhs[i]);
  insist(rOO);
  const int m = mxGetM(prhs[i]);
  const int n = mxGetN(prhs[i]);
  
  if(n > 1) 
    mexErrMsgTxt("DMBEIVMex input one dimensional vector only");
  
  i++;
  const double *rOH1 = mxGetPr(prhs[i]);
  insist(rOH1);
  insist(m == mxGetM(prhs[i]) && n == mxGetN(prhs[i]));
  
  i++;
  const double *rOH2 = mxGetPr(prhs[i]);
  insist(rOH2);
  insist(m == mxGetM(prhs[i]) && n == mxGetN(prhs[i]));

  if(nlhs == 0 || nlhs == 1) {
    plhs[0] = mxCreateDoubleMatrix(m, n, mxREAL);
    double *pl0 = mxGetPr(plhs[0]);
    
#pragma omp parallel for			\
  default(shared) schedule(static, 1)
    for(int i = 0; i < m*n; i++) {
      double &v = pl0[i];
      const double R [] = { rOO[i], rOH1[i], rOH2[i] };
      FORT(ho2sur)(R, v);
    }
  } 
  
  cout << flush;

  return;
}
