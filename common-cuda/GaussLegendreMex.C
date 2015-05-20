
#include <iostream>
using namespace std;
#include <cassert>
#include <cstring>
#include <mex.h>

#include "fort.h"

extern "C" void FORT(gausslegendre)(const int &n, double *point, double *weight);

void mexFunction(int nlhs, mxArray *plhs[], 
                 int nrhs, const mxArray **prhs)
{
  if(nrhs != 1) 
    mexErrMsgTxt("GaussLegendreMex requires 1 input argument");
  
  const int n = *((int *) mxGetData(prhs[0]));
  
  if (n > 199)
    mexErrMsgTxt("GaussLegendreMex: highest order is 199");
  
  if (nlhs > 2)
    mexErrMsgTxt("GaussLegendreMex output arguments <= 2");
  
  plhs[0] = mxCreateDoubleMatrix(n, 1, mxREAL);
  plhs[1] = mxCreateDoubleMatrix(n, 1, mxREAL);
  
  FORT(gausslegendre)(n, mxGetPr(plhs[0]), mxGetPr(plhs[1]));
  
  cout << flush;
  
  return;
}
