
/* $Id$ */

#include <iostream>
using namespace std;
#include <cassert>
#include <cstring>

#include <mex.h>
#include "fort.h"
#include "matutils.h"

extern "C" void FORT(fh2fxz)(const double *r, double &v, const int &read_data_only);


void FH2FXZ(const double &r1, const double &r2, const double &r3,
	    double &v, const int read_data_only=0)
{
  double r[3] = { r1, r2, r3 };
  FORT(fh2fxz)(r, v, read_data_only);
}

void mexFunction(int nlhs, mxArray *plhs[], 
                 int nrhs, const mxArray **prhs)
{
  if(nrhs != 3) 
    mexErrMsgTxt("FXZMex requires 3 input arguments");
  
  if (nlhs > 3)
    mexErrMsgTxt("FXZMex output arguments <= 2");
  
  int i = 0;
  const double *r1 = mxGetPr(prhs[i]);
  insist(r1);
  const int m = mxGetM(prhs[i]);
  const int n = mxGetN(prhs[i]);

  if(n > 1) 
    mexErrMsgTxt("FXZMex input one dimensional vector only");
  
  static int read_data_only = 1;
  if(read_data_only == 1) {
    double v = 0.0;
    FH2FXZ(0.0, 0.0, 0.0, v, read_data_only);
    read_data_only = 0;
  }

  i++;
  const double *r2 = mxGetPr(prhs[i]);
  insist(r2);
  insist(m == mxGetM(prhs[i]) && n == mxGetN(prhs[i]));
  
  i++;
  const double *r3 = mxGetPr(prhs[i]);
  insist(r3);
  insist(m == mxGetM(prhs[i]) && n == mxGetN(prhs[i]));
  
  plhs[0] = mxCreateDoubleMatrix(m, n, mxREAL);
  double *pl0 = mxGetPr(plhs[0]);
    
  for(int i = 0; i < m*n; i++) {
    double &v = pl0[i];
    FH2FXZ(r1[i], r2[i], r3[i], v);
  }
  
  cout << flush;
}
