
/* $Id$ */

#include <iostream>
using namespace std;
#include <cassert>
#include <cstring>

#include <mex.h>
#include "fort.h"
#include "matutils.h"

#define ATTPACKED(x) __attribute__((packed)) FORT(x)

extern "C" struct {
  int length;
  char data_dir[512];
} ATTPACKED(hswdatadir);

extern "C" double FORT(pot)(const double &rhf1, const double &rH2, const double &rhf2, 
			    const int &read_data_only);


void FH2HSW(const double &r1, const double &r2, const double &r3,
	    double &v, const int read_data_only=0)
{
  v = FORT(pot)(r1, r2, r3, read_data_only);
}

void mexFunction(int nlhs, mxArray *plhs[], 
                 int nrhs, const mxArray **prhs)
{
  if(nrhs != 3) 
    mexErrMsgTxt("HSWMex requires 3 input arguments");
  
  if (nlhs > 3)
    mexErrMsgTxt("HSWMex output arguments <= 2");
  
  int i = 0;
  const double *r1 = mxGetPr(prhs[i]);
  insist(r1);
  const int m = mxGetM(prhs[i]);
  const int n = mxGetN(prhs[i]);

  if(n > 1) 
    mexErrMsgTxt("HSWMex input one dimensional vector only");
  
  static int read_data_only = 1;
  if(read_data_only == 1) {
    
    const char *data_dir = getenv("HSW_DATA_DIR");
    if(!data_dir) {
      mexErrMsgTxt("HSWMex: no enviorment variable 'HSW_DATA_DIR' defined");
    } else {
      FORT(hswdatadir).length = strlen(data_dir);
      memcpy(FORT(hswdatadir).data_dir, data_dir, sizeof(char)*strlen(data_dir));
    }
    
    double v = 0.0;
    FH2HSW(0.0, 0.0, 0.0, v, read_data_only);
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

#pragma omp parallel for if(m*n > 100)          \
  default(shared) schedule(static, 1)            
  for(int i = 0; i < m*n; i++) {
    double &v = pl0[i];
    FH2HSW(r1[i], r2[i], r3[i], v);
  }

  cout << flush;
}
