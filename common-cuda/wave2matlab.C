
/* $Id$ */

#include <iostream>
using namespace std;
#include <cassert>
#include <cstring>
#include <cassert>
#include <mex.h>
#include "str.h"
#include "matutils.h"

int file_exist(const char *file_name)
{
  return access(file_name, F_OK) ? 0 : 1;
}

void wavepacket_to_matlab(const char *script, const int nrhs, mxArray *prhs[])
{
  if(!file_exist(script + Str(".m"))) return;

  cout << " Matlab script " << script << endl;

  insist(!mexCallMATLAB(0, NULL, nrhs, prhs, script));
}

void wavepacket_to_matlab(const char *script)
{
  if(!file_exist(script + Str(".m"))) return;

  cout << " Matlab script " << script << endl;

  insist(!mexCallMATLAB(0, NULL, 0, NULL, script));
}
