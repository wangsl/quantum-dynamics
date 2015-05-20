
/* $Id$ */

#ifndef MATUTILS_H
#define MATUTILS_H

#define MCrash(x) MatlabCrashLoc(x, __FILE__, __LINE__)
#define MatCrash(x) MatlabCrashLoc(x, __FILE__, __LINE__)
#define MatlabCrash(x) MatlabCrashLoc(x, __FILE__, __LINE__)

#ifdef __cplusplus

#include <mex.h>
#include <cstring>

#define insist(x) if (!(x)) MatlabCrashLoc("insist failed: " #x, __FILE__, __LINE__)

void MatlabCrashLoc(const char *message, const char *file_name, const int line);

void wavepacket_to_matlab(const char *script, const int nrhs, mxArray *prhs[]);
void wavepacket_to_matlab(const char *script);

inline void *mxGetData(const mxArray *mx, const char *field)
{
  insist(mx);
  mxArray *mxPtr = mxGetField(mx, 0, field);
  if(!mxPtr) return 0;
  void *ptr = mxGetData(mxPtr);
  if(!mxPtr) return 0;
  return ptr;
}

inline char *mxGetString(const mxArray *mx, const char *field)
{
  insist(mx);
  char *tmp = mxArrayToString(mxGetField(mx, 0, field));
  if(!tmp) return 0;
  
  char *string = new char [strlen(tmp) + 1];
  insist(string);
  memcpy(string, tmp, strlen(tmp)*sizeof(char));
  string[strlen(tmp)] = '\0';
  if(tmp) { mxFree(tmp); tmp = 0; }
  return string;
}

#else /* For Fortran */



#endif

#endif /* MATUTILS_H */
