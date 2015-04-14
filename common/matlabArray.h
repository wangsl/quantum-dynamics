
/* $Id$ */

#ifndef MATLABARRAY_H
#define MATLABARRAY_H

#include <iostream>
#include <mex.h>
#include "matutils.h"

template<class T> class MatlabArray
{
public:
  const mxArray *mx;
  T *data;
  
  MatlabArray(const mxArray *mx_) :
    mx(mx_), data(0)
  {    
    data = (T *) mxGetData(mx);
    insist(data);
  }

  ~MatlabArray()
  { 
    mx = 0;
    data = 0;
  }
};

#endif /* MATLABARRAY_H */
