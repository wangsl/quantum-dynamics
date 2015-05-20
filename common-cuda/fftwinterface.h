
/* $Id$ */

#ifndef FFTWINTERFACE_H
#define FFTWINTERFACE_H

#include <fftw3.h>
#include "complex.h"
#include "rmat.h"
#include "fftwinterface.h"

class FFTWInterface
{
private:
  fftw_plan forward_plan;
  fftw_plan backward_plan;
  int has_forward_plan;
  int has_backward_plan;

  int fftw_flag;
  
  double *f;
  int m;
  int n;
  int column_major;
  
  // 2d matrix is m X n, similar as in Matlab

  void create_fftw_plan(int &has_plan, fftw_plan &plan,
			const int fftw_direction, const int fftw_flag);
  
  void create_forward_plan();
  void create_backward_plan();
  
public:
  FFTWInterface(double *f, const int n, const int flag = FFTW_ESTIMATE);
  FFTWInterface(double *f, const int m, const int n, int flag = FFTW_ESTIMATE, int column_major = 1);
  
  ~FFTWInterface();
  
  void forward_transform();
  void backward_transform();

  static void get_momentum_for_fftw(double *p, const int nm, const double xl);
  static void get_momentum_for_fftw(RVec &p, const double xl);
};

#endif /* FFTWINTERFACE */

