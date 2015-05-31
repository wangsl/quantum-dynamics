
#ifndef GRADIENTS_H
#define GRADIENTS_H

#include "complex.h"

template<class T> __global__ void gradients_3d(const int n1, const int n2, const int n3,
					       const int n2p, const double dx,
					       const T *f, T *v, T *g, const int n_points);

#endif /* GRADIENTS_H */
