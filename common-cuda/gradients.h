
#ifndef GRADIENTS_H
#define GRADIENTS_H

#include "complex.h"

void copy_gradient_coefficients_to_device(const int n_points);

template<class T> __global__ void gradients_3d(const int n1, const int n2, const int n3,
					       const int n2p, const double dx2,
					       const T *f, T *v, T *g, const int n_points);

template<class T> __global__ void gradients2_3d(const int n1, const int n2, const int n3,
						const int n2p, const double dx2,
						const T *f, T *v, T *g, const int n_points);

#endif /* GRADIENTS_H */
