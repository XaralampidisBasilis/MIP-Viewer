
#if GRADIENTS_METHOD == 0
#include "./compute_gradient/compute_gradient_trilinear_analytic"

#elif GRADIENTS_METHOD == 2
#include "./compute_gradient/compute_gradient_triquadratic_bspline"

#endif