
#if GRADIENTS_METHOD == 0
#include "./computeGradient/computeGradientTrilinearAnalytic"

#elif GRADIENTS_METHOD == 1
#include "./computeGradient/computeGradientTriquadraticBspline"

#endif