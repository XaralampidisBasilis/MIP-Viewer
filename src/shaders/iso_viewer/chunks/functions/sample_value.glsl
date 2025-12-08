
#if INTERPOLATION_METHOD == 0
#include "./sample_value/sample_value_trilinear"

#elif INTERPOLATION_METHOD == 1
#include "./sample_value/sample_value_tricubic"
#endif

