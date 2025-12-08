
#if SKIPPING_ENABLED == 1
#if SKIPPING_METHOD == 0
#include "./sample_distance/sample_occupancy"

#elif SKIPPING_METHOD == 1
#include "./sample_distance/sample_distance_isotropic"
    
#elif SKIPPING_METHOD == 2
#include "./sample_distance/sample_distance_anisotropic"

#elif SKIPPING_METHOD == 3
#include "./sample_distance/sample_distance_extended_isotropic"
   
#elif SKIPPING_METHOD == 4
#include "./sample_distance/sample_distance_extended_anisotropic"
#endif    
#endif
