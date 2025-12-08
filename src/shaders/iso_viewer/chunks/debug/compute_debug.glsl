#include "./modules/debug_ray" 
#include "./modules/debug_cell"  
#include "./modules/debug_trace"  
#include "./modules/debug_block"  
#include "./modules/debug_hit"  
#include "./modules/debug_frag"               
#include "./modules/debug_box"   
#include "./modules/debug_camera"          
#include "./modules/debug_variables"               

#if DEBUG_ENABLED == 1
#include "./modules/debug_stats"  
#endif

#if INTERPOLATION_METHOD == 0
#include "./modules/debug_cubic"          

#elif INTERPOLATION_METHOD == 1
#include "./modules/debug_quintic"          
#endif
