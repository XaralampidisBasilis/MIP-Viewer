#include "./struct_camera"
#include "./struct_frag"
#include "./struct_box"
#include "./struct_ray"
#include "./struct_trace"
#include "./struct_hit"
#include "./struct_cell"
#include "./struct_block"

#if DEBUG_ENABLED == 1
#include "./struct_debug"
#endif

#if DEBUG_ENABLED == 1
#include "./struct_stats"
#endif

#if INTERPOLATION_METHOD == 0
#include "./struct_cubic"

#elif INTERPOLATION_METHOD == 1
#include "./struct_quintic"
#endif
