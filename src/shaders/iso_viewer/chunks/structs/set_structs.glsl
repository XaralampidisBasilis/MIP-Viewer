set_camera();
set_box();
set_ray();
set_trace();
set_hit();
set_cell();
set_block();
set_frag();

#if DEBUG_ENABLED == 1
set_debug();
#endif

#if DEBUG_ENABLED == 1
set_stats();
#endif

#if INTERPOLATION_METHOD == 0
set_cubic();

#elif INTERPOLATION_METHOD == 1
set_quintic();
#endif
