#include "./math/math"
#include "./random/random"
#include "./inclusions/inside_open"
#include "./inclusions/inside_closed"
#include "./intersections/intersect_box"
#include "./intersections/intersect_box_exit"
#include "./intersections/intersection_box_bounds"
#include "./matrix/sum_anti_diags"
#include "./colors/to_color"
#include "./colors/colormap"
#include "./colors/rgb2hsv"
#include "./colors/hsv2rgb"
#include "./colors/constants"

#if INTERPOLATION_METHOD == 0
#include "./solvers/cubic_roots"
#include "./solvers/cubic_has_root"          

#elif INTERPOLATION_METHOD == 1
#include "./solvers/quintic_roots"
#include "./solvers/quintic_has_root"       
#endif
