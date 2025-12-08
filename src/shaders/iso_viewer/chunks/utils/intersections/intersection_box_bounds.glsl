/**
 * Calculates the minimum and maximum distance from a point to a 3D box.
 *
 * @param b_min   The minimum corner (bottom-left-front) of the box.
 * @param b_max   The maximum corner (top-right-back) of the box.
 * @param P       The 3D point in space.
 * @return float    The min distance from the point to the box.
 */

#ifndef INTERSECTION_BOX_BOUNDS
#define INTERSECTION_BOX_BOUNDS

#ifndef MMAX
#include "../math/mmax"
#endif

vec2 intersection_box_bounds(vec3 b_min, vec3 b_max, vec3 p) 
{
    vec3 c = (b_max + b_min) * 0.5;
    vec3 s = (b_max - b_min) * 0.5;
    vec3 aq = abs(p - c);
    vec3 d_min = aq - s;
    vec3 d_max = aq + s;    
    return vec2(length(max(d_min, 0.0) + min(mmax(d_min), 0.0)), length(d_max));
}

#endif // BOX_BOUNDS
