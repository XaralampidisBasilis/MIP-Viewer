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

vec2 intersection_box_bounds(vec3 boxMin, vec3 boxMax, vec3 point)
{
    vec3 center = 0.5 * (boxMin + boxMax);
    vec3 halfExtent = 0.5 * (boxMax - boxMin);
    vec3 q = abs(point - center) - halfExtent;

    float minDist = length(max(q, 0.0)) + min(mmax(q), 0.0);
    float maxDist = length(abs(point - center) + halfExtent);

    return vec2(minDist, maxDist);
}

#endif // BOX_BOUNDS
