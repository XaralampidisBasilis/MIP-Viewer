
// Ray-AABB (Axis Aligned Bounding Box) intersection.
// Mathematics: https://tavianator.com/2022/ray_box_boundary.html
// box: https://iquilezles.org/articles/intersectors/

#ifndef INTERSECTED_BOX
#define INTERSECTED_BOX

#ifndef MMIN
#include "../math/mmin"
#endif
#ifndef MMAX
#include "../math/mmax"
#endif

bool intersected_box(vec3 box_min, vec3 box_max, vec3 start, vec3 dir) 
{
    vec3 inv_dir = 1.0 / dir;
    vec3 b_min = (box_min - start) * inv_dir;
    vec3 b_max = (box_max - start) * inv_dir;
    float t_entry = mmax(min(b_min, b_max));
    float t_exit  = mmin(max(b_min, b_max));
    return t_exit >= max(t_entry, 0.0);
}

#endif