#ifndef COMPUTE_SECOND_DERIVATIVES
#define COMPUTE_SECOND_DERIVATIVES

#ifndef SAMPLE_TRILINEAR_VOLUME
#include "../sample_volume"
#endif

vec3 compute_second_derivatives(in vec3 p)
{
    // Central differencing samples
    vec3 s_x0yz_xy0z_xyz0 = vec3(
        sample_volume(vec3(p.x - 1.0, p.y, p.z)),
        sample_volume(vec3(p.x, p.y - 1.0, p.z)),
        sample_volume(vec3(p.x, p.y, p.z - 1.0))
    );

    vec3 s_x1yz_xy1z_xyz1 = vec3(
        sample_volume(vec3(p.x + 1.0, p.y, p.z)),
        sample_volume(vec3(p.x, p.y + 1.0, p.z)),
        sample_volume(vec3(p.x, p.y, p.z + 1.0))
    );

    // Pure second derivatives
    return s_x0yz_xy0z_xyz0 + s_x1yz_xy1z_xyz1 - sample_volume(p) * 2.0;
}

#endif