#ifndef COMPUTE_SECOND_DERIVATIVES
#define COMPUTE_SECOND_DERIVATIVES

#ifndef SAMPLE_TRILINEAR_VOLUME
#include "../sampleVolume"
#endif

vec3 computeSecondDerivatives(in vec3 p)
{
    // Central differencing samples
    vec3 s_x0yz_xy0z_xyz0 = vec3(
        sampleVolume(vec3(p.x - 1.0, p.y, p.z)),
        sampleVolume(vec3(p.x, p.y - 1.0, p.z)),
        sampleVolume(vec3(p.x, p.y, p.z - 1.0))
    );

    vec3 s_x1yz_xy1z_xyz1 = vec3(
        sampleVolume(vec3(p.x + 1.0, p.y, p.z)),
        sampleVolume(vec3(p.x, p.y + 1.0, p.z)),
        sampleVolume(vec3(p.x, p.y, p.z + 1.0))
    );

    // Pure second derivatives
    return s_x0yz_xy0z_xyz0 + s_x1yz_xy1z_xyz1 - sampleVolume(p) * 2.0;
}

#endif