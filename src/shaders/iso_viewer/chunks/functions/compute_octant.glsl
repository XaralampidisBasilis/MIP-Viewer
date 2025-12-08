#ifndef COMPUTE_OCTANT
#define COMPUTE_OCTANT

int compute_octant(in vec3 d)
{
    ivec3 b = ivec3(step(0.0, d));         
    return (b.z << 2) | (b.y << 1) | b.x;
}

#endif