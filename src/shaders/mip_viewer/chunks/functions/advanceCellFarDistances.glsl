#ifndef ADVANCE_CELL_FAR_DISTANCES
#define ADVANCE_CELL_FAR_DISTANCES

vec3 advanceCellFarDistances(vec3 tFar, ivec3 exitStep)
{
    return tFar + vec3(exitStep) * u_ray.step_distances;
}

#endif