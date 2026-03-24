#ifndef CELL_COORDS_TO_FAR_DISTANCES
#define CELL_COORDS_TO_FAR_DISTANCES

#ifndef CELL_COORDS_TO_MIN_POSITION
#include "./cellCoordsToMinPosition"
#endif

vec3 cellCoordsToFarDistances(ivec3 coords)
{
    vec3 cMin = cellCoordsToMinPosition(coords);
    vec3 cMax = cMin + vec3(1.0);
    
    vec3 cFar = vec3(
        u_ray.sign_direction.x > 0 ? cMax.x : cMin.x,
        u_ray.sign_direction.y > 0 ? cMax.y : cMin.y,
        u_ray.sign_direction.z > 0 ? cMax.z : cMin.z
    );

    vec3 tFar = (cFar - v_ray_origin) * u_ray.inv_direction;
    
    return tFar;
}

#endif
