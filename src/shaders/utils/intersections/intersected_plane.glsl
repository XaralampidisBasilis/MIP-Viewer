
#ifndef INTERSECTED_PLANE
#define INTERSECTED_PLANE

#ifndef MICRO_TOLERANCE
#define MICRO_TOLERANCE 1e-6
#endif

bool intersected_plane(vec4 hessian, vec3 origin, vec3 direction) 
{
    float denominator = dot(hessian.xyz, direction);
    if ( abs(denominator) < MICRO_TOLERANCE ) 
    {
        return false;
    }

    float distance = -dot(hessian, vec4(origin, 1.0)) / denominator;
    return distance > 0.0;
}

#endif 