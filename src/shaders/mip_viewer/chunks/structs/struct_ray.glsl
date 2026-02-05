#ifndef STRUCT_RAY
#define STRUCT_RAY

struct Ray 
{
    bool  discarded;       // flag indicating if the ray has been discarded
    vec3  direction;       // direction vector for each step along the ray
    vec3  inv_direction;   // inverse of the direction vector
    float spacing;         // fixed step distance for each ray 
    ivec3 signs;           // the sign of the direction vector
    int   octant;          // the group index of the ray direction vector for 8 octant groups
    float start_distance;  // starting distance along the current ray from origin for ray march  
    vec3  start_position;  // starting position of the current ray in 3d model coordinates for ray march
    float end_distance;    // ending distance along the current ray from origin for ray march
    vec3  end_position;    // ending position of the current ray in 3d model coordinates for ray march
    float span_distance;   // total distance that can be covered by the current ray for ray march
};

Ray ray; // Global mutable struct

void set_ray()
{
    ray.discarded      = false;
    ray.direction      = vec3(0.0);
    ray.inv_direction  = vec3(0.0);
    ray.signs          = ivec3(0);
    ray.octant         = 0;
    ray.spacing        = 0.0;
    ray.start_position = vec3(0.0);
    ray.end_position   = vec3(0.0);
    ray.start_distance = 0.0;
    ray.end_distance   = 0.0;
    ray.span_distance  = 0.0;
}

#endif // STRUCT_RAY
