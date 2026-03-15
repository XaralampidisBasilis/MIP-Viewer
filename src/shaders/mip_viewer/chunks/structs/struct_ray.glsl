#ifndef STRUCT_RAY
#define STRUCT_RAY

struct Ray 
{
    bool  reversed;
    uint  map;       
    uint  axis;       
    uint  idx;
    bool  discarded;       // flag indicating if the ray has been discarded
    vec3  origin;
    vec3  direction;       // direction vector for each step along the ray
    vec3  inv_direction;   // inverse of the direction vector
    float spacing;         // fixed step distance for each ray 
    ivec3 signs;           // the sign of the direction vector
    float start_distance;  // starting distance along the current ray from origin for ray march  
    vec3  start_position;  // starting position of the current ray in 3d model coordinates for ray march
    float end_distance;    // ending distance along the current ray from origin for ray march
    vec3  end_position;    // ending position of the current ray in 3d model coordinates for ray march
    float span_distance;   // total distance that can be covered by the current ray for ray march
};

Ray ray; // Global mutable struct

void set_ray()
{
    ray.reversed       = false;
    ray.discarded      = false;
    ray.map            = uint(u_ray.map);
    ray.axis           = uint(u_ray.axis);
    ray.idx            = uint(u_ray.idx);
    ray.origin         = v_ray_origin;
    ray.direction      = u_ray.direction;
    ray.inv_direction  = u_ray.inv_direction;
    ray.signs          = u_ray.signs;
    ray.spacing        = u_ray.spacing;
    ray.start_position = vec3(0.0);
    ray.end_position   = vec3(0.0);
    ray.start_distance = 0.0;
    ray.end_distance   = 0.0;
    ray.span_distance  = 0.0;
}

#endif // STRUCT_RAY
