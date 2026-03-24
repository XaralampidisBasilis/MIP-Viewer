#ifndef STRUCT_RAY
#define STRUCT_RAY

const float eps = 0.001;

struct Ray 
{
    bool  reversed;
    bool  discarded;       // flag indicating if the ray has been discarded
    float phase;     
    vec3  origin;
    vec3  direction;       // direction vector for each step along the ray
    float step_distance;         // fixed step distance for each ray 
    float eps_distance;
    vec3  eps_direction;
    float start_distance;  // starting distance along the current ray from origin for ray march  
    vec3  start_position;  // starting position of the current ray in 3d model coordinates for ray march
    float end_distance;    // ending distance along the current ray from origin for ray march
    vec3  end_position;    // ending position of the current ray in 3d model coordinates for ray march
    float span_distance;   // total distance that can be covered by the current ray for ray march
};

Ray ray; // Global mutable struct

void set_ray()
{
    ray.phase          = random(v_ray_origin);
    ray.discarded      = false;
    ray.reversed       = u_ray.reverse;
    ray.origin         = v_ray_origin;
    ray.direction      = u_ray.direction;
    ray.step_distance  = u_ray.step_distance;
    ray.eps_direction  = u_ray.direction * eps;
    ray.eps_distance    = u_ray.step_distance * eps;
    ray.start_position = vec3(0.0);
    ray.end_position   = vec3(0.0);
    ray.start_distance = 0.0;
    ray.end_distance   = 0.0;
    ray.span_distance  = 0.0;
}

#endif // STRUCT_RAY
