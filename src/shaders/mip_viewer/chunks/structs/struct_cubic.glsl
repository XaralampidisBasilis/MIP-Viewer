#ifndef STRUCT_CUBIC
#define STRUCT_CUBIC

struct Cubic 
{
    vec4  values;
    vec4  distances;
    vec4  coeffs;    
    vec4  bernstein_coeffs; 
    vec4  roots;
    float root;
    int   num_roots;
    vec2  crit_points;
    float  max_value;
};

Cubic cubic; // Global mutable struct

void set_cubic()
{
    cubic.roots = vec4(0);
    cubic.values = vec4(0);
    cubic.distances = vec4(0);
    cubic.coeffs = vec4(0);
    cubic.bernstein_coeffs = vec4(0);
    cubic.root = 0.0;
    cubic.num_roots = 0;
    cubic.crit_points = vec2(0);
    cubic.max_value = 0.0;
}

#endif 
