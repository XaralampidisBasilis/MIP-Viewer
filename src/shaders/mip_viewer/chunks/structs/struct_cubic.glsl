#ifndef STRUCT_CUBIC
#define STRUCT_CUBIC

struct Cubic 
{
    vec4   values;
    vec4   coeffs;    
    vec4   bernstein_coeffs; 
    float  max_value;
    float  argmax_time;
};

Cubic cubic; // Global mutable struct

void set_cubic()
{
    cubic.values = vec4(0);
    cubic.coeffs = vec4(0);
    cubic.bernstein_coeffs = vec4(0);
    cubic.max_value = 0.0;
    cubic.argmax_time = 0.0;
}

#endif 
