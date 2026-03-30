#ifndef STRUCT_CUBIC
#define STRUCT_CUBIC

const mat4 CUBIC_INV_VANDER = mat4(
    2, 0, 0, 0, 
    -11, 18, -9, 2, 
    18, -45, 36, -9, 
    -9, 27, -27, 9
) / 2.0;

const mat4 CUBIC_INV_BERNSTEIN = mat4(
    6, 0, 0, 0,  
    -5, 18, -9, 2, 
    2, -9, 18, -5,
    0, 0, 0, 6   
) / 6.0;

struct Cubic 
{
    bool   maximize;
    vec4   values;
    vec4   coeffs;    
    vec4   bernstein_coeffs; 
    float  max_value;
    float  argmax_point;
};

Cubic cubic; // Global mutable struct

#endif 
