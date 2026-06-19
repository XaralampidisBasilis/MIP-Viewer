

#ifndef EVAL_POLY_SIGN_CHANGE
#define EVAL_POLY_SIGN_CHANGE

#ifndef EVAL_POLY
#include "./eval_poly"
#endif
#ifndef SIGN_CHANGE
#include "./sign_change"
#endif
#ifndef MMIX
#include "./mmix"
#endif

// Compute samples with non overlapping batches
bool eval_poly_sign_change(vec3 coeffs)
{
    const int batches = 2;
    const int samples = batches * 4;
    const float spacing = 1.0 / float(samples - 1);
    const vec4 stride = vec4(spacing * 4.0);

    vec4 points = vec4(0, 1, 2, 3) * spacing;
    vec4 values = eval_poly(coeffs, points);
    bool change = sign_change(values);
    float prev = values.w;

    #pragma unroll
    for (int i = 1; i < batches; ++i) 
    {   
        points += stride;
        values = eval_poly(coeffs, points);
        change = change || sign_change(prev, values.x) || sign_change(values);
        prev = values.w;
    }

    return change;
}

bool eval_poly_sign_change(vec4 coeffs)
{
    const int batches = 3;
    const int samples = batches * 4;
    const float spacing = 1.0 / float(samples - 1);
    const vec4 stride = vec4(spacing * 4.0);

    vec4 points = vec4(0, 1, 2, 3) * spacing;
    vec4 values = eval_poly(coeffs, points);
    bool change = sign_change(values);
    float prev = values.w;

    #pragma unroll
    for (int i = 1; i < batches; ++i) 
    {   
        points += stride;
        values = eval_poly(coeffs, points);
        change = change || sign_change(prev, values.x) || sign_change(values);
        prev = values.w;
    }

    return change;
}

bool eval_poly_sign_change(float[5] coeffs)
{
    const int batches = 4;
    const int samples = batches * 4;
    const float spacing = 1.0 / float(samples - 1);
    const vec4 stride = vec4(spacing * 4.0);

    vec4 points = vec4(0, 1, 2, 3) * spacing;
    vec4 values = eval_poly(coeffs, points);
    bool change = sign_change(values);
    float prev = values.w;

    #pragma unroll
    for (int i = 1; i < batches; ++i) 
    {   
        points += stride;
        values = eval_poly(coeffs, points);
        change = change || sign_change(prev, values.x) || sign_change(values);
        prev = values.w;
    }

    return change;
}

bool eval_poly_sign_change(float[6] coeffs)
{
    const int batches = 4;
    const int samples = batches * 4;
    const float spacing = 1.0 / float(samples - 1);
    const vec4 stride = vec4(spacing * 4.0);

    vec4 points = vec4(0, 1, 2, 3) * spacing;
    vec4 values = eval_poly(coeffs, points);
    bool change = sign_change(values);
    float prev = values.w;

    #pragma unroll
    for (int i = 1; i < batches; ++i) 
    {   
        points += stride;
        values = eval_poly(coeffs, points);
        change = change || sign_change(prev, values.x) || sign_change(values);
        prev = values.w;
    }

    return change;
}

#endif