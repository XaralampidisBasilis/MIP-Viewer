

#ifndef EVAL_POLY_SIGN_CHANGES
#define EVAL_POLY_SIGN_CHANGES

#ifndef EVAL_POLY
#include "./eval_poly"
#endif
#ifndef SIGN_CHANGE
#include "./sign_changes"
#endif
#ifndef MMIX
#include "./mmix"
#endif

// Compute samples with non overlapping batches

int eval_poly_sign_changes(vec3 coeffs)
{
    const int batches = 2;
    const int samples = batches * 4;
    const float spacing = 1.0 / float(samples - 1);
    const vec4 stride = vec4(spacing * 4.0);

    vec4 points = vec4(0, 1, 2, 3) * spacing;
    vec4 values = eval_poly(coeffs, points);
    int changes = sign_changes(values);
    float prev_value = values.w;

    #pragma unroll
    for (int i = 1; i < batches; ++i) 
    {   
        points += stride;
        values = eval_poly(coeffs, points);
        changes += sign_changes(prev_value, values.x) + sign_changes(values);
        prev_value = values.w;
    }

    return changes;
}

int eval_poly_sign_changes(vec4 coeffs)
{
    const int batches = 3;
    const int samples = batches * 4;
    const float spacing = 1.0 / float(samples - 1);
    const vec4 stride = vec4(spacing * 4.0);

    vec4 points = vec4(0, 1, 2, 3) * spacing;
    vec4 values = eval_poly(coeffs, points);
    int changes = sign_changes(values);
    float prev_value = values.w;

    #pragma unroll
    for (int i = 1; i < batches; ++i) 
    {   
        points += stride;
        values = eval_poly(coeffs, points);
        changes += sign_changes(prev_value, values.x) + sign_changes(values);
        prev_value = values.w;
    }

    return changes;
}

int eval_poly_sign_changes(float[5] coeffs)
{
    const int batches = 4;
    const int samples = batches * 4;
    const float spacing = 1.0 / float(samples - 1);
    const vec4 stride = vec4(spacing * 4.0);

    vec4 points = vec4(0, 1, 2, 3) * spacing;
    vec4 values = eval_poly(coeffs, points);
    int changes = sign_changes(values);
    float prev_value = values.w;

    #pragma unroll
    for (int i = 1; i < batches; ++i) 
    {   
        points += stride;
        values = eval_poly(coeffs, points);
        changes += sign_changes(prev_value, values.x) + sign_changes(values);
        prev_value = values.w;
    }

    return changes;
}

int eval_poly_sign_changes(float[6] coeffs)
{
    const int batches = 4;
    const int samples = batches * 4;
    const float spacing = 1.0 / float(samples - 1);
    const vec4 stride = vec4(spacing * 4.0);

    vec4 points = vec4(0, 1, 2, 3) * spacing;
    vec4 values = eval_poly(coeffs, points);
    int changes = sign_changes(values);
    float prev_value = values.w;

    #pragma unroll
    for (int i = 1; i < batches; ++i) 
    {   
        points += stride;
        values = eval_poly(coeffs, points);
        changes += sign_changes(prev_value, values.x) + sign_changes(values);
        prev_value = values.w;
    }

    return changes;
}



#endif