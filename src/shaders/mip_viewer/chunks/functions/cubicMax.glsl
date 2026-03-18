#ifndef CUBIC_MAX
#define CUBIC_MAX

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

struct CubicMax {
    float v;
    float t;
};

CubicMax cubicMaxFromValues(vec4 vals) 
{
    // quadratic coefficients
    vec4 c = vals * CUBIC_INV_VANDER;
    vec3 q = vec3(c.y, 2.0 * c.z, 3.0 * c.w);

    // derivative roots
    vec2 t_ext = clamp(quadratic_roots(q), 0.0, 1.0);

    // evaluate extrema
    vec2 v_ext = eval_poly(c, t_ext);

    // endpoints + extrema
    vec4 v = vec4(vals.x, v_ext.x, v_ext.y, vals.w);
    vec4 t = vec4(0.0, t_ext.x, t_ext.y, 1.0);

    int i = argmax(v);
    return CubicMax(v[i], t[i]);
}

CubicMax cubicMaxFromCoeffs(vec4 c)
{
    // quadratic coefficients
    vec3 q = vec3(c.y, 2.0 * c.z, 3.0 * c.w);

    // derivative roots
    vec2 t_ext = clamp(quadratic_roots(q), 0.0, 1.0);

    // evaluate extrema
    vec2 v_ext = eval_poly(c, t_ext);

    // evaluate endpoints directly
    float v0 = c.x;
    float v1 = c.x + c.y + c.z + c.w;

    // pack for argmax
    vec4 v = vec4(v0, v_ext.x, v_ext.y, v1);
    vec4 t = vec4(0.0, t_ext.x, t_ext.y, 1.0);

    int i = argmax(v);
    return CubicMax(v[i], t[i]);
}

#endif