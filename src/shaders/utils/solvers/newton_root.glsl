
#ifndef NEWTON_ROOT
#define NEWTON_ROOT

#ifndef NEWTON_ITERATIONS
#define NEWTON_ITERATIONS 3
#endif
#ifndef MICRO_TOLERANCE
#define MICRO_TOLERANCE 1e-6
#endif
#ifndef EVAL_POLY
#include "../math/eval_poly"
#endif

float newton_root(in vec4 c, in float x)
{
    float y, dydx;

    #pragma unroll
    for (int i = 0; i < NEWTON_ITERATIONS; ++i)
    {
        // compute polynomial
        y = eval_poly(c, x, dydx);

        // newtons
        x -= y / dydx;
    }

    return x;
}

float newton_root(in float c[5], in float x)
{
    float y, dydx;

    #pragma unroll
    for (int i = 0; i < NEWTON_ITERATIONS; ++i)
    {
        // compute polynomial
        y = eval_poly(c, x, dydx);

        // newtons
        x -= y / dydx;
    }

    return x;
}

float newton_root(in float c[6], in float x)
{
    float y, dydx;

    #pragma unroll
    for (int i = 0; i < NEWTON_ITERATIONS; ++i)
    {
        // compute polynomial
        y = eval_poly(c, x, dydx);

        // newtons
        x -= y / dydx;
    }

    return x;
}


#endif
