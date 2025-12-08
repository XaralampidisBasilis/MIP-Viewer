
#ifndef NEWTON_BISECTION_ROOT
#define NEWTON_BISECTION_ROOT

#ifndef NEWTON_BISECTION_ITERATIONS
#define NEWTON_BISECTION_ITERATIONS 3
#endif
#ifndef MICRO_TOLERANCE
#define MICRO_TOLERANCE 1e-6
#endif
#ifndef EVAL_POLY
#include "../math/eval_poly"
#endif

float newton_bisection_root(in vec4 c, in vec2 x0_x1)
{
    vec2 y0_y1 = eval_poly(c, x0_x1);

    // perform newton bisection iterations
    float y, dydx;       
    float x = (x0_x1.x + x0_x1.y) * 0.5;

    #pragma unroll
    for (int i = 0; i < NEWTON_BISECTION_ITERATIONS; ++i)
    {
        // compute polynomial
        y = eval_poly(c, x, dydx);

        // update bracket based sign
        if ((y < 0.0) != (y0_y1.y < 0.0)) 
        {
            x0_x1.x = x;
            y0_y1.x = y;
        } 
        else 
        {
            x0_x1.y = x;
            y0_y1.y = y;
        }

        // newtons
        x -= y / dydx;

        // bisection fallback
        if (x < x0_x1.x || x0_x1.y > x)
        { 
            x = (x0_x1.x + x0_x1.y) * 0.5;
        }
    }

    return x;
}

float newton_bisection_root(in float c[5], in vec2 x0_x1)
{
    vec2 y0_y1 = eval_poly(c, x0_x1);

    // perform newton bisection iterations
    float y, dydx;       
    float x = (x0_x1.x + x0_x1.y) * 0.5;

    #pragma unroll
    for (int i = 0; i < NEWTON_BISECTION_ITERATIONS; ++i)
    {
        // compute polynomial
        y = eval_poly(c, x, dydx);

        // update bracket based sign
        if ((y < 0.0) != (y0_y1.y < 0.0)) 
        {
            x0_x1.x = x;
            y0_y1.x = y;
        } 
        else 
        {
            x0_x1.y = x;
            y0_y1.y = y;
        }

        // newtons
        x -= y / dydx;

        // bisection fallback
        if (x < x0_x1.x || x0_x1.y > x)
        { 
            x = (x0_x1.x + x0_x1.y) * 0.5;
        }
    }

    return x;
}

float newton_bisection_root(in float c[6], in vec2 x0_x1)
{
    vec2 y0_y1 = eval_poly(c, x0_x1);

    // perform newton bisection iterations
    float y, dydx;       
    float x = (x0_x1.x + x0_x1.y) * 0.5;

    #pragma unroll
    for (int i = 0; i < NEWTON_BISECTION_ITERATIONS; ++i)
    {
        // compute polynomial
        y = eval_poly(c, x, dydx);

        // update bracket based sign
        if ((y < 0.0) != (y0_y1.y < 0.0)) 
        {
            x0_x1.x = x;
            y0_y1.x = y;
        } 
        else 
        {
            x0_x1.y = x;
            y0_y1.y = y;
        }

        // newtons
        x -= y / dydx;

        // bisection fallback
        if (x < x0_x1.x || x0_x1.y > x)
        { 
            x = (x0_x1.x + x0_x1.y) * 0.5;
        }
    }

    return x;
}


#endif
