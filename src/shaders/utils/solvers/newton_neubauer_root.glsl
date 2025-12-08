
#ifndef NEWTON_NEUBAUER_ROOT
#define NEWTON_NEUBAUER_ROOT

#ifndef NEWTON_NEUBAUER_ITERATIONS
#define NEWTON_NEUBAUER_ITERATIONS 3
#endif
#ifndef MICRO_TOLERANCE
#define MICRO_TOLERANCE 1e-6
#endif
#ifndef EVAL_POLY
#include "../math/eval_poly"
#endif

float newton_neubauer_root(in vec4 c, in vec2 x0_x1)
{
    vec2 y0_y1 = eval_poly(c, x0_x1);

    // initialize neubauer
    float dx = x0_x1.y - x0_x1.x;
    float dy = y0_y1.y - y0_y1.x;
    float dydx, y, x = x0_x1.x - (y0_y1.x * dx) / dy;

    #pragma unroll
    for (int i = 0; i < NEWTON_NEUBAUER_ITERATIONS; ++i)
    {
        // compute polynomial
        y = eval_poly(c, x, dydx);

        // update bracket based on value signs
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

        // neubauer fallback
        if (x < x0_x1.x || x0_x1.y > x)
        { 
            dx = x0_x1.y - x0_x1.x;
            dy = y0_y1.y - y0_y1.x + MICRO_TOLERANCE;
            x = x0_x1.x - (y0_y1.x * dx) / dy;
        }
    }
    
    return x;
}

float newton_neubauer_root(in float c[5], in vec2 x0_x1)
{
    vec2 y0_y1 = eval_poly(c, x0_x1);

    // neubauer
    float dx = x0_x1.y - x0_x1.x;
    float dy = y0_y1.y - y0_y1.x;
    float dydx, y, x = x0_x1.x - (y0_y1.x * dx) / dy;

    #pragma unroll
    for (int i = 0; i < NEWTON_NEUBAUER_ITERATIONS; ++i)
    {
        // compute polynomial
        y = eval_poly(c, x, dydx);

        // update bracket based on value signs
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

        // neubauer fallback
        if (x < x0_x1.x || x0_x1.y > x)
        { 
            dx = x0_x1.y - x0_x1.x;
            dy = y0_y1.y - y0_y1.x + MICRO_TOLERANCE;
            x = x0_x1.x - (y0_y1.x * dx) / dy;
        }
    }
    
    return x;
}

float newton_neubauer_root(in float c[6], in vec2 x0_x1)
{
    vec2 y0_y1 = eval_poly(c, x0_x1);

    // neubauer
    float dx = x0_x1.y - x0_x1.x;
    float dy = y0_y1.y - y0_y1.x;
    float dydx, y, x = x0_x1.x - (y0_y1.x * dx) / dy;

    #pragma unroll
    for (int i = 0; i < NEWTON_NEUBAUER_ITERATIONS; ++i)
    {
        // compute polynomial
        y = eval_poly(c, x, dydx);

        // update bracket based on value signs
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

        // neubauer fallback
        if (x < x0_x1.x || x0_x1.y > x)
        { 
            dx = x0_x1.y - x0_x1.x;
            dy = y0_y1.y - y0_y1.x + MICRO_TOLERANCE;
            x = x0_x1.x - (y0_y1.x * dx) / dy;
        }
    }
    
    return x;
}


#endif