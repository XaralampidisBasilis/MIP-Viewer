
#ifndef BISECTION_ROOT
#define BISECTION_ROOT

#ifndef BISECTION_ITERATIONS
#define BISECTION_ITERATIONS 3
#endif
#ifndef EVAL_POLY
#include "../math/eval_poly"
#endif

float bisection_root(in vec4 c, in vec2 x0_x1)
{
    vec2 y0_y1 = eval_poly(c, x0_x1);
    
    float y;
    float x = (x0_x1.x + x0_x1.y) * 0.5;

    #pragma unroll
    for (int i = 0; i < BISECTION_ITERATIONS; ++i)
    {
        // evaluate new y using honers method
        y = eval_poly(c, x);

        // determine if the root is in the left or right sub-interval
        if ((y < 0.0) == (y0_y1.y < 0.0))
        {
            x0_x1.x = x;
            y0_y1.x = y;
        }
        else
        {
            x0_x1.y = x;
            y0_y1.y = y;
        }

        // bisection update
        x = (x0_x1.x + x0_x1.y) * 0.5;
    }

    return x;
}

float bisection_root(in float c[5], in vec2 x0_x1)
{;
    vec2 y0_y1 = eval_poly(c, x0_x1);

    float y;
    float x = (x0_x1.x + x0_x1.y) * 0.5;

    #pragma unroll
    for (int i = 0; i < BISECTION_ITERATIONS; ++i)
    {
        // evaluate new y using honers method
        y = eval_poly(c, x);

        // determine if the root is in the left or right sub-interval
        if ((y < 0.0) == (y0_y1.y < 0.0))
        {
            x0_x1.x = x;
            y0_y1.x = y;
        }
        else
        {
            x0_x1.y = x;
            y0_y1.y = y;
        }

        // bisection update
        x = (x0_x1.x + x0_x1.y) * 0.5;
    }

    return x;
}

float bisection_root(in float c[6], in vec2 x0_x1)
{
    vec2 y0_y1 = eval_poly(c, x0_x1);

    float y;
    float x = (x0_x1.x + x0_x1.y) * 0.5;

    #pragma unroll
    for (int i = 0; i < BISECTION_ITERATIONS; ++i)
    {
        // evaluate new y using honers method
        y = eval_poly(c, x);

        // determine if the root is in the left or right sub-interval
        if ((y < 0.0) == (y0_y1.y < 0.0))
        {
            x0_x1.x = x;
            y0_y1.x = y;
        }
        else
        {
            x0_x1.y = x;
            y0_y1.y = y;
        }

        // bisection update
        x = (x0_x1.x + x0_x1.y) * 0.5;
    }

    return x;
}


#endif