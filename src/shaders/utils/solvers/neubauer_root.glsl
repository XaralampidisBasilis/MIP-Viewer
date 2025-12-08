
#ifndef NEUBAUER_ROOT
#define NEUBAUER_ROOT

#ifndef NEUBAUER_ITERATIONS
#define NEUBAUER_ITERATIONS 3
#endif
#ifndef MICRO_TOLERANCE
#define MICRO_TOLERANCE 1e-6
#endif
#ifndef EVAL_POLY
#include "../math/eval_poly"
#endif

// Find a root of the polynomial c0 + c1x^1 + ... + cnx^n = 0 for x in [x0, x1]

float neubauer_root(in vec4 c, in vec2 x0_x1)
{
    vec2 y0_y1 = eval_poly(c, x0_x1);

    // initialize neubauer
    float dx = x0_x1.y - x0_x1.x;
    float dy = y0_y1.y - y0_y1.x;
    float y, x = x0_x1.x - (y0_y1.x * dx) / dy;

    #pragma unroll
    for (int i = 0; i < NEUBAUER_ITERATIONS; ++i)
    {
        // evaluate polynomial
        y = eval_poly(c, x);

        // determine bracket based on sign
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

        // compute neubauer update
        dx = x0_x1.y - x0_x1.x;
        dy = y0_y1.y - y0_y1.x + MICRO_TOLERANCE;
        x = x0_x1.x - (y0_y1.x * dx) / dy;
    }

    return x;
}

float neubauer_root(in float c[5], in vec2 x0_x1)
{
    vec2 y0_y1 = eval_poly(c, x0_x1);

    // initialize neubauer
    float dx = x0_x1.y - x0_x1.x;
    float dy = y0_y1.y - y0_y1.x;
    float y, x = x0_x1.x - (y0_y1.x * dx) / dy;

    #pragma unroll
    for (int i = 0; i < NEUBAUER_ITERATIONS; ++i)
    {
        // evaluate polynomial
        y = eval_poly(c, x);

        // determine bracket based on sign
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

        // compute neubauer update
        dx = x0_x1.y - x0_x1.x;
        dy = y0_y1.y - y0_y1.x + MICRO_TOLERANCE;
        x = x0_x1.x - (y0_y1.x * dx) / dy;
    }

    return x;
}

float neubauer_root(in float c[6], in vec2 x0_x1)
{
    vec2 y0_y1 = eval_poly(c, x0_x1);

    // initialize neubauer
    float dx = x0_x1.y - x0_x1.x;
    float dy = y0_y1.y - y0_y1.x;
    float y, x = x0_x1.x - (y0_y1.x * dx) / dy;

    #pragma unroll
    for (int i = 0; i < NEUBAUER_ITERATIONS; ++i)
    {
        // evaluate polynomial
        y = eval_poly(c, x);

        // determine bracket based on sign
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

        // compute neubauer update
        dx = x0_x1.y - x0_x1.x;
        dy = y0_y1.y - y0_y1.x + MICRO_TOLERANCE;
        x = x0_x1.x - (y0_y1.x * dx) / dy;
    }

    return x;
}

#endif