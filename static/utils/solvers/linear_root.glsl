#ifndef LINEAR_ROOT
#define LINEAR_ROOT

#ifndef MICRO_TOLERANCE
#define MICRO_TOLERANCE 1e-6
#endif

// Solves the linear equation: c[0] + c[1]*x^1 = 0
// We assume non zero linear coefficient

float linear_root(in vec2 c)
{
    // compute linear root
    float x = - c.x / c.y;
    return x;
}


#endif
