#ifndef SHOULD_INTERSECT_MIP
#define SHOULD_INTERSECT_MIP

bool shouldIntersectMip(float mipValue, float newValue)
{
    #if VARIATION_ENABLED == 1

    return abs(1.0 - newValue / mipValue) < u_debug.variable1;

    #else
    
    return abs(mipValue - newValue) < u_debug.variable1;

    #endif
}

#endif
