#ifndef SHOULD_INTERSECT_MIP
#define SHOULD_INTERSECT_MIP

bool shouldIntersectMip(float mipValue, float newValue)
{
    return abs(mipValue - newValue) < u_debug.variable1;
}

#endif
