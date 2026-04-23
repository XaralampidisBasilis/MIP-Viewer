#ifndef SHOULD_MAXIMIZE_CUBIC
#define SHOULD_MAXIMIZE_CUBIC

bool shouldMaximizeCubic(vec4 bernsteinCoeffs, float mipValue)
{
    return any(greaterThan(bernsteinCoeffs, vec4(mipValue)));
}

bool shouldMaximizeCubic(vec4 bernsteinCoeffs, float mipValue, float tolerance)
{
    return any(greaterThan(bernsteinCoeffs, vec4(mipValue + tolerance)));
}

#endif
