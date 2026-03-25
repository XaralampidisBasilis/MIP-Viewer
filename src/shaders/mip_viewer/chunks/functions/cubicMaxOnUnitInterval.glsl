#ifndef CUBIC_MAX_ON_UNIT_INTERVAL
#define CUBIC_MAX_ON_UNIT_INTERVAL

struct CubicMax {
    float value;
    float point;
};

// Evaluate cubic with Horner form
float evalCubic(vec4 c, float t)
{
    return ((c.w * t + c.z) * t + c.y) * t + c.x;
}

float signNonZero(float x)
{
    return (x >= 0.0) ? 1.0 : -1.0;
}

// Solve the quadratic derivative and select the stationary point
// with negative second derivative, which is the local maximum. 
// Uses a stable quadratic solve for p'(t) from Vieta.

CubicMax cubicMaxOnUnitInterval(vec4 c, float v0, float v1)
{
    const float eps = 1e-6;

    float bestV = v0;
    float bestT = 0.0;

    if (v1 > bestV)
    {
        bestV = v1;
        bestT = 1.0;
    }

    float d = c.y;
    float b = 2.0 * c.z;
    float a = 3.0 * c.w;

    float disc = b * b - 4.0 * a * d;
    if (disc < 0.0)
    {
        return CubicMax(bestV, bestT);
    }

    float s = sqrt(disc);
    float q  = -0.5 * (b + signNonZero(b) * s);

    // First root stays stable in the linear limit a -> 0: t1 = d/q -> -d/b
    float t0 = d / q;
    float t1 = q / a;
       
    // p''(t) = 2*a*t + b
    float dd0 = 2.0 * a * t0 + b;

    // Pick the stationary point that is a local maximum
    float t = (dd0 < 0.0) ? t0 : t1;

    if (t > 0.0 && t < 1.0)
    {
        float v = evalCubic(c, t);
        if (v > bestV)
        {
            bestV = v;
            bestT = t;
        }
    }

    return CubicMax(bestV, bestT);
}

#endif

/*

// Treat p'(t) as linear when the quadratic term is negligible.
// Otherwise solve the quadratic derivative and select the stationary point
// with negative second derivative, which is the local maximum

CubicMax cubicMaxOnUnitInterval(vec4 c, float v0, float v1)
{
    const float eps = 1e-6;

    float bestV = v0;
    float bestT = 0.0;

    if (v1 > bestV)
    {
        bestV = v1;
        bestT = 1.0;
    }

    float d = c.y;
    float b = 2.0 * c.z;
    float a = 3.0 * c.w;

    if (abs(a) < eps)
    {
        if (abs(b) >= eps)
        {
            float t = -d / b;
            if (t > 0.0 && t < 1.0)
            {
                float v = evalCubic(c, t);
                if (v > bestV)
                {
                    bestV = v;
                    bestT = t;
                }
            }
        }

        return CubicMax(bestV, bestT);
    }

    float disc = b * b - 4.0 * a * d;
    if (disc < 0.0)
    {
        return CubicMax(bestV, bestT);
    }

    float s = sqrt(disc);
    float inv2a = 0.5 / a;

    float t0 = (-b - s) * inv2a;
    float t1 = (-b + s) * inv2a;

    // p''(t) = 2*a*t + b
    float dd0 = 2.0 * a * t0 + b;

    // Pick the stationary point that is a local maximum
    float t = (dd0 < 0.0) ? t0 : t1;

    if (t > 0.0 && t < 1.0)
    {
        float v = evalCubic(c, t);
        if (v > bestV)
        {
            bestV = v;
            bestT = t;
        }
    }

    return CubicMax(bestV, bestT);
}

// Handle the linear-derivative degeneracy explicitly;
// otherwise solve the quadratic derivative directly and test both stationary points.

CubicMax cubicMaxOnUnitInterval(vec4 c, float v0, float v1)
{
    const float eps = 1e-6;

    float bestV = v0;
    float bestT = 0.0;

    if (v1 > bestV)
    {
        bestV = v1;
        bestT = 1.0;
    }

    // p'(t) = d + b*t + a*t^2
    float d = c.y;
    float b = 2.0 * c.z;
    float a = 3.0 * c.w;

    // Linear derivative case
    if (abs(a) < eps)
    {
        if (abs(b) >= eps)
        {
            float t = -d / b;
            if (t > 0.0 && t < 1.0)
            {
                float v = evalCubic(c, t);
                if (v > bestV)
                {
                    bestV = v;
                    bestT = t;
                }
            }
        }

        return CubicMax(bestV, bestT);
    }

    // True quadratic derivative case
    float disc = b * b - 4.0 * a * d;
    if (disc < 0.0)
    {
        return CubicMax(bestV, bestT);
    }
        
    float s = sqrt(disc);
    float inv2a = 0.5 / a;

    float t0 = (-b - s) * inv2a;
    if (t0 > 0.0 && t0 < 1.0)
    {
        float v = evalCubic(c, t0);
        if (v > bestV)
        {
            bestV = v;
            bestT = t0;
        }
    }

    float t1 = (-b + s) * inv2a;
    if (t1 > 0.0 && t1 < 1.0)
    {
        float v = evalCubic(c, t1);
        if (v > bestV)
        {
            bestV = v;
            bestT = t1;
        }
    }

    return CubicMax(bestV, bestT);
}

// Treat p'(t) as linear when the quadratic term is negligible.
// Otherwise solve the quadratic derivative and select the stationary point
// with negative second derivative, which is the local maximum

CubicMax cubicMaxOnUnitInterval(vec4 c, float v0, float v1)
{
    const float eps = 1e-6;

    float bestV = v0;
    float bestT = 0.0;

    if (v1 > bestV)
    {
        bestV = v1;
        bestT = 1.0;
    }

    float d = c.y;
    float b = 2.0 * c.z;
    float a = 3.0 * c.w;

    if (abs(a) < eps)
    {
        if (abs(b) >= eps)
        {
            float t = -d / b;
            if (t > 0.0 && t < 1.0)
            {
                float v = evalCubic(c, t);
                if (v > bestV)
                {
                    bestV = v;
                    bestT = t;
                }
            }
        }

        return CubicMax(bestV, bestT);
    }

    float disc = b * b - 4.0 * a * d;
    if (disc <= 0.0)
    {
        return CubicMax(bestV, bestT);
    }

    float s = sqrt(disc);
    float inv2a = 0.5 / a;

    float t0 = (-b - s) * inv2a;
    float t1 = (-b + s) * inv2a;

    // p''(t) = 2*a*t + b
    float dd0 = 2.0 * a * t0 + b;

    // Pick the stationary point that is a local maximum
    float t = (dd0 < 0.0) ? t0 : t1;

    if (t > 0.0 && t < 1.0)
    {
        float v = evalCubic(c, t);
        if (v > bestV)
        {
            bestV = v;
            bestT = t;
        }
    }

    return CubicMax(bestV, bestT);
}

// Find the cubic maximum on [0,1]. Uses a stable quadratic solve for p'(t),
// with the second root from Vieta; t1 = d/q also degenerates to the linear
// root -d/b as a -> 0.

CubicMax cubicMaxOnUnitInterval(vec4 c, float v0, float v1)
{
    const float eps = 1e-6;

    float bestT = 0.0;
    float bestV = v0;

    if (v1 > bestV)
    {
        bestV = v1;
        bestT = 1.0;
    }

    // p'(t) = d + b*t + a*t^2
    float a = 3.0 * c.w;
    float b = 2.0 * c.z;
    float d = c.y;

    float disc = b * b - 4.0 * a * d;
    if (disc >= 0.0)
    {
        float s = sqrt(disc);
        float sb = (b >= 0.0) ? 1.0 : -1.0;
        float q  = -0.5 * (b + sb * s);

        // This root stays meaningful in the linear limit a -> 0: t1 = d/q -> -d/b
        if (abs(q) > eps)
        {
            float t1 = d / q;
            if (t1 > 0.0 && t1 < 1.0)
            {
                float v = evalCubic(c, t1);
                if (v > bestV)
                {
                    bestV = v;
                    bestT = t1;
                }
            }
        }

        // Only valid as a true quadratic root
        if (abs(a) > eps)
        {
            float t0 = q / a;
            if (t0 > 0.0 && t0 < 1.0)
            {
                float v = evalCubic(c, t0);
                if (v > bestV)
                {
                    bestV = v;
                    bestT = t0;
                }
            }
        }
    }

    return CubicMax(bestV, bestT);
}
*/