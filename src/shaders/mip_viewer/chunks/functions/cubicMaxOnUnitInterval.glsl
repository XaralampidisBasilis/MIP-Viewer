#ifndef CUBIC_MAX_ON_UNIT_INTERVAL
#define CUBIC_MAX_ON_UNIT_INTERVAL

struct CubicMax {
    float v;
    float t;
};

// Find the maximum value of a cubic on [0, 1] from power-basis coefficients
CubicMax cubicMaxOnUnitInterval(vec4 c)
{
    // Start with endpoint t = 0
    float bestT = 0.0;
    float bestV = c.x;

    // Test endpoint t = 1
    float v1 = c.x + c.y + c.z + c.w;
    if (v1 > bestV)
    {
        bestV = v1;
        bestT = 1.0;
    }

    // Derivative: c.y + 2*c.z*t + 3*c.w*t^2
    float a = c.w * 3.0;
    float b = c.z * 2.0;
    float d = c.y;

    float disc = b * b - 4.0 * a * d;
    if (disc >= 0.0)
    {
        float s = sqrt(disc);
        float q = -0.5 * (b + ssign(b) * s);

        // t1 also becomes the linear root when a == 0
        if (q != 0.0)
        {
            float t1 = d / q;
            if (t1 > 0.0 && t1 < 1.0)
            {
                // Evaluate cubic at t1 using Horner form
                float v = ((c.w * t1 + c.z) * t1 + c.y) * t1 + c.x;
                if (v > bestV)
                {
                    bestV = v;
                    bestT = t1;
                }
            }
        }

        // Only valid for the true quadratic case
        if (a != 0.0)
        {
            float t0 = q / a;
            if (t0 > 0.0 && t0 < 1.0)
            {
                // Evaluate cubic at t0 using Horner 
                float v = ((c.w * t0 + c.z) * t0 + c.y) * t0 + c.x;
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

CubicMax cubicMaxOnUnitInterval(vec4 c, float v0, float v1)
{
    // Start with endpoint t = 0
    float bestT = 0.0;
    float bestV = v0;

    // Test endpoint t = 1
    if (v1 > bestV)
    {
        bestV = v1;
        bestT = 1.0;
    }

    // Derivative: c.y + 2*c.z*t + 3*c.w*t^2
    float a = c.w * 3.0;
    float b = c.z * 2.0;
    float d = c.y;

    float disc = b * b - 4.0 * a * d;
    if (disc >= 0.0)
    {
        float s = sqrt(disc);
        float q = -0.5 * (b + ssign(b) * s);

        // t1 also becomes the linear root when a == 0
        if (q != 0.0)
        {
            float t1 = d / q;
            if (t1 > 0.0 && t1 < 1.0)
            {
                // Evaluate cubic at t1 using Horner form
                float v = ((c.w * t1 + c.z) * t1 + c.y) * t1 + c.x;
                if (v > bestV)
                {
                    bestV = v;
                    bestT = t1;
                }
            }
        }

        // Only valid for the true quadratic case
        if (a != 0.0)
        {
            float t0 = q / a;
            if (t0 > 0.0 && t0 < 1.0)
            {
                // Evaluate cubic at t0 using Horner 
                float v = ((c.w * t0 + c.z) * t0 + c.y) * t0 + c.x;
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

/*
CubicMax cubicMaxOnUnitInterval(vec4 c)
{
    // quadratic coefficients
    vec3 q = vec3(c.y, 2.0 * c.z, 3.0 * c.w);

    // derivative roots
    vec2 t_ext = clamp(quadratic_roots(q), 0.0, 1.0);

    // evaluate extrema
    vec2 v_ext = eval_poly(c, t_ext);

    // evaluate endpoints directly
    float v0 = c.x;
    float v1 = c.x + c.y + c.z + c.w;

    // pack for argmax
    vec4 v = vec4(v0, v_ext.x, v_ext.y, v1);
    vec4 t = vec4(0.0, t_ext.x, t_ext.y, 1.0);

    int i = argmax(v);
    return CubicMax(v[i], t[i]);
}
*/

#endif