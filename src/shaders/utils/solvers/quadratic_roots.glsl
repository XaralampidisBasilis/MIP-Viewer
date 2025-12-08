/* Sources
Numerical Recipes in C: The Art of Scientific Computing, 2nd Edition Section: Chapter 5.6 – Quadratic and Cubic Equations
(https://www.cec.uchile.cl/cinetica/pcordero/MC_libros/NumericalRecipesinC.pdf),
*/

#ifndef QUADRATIC_ROOTS
#define QUADRATIC_ROOTS

#ifndef SSIGN
#include "../math/ssign"
#endif

// Solves the quadratic equation: c[0] + c[1]*x^1 + c[2]*x^2 = 0
// We assume non zero quadratic coefficient
// xd is the fallback root

vec2 quadratic_roots(in vec3 c)
{
    // adjust quadratic coefficients 
    vec2 n = c.xy / c.z;
    n.y /= -2.0;

    // compute quadratic discriminant
    float d = n.y * n.y - n.x;
    float sqrt_d = sqrt(max(0.0, d));
    float xq = n.y + sqrt_d * ssign(n.y);

    // compute quadratic roots via stable formula
    return vec2(xq, n.x / xq);
}

vec2 quadratic_roots(in vec3 c, in float xd)
{
    // adjust quadratic coefficients 
    vec2 n = c.xy / c.z;
    n.y /= -2.0;

    // compute quadratic discriminant
    float d = n.y * n.y - n.x;
    float sqrt_d = sqrt(max(0.0, d));
    float xq = n.y + sqrt_d * ssign(n.y);

    // compute quadratic roots via stable formula
    vec2 x = vec2(xq, n.x / xq);

    // select roots based on determinant
    return (d >= 0.0) ? x : vec2(xd);
}



#endif






