
#ifndef SHAPE_OPERATOR
#define SHAPE_OPERATOR

mat3 shape_operator(in vec3 gradient, in mat3 hessian)
{
    vec3 normal = normalize(gradient);
    mat3 P = mat3(1.0) - outerProduct(normal, normal);
    mat3 S = (P * hessian * P) / length(gradient);

    return S;
}

#endif