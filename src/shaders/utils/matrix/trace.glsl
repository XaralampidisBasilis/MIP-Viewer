#ifndef TRACE
#define TRACE

float trace(mat2 M) 
{
    return M[0][0] + M[1][1];
}

float trace(mat3 M) 
{
    return M[0][0] + M[1][1] + M[2][2];
}

float trace(mat4 M) 
{
    return M[0][0] + M[1][1] + M[2][2] + M[3][3];
}

#endif
