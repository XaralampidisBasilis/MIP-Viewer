#ifndef RESHAPE
#define RESHAPE

// pos in [0, dims]
int reshape_2d_to_1d(in ivec2 pos2, in ivec2 dims) 
{
    return pos2.x + pos2.y * dims.x;
}

// pos in [0, dims]
ivec2 reshape_1d_to_2d(in int pos1, in ivec2 dims) 
{
    ivec2 pos2;
    pos2.y = pos1 / dims.x;
    pos2.x = pos1 % dims.x;
    return pos2;
}

// pos in 0, ..., dims - 1
int reshape_3d_to_1d(in ivec3 pos3, in ivec3 dims) 
{
    return pos3.z * dims.x * dims.y + pos3.y * dims.x + pos3.x;
}

// pos in 0, ..., dims - 1
ivec3 reshape_1d_to_3d(in int pos1, in ivec3 dims) 
{
    ivec3 pos3;
    int dims_xy = dims.x * dims.y;

    pos3.z = pos1 / dims_xy;
    int pos_xy = pos1 % dims_xy;
    pos3.y = pos_xy / dims.x;
    pos3.x = pos_xy % dims.x;
    return pos3;
}

// pos in 0, ..., dims - 1
ivec2 reshape_3d_to_2d(in ivec3 pos3, in ivec3 dims) 
{
    ivec2 pos2;
    pos2.x = pos3.x;
    pos2.y = pos3.z * dims.y + pos3.y;
    return pos2;
}

// pos in 0, ..., dims - 1
ivec3 reshape_2d_to_3d(in ivec2 pos2, in ivec3 dims) 
{
    ivec3 pos3;
    pos3.x = pos2.x;
    pos3.y = pos2.y % dims.y;
    pos3.z = pos2.y / dims.y;
    return pos3;
}

#endif // RESHAPE