#ifndef RAY_DISTANCE_TERMINATED
#define RAY_DISTANCE_TERMINATED

bool rayDistanceTerminated(float a, float b, float t)
{
    return t < a || t > b
}

#endif
