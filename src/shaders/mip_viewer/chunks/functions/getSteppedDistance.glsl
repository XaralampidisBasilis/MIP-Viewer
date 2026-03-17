#ifndef GET_STEPPED_DISTANCE
#define GET_STEPPED_DISTANCE

float getSteppedDistance(float t, float spacing, float phase)
{
    return spacing * (floor(t / spacing) + phase);
}

#endif
