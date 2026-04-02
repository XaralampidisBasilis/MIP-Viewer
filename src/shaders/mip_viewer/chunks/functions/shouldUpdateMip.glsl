#ifndef SHOULD_UPDATE_MIP
#define SHOULD_UPDATE_MIP

bool shouldUpdateMip(float mipValue, float newValue)
{
    return mipValue < newValue;
}

#endif
