
// UPDATE_MIP_IN_TRACE
mip.update = shouldUpdateMip(mip.value, trace.value);

if (mip.update)
{
    mip.distance = trace.distance;
    mip.value = trace.value;

    #if DEBUG_ENABLED == 1

        stats.num_mips += 1;

    #endif

}

