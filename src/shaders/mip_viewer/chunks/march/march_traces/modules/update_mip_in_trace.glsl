
// Compare trace to mip value
mip.update = trace.value > mip.value;

if (mip.update)
{
    // Update value
    mip.value = trace.value;

    // Update distance
    mip.distance = trace.distance;

    // update stats
    #if DEBUG_ENABLED == 1

        stats.num_mips += 1;

    #endif

}

