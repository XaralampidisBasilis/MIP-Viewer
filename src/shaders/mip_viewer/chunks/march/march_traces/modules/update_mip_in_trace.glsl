
// Compare trace to mip value
if (trace.value > mip.value)
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

