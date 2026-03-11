
// set value
mip.value = trace.value;

// set distance
mip.distance = trace.distance;

// update stats
#if DEBUG_ENABLED == 1

    stats.num_mips += 1;

#endif
