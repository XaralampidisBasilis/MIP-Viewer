
// START_TRACE_IN_CELL
if(block.prev_empty)
{
    // Increment distance
    trace.distance = block.entry_distance;

    // Compute position
    trace.position = block.entry_position; 

    // Update value
    trace.value = sampleVolume(trace.position);

    #if DEBUG_ENABLED == 1

        stats.num_volume_fetches += 1;
        stats.num_traces += 1;
    
    #endif
}