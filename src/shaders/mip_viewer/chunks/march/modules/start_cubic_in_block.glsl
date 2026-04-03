

// START_CUBIC_IN_BLOCK
if(block.prev_empty)
{
    cubic.values.w = sampleVolume(block.entry_position);

    #if DEBUG_ENABLED == 1

        stats.num_volume_fetches += 1;
    
    #endif
}