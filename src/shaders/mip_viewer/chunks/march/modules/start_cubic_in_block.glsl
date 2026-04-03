

// START_CUBIC_IN_BLOCK
if(block.prev_empty)
{
    cubic.values.w = sampleVolume(block.entry_position);
    cubic.max_value = cubic.values.w;
    cubic.argmax_point = 0.0;

    #if DEBUG_ENABLED == 1

        stats.num_volume_fetches += 1;
    
    #endif
}