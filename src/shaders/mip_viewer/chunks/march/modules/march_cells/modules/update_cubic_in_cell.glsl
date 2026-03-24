
// UPDATE_CUBIC     
vec3 span_vector = cell.exit_position - cell.entry_position;

cubic.values.x = cubic.values.w;
cubic.values.y = sampleVolume(cell.entry_position + span_vector * (1.0 / 3.0));
cubic.values.z = sampleVolume(cell.entry_position + span_vector * (2.0 / 3.0));
cubic.values.w = sampleVolume(cell.exit_position);

#if DEBUG_ENABLED == 1

    stats.num_volume_fetches += 3;

#endif