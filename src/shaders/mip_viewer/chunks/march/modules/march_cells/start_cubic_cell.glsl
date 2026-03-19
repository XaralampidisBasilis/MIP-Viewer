

cell.exit_distance = ray.start_distance;
cell.exit_position = rayDistanceToPosition(cell.exit_distance); 

cell.coords = positionToCellCoords(cell.exit_position); 
cubic.values[3] = sampleVolume(cell.exit_position);

#if DEBUG_ENABLED == 1

    stats.num_volume_fetches += 1;
    
#endif

