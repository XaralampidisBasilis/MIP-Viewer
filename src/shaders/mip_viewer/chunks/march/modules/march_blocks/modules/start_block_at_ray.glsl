
// START_BLOCK
block.coords = positionToBlockCoords(ray.start_position);
block.exit_distance = ray.start_distance;
block.exit_position = ray.start_position; 
block.exit_step = ivec3(0);
block.empty = false;
