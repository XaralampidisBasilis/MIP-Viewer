
// start block
block.exit_distance = ray.start_distance;
block.exit_position = rayDistanceToPosition(block.exit_distance); 

block.entry_distance = block.exit_distance;
block.entry_position = block.exit_position; 

block.coords = positionToBlockCoords(block.exit_position);
