#ifdef STRUCT_BLOCK

block.step_radius    = 0;
block.prev_empty     = false;
block.empty          = false;
block.terminated     = false;
block.coords         = ivec3(0);
block.entry_distance = 0.0;
block.exit_distance  = 0.0;
block.span_distance  = 0.0;
block.entry_position = vec3(0.0);
block.exit_position  = vec3(0.0);
block.entry_step     = ivec3(0);
block.exit_step      = ivec3(0);

#endif // STRUCT_BLOCK
