// COMPUTE DEBUG 

// intersected
vec4 debug_trace_intersected = to_color(trace.intersected);

// terminated 
vec4 debug_trace_terminated = to_color(trace.terminated);

// distance
vec4 debug_trace_distance = to_color(map(box.min_entry_distance, box.max_exit_distance, trace.distance));

// position
vec4 debug_trace_position = to_color(map(box.min_position, box.max_position, trace.position));

// error
vec4 debug_trace_residue = to_color(mmix(COLOR.BLUE, COLOR.BLACK, COLOR.RED, map(-1.0, 1.0, trace.residue / MILLI_TOLERANCE)));

// Span residue
vec4 debug_trace_span_residue = to_color(mmix(COLOR.BLUE, COLOR.BLACK, COLOR.RED, map(-1.0, 1.0, (trace.residue - trace.prev_residue) / MILLI_TOLERANCE)));

// PRINT DEBUG
switch (u_debug.option - 300)
{ 
    case  1: fragColor = debug_trace_intersected;     break;
    case  2: fragColor = debug_trace_terminated;      break;
    case  3: fragColor = debug_trace_distance;        break;
    case  4: fragColor = debug_trace_position;        break;
    case  5: fragColor = debug_trace_residue;         break;
    case  6: fragColor = debug_trace_span_residue;    break;
}