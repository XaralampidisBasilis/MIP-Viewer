
trace.prev_residue = trace.residue;

trace.residue = sample_residue_tricubic(trace.position);

trace.intersected = sign_change(trace.residue, trace.prev_residue);
