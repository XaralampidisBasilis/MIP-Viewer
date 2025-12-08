
// compute the intensity samples inside the cell from the intensity map texture
cubic.residuals[0] = cubic.residuals[3];

#pragma unroll
for (int i = 1; i < 4; i++) 
{
    vec3 position = mix(cell.entry_position, cell.exit_position, sampling_points[i]);

    cubic.residuals[i] = sample_residue_trilinear(position);
}
