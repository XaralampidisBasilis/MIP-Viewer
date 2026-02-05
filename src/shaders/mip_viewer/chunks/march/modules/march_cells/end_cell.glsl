
if (cell.intersected) 
{
    // Compute the hit details for the intersected cell
    #include "./compute_hit"

    hit.undefined = false;

    // Mark as escaped if the hit lies outside the ray's valid range
    hit.escaped = (hit.distance < ray.start_distance || hit.distance > ray.end_distance);

    // Keep the hit only if it’s within the valid distance range
    hit.discarded = hit.escaped;

    // Clamp hit distance
    hit.distance = clamp(hit.distance, ray.start_distance, ray.end_distance);
    hit.position = camera.position + ray.direction * hit.distance;
}
else
{
    // Mark hit as undefined when the cell has neither an intersection nor a termination point
    hit.undefined = !cell.terminated;
    hit.discarded = true;
    hit.escaped = false;

    // Set discarded distance the ray end
    hit.distance = ray.end_distance;
    hit.position = camera.position + ray.direction * hit.distance;
}
