

debug.variable0 = to_color(turbo(box.entry_position.x / float(u_volume.dimensions.x)));
debug.variable1 = to_color(turbo(box.entry_position.y / float(u_volume.dimensions.y)));
debug.variable2 = to_color(turbo(box.entry_position.z / float(u_volume.dimensions.z)));

// PRINT DEBUG
switch (u_debug.option - 1000)
{ 
    case 0  : fragColor = debug.variable0; break;
    case 1  : fragColor = debug.variable1; break;
    case 2  : fragColor = debug.variable2; break;
    case 3  : fragColor = debug.variable3; break;
    case 4  : fragColor = debug.variable4; break;
    case 5  : fragColor = debug.variable5; break;
    case 6  : fragColor = debug.variable6; break;
    case 7  : fragColor = debug.variable7; break;
    case 8  : fragColor = debug.variable8; break;
    case 9  : fragColor = debug.variable9; break;
}