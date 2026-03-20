#ifndef ADVANCE_CELL_COORDS_AT_BLOCK
#define ADVANCE_CELL_COORDS_AT_BLOCK

ivec3 advanceCellCoordsAtBlock(ivec3 blockCoords, vec3 entryPosition, ivec3 entryStep)
{
    ivec3 entryCoords = positionToCellCoords(entryPosition);
    ivec3 stepCoords = blockCoords * u_volume.block_size;

    return entryCoords + (stepCoords - entryCoords) * entryStep;
}

#endif