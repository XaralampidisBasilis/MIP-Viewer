#ifndef ADVANCE_CELL_COORDS_AT_BLOCK
#define ADVANCE_CELL_COORDS_AT_BLOCK

ivec3 advanceCellCoordsAtBlock(ivec3 blockCoords, ivec3 entryStep, vec3 entryPosition)
{
    ivec3 entryCoords = positionToCellCoords(entryPosition);
    ivec3 stepCoords = blockCoords * BLOCK_SIZE;

    return entryCoords + (stepCoords - entryCoords) * entryStep;
}

#endif