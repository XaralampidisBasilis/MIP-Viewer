#ifndef START_CELL_COORDS_IN_BLOCK
#define START_CELL_COORDS_IN_BLOCK

#ifndef POSITION_TO_CELL_COORDS
#include "./positionToCellCoords"
#endif

ivec3 startCellCoordsInBlock(ivec3 blockCoords, ivec3 entryStep, vec3 entryPosition)
{
    ivec3 entryCoords = positionToCellCoords(entryPosition);
    
    #if BLOCK_SIZE == 1
    ivec3 stepCoords = blockCoords;
    #else
    ivec3 stepCoords = blockCoords * BLOCK_SIZE;
    #endif

    return entryCoords + (stepCoords - entryCoords) * entryStep;
}

#endif