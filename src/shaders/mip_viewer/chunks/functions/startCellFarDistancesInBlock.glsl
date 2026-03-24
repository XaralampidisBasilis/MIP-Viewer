#ifndef START_CELL_FAR_DISTANCES_IN_BLOCK
#define START_CELL_FAR_DISTANCES_IN_BLOCK

#ifndef START_CELL_COORDS_IN_BLOCK
#include "./startCellCoordsInBlock"
#endif
#ifndef CELL_COORDS_TO_FAR_DISTANCES
#include "./cellCoordsToFarDistances"
#endif

vec3 startCellFarDistancesInBlock(ivec3 blockCoords, ivec3 entryStep, vec3 entryPosition)
{
    ivec3 cellCoords = startCellCoordsInBlock(blockCoords, entryStep, entryPosition);
    vec3 tFar = cellCoordsToFarDistances(cellCoords);
    return tFar;
}

#endif