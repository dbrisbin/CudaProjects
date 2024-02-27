/// @file types/types.h
/// @brief Types used in Bezier curve tessellation and Quadtree construction.

#ifndef CHAPTER_21_CONSTANTS_TYPES_H
#define CHAPTER_21_CONSTANTS_TYPES_H

#include "constants.h"

struct BezierLineFixedSize
{
    float2 CP[3U];
    float2 vertex_pos[MAX_TESS_POINTS];
    int num_vertices;
};

struct BezierLineDynamic
{
    float2 CP[3U];
    float2* vertex_pos;
    int num_vertices;
};

#endif  // CHAPTER_21_CONSTANTS_TYPES_H