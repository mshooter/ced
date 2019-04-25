#ifndef ASSIGNPIXTOTRI_CUH_INCLUDED
#define ASSIGNPIXTOTRI_CUH_INCLUDED

namespace ced
{
    namespace gpu
    {
        void assignPixToTri(
                std::vector<int>& map_triangleID,
                std::vector<int>& map_pixelID,
                const std::vector<int>& triangleIDs,
                const std::vector<float2>& coordinates, 
                const int& height,
                const int& width
                );
    }
}

#endif // ASSIGNPIXTOTRI_CUH_INCLUDED
