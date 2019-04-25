#include "AssignPixToTri.cuh"

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
                )
        {
            int amountOfTri = triangleIDs.size()/3;
            int amountOfPix = (width *height);
            for(unsigned int t = 0; t < amountOfTri; ++t)
            {

                int idTri0 = triangleIDs[(t*3+0)]; 
                int idTri1 = triangleIDs[(t*3+1)]; 
                int idTri2 = triangleIDs[(t*3+2)];
                Point v0 = coordinates[idTri0];
                Point v1 = coordinates[idTri1];
                Point v2 = coordinates[idTri2];
                int minx = std::min({v0.x, v1.x, v2.x});
                int miny = std::min({v0.y, v1.y, v2.y});
                int maxx = std::max({v0.x, v1.x, v2.x});
                int maxy = std::max({v0.y, v1.y, v2.y});
                // serialized
                for(int x = miny; x < maxy; ++x)
                {
                    for(int y = minx; y < maxx; ++y)
                    {
                        Point p = {y, x};   
                        if(isPixInTri(v0,v1,v2,p))
                        {
                            trianglePixels.insert({t, (y+x*width)});
                        }
                    }
                }
            }
        }
    }
}
