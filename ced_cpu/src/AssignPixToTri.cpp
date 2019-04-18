#include "AssignPixToTri.hpp"
#include "IsPixInTri.hpp"


namespace ced
{
    namespace cpu
    {
        void assignPixToTri(
                std::multimap<unsigned int, unsigned int>& trianglePixels, 
                const std::vector<unsigned int>& triangleIDs,
                const std::vector<Point>& coordinates, 
                const unsigned int& height,
                const unsigned int& width
                )
        {
            unsigned int amountOfTri = triangleIDs.size()/3;
            unsigned int amountOfPix = (width *height);
            for(unsigned int t = 0; t < amountOfTri; ++t)
            {

                unsigned int idTri0 = triangleIDs[(t*3+0)]; 
                unsigned int idTri1 = triangleIDs[(t*3+1)]; 
                unsigned int idTri2 = triangleIDs[(t*3+2)];
                Point v0 = coordinates[idTri0];
                Point v1 = coordinates[idTri1];
                Point v2 = coordinates[idTri2];
                for(unsigned int i = 0; i < amountOfPix ; ++i)
                {
                        unsigned int h = i / width;
                        unsigned int w = i % width;
                        Point p = {w, h};   
                        // if point is in the triangle, insert the point ID 
                        if(isPixInTri(v0, v1, v2, p))
                        {
                            trianglePixels.insert({t, i});
                        }
                }
            }
        }
    }
}
