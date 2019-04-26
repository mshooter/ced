#include "AssignPixToTri.hpp"
#include "IsPixInTri.hpp"
#include <algorithm>
#include <iostream>
namespace ced
{
    namespace cpu
    {
        void assignPixToTri(
                std::multimap<unsigned int, unsigned int>& pixIDdepTri, 
                const std::vector<unsigned int>& triangleIDs,
                const std::vector<Point>& coordinates, 
                const unsigned int& width
                )
        {
            unsigned int amountOfTri = triangleIDs.size()/3;
            for(unsigned int t = 0; t < amountOfTri; ++t)
            {

                unsigned int idTri0 = triangleIDs[(t*3+0)]; 
                unsigned int idTri1 = triangleIDs[(t*3+1)]; 
                unsigned int idTri2 = triangleIDs[(t*3+2)];
                Point v0 = coordinates[idTri0];
                Point v1 = coordinates[idTri1];
                Point v2 = coordinates[idTri2];
                int minx = std::min({v0.x, v1.x, v2.x});
                int miny = std::min({v0.y, v1.y, v2.y});
                int maxx = std::max({v0.x, v1.x, v2.x});
                int maxy = std::max({v0.y, v1.y, v2.y});
                for(int x = miny; x < maxy; ++x)
                {
                    for(int y = minx; y < maxx; ++y)
                    {
                        Point p = {y, x};   
                        if(isPixInTri(v0,v1,v2,p))
                        {
                            pixIDdepTri.insert({t,(y+x*width)});
                        }
                    }
                }
            }
        }
    }
}
