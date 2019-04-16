#include "AssignPixToTri.hpp"
#include "IsPixInTri.hpp"

int area(int x1, int y1, int x2, int y2, int x3, int y3)
{
   return abs((x1*(y2-y3) + x2*(y3-y1)+ x3*(y1-y2))/2);
}
  
/* A function to check whether point P(x, y) lies inside the triangle  */
bool isInside(int x1, int y1, int x2, int y2, int x3, int y3, int x, int y)
{  
   /* Calculate area of triangle ABC */
   float A = area (x1, y1, x2, y2, x3, y3);
  
   /* Calculate area of triangle PBC */ 
   float A1 = area (x, y, x2, y2, x3, y3);
  
   /* Calculate area of triangle PAC */ 
   float A2 = area (x1, y1, x, y, x3, y3);
  
   /* Calculate area of triangle PAB */  
   float A3 = area (x1, y1, x2, y2, x, y);
    
   /* Check if sum of A1, A2 and A3 is same as A */
   return (A == A1 + A2 + A3);
}


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
                unsigned int maxx = std::max(std::max(v0.x,v1.x), v2.x);
                unsigned int maxy = std::max(std::max(v0.y,v1.y), v2.y);
                unsigned int minx = std::min(std::min(v0.x,v1.x), v2.x);
                unsigned int miny = std::min(std::min(v0.y,v1.y), v2.y);
                for(unsigned int i = 0; i < amountOfPix ; ++i)
                {
                        unsigned int h = i / width;
                        unsigned int w = i % width;
                        Point p = {w, h};   
                        // if point is in the triangle, insert the point ID 
                        if(isInside(v0.x, v0.y, v1.x, v1.y, v2.x, v2.y, p.x, p.y))
                        {
                            trianglePixels.insert({t, i});
                        }
                }
            }
        }
    }
}
