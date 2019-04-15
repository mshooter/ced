#include "AssignTriToPix.hpp"

namespace ced
{
    namespace cpu
    {
        void assignTriToPix(
                std::vector<unsigned int>& triangleIDs,
                std::vector<Point>& pixelInTriangle, 
                const std::vector<unsigned int>& _triangles,
                const std::vector<Point>& coordinates, 
                const unsigned int& height,
                const unsigned int& width
                )
        {
            for(unsigned int t = 0; t < _triangles.size(); t+=3)
            {
                Point p0 = coordinates[_triangles[t+0]];
                Point p1 = coordinates[_triangles[t+1]];
                Point p2 = coordinates[_triangles[t+2]];
                for(unsigned int i =0; i < height; ++i)
                {
                    for(unsigned int w = 0; w < width; ++w)
                    {
                        Point p = {w,i};
                        if(isInside(p0, p1, p2, p))
                        {
                            pixelInTriangle.push_back(p);
                            triangleIDs.push_back(t/3);
                        }
                    }

                }
            }
        }

        float area(Point a, Point b, Point c)
        {
            return abs((a.x*(b.y-c.y) + b.x*(c.y-a.y)+ c.x*(a.y-b.y))/2.0f);
        }

        bool isInside(Point a, Point b, Point c, Point p)
        {
            /* Calculate area of triangle ABC */
            float A = area (a, b, c);

            /* Calculate area of triangle PBC */
            float A1 = area (p, b, c);

            /* Calculate area of triangle PAC */
            float A2 = area (p, a, c);

            /* Calculate area of triangle PAB */
            float A3 = area (a, b, p);

            /* Check if sum of A1, A2 and A3 is same as A */
            return (A == A1 + A2 + A3);
        }
    }
}
