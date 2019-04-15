#ifndef ASSIGNTRITOPIX_H_INCLUDED
#define ASSIGNTRITOPIX_H_INCLUDED

#include <vector>
#include "Point.hpp"

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
                );
        float area(Point a, Point b, Point c);
        bool isInside(Point a, Point b, Point c, Point p);
    }
}

#endif // ASSIGNTRITOPIX_H_INCLUDED
