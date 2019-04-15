#ifndef ASSIGNCOLTOPIX_H_INCLUDED
#define ASSIGNCOLTOPIX_H_INCLUDED

#include <vector>
#include "Point.hpp"


namespace ced
{
    namespace cpu
    {
        void assignColToPix(
                std::vector<float>& imgData,
                const unsigned int& amountOfTriangles, 
                const std::vector<unsigned int> triangleIDs, 
                const std::vector<Point> triPixelIdx,
                const unsigned int& height,
                const unsigned int& width);
    }
}

#endif // ASSIGNCOLTOPIX_H_INCLUDED
