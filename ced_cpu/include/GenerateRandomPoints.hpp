// @file: GenerateRandomPoints.hpp
#ifndef GENERATERANDOMPOINTS_H_INCLUDED
#define GENERATERANDOMPOINTS_H_INCLUDED

#include "Point.hpp"
#include <vector>

namespace ced
{
    namespace cpu
    {
        void generateRandomPoints(   
                std::vector<Point>& coordinateList,
                const int& _amountOfPoints, 
                const int& height, 
                const int& width); 
    }
}

#endif // GENERATERANDOMPOINTS_H_INCLUDED
