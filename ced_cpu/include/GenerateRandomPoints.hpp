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
                const std::vector<Point>& whites,
                const unsigned int& _amountOfPoints, 
                const unsigned int& height, 
                const unsigned int& width); 
    }
}

#endif // GENERATERANDOMPOINTS_H_INCLUDED
