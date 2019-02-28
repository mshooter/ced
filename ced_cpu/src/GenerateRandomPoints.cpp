#include "GenerateRandomPoints.hpp"
#include <cstdlib>

namespace ced
{
    namespace cpu
    {
        std::vector<Point> generateRandomPoints(const unsigned int _amountOfPoints, const unsigned int height, const unsigned int width)
        {
            std::vector<Point> coordinateList(_amountOfPoints);
            for(unsigned int i=0; i < _amountOfPoints; ++i)
            {     
                coordinateList[i] = Point(rand()%width, rand()%height);
            } 
            return coordinateList;
        }
    }
}
