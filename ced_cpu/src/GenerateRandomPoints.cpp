#include "GenerateRandomPoints.hpp"
#include <cstdlib>

namespace ced
{
    namespace cpu
    {
        void generateRandomPoints(
                std::vector<Point>& coordinateList,
                const std::vector<Point>& whites,
                const unsigned int& _amountOfPoints, 
                const unsigned int& height, 
                const unsigned int& width)
        {

            for(unsigned int i = 0 ; i < _amountOfPoints; ++i)
            {
                coordinateList.push_back(Point(rand()%width, rand()%height));
            }
        }
    }
}
