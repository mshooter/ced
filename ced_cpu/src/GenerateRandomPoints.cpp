#include "GenerateRandomPoints.hpp"
#include <cstdlib>

namespace ced
{
    namespace cpu
    {
        void generateRandomPoints(
                std::vector<Point>& coordinateList,
                const int& _amountOfPoints, 
                const int& height, 
                const int& width)
        {

            for(int i = 0 ; i < _amountOfPoints; ++i)
            {
                coordinateList.push_back(Point(rand()%width, rand()%height));
            }
        }
    }
}
