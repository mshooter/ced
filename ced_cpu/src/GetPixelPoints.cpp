#include "GetPixelPoints.hpp"

namespace ced
{
    namespace cpu
    {
        void getWhitePixelsCoords(
                std::vector<Point>& _whitePixelsCoord, 
                const std::vector<float>& edgePixData, 
                const int& height, 
                const int& width)
        {
            for(int i =0; i < height; ++i)
            {
                for(int j =0; j < width; ++j)
                {
                    if((edgePixData[(j + i * width) * 3 + 0]+edgePixData[(j + i * width) * 3 + 1] + edgePixData[(j + i * width) * 3 + 2])/3.0f > 0 )
                    {
                        _whitePixelsCoord.push_back(Point(j, i));
                    }
                }
            }
        }
    }
}

