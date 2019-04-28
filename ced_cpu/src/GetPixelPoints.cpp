#include "GetPixelPoints.hpp"
#include <iostream>
namespace ced
{
    namespace cpu
    {
        void getWhitePixelsCoords(
                std::vector<Point>& _whitePixelsCoord, 
                const std::vector<float>& _red, 
                const std::vector<float>& _green, 
                const std::vector<float>& _blue, 
                const int& height, 
                const int& width)
        {
            for(int i =0; i < height; ++i)
            {
                for(int j =0; j < width; ++j)
                {
                    
                    if((_red[(j + i * width)]+_green[(j + i * width)] + _blue[(j + i * width)]) == 3.0f )
                    {
                        _whitePixelsCoord.push_back(Point(j, i));
                    }
                }
            }
        }
    }
}

