#include "GetPixelPoints.hpp"

namespace ced
{
    namespace cpu
    {
        void getWhitePixelsCoords(std::vector<Point>& _whitePixelsCoord, std::vector<Point>& _allPixCoord, std::vector<float> _imgPix, int height, int width)
        {
            for(int i=0; i < height; ++i)
            {
                for(int j=0; j < width; ++j)
                {
                    _allPixCoord.push_back(Point(j,i));
                    const unsigned int currentIndex = (j + i * width) * 3;
                    if(_imgPix[currentIndex] == 1)
                    {
                        _whitePixelsCoord.push_back(Point(j,i));
                    }
                }
            }
        }
    }
}

