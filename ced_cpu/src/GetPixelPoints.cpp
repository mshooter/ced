#include "GetPixelPoints.hpp"

namespace ced
{
    namespace cpu
    {
        void getWhitePixelsCoords(std::vector<Point>& _whitePixelsCoord, std::vector<float> _imgPix, int width)
        {
            for(unsigned int i = 0; i < _imgPix.size(); ++i)
            {
                int w = i / 3;
                int h = i % 3;
                if( _imgPix[(w + h * width) * 3 + 0] == 1 || 
                    _imgPix[(w + h * width) * 3 + 1] == 1 ||
                    _imgPix[(w + h * width) * 3 + 2] == 1)
                {
                    _whitePixelsCoord.push_back(Point(w,h));
                }
            }
        }
    }
}

