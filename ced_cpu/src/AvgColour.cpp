#include "AvgColour.hpp"

namespace ced
{
    namespace cpu
    {
        void avgColour( std::vector<float>& _imgData, 
                        const std::vector<Point>& pixelIdxTri, 
                        float& r,
                        float& g, 
                        float& b,
                        const unsigned int& height,
                        const unsigned int& width)
        {
            for(auto p : pixelIdxTri)
            {
                r += _imgData[(p.x + p.y * width) * 3 + 0];
                g += _imgData[(p.x + p.y * width) * 3 + 1];  
                b += _imgData[(p.x + p.y * width) * 3 + 2];

            }
            r /= static_cast<float>(pixelIdxTri.size());
            g /= static_cast<float>(pixelIdxTri.size());
            b /= static_cast<float>(pixelIdxTri.size());
        }
    }
}
