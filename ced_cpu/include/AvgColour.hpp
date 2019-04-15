#ifndef AVGCOLOUR_H_INCLUDED
#define AVGCOLOUR_H_INCLUDED

#include <vector>
#include "Point.hpp"

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
                        const unsigned int& width);
    }
}


#endif // AVGCOLOUR_H_INCLUDED
