#ifndef ASSIGNTRITOPIX_H_INCLUDED
#define ASSIGNTRITOPIX_H_INCLUDED

#include "Point.hpp"
#include <vector>

namespace ced
{
    namespace cpu
    {
        // iterate over every pixel (width, height) check which midpoint is closest to pix -> assign index to that pix
        void assignTriToPix(    const int& height, 
                                const int& width, 
                                const std::vector<Point>& _mpts, 
                                std::vector<unsigned int>& _pixTriIdx);
    }
}

#endif // ASSIGNTRITOPIX_H_INCLUDED
