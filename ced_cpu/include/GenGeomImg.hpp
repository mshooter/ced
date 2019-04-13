#ifndef GENGEOMIMG_H_INCLUDED
#define GENGEOMIMG_H_INCLUDED

#include <vector>
#include "Point.hpp"

namespace ced
{
    namespace cpu
    {
        void genGeomImg(    std::vector<float>& _pixData,
                            std::vector<unsigned int> _tri, 
                            std::vector<Point> _pts,
                            const int& width,
                            const int& height);
    }
}

#endif // GENGEOMIMG_H_INCLUDED
