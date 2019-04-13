#ifndef COLOURTRIANGLE_H_INCLUDED
#define COLOURTRIANGLE_H_INCLUDED

#include <vector>
#include "Point.hpp"
#include "TriOrientation.hpp"

namespace ced
{
    namespace cpu
    {
        void findMinMax(    const Point& p0,
                            const Point& p1,
                            const Point& p2,
                            const int& width, 
                            const int& height,
                            int& minx, 
                            int& maxx, 
                            int& miny, 
                            int& maxy);

        void rasterise( const Point& p0,
                        const Point& p1, 
                        const Point& p2,
                        const int& minx, 
                        const int& maxx, 
                        const int& miny, 
                        const int& maxy, 
                        std::vector<Point>& _ptsInsideTri);
    }
}

#endif // COLOURTRIANGLE_H_INCLUDED
