#ifndef FINDMIDPOINTTRI_H_INCLUDED
#define FINDMIDPOINTTRI_H_INCLUDED

#include <vector>
#include "Point.hpp"

namespace ced
{
    namespace cpu
    {
        void findMidpointTri(std::vector<Point> _pts, std::vector<unsigned int> _tris, std::vector<Point>& _midpoints);
    }
}

#endif // FINDMIDPOINTTRI_H_INCLUDED
