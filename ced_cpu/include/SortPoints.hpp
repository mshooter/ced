// @file : SortPoints.hpp
#ifndef SORTPOINTS_H_INCLUDED
#define SORTPOINTS_H_INCLUDED

#include <vector>
#include "Point.hpp"

namespace ced
{
    namespace cpu
    {
        int partition(std::vector<Point>& _points, int lo, int hi);
    }
}

#endif // SORTPOINTS_H_INCLUDED
