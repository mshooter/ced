// @file : SortPoints.hpp
#ifndef SORTPOINTS_H_INCLUDED
#define SORTPOINTS_H_INCLUDED

#include <vector>

namespace ced
{
    namespace cpu
    {
        template <typename T>
        int partition(std::vector<T>& _points, int lo, int hi);

        template <typename T>
        void quickSort(std::vector<T>& _pts, int lo, int hi);
        
        #include "SortPoints.inl"
    }
}

#endif // SORTPOINTS_H_INCLUDED
