// @file : SortPoints.hpp
#ifndef SORTPOINTS_H_INCLUDED
#define SORTPOINTS_H_INCLUDED

#include <vector>
#include "Distance2P.hpp"

namespace ced
{
    namespace cpu
    {
        //  ----------------------------------------------------------
        /// @build : part the data to elements lower than piv point 
        /// and higher than piv point 
        /// @param[_in] : low range
        /// @param[_in] : high range
        //  ----------------------------------------------------------
        template <typename T>
        int partition(std::vector<T>& _points, int lo, int hi);
        //  ----------------------------------------------------------
        /// @build : normal quickSort
        /// @param[_in] : low range
        /// @param[_in] : high range
        //  ----------------------------------------------------------
        template <typename T>
        void quickSort(std::vector<T>& _pts, int lo, int hi);
        //  ----------------------------------------------------------
        /// @build : partition Distance2P
        /// @param[_in] : points vector
        /// @param[_in] : low range
        /// @param[_in] : high range
        //  ----------------------------------------------------------
        template <typename T, typename U>
        int partitionDist(std::vector<T>& _ids, std::vector<U> _points, U cc, int lo, int hi);
        //  ----------------------------------------------------------
        /// @build : quickSort Distance2P
        /// @param[_in] : ids vector
        /// @param[_in] : points vector
        /// @param[_in] : low range
        /// @param[_in] : high range
        //  ----------------------------------------------------------
        template <typename T, typename U>
        void quickSortDist(std::vector<T>& _ids, std::vector<U> _points, U cc, int lo, int hi);
        //  ----------------------------------------------------------
        /// @build: implementation of sorting points template 
        //  ----------------------------------------------------------
        #include "SortPoints.inl"
    }
}

#endif // SORTPOINTS_H_INCLUDED
