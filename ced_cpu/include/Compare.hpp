#ifndef COMPARE_H_INCLUDED
#define COMPARE_H_INCLUDED

#include "Point.hpp"
#include <utility>
namespace ced
{
    namespace cpu
    {
        //  -------------------------------------------------------
        /// @build: compares the x coordinate
        //  -------------------------------------------------------
        bool compareX(const Point& lhs, const Point& rhs);
        //  -------------------------------------------------------
        /// @build: compares the y coordinate
        //  -------------------------------------------------------
        bool compareY(const Point& lhs, const Point& rhs);
        
        bool compareXofPair(const std::pair<int, int>& lhs, const std::pair<int, int>& rhs);
    }
}

#endif // COMPARE_H_INCLUDED
