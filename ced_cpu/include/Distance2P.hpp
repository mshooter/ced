#ifndef DISTANCE2P_H_INCLUDED
#define DISTANCE2P_H_INCLUDED

#include <cmath>
#include <Point.hpp>

namespace ced
{
    namespace cpu
    {
        //  ------------------------------------
        /// @build : distance between two points
        /// @param[_in] : first point
        /// @param[_in] : second point
        //  ------------------------------------
        template <typename T>
        T distance2P(Point _p1, Point _p2);
        //  ------------------------------------
        /// @build : template implementation of the distance function
        //  ------------------------------------
        #include "Distance2P.inl"
    }
}

#endif // DISTANCE2P_H_INCLUDED
