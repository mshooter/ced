#ifndef DISTANCE2P_H_INCLUDED
#define DISTANCE2P_H_INCLUDED

#include <cmath>

namespace ced
{
    namespace cpu
    {
        //  ------------------------------------
        /// @build : distance between two points
        /// @param[_in] : first point
        /// @param[_in] : second point
        //  ------------------------------------
        template <typename T, typename U>
        T distance2P(U _p1, U _p2);
        //  ------------------------------------
        /// @build : template implementation of the distance function
        //  ------------------------------------
        #include "Distance2P.inl"
    }
}

#endif // DISTANCE2P_H_INCLUDED
