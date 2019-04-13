#ifndef TRIORIENTATION_H_INCLUDED
#define TRIORIENTATION_H_INCLUDED

#include "Point.hpp"

namespace ced
{
    namespace cpu
    {
        //  ------------------------------------------------
        /// @build :  are the three points counter clockwise ? 
        /// @param[_in] _p1 : first point 
        /// @param[_in] _p2 : second point 
        /// @param[_in] _p3 : third point 
        //  ------------------------------------------------
        template <typename T>
        T isCCW(T _p1, T _p2, T _p3);
        #include "TriOrientation.inl"
    }
}
#endif // TRIORIENTATION_H_INCLUDED 
