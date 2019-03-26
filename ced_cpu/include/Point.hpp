// @Point.hpp
#ifndef POINT_H_INCLUDED
#define POINT_H_INCLUDED

#include <utility>

namespace ced
{
    namespace cpu
    {
        struct Point
        {
            //  ----------------------------------------------------------
            /// @build: Point constructor
            /// @param[_in] : coordinate x 
            /// @param[_in] : coordinate y 
            //  ----------------------------------------------------------
            template <typename T>
            Point(const T _x, const T _y);
            //  ----------------------------------------------------------
            /// @build: default constructor
            //  ----------------------------------------------------------
            Point() = default;
            //  ----------------------------------------------------------
            /// @build: assign operator 
            //  ----------------------------------------------------------
            Point& operator=(const Point&) = default;
            //  ----------------------------------------------------------
            /// @build: compare operator 
            //  ----------------------------------------------------------
            bool operator==(const Point& rhs) const;
            //  ----------------------------------------------------------
            /// @build: add operator 
            //  ----------------------------------------------------------
            Point operator+(const Point& rhs) const;
            //  ----------------------------------------------------------
            /// @build: substract operator 
            //  ----------------------------------------------------------
            Point operator-(const Point& rhs) const;
            //  ----------------------------------------------------------
            /// @build: multiply operator 
            //  ----------------------------------------------------------
            Point operator*(const Point& rhs) const;
            //  ----------------------------------------------------------
            /// @build: divide operator 
            //  ----------------------------------------------------------
            Point operator/(const float scalar) const;
            //  ----------------------------------------------------------
            /// @build: x coordinate 
            //  ----------------------------------------------------------
            float x;
            //  ----------------------------------------------------------
            /// @build: y coordinate 
            //  ----------------------------------------------------------
            float y;
        };
        //  ----------------------------------------------------------
        /// @build: template implementation
        //  ----------------------------------------------------------
        #include "Point.inl"
    }
}



#endif //POINT_H_INCLUDED
