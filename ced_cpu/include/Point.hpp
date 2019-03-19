// @Point.hpp
#ifndef POINT_H_INCLUDED
#define POINT_H_INCLUDED

#include <utility>

namespace ced
{
    namespace cpu
    {
        class Point
        {
            public:
                template <typename T>
                Point(const T _x, const T _y);
                Point() = default;
                ~Point() = default;
                Point& operator=(const Point&) = default;
                int x;
                int y;
        };
        #include "Point.inl"
    }
}



#endif //POINT_H_INCLUDED
