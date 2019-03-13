// @Point.hpp
#ifndef POINT_H_INCLUDED
#define POINT_H_INCLUDED

namespace ced
{
    namespace cpu
    {
        class Point
        {
            public:
                Point(const int _x, const int _y);
                Point() = default;
                ~Point() = default;
                Point& operator=(const Point&) = default;
                int x;
                int y;
        };
    }
}

#endif //POINT_H_INCLUDED
