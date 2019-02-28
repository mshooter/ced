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
                Point(const int x, const int y);
                Point() = default;
                ~Point() = default;
                int getX();
                int getY();
                void setX(int x);
                void setY(int y);
                Point& operator=(const Point&) = default;
            private:
                int m_x; 
                int m_y;
        };
    }
}

#endif //POINT_H_INCLUDED
