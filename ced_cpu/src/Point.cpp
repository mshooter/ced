#include "Point.hpp"

namespace ced
{
    namespace cpu
    {
        bool Point::operator==(const Point& rhs) const
        {
            return ((x == rhs.x) && (y == rhs.y));
        }
        //  ----------------------------------------------------------
        Point Point::operator+(const Point& rhs) const
        {
            Point p;
            p.x = x + rhs.x;
            p.y = y + rhs.y;
            return p;
        }
        //  ----------------------------------------------------------
        Point Point::operator-(const Point& rhs) const
        {
            Point p;
            p.x = x - rhs.x;
            p.y = y - rhs.y;
            return p;
        }
        //  ----------------------------------------------------------
        Point Point::operator*(const Point& rhs) const
        {
            Point p;
            p.x = x * rhs.x;
            p.y = y * rhs.y;
            return p;
        }
        //  ----------------------------------------------------------
        Point Point::operator/(const float scalar) const
        {
            Point p;
            p.x = x / scalar;
            p.y = y / scalar;
            return p;
        }
    }
}
