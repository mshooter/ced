#include "Point.hpp"

namespace ced
{
    namespace cpu
    {
        bool Point::operator==(const Point& rhs) const
        {
            return ((x == rhs.x) && (y == rhs.y));
        }
    }
}
