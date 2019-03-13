#include "Point.hpp"
#include <utility>

namespace ced
{
    namespace cpu
    {
        Point::Point(const int _x, const int _y)
        {
            x = std::move(_x);
            y = std::move(_y);
        }
    }
}
