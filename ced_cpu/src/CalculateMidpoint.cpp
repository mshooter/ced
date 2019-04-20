#include "CalculateMidpoint.hpp"

namespace ced
{
    namespace cpu
    {
        Point calculateMidpoint(
            const Point& a,
            const Point& b,
            const Point& c)
        {
            float x = (a.x + b.x + c.x)/3.0f;
            float y = (a.y + b.y + c.y)/3.0f;

            return Point(x, y);
        }
    }
}
