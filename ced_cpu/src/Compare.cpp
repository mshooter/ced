#include "Compare.hpp"

namespace ced
{
    namespace cpu
    {
        bool compareX(const Point& lhs, const Point& rhs)
        {
            return lhs.x < rhs.x;
        }
        //  -------------------------------------------------------
        bool compareY(const Point& lhs, const Point& rhs)
        {
            return lhs.y < rhs.y;
        }
        //  -------------------------------------------------------
        bool compareXofPair(const std::pair<int, int>& lhs, const std::pair<int, int>& rhs)
        {
            return lhs.second < rhs.second;
        }
    }
}
