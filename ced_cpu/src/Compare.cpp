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
        bool compareXofPair(const std::pair<unsigned int, unsigned int>& lhs, const std::pair<unsigned int, unsigned int>& rhs)
        {
            return lhs.first < rhs.second;
        }
    }
}
