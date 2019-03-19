#include "Edge.hpp"
#include <utility>

namespace ced
{
    namespace cpu
    {
        bool Edge::operator==(const Edge& rhs) const
        {
            return ((startPoint.x == rhs.startPoint.x) &&
                    (startPoint.y == rhs.startPoint.y) &&
                    (endPoint.x   == rhs.endPoint.x  ) &&
                    (endPoint.y   == rhs.endPoint.y));
        }
        Edge::Edge(Point _p1, Point _p2)
        {
           startPoint = std::move(_p1);
           endPoint = std::move(_p2);
        }
    }
}
