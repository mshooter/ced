#include "Edge.hpp"
#include <utility>

namespace ced
{
    namespace cpu
    {
        Edge::Edge(Point* _p1, Point* _p2)
        {
           startPoint = std::move(_p1);
           endPoint = std::move(_p2);
        }
    }
}
