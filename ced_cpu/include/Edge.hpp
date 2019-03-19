// @ file: Edge.hpp
#ifndef EDGE_H_INCLUDED
#define EDGE_H_INCLUDED

#include "Point.hpp"
// TODO: smart pointers
namespace ced
{
    namespace cpu
    {
        class Edge
        {
            public:
                Edge(Point _p1, Point _p2);
                Edge(const Edge&) = default;
                Edge& operator=(const Edge&) = default; 
                Edge(Edge&&) = default; 
                Edge& operator=(Edge&&) = default; 
                bool operator==(const Edge& rhs) const;
                Point startPoint; 
                Point endPoint;
        };

    }
}

#endif // EDGE_H_INCLUDED
