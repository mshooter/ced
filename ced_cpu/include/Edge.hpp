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
                Edge(Point* _p1, Point* _p2);
                Point* startPoint; 
                Point* endPoint;
        };
    }
}

#endif // EDGE_H_INCLUDED
