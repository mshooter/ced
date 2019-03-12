// @ file: Edge.hpp
#ifndef EDGE_H_INCLUDED
#define EDGE_H_INCLUDED

#include "Point.hpp"
namespace ced
{
    namespace cpu
    {
        class Edge
        {
            public:
                Edge(Point _p1, Point _p2)
                {
                   fPoint = Point(_p1);
                   ePoint = Point(_p2);
                }

                Point fPoint; 
                Point ePoint;
        };
    }
}

#endif // EDGE_H_INCLUDED
