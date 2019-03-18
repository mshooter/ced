#ifndef DELAUNAY_H_INCLUDED
#define DELAUNAY_H_INCLUDED

#include <vector>
#include "Point.hpp"
namespace ced
{
    namespace cpu
    {
       template <typename T>
       bool isReverseInHull(T _edge, std::vector<T> _hull)
       {
           for(auto edge : _hull)
           {
               if(_edge == edge)
               {
                   return true;
               }
           }
           return false;
       }
       //--------------------------------------------------------------------------
       bool isLeft(Point* p1, Point* p2, Point* p3)
       {
           int v = ((p3->x - p1->x) * (p2->y - p1->y)) - ((p2->x - p1->x) * (p3->y - p1->y));
           // is left
           if( v <= 0 )    { return true;  }
           // is right
           else            { return false; }
       }
    }
}

#endif //DELAUNAY_H_INCLUDED
