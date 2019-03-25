#ifndef CIRCUMCIRCLE_H_INCLUDED
#define CIRCUMCIRCLE_H_INCLUDED

#include <cmath>
#include "Point.hpp"

namespace ced
{
    namespace cpu
    {
        bool circumCircle(Point A, Point B, Point C, Point D); 
        float circumRadius(Point A, Point B, Point C);
    }
}

#endif // CIRCUMCIRCLE_H_INCLUDED
