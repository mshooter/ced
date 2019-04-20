#ifndef CALCULATEMIDPOINT_H_INCLUDED
#define CALCULATEMIDPOINT_H_INCLUDED

#include "Point.hpp"
namespace ced 
{
    namespace cpu
    {
        Point calculateMidpoint(
            const Point& a, 
            const Point& b, 
            const Point& c);
    }
}

#endif // CALCULATEMIDPOINT_H_INCLUDED
