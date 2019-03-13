#include "isLeft.hpp"

namespace ced
{
    namespace cpu
    {
        bool isLeft(Point* p1, Point* p2, Point* p3)
        {
            int v = ((p3->x - p1->x) * (p2->y - p1->y)) - ((p2->x - p1->x) * (p3->y - p1->y));
            if( v <= 0 )    { return true;  }
            else            { return false; }
        }
    }
}
