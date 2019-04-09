#ifndef PSEUDO_ANGLE_H
#define PSEUDO_ANGLE_H

#include "Point.hpp"

namespace ced
{
    namespace cpu
    {
        // monotonically increases with real angle, but doenst need expensive trigonometry
        float pseudo_angle(Point p);
    }
}

#endif // PSEUDO_ANGLE_H
