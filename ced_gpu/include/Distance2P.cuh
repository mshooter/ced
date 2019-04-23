#ifndef DISTANCE2P_CUH_INCLUDED
#define DISTANCE2P_CUH_INCLUDED

#include "Point.cuh"

namespace ced
{
    namespace gpu
    {
        template <typename T>
        struct distance2P
        {
            const Point cc;

            distance2P(Point _cc) : cc(_cc) {}

            __host__ __device__
            T operator()(const Point& ptItr) const 
            {
                T distanceX = cc.x - ptItr.x; 
                T distanceY = cc.y - ptItr.y;
                T dist = distanceX * distanceX + distanceY * distanceY;
                return dist;
            }
        };
    }
}


#endif //DISTANCE2P_CUH_INCLUDED
