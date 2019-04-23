#ifndef DISTANCE2P_CUH_INCLUDED
#define DISTANCE2P_CUH_INCLUDED

namespace ced
{
    namespace gpu
    {
        template <typename T>
        struct distance2P
        {
            const float2 cc;

            distance2P(float2 _cc) : cc(_cc) {}

            __host__ __device__
            T operator()(const float2& ptItr) const 
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
