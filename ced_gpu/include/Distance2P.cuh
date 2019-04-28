#ifndef DISTANCE2P_CUH_INCLUDED
#define DISTANCE2P_CUH_INCLUDED

#include <thrust/tuple.h>
namespace ced
{
    namespace gpu
    {
        template <typename T>
        struct distance2P
        {
            const float cx;
            const float cy;
            distance2P(float _cx, float _cy) : cx(_cx), cy(_cy) {}

            __host__ __device__
            T operator()(const thrust::tuple<float, float>& elements) const 
            {
                T distanceX = cx - elements.get<0>(); 
                T distanceY = cy - elements.get<1>();
                T dist = distanceX * distanceX + distanceY * distanceY;
                return dist;
            }
        };
    }
}


#endif //DISTANCE2P_CUH_INCLUDED
