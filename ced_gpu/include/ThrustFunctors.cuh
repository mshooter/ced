#ifndef THRUSTFUNCTORS_CUH_INCLUDED
#define THRUSTFUNCTORS_CUH_INCLUDED

#include "Distance2P.cuh"

namespace ced
{
    namespace gpu
    {
        struct DistanceCP
        {
            const Point centroid;
            __device__ 
            float operator()(Point& p1)
            {
                return distance2P<float>(centroid.x, centroid.y, p1.x, p1.y);
            }
        }
    }
}

#endif // THRUSTFUNCTORS_CUH_INCLUDED

