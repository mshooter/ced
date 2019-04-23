#ifndef PSEUDOANGLE_CUH_INCLUDED
#define PSEUDOANGLE_CUH_INCLUDED

#include "Point.cuh"

namespace ced
{
    namespace gpu
    {
        struct angle_funct
        {
            __host__ __device__ 
            float operator()(const Point& p)
            {
                const float d = p.x / (std::abs(p.x) + std::abs(p.y));
                return (p.y > 0.0f ? 3.0f - d : 1.0f + d) / 4.0f; 
            }
        };
    }
}

#endif // PSEUDOANGLE_CUH_INCLUDED

