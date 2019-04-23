#ifndef CCW_CUH_INCLUDED
#define CCW_CUH_INCLUDED

namespace ced
{
    namespace gpu
    {
        __host__ __device__ 
        float CCW(float2 p0, float2 p1, float2 p2); 
    }
}

#endif // CCW_CUH_INCLUDED
