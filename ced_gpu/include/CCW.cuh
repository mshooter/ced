#ifndef CCW_CUH_INCLUDED
#define CCW_CUH_INCLUDED

namespace ced
{
    namespace gpu
    {
        __host__ __device__ 
        float CCW(  float p1x, 
                    float p1y, 
                    float p2x, 
                    float p2y, 
                    float p3x, 
                    float p3y);
    }
}

#endif // CCW_CUH_INCLUDED
