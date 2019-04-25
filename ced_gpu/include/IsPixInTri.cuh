#ifndef ISPIXINTRI_CUH_INCLUDED
#define ISPIXINTRI_CUH_INCLUDED

namespace ced
{
    namespace gpu
    {
        __host__ __device__ bool isPixInTri(float2 a, float2 b, float2 c, float2 p);
    }
}

#endif // ISPIXINTRI_CUH_INCLUDED
