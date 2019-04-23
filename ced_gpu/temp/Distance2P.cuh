#ifndef DISTANCE2P_CUH_INCLUDED
#define DISTANCE2P_CUH_INCLUDED

namespace ced
{
    namespace gpu
    {
        template <typename T>
        __device__ T distance2P(T x1, T y1, T x2, T y2);
        #include "Distance2P.inl"
    }
}

#endif // DISTANCE2P_CUH_INCLUDED
