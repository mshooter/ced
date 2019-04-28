#ifndef GETWHITEPIXELS_CUH_INCLUDED
#define GETWHITEPIXELS_CUH_INCLUDED

#include <thrust/device_vector.h>

namespace ced
{
    namespace gpu
    {
        void getWhitePixels(    thrust::device_vector<int>& _whitePixels,
                                thrust::device_vector<float> _red,
                                thrust::device_vector<float> _green,
                                thrust::device_vector<float> _blue
                                );
        //  --------------------------------------------------------------------------
        struct isWhite
        {
            __host__ __device__ 
            int operator()(const float& f)
            {
                if(f == 3.0f)
                {
                    return 1;
                }
                else
                {
                    return 0;
                }
            }
        };
        //  --------------------------------------------------------------------------
        struct isIdentity
        {
            __host__ __device__ 
            int operator()(const thrust::tuple<int, int>& f)
            {
                if(thrust::get<0>(f) == 1)
                {
                    return thrust::get<1>(f);
                }
                else
                {
                    return -1;
                }
            }
        };
    }
}

#endif // GETWHITEPIXELS_CUH_INCLUDED
