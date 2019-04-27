#ifndef GETWHITEPIXELS_CUH_INCLUDED
#define GETWHITEPIXELS_CUH_INCLUDED

#include <thrust/device_vector.h>

namespace ced
{
    namespace gpu
    {
        void getWhitePixels(    thrust::device_vector<float2>& _whitePixels,
                                thrust::device_vector<float> _red,
                                thrust::device_vector<float> _green,
                                thrust::device_vector<float> _blue,
                                );
    }
}

#endif // GETWHITEPIXELS_CUH_INCLUDED
