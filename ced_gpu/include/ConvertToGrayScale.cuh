#ifndef CONVERTTOGRAYSCALE_CUH_INCLUDED
#define CONVERTTOGRAYSCALE_CUH_INCLUDED

#include <thrust/device_vector.h>

namespace ced
{
    namespace gpu
    {
            void convertToGrayScale(
            thrust::device_vector<float>& red,
            thrust::device_vector<float>& green,
            thrust::device_vector<float>& blue
            );
    }
}

#endif // CONVERTTOGRAYSCALE_CUH_INCLUDED
