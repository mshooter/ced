#ifndef CALCULATECENTROID_CUH_INCLUDED
#define CALCULATECENTROID_CUH_INCLUDED

#include <thrust/device_vector.h>
#include "Point.cuh"

namespace ced
{
    namespace gpu
    {
        __host__ Point calculateCentroid(const thrust::device_vector<float>& d_x, const thrust::device_vector<float>& d_y);
    }
}

#endif // CALCULATECENTROID_CUH_INCLUDED
