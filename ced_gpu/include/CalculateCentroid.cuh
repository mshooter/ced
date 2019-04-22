#ifndef CALCULATECENTROID_CUH_INCLUDED
#define CALCULATECENTROID_CUH_INCLUDED

#include <thrust/device_vector.h>

namespace ced
{
    namespace gpu
    {
        __host__ Point calculateCentroid(thrust::device_vector<Point> pts);
    }
}

#endif // CALCULATECENTROID_CUH_INCLUDED
