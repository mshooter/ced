#ifndef CALCULATECENTROID_CUH_INCLUDED
#define CALCULATECENTROID_CUH_INCLUDED

#include <thrust/device_vector.h>

namespace ced
{
    namespace gpu
    {
        __host__ float calculateCentroidx(const thrust::device_vector<float>& d_x);
        __host__ float calculateCentroidy( const thrust::device_vector<float>& d_y);
    }
}

#endif // CALCULATECENTROID_CUH_INCLUDED
