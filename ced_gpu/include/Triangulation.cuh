#ifndef TRIANGULATION_CUH_INCLUDED
#define TRIANGULATION_CUH_INCLUDED

#include <thrust/device_vector.h>

namespace ced
{   
    namespace gpu
    {
        __host__ void createFirstTri(
            const thrust::device_vector<float2>& d_pts,
            int& i0, 
            int& i1, 
            int& i2, 
            const float2& centroid );
    }
}

#endif // TRIANGULATION_CUH_INCLUDED
