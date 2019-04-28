#ifndef TRIANGULATION_CUH_INCLUDED
#define TRIANGULATION_CUH_INCLUDED

#include <thrust/device_vector.h>

namespace ced
{   
    namespace gpu
    {
        __host__ void createFirstTri(
            const thrust::device_vector<float>& d_x,
            const thrust::device_vector<float>& d_y,
            int& i0, 
            int& i1, 
            int& i2, 
            const float2& centroid );
        //  ----------------------------------------------------------------------------------------------
        __host__ void triangulate(   thrust::device_vector<float>& x,
                                     thrust::device_vector<float>& y,
                                     thrust::device_vector<int>& triangles);
    }
}

#endif // TRIANGULATION_CUH_INCLUDED
