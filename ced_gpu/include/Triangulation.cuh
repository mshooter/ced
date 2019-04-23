#ifndef TRIANGULATION_CUH_INCLUDED
#define TRIANGULATION_CUH_INCLUDED

#include <thrust/device_vector.h>
#include "Point.cuh"

namespace ced
{   
    namespace gpu
    {
        __host__ void createFirstTri(
            const thrust::device_vector<Point>& d_pts,
            int& i0, 
            int& i1, 
            int& i2, 
            const Point& centroid );
    }
}

#endif // TRIANGULATION_CUH_INCLUDED
