#ifndef TRIANGULATION_CUH_INCLUDED
#define TRIANGULATION_CUH_INCLUDED

#include <thrust/device_vector.h>
#include "Point.hpp"

namespace ced
{
    namespace gpu
    {
        __host__ void createFirstTri(thrust::device_vector<Point> pts, unsigned int& i0, unsigned int& i1, unsigned int& i2, Point centroid);
        __device__ unsigned int hashKey(Point p, Point cc, unsigned int hashSize);
    }
}

#endif // TRIANGULATION_CUH_INCLUDED
