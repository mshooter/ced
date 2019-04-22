#include "CalculateCentroid.cuh"

#include <thrust/extrema.h>

namespace ced
{
    namespace gpu
    {
        __host__ Point calculateCentroid(thrust::device_vector<float> x, thrust::device_vector<float> y)
        {
            float minx = thrust::min_element(x.begin(), x.end());
            float maxx = thrust::max_element(x.begin(), x.end());
            float miny = thrust::min_element(y.begin(), y.end());
            float maxy = thrust::max_element(y.begin(), y.end());

            float cx = (minx + maxx) / 2.0f;
            float cy = (miny + maxy) / 2.0f;
            return Point(cx, cy);
        }
    }
}
