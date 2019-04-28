#include "CalculateCentroid.cuh"
#include <thrust/extrema.h>

namespace ced
{
    namespace gpu
    {
        __host__ float calculateCentroidx(const thrust::device_vector<float>& d_x)
        {
            float minx = d_x[thrust::min_element(thrust::device, d_x.begin(), d_x.end())- d_x.begin()];
            float maxx = d_x[thrust::max_element(thrust::device, d_x.begin(), d_x.end())- d_x.begin()];
            float cx = (minx + maxx) / 2.0f;
            return cx;
        }

        __host__ float calculateCentroidy(const thrust::device_vector<float>& d_y)
        {
            float miny = d_y[thrust::min_element(thrust::device, d_y.begin(), d_y.end())- d_y.begin()];
            float maxy = d_y[thrust::max_element(thrust::device, d_y.begin(), d_y.end())- d_y.begin()];
            float cy = (miny + maxy) / 2.0f;
            return cy;
        }
    }
}
