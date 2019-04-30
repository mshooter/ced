#ifndef CALCULATEDIRECTIONS_CUH_INCLUDED
#define CALCULATEDIRECTIONS_CUH_INCLUDED

#include <thrust/device_vector.h>

namespace ced
{
    namespace gpu
    {
        void calculateDirections(   thrust::device_vector<float>& _gx, 
                                    thrust::device_vector<float>& _gy,
                                    thrust::device_vector<float>& _directions);
        struct calculate_directions
        {
            __host__ __device__
            float operator()(const thrust::tuple<float, float>& t)
            {
                return fabs(atan(thrust::get<1>(t)/thrust::get<0>(t)) * 180.0f/ 3.142f);
            }
        };
    }
}

#endif// CALCULATEDIRECTIONS_CUH_INCLUDED
