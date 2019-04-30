#ifndef CALCULATEMAGNITUDE_CUH_INCLUDED
#define CALCULATEMAGNITUDE_CUH_INCLUDED

#include <thrust/device_vector.h>
#include <thrust/tuple.h>

namespace ced
{
    namespace gpu
    {
        void calculateMagnitude(thrust::device_vector<float>& _gx,
                                thrust::device_vector<float>& _gy,
                                thrust::device_vector<float>& _magnitude);
        //  ---------------------------------------------------------------------------------
        struct calculate_magnitude
        {
            __host__ __device__
            float operator()(const thrust::tuple<float, float>& t)
            {
                return fabs(thrust::get<0>(t)) + fabs(thrust::get<1>(t));
            }
        };
    }
}

#endif// CALCULATEMAGNITUDE_CUH_INCLUDED
