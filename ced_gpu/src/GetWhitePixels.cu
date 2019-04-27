#include "GetWhitePixels.cuh"

namespace ced
{
    namespace gpu
    {
        void getWhitePixels(    thrust::device_vector<float2>& _whitePixels,
                                thrust::device_vector<float> _red,
                                thrust::device_vector<float> _green,
                                thrust::device_vector<float> _blue,
                                )
        {
            thrust::device_vector<float> d_result(m_red.size());
            // sum red and green
            thrust::transform(  d_red.begin(), 
                                d_red.end(), 
                                d_green.begin(), 
                                d_result.begin(), 
                                thrust::plus<float>());

            thrust::transform(  d_result.begin(), 
                                d_result.end(), 
                                d_blue.begin(), 
                                d_result.begin(), 
                                thrust::plus<float>());
        }
    }
}


