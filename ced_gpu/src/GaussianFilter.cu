#include "GaussianFilter.cuh"

#include <thrust/transform.h>
#include <thrust/device_vector.h>

#include "ThrustFunctors.cuh"

namespace ced
{
    namespace gpu
    {
        std::vector<float> gaussianFilter(int _dimension, float _sigma)
        {
            const float sigma2 = _sigma * _sigma;
            thrust::device_vector<float> filter(_dimension * _dimension);
            thrust::device_vector<int> itr(_dimension * _dimension);
            thrust::sequence(itr.begin(), itr.end());
            thrust::transform(itr.begin(), itr.end(), filter.begin(), g(_dimension, sigma2)); 

            float sum = thrust::reduce(filter.begin(), filter.end());
            thrust::device_vector<float> d_filter(_dimension * _dimension);
            thrust::transform(filter.begin(), filter.end(), filter.begin(), multiplyConst(sum));
            std::vector<float> h_filter(filter.begin(), filter.end()); 
            return h_filter;
        }
    }
}
