#include "GaussianFilter.cuh"

#include <thrust/transform.h>

namespace ced
{
    namespace gpu
    {
        __host__ std::device_vector<float> gaussianFilter(int _dimension, float _sigma)
        {
            thrust::device_vector<float> filter(_dimension * _dimension);
            thrust::device_vector<int> itr(_dimension * _dimension);
            thrust::sequence(itr.begin(), itr.end());
            const int middle = _dimension / 2; 
            const float sigma2 = _sigma * _sigma;
            
            float g = [=] __device__ (int& id)
            {
                int y = id / _dimension; 
                int x = id &(_dimension-1); 
                
                return std::exp(-((x*x + y*y)/(2.0f*sigma2))) / (2.0f * std::acos(-1.0f) *sigma2);
            };
            
            thrust::transform(itr.begin(), itr.end(), filter.begin(), g); 
        }
    }
}
