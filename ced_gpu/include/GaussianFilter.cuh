// GaussianFilter.cuh
#ifndef GAUSSIANFILTER_H_INCLUDED 
#define GAUSSIANFILTER_H_INCLUDED

#include <thrust/device_vector.h>

namespace ced
{
    namespace gpu
    {
        __global__ thrust::device_vector<float> gaussianFilter(int _dimension = 5, float = _sigma 1.4f); 
        
    }
}

#endif //GAUSSIANFILTER_H_INCLUDED
  
