#ifndef GAUSSIANFILTER_CUH_INCLUDED
#define GAUSSIANFILTER_CUH_INCLUDED

#include <thrust/device_vector.h>
namespace ced
{
    namespace gpu
    {
        __host__ std::device_vector<float> gaussianFilter(int _dimension = 5, float _sigma = 1.4f); 
    }
}

#endif // GAUSSIANFILTER_CUH_INCLUDED
