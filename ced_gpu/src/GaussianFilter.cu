#include "GaussianFilter.cuh"
#include <thrust/device_vector.h>
#include <thrust/transform.h>
namespace ced
{
    namespace gpu
    {
        __global__ std::vector<float> gaussianFilter(int _dimension, float _sigma);
        {

        }
    }
}
