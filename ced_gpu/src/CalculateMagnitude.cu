#include "CalculateMagnitude.cuh"
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform.h>
#include <thrust/functional.h>

namespace ced
{
    namespace gpu
    {
        void calculateMagnitude(thrust::device_vector<float>& _gx,
                                thrust::device_vector<float>& _gy,
                                thrust::device_vector<float>& _magnitude)
        {
            thrust::transform(
                        thrust::make_zip_iterator(
                                            thrust::make_tuple(_gx.begin(), _gy.begin())),
                        thrust::make_zip_iterator(
                                            thrust::make_tuple(_gx.end(), _gy.end())),
                        _magnitude.begin(),
                        calculate_magnitude()); 
        }
    }
}
