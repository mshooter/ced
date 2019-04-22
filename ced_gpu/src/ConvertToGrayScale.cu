#include "ConvertToGrayScale.cuh"
#include "Math.cuh"

#include <thrust/copy.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform.h>
#include <thrust/for_each.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>

namespace ced
{
    namespace gpu
    {
        __host__ void converToGrayScale(
                thrust::device_vector<float>& red,
                thrust::device_vector<float>& green,
                thrust::device_vector<float>& blue )
        {
            thrust::device_vector<float> result(red.size());
            // sum red and green
            thrust::transform(thrust::device, red.begin(), red.end(), green.begin(), result.begin(), thrust::plus<float>());
            thrust::transform(thrust::device, result.begin(), result.end(), blue.begin(), result.begin(), thrust::plus<float>());
            // DIVIDE
            thrust::transform(thrust::device, result.begin(), result.end(), result.begin(), divideByConstant<float>(3.0f));
            // assign result element to red, green and blue
            thrust::copy(thrust::device, result.begin(), result.end(), red.begin());    
            thrust::copy(thrust::device, result.begin(), result.end(), green.begin());    
            thrust::copy(thrust::device, result.begin(), result.end(), blue.begin());    
        }   
    }
}
