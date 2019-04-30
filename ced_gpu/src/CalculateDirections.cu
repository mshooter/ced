#include "CalculateDirections.cuh"

namespace ced
{
    namespace gpu
    {
        void calculateDirections(   thrust::device_vector<float>& _gx, 
                                    thrust::device_vector<float>& _gy,
                                    thrust::device_vector<float>& _directions)
        {
            thrust::transform(
                        thrust::make_zip_iterator(
                                            thrust::make_tuple(_gx.begin(), _gy.begin())),
                        thrust::make_zip_iterator(
                                            thrust::make_tuple(_gx.end(), _gy.end())),
                        _directions.begin(),
                        calculate_directions()); 
        }
    }
}
