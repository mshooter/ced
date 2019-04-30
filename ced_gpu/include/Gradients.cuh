#ifndef GRADIENTS_CUH_INCLUDED
#define GRADIENTS_CUH_INCLUDED

#include <thrust/device_vector.h>
namespace ced
{
    namespace gpu
    {
        // calculate the gradients , use convolutional 
        void gradients( thrust::device_vector<float>& _gx, 
                        thrust::device_vector<float>& _gy,
                        thrust::device_vector<float>& _red,
                        thrust::device_vector<float>& _green,
                        thrust::device_vector<float>& _blue,
                        int& _height,
                        int& _width);
    }
}


#endif // GRADIENTS_CUH_INCLUDED
